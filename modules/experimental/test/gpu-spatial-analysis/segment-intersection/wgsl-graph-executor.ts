// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Executes command nodes on the CPU with the wgsl_reflect interpreter (WgslExec), for node specs
// that check what a kernel computes, not only how it is wired. Nodes run in the order given, each
// logical graph buffer gets its own zero-initialized memory (no transient aliasing), and
// GPU-written indirect dispatch gates are honored. Only small inputs are practical: the
// interpreter runs every invocation in JavaScript.

import {NullDevice} from '@luma.gl/test-utils';
import type {GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {ScalarData, TypedData, WgslExec, WgslParser} from 'wgsl_reflect';

type LogicalBuffer = {byteLength: number};
type FakeBuffer = {backing: ArrayBuffer; byteLength: number};
type FakeBinding = {buffer: FakeBuffer; offset: number; size: number};
type FakePipeline = {source: string; constants: Record<string, number>};

type ExecutableNode = {
  id: string;
  type?: string;
  condition?: {source: string; mode?: string; buffer?: LogicalBuffer; byteOffset?: number};
  compile?: (context: {device: unknown}) => {
    encode: (context: Record<string, unknown>) => void;
    destroy?: () => void;
  };
};

/** CPU memory of every logical buffer the executed nodes touched. */
export type CPUGraphMemory = {
  /** Writes `values` at the view's first element (bit copy). */
  write: (view: GraphDataView, values: ArrayLike<number> | ArrayBufferView) => void;
  /** Reads `length` u32 words (default: the view's components) from the view's first element. */
  readUint32: (view: GraphDataView, length?: number) => Uint32Array;
  /** Reads `length` f32 words from the view's first element. */
  readFloat32: (view: GraphDataView, length?: number) => Float32Array;
  /** Number of dispatches executed per node id (gated-off nodes count zero workgroups). */
  workgroupCounts: Map<string, number>;
};

function getComponentCount(view: GraphDataView): number {
  return (view.length * view.rowByteLength) / 4;
}

/** Creates the CPU memory used by {@link executeCommandNodesOnCPU}. */
export function createCPUGraphMemory(): CPUGraphMemory & {
  getBuffer: (logical: LogicalBuffer) => FakeBuffer;
} {
  const buffers = new Map<LogicalBuffer, FakeBuffer>();
  const getBuffer = (logical: LogicalBuffer): FakeBuffer => {
    let buffer = buffers.get(logical);
    if (!buffer) {
      buffer = {
        backing: new ArrayBuffer(Math.ceil(logical.byteLength / 4) * 4 + 16),
        byteLength: logical.byteLength
      };
      buffers.set(logical, buffer);
    }
    return buffer;
  };
  const words = (view: GraphDataView, length?: number) => {
    const buffer = getBuffer(view.buffer as unknown as LogicalBuffer);
    return {buffer, offset: view.byteOffset, length: length ?? getComponentCount(view)};
  };
  return {
    getBuffer,
    workgroupCounts: new Map(),
    write(view, values) {
      const {buffer, offset} = words(view);
      const source = ArrayBuffer.isView(values)
        ? new Uint8Array(values.buffer, values.byteOffset, values.byteLength)
        : new Uint8Array(Float32Array.from(values as ArrayLike<number>).buffer);
      new Uint8Array(buffer.backing, offset, source.byteLength).set(source);
    },
    readUint32(view, length) {
      const {buffer, offset, length: count} = words(view, length);
      return new Uint32Array(buffer.backing.slice(offset, offset + 4 * count));
    },
    readFloat32(view, length) {
      const {buffer, offset, length: count} = words(view, length);
      return new Float32Array(buffer.backing.slice(offset, offset + 4 * count));
    }
  };
}

function createExecutorDevice(): NullDevice {
  const device = new NullDevice({});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  Object.defineProperty(device, 'props', {
    value: {...device.props, _cacheShaders: false, _cachePipelines: false}
  });
  Object.defineProperty(device, 'limits', {
    value: {
      ...device.limits,
      maxComputeWorkgroupsPerDimension: 65535,
      maxStorageBufferBindingSize: 134217728,
      maxStorageBuffersPerShaderStage: 8,
      maxComputeWorkgroupStorageSize: 16384,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 256,
      maxComputeWorkgroupSizeZ: 64
    }
  });
  Object.defineProperty(device, 'createComputePipeline', {
    value: (props: {shader: {source: string}; constants?: Record<string, number>}) => ({
      source: props.shader.source,
      constants: {...props.constants},
      hash: '',
      destroy: () => {}
    })
  });
  return device;
}

function installInterpreterFixes(): () => void {
  const originals = [
    [WgslExec.prototype, '_call'],
    [WgslExec.prototype, '_evalCall'],
    [WgslExec.prototype, '_evalBinaryOp'],
    [TypedData.prototype, 'setDataValue']
  ] as const;
  const descriptors = originals.map(
    ([prototype, name]) => Object.getOwnPropertyDescriptor(prototype, name)!
  );
  // wgsl_reflect 1.2.3 binds call arguments in the callee's scope, whose parent is the CALLER's
  // scope: an argument expression can read a parameter bound just before it, and a statement call
  // (`_call`) assigns parameters with `setVariable`, which overwrites a caller variable of the same
  // name. Calls are rebound here with WGSL semantics: arguments evaluated in the caller's scope,
  // parameters created in a fresh scope whose parent is the module scope.
  type InterpreterContext = {
    parent: InterpreterContext | null;
    functions: Map<string, unknown>;
    currentFunctionName: string;
    getFunction: (name: string) => {node: {args: {name: string}[]; body: unknown[]}} | null;
    createVariable: (name: string, value: unknown, node: unknown) => void;
  };
  type CallExpression = {
    name: string;
    args: unknown[];
    cachedReturnValue: unknown;
    postfix?: unknown;
  };
  type InterpreterPrototype = {
    _call: (call: CallExpression, context: InterpreterContext) => unknown;
    _evalCall: (call: CallExpression, context: InterpreterContext) => unknown;
    evalExpression: (expression: unknown, context: InterpreterContext) => unknown;
    _execStatements: (statements: unknown[], context: InterpreterContext) => unknown;
  };
  const interpreter = WgslExec.prototype as unknown as InterpreterPrototype;
  {
    const originalCall = interpreter._call;
    const originalEvalCall = interpreter._evalCall;
    const getModuleScope = (context: InterpreterContext): InterpreterContext => {
      let scope: InterpreterContext | null = context;
      while (scope && scope.functions.size === 0) {
        scope = scope.parent;
      }
      return scope ?? context;
    };
    const invoke = function (
      this: InterpreterPrototype,
      call: CallExpression,
      context: InterpreterContext
    ): {handled: boolean; value?: unknown} {
      const callee = context.getFunction(call.name);
      if (!callee) {
        return {handled: false};
      }
      const values = callee.node.args.map((_, index) =>
        this.evalExpression(call.args[index], context)
      );
      const Scope = (
        context as unknown as {constructor: new (parent: InterpreterContext) => InterpreterContext}
      ).constructor;
      const scope = new Scope(getModuleScope(context));
      scope.currentFunctionName = call.name;
      callee.node.args.forEach((argument, index) =>
        scope.createVariable(argument.name, values[index], argument)
      );
      const value = this._execStatements(callee.node.body, scope) as {
        getSubData?: (exec: unknown, postfix: unknown, context: InterpreterContext) => unknown;
      } | null;
      // Fourth interpreter bug: a postfix on a call result (`f(x).member`, `f(x)[i]`) is dropped, so
      // the whole struct is compared or assigned instead of the member. Apply it here.
      return {
        handled: true,
        value:
          call.postfix && value?.getSubData ? value.getSubData(this, call.postfix, context) : value
      };
    };
    interpreter._call = function (call, context) {
      const result = invoke.call(this, call, context);
      return result.handled ? undefined : originalCall.call(this, call, context);
    };
    interpreter._evalCall = function (call, context) {
      if (call.cachedReturnValue !== null) {
        return call.cachedReturnValue;
      }
      const result = invoke.call(this, call, context);
      return result.handled ? result.value : originalEvalCall.call(this, call, context);
    };
  }

  // wgsl_reflect 1.2.3 rejects `x = y` for two array values ("SetDataValue: Type mismatch
  // array<u32, 3> array<u32, 3>") and leaves `x` unchanged, because it compares the two TypeInfo
  // objects by identity and every array expression builds its own. Repro:
  //   var a = array<u32, 2>(1u, 2u); var b = array<u32, 2>(3u, 4u); a = b;  // a stays (1, 2)
  // Same-named types are interchangeable, so give the value the target's TypeInfo before the copy.
  type TypedDataPrototype = {
    typeInfo: {getTypeName: () => string};
    setDataValue: (exec: unknown, value: unknown, postfix: unknown, context: unknown) => void;
  };
  const typedData = TypedData.prototype as unknown as TypedDataPrototype;
  {
    const originalSetDataValue = typedData.setDataValue;
    typedData.setDataValue = function (exec, value, postfix, context) {
      if (
        value instanceof TypedData &&
        value.typeInfo !== (this.typeInfo as unknown) &&
        value.typeInfo?.getTypeName() === this.typeInfo.getTypeName()
      ) {
        (value as unknown as {typeInfo: unknown}).typeInfo = this.typeInfo;
      }
      return originalSetDataValue.call(this, exec, value, postfix, context);
    };
  }

  // wgsl_reflect 1.2.3 shifts u32 values right with the JavaScript signed `>>`, so a u32 of 2^31 or
  // more shifts in ones: `0xf0000000u >> 16u` gives 0xfffff000u instead of 0xf000u. Repro:
  //   var t = 65535u * 65535u + 131070u;  // 0xffffffffu
  //   t >> 16u                            // 0xffffffffu, expected 0xffffu
  // Scalar u32 shifts are redone with `>>>`.
  type ShiftPrototype = {
    _evalBinaryOp: (node: BinaryNode, context: unknown) => unknown;
    evalExpression: (expression: unknown, context: unknown) => unknown;
  };
  type BinaryNode = {operator: string; left: unknown; right: unknown};
  const shifter = WgslExec.prototype as unknown as ShiftPrototype;
  {
    const originalBinaryOp = shifter._evalBinaryOp;
    shifter._evalBinaryOp = function (node, context) {
      // wgsl_reflect eagerly evaluates both sides of logical operators. WGSL short-circuits:
      // e.g. ring assembly guards claim[NONE] with successor != NONE && claim[successor] ... .
      // Besides throwing on that invalid read, eagerness incorrectly executes RHS side effects.
      if (node.operator === '&&' || node.operator === '||') {
        const left = this.evalExpression(node.left, context);
        if (left instanceof ScalarData) {
          if (node.operator === '&&' ? !left.value : Boolean(left.value)) {
            return left;
          }
          return this.evalExpression(node.right, context);
        }
      }
      if (node.operator === '>>') {
        const left = this.evalExpression(node.left, context);
        const right = this.evalExpression(node.right, context);
        if (
          left instanceof ScalarData &&
          right instanceof ScalarData &&
          left.typeInfo.name === 'u32'
        ) {
          return new ScalarData(left.value >>> right.value, left.typeInfo);
        }
      }
      return originalBinaryOp.call(this, node, context);
    };
  }

  return () => {
    originals.forEach(([prototype, name], index) =>
      Object.defineProperty(prototype, name, descriptors[index])
    );
  };
}

const programs = new Map<
  string,
  {exec: WgslExec; bindings: Map<string, {location: number; type: string}>}
>();

function getProgram(source: string) {
  let program = programs.get(source);
  if (!program) {
    const bindings = new Map<string, {location: number; type: string}>();
    for (const match of source.matchAll(
      /@group\(0\)\s*@binding\((\d+)\)\s*var<storage,\s*\w+>\s*(\w+)\s*:\s*array<([^;]+)>;/g
    )) {
      bindings.set(match[2], {location: Number(match[1]), type: match[3]});
    }
    if (/\bworkgroupBarrier\s*\(|\bstorageBarrier\s*\(/.test(source)) {
      // Invocations run one after another, so a barrier cannot synchronize them: replace the
      // producer with a CPU stand-in (cpu-gpu-core-emulations.ts) instead.
      throw new Error('CPU executor: cooperative kernels (barriers) cannot run on the interpreter');
    }
    // Phony assignments keep bindings in the layout; the interpreter does not know '_'.
    // wgsl_reflect 1.2.3 indexes a one-value workgroup_size attribute as a string ('32' -> 3 x 2).
    const executable = source
      .replace(/^\s*_\s*=[^;]*;/gm, '')
      .replace(/@workgroup_size\(\s*(\d+)u?\s*\)/g, '@workgroup_size($1, 1, 1)')
      // wgsl_reflect 1.2.3 cannot assign a vector lane by dynamic index inside a struct
      // ("SetDataValue: Type vec4i is not an array", the lane is dropped). Repro:
      //   struct S { v: vec4i } var s = S(vec4i(0)); var k = 1u; s.v[k] = 5;  // s.v unchanged
      // Lane writes are rewritten to member swizzles (`exactOrientSigns` writes `result.signs[k]`).
      // `var a = 0;` is an i32 in WGSL, but wgsl_reflect 1.2.3 types the literal "abstract" and
      // backs it with a Uint32Array when it is not negative, so `a = -1; a > 0` is true. Repro:
      // `var a = 0; a = -1; if (a > 0) { ... }` takes the branch. Integer literal initializers
      // of `var` declarations get the `i` suffix, which types them i32.
      .replace(/\bvar\s+(\w+)\s*(?::\s*i32\s*)?=\s*(-?\d+)\s*;/g, 'var $1 = $2i;')
      // A `bool` struct member has no layout size in wgsl_reflect 1.2.3: it aliases the offset of the
      // previous member (`struct S { v: vec4i, f: bool }`: writing `s.v.x` changes `s.f`), and a read
      // is TypedData, which `select` rejects as a condition. Members declared on their own line are
      // declared `u32` instead; the interpreter does not type-check, and assigning or testing a bool
      // against a u32 member works.
      .replace(/^(\s+\w+\s*:\s*)bool(\s*,?\s*)$/gm, '$1u32$2')
      .replace(
        /^(\s*)([A-Za-z_]\w*\.[A-Za-z_]\w*)\[(\w+)\] = ([^;]+);/gm,
        (_, indent, vector, lane, value) =>
          `${indent}if (${lane} == 0u) { ${vector}.x = ${value}; } ` +
          `else if (${lane} == 1u) { ${vector}.y = ${value}; } ` +
          `else if (${lane} == 2u) { ${vector}.z = ${value}; } else { ${vector}.w = ${value}; }`
      );
    program = {exec: new WgslExec(new WgslParser().parse(executable)), bindings};
    programs.set(source, program);
  }
  return program;
}

function getTypedArray(binding: FakeBinding, type: string): ArrayBufferView {
  const {backing} = binding.buffer;
  const length = Math.floor(binding.size / 4);
  if (type === 'f32' || /^vec\dx?<f32>|^vec\df/.test(type)) {
    return new Float32Array(backing, binding.offset, length);
  }
  if (type === 'i32' || type === 'atomic<i32>') {
    return new Int32Array(backing, binding.offset, length);
  }
  return new Uint32Array(backing, binding.offset, length);
}

/**
 * Executes `nodes` in order on `memory`. Compute nodes (including upstream `Kernel`-based ones)
 * and copy nodes are supported; render nodes and CPU conditions throw.
 */
export function executeCommandNodesOnCPU(
  nodes: readonly unknown[],
  memory: ReturnType<typeof createCPUGraphMemory>
): void {
  const device = createExecutorDevice();
  const restoreInterpreter = installInterpreterFixes();
  try {
    for (const node of nodes as ExecutableNode[]) {
      if (!node.compile) {
        throw new Error(`CPU executor: node ${node.id} has no compile callback`);
      }
      if (node.condition && node.condition.source !== 'gpu') {
        throw new Error(`CPU executor: node ${node.id} has an unsupported condition`);
      }
      const compiled = node.compile({device});
      let pipeline: FakePipeline | undefined;
      let bindings: Record<string, FakeBinding> = {};
      const run = (x: number, y: number, z: number) => {
        memory.workgroupCounts.set(node.id, (memory.workgroupCounts.get(node.id) ?? 0) + x * y * z);
        if (x * y * z === 0) {
          return;
        }
        if (!pipeline) {
          throw new Error(`CPU executor: node ${node.id} dispatched without a pipeline`);
        }
        const program = getProgram(pipeline.source);
        const group: Record<number, ArrayBufferView> = {};
        for (const [name, binding] of Object.entries(bindings)) {
          const declaration = program.bindings.get(name);
          if (!declaration) {
            throw new Error(`CPU executor: node ${node.id} binding ${name} is not declared`);
          }
          group[declaration.location] = getTypedArray(binding, declaration.type);
        }
        try {
          program.exec.dispatchWorkgroups(
            'main',
            [x, y, z],
            {0: group},
            {constants: pipeline.constants}
          );
        } catch (error) {
          // Name the kernel: interpreter errors carry no WGSL location.
          throw new Error(`CPU executor: node ${node.id} failed: ${String(error)}`, {cause: error});
        }
      };
      const computePass = {
        setPipeline: (next: FakePipeline) => {
          pipeline = next;
        },
        setBindings: (next: Record<string, FakeBinding>) => {
          bindings = next;
        },
        dispatch: (x: number, y = 1, z = 1) => {
          const condition = node.condition;
          if (condition?.source === 'gpu' && condition.buffer) {
            const command = new Uint32Array(
              memory.getBuffer(condition.buffer).backing,
              condition.byteOffset ?? 0,
              3
            );
            run(command[0], command[1], command[2]);
          } else {
            run(x, y, z);
          }
        },
        dispatchIndirect: () => {
          throw new Error(
            `CPU executor: node ${node.id} dispatched indirectly without a condition`
          );
        }
      };
      const commandEncoder = {
        copyBufferToBuffer: (props: {
          sourceBuffer: FakeBuffer;
          sourceOffset?: number;
          destinationBuffer: FakeBuffer;
          destinationOffset?: number;
          size: number;
        }) => {
          const source = new Uint8Array(
            props.sourceBuffer.backing,
            props.sourceOffset ?? 0,
            props.size
          );
          new Uint8Array(
            props.destinationBuffer.backing,
            props.destinationOffset ?? 0,
            props.size
          ).set(source.slice());
        },
        clearBuffer: (buffer: FakeBuffer, offset = 0, size?: number) => {
          new Uint8Array(buffer.backing, offset, size ?? buffer.byteLength - offset).fill(0);
        }
      };
      try {
        compiled.encode({
          computePass,
          commandEncoder,
          getBuffer: (view: GraphDataView) =>
            memory.getBuffer(view.buffer as unknown as LogicalBuffer),
          getTextureView: () => {
            throw new Error(`CPU executor: node ${node.id} uses textures`);
          },
          parameters: undefined
        });
      } finally {
        compiled.destroy?.();
      }
    }
  } finally {
    // Interpreters retain their last dispatch bindings; release them with the case's buffers.
    programs.clear();
    restoreInterpreter();
    device.destroy();
  }
}
