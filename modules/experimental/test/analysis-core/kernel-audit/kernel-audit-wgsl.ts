// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Static checks over generated WGSL that `wgsl_reflect` does not make, for the library-wide kernel
 * audit (quality rule 16 plus the compile traps of BRIEF rule 6). Every check here caught, or would
 * have caught, a kernel that Dawn rejected on a real GPU while `wgsl_reflect` accepted it.
 */

/** WGSL keywords (WGSL spec, "Keyword Summary"). */
export const WGSL_KEYWORDS: ReadonlySet<string> = new Set([
  'alias',
  'break',
  'case',
  'const',
  'const_assert',
  'continue',
  'continuing',
  'default',
  'diagnostic',
  'discard',
  'else',
  'enable',
  'false',
  'fn',
  'for',
  'if',
  'let',
  'loop',
  'override',
  'requires',
  'return',
  'struct',
  'switch',
  'true',
  'var',
  'while'
]);

/**
 * WGSL reserved words (WGSL spec, "Reserved Words"). Any use as an identifier is a shader-creation
 * error, wherever it appears (declaration, parameter, struct member or reference).
 */
export const WGSL_RESERVED_WORDS: ReadonlySet<string> = new Set([
  'NULL',
  'Self',
  'abstract',
  'active',
  'alignas',
  'alignof',
  'as',
  'asm',
  'asm_fragment',
  'async',
  'attribute',
  'auto',
  'await',
  'become',
  'cast',
  'catch',
  'class',
  'co_await',
  'co_return',
  'co_yield',
  'coherent',
  'column_major',
  'common',
  'compile',
  'compile_fragment',
  'concept',
  'const_cast',
  'consteval',
  'constexpr',
  'constinit',
  'crate',
  'debugger',
  'decltype',
  'delete',
  'demote',
  'demote_to_helper',
  'do',
  'dynamic_cast',
  'enum',
  'explicit',
  'export',
  'extends',
  'extern',
  'external',
  'fallthrough',
  'filter',
  'final',
  'finally',
  'friend',
  'from',
  'fxgroup',
  'get',
  'goto',
  'groupshared',
  'highp',
  'impl',
  'implements',
  'import',
  'inline',
  'instanceof',
  'interface',
  'layout',
  'lowp',
  'macro',
  'macro_rules',
  'match',
  'mediump',
  'meta',
  'mod',
  'module',
  'move',
  'mut',
  'mutable',
  'namespace',
  'new',
  'nil',
  'noexcept',
  'noinline',
  'nointerpolation',
  'non_coherent',
  'noncoherent',
  'noperspective',
  'null',
  'nullptr',
  'of',
  'operator',
  'package',
  'packoffset',
  'partition',
  'pass',
  'patch',
  'pixelfragment',
  'precise',
  'precision',
  'premerge',
  'priv',
  'protected',
  'pub',
  'public',
  'readonly',
  'ref',
  'regardless',
  'register',
  'reinterpret_cast',
  'require',
  'resource',
  'restrict',
  'self',
  'set',
  'shared',
  'sizeof',
  'smooth',
  'snorm',
  'static',
  'static_assert',
  'static_cast',
  'std',
  'subroutine',
  'super',
  'target',
  'template',
  'this',
  'thread_local',
  'throw',
  'trait',
  'try',
  'type',
  'typedef',
  'typeid',
  'typename',
  'typeof',
  'union',
  'unless',
  'unorm',
  'unsafe',
  'unsized',
  'use',
  'using',
  'varying',
  'virtual',
  'volatile',
  'wgsl',
  'where',
  'with',
  'writeonly',
  'yield'
]);

/** WGSL built-in functions (WGSL spec, "Built-in Functions", including subgroup and quad). */
const WGSL_BUILTIN_FUNCTIONS: ReadonlySet<string> = new Set([
  'bitcast',
  'all',
  'any',
  'select',
  'arrayLength',
  'abs',
  'acos',
  'acosh',
  'asin',
  'asinh',
  'atan',
  'atanh',
  'atan2',
  'ceil',
  'clamp',
  'cos',
  'cosh',
  'countLeadingZeros',
  'countOneBits',
  'countTrailingZeros',
  'cross',
  'degrees',
  'determinant',
  'distance',
  'dot',
  'dot4U8Packed',
  'dot4I8Packed',
  'exp',
  'exp2',
  'extractBits',
  'faceForward',
  'firstLeadingBit',
  'firstTrailingBit',
  'floor',
  'fma',
  'fract',
  'frexp',
  'insertBits',
  'inverseSqrt',
  'ldexp',
  'length',
  'log',
  'log2',
  'max',
  'min',
  'mix',
  'modf',
  'normalize',
  'pow',
  'quantizeToF16',
  'radians',
  'reflect',
  'refract',
  'reverseBits',
  'round',
  'saturate',
  'sign',
  'sin',
  'sinh',
  'smoothstep',
  'sqrt',
  'step',
  'tan',
  'tanh',
  'transpose',
  'trunc',
  'dpdx',
  'dpdxCoarse',
  'dpdxFine',
  'dpdy',
  'dpdyCoarse',
  'dpdyFine',
  'fwidth',
  'fwidthCoarse',
  'fwidthFine',
  'textureDimensions',
  'textureGather',
  'textureGatherCompare',
  'textureLoad',
  'textureNumLayers',
  'textureNumLevels',
  'textureNumSamples',
  'textureSample',
  'textureSampleBias',
  'textureSampleCompare',
  'textureSampleCompareLevel',
  'textureSampleGrad',
  'textureSampleLevel',
  'textureSampleBaseClampToEdge',
  'textureStore',
  'atomicLoad',
  'atomicStore',
  'atomicAdd',
  'atomicSub',
  'atomicMax',
  'atomicMin',
  'atomicAnd',
  'atomicOr',
  'atomicXor',
  'atomicExchange',
  'atomicCompareExchangeWeak',
  'pack4x8snorm',
  'pack4x8unorm',
  'pack4xI8',
  'pack4xU8',
  'pack4xI8Clamp',
  'pack4xU8Clamp',
  'pack2x16snorm',
  'pack2x16unorm',
  'pack2x16float',
  'unpack4x8snorm',
  'unpack4x8unorm',
  'unpack4xI8',
  'unpack4xU8',
  'unpack2x16snorm',
  'unpack2x16unorm',
  'unpack2x16float',
  'storageBarrier',
  'textureBarrier',
  'workgroupBarrier',
  'workgroupUniformLoad',
  'subgroupAdd',
  'subgroupExclusiveAdd',
  'subgroupInclusiveAdd',
  'subgroupAll',
  'subgroupAnd',
  'subgroupAny',
  'subgroupBallot',
  'subgroupBroadcast',
  'subgroupBroadcastFirst',
  'subgroupElect',
  'subgroupMax',
  'subgroupMin',
  'subgroupMul',
  'subgroupExclusiveMul',
  'subgroupInclusiveMul',
  'subgroupOr',
  'subgroupShuffle',
  'subgroupShuffleDown',
  'subgroupShuffleUp',
  'subgroupShuffleXor',
  'subgroupXor',
  'quadBroadcast',
  'quadSwapDiagonal',
  'quadSwapX',
  'quadSwapY'
]);

/** Predeclared type generators and scalar types that may be called as value constructors. */
const WGSL_CONSTRUCTOR_TYPES =
  /^(?:bool|i32|u32|f32|f16|array|vec[234][ifuh]?|mat[234]x[234][fh]?)$/;

/** One audit finding inside one WGSL module. */
export type WGSLFinding = {
  /** Check that failed. */
  check: string;
  /** Short description with the offending text. */
  detail: string;
};

/** Removes `//` line comments and (possibly nested) block comments, keeping line breaks. */
export function stripWGSLComments(source: string): string {
  let output = '';
  let depth = 0;
  for (let index = 0; index < source.length; index++) {
    const pair = source.slice(index, index + 2);
    if (pair === '/*') {
      depth++;
      index++;
    } else if (depth > 0 && pair === '*/') {
      depth--;
      index++;
    } else if (depth > 0) {
      if (source[index] === '\n') output += '\n';
    } else if (pair === '//') {
      while (index < source.length && source[index] !== '\n') index++;
      output += '\n';
    } else {
      output += source[index];
    }
  }
  return output;
}

/** Rejects reserved words used anywhere as identifiers, and keywords in declaration positions. */
export function findReservedIdentifiers(code: string): WGSLFinding[] {
  const findings = new Map<string, WGSLFinding>();
  for (const match of code.matchAll(/[A-Za-z_][A-Za-z0-9_]*/g)) {
    if (WGSL_RESERVED_WORDS.has(match[0])) {
      const context = code.slice(Math.max(0, match.index - 24), match.index + 24).trim();
      findings.set(match[0], {
        check: 'reserved word',
        detail: `\`${match[0]}\` in "${context.replace(/\s+/g, ' ')}"`
      });
    }
  }
  const declaration =
    /\b(?:let|var(?:<[^>]*>)?|const|override|fn|struct|alias)\s+([A-Za-z_]\w*)|[(,{]\s*([A-Za-z_]\w*)\s*:(?!:)/g;
  for (const match of code.matchAll(declaration)) {
    const name = match[1] ?? match[2];
    if (WGSL_KEYWORDS.has(name) && !(match[2] && name === 'default')) {
      findings.set(`keyword ${name}`, {check: 'keyword identifier', detail: `\`${name}\``});
    }
  }
  return [...findings.values()];
}

const OPERAND = String.raw`[A-Za-z_]\w*(?:\s*(?:\.\s*[A-Za-z_]\w*|\[[^\[\]]*\]))*`;
const SELF_COMPARE = new RegExp(
  String.raw`(^|[(,;{=?]|&&|\|\||\breturn\b)\s*(${OPERAND})\s*(==|!=)\s*(${OPERAND})\s*(?=$|[),;{\]]|&&|\|\|)`,
  'gm'
);

const F32_MAX = String.raw`3\.402823466e\+38`;

/**
 * Finite tests of upstream `@luma.gl/gpgpu` primitives that use the folded `v == v` NaN test
 * (`v == v && abs(v) <= F32_MAX`). The audit cannot fix upstream files (BRIEF rule 2), so these
 * exact idioms are filed as a request (see the T-AUDIT handoff) and ignored here. The analysis
 * sources themselves must not contain the idiom: `findSourceSelfCompareIdioms` checks that.
 */
export const UPSTREAM_SELF_COMPARE_IDIOMS: readonly {pattern: RegExp; source: string}[] = [
  {
    // gpu-bvh.ts:273,412, gpu-bvh-query.ts:343,428, gpu-segmented-bvh.ts:402,
    // gpu-spatial-utils.ts:19, gpu-grid-index-query.ts:420,603, gpu-point-spatial-filter.ts:322,
    // gpu-virtual-geometry-selection.ts:447, gpu-group-aggregation.ts:543, gpu-reduction.ts:419,
    // gpu-histogram.ts:438,565, gpu-vector-search/gpu-k-means.ts:403
    pattern: new RegExp(String.raw`\bvalue == value && abs\(value\) <= ${F32_MAX}`, 'g'),
    source: 'gpgpu finite(value) helpers'
  },
  {
    pattern: new RegExp(String.raw`\btotal == total && abs\(total\) <= ${F32_MAX}`, 'g'),
    source: 'gpgpu/src/gpu-graph/gpu-graph-page-rank-internals.ts:466'
  },
  {
    pattern: new RegExp(
      String.raw`\bx == x && y == y && abs\(x\) <= ${F32_MAX} && abs\(y\) <= ${F32_MAX}`,
      'g'
    ),
    source: 'gpu-grid-index-internals.ts:271, gpu-grid-binning.ts:319, gpu-grid-aggregation.ts:376'
  },
  {
    pattern: new RegExp(String.raw`\bz == z && abs\(z\) <= ${F32_MAX}`, 'g'),
    source: 'gpu-grid-index-internals.ts:244'
  },
  {
    pattern: new RegExp(String.raw`\bweight == weight && abs\(weight\) <= ${F32_MAX}`, 'g'),
    source: 'gpu-grid-aggregation.ts:377'
  },
  {
    pattern: new RegExp(
      String.raw`(boundsValues\[BOUNDS_OFFSET \+ \d+u\]) == \1 && (boundsValues\[BOUNDS_OFFSET \+ \d+u\]) == \2 && abs\(\1\) <= ${F32_MAX}`,
      'g'
    ),
    source: 'gpu-grid-index-internals.ts:345'
  }
];

/** Rejects `x == x` / `x != x` NaN tests, which Tint and Metal fold to a constant. */
export function findSelfCompares(code: string): WGSLFinding[] {
  const findings: WGSLFinding[] = [];
  for (const {pattern} of UPSTREAM_SELF_COMPARE_IDIOMS) {
    code = code.replace(pattern, 'true');
  }
  for (const match of code.matchAll(SELF_COMPARE)) {
    const left = match[2].replace(/\s+/g, '');
    if (left === match[4].replace(/\s+/g, '')) {
      findings.push({check: 'NaN self-compare', detail: `\`${left} ${match[3]} ${left}\``});
    }
  }
  return findings;
}

function parseIntegerLiteral(text: string): number | undefined {
  const match = /^\s*(0[xX][0-9a-fA-F]+|\d+)[iu]?\s*$/.exec(text);
  return match ? Number(match[1]) : undefined;
}

/**
 * Rejects a const-expression `bitcast<f32>(<integer constant>)` whose bits are an infinity or NaN.
 * Const evaluation of a non-finite value is a shader-creation error; the value must come from a
 * runtime `var`.
 */
export function findConstantNonFiniteBitcasts(code: string): WGSLFinding[] {
  const constants = new Map<string, number>();
  for (const match of code.matchAll(/\bconst\s+(\w+)\s*(?::\s*\w+\s*)?=\s*([^;]+);/g)) {
    const value = parseIntegerLiteral(match[2]);
    if (value !== undefined) constants.set(match[1], value);
  }
  const findings: WGSLFinding[] = [];
  const bitcast = /bitcast<\s*(f32|f16|vec[234]<\s*f32\s*>|vec[234]f)\s*>\s*\(\s*([\w]+)\s*\)/g;
  for (const match of code.matchAll(bitcast)) {
    const value = parseIntegerLiteral(match[2]) ?? constants.get(match[2]);
    if (value === undefined) continue;
    const nonFinite =
      match[1] === 'f16' ? (value & 0x7c00) === 0x7c00 : (value & 0x7f800000) === 0x7f800000;
    if (nonFinite) {
      findings.push({check: 'const non-finite bitcast', detail: `\`${match[0]}\``});
    }
  }
  for (const match of code.matchAll(
    /(?<![\w.])(\d+(?:\.\d*)?(?:e[+-]?\d+)?[fh]?)\s*\/\s*(0(?:\.0*)?[fhiu]?)(?![\w.])/g
  )) {
    findings.push({check: 'const division by zero', detail: `\`${match[0]}\``});
  }
  return findings;
}

// A template list follows its name directly (`bitcast<f32>(`), so `a < vec2<f32>(0.0)` is no call of `a`.
const CALL = /(?<![\w.@])([A-Za-z_]\w*)(?:<\w[\w\s,<>]*>)?\s*\(/g;
const CALL_KEYWORDS = new Set(['if', 'for', 'while', 'switch', 'return', 'loop', 'else', 'case']);

/** Rejects calls to functions that the module neither declares nor gets from WGSL. */
export function findUndeclaredCalls(code: string): WGSLFinding[] {
  const declared = new Set<string>();
  for (const match of code.matchAll(/\b(?:fn|struct|alias)\s+([A-Za-z_]\w*)/g)) {
    declared.add(match[1]);
  }
  const findings = new Map<string, WGSLFinding>();
  for (const match of code.matchAll(CALL)) {
    const name = match[1];
    if (/\bfn\s*$/.test(code.slice(Math.max(0, match.index - 8), match.index))) continue;
    if (
      declared.has(name) ||
      CALL_KEYWORDS.has(name) ||
      WGSL_BUILTIN_FUNCTIONS.has(name) ||
      WGSL_CONSTRUCTOR_TYPES.test(name)
    ) {
      continue;
    }
    findings.set(name, {check: 'undeclared call', detail: `\`${name}(\``});
  }
  return [...findings.values()];
}

// Memory and binding accounting.

/** Values of module `const` and `override` declarations, with pipeline constants applied. */
export function getWGSLConstantValues(
  code: string,
  pipelineConstants: Readonly<Record<string, number>> = {}
): Map<string, number> {
  const values = new Map<string, number>();
  const declarations = /\b(const|override)\s+(\w+)\s*(?::\s*\w+\s*)?(?:=\s*([^;]+))?;/g;
  for (const match of code.matchAll(declarations)) {
    const [, kind, name, expression] = match;
    if (kind === 'override' && name in pipelineConstants) {
      values.set(name, pipelineConstants[name]);
      continue;
    }
    const value =
      expression === undefined ? undefined : evaluateIntegerExpression(expression, values);
    if (value !== undefined) values.set(name, value);
  }
  return values;
}

/** Evaluates a small integer constant expression (`+ - * / % << >>`, parentheses, names). */
export function evaluateIntegerExpression(
  expression: string,
  values: ReadonlyMap<string, number>
): number | undefined {
  const substituted = expression
    .replace(/\b(?:u32|i32)\s*\(/g, '(')
    .replace(/\b(0[xX][0-9a-fA-F]+|\d+)[iu]\b/g, '$1')
    .replace(/\b[A-Za-z_]\w*\b/g, name => (values.has(name) ? String(values.get(name)) : name));
  if (!/^[\s\dxXa-fA-F+\-*/%()<>]*$/.test(substituted) || /[g-wyzG-WYZ]/.test(substituted)) {
    return undefined;
  }
  try {
    // The text is digits and operators only (checked above).
    // eslint-disable-next-line no-new-func
    const value = Function(`"use strict"; return (${substituted});`)() as number;
    return Number.isFinite(value) ? Math.trunc(value) : undefined;
  } catch {
    return undefined;
  }
}

type TypeLayout = {size: number; align: number};

function roundUp(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

/** Splits `a, b<c, d>, e` at top-level commas. */
function splitTopLevel(text: string): string[] {
  const parts: string[] = [];
  let depth = 0;
  let start = 0;
  for (let index = 0; index < text.length; index++) {
    const character = text[index];
    if (character === '<' || character === '(') depth++;
    else if (character === '>' || character === ')') depth--;
    else if (character === ',' && depth === 0) {
      parts.push(text.slice(start, index));
      start = index + 1;
    }
  }
  parts.push(text.slice(start));
  return parts.map(part => part.trim()).filter(Boolean);
}

/** WGSL host-shareable size and alignment of the types of one module. */
export class WGSLTypeLayouts {
  private readonly structs = new Map<string, string[]>();
  private readonly aliases = new Map<string, string>();
  private readonly layouts = new Map<string, TypeLayout>();
  /** Array lengths that could not be resolved, for diagnostics. */
  readonly unresolvedLengths = new Set<string>();

  constructor(
    code: string,
    private readonly constants: ReadonlyMap<string, number>
  ) {
    for (const match of code.matchAll(/\bstruct\s+(\w+)\s*\{([^}]*)\}/g)) {
      const memberTypes = splitTopLevel(match[2])
        .map(member => member.replace(/@\w+(?:\([^)]*\))?/g, '').trim())
        .map(member => member.slice(member.indexOf(':') + 1).trim());
      this.structs.set(match[1], memberTypes);
    }
    for (const match of code.matchAll(/\balias\s+(\w+)\s*=\s*([^;]+);/g)) {
      this.aliases.set(match[1], match[2].trim());
    }
  }

  /** Returns the layout of a WGSL type; unknown names count as one 4-byte word. */
  getLayout(type: string): TypeLayout {
    const key = type.replace(/\s+/g, '');
    const cached = this.layouts.get(key);
    if (cached) return cached;
    const layout = this.computeLayout(key);
    this.layouts.set(key, layout);
    return layout;
  }

  private computeLayout(type: string): TypeLayout {
    if (this.aliases.has(type)) return this.getLayout(this.aliases.get(type)!);
    const struct = this.structs.get(type);
    if (struct) {
      let offset = 0;
      let align = 1;
      for (const member of struct) {
        const layout = this.getLayout(member);
        offset = roundUp(offset, layout.align) + layout.size;
        align = Math.max(align, layout.align);
      }
      return {size: roundUp(offset, align), align};
    }
    if (/^(?:f16)$/.test(type)) return {size: 2, align: 2};
    if (/^(?:bool|i32|u32|f32)$/.test(type) || /^atomic</.test(type)) return {size: 4, align: 4};
    const vector = /^vec([234])(?:<(\w+)>|([ifuh]))$/.exec(type);
    if (vector) {
      const component = vector[2] === 'f16' || vector[3] === 'h' ? 2 : 4;
      const count = Number(vector[1]);
      const align = (count === 2 ? 2 : 4) * component;
      return {size: count * component, align};
    }
    const matrix = /^mat([234])x([234])(?:<(\w+)>|([fh]))?$/.exec(type);
    if (matrix) {
      const column = this.getLayout(
        `vec${matrix[2]}<${matrix[3] ?? (matrix[4] === 'h' ? 'f16' : 'f32')}>`
      );
      return {size: Number(matrix[1]) * roundUp(column.size, column.align), align: column.align};
    }
    const array = /^array<(.+)>$/.exec(type);
    if (array) {
      const [elementType, lengthExpression] = splitTopLevel(array[1]);
      const element = this.getLayout(elementType);
      const stride = roundUp(element.size, element.align);
      if (lengthExpression === undefined) return {size: stride, align: element.align};
      const length = evaluateIntegerExpression(lengthExpression, this.constants);
      if (length === undefined) {
        this.unresolvedLengths.add(lengthExpression);
        return {size: stride, align: element.align};
      }
      return {size: stride * length, align: element.align};
    }
    return {size: 4, align: 4};
  }
}

/** Finds the matching closing brace of the block opened at `open`. */
function findBlockEnd(code: string, open: number): number {
  let depth = 0;
  for (let index = open; index < code.length; index++) {
    if (code[index] === '{') depth++;
    else if (code[index] === '}' && --depth === 0) return index;
  }
  return code.length;
}

/** Type text of a declaration `NAME : TYPE` up to `=` or `;` at depth zero. */
function readDeclaredType(text: string): string | undefined {
  const colon = /^\s*:\s*/.exec(text);
  if (!colon) return undefined;
  let depth = 0;
  let index = colon[0].length;
  for (; index < text.length; index++) {
    const character = text[index];
    if (character === '<') depth++;
    else if (character === '>') depth--;
    else if (depth === 0 && (character === '=' || character === ';')) break;
  }
  return text.slice(colon[0].length, index).trim();
}

/** Byte totals of one WGSL module against the core limits. */
export type WGSLModuleUsage = {
  /** `var<private>` bytes. */
  privateBytes: number;
  /** Largest function-address-space bytes of any one function, and that function. */
  functionBytes: number;
  functionName: string;
  /** `var<workgroup>` bytes. */
  workgroupBytes: number;
  /** `var<storage, ...>` bindings. */
  storageBuffers: number;
  /** `var<uniform>` bindings. */
  uniformBuffers: number;
  /** Storage-texture bindings. */
  storageTextures: number;
  /** Sampled-texture bindings. */
  sampledTextures: number;
  /** Sampler bindings. */
  samplers: number;
  /** `@workgroup_size` of each entry point, resolved. */
  workgroupSizes: number[][];
  /** Array lengths the accounting could not resolve (counted as one element). */
  unresolvedLengths: string[];
};

/** Accounts the private, function, workgroup memory and the bindings of one module. */
export function getWGSLModuleUsage(
  code: string,
  pipelineConstants: Readonly<Record<string, number>> = {}
): WGSLModuleUsage {
  const constants = getWGSLConstantValues(code, pipelineConstants);
  const layouts = new WGSLTypeLayouts(code, constants);
  const sumModuleVars = (addressSpace: string): number => {
    let total = 0;
    const pattern = new RegExp(String.raw`var<\s*${addressSpace}\s*>\s*\w+`, 'g');
    for (const match of code.matchAll(pattern)) {
      const type = readDeclaredType(code.slice(match.index + match[0].length));
      if (type) total += layouts.getLayout(type).size;
    }
    return total;
  };
  let functionBytes = 0;
  let functionName = '';
  for (const match of code.matchAll(/\bfn\s+(\w+)\s*\([^{]*\{/g)) {
    const open = match.index + match[0].length - 1;
    const body = code.slice(open, findBlockEnd(code, open));
    let total = 0;
    for (const variable of body.matchAll(/\bvar\s+\w+/g)) {
      const rest = body.slice(variable.index + variable[0].length);
      const type =
        readDeclaredType(rest) ??
        /^\s*=\s*((?:array|vec[234]|mat[234]x[234])<[^(]+>)\s*\(/.exec(rest)?.[1];
      total += type ? layouts.getLayout(type).size : 4;
    }
    if (total > functionBytes) {
      functionBytes = total;
      functionName = match[1];
    }
  }
  const workgroupSizes: number[][] = [];
  for (const match of code.matchAll(/@workgroup_size\s*\(([^)]*)\)/g)) {
    workgroupSizes.push(
      splitTopLevel(match[1]).map(part => evaluateIntegerExpression(part, constants) ?? Number.NaN)
    );
  }
  const count = (pattern: RegExp): number => (code.match(pattern) ?? []).length;
  return {
    privateBytes: sumModuleVars('private'),
    functionBytes,
    functionName,
    workgroupBytes: sumModuleVars('workgroup'),
    storageBuffers: count(/var<\s*storage\b/g),
    uniformBuffers: count(/var<\s*uniform\s*>/g),
    storageTextures: count(/:\s*texture_storage_\w+/g),
    sampledTextures: count(
      /:\s*texture_(?:1d|2d|2d_array|3d|cube|cube_array|multisampled_2d|depth_\w+|external)\b/g
    ),
    samplers: count(/:\s*sampler(?:_comparison)?\b/g),
    workgroupSizes,
    unresolvedLengths: [...layouts.unresolvedLengths]
  };
}

/** Core WebGPU limits the audit checks every kernel against. */
export const CORE_SHADER_LIMITS = {
  maxStorageBuffersPerShaderStage: 8,
  maxUniformBuffersPerShaderStage: 12,
  maxStorageTexturesPerShaderStage: 4,
  maxSampledTexturesPerShaderStage: 16,
  maxSamplersPerShaderStage: 16,
  maxPrivateBytes: 8192,
  maxFunctionBytes: 8192,
  maxComputeWorkgroupStorageSize: 16384,
  maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupSize: [256, 256, 64]
} as const;

/** Every static finding of one module: compile traps plus core-limit violations. */
export function auditWGSLModule(
  source: string,
  pipelineConstants: Readonly<Record<string, number>> = {}
): WGSLFinding[] {
  const code = stripWGSLComments(source);
  const findings = [
    ...findReservedIdentifiers(code),
    ...findSelfCompares(code),
    ...findConstantNonFiniteBitcasts(code),
    ...findUndeclaredCalls(code)
  ];
  const usage = getWGSLModuleUsage(code, pipelineConstants);
  const limits = CORE_SHADER_LIMITS;
  const over = (check: string, value: number, limit: number, detail = '') => {
    if (value > limit) findings.push({check, detail: `${value} > ${limit}${detail}`});
  };
  over('storage buffers per stage', usage.storageBuffers, limits.maxStorageBuffersPerShaderStage);
  over('uniform buffers per stage', usage.uniformBuffers, limits.maxUniformBuffersPerShaderStage);
  over(
    'storage textures per stage',
    usage.storageTextures,
    limits.maxStorageTexturesPerShaderStage
  );
  over(
    'sampled textures per stage',
    usage.sampledTextures,
    limits.maxSampledTexturesPerShaderStage
  );
  over('samplers per stage', usage.samplers, limits.maxSamplersPerShaderStage);
  over('private bytes', usage.privateBytes, limits.maxPrivateBytes);
  over(
    'function bytes',
    usage.functionBytes,
    limits.maxFunctionBytes,
    ` in fn ${usage.functionName}`
  );
  over('workgroup bytes', usage.workgroupBytes, limits.maxComputeWorkgroupStorageSize);
  for (const size of usage.workgroupSizes) {
    const [x = 1, y = 1, z = 1] = size;
    if ([x, y, z].some(value => !Number.isFinite(value))) {
      findings.push({check: 'workgroup size', detail: `unresolved @workgroup_size(${size})`});
      continue;
    }
    over('workgroup invocations', x * y * z, limits.maxComputeInvocationsPerWorkgroup);
    limits.maxComputeWorkgroupSize.forEach((limit, axis) =>
      over(`workgroup size ${'xyz'[axis]}`, [x, y, z][axis], limit)
    );
  }
  return findings;
}
