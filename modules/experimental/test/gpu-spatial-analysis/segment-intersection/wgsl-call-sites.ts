// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Inlined call-site counting for big WGSL helpers. Shader compilers inline every function call, so
// a second call site of a large exact predicate (the 256-bit orientation fallback, the limb
// incircle) duplicates it in the pipeline; a second copy made the spatial-join kernels 2.5x slower.

/** Function bodies by name, and the names of the compute entry points. */
export function parseWGSLFunctions(source: string): {
  bodies: Map<string, string>;
  entries: string[];
} {
  const bodies = new Map<string, string>();
  const entries: string[] = [];
  for (const match of source.matchAll(/(@compute[^\n]*\n?\s*)?fn\s+(\w+)\s*\(/g)) {
    const name = match[2];
    if (match[1]) {
      entries.push(name);
    }
    const open = source.indexOf('{', match.index);
    let depth = 0;
    let end = open;
    for (; end < source.length; end++) {
      if (source[end] === '{') {
        depth++;
      } else if (source[end] === '}') {
        depth--;
        if (depth === 0) {
          break;
        }
      }
    }
    bodies.set(name, source.slice(open, end));
  }
  return {bodies, entries};
}

function countCalls(body: string, name: string): number {
  return body.match(new RegExp(`\\b${name}\\s*\\(`, 'g'))?.length ?? 0;
}

/**
 * Number of call sites of every function on a call chain to `target`, counted over the functions
 * reachable from the compute entry point (dead helpers are dropped by compilers). A value above
 * one means the target is inlined more than once.
 */
export function getInlinedCallSites(source: string, target: string): Map<string, number> {
  const {bodies, entries} = parseWGSLFunctions(source);
  const reachable = new Set(entries);
  const queue = [...entries];
  while (queue.length > 0) {
    const body = bodies.get(queue.pop() ?? '') ?? '';
    for (const name of bodies.keys()) {
      if (!reachable.has(name) && countCalls(body, name) > 0) {
        reachable.add(name);
        queue.push(name);
      }
    }
  }
  const chain = new Set([target]);
  let grown = true;
  while (grown) {
    grown = false;
    for (const name of reachable) {
      const body = bodies.get(name) ?? '';
      if (!chain.has(name) && [...chain].some(callee => countCalls(body, callee) > 0)) {
        chain.add(name);
        grown = true;
      }
    }
  }
  const sites = new Map<string, number>();
  if (!reachable.has(target)) {
    return sites;
  }
  for (const callee of chain) {
    let count = 0;
    for (const name of reachable) {
      count += countCalls(bodies.get(name) ?? '', callee);
    }
    if (!entries.includes(callee)) {
      sites.set(callee, count);
    }
  }
  return sites;
}

/** Names of functions a kernel calls but never declares (wgsl_reflect does not check calls). */
export function getUndeclaredCalls(source: string): string[] {
  const {bodies} = parseWGSLFunctions(source.replace(/\/\/[^\n]*/g, ''));
  const builtins = new Set([
    'select',
    'min',
    'max',
    'abs',
    'sign',
    'floor',
    'ceil',
    'round',
    'sqrt',
    'clamp',
    'dot',
    'any',
    'all',
    'bitcast',
    'atomicAdd',
    'atomicSub',
    'atomicMin',
    'atomicMax',
    'atomicLoad',
    'atomicStore',
    'atomicOr',
    'atomicAnd',
    'atomicExchange',
    'atomicCompareExchangeWeak',
    'workgroupBarrier',
    'storageBarrier',
    'arrayLength',
    'vec2',
    'vec3',
    'vec4',
    'vec2f',
    'vec3f',
    'vec4f',
    'vec2i',
    'vec3i',
    'vec4i',
    'vec2u',
    'vec3u',
    'vec4u',
    'array',
    'u32',
    'i32',
    'f32',
    'bool',
    'countLeadingZeros',
    'countTrailingZeros',
    'firstLeadingBit',
    'firstTrailingBit',
    'countOneBits',
    'reverseBits',
    'extractBits',
    'insertBits',
    'fma',
    'fract',
    'trunc',
    'exp2',
    'log2',
    'ldexp',
    'frexp',
    'pow',
    'exp',
    'log',
    'cos',
    'sin',
    'atan2',
    'length',
    'cross',
    'normalize',
    'mix',
    'step',
    'smoothstep',
    'textureLoad',
    'textureDimensions',
    'pack2x16float',
    'unpack2x16float',
    'distance',
    'modf',
    'transpose',
    'determinant',
    'mat2x2',
    'mat3x3',
    'mat4x4',
    'saturate',
    'quantizeToF16',
    'dot4U8Packed',
    'dot4I8Packed',
    'if',
    'for',
    'while',
    'switch',
    'return',
    'loop'
  ]);
  const structs = new Set([...source.matchAll(/struct\s+(\w+)/g)].map(match => match[1]));
  const missing = new Set<string>();
  for (const body of bodies.values()) {
    for (const match of body.matchAll(/\b([A-Za-z_]\w*)\s*(?:<[^>()]*>)?\s*\(/g)) {
      const name = match[1];
      if (!bodies.has(name) && !builtins.has(name) && !structs.has(name)) {
        missing.add(name);
      }
    }
  }
  return [...missing];
}
