// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Core WebGPU limit on private (function-scope) memory per invocation, in bytes. */
export const CORE_PRIVATE_MEMORY_LIMIT_BYTES = 8192;
/** Core WebGPU `maxComputeWorkgroupStorageSize`, in bytes. */
export const CORE_WORKGROUP_STORAGE_LIMIT_BYTES = 16384;

function getTypeBytes(type: string): number {
  const array = /^array<\s*([^,>]+(?:<[^>]*>)?)\s*,\s*(\d+)u?\s*>$/.exec(type.trim());
  if (array) {
    return getTypeBytes(array[1]) * Number(array[2]);
  }
  const vector = /^vec([234])</.exec(type.trim());
  if (vector) {
    return vector[1] === '2' ? 8 : 16;
  }
  return 4;
}

function sumDeclarations(source: string, pattern: RegExp): number {
  let total = 0;
  for (const match of source.matchAll(pattern)) {
    total += getTypeBytes(match[1]);
  }
  return total;
}

/**
 * Upper bound of the private memory one invocation of a kernel uses: every `var<private>` plus
 * every function-scope `var` of array type (scalar locals are registers). Arrays are counted at
 * their declared length.
 */
export function getPrivateMemoryBytes(source: string): number {
  return (
    sumDeclarations(source, /var<private>\s+\w+\s*:\s*([^;]+);/g) +
    sumDeclarations(source, /^\s+var\s+\w+\s*:\s*(array<[^;]+>);/gm)
  );
}

/** Bytes of `var<workgroup>` declarations of a kernel. */
export function getWorkgroupMemoryBytes(source: string): number {
  return sumDeclarations(source, /var<workgroup>\s+\w+\s*:\s*([^;]+);/g);
}

/** Number of `var<storage, ...>` bindings of a kernel. */
export function getStorageBindingCount(source: string): number {
  return (source.match(/var<storage,/g) ?? []).length;
}
