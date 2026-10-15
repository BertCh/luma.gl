// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Formats a cross-form node reference `<graphId>:<nodeId>` used by `derivedFrom`. */
export function formatGPUComputationNodeRef(graphId: string, nodeId: string): string {
  return `${graphId}:${nodeId}`;
}

/** Parses at the first colon; throws when the separator or either part is missing. */
export function parseGPUComputationNodeRef(ref: string): {graph: string; node: string} {
  const separator = ref.indexOf(':');
  if (separator <= 0 || separator === ref.length - 1) {
    throw new Error('Expected <graphId>:<nodeId>');
  }
  return {graph: ref.slice(0, separator), node: ref.slice(separator + 1)};
}
