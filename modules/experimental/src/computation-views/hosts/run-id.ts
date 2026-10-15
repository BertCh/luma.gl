// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Computes FNV-1a 64 over UTF-8 canonical JSON, using two unsigned 32-bit lanes. */
export function computeRunId(props: {
  structureIds: readonly string[];
  datasetVersion: string;
  seed?: number | string;
  adapter: string;
  variant?: string;
}): string {
  // A positional JSON join preserves boundaries, optional values, seed types and structure order.
  const canonical = JSON.stringify([
    props.structureIds,
    props.datasetVersion,
    props.seed ?? null,
    props.adapter,
    props.variant ?? null
  ]);
  let high = 0xcbf29ce4;
  let low = 0x84222325;
  for (const byte of new TextEncoder().encode(canonical)) {
    low = (low ^ byte) >>> 0;
    // FNV prime = 2**40 + 0x1b3. The low-lane product remains exactly representable.
    const product = low * 0x1b3;
    high = (Math.imul(high, 0x1b3) + Math.floor(product / 0x100000000) + (low << 8)) >>> 0;
    low = product >>> 0;
  }
  return `run-${high.toString(16).padStart(8, '0')}${low.toString(16).padStart(8, '0')}`;
}
