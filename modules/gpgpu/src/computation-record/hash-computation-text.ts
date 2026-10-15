// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

const FNV_OFFSET_BASIS = 0x811c9dc5;
const FNV_PRIME = 0x01000193;

/**
 * Returns a deterministic 8-digit hexadecimal FNV-1a hash of `text` over its UTF-16 code units.
 *
 * Used for source-table ids (`wgsl:<hash>`) and body hashes. It is an identity hash for
 * deduplication and drift detection, not a cryptographic digest.
 */
export function hashGPUComputationText(text: string): string {
  let hash = FNV_OFFSET_BASIS;
  for (let index = 0; index < text.length; index++) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, FNV_PRIME);
  }
  return (hash >>> 0).toString(16).padStart(8, '0');
}
