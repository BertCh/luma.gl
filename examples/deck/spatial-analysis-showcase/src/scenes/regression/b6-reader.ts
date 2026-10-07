// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import type {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';

/** One named slice of a {@link createNamedReader} read: `bytes` of `buffer`. */
export type NamedSource = {name: string; buffer: Buffer; bytes: number};

/** Typed views of one named slice. Both views share a private copy of the bytes. */
export type NamedSlice = {f32: Float32Array; u32: Uint32Array};

/**
 * A `SummaryReader` over several named buffers: the copies run back to back into one ring
 * ticket and `onResult` receives a lookup of copies by name. Results are small (summaries and
 * per-row result columns of a few thousand rows) and are only read when an input changed.
 */
export function createNamedReader(
  resources: SpatialAnalysisResources,
  id: string,
  sources: readonly NamedSource[],
  onResult: (get: (name: string) => NamedSlice) => void
): SummaryReader {
  return new SummaryReader(
    resources,
    id,
    sources.map(({buffer, bytes}) => ({buffer, size: bytes})),
    bytes => {
      const offsets = new Map<string, number>();
      const lengths = new Map<string, number>();
      let cursor = 0;
      for (const source of sources) {
        offsets.set(source.name, cursor);
        lengths.set(source.name, source.bytes);
        cursor += source.bytes;
      }
      try {
        onResult(name => {
          const offset = offsets.get(name);
          if (offset === undefined) throw new Error(`Unknown reader slice "${name}"`);
          const copy = bytes.slice(offset, offset + lengths.get(name)!);
          return {f32: new Float32Array(copy), u32: new Uint32Array(copy)};
        });
      } catch (error) {
        // SummaryReader swallows errors (it expects a destroyed device); report real bugs.
        // biome-ignore lint/suspicious/noConsole: surfaces errors the reader would otherwise swallow
        console.error(`reader ${id}:`, error);
      }
    }
  );
}
