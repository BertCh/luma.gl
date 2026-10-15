// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationCapture} from '@luma.gl/gpgpu';
import type {LensRecord} from '../../../src/computation-views/registry/view-registration';

/** One authored card capture, including its recording provenance. */
export function makeY2Capture(
  name: string,
  values: number[],
  metadata: Partial<GPUComputationCapture> = {}
): GPUComputationCapture {
  return {
    id: `${name}-id`,
    name,
    values,
    elementType: 'f32',
    provenance: 'gpu-readback',
    ...metadata
  };
}

/** Minimal real-record schema with caller-owned snapshots. */
export function makeY2Record(snapshots: GPUComputationCapture[]): LensRecord {
  return {
    schema: 'luma.computation/1',
    id: 'gpu-core.cards',
    runId: 'y2-cards',
    createdAt: '2026-10-14T00:00:00.000Z',
    adapter: {backend: 'webgpu', description: 'Card fixture adapter'},
    domains: ['gpu-core'],
    graphs: [],
    axes: {axes: [{id: 'entity', label: 'entity', kind: 'data'}], keySpaces: [], relations: []},
    refs: {formulas: [], sources: []},
    decisions: [],
    choices: [],
    captures: {
      schema: 'luma.computation-captures/1',
      runId: 'y2-cards',
      snapshots,
      stamps: [],
      replays: []
    }
  };
}

/** Points with separate value captures and an explicitly authored context layer. */
export function makeY2Points(values?: number[]): LensRecord {
  const coordinates = Array.from({length: values?.length ?? 3}, (_, index) => [
    index,
    index % 2
  ]).flat();
  const geometry = {kind: 'points', coordinateModel: 'projected'} as const;
  return makeY2Record([
    makeY2Capture('context', [-1, -1, 8, 3], {components: 2, role: 'context', geometry}),
    makeY2Capture('input', coordinates, {components: 2, role: 'input', geometry}),
    makeY2Capture('output', coordinates, {
      components: 2,
      role: 'output',
      geometry: {...geometry, ...(values ? {valuesCapture: 'values'} : {})}
    }),
    ...(values ? [makeY2Capture('values', values)] : [])
  ]);
}

/** Array or column captures with intentionally unequal lengths. */
export function makeY2Arrays(table = false): LensRecord {
  return makeY2Record([
    makeY2Capture(
      'input',
      Array.from({length: 20}, (_, index) => index),
      {role: 'input', ...(table ? {keySpace: 'rows'} : {})}
    ),
    makeY2Capture(
      'output',
      Array.from({length: table ? 4 : 20}, (_, index) => index * 2),
      {role: 'output', ...(table ? {keySpace: 'rows'} : {})}
    )
  ]);
}

/** Recorded raster rounds, with no synthetic intermediate states. */
export function makeY2Frames(count = 3): LensRecord {
  return makeY2Record(
    Array.from({length: count}, (_, iteration) =>
      makeY2Capture('arrival', [0, iteration, iteration * 2, 5], {
        id: `arrival-${iteration}`,
        role: 'output',
        iteration,
        meaning: 'settledAt round',
        geometry: {kind: 'raster', shape: [2, 2], cellSize: 30, unit: 'm'}
      })
    )
  );
}
