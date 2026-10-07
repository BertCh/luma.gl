// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, CommandEncoder, Device} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {addKernelPass} from '../../engine/mode-kernels';
import type {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {FOCUS_SLOTS, type WeightsCsrBuffers} from './b4-weights-kit';

/** Id the gather writes into the slots past the end of the focus row. */
const UNUSED_SLOT = 0xffffffff;

const GATHER_BODY = /* wgsl */ `
  let focus = min(focusRow[focusRowOffset], ROW_COUNT - 1u);
  let start = offsets[offsetsOffset + focus];
  let end = offsets[offsetsOffset + focus + 1u];
  let slot = start + index;
  var other = 0xffffffffu;
  var value = 0.0;
  if (slot < end) {
    other = neighbors[neighborsOffset + slot];
    value = weights[weightsOffset + slot];
  }
  neighborIds[neighborIdsOffset + index] = other;
  slotWeights[slotWeightsOffset + index] = value;`;

/** The focus row of the analysed weights, as read back from the GPU. */
export type FocusRow = {
  /** Neighbour row ids of the first `degree` slots. */
  ids: Uint32Array;
  /** Weight of each slot, 0 past `degree`. */
  weights: Float32Array;
  /** Number of neighbours listed by the focus row (capped at the slot count). */
  degree: number;
};

/** Gathers the focus row of a CSR into two small buffers and reads them back. */
export type FocusRowGatherer = {
  /** Weight of each focus-row slot (float32), 0 past the row: the `width` channel of the bundle. */
  slotWeights: Buffer;
  /** Compiled graph, for the "rebuilds" accounting. */
  graph: CompiledGPUCommandGraph<never>;
  /** Encodes the gather; call after the CSR it reads was written in the same frame. */
  encode: (commandEncoder: CommandEncoder) => void;
  /** Reads the gathered row back; `onRow` fires when the bytes arrive. */
  request: (commandEncoder: CommandEncoder) => void;
  /** True while a read is in flight. */
  isReading: () => boolean;
  stop: () => void;
};

/**
 * One kernel that copies the focus row of a weights CSR to `FOCUS_SLOTS` ids and weights. The
 * focus bundle draws its link widths from these weights, and the weights display, the kernel
 * chart and the tooltips read them back. The row index is a one-word buffer, so moving the focus
 * is a buffer write, never a recompile.
 */
export function createFocusRowGatherer(props: {
  device: Device;
  resources: SpatialAnalysisResources;
  id: string;
  rows: number;
  slots: number;
  csr: WeightsCsrBuffers;
  focusRow: Buffer;
  onRow: (row: FocusRow) => void;
}): FocusRowGatherer {
  const {device, resources, id, rows, slots, csr, focusRow} = props;
  const neighborIds = resources.createBuffer(`${id}-ids`, FOCUS_SLOTS * 4);
  const slotWeights = resources.createBuffer(`${id}-weights`, FOCUS_SLOTS * 4);
  const graph = new GPUCommandGraph<void>(device, {id: `${id}-gather`});
  addKernelPass(graph, {
    id: `${id}-gather`,
    invocationCount: FOCUS_SLOTS,
    bindings: [
      {
        name: 'focusRow',
        view: importGraphBuffer(graph, 'focus', focusRow, 'uint32', 1),
        type: 'u32',
        access: 'read'
      },
      {
        name: 'offsets',
        view: importGraphBuffer(graph, 'offsets', csr.offsets, 'uint32', rows + 1),
        type: 'u32',
        access: 'read'
      },
      {
        name: 'neighbors',
        view: importGraphBuffer(graph, 'neighbors', csr.neighbors, 'uint32', slots),
        type: 'u32',
        access: 'read'
      },
      {
        name: 'weights',
        view: importGraphBuffer(graph, 'weights', csr.weights, 'float32', slots),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'neighborIds',
        view: importGraphBuffer(graph, 'ids', neighborIds, 'uint32', FOCUS_SLOTS),
        type: 'u32',
        access: 'read_write'
      },
      {
        name: 'slotWeights',
        view: importGraphBuffer(graph, 'slot-weights', slotWeights, 'float32', FOCUS_SLOTS),
        type: 'f32',
        access: 'read_write'
      }
    ],
    declarations: `const ROW_COUNT: u32 = ${rows}u;`,
    body: GATHER_BODY
  });
  const compiled = resources.track(graph.compile());
  const reader = new SummaryReader(
    resources,
    `${id}-focus-row`,
    [
      {buffer: neighborIds, size: FOCUS_SLOTS * 4},
      {buffer: slotWeights, size: FOCUS_SLOTS * 4}
    ],
    bytes => {
      const ids = new Uint32Array(bytes, 0, FOCUS_SLOTS);
      const weights = new Float32Array(bytes, FOCUS_SLOTS * 4, FOCUS_SLOTS);
      // Unused slots hold the sentinel 0xffffffff; a row's slots are contiguous from the start.
      let degree = 0;
      while (degree < FOCUS_SLOTS && ids[degree] !== UNUSED_SLOT) degree++;
      props.onRow({ids, weights, degree});
    }
  );
  return {
    slotWeights,
    graph: compiled as CompiledGPUCommandGraph<never>,
    encode: commandEncoder => compiled.encode(commandEncoder, {parameters: undefined}),
    request: commandEncoder => reader.request(commandEncoder),
    isReading: () => reader.isPending,
    stop: () => reader.stop()
  };
}
