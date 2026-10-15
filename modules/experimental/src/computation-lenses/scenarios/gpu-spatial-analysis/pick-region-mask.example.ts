// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, Buffer, type Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUPickRegionMask} from '@luma.gl/experimental/gpu-spatial-analysis';

/** A region-pick result and the number of source rows; the caller owns the recording graph. */
export type PickRegionMaskInput = {
  graph: GPUCommandGraph;
  /** `[count, overflow, (objectIndex, batchIndex)...]`, as `GPUIndexPickingTarget.addRegionPass` writes it. */
  picks: GraphDataView<'uint32'>;
  rowCount: number;
};

/** Runs the exact public-API example shown below the pick region mask result figure. */
export async function runExample(device: Device, input: PickRegionMaskInput) {
  const {graph, picks, rowCount} = input;
  // Every view must belong to the supplied device's graph.
  assert(graph.device === device);
  const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
  const mask = createTransientView(graph, 'mask', 'uint32', rowCount, usage);
  const overflow = createTransientView(graph, 'overflow', 'uint32', 1, usage);
  // #region example
  // Only pairs from batch 0 count; duplicate picks of a row collapse into one 1.
  graph.add(
    new GPUPickRegionMask({
      picks,
      batchIndex: 0,
      output: {mask, status: {overflow}}
    })
  );
  // #endregion example
  return {
    graph,
    encodeOptions: {parameters: undefined},
    views: {picks, mask, overflow}
  };
}
