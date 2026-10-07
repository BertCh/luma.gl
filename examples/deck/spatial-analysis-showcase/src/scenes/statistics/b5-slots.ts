// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/gpgpu/gpu-data';
import {addKernelPass} from '../../engine/mode-kernels';

/**
 * Hands out slices of one `Buffer` as graph data views. Every slice starts on a whole number of
 * `rowCapacity`-row slots, so each contributor output of a table with `rowCapacity` rows lands at a
 * fixed slot and one display kernel can select among them with a slot number.
 */
export class SlotSlices<Parameters> {
  private readonly graph: GPUCommandGraph<Parameters>;
  private readonly handle: ReturnType<GPUCommandGraph<Parameters>['importBuffer']>;
  private readonly rowCapacity: number;

  /**
   * @param graph Graph that imports `buffer`. A graph may import a buffer only once.
   * @param buffer Metrics buffer of `slotCount * rowCapacity` four-byte rows.
   * @param rowCapacity Rows per slot; a multiple of 64 keeps slot offsets 256-byte aligned.
   */
  constructor(graph: GPUCommandGraph<Parameters>, buffer: Buffer, rowCapacity: number) {
    this.graph = graph;
    this.rowCapacity = rowCapacity;
    this.handle = graph.importBuffer(
      {id: buffer.id, byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    );
  }

  /** One view over every row of the buffer, for kernels that address slots themselves. */
  whole(length: number): GraphDataView<'uint32'> {
    return this.graph.createDataView(this.handle, {format: 'uint32', length});
  }

  /**
   * The view over slot `slot` (a four-byte format uses one slot; `uint32x2` spans two).
   * `length` defaults to the slot's row capacity; a dense table of `keyCount` rows passes it.
   */
  view<Format extends GPUVectorFormat>(
    slot: number,
    format: Format,
    length: number = this.rowCapacity
  ): GraphDataView<Format> {
    return this.graph.createDataView(this.handle, {
      format,
      length,
      byteOffset: slot * this.rowCapacity * 4
    });
  }
}

/**
 * Adds `output[row] = metrics[slot * capacity + zone(row)]`: every source row takes the value its
 * zone has in one slot of a zone-metrics buffer. `selection` holds `[slot, isUnsigned]`; rows
 * whose zone is out of range or whose zone value is NaN get NaN, so the layer shows no data.
 * Switching the metric is a two-word buffer write and one re-encode of this pass.
 */
export function addZoneGatherPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    rowCount: number;
    zoneCapacity: number;
    zoneCount: number;
    metrics: GraphDataView<'uint32'>;
    selection: GraphDataView<'uint32'>;
    zoneIds: GraphDataView<'uint32'>;
    output: GraphDataView<'float32'>;
  }
): void {
  addKernelPass(graph, {
    id: props.id,
    invocationCount: props.rowCount,
    bindings: [
      {name: 'metrics', view: props.metrics, type: 'u32', access: 'read'},
      {name: 'selection', view: props.selection, type: 'u32', access: 'read'},
      {name: 'zoneIds', view: props.zoneIds, type: 'u32', access: 'read'},
      {name: 'output', view: props.output, type: 'f32', access: 'read_write'}
    ],
    body: /* wgsl */ `
  let zone = zoneIds[zoneIdsOffset + index];
  // Keeps the NaN pattern a runtime value because WGSL rejects a constant NaN.
  var value = bitcast<f32>(0x7fc00000u | (zone >> 31u));
  if (zone < ${props.zoneCount}u) {
    let raw = metrics[metricsOffset + selection[selectionOffset] * ${props.zoneCapacity}u + zone];
    value = select(bitcast<f32>(raw), f32(raw), selection[selectionOffset + 1u] != 0u);
  }
  output[outputOffset + index] = value;`
  });
}
