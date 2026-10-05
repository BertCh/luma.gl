// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Mode-local helpers of the group-statistics mode: a kernel that selects one metric column of a
 * packed metrics buffer per frame (so changing the statistic never recompiles), slice helpers for
 * writing several contributor outputs into one buffer, and a class-legend renderer.
 */

import {Buffer, type Binding} from '@luma.gl/core';
import {Computation} from '@luma.gl/engine';
import {
  getViewBinding,
  getViewElementOffset,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/gpgpu/gpu-data';
import {escapeHtml, formatCompact, getPackedColorCss} from './classification-layers';

const WORKGROUP_SIZE = 256;

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
   * @param rowCapacity Rows per slot.
   */
  constructor(graph: GPUCommandGraph<Parameters>, buffer: Buffer, rowCapacity: number) {
    this.graph = graph;
    this.rowCapacity = rowCapacity;
    this.handle = graph.importBuffer(
      {id: buffer.id, byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    );
  }

  /** Returns one view over every row of the buffer, for kernels that address slots themselves. */
  whole(length: number): GraphDataView<'uint32'> {
    return this.graph.createDataView(this.handle, {format: 'uint32', length});
  }

  /**
   * Returns the view over slot `slot` (a four-byte format uses one slot; `uint32x2` spans two).
   * Slot byte offsets are multiples of 256 as long as `rowCapacity` is a multiple of 64.
   */
  view<Format extends GPUVectorFormat>(slot: number, format: Format): GraphDataView<Format> {
    return this.graph.createDataView(this.handle, {
      format,
      length: this.rowCapacity,
      byteOffset: slot * this.rowCapacity * 4
    });
  }
}

/**
 * Adds `display[row] = metric(row)` for a table of `capacity` rows: `selection` holds
 * `[slot, isUnsigned]`, where `slot` indexes `capacity`-row slots of `metrics` and `isUnsigned`
 * converts a `uint32` slot to float. Rows at or past `count[0]` (the GPU-written occupied-row
 * count) get NaN, as does every NaN metric, so downstream class breaks skip them.
 *
 * Both `selection` contents and the metric buffer are per-frame data: switching statistic is a
 * two-word buffer write and one re-encode of this pass.
 */
export function addMetricSelectPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    capacity: number;
    metrics: GraphDataView<'uint32'>;
    selection: GraphDataView<'uint32'>;
    count: GraphDataView<'uint32'>;
    display: GraphDataView<'float32'>;
  }
): void {
  const workgroupCount = Math.ceil(props.capacity / WORKGROUP_SIZE);
  const bindings = [
    {name: 'metrics', view: props.metrics, access: 'read' as const, type: 'u32'},
    {name: 'selection', view: props.selection, access: 'read' as const, type: 'u32'},
    {name: 'tableCount', view: props.count, access: 'read' as const, type: 'u32'},
    {name: 'display', view: props.display, access: 'read_write' as const, type: 'f32'}
  ];
  const declarations = bindings
    .map(
      (
        binding,
        location
      ) => `const ${binding.name}Offset: u32 = ${getViewElementOffset(binding.view)}u;
@group(0) @binding(${location}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;`
    )
    .join('\n');
  const source = /* wgsl */ `
${declarations}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let row = invocation.x;
  if (row >= ${props.capacity}u) {
    return;
  }
  // The count is below 2^31, so the OR adds nothing; it keeps the NaN pattern a runtime value
  // because WGSL rejects a constant NaN.
  var value = bitcast<f32>(0x7fc00000u | (tableCount[tableCountOffset] >> 31u));
  if (row < tableCount[tableCountOffset]) {
    let raw = metrics[metricsOffset + selection[selectionOffset] * ${props.capacity}u + row];
    value = select(bitcast<f32>(raw), f32(raw), selection[selectionOffset + 1u] != 0u);
  }
  display[displayOffset + row] = value;
}`;
  graph.addComputePass({
    id: props.id,
    workload: {
      operation: 'GroupStatisticsMetricSelect',
      variant: props.id,
      commandCount: 1,
      maximumWorkgroupCount: workgroupCount,
      maximumInvocationCount: workgroupCount * WORKGROUP_SIZE,
      readByteLength: props.capacity * 4,
      writeByteLength: props.capacity * 4
    },
    resources: bindings.map(binding => ({
      buffer: binding.view as GraphDataView<'uint32'>,
      usage: binding.access === 'read' ? ('storage-read' as const) : ('storage-read-write' as const)
    })),
    compile: ({device}) => {
      const computation = new Computation(device, {
        id: props.id,
        source,
        shaderLayout: {
          bindings: bindings.map((binding, location) => ({
            name: binding.name,
            type: binding.access === 'read' ? ('read-only-storage' as const) : ('storage' as const),
            group: 0,
            location
          }))
        }
      });
      return {
        encode: ({computePass, getBuffer}) => {
          const resolved: Record<string, Binding> = {};
          for (const binding of bindings) {
            resolved[binding.name] = getViewBinding(
              binding.view as GraphDataView<'uint32'>,
              getBuffer
            );
          }
          computation.setBindings(resolved);
          computation.dispatch(computePass, workgroupCount);
        },
        destroy: () => computation.destroy()
      };
    }
  });
}

/**
 * Adds a pass that copies `sources.length` float32 or uint32 columns (bit for bit) into
 * consecutive `capacity`-row slots of a metrics buffer, starting at `firstSlot`.
 *
 * `GPUCellTableCompare` writes its outputs and reads them back inside the same contributor, so they
 * cannot be slices of one buffer (WebGPU rejects one buffer bound both read-only and writable in
 * one dispatch scope). They get their own buffers and this pass packs them for the metric select.
 */
export function addSlotPackPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    capacity: number;
    sources: readonly (GraphDataView<'float32'> | GraphDataView<'uint32'>)[];
    destination: GraphDataView<'uint32'>;
    firstSlot: number;
  }
): void {
  const workgroupCount = Math.ceil(props.capacity / WORKGROUP_SIZE);
  const bindings = [
    ...props.sources.map((view, index) => ({
      name: `source${index}`,
      view,
      access: 'read' as const
    })),
    {name: 'destination', view: props.destination, access: 'read_write' as const}
  ];
  const declarations = bindings
    .map(
      (
        binding,
        location
      ) => `const ${binding.name}Offset: u32 = ${getViewElementOffset(binding.view)}u;
@group(0) @binding(${location}) var<storage, ${binding.access}> ${binding.name}: array<u32>;`
    )
    .join('\n');
  const copies = props.sources
    .map(
      (_, index) =>
        `  destination[destinationOffset + ${props.firstSlot + index}u * ${props.capacity}u + row] = source${index}[source${index}Offset + row];`
    )
    .join('\n');
  const source = /* wgsl */ `
${declarations}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let row = invocation.x;
  if (row >= ${props.capacity}u) {
    return;
  }
${copies}
}`;
  graph.addComputePass({
    id: props.id,
    workload: {
      operation: 'GroupStatisticsSlotPack',
      variant: props.id,
      commandCount: 1,
      maximumWorkgroupCount: workgroupCount,
      maximumInvocationCount: workgroupCount * WORKGROUP_SIZE,
      readByteLength: props.capacity * 4 * props.sources.length,
      writeByteLength: props.capacity * 4 * props.sources.length
    },
    resources: bindings.map(binding => ({
      buffer: binding.view as GraphDataView<'uint32'>,
      usage: binding.access === 'read' ? ('storage-read' as const) : ('storage-read-write' as const)
    })),
    compile: ({device}) => {
      const computation = new Computation(device, {
        id: props.id,
        source,
        shaderLayout: {
          bindings: bindings.map((binding, location) => ({
            name: binding.name,
            type: binding.access === 'read' ? ('read-only-storage' as const) : ('storage' as const),
            group: 0,
            location
          }))
        }
      });
      return {
        encode: ({computePass, getBuffer}) => {
          const resolved: Record<string, Binding> = {};
          for (const binding of bindings) {
            resolved[binding.name] = getViewBinding(
              binding.view as GraphDataView<'uint32'>,
              getBuffer
            );
          }
          computation.setBindings(resolved);
          computation.dispatch(computePass, workgroupCount);
        },
        destroy: () => computation.destroy()
      };
    }
  });
}

/** Builds the HTML of a class legend: swatch, range, share bar and count per class. */
export function getClassLegendHtml(props: {
  heading: string;
  edges: ArrayLike<number>;
  counts: ArrayLike<number>;
  palette: ArrayLike<number>;
  classCount: number;
  noDataLabel?: string;
}): string {
  const {edges, counts, palette, classCount} = props;
  let total = 0;
  for (let index = 0; index < classCount; index++) total += counts[index];
  const html = [`<div>${escapeHtml(props.heading)}</div>`];
  for (let index = 0; index < classCount; index++) {
    const share = total > 0 ? counts[index] / total : 0;
    const color = getPackedColorCss(palette[index] ?? 0);
    html.push(
      `<div style="display:flex;align-items:center;gap:6px;margin-top:2px">` +
        `<span style="width:12px;height:12px;border-radius:2px;flex:none;background:${color}"></span>` +
        `<span style="width:116px;flex:none;font:10px ui-monospace,monospace">${formatCompact(edges[index])} – ${formatCompact(edges[index + 1])}</span>` +
        `<span style="flex:1;height:6px;background:rgba(255,255,255,.08);border-radius:3px"><span style="display:block;height:6px;border-radius:3px;width:${(share * 100).toFixed(1)}%;background:${color}"></span></span>` +
        `<span style="width:40px;text-align:right;font:10px ui-monospace,monospace">${counts[index]}</span></div>`
    );
  }
  if (props.noDataLabel) {
    html.push(`<div style="margin-top:3px">${escapeHtml(props.noDataLabel)}</div>`);
  }
  return html.join('');
}
