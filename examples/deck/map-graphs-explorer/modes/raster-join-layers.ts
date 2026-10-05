// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Mode-local helpers of the raster-join mode: a per-cell display kernel that turns the zone raster
 * and the per-zone join columns into one float per cell (the metric is a buffer, so switching it
 * never recompiles), and a CPU point-in-polygon lookup for the hover tooltip.
 */

import type {Binding} from '@luma.gl/core';
import {Computation} from '@luma.gl/engine';
import {
  getViewBinding,
  getViewElementOffset,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {MapGraphsPolygons} from '../map-graphs-data';

const WORKGROUP_SIZE = 256;

/** Metric written per cell by {@link addRasterJoinDisplayPass}. */
export const RASTER_JOIN_METRIC = {
  count: 0,
  sum: 1,
  boundaryShare: 2
} as const;

/** Display value of a cell that holds no zone; hide it with `discardAtOrBelow`. */
export const RASTER_JOIN_HIDDEN = -1;

/**
 * Adds `display[cell] = metric(zone(cell))`, or {@link RASTER_JOIN_HIDDEN} for cells without a
 * zone. Metric 0 is the joined point count, 1 the joined value sum, 2 the share of the zone's
 * joined points that sit in boundary cells (the per-zone error bound, as a fraction).
 *
 * The raster layer cannot do this lookup itself through `valueIndices`: cells without a zone hold
 * `0xffffffff`, which would index far past the per-zone columns.
 */
export function addRasterJoinDisplayPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    cellCount: number;
    zoneCount: number;
    zones: GraphDataView<'uint32'>;
    counts: GraphDataView<'uint32'>;
    sums: GraphDataView<'float32'>;
    boundaryCounts: GraphDataView<'uint32'>;
    /** One uint32: a {@link RASTER_JOIN_METRIC} value. */
    metric: GraphDataView<'uint32'>;
    display: GraphDataView<'float32'>;
  }
): void {
  const workgroupCount = Math.ceil(props.cellCount / WORKGROUP_SIZE);
  const bindings = [
    {name: 'zones', view: props.zones, access: 'read' as const, type: 'u32'},
    {name: 'counts', view: props.counts, access: 'read' as const, type: 'u32'},
    {name: 'sums', view: props.sums, access: 'read' as const, type: 'f32'},
    {name: 'boundaryCounts', view: props.boundaryCounts, access: 'read' as const, type: 'u32'},
    {name: 'metric', view: props.metric, access: 'read' as const, type: 'u32'},
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
  let cell = invocation.x;
  if (cell >= ${props.cellCount}u) {
    return;
  }
  let zone = zones[zonesOffset + cell];
  var value = ${RASTER_JOIN_HIDDEN}.0;
  if (zone < ${props.zoneCount}u) {
    let count = f32(counts[countsOffset + zone]);
    let mode = metric[metricOffset];
    if (mode == ${RASTER_JOIN_METRIC.count}u) {
      value = count;
    } else if (mode == ${RASTER_JOIN_METRIC.sum}u) {
      value = sums[sumsOffset + zone];
    } else {
      value = select(0.0, f32(boundaryCounts[boundaryCountsOffset + zone]) / count, count > 0.0);
    }
  }
  display[displayOffset + cell] = value;
}`;
  graph.addComputePass({
    id: props.id,
    workload: {
      operation: 'RasterJoinModeKernel',
      variant: props.id,
      commandCount: 1,
      maximumWorkgroupCount: workgroupCount,
      maximumInvocationCount: workgroupCount * WORKGROUP_SIZE,
      readByteLength: props.cellCount * 4,
      writeByteLength: props.cellCount * 4
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
 * Returns the feature row of the first polygon feature containing `[x, y]` (even-odd over each
 * polygon's rings, smallest row wins), or -1.
 */
export function findContainingFeature(
  polygons: Pick<
    MapGraphsPolygons,
    'polygonPositions' | 'featureOffsets' | 'polygonOffsets' | 'ringOffsets'
  >,
  x: number,
  y: number
): number {
  const {polygonPositions, featureOffsets, polygonOffsets, ringOffsets} = polygons;
  const featureCount = featureOffsets.length - 1;
  for (let feature = 0; feature < featureCount; feature++) {
    for (let polygon = featureOffsets[feature]; polygon < featureOffsets[feature + 1]; polygon++) {
      let inside = false;
      for (let ring = polygonOffsets[polygon]; ring < polygonOffsets[polygon + 1]; ring++) {
        const first = ringOffsets[ring];
        const vertexCount = ringOffsets[ring + 1] - first;
        for (let vertex = 0; vertex < vertexCount; vertex++) {
          const start = first + vertex;
          const end = first + ((vertex + 1) % vertexCount);
          const startY = polygonPositions[start * 2 + 1];
          const endY = polygonPositions[end * 2 + 1];
          if (startY > y === endY > y) continue;
          const startX = polygonPositions[start * 2];
          const endX = polygonPositions[end * 2];
          if (x < startX + ((y - startY) / (endY - startY)) * (endX - startX)) inside = !inside;
        }
      }
      if (inside) return feature;
    }
  }
  return -1;
}
