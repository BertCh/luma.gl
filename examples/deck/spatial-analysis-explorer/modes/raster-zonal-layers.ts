// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Mode-local helpers of the raster-zonal mode: a scanline polygon rasterizer that turns ZIP
 * polygons into a zone raster on the CPU, and a small kernel that replaces NaN for display.
 */

import type {Binding} from '@luma.gl/core';
import {Computation} from '@luma.gl/engine';
import {
  getViewBinding,
  getViewElementOffset,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {SpatialAnalysisPolygons} from '../spatial-analysis-data';

const WORKGROUP_SIZE = 256;

/**
 * Adds `output = isNaN(input) ? -1 : input`. The shared raster layer's NaN test is a float
 * comparison that Metal's shader compiler may fold away, so NaN zone means would color as the
 * lowest value; a -1 sentinel plus `discardAtOrBelow: 0` hides empty zones reliably.
 */
export function addHideNaNPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    length: number;
    input: GraphDataView<'float32'>;
    output: GraphDataView<'float32'>;
  }
): void {
  const workgroupCount = Math.ceil(props.length / WORKGROUP_SIZE);
  const bindings = [
    {name: 'source', view: props.input, access: 'read' as const},
    {name: 'display', view: props.output, access: 'read_write' as const}
  ];
  const declarations = bindings
    .map(
      (
        binding,
        location
      ) => `const ${binding.name}Offset: u32 = ${getViewElementOffset(binding.view)}u;
@group(0) @binding(${location}) var<storage, ${binding.access}> ${binding.name}: array<f32>;`
    )
    .join('\n');
  const source = /* wgsl */ `
${declarations}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= ${props.length}u) {
    return;
  }
  let value = source[sourceOffset + index];
  let isNaN = (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u;
  display[displayOffset + index] = select(value, -1.0, isNaN);
}`;
  graph.addComputePass({
    id: props.id,
    workload: {
      operation: 'RasterZonalModeKernel',
      variant: props.id,
      commandCount: 1,
      maximumWorkgroupCount: workgroupCount,
      maximumInvocationCount: workgroupCount * WORKGROUP_SIZE,
      readByteLength: props.length * 4,
      writeByteLength: props.length * 4
    },
    resources: [
      {buffer: props.input, usage: 'storage-read' as const},
      {buffer: props.output, usage: 'storage-read-write' as const}
    ],
    compile: ({device}) => {
      const computation = new Computation(device, {
        id: props.id,
        source,
        shaderLayout: {
          bindings: [
            {name: 'source', type: 'read-only-storage' as const, group: 0, location: 0},
            {name: 'display', type: 'storage' as const, group: 0, location: 1}
          ]
        }
      });
      return {
        encode: ({computePass, getBuffer}) => {
          const resolved: Record<string, Binding> = {
            source: getViewBinding(props.input, getBuffer),
            display: getViewBinding(props.output, getBuffer)
          };
          computation.setBindings(resolved);
          computation.dispatch(computePass, workgroupCount);
        },
        destroy: () => computation.destroy()
      };
    }
  });
}

/**
 * Rasterizes polygon features onto a raster grid with a scanline, even-odd fill.
 *
 * For each feature and each raster row, every ring edge that straddles the row's center line adds
 * one crossing; crossings are sorted and cells whose centers lie between consecutive pairs are
 * filled. The cost is O(features x rows + edges x rows touched), with no per-cell
 * point-in-polygon tests. Holes work through the even-odd rule. A later feature overwrites an
 * earlier one where they overlap.
 *
 * @param polygons Polygons in the same planar meters as `bounds`.
 * @param grid Raster `width`, `height`, `[minX, minY, maxX, maxY]` outer edges, and cell size.
 * @returns One value per cell, row 0 at the north edge: feature row + 1, or 0 outside every feature.
 */
export function rasterizePolygonZones(
  polygons: Pick<
    SpatialAnalysisPolygons,
    'polygonPositions' | 'featureOffsets' | 'polygonOffsets' | 'ringOffsets'
  >,
  grid: {
    width: number;
    height: number;
    bounds: readonly [number, number, number, number];
    cellSize: readonly [number, number];
  }
): Uint32Array {
  const {width, height, bounds, cellSize} = grid;
  const {polygonPositions, featureOffsets, polygonOffsets, ringOffsets} = polygons;
  const zones = new Uint32Array(width * height);
  const featureCount = featureOffsets.length - 1;
  const crossings: number[][] = Array.from({length: height}, () => []);
  for (let feature = 0; feature < featureCount; feature++) {
    for (const rowCrossings of crossings) rowCrossings.length = 0;
    let touchedRowMinimum = height;
    let touchedRowMaximum = -1;
    for (let polygon = featureOffsets[feature]; polygon < featureOffsets[feature + 1]; polygon++) {
      for (let ring = polygonOffsets[polygon]; ring < polygonOffsets[polygon + 1]; ring++) {
        const first = ringOffsets[ring];
        const last = ringOffsets[ring + 1];
        const vertexCount = last - first;
        for (let vertex = 0; vertex < vertexCount; vertex++) {
          // Rings close implicitly: the last vertex connects to the first.
          const start = first + vertex;
          const end = first + ((vertex + 1) % vertexCount);
          const startX = polygonPositions[start * 2];
          const startY = polygonPositions[start * 2 + 1];
          const endX = polygonPositions[end * 2];
          const endY = polygonPositions[end * 2 + 1];
          if (startY === endY) continue;
          const rowMinimum = Math.max(
            0,
            Math.floor((bounds[3] - Math.max(startY, endY)) / cellSize[1] - 0.5) - 1
          );
          const rowMaximum = Math.min(
            height - 1,
            Math.ceil((bounds[3] - Math.min(startY, endY)) / cellSize[1] - 0.5) + 1
          );
          for (let row = rowMinimum; row <= rowMaximum; row++) {
            const centerY = bounds[3] - (row + 0.5) * cellSize[1];
            // Half-open rule: the edge counts when exactly one endpoint is above the center line.
            if (startY <= centerY === endY <= centerY) continue;
            const fraction = (centerY - startY) / (endY - startY);
            crossings[row].push(startX + fraction * (endX - startX));
            touchedRowMinimum = Math.min(touchedRowMinimum, row);
            touchedRowMaximum = Math.max(touchedRowMaximum, row);
          }
        }
      }
    }
    for (let row = touchedRowMinimum; row <= touchedRowMaximum; row++) {
      const rowCrossings = crossings[row].sort((a, b) => a - b);
      for (let pair = 0; pair + 1 < rowCrossings.length; pair += 2) {
        const firstColumn = Math.max(
          0,
          Math.ceil((rowCrossings[pair] - bounds[0]) / cellSize[0] - 0.5)
        );
        const lastColumn = Math.min(
          width - 1,
          Math.ceil((rowCrossings[pair + 1] - bounds[0]) / cellSize[0] - 0.5) - 1
        );
        for (let column = firstColumn; column <= lastColumn; column++) {
          zones[row * width + column] = feature + 1;
        }
      }
    }
  }
  return zones;
}
