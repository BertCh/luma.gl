// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Mode-local compute kernels shared by the hydrology and cost-distance modes: a small linear
 * compute-node builder (the public equivalent of the contributors' internal kernel helper) and the
 * passes that prepare display rasters, a slope-based friction surface and path line segments.
 */

import type {Binding} from '@luma.gl/core';
import {Computation} from '@luma.gl/engine';
import {
  getViewBinding,
  getViewElementOffset,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';

const WORKGROUP_SIZE = 256;

/** One storage binding of a {@link addKernelPass} kernel. */
type KernelBinding = {
  /** WGSL variable name. `${name}Offset` holds the view's element offset. */
  name: string;
  view: GraphDataView;
  type: 'u32' | 'f32';
  access: 'read' | 'read_write';
};

/**
 * Adds one linear compute pass to `graph`: `invocationCount` invocations in a 1D dispatch of
 * `ceil(invocationCount / 256)` workgroups, each running `body` with `index` in range.
 */
function addKernelPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    bindings: readonly KernelBinding[];
    invocationCount: number;
    declarations?: string;
    body: string;
  }
): void {
  const workgroupCount = Math.ceil(props.invocationCount / WORKGROUP_SIZE);
  if (workgroupCount > graph.device.limits.maxComputeWorkgroupsPerDimension) {
    throw new Error(`${props.id} needs more workgroups than one dispatch dimension allows`);
  }
  const declarations = props.bindings
    .map(
      (
        binding,
        location
      ) => `const ${binding.name}Offset: u32 = ${getViewElementOffset(binding.view)}u;
@group(0) @binding(${location}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;`
    )
    .join('\n');
  const source = /* wgsl */ `
const INVOCATION_COUNT: u32 = ${props.invocationCount}u;
${declarations}
${props.declarations ?? ''}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= INVOCATION_COUNT) {
    return;
  }
  ${props.body}
}`;
  let readByteLength = 0;
  let writeByteLength = 0;
  for (const binding of props.bindings) {
    const byteLength = binding.view.length * binding.view.rowByteLength;
    if (binding.access === 'read') readByteLength += byteLength;
    else writeByteLength += byteLength;
  }
  graph.addComputePass({
    id: props.id,
    workload: {
      operation: 'TerrainModeKernel',
      variant: props.id,
      commandCount: 1,
      maximumWorkgroupCount: workgroupCount,
      maximumInvocationCount: workgroupCount * WORKGROUP_SIZE,
      readByteLength,
      writeByteLength
    },
    resources: props.bindings.map(binding => ({
      buffer: binding.view,
      usage: binding.access === 'read' ? ('storage-read' as const) : ('storage-read-write' as const)
    })),
    compile: ({device}) => {
      const computation = new Computation(device, {
        id: props.id,
        source,
        shaderLayout: {
          bindings: props.bindings.map((binding, location) => ({
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
          for (const binding of props.bindings) {
            resolved[binding.name] = getViewBinding(binding.view, getBuffer);
          }
          computation.setBindings(resolved);
          computation.dispatch(computePass, workgroupCount);
        },
        destroy: () => computation.destroy()
      };
    }
  });
}

const NAN_DECLARATIONS = /* wgsl */ `
fn isNaNValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u; }
fn getQuietNaN() -> f32 { var bits = 0x7fc00000u; return bitcast<f32>(bits); }`;

/**
 * Adds `output = log10(max(accumulation, 1))`, and -1 where accumulation is NaN (invalid or
 * unresolved cells). The sentinel is finite because the raster layer's NaN test is not reliable
 * on every WebGPU compiler; hide those cells with `discardAtOrBelow: -0.5`.
 */
export function addLogAccumulationPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    cellCount: number;
    accumulation: GraphDataView<'float32'>;
    output: GraphDataView<'float32'>;
  }
): void {
  addKernelPass(graph, {
    id: props.id,
    bindings: [
      {name: 'accumulation', view: props.accumulation, type: 'f32', access: 'read'},
      {name: 'logAccumulation', view: props.output, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.cellCount,
    declarations: NAN_DECLARATIONS,
    body: `let value = accumulation[accumulationOffset + index];
  logAccumulation[logAccumulationOffset + index] =
    select(log2(max(value, 1.0)) * 0.30102999566, -1.0, isNaNValue(value));`
  });
}

/**
 * Adds `depth = filled - elevation` (0 where either is NaN), the thickness of the depression
 * fill, so a raster layer can show which cells were raised and by how much.
 */
export function addFillDepthPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    cellCount: number;
    elevation: GraphDataView<'float32'>;
    filled: GraphDataView<'float32'>;
    output: GraphDataView<'float32'>;
  }
): void {
  addKernelPass(graph, {
    id: props.id,
    bindings: [
      {name: 'elevation', view: props.elevation, type: 'f32', access: 'read'},
      {name: 'filled', view: props.filled, type: 'f32', access: 'read'},
      {name: 'depth', view: props.output, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.cellCount,
    declarations: NAN_DECLARATIONS,
    body: `let value = filled[filledOffset + index] - elevation[elevationOffset + index];
  depth[depthOffset + index] = select(value, 0.0, isNaNValue(value));`
  });
}

/**
 * Adds `index = firstTrailingBit(code)` (0 east through 7 north-east, clockwise) for D8 codes
 * 1 to 128, and `0xffffffff` (no data) for terminal cells (code 0) and invalid cells.
 */
export function addDirectionIndexPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    cellCount: number;
    directions: GraphDataView<'uint32'>;
    output: GraphDataView<'uint32'>;
  }
): void {
  addKernelPass(graph, {
    id: props.id,
    bindings: [
      {name: 'directions', view: props.directions, type: 'u32', access: 'read'},
      {name: 'directionIndices', view: props.output, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.cellCount,
    body: `let code = directions[directionsOffset + index];
  let isRoute = code > 0u && code <= 128u && countOneBits(code) == 1u;
  directionIndices[directionIndicesOffset + index] =
    select(0xffffffffu, firstTrailingBit(code), isRoute);`
  });
}

/** Formats a number as a WGSL `f32` literal. */
function getFloatLiteral(value: number): string {
  const text = String(Math.fround(value));
  return /[.eE]/.test(text) ? text : `${text}.0`;
}

/**
 * Adds a kernel turning a cell-index path (target first) into `float32` segments
 * `x0, y0, x1, y1` in raster meters. Segment `i` joins `ids[i]` to `ids[min(i + 1, count - 1)]`;
 * rows at or after `count` are NaN. `segments` is a float32 view of `4 * capacity` values.
 */
export function addPathSegmentsPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    width: number;
    capacity: number;
    bounds: readonly [number, number, number, number];
    cellSize: readonly [number, number];
    ids: GraphDataView<'uint32'>;
    count: GraphDataView<'uint32'>;
    segments: GraphDataView<'float32'>;
  }
): void {
  const {bounds, cellSize} = props;
  addKernelPass(graph, {
    id: props.id,
    bindings: [
      {name: 'ids', view: props.ids, type: 'u32', access: 'read'},
      {name: 'count', view: props.count, type: 'u32', access: 'read'},
      {name: 'segments', view: props.segments, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.capacity,
    declarations: /* wgsl */ `
const GRID_WIDTH: u32 = ${props.width}u;
const ORIGIN_X: f32 = ${getFloatLiteral(bounds[0])};
const ORIGIN_Y: f32 = ${getFloatLiteral(bounds[3])};
const CELL_WIDTH: f32 = ${getFloatLiteral(cellSize[0])};
const CELL_HEIGHT: f32 = ${getFloatLiteral(cellSize[1])};
${NAN_DECLARATIONS}
fn getCellCenter(cell: u32) -> vec2<f32> {
  return vec2<f32>(
    ORIGIN_X + (f32(cell % GRID_WIDTH) + 0.5) * CELL_WIDTH,
    ORIGIN_Y - (f32(cell / GRID_WIDTH) + 0.5) * CELL_HEIGHT
  );
}`,
    body: `let pathCount = count[countOffset];
  let base = segmentsOffset + index * 4u;
  if (index < pathCount) {
    let start = getCellCenter(ids[idsOffset + index]);
    let end = getCellCenter(ids[idsOffset + min(index + 1u, pathCount - 1u)]);
    segments[base] = start.x;
    segments[base + 1u] = start.y;
    segments[base + 2u] = end.x;
    segments[base + 3u] = end.y;
  } else {
    let nan = getQuietNaN();
    segments[base] = nan;
    segments[base + 1u] = nan;
    segments[base + 2u] = nan;
    segments[base + 3u] = nan;
  }`
  });
}

/**
 * Adds `friction = 1 + weight * (slope / 10 degrees)^2`, cost per ground meter, with NaN (an
 * impassable barrier) where slope is NaN. `weight` is the first value of a one-row float32 view
 * the application rewrites per frame; weight 0 gives uniform friction 1 (straight-line distance).
 * GPUCostDistance reads friction through a band whose calibration is compile-time, so a live
 * slope weight needs this small kernel in the graph.
 */
export function addSlopeFrictionPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    cellCount: number;
    slope: GraphDataView<'float32'>;
    weight: GraphDataView<'float32'>;
    output: GraphDataView<'float32'>;
  }
): void {
  addKernelPass(graph, {
    id: props.id,
    bindings: [
      {name: 'slope', view: props.slope, type: 'f32', access: 'read'},
      {name: 'weight', view: props.weight, type: 'f32', access: 'read'},
      {name: 'friction', view: props.output, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.cellCount,
    body: `let steepness = slope[slopeOffset + index] / 10.0;
  // A NaN slope propagates to NaN friction, which the cost contributor treats as a barrier.
  friction[frictionOffset + index] = 1.0 + weight[weightOffset] * steepness * steepness;`
  });
}
