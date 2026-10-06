// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Shared scaffolding of the Recipes mode: the scene contract, a toolkit that creates the buffers
 * and the one graph of a scene, a generic element-wise compute pass for the small adapters the
 * mode needs (the recipes themselves are all contributors), and procedural district polygons.
 */

import type {Layer} from '@deck.gl/core';
import {Buffer, type Binding, type CommandEncoder, type Device} from '@luma.gl/core';
import {Computation} from '@luma.gl/engine';
import {
  getViewBinding,
  getViewElementOffset,
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {getGPUVectorFormatInfo, type GPUVectorFormat} from '@luma.gl/gpgpu/gpu-data';
import type {GPUParameterBuffer} from '@luma.gl/experimental/gpu-spatial-analysis';
import {importGraphBuffer} from '../graph-buffers';
import type {
  SpatialAnalysisFrame,
  SpatialAnalysisModeContext,
  SpatialAnalysisPointerEvent
} from '../spatial-analysis-mode';
import {SpatialAnalysisResources} from '../spatial-analysis-resources';
import {createSeededRandom} from '../spatial-analysis-data';
import {SummaryReader} from './summary-reader';

/** A live slider of the active recipe. `position` and `value` are in the parameter's own units. */
export type RecipeSliderParameter = {
  kind: 'slider';
  /** Name shown above the slider. */
  label: string;
  minimum: number;
  maximum: number;
  step: number;
  value: number;
  /** Text shown beside the slider. */
  format: (value: number) => string;
  /** Called with the new value; writes a parameter buffer, never recompiles. */
  onChange: (value: number) => void;
};

/** A live toggle of the active recipe. */
export type RecipeToggleParameter = {
  kind: 'toggle';
  label: string;
  value: boolean;
  onChange: (value: boolean) => void;
};

/** One per-frame parameter exposed by a recipe scene. */
export type RecipeParameter = RecipeSliderParameter | RecipeToggleParameter;

/** What a scene may ask of the mode while it runs. */
export type RecipeSceneHost = {
  /** Shows up to six `[label, value]` result lines. Unused lines are hidden. */
  setOutputs: (lines: readonly (readonly [string, string])[]) => void;
  /** Rebuilds the deck layers from {@link RecipeScene.getLayers}. */
  updateLayers: () => void;
  /** Data and device services of the explorer. */
  context: SpatialAnalysisModeContext;
};

/** A built recipe: one compiled graph, its parameters, readouts and layers. */
export type RecipeScene = {
  /** The single compiled graph of the recipe. */
  compiled: CompiledGPUCommandGraph<void>;
  /** Number of contributors the recipe added to the graph. */
  contributorCount: number;
  /** Contributors and adapters in chain order, one string each. */
  chain: readonly string[];
  /** Up to four sliders and two toggles. */
  parameters: readonly RecipeParameter[];
  /** What the colors and marks mean. */
  legend: string;
  /** Provenance of the data and of any synthetic input. */
  dataNote: string;
  /** Writes parameters and encodes the graph when something changed. */
  encode: (commandEncoder: CommandEncoder, frame: SpatialAnalysisFrame) => void;
  /** Layers drawing the recipe outputs. */
  getLayers: () => Layer[];
  onClick?: (event: SpatialAnalysisPointerEvent) => boolean;
  getTooltip?: (event: SpatialAnalysisPointerEvent) => string | null;
  /** Releases every GPU resource of the scene. */
  destroy: () => void;
};

/** Builds one scene. */
export type RecipeSceneBuilder = (host: RecipeSceneHost) => Promise<RecipeScene>;

/** Buffer plus the graph view of it. */
export type KitBuffer<Format extends GPUVectorFormat> = {
  buffer: Buffer;
  view: GraphDataView<Format>;
};

/**
 * Owns the resources and the one graph of a scene. `input`, `output` and `parameter` create a
 * tracked buffer and import it into the graph, so a scene reads like the recipe specs.
 */
export class RecipeKit {
  readonly graph: GPUCommandGraph<void>;
  readonly resources: SpatialAnalysisResources;

  constructor(
    readonly device: Device,
    readonly id: string
  ) {
    this.graph = new GPUCommandGraph<void>(device, {id});
    this.resources = new SpatialAnalysisResources(device, id);
  }

  /** Uploads `data` and imports it as a read-only view of `length` rows. */
  input<Format extends GPUVectorFormat>(
    name: string,
    data: Float32Array | Uint32Array | Int32Array,
    format: Format,
    length?: number
  ): KitBuffer<Format> {
    const buffer = this.resources.createBuffer(name, data);
    return {buffer, view: importGraphBuffer(this.graph, name, buffer, format, length)};
  }

  /** Creates a zero-filled caller-owned output of `length` rows, plus `extraRows` spare rows. */
  output<Format extends GPUVectorFormat>(
    name: string,
    format: Format,
    length: number,
    extraRows = 0
  ): KitBuffer<Format> {
    const rowBytes = getGPUVectorFormatInfo(format).byteLength;
    const buffer = this.resources.createBuffer(name, Math.max(1, length + extraRows) * rowBytes);
    return {buffer, view: importGraphBuffer(this.graph, name, buffer, format, length)};
  }

  /** Creates a per-frame parameter buffer and its graph view. */
  parameter<Format extends 'float32' | 'uint32' | 'sint32'>(
    name: string,
    format: Format,
    values: Float32Array | Uint32Array | Int32Array
  ): {parameters: GPUParameterBuffer<Format>; view: GraphDataView<Format>} {
    const parameters = this.resources.createParameterBuffer(name, format, values.length, values);
    return {parameters, view: parameters.importToGraph(this.graph) as GraphDataView<Format>};
  }

  /** Compiles the graph and tracks it for release. */
  compile(): CompiledGPUCommandGraph<void> {
    return this.resources.track(this.graph.compile());
  }
}

/** One binding of {@link addKernelPass}: a graph view read or written by the kernel body. */
export type KernelBinding = {
  name: string;
  view: GraphDataView<GPUVectorFormat>;
  access: 'read' | 'read_write';
  /** WGSL element type, for example `u32`, `f32`, `vec2<f32>`. */
  type: string;
};

/**
 * Adds a one-invocation-per-element compute pass to `graph`. `body` runs for `index < count`;
 * every binding `name` is an array, and `<name>Offset` is the element offset of its view.
 */
export function addKernelPass(
  graph: GPUCommandGraph<void>,
  props: {id: string; count: number; bindings: readonly KernelBinding[]; body: string}
): void {
  const workgroupSize = 256;
  const workgroupCount = Math.max(1, Math.ceil(props.count / workgroupSize));
  const declarations = props.bindings
    .map(
      (binding, location) =>
        `const ${binding.name}Offset: u32 = ${getViewElementOffset(binding.view)}u;\n` +
        `@group(0) @binding(${location}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;`
    )
    .join('\n');
  const source = /* wgsl */ `
${declarations}
@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= ${props.count}u) {
    return;
  }
  ${props.body}
}`;
  graph.addComputePass({
    id: props.id,
    workload: {
      operation: 'RecipesDemoKernel',
      variant: props.id,
      commandCount: 1,
      maximumWorkgroupCount: workgroupCount,
      maximumInvocationCount: workgroupCount * workgroupSize,
      readByteLength: props.count * 4,
      writeByteLength: props.count * 4
    },
    resources: props.bindings.map(binding => ({
      buffer: binding.view as GraphDataView<'uint32'>,
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

/** Polygon features in the GeoArrow-style layout the recipes read, plus drawable outlines. */
export type DistrictPolygons = {
  polygonPositions: Float32Array;
  featureOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  ringOffsets: Uint32Array;
  /** `x0, y0, x1, y1` per ring edge. */
  outlineSegments: Float32Array;
  outlineFeatureRows: Uint32Array;
  featureCount: number;
  columns: number;
  rows: number;
  bounds: readonly [number, number, number, number];
};

/**
 * Builds a `columns x rows` mesh of quadrilateral districts over `bounds` (planar meters). Interior
 * vertices are shared by neighbors and jittered by `jitter` of a cell, so the districts touch
 * exactly (queen contiguity works) without being a plain grid. `jitter = 0` gives rectangles.
 */
export function createDistrictPolygons(
  bounds: readonly [number, number, number, number],
  columns: number,
  rows: number,
  jitter: number,
  seed: number
): DistrictPolygons {
  const random = createSeededRandom(seed);
  const cellWidth = (bounds[2] - bounds[0]) / columns;
  const cellHeight = (bounds[3] - bounds[1]) / rows;
  const vertexX = new Float32Array((columns + 1) * (rows + 1));
  const vertexY = new Float32Array((columns + 1) * (rows + 1));
  for (let row = 0; row <= rows; row++) {
    for (let column = 0; column <= columns; column++) {
      const interior = column > 0 && column < columns && row > 0 && row < rows;
      const shake = interior ? jitter : 0;
      vertexX[row * (columns + 1) + column] =
        bounds[0] + (column + (random() - 0.5) * shake) * cellWidth;
      vertexY[row * (columns + 1) + column] =
        bounds[1] + (row + (random() - 0.5) * shake) * cellHeight;
    }
  }
  const featureCount = columns * rows;
  const polygonPositions = new Float32Array(featureCount * 8);
  const offsets = new Uint32Array(featureCount + 1);
  const ringOffsets = new Uint32Array(featureCount + 1);
  const outlineSegments = new Float32Array(featureCount * 16);
  const outlineFeatureRows = new Uint32Array(featureCount * 4);
  for (let feature = 0; feature < featureCount; feature++) {
    const column = feature % columns;
    const row = Math.floor(feature / columns);
    const corners = [
      row * (columns + 1) + column,
      row * (columns + 1) + column + 1,
      (row + 1) * (columns + 1) + column + 1,
      (row + 1) * (columns + 1) + column
    ];
    corners.forEach((corner, index) => {
      polygonPositions[feature * 8 + index * 2] = vertexX[corner];
      polygonPositions[feature * 8 + index * 2 + 1] = vertexY[corner];
    });
    for (let edge = 0; edge < 4; edge++) {
      const start = corners[edge];
      const end = corners[(edge + 1) % 4];
      outlineSegments.set(
        [vertexX[start], vertexY[start], vertexX[end], vertexY[end]],
        feature * 16 + edge * 4
      );
      outlineFeatureRows[feature * 4 + edge] = feature;
    }
    offsets[feature] = feature;
    ringOffsets[feature] = feature * 4;
  }
  offsets[featureCount] = featureCount;
  ringOffsets[featureCount] = featureCount * 4;
  return {
    polygonPositions,
    featureOffsets: offsets,
    polygonOffsets: offsets,
    ringOffsets,
    outlineSegments,
    outlineFeatureRows,
    featureCount,
    columns,
    rows,
    bounds
  };
}

/** Returns the 0..1 quantile of `values` (a copy is sorted). */
export function getQuantile(values: ArrayLike<number>, fraction: number): number {
  const sorted = Float64Array.from(values).sort();
  return sorted[Math.min(sorted.length - 1, Math.floor(fraction * (sorted.length - 1)))];
}

/** `[minX, minY, maxX, maxY]` of the `margin` to `1 - margin` quantile of interleaved `x, y` positions. */
export function getCoreBounds(
  positions: Float32Array,
  margin = 0.02
): [number, number, number, number] {
  const count = positions.length / 2;
  const xs = new Float32Array(count);
  const ys = new Float32Array(count);
  for (let index = 0; index < count; index++) {
    xs[index] = positions[index * 2];
    ys[index] = positions[index * 2 + 1];
  }
  return [
    getQuantile(xs, margin),
    getQuantile(ys, margin),
    getQuantile(xs, 1 - margin),
    getQuantile(ys, 1 - margin)
  ];
}

/**
 * Rasterizes polygons by point-in-polygon on cell centers (even-odd). Returns, per cell with row 0
 * at the south edge, the feature row or `featureCount` outside every feature. The extra value is
 * the row of a transparent entry the caller appends to a per-feature color buffer.
 */
export function rasterizeFeatureRows(
  polygons: DistrictPolygons,
  bounds: readonly [number, number, number, number],
  width: number,
  height: number
): Uint32Array {
  const cells = new Uint32Array(width * height).fill(polygons.featureCount);
  const cellWidth = (bounds[2] - bounds[0]) / width;
  const cellHeight = (bounds[3] - bounds[1]) / height;
  const {polygonPositions: positions} = polygons;
  for (let feature = 0; feature < polygons.featureCount; feature++) {
    const ring = feature * 8;
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (let vertex = 0; vertex < 4; vertex++) {
      minX = Math.min(minX, positions[ring + vertex * 2]);
      maxX = Math.max(maxX, positions[ring + vertex * 2]);
      minY = Math.min(minY, positions[ring + vertex * 2 + 1]);
      maxY = Math.max(maxY, positions[ring + vertex * 2 + 1]);
    }
    const firstColumn = Math.max(0, Math.floor((minX - bounds[0]) / cellWidth));
    const lastColumn = Math.min(width - 1, Math.ceil((maxX - bounds[0]) / cellWidth));
    const firstRow = Math.max(0, Math.floor((minY - bounds[1]) / cellHeight));
    const lastRow = Math.min(height - 1, Math.ceil((maxY - bounds[1]) / cellHeight));
    for (let row = firstRow; row <= lastRow; row++) {
      const y = bounds[1] + (row + 0.5) * cellHeight;
      for (let column = firstColumn; column <= lastColumn; column++) {
        const x = bounds[0] + (column + 0.5) * cellWidth;
        let inside = false;
        for (let vertex = 0, previous = 3; vertex < 4; previous = vertex++) {
          const ax = positions[ring + vertex * 2];
          const ay = positions[ring + vertex * 2 + 1];
          const bx = positions[ring + previous * 2];
          const by = positions[ring + previous * 2 + 1];
          if (ay > y !== by > y && x < ((bx - ax) * (y - ay)) / (by - ay) + ax) inside = !inside;
        }
        if (inside) cells[row * width + column] = feature;
      }
    }
  }
  return cells;
}

/** Converts planar meters to longitude/latitude degrees for a whole position array. */
export function metersToDegrees(
  positions: Float32Array,
  unproject: (x: number, y: number) => [number, number]
): Float32Array {
  const degrees = new Float32Array(positions.length);
  for (let index = 0; index < positions.length; index += 2) {
    const [longitude, latitude] = unproject(positions[index], positions[index + 1]);
    degrees[index] = longitude;
    degrees[index + 1] = latitude;
  }
  return degrees;
}

export {SummaryReader};

/**
 * Encodes `compiled` only when {@link markDirty} was called (every recipe here has static inputs),
 * and requests the summary readback afterwards.
 */
export class DirtyEncoder {
  private dirty = true;

  constructor(
    private readonly compiled: CompiledGPUCommandGraph<void>,
    private readonly reader: SummaryReader | null
  ) {}

  /** Schedules one encode of the graph on the next frame. */
  markDirty(): void {
    this.dirty = true;
  }

  /** Encodes when dirty, then lets the reader record or deliver its copy. */
  encode(commandEncoder: CommandEncoder): void {
    if (this.dirty) {
      this.compiled.encode(commandEncoder, {parameters: undefined});
      this.dirty = false;
      this.reader?.request(commandEncoder);
    } else {
      this.reader?.flush(commandEncoder);
    }
  }
}

/** Splits a summary readback into typed slices. `sizes` are byte lengths in order. */
export function sliceSummary(bytes: ArrayBuffer, sizes: readonly number[]) {
  const slices: {u32: Uint32Array; i32: Int32Array; f32: Float32Array}[] = [];
  let offset = 0;
  for (const size of sizes) {
    const part = bytes.slice(offset, offset + size);
    slices.push({
      u32: new Uint32Array(part),
      i32: new Int32Array(part),
      f32: new Float32Array(part)
    });
    offset += size;
  }
  return slices;
}

/** Returns the district containing `[x, y]`, or -1. Districts are the quads of {@link createDistrictPolygons}. */
export function createDistrictLookup(polygons: DistrictPolygons) {
  const {polygonPositions: positions, featureCount} = polygons;
  const boxes = new Float32Array(featureCount * 4);
  for (let feature = 0; feature < featureCount; feature++) {
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (let vertex = 0; vertex < 4; vertex++) {
      minX = Math.min(minX, positions[feature * 8 + vertex * 2]);
      maxX = Math.max(maxX, positions[feature * 8 + vertex * 2]);
      minY = Math.min(minY, positions[feature * 8 + vertex * 2 + 1]);
      maxY = Math.max(maxY, positions[feature * 8 + vertex * 2 + 1]);
    }
    boxes.set([minX, minY, maxX, maxY], feature * 4);
  }
  return (x: number, y: number): number => {
    for (let feature = 0; feature < featureCount; feature++) {
      if (x < boxes[feature * 4] || x > boxes[feature * 4 + 2]) continue;
      if (y < boxes[feature * 4 + 1] || y > boxes[feature * 4 + 3]) continue;
      let inside = false;
      for (let vertex = 0, previous = 3; vertex < 4; previous = vertex++) {
        const ax = positions[feature * 8 + vertex * 2];
        const ay = positions[feature * 8 + vertex * 2 + 1];
        const bx = positions[feature * 8 + previous * 2];
        const by = positions[feature * 8 + previous * 2 + 1];
        if (ay > y !== by > y && x < ((bx - ax) * (y - ay)) / (by - ay) + ax) inside = !inside;
      }
      if (inside) return feature;
    }
    return -1;
  };
}

/** Counts interleaved `x, y` points per district (an input column the recipes read, built on the CPU). */
export function countPointsPerDistrict(
  polygons: DistrictPolygons,
  positions: Float32Array
): Float32Array {
  const find = createDistrictLookup(polygons);
  const counts = new Float32Array(polygons.featureCount);
  for (let index = 0; index < positions.length; index += 2) {
    const feature = find(positions[index], positions[index + 1]);
    if (feature >= 0) counts[feature]++;
  }
  return counts;
}

/** Interleaved first and last vertex of every trip: its pickup and drop-off. */
export function getTripEndpoints(
  vertexPositions: Float32Array,
  tripOffsets: Uint32Array
): Float32Array {
  const tripCount = tripOffsets.length - 1;
  const endpoints = new Float32Array(tripCount * 4);
  for (let trip = 0; trip < tripCount; trip++) {
    const first = tripOffsets[trip];
    const last = tripOffsets[trip + 1] - 1;
    endpoints.set(vertexPositions.subarray(first * 2, first * 2 + 2), trip * 4);
    endpoints.set(vertexPositions.subarray(last * 2, last * 2 + 2), trip * 4 + 2);
  }
  return endpoints;
}

/** One `x0, y0, x1, y1` row per consecutive vertex pair inside a trip, plus the trip of each row. */
export function getTripSegments(
  vertexPositions: Float32Array,
  tripOffsets: Uint32Array
): {segments: Float32Array; segmentCount: number} {
  const tripCount = tripOffsets.length - 1;
  let segmentCount = 0;
  for (let trip = 0; trip < tripCount; trip++) {
    segmentCount += Math.max(0, tripOffsets[trip + 1] - tripOffsets[trip] - 1);
  }
  const segments = new Float32Array(segmentCount * 4);
  let row = 0;
  for (let trip = 0; trip < tripCount; trip++) {
    for (let vertex = tripOffsets[trip]; vertex < tripOffsets[trip + 1] - 1; vertex++, row++) {
      segments.set(vertexPositions.subarray(vertex * 2, vertex * 2 + 4), row * 4);
    }
  }
  return {segments, segmentCount};
}
