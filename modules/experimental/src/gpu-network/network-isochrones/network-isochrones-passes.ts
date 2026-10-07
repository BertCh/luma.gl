// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../utils/wgsl-kernel-nodes';

const OPERATION = 'GPUNetworkIsochrones';

/** Raster accumulation rule of the splat: smallest or largest cost per pixel. */
export type GPUNetworkIsochronesMode = 'min' | 'max';

/**
 * Largest facility count of the facility-labelled raster: the low 8 bits of each accumulated pixel
 * word carry the facility row (255 is reserved), the high 24 bits the cost. @internal
 */
export const GPU_NETWORK_ISOCHRONES_MAXIMUM_RASTER_FACILITIES = 255;

/** Encoded value of an untouched pixel for each mode. @internal */
export function getIsochronesEncodedInitialValue(mode: GPUNetworkIsochronesMode): number {
  return mode === 'min' ? 0xffffffff : 0;
}

/**
 * Writes the {@link GPUIsobands} parameter view from the isochrone parameters (cell sizes from the
 * extent and the raster size). @internal
 */
export function createIsochronesBandParametersNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    width: number;
    height: number;
    parameters: GraphDataView<'float32'>;
    bandParameters: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'band-parameters',
    bindings: [
      {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'bandParams', view: props.bandParameters, type: 'f32', access: 'read_write'}
    ],
    invocationCount: 1,
    declarations: `const WIDTH: f32 = ${props.width}.0;
const HEIGHT: f32 = ${props.height}.0;`,
    body: `bandParams[bandParamsOffset] = params[paramsOffset];
  bandParams[bandParamsOffset + 1u] = params[paramsOffset + 7u];
  bandParams[bandParamsOffset + 2u] = params[paramsOffset + 8u];
  bandParams[bandParamsOffset + 3u] = 0.0;
  bandParams[bandParamsOffset + 4u] = params[paramsOffset + 1u];
  bandParams[bandParamsOffset + 5u] = params[paramsOffset + 2u];
  bandParams[bandParamsOffset + 6u] = params[paramsOffset + 3u];
  bandParams[bandParamsOffset + 7u] = params[paramsOffset + 4u];
  bandParams[bandParamsOffset + 8u] = (params[paramsOffset + 3u] - params[paramsOffset + 1u]) / WIDTH;
  bandParams[bandParamsOffset + 9u] = (params[paramsOffset + 4u] - params[paramsOffset + 2u]) / HEIGHT;
  bandParams[bandParamsOffset + 10u] = 0.0;
  bandParams[bandParamsOffset + 11u] = 0.0;`
  });
}

/**
 * One thread per edge: samples the edge at a bounded number of points, interpolates the cost
 * linearly from the source node cost along the edge weight, and applies `atomicMin` or `atomicMax`
 * to every pixel within the buffer of each sample. Costs are non-negative f32, so their bit
 * patterns order like integers; max mode stores `bits + 1` so zero means untouched. With
 * `assignments` (min mode) the low 8 bits of the word hold the facility row of the edge's source
 * node and the cost keeps its top 24 bits (relative precision 2^-15, rounded down), so the
 * smallest word is the smallest cost with the lowest facility row on ties. @internal
 */
export function createIsochronesSplatNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    nodeCount: number;
    width: number;
    height: number;
    mode: GPUNetworkIsochronesMode;
    maximumBufferPixels: number;
    maximumSamplesPerEdge: number;
    offsets: GraphDataView<'uint32'>;
    neighbors: GraphDataView<'uint32'>;
    weights: GraphDataView<'float32'>;
    costs: GraphDataView<'float32'>;
    nodePositions: GraphDataView<'float32x2'>;
    parameters: GraphDataView<'float32'>;
    encoded: GraphDataView<'uint32'>;
    assignments?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const isMin = props.mode === 'min';
  const labelled = Boolean(props.assignments);
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: `splat-${props.mode}${labelled ? '-labelled' : ''}`,
    bindings: [
      {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: props.neighbors, type: 'u32', access: 'read'},
      {name: 'weights', view: props.weights, type: 'f32', access: 'read'},
      {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
      {name: 'positions', view: props.nodePositions, type: 'f32', access: 'read'},
      {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'encoded', view: props.encoded, type: 'atomic<u32>', access: 'read_write'},
      ...(props.assignments
        ? [{name: 'assignments', view: props.assignments, type: 'u32', access: 'read'} as const]
        : [])
    ],
    invocationCount: props.neighbors.length,
    declarations: `const NODE_COUNT: u32 = ${props.nodeCount}u;
const WIDTH: i32 = ${props.width};
const HEIGHT: i32 = ${props.height};
const MAXIMUM_BUFFER_PIXELS: u32 = ${props.maximumBufferPixels}u;
const MAXIMUM_SAMPLES: u32 = ${props.maximumSamplesPerEdge}u;
const COST_CEILING: f32 = ${getWGSLFloatLiteral(3.0e38)};

fn isochronesIsFinite(value: f32) -> bool {
  return value == value && abs(value) < COST_CEILING;
}

fn isochronesFindSource(edge: u32) -> u32 {
  var low = 0u;
  var high = NODE_COUNT;
  while (low + 1u < high) {
    let middle = (low + high) >> 1u;
    if (offsets[offsetsOffset + middle] <= edge) {
      low = middle;
    } else {
      high = middle;
    }
  }
  return low;
}

fn isochronesGetPosition(node: u32) -> vec2f {
  return vec2f(positions[positionsOffset + 2u * node], positions[positionsOffset + 2u * node + 1u]);
}`,
    body: `if (index >= offsets[offsetsOffset + NODE_COUNT]) { return; }
  let targetNode = neighbors[neighborsOffset + index];
  if (targetNode >= NODE_COUNT) { return; }
  let weight = weights[weightsOffset + index];
  let source = isochronesFindSource(index);
  let sourceCost = costs[costsOffset + source];
  ${
    labelled
      ? `let facility = assignments[assignmentsOffset + source];
  if (facility >= ${GPU_NETWORK_ISOCHRONES_MAXIMUM_RASTER_FACILITIES}u) { return; }`
      : ''
  }
  if (!(weight >= 0.0) || !isochronesIsFinite(weight) || !isochronesIsFinite(sourceCost) || sourceCost < 0.0) { return; }
  let startPoint = isochronesGetPosition(source);
  let endPoint = isochronesGetPosition(targetNode);
  if (!isochronesIsFinite(startPoint.x) || !isochronesIsFinite(startPoint.y) || !isochronesIsFinite(endPoint.x) || !isochronesIsFinite(endPoint.y)) { return; }
  let minimum = vec2f(params[paramsOffset + 1u], params[paramsOffset + 2u]);
  let cellSize = vec2f(
    (params[paramsOffset + 3u] - minimum.x) / f32(WIDTH),
    (params[paramsOffset + 4u] - minimum.y) / f32(HEIGHT)
  );
  let walk = max(params[paramsOffset + 6u], 0.0);
  let radius = max(max(params[paramsOffset + 5u], 0.0), 0.5 * length(cellSize));
  let reachX = i32(min(u32(ceil(radius / cellSize.x)), MAXIMUM_BUFFER_PIXELS));
  let reachY = i32(min(u32(ceil(radius / cellSize.y)), MAXIMUM_BUFFER_PIXELS));
  let spacing = 0.5 * radius;
  let sampleCount = clamp(u32(ceil(length(endPoint - startPoint) / spacing)) + 1u, 2u, MAXIMUM_SAMPLES);
  for (var sampleIndex = 0u; sampleIndex < sampleCount; sampleIndex++) {
    let t = f32(sampleIndex) / f32(sampleCount - 1u);
    let point = mix(startPoint, endPoint, vec2f(t));
    let edgeCost = sourceCost + t * weight;
    let centerPixel = vec2i(floor((point - minimum) / cellSize));
    for (var dy = -reachY; dy <= reachY; dy++) {
      let row = centerPixel.y + dy;
      if (row < 0 || row >= HEIGHT) { continue; }
      for (var dx = -reachX; dx <= reachX; dx++) {
        let column = centerPixel.x + dx;
        if (column < 0 || column >= WIDTH) { continue; }
        let pixelCenter = minimum + (vec2f(f32(column), f32(row)) + 0.5) * cellSize;
        let pixelDistance = length(pixelCenter - point);
        if (pixelDistance > radius) { continue; }
        let cost = abs(edgeCost + walk * pixelDistance);
        if (!isochronesIsFinite(cost)) { continue; }
        let slot = encodedOffset + u32(row * WIDTH + column);
        // Test-before-RMW: overlapping samples and edges rewrite most pixels with a no-op, and a
        // plain atomic load is far cheaper than a contended read-modify-write.
        ${
          labelled
            ? `let encodedValue = (bitcast<u32>(cost) & 0xffffff00u) | facility;
        if (encodedValue < atomicLoad(&encoded[slot])) { atomicMin(&encoded[slot], encodedValue); }`
            : isMin
              ? `let encodedValue = bitcast<u32>(cost);
        if (encodedValue < atomicLoad(&encoded[slot])) { atomicMin(&encoded[slot], encodedValue); }`
              : `let encodedValue = bitcast<u32>(cost) + 1u;
        if (encodedValue > atomicLoad(&encoded[slot])) { atomicMax(&encoded[slot], encodedValue); }`
        }
      }
    }
  }`
  });
}

/** Decodes the accumulated pixels to float32 costs, `unreachedCost` where nothing landed. @internal */
export function createIsochronesDecodeNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    mode: GPUNetworkIsochronesMode;
    unreachedCost: number;
    encoded: GraphDataView<'uint32'>;
    values: GraphDataView<'float32'>;
    pixelCount: number;
    /** Facility-labelled encoding (min mode): also writes the facility row per pixel. */
    pixelFacilities?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const isMin = props.mode === 'min';
  const labelled = Boolean(props.pixelFacilities);
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: `decode-${props.mode}${labelled ? '-labelled' : ''}`,
    bindings: [
      {name: 'encoded', view: props.encoded, type: 'u32', access: 'read'},
      {name: 'values', view: props.values, type: 'f32', access: 'read_write'},
      ...(props.pixelFacilities
        ? [
            {
              name: 'facilitiesOut',
              view: props.pixelFacilities,
              type: 'u32',
              access: 'read_write'
            } as const
          ]
        : [])
    ],
    invocationCount: props.pixelCount,
    declarations: `const UNREACHED_COST: f32 = ${getWGSLFloatLiteral(props.unreachedCost)};`,
    body: `let word = encoded[encodedOffset + index];
  ${
    labelled
      ? `values[valuesOffset + index] = select(bitcast<f32>(word & 0xffffff00u), UNREACHED_COST, word == 0xffffffffu);
  facilitiesOut[facilitiesOutOffset + index] = select(word & 0xffu, 0xffffffffu, word == 0xffffffffu);`
      : isMin
        ? 'values[valuesOffset + index] = select(bitcast<f32>(word), UNREACHED_COST, word == 0xffffffffu);'
        : 'values[valuesOffset + index] = select(bitcast<f32>(word - 1u), UNREACHED_COST, word == 0u);'
  }`
  });
}

/** One thread per node: mask of nodes with a finite cost at or below the cell cost limit. @internal */
export function createIsochronesNodeMaskNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    costs: GraphDataView<'float32'>;
    parameters: GraphDataView<'float32'>;
    mask: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'node-mask',
    bindings: [
      {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
      {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'mask', view: props.mask, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.costs.length,
    body: `let cost = costs[costsOffset + index];
  let isReached = cost == cost && abs(cost) < ${getWGSLFloatLiteral(3.0e38)} && cost <= params[paramsOffset + 9u];
  mask[maskOffset + index] = select(0u, 1u, isReached);`
  });
}

/**
 * Labels every table cell with the facility of its cheapest reached node. Two atomic passes over
 * the nodes: the minimum cost bits per cell, then the smallest facility row among nodes whose cost
 * equals that minimum, so ties resolve to the lowest facility row independent of thread order.
 * `nodeCells` rows that are zero (masked out) or absent from the table are ignored; table rows
 * past the cell count keep `0xffffffff`. @internal
 */
export function createIsochronesCellFacilityNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    nodeCount: number;
    nodeCells: GraphDataView<'uint32x2'>;
    tableCells: GraphDataView<'uint32x2'>;
    tableCount: GraphDataView<'uint32'>;
    costs: GraphDataView<'float32'>;
    assignments: GraphDataView<'uint32'>;
    cellMinimumCosts: GraphDataView<'uint32'>;
    cellFacilities: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  const declarations = `const NODE_COUNT: u32 = ${props.nodeCount}u;

// Row of the node's cell in the sorted table, or 0xffffffff.
fn isochronesFindCellRow(node: u32) -> u32 {
  let low = nodeCells[nodeCellsOffset + 2u * node];
  let high = nodeCells[nodeCellsOffset + 2u * node + 1u];
  if (low == 0u && high == 0u) {
    return 0xffffffffu;
  }
  var first = 0u;
  var last = tableCount[tableCountOffset];
  while (first < last) {
    let middle = (first + last) >> 1u;
    let rowLow = tableCells[tableCellsOffset + 2u * middle];
    let rowHigh = tableCells[tableCellsOffset + 2u * middle + 1u];
    if (rowHigh < high || (rowHigh == high && rowLow < low)) {
      first = middle + 1u;
    } else {
      last = middle;
    }
  }
  if (first < tableCount[tableCountOffset]
    && tableCells[tableCellsOffset + 2u * first] == low
    && tableCells[tableCellsOffset + 2u * first + 1u] == high) {
    return first;
  }
  return 0xffffffffu;
}`;
  const common = [
    {name: 'nodeCells', view: props.nodeCells, type: 'u32', access: 'read'},
    {name: 'tableCells', view: props.tableCells, type: 'u32', access: 'read'},
    {name: 'tableCount', view: props.tableCount, type: 'u32', access: 'read'},
    {name: 'costs', view: props.costs, type: 'f32', access: 'read'}
  ] as const;
  return [
    createFillNode<Parameters>(graph, {
      id: `${props.id}-minimum-fill`,
      operation: OPERATION,
      view: props.cellMinimumCosts,
      type: 'u32',
      value: '0xffffffffu'
    }),
    createFillNode<Parameters>(graph, {
      id: `${props.id}-facility-fill`,
      operation: OPERATION,
      view: props.cellFacilities,
      type: 'u32',
      value: '0xffffffffu'
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-minimum`,
      operation: OPERATION,
      variant: 'cell-minimum-cost',
      bindings: [
        ...common,
        {
          name: 'cellMinimum',
          view: props.cellMinimumCosts,
          type: 'atomic<u32>',
          access: 'read_write'
        }
      ],
      invocationCount: props.nodeCount,
      declarations,
      body: `let row = isochronesFindCellRow(index);
  if (row != 0xffffffffu) {
    atomicMin(&cellMinimum[cellMinimumOffset + row], bitcast<u32>(costs[costsOffset + index]));
  }`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-facility`,
      operation: OPERATION,
      variant: 'cell-facility',
      bindings: [
        ...common,
        {name: 'assignments', view: props.assignments, type: 'u32', access: 'read'},
        {name: 'cellMinimum', view: props.cellMinimumCosts, type: 'u32', access: 'read'},
        {
          name: 'cellFacility',
          view: props.cellFacilities,
          type: 'atomic<u32>',
          access: 'read_write'
        }
      ],
      invocationCount: props.nodeCount,
      declarations,
      body: `let row = isochronesFindCellRow(index);
  if (row != 0xffffffffu && bitcast<u32>(costs[costsOffset + index]) == cellMinimum[cellMinimumOffset + row]) {
    atomicMin(&cellFacility[cellFacilityOffset + row], assignments[assignmentsOffset + index]);
  }`
    })
  ];
}

/**
 * One thread per triangle slot: the facility of the cell sample nearest to the triangle centroid
 * among the four samples around it that carry a facility, `0xffffffff` when none does or past the
 * triangle count. @internal
 */
export function createIsochronesTriangleFacilityNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    width: number;
    height: number;
    triangles: GraphDataView<'float32x2'>;
    triangleCount: GraphDataView<'uint32'>;
    parameters: GraphDataView<'float32'>;
    pixelFacilities: GraphDataView<'uint32'>;
    triangleFacilities: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'triangle-facility',
    bindings: [
      {name: 'triangles', view: props.triangles, type: 'f32', access: 'read'},
      {name: 'triangleCount', view: props.triangleCount, type: 'u32', access: 'read'},
      {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'pixelFacilities', view: props.pixelFacilities, type: 'u32', access: 'read'},
      {name: 'facilitiesOut', view: props.triangleFacilities, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.triangleFacilities.length,
    declarations: `const WIDTH: i32 = ${props.width};
const HEIGHT: i32 = ${props.height};`,
    body: `if (index >= triangleCount[triangleCountOffset]) {
    facilitiesOut[facilitiesOutOffset + index] = 0xffffffffu;
    return;
  }
  let base = trianglesOffset + 6u * index;
  let centroid = vec2f(
    (triangles[base] + triangles[base + 2u] + triangles[base + 4u]) / 3.0,
    (triangles[base + 1u] + triangles[base + 3u] + triangles[base + 5u]) / 3.0
  );
  let minimum = vec2f(params[paramsOffset + 1u], params[paramsOffset + 2u]);
  let cellSize = vec2f(
    (params[paramsOffset + 3u] - minimum.x) / f32(WIDTH),
    (params[paramsOffset + 4u] - minimum.y) / f32(HEIGHT)
  );
  // Sample (column, row) sits at minimum + (column + 0.5, row + 0.5) * cellSize.
  let gridPosition = (centroid - minimum) / cellSize - vec2f(0.5);
  let lower = vec2i(floor(gridPosition));
  var best = 0xffffffffu;
  var bestDistance = 1e30;
  for (var corner = 0u; corner < 4u; corner++) {
    let column = lower.x + i32(corner & 1u);
    let row = lower.y + i32(corner >> 1u);
    if (column < 0 || row < 0 || column >= WIDTH || row >= HEIGHT) { continue; }
    let facility = pixelFacilities[pixelFacilitiesOffset + u32(row * WIDTH + column)];
    if (facility == 0xffffffffu) { continue; }
    let offset = gridPosition - vec2f(f32(column), f32(row));
    let distance = dot(offset, offset);
    if (distance < bestDistance || (distance == bestDistance && facility < best)) {
      bestDistance = distance;
      best = facility;
    }
  }
  facilitiesOut[facilitiesOutOffset + index] = best;`
  });
}
