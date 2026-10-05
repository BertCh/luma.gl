// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';

const OPERATION = 'GPUEdgeBundling';

/** Work-state coordinate that marks every point of a dead edge. @internal */
export const EDGE_BUNDLING_DEAD_COORDINATE = -1.0e8;
/** Work-state x below this value means "dead point". @internal */
export const EDGE_BUNDLING_DEAD_LIMIT = -1.0e7;
/** Segment lengths at or under this read as zero when resampling (unit work box). @internal */
export const EDGE_BUNDLING_LENGTH_EPSILON = 1e-9;
/**
 * Gradients shorter than this many fixed-point density quanta leave a point where it is.
 *
 * Density is quantized, so a gradient of a few quanta is rounding noise rather than signal.
 * Normalizing such a gradient turns noise into a full-radius step in a random direction: lone
 * edges wandered a few percent of the box. The threshold scales with the quantum, which follows
 * the point-count-dependent fixed-point scale.
 * @internal
 */
export const EDGE_BUNDLING_GRADIENT_QUANTA = 4;
/** Kernel radii at or under this (work-box units) end the schedule. @internal */
export const EDGE_BUNDLING_RADIUS_EPSILON = 1e-9;

/** Parameter views accepted by the contributor. @internal */
export type EdgeBundlingParameterView = GraphDataView<'uint32'> | GraphDataView<'float32'>;

/** Shared, compile-time constants of one bundling graph. @internal */
export type EdgeBundlingConstants = {
  edgeCount: number;
  vertexCount: number;
  pointsPerEdge: number;
  densityResolution: number;
  fixedPointExponent: number;
  iterationCount: number;
  boxPadding: number;
};

function getConstantsWGSL(constants: EdgeBundlingConstants, includeIteration?: number): string {
  return /* wgsl */ `
const EDGE_COUNT: u32 = ${constants.edgeCount}u;
const VERTEX_COUNT: u32 = ${constants.vertexCount}u;
const POINTS: u32 = ${constants.pointsPerEdge}u;
const RESOLUTION: u32 = ${constants.densityResolution}u;
const RESOLUTION_F: f32 = ${getWGSLFloatLiteral(constants.densityResolution)};
const FIXED_POINT_SCALE: f32 = ${getWGSLFloatLiteral(2 ** constants.fixedPointExponent)};
const FIXED_POINT_INVERSE: f32 = ${getWGSLFloatLiteral(2 ** -constants.fixedPointExponent)};
const MAXIMUM_ITERATIONS: u32 = ${constants.iterationCount}u;
const BOX_SCALE: f32 = ${getWGSLFloatLiteral(1 + 2 * constants.boxPadding)};
const DEAD_COORDINATE: f32 = ${getWGSLFloatLiteral(EDGE_BUNDLING_DEAD_COORDINATE)};
const DEAD_LIMIT: f32 = ${getWGSLFloatLiteral(EDGE_BUNDLING_DEAD_LIMIT)};
const LENGTH_EPSILON: f32 = ${getWGSLFloatLiteral(EDGE_BUNDLING_LENGTH_EPSILON)};
const GRADIENT_MINIMUM: f32 = ${getWGSLFloatLiteral(EDGE_BUNDLING_GRADIENT_QUANTA)} * FIXED_POINT_INVERSE;
const RADIUS_EPSILON: f32 = ${getWGSLFloatLiteral(EDGE_BUNDLING_RADIUS_EPSILON)};
${includeIteration === undefined ? '' : `const ITERATION: u32 = ${includeIteration}u;`}`;
}

/**
 * WGSL for per-frame parameter access. Word layout: 0 activeIterations, 1 initial radius,
 * 2 lambda, 3 smoothing, 4 step scale (optional). A `uint32` view stores word 0 as an integer and
 * words 1 to 4 as float bit patterns; a `float32` view stores every word as a float.
 */
function getParametersWGSL(parameters: EdgeBundlingParameterView | undefined): string {
  let readers: string;
  if (!parameters) {
    readers = /* wgsl */ `
fn readActiveIterations() -> u32 { return MAXIMUM_ITERATIONS; }
fn readInitialRadius() -> f32 { return 0.03; }
fn readLambda() -> f32 { return 0.85; }
fn readSmoothing() -> f32 { return 0.5; }
fn readStepScale() -> f32 { return 1.0; }`;
  } else {
    const isFloat = parameters.format === 'float32';
    const readFloat = (word: number) =>
      isFloat
        ? `parameters[parametersOffset + ${word}u]`
        : `bitcast<f32>(parameters[parametersOffset + ${word}u])`;
    readers = /* wgsl */ `
fn readActiveIterations() -> u32 {
  ${
    isFloat
      ? 'return u32(clamp(parameters[parametersOffset], 0.0, f32(MAXIMUM_ITERATIONS)));'
      : 'return min(parameters[parametersOffset], MAXIMUM_ITERATIONS);'
  }
}
fn readInitialRadius() -> f32 { return max(${readFloat(1)}, 0.0); }
fn readLambda() -> f32 { return clamp(${readFloat(2)}, 0.5, 0.9); }
fn readSmoothing() -> f32 { return clamp(${readFloat(3)}, 0.0, 1.0); }
fn readStepScale() -> f32 { return ${parameters.length > 4 ? `max(${readFloat(4)}, 0.0)` : '1.0'}; }`;
  }
  return `${readers}
fn isIterationActive() -> bool { return ITERATION < readActiveIterations(); }
fn getIterationRadius() -> f32 {
  var radius = readInitialRadius();
  let lambda = readLambda();
  for (var round = 0u; round < ITERATION; round++) {
    radius = radius * lambda;
  }
  return radius;
}`;
}

function getEdgeHelpersWGSL(hasMask: boolean): string {
  return /* wgsl */ `
fn isFiniteValue(value: f32) -> bool { return abs(value) < 3.0e38; }
fn getVertexPosition(vertex: u32) -> vec2<f32> {
  return vec2<f32>(
    vertexPositions[vertexPositionsOffset + vertex * 2u],
    vertexPositions[vertexPositionsOffset + vertex * 2u + 1u]
  );
}
fn isEdgeLive(edge: u32) -> bool {
  let source = edgeSources[edgeSourcesOffset + edge];
  let targetVertex = edgeTargets[edgeTargetsOffset + edge];
  if (source >= VERTEX_COUNT || targetVertex >= VERTEX_COUNT) {
    return false;
  }
  ${hasMask ? 'if (edgeMask[edgeMaskOffset + edge] == 0u) {\n    return false;\n  }' : ''}
  let a = getVertexPosition(source);
  let b = getVertexPosition(targetVertex);
  return isFiniteValue(a.x) && isFiniteValue(a.y) && isFiniteValue(b.x) && isFiniteValue(b.y);
}`;
}

const BOX_READ_WGSL = /* wgsl */ `
fn decodeKey(key: u32) -> f32 {
  return bitcast<f32>(select(~key, key & 0x7fffffffu, (key & 0x80000000u) != 0u));
}
/** Returns (origin x, origin y, side) of the square work box. */
fn getBox() -> vec3<f32> {
  if (boxKeys[boxKeysOffset] == 0xffffffffu) {
    return vec3<f32>(0.0, 0.0, 1.0);
  }
  let minX = decodeKey(boxKeys[boxKeysOffset]);
  let minY = decodeKey(boxKeys[boxKeysOffset + 1u]);
  let maxX = decodeKey(boxKeys[boxKeysOffset + 2u]);
  let maxY = decodeKey(boxKeys[boxKeysOffset + 3u]);
  let extent = max(maxX - minX, maxY - minY);
  let side = select(extent * BOX_SCALE, 1.0, !(extent > 0.0));
  return vec3<f32>(0.5 * (minX + maxX) - 0.5 * side, 0.5 * (minY + maxY) - 0.5 * side, side);
}`;

/** Edge inputs shared by the passes that read the caller's geometry. @internal */
export type EdgeBundlingInputs = {
  positions: GraphDataView<'float32x2'>;
  sources: GraphDataView<'uint32'>;
  targets: GraphDataView<'uint32'>;
  mask?: GraphDataView<'uint32'>;
};

function getInputBindings(inputs: EdgeBundlingInputs): WGSLKernelBinding[] {
  const bindings: WGSLKernelBinding[] = [
    {
      name: 'vertexPositions',
      view: inputs.positions,
      type: 'f32',
      access: 'read'
    },
    {name: 'edgeSources', view: inputs.sources, type: 'u32', access: 'read'},
    {name: 'edgeTargets', view: inputs.targets, type: 'u32', access: 'read'}
  ];
  if (inputs.mask) {
    bindings.push({
      name: 'edgeMask',
      view: inputs.mask,
      type: 'u32',
      access: 'read'
    });
  }
  return bindings;
}

/** Resets the work-box keys to an empty box. @internal */
export function createEdgeBundlingBoxResetNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {id: string; boxKeys: GraphDataView<'uint32'>}
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'box-reset',
    bindings: [
      {
        name: 'boxKeys',
        view: props.boxKeys,
        type: 'u32',
        access: 'read_write'
      }
    ],
    invocationCount: 4,
    body: `boxKeys[boxKeysOffset + index] = select(0u, 0xffffffffu, index < 2u);`
  });
}

/** Accumulates the live-edge endpoint bounds with order-preserving atomic min and max. @internal */
export function createEdgeBundlingBoxNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    constants: EdgeBundlingConstants;
    inputs: EdgeBundlingInputs;
    boxKeys: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'box',
    bindings: [
      ...getInputBindings(props.inputs),
      {
        name: 'boxKeys',
        view: props.boxKeys,
        type: 'atomic<u32>',
        access: 'read_write'
      }
    ],
    invocationCount: props.constants.edgeCount,
    declarations: `${getConstantsWGSL(props.constants)}
${getEdgeHelpersWGSL(Boolean(props.inputs.mask))}
fn encodeKey(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  return select(bits | 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}`,
    body: `if (!isEdgeLive(index)) {
    return;
  }
  let a = getVertexPosition(edgeSources[edgeSourcesOffset + index]);
  let b = getVertexPosition(edgeTargets[edgeTargetsOffset + index]);
  let low = min(a, b);
  let high = max(a, b);
  atomicMin(&boxKeys[boxKeysOffset], encodeKey(low.x));
  atomicMin(&boxKeys[boxKeysOffset + 1u], encodeKey(low.y));
  atomicMax(&boxKeys[boxKeysOffset + 2u], encodeKey(high.x));
  atomicMax(&boxKeys[boxKeysOffset + 3u], encodeKey(high.y));`
  });
}

/** Writes the straight-line subdivision of every edge in normalized work-box coordinates. @internal */
export function createEdgeBundlingInitializeNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    constants: EdgeBundlingConstants;
    inputs: EdgeBundlingInputs;
    boxKeys: GraphDataView<'uint32'>;
    work: GraphDataView<'float32x2'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'initialize',
    bindings: [
      ...getInputBindings(props.inputs),
      {name: 'boxKeys', view: props.boxKeys, type: 'u32', access: 'read'},
      {name: 'work', view: props.work, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.constants.edgeCount * props.constants.pointsPerEdge,
    declarations: `${getConstantsWGSL(props.constants)}
${getEdgeHelpersWGSL(Boolean(props.inputs.mask))}
${BOX_READ_WGSL}`,
    body: `let edge = index / POINTS;
  let point = index % POINTS;
  if (!isEdgeLive(edge)) {
    work[workOffset + index * 2u] = DEAD_COORDINATE;
    work[workOffset + index * 2u + 1u] = DEAD_COORDINATE;
    return;
  }
  let workBox = getBox();
  let a = (getVertexPosition(edgeSources[edgeSourcesOffset + edge]) - workBox.xy) / workBox.z;
  let b = (getVertexPosition(edgeTargets[edgeTargetsOffset + edge]) - workBox.xy) / workBox.z;
  let t = f32(point) / f32(POINTS - 1u);
  var placed = a + (b - a) * t;
  if (point == 0u) {
    placed = a;
  } else if (point == POINTS - 1u) {
    placed = b;
  }
  work[workOffset + index * 2u] = placed.x;
  work[workOffset + index * 2u + 1u] = placed.y;`
  });
}

/** Zeroes the density grid. @internal */
export function createEdgeBundlingClearNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {id: string; density: GraphDataView<'uint32'>}
): GPUCommandNode<Parameters> {
  return createFillNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    view: props.density,
    type: 'u32',
    value: '0u'
  });
}

/** Splats Epanechnikov fixed-point weights, one invocation per control point. @internal */
export function createEdgeBundlingSplatNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    constants: EdgeBundlingConstants;
    iteration: number;
    parameters?: EdgeBundlingParameterView;
    work: GraphDataView<'float32x2'>;
    density: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const {constants} = props;
  const bindings: WGSLKernelBinding[] = [
    {name: 'work', view: props.work, type: 'f32', access: 'read'},
    {
      name: 'density',
      view: props.density,
      type: 'atomic<u32>',
      access: 'read_write'
    }
  ];
  if (props.parameters) {
    bindings.push({
      name: 'parameters',
      view: props.parameters,
      type: props.parameters.format === 'float32' ? 'f32' : 'u32',
      access: 'read'
    });
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'splat',
    bindings,
    invocationCount: constants.edgeCount * constants.pointsPerEdge,
    declarations: `${getConstantsWGSL(constants, props.iteration)}
${getParametersWGSL(props.parameters)}`,
    body: `if (!isIterationActive()) {
    return;
  }
  let radius = getIterationRadius();
  if (!(radius > RADIUS_EPSILON)) {
    return;
  }
  let px = work[workOffset + index * 2u];
  let py = work[workOffset + index * 2u + 1u];
  if (px < DEAD_LIMIT) {
    return;
  }
  let gx = px * RESOLUTION_F;
  let gy = py * RESOLUTION_F;
  let radiusCells = radius * RESOLUTION_F;
  let reach = i32(max(1.0, ceil(radiusCells)));
  let inverseSquare = 1.0 / (radiusCells * radiusCells);
  let cx = i32(floor(gx));
  let cy = i32(floor(gy));
  let maximumCell = i32(RESOLUTION) - 1;
  let x0 = max(cx - reach, 0);
  let x1 = min(cx + reach, maximumCell);
  let y0 = max(cy - reach, 0);
  let y1 = min(cy + reach, maximumCell);
  for (var y = y0; y <= y1; y++) {
    let dy = f32(y) + 0.5 - gy;
    for (var x = x0; x <= x1; x++) {
      let dx = f32(x) + 0.5 - gx;
      let q = (dx * dx + dy * dy) * inverseSquare;
      if (q < 1.0) {
        let weight = u32((1.0 - q) * FIXED_POINT_SCALE + 0.5);
        if (weight > 0u) {
          atomicAdd(&density[densityOffset + u32(y) * RESOLUTION + u32(x)], weight);
        }
      }
    }
  }`
  });
}

/**
 * Advects, arc-length resamples, and Laplacian-smooths every live edge in one invocation per
 * edge. Reads density only and rewrites only its own edge rows, so positions update in place.
 * @internal
 */
export function createEdgeBundlingUpdateNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    constants: EdgeBundlingConstants;
    iteration: number;
    parameters?: EdgeBundlingParameterView;
    work: GraphDataView<'float32x2'>;
    density: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const {constants} = props;
  const bindings: WGSLKernelBinding[] = [
    {name: 'work', view: props.work, type: 'f32', access: 'read_write'},
    {name: 'density', view: props.density, type: 'u32', access: 'read'}
  ];
  if (props.parameters) {
    bindings.push({
      name: 'parameters',
      view: props.parameters,
      type: props.parameters.format === 'float32' ? 'f32' : 'u32',
      access: 'read'
    });
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'update',
    bindings,
    invocationCount: constants.edgeCount,
    workgroupSize: 64,
    declarations: `${getConstantsWGSL(constants, props.iteration)}
${getParametersWGSL(props.parameters)}
fn readDensity(x: i32, y: i32) -> f32 {
  return f32(density[densityOffset + u32(y) * RESOLUTION + u32(x)]) * FIXED_POINT_INVERSE;
}
/** Bilinear density at grid coordinates where integers are cell centers; edges clamp. */
fn sampleDensity(gx: f32, gy: f32) -> f32 {
  let maximumCell = f32(RESOLUTION) - 1.0;
  let cx = clamp(gx, 0.0, maximumCell);
  let cy = clamp(gy, 0.0, maximumCell);
  let x0 = i32(floor(cx));
  let y0 = i32(floor(cy));
  let x1 = min(x0 + 1, i32(RESOLUTION) - 1);
  let y1 = min(y0 + 1, i32(RESOLUTION) - 1);
  let fx = cx - f32(x0);
  let fy = cy - f32(y0);
  return readDensity(x0, y0) * (1.0 - fx) * (1.0 - fy) +
    readDensity(x1, y0) * fx * (1.0 - fy) +
    readDensity(x0, y1) * (1.0 - fx) * fy +
    readDensity(x1, y1) * fx * fy;
}`,
    body: `if (!isIterationActive()) {
    return;
  }
  let radius = getIterationRadius();
  if (!(radius > RADIUS_EPSILON)) {
    return;
  }
  let base = index * POINTS;
  if (work[workOffset + base * 2u] < DEAD_LIMIT) {
    return;
  }
  var points: array<vec2<f32>, ${constants.pointsPerEdge}>;
  var resampled: array<vec2<f32>, ${constants.pointsPerEdge}>;
  for (var i = 0u; i < POINTS; i++) {
    points[i] = vec2<f32>(
      work[workOffset + (base + i) * 2u],
      work[workOffset + (base + i) * 2u + 1u]
    );
  }
  // Advect interior points one scaled bandwidth along the bilinearly sampled, normalized gradient.
  let advectionStep = radius * readStepScale();
  for (var i = 1u; i < POINTS - 1u; i++) {
    let gx = points[i].x * RESOLUTION_F - 0.5;
    let gy = points[i].y * RESOLUTION_F - 0.5;
    if (gx < 1.0 || gy < 1.0 || gx >= RESOLUTION_F - 2.0 || gy >= RESOLUTION_F - 2.0) {
      continue;
    }
    let gradientX = sampleDensity(gx + 1.0, gy) - sampleDensity(gx - 1.0, gy);
    let gradientY = sampleDensity(gx, gy + 1.0) - sampleDensity(gx, gy - 1.0);
    let gradientLength = sqrt(gradientX * gradientX + gradientY * gradientY);
    if (gradientLength < GRADIENT_MINIMUM) {
      continue;
    }
    points[i] = points[i] + vec2<f32>(gradientX / gradientLength, gradientY / gradientLength) * advectionStep;
  }
  // Resample to uniform arc length, keeping both endpoints.
  var total = 0.0;
  for (var i = 1u; i < POINTS; i++) {
    let delta = points[i] - points[i - 1u];
    total = total + sqrt(delta.x * delta.x + delta.y * delta.y);
  }
  var segment = 1u;
  var consumed = 0.0;
  var delta = points[1] - points[0];
  var segmentLength = sqrt(delta.x * delta.x + delta.y * delta.y);
  for (var i = 0u; i < POINTS; i++) {
    let arcTarget = total * f32(i) / f32(POINTS - 1u);
    while (segment < POINTS - 1u && consumed + segmentLength < arcTarget) {
      consumed = consumed + segmentLength;
      segment = segment + 1u;
      delta = points[segment] - points[segment - 1u];
      segmentLength = sqrt(delta.x * delta.x + delta.y * delta.y);
    }
    let fraction = select((arcTarget - consumed) / segmentLength, 0.0, segmentLength < LENGTH_EPSILON);
    let previous = points[segment - 1u];
    resampled[i] = previous + (points[segment] - previous) * fraction;
  }
  resampled[0] = points[0];
  resampled[POINTS - 1u] = points[POINTS - 1u];
  // One simultaneous Laplacian pass from the resampled snapshot.
  let smoothing = readSmoothing();
  for (var i = 0u; i < POINTS; i++) {
    var position = resampled[i];
    if (i > 0u && i < POINTS - 1u) {
      let midpoint = 0.5 * (resampled[i - 1u] + resampled[i + 1u]) - position;
      position = position + smoothing * midpoint;
    }
    work[workOffset + (base + i) * 2u] = position.x;
    work[workOffset + (base + i) * 2u + 1u] = position.y;
  }`
  });
}

/** Maps work-box points back to caller coordinates, pins endpoints, and collapses dead edges. @internal */
export function createEdgeBundlingFinalizeNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    constants: EdgeBundlingConstants;
    inputs: EdgeBundlingInputs;
    boxKeys: GraphDataView<'uint32'>;
    work: GraphDataView<'float32x2'>;
    paths: GraphDataView<'float32x2'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'finalize',
    bindings: [
      ...getInputBindings(props.inputs),
      {name: 'boxKeys', view: props.boxKeys, type: 'u32', access: 'read'},
      {name: 'work', view: props.work, type: 'f32', access: 'read'},
      {name: 'paths', view: props.paths, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.constants.edgeCount * props.constants.pointsPerEdge,
    declarations: `${getConstantsWGSL(props.constants)}
${getEdgeHelpersWGSL(Boolean(props.inputs.mask))}
${BOX_READ_WGSL}`,
    body: `let edge = index / POINTS;
  let point = index % POINTS;
  let source = edgeSources[edgeSourcesOffset + edge];
  var outputPoint = vec2<f32>(0.0, 0.0);
  if (!isEdgeLive(edge)) {
    if (source < VERTEX_COUNT) {
      outputPoint = getVertexPosition(source);
    }
  } else if (point == 0u) {
    outputPoint = getVertexPosition(source);
  } else if (point == POINTS - 1u) {
    outputPoint = getVertexPosition(edgeTargets[edgeTargetsOffset + edge]);
  } else {
    let workBox = getBox();
    let normalized = vec2<f32>(work[workOffset + index * 2u], work[workOffset + index * 2u + 1u]);
    outputPoint = workBox.xy + normalized * workBox.z;
  }
  paths[pathsOffset + index * 2u] = outputPoint.x;
  paths[pathsOffset + index * 2u + 1u] = outputPoint.y;`
  });
}

/** Writes `startIndices` (`edge * pointsPerEdge`) and the four-word indirect draw record. @internal */
export function createEdgeBundlingIndicesNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    constants: EdgeBundlingConstants;
    startIndices?: GraphDataView<'uint32'>;
    drawRecord?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [];
  if (props.startIndices) {
    bindings.push({
      name: 'startIndices',
      view: props.startIndices,
      type: 'u32',
      access: 'read_write'
    });
  }
  if (props.drawRecord) {
    bindings.push({
      name: 'drawRecord',
      view: props.drawRecord,
      type: 'u32',
      access: 'read_write'
    });
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'indices',
    bindings,
    invocationCount: Math.max(props.constants.edgeCount + 1, 4),
    declarations: getConstantsWGSL(props.constants),
    body: `${
      props.startIndices
        ? 'if (index <= EDGE_COUNT) {\n    startIndices[startIndicesOffset + index] = index * POINTS;\n  }'
        : ''
    }
  ${
    props.drawRecord
      ? `if (index < 4u) {
    drawRecord[drawRecordOffset + index] = select(select(0u, EDGE_COUNT, index == 1u), POINTS, index == 0u);
  }`
      : ''
  }`
  });
}
