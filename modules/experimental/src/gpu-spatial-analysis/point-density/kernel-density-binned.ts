// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandGraphNodeCondition,
  type GPUCommandNode,
  type GraphBufferUse,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  createFixedPointClearNode,
  createFixedPointScaleNodes,
  FIXED_POINT_WGSL,
  FIXED_POINT_WORD_COUNT,
  getFixedPointAtomicAddWGSL,
  getFixedPointHeadroomBits
} from '../../utils/fixed-point-accumulator';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {GRID_COORDINATE_WGSL} from '../spatial-interpolation/gpu-inverse-distance-weighting';
import type {
  GPUKernelDensityEdgeCorrection,
  GPUKernelDensityNormalization
} from './gpu-kernel-density';
import {
  GPU_KERNEL_DENSITY_AUTO_SUPPORT_CELLS,
  GPU_KERNEL_DENSITY_BINNED_BOX_PASS_COUNT,
  GPU_KERNEL_DENSITY_BINNED_EXPONENTIAL_VARIANCE_SHARE,
  GPU_KERNEL_DENSITY_BINNED_MARGIN_BANDWIDTHS,
  GPU_KERNEL_DENSITY_GAUSSIAN_TRUNCATION
} from './kernel-density-math';
import type {GPUPolygonStudyWindowViews} from '../../analysis-core/contracts/study-window';
import {getStudyWindowBinding} from '../../utils/study-window-wgsl';
import {
  getKernelDensityResultWGSL,
  getKernelDensityStudyWindowWGSL,
  getWindowMassWGSL
} from './kernel-density-wgsl';

const OPERATION = 'GPUKernelDensity';

/** Smallest refresh interval, in cells, of the running box sums. */
const MINIMUM_REFRESH_INTERVAL = 32;

/** Largest blur sigma in cells; beyond it the filter radius would overflow 32-bit indexing. */
const MAXIMUM_SIGMA_CELLS = 1e6;

/** Float32 words of one axis block of the blur plan. */
const AXIS_PLAN_LENGTH = 5;

/** `[valid, 1 / cellArea, x block, y block]`. */
const PLAN_LENGTH = 2 + 2 * AXIS_PLAN_LENGTH;
const PLAN_X_BASE = 2;
const PLAN_Y_BASE = 2 + AXIS_PLAN_LENGTH;

/** Dispatch gate slots of the binned nodes under `method: 'auto'`; slot 0 gates the exact gather. */
const GATE_SLOT = {gather: 0, bin: 1, decode: 2, blurX: 3, blurY: 4, finish: 5} as const;

/** The workgroup sizes the gate shapes below and the gated kernels share. */
const POINT_WORKGROUP_SIZE = 64;
const LINE_WORKGROUP_SIZE = 32;

/**
 * Maximum padded cells whose three-word fixed-point sums fit one 128 MiB storage binding, the
 * core-device limit.
 * @internal
 */
export const GPU_KERNEL_DENSITY_BINNED_MAXIMUM_CELL_COUNT = Math.floor(
  (128 * 1024 * 1024) / (4 * FIXED_POINT_WORD_COUNT)
);

/** Inputs of {@link createGPUKernelDensityBinnedNodes}. @internal */
export type GPUKernelDensityBinnedProps = {
  id: string;
  /** Prepared polygon window, sharing the point-weight inclusion rule. */
  studyWindow?: GPUPolygonStudyWindowViews;
  positions: GraphDataView<'float32x2'>;
  parameters: GraphDataView<'float32'>;
  /** Included point weights (exclusion and `'diggle'` reweighting applied; 0 for excluded rows). */
  pointWeights: GraphDataView<'float32'>;
  /** `bandwidthState` binding when the GPU computes the bandwidth, else empty. */
  bandwidthBindings: readonly WGSLKernelBinding[];
  /** WGSL expression of the bandwidth. */
  bandwidthExpression: string;
  /** Sum of the included weights, bound for `'probability'`. */
  weightSum?: GraphDataView<'float32'>;
  values: GraphDataView<'float32'>;
  width: number;
  height: number;
  /** Cells added on every side of the output grid to the binned region (before the guard cell). */
  padding: number;
  edgeCorrection: GPUKernelDensityEdgeCorrection;
  normalization: GPUKernelDensityNormalization;
  /** Optional one-row uint32: 1 when `method: 'binned'` ran with less margin than it needs. */
  insufficientPadding?: GraphDataView<'uint32'>;
  /** Gate the binned nodes and the exact gather on the bandwidth (`method: 'auto'`). */
  autoGate: boolean;
};

/** Nodes of the binned estimate and the exact-path gate under `method: 'auto'`. @internal */
export type GPUKernelDensityBinnedNodes<Parameters> = {
  /** Plan node (and the gate writer under `'auto'`): must run before the exact gather. */
  planNodes: GPUCommandNode<Parameters>[];
  /** Binning, decode, blur and finish nodes, in order. */
  bodyNodes: GPUCommandNode<Parameters>[];
  /** Condition for the exact gather under `'auto'`. */
  exactGate?: {
    condition: GPUCommandGraphNodeCondition<Parameters>;
    extraResources: GraphBufferUse[];
  };
};

/**
 * Default `binnedPadding`: one grid extent (the larger side), which gives the margin `2.5 h` the
 * blur needs for bandwidths up to 40% of the grid, reduced until the three fixed-point words per
 * cell fit one storage binding.
 * @internal
 */
export function getGPUKernelDensityDefaultBinnedPadding(width: number, height: number): number {
  let padding = Math.max(width, height);
  while (padding > 0) {
    const size = getGPUKernelDensityBinnedSize(width, height, padding);
    if (size.paddedWidth * size.paddedHeight <= GPU_KERNEL_DENSITY_BINNED_MAXIMUM_CELL_COUNT) {
      break;
    }
    padding = Math.floor(padding / 2);
  }
  return padding;
}

/**
 * Cells per side of the padded binning grid: the output grid plus `padding` on each side plus one
 * guard cell on each side, so every point that is inside the counted region has both linear-binning
 * neighbors in the array.
 * @internal
 */
export function getGPUKernelDensityBinnedSize(
  width: number,
  height: number,
  padding: number
): {paddedWidth: number; paddedHeight: number; margin: number} {
  const margin = padding + 1;
  return {paddedWidth: width + 2 * margin, paddedHeight: height + 2 * margin, margin};
}

/**
 * Builds the binned estimate (gaussian kernel only), in order:
 *
 * 1. `blur-plan`: one invocation reads the bandwidth and grid and writes the per-axis filter
 *    parameters (box radius and edge weight, normalisation, refresh interval, exponential pole)
 *    and, under `'auto'`, arms the dispatch gate. Every bandwidth-dependent number lives in this
 *    GPU word (library rule 12), so one set of pipelines serves every bandwidth.
 * 2. Fixed-point scale and clear, then `bin`: linear (cloud-in-cell) binning of each included
 *    weight onto the 4 nearest cell centers of the padded grid with the shared fixed-point
 *    accumulator (deterministic, no float CAS), and `decode` to f32.
 * 3. Per axis, three running-sum box blurs and one two-sided exponential (forward and backward
 *    first-order recursion) pass; x then y, ping-ponging between two buffers.
 * 4. `finish`: scale by 1 / cell area, then the shared edge-correction and normalization tail.
 *
 * @internal
 */
export function createGPUKernelDensityBinnedNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUKernelDensityBinnedProps
): GPUKernelDensityBinnedNodes<Parameters> {
  const {id, width, height, edgeCorrection, normalization, studyWindow} = props;
  const pointCount = props.positions.length;
  const cellCount = width * height;
  const {paddedWidth, paddedHeight, margin} = getGPUKernelDensityBinnedSize(
    width,
    height,
    props.padding
  );
  const paddedCellCount = paddedWidth * paddedHeight;

  const gate = props.autoGate
    ? createWGSLDispatchGate(graph, {
        id: `${id}-method-gate`,
        operation: OPERATION,
        shapes: [
          [cellCount, POINT_WORKGROUP_SIZE],
          [pointCount, POINT_WORKGROUP_SIZE],
          [paddedCellCount, POINT_WORKGROUP_SIZE],
          [paddedHeight, LINE_WORKGROUP_SIZE],
          [paddedWidth, LINE_WORKGROUP_SIZE],
          [cellCount, POINT_WORKGROUP_SIZE]
        ]
      })
    : undefined;
  const getGate = (slot: number) => (gate ? gate.getGate(slot) : {});

  const plan = createTransientView(graph, `${id}-blur-plan-words`, 'float32', PLAN_LENGTH);
  const scale = createTransientView(graph, `${id}-bin-scale-word`, 'uint32', 1);
  const words = createTransientView(
    graph,
    `${id}-bin-words`,
    'uint32',
    paddedCellCount * FIXED_POINT_WORD_COUNT
  );
  const density = createTransientView(graph, `${id}-bin-density`, 'float32', paddedCellCount);
  const scratch = createTransientView(graph, `${id}-bin-scratch`, 'float32', paddedCellCount);

  const gridBindings = (name: string): WGSLKernelBinding => ({
    name,
    view: props.parameters,
    type: 'f32',
    access: 'read'
  });
  const gridConstants = `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const PADDED_WIDTH: u32 = ${paddedWidth}u;
const PADDED_HEIGHT: u32 = ${paddedHeight}u;
const MARGIN: u32 = ${margin}u;`;

  const planNodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-blur-plan`,
      operation: OPERATION,
      variant: `blur-plan-${props.autoGate ? 'auto' : 'binned'}`,
      bindings: [
        gridBindings('params'),
        ...props.bandwidthBindings,
        {name: 'plan', view: plan, type: 'f32', access: 'read_write'},
        ...(gate ? [gate.binding] : []),
        ...(props.insufficientPadding
          ? [
              {
                name: 'insufficientPadding',
                view: props.insufficientPadding,
                type: 'u32',
                access: 'read_write'
              } as const
            ]
          : [])
      ],
      invocationCount: 1,
      declarations: `${gridConstants}
${GRID_COORDINATE_WGSL}
const EXPONENTIAL_SHARE: f32 = ${getWGSLFloatLiteral(GPU_KERNEL_DENSITY_BINNED_EXPONENTIAL_VARIANCE_SHARE)};
const BOX_PASS_COUNT: f32 = ${getWGSLFloatLiteral(GPU_KERNEL_DENSITY_BINNED_BOX_PASS_COUNT)};
const MINIMUM_REFRESH_INTERVAL: f32 = ${getWGSLFloatLiteral(MINIMUM_REFRESH_INTERVAL)};
const SUPPORT_FACTOR: f32 = ${getWGSLFloatLiteral(GPU_KERNEL_DENSITY_GAUSSIAN_TRUNCATION)};
const AUTO_SUPPORT_CELLS: f32 = ${getWGSLFloatLiteral(GPU_KERNEL_DENSITY_AUTO_SUPPORT_CELLS)};
const MARGIN_BANDWIDTHS: f32 = ${getWGSLFloatLiteral(GPU_KERNEL_DENSITY_BINNED_MARGIN_BANDWIDTHS)};
// Axis block: [box radius, box edge weight, box normalisation, refresh interval, exponential pole].
// Boxes are 2R + 1 cells plus fractional end taps of weight 'edge', so the variance of a pass
// is matched exactly: v = (R (R + 1) (2R + 1) / 3 + 2 edge (R + 1)^2) / (2R + 1 + 2 edge).
fn writeAxisPlan(base: u32, sigma: f32, lineLength: f32) {
  let varianceTotal = sigma * sigma;
  let varianceExponential = EXPONENTIAL_SHARE * varianceTotal;
  let varianceBox = (varianceTotal - varianceExponential) / BOX_PASS_COUNT;
  let radius = floor((sqrt(1.0 + 12.0 * varianceBox) - 1.0) * 0.5);
  let edge = clamp(
    (2.0 * radius + 1.0) * (varianceBox - radius * (radius + 1.0) / 3.0) /
      (2.0 * ((radius + 1.0) * (radius + 1.0) - varianceBox)),
    0.0,
    1.0
  );
  plan[planOffset + base] = radius;
  plan[planOffset + base + 1u] = edge;
  plan[planOffset + base + 2u] = 1.0 / (2.0 * radius + 1.0 + 2.0 * edge);
  plan[planOffset + base + 3u] = max(MINIMUM_REFRESH_INTERVAL, min(2.0 * radius + 3.0, lineLength));
  // Two-sided geometric with variance 2a / (1 - a)^2: the stable root of v a^2 - (2v + 2) a + v.
  plan[planOffset + base + 4u] = varianceExponential / ((varianceExponential + 1.0) + sqrt(2.0 * varianceExponential + 1.0));
}`,
      body: `let gridStart = vec2f(params[paramsOffset], params[paramsOffset + 1u]);
  let gridEnd = vec2f(params[paramsOffset + 2u], params[paramsOffset + 3u]);
  let bandwidth = ${props.bandwidthExpression};
  let cellSize = (gridEnd - gridStart) / vec2f(f32(WIDTH), f32(HEIGHT));
  let cellArea = abs(cellSize.x * cellSize.y);
  let sigmaCells = abs(vec2f(bandwidth) / cellSize);
  let valid = isFiniteValue(bandwidth) && bandwidth > 0.0 && isFiniteValue(cellSize.x) &&
    isFiniteValue(cellSize.y) && all(gridEnd != gridStart) && isFiniteValue(cellArea) &&
    cellArea > 0.0 && isFiniteValue(sigmaCells.x) && isFiniteValue(sigmaCells.y);
  let sigma = select(vec2f(0.0), min(sigmaCells, vec2f(${getWGSLFloatLiteral(MAXIMUM_SIGMA_CELLS)})), valid);
  plan[planOffset] = select(0.0, 1.0, valid);
  plan[planOffset + 1u] = select(0.0, 1.0 / cellArea, valid);
  writeAxisPlan(${PLAN_X_BASE}u, sigma.x, f32(PADDED_WIDTH));
  writeAxisPlan(${PLAN_Y_BASE}u, sigma.y, f32(PADDED_HEIGHT));
  // The cascade drops mass that leaves the array, so it needs a margin of a few bandwidths.
  let sufficient = f32(MARGIN) >= MARGIN_BANDWIDTHS * max(sigma.x, sigma.y);${
    props.insufficientPadding
      ? `
  insufficientPadding[insufficientPaddingOffset] = ${props.autoGate ? '0u' : 'select(0u, 1u, valid && !sufficient)'};`
      : ''
  }${
    gate
      ? `
  // The exact gather visits (support / cell)^2 cells per point; the blur costs the same at any
  // bandwidth, so it wins once the support radius exceeds the threshold, provided the margin
  // suffices (otherwise the exact path stays correct).
  let useBinned = valid && sufficient && SUPPORT_FACTOR * max(sigma.x, sigma.y) > AUTO_SUPPORT_CELLS;
  ${gate.armWGSL}
  if (useBinned) {
    dispatch[dispatchOffset + ${3 * GATE_SLOT.gather}u] = 0u;
  } else {
    ${[GATE_SLOT.bin, GATE_SLOT.decode, GATE_SLOT.blurX, GATE_SLOT.blurY, GATE_SLOT.finish]
      .map(slot => `dispatch[dispatchOffset + ${3 * slot}u] = 0u;`)
      .join('\n    ')}
  }`
      : ''
  }`
    })
  ];

  const bodyNodes: GPUCommandNode<Parameters>[] = [
    ...createFixedPointScaleNodes(graph, {
      id: `${id}-bin-scale`,
      operation: OPERATION,
      values: props.pointWeights,
      scale
    }),
    createFixedPointClearNode(graph, {id: `${id}-bin-clear`, operation: OPERATION, words}),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-bin`,
      operation: OPERATION,
      variant: 'bin',
      bindings: [
        ...(studyWindow ? [getStudyWindowBinding(studyWindow)] : []),
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        {name: 'pointWeights', view: props.pointWeights, type: 'f32', access: 'read'},
        gridBindings('params'),
        {name: 'binScale', view: scale, type: 'u32', access: 'read'},
        {name: 'words', view: words, type: 'atomic<u32>', access: 'read_write'}
      ],
      invocationCount: pointCount,
      workgroupSize: POINT_WORKGROUP_SIZE,
      ...getGate(GATE_SLOT.bin),
      declarations: `override FIXED_POINT_HEADROOM_BITS: u32;
${gridConstants}
${FIXED_POINT_WGSL}
${getFixedPointAtomicAddWGSL('words')}${studyWindow ? getKernelDensityStudyWindowWGSL(studyWindow) : ''}`,
      constants: {FIXED_POINT_HEADROOM_BITS: getFixedPointHeadroomBits(pointCount)},
      // Linear binning splits the weight over the four cell centers around the point:
      // (1 - fx)(1 - fy), fx (1 - fy), (1 - fx) fy, fx fy. A point within half a cell of the padded
      // array's outer edge lacks a neighbor and is not counted (comparisons are false for NaN).
      body: `let weight = pointWeights[pointWeightsOffset + index];
  if (!(weight > 0.0)) {
    return;
  }
  let gridStart = vec2f(params[paramsOffset], params[paramsOffset + 1u]);
  let gridEnd = vec2f(params[paramsOffset + 2u], params[paramsOffset + 3u]);
  let cellSize = (gridEnd - gridStart) / vec2f(f32(WIDTH), f32(HEIGHT));
  let position = vec2f(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
  ${studyWindow ? 'if (!isInsideStudyWindow(position)) { return; }\n  ' : ''}let coordinate = (position - gridStart) / cellSize + vec2f(f32(MARGIN) - 0.5);
  if (!(coordinate.x >= 0.0 && coordinate.y >= 0.0 &&
      coordinate.x < f32(PADDED_WIDTH - 1u) && coordinate.y < f32(PADDED_HEIGHT - 1u))) {
    return;
  }
  let base = floor(coordinate);
  let fraction = coordinate - base;
  let cell = u32(base.y) * PADDED_WIDTH + u32(base.x);
  let unitExponent = getFixedPointUnitExponent(binScale[binScaleOffset], FIXED_POINT_HEADROOM_BITS);
  addFixedPointToWords(cell, quantizeFixedPoint(weight * (1.0 - fraction.x) * (1.0 - fraction.y), unitExponent));
  addFixedPointToWords(cell + 1u, quantizeFixedPoint(weight * fraction.x * (1.0 - fraction.y), unitExponent));
  addFixedPointToWords(cell + PADDED_WIDTH, quantizeFixedPoint(weight * (1.0 - fraction.x) * fraction.y, unitExponent));
  addFixedPointToWords(cell + PADDED_WIDTH + 1u, quantizeFixedPoint(weight * fraction.x * fraction.y, unitExponent));`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-bin-decode`,
      operation: OPERATION,
      variant: 'bin-decode',
      bindings: [
        {name: 'words', view: words, type: 'u32', access: 'read'},
        {name: 'binScale', view: scale, type: 'u32', access: 'read'},
        {name: 'density', view: density, type: 'f32', access: 'read_write'}
      ],
      invocationCount: paddedCellCount,
      workgroupSize: POINT_WORKGROUP_SIZE,
      ...getGate(GATE_SLOT.decode),
      declarations: `override FIXED_POINT_HEADROOM_BITS: u32;
${FIXED_POINT_WGSL}`,
      constants: {FIXED_POINT_HEADROOM_BITS: getFixedPointHeadroomBits(pointCount)},
      body: `let base = wordsOffset + index * ${FIXED_POINT_WORD_COUNT}u;
  let unitExponent = getFixedPointUnitExponent(binScale[binScaleOffset], FIXED_POINT_HEADROOM_BITS);
  density[densityOffset + index] = getFixedPointFloat(
    vec3<u32>(words[base], words[base + 1u], words[base + 2u]),
    unitExponent
  );`
    })
  ];

  // x passes run one invocation per row, y passes one per column. Ping-pong: density -> scratch
  // (box), scratch -> density (box), density -> scratch (box), scratch -> density (exponential).
  for (const axis of ['x', 'y'] as const) {
    const isX = axis === 'x';
    const lineCount = isX ? paddedHeight : paddedWidth;
    const lineLength = isX ? paddedWidth : paddedHeight;
    const constants = {
      LINE_LENGTH: lineLength,
      LINE_STEP: isX ? 1 : paddedWidth,
      LINE_STRIDE: isX ? paddedWidth : 1,
      PLAN_BASE: isX ? PLAN_X_BASE : PLAN_Y_BASE
    };
    const passes: ['box' | 'exponential', GraphDataView<'float32'>, GraphDataView<'float32'>][] = [
      ['box', density, scratch],
      ['box', scratch, density],
      ['box', density, scratch],
      ['exponential', scratch, density]
    ];
    for (const [passIndex, [kind, source, target]] of passes.entries()) {
      bodyNodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-blur-${axis}-${kind}-${passIndex}`,
          operation: OPERATION,
          variant: `blur-${kind}`,
          bindings: [
            {name: 'source', view: source, type: 'f32', access: 'read'},
            {name: 'destination', view: target, type: 'f32', access: 'read_write'},
            {name: 'plan', view: plan, type: 'f32', access: 'read'}
          ],
          invocationCount: lineCount,
          workgroupSize: LINE_WORKGROUP_SIZE,
          ...getGate(isX ? GATE_SLOT.blurX : GATE_SLOT.blurY),
          declarations: `override LINE_LENGTH: u32;
override LINE_STEP: u32;
override LINE_STRIDE: u32;
override PLAN_BASE: u32;
${kind === 'box' ? BOX_BLUR_DECLARATIONS : ''}`,
          constants,
          body: kind === 'box' ? BOX_BLUR_BODY : EXPONENTIAL_BLUR_BODY
        })
      );
    }
  }

  const finishBindings: WGSLKernelBinding[] = [
    ...(studyWindow ? [getStudyWindowBinding(studyWindow)] : []),
    {name: 'smoothed', view: density, type: 'f32', access: 'read'},
    gridBindings('params'),
    {name: 'plan', view: plan, type: 'f32', access: 'read'},
    ...props.bandwidthBindings,
    ...(normalization === 'probability' && props.weightSum
      ? [{name: 'weightSum', view: props.weightSum, type: 'f32', access: 'read'} as const]
      : []),
    {name: 'valuesOut', view: props.values, type: 'f32', access: 'read_write'}
  ];
  bodyNodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-finish`,
      operation: OPERATION,
      variant: `finish-${edgeCorrection}-${normalization}`,
      bindings: finishBindings,
      invocationCount: cellCount,
      workgroupSize: POINT_WORKGROUP_SIZE,
      ...getGate(GATE_SLOT.finish),
      declarations: `${gridConstants}
${GRID_COORDINATE_WGSL}
${studyWindow ? getKernelDensityStudyWindowWGSL(studyWindow, edgeCorrection === 'uniform' ? 'gaussian' : undefined) : edgeCorrection === 'none' ? '' : getWindowMassWGSL('gaussian')}`,
      body: `let column = index % WIDTH;
  let rasterRow = index / WIDTH;
  let gridStart = vec2f(params[paramsOffset], params[paramsOffset + 1u]);
  let gridEnd = vec2f(params[paramsOffset + 2u], params[paramsOffset + 3u]);
  let bandwidth = ${props.bandwidthExpression};
  let windowMin = vec2f(params[paramsOffset + 5u], params[paramsOffset + 6u]) - gridStart;
  let windowMax = vec2f(params[paramsOffset + 7u], params[paramsOffset + 8u]) - gridStart;
  let cellSize = (gridEnd - gridStart) / vec2f(f32(WIDTH), f32(HEIGHT));
  let cellOffset = (vec2f(f32(column), f32(rasterRow)) + vec2f(0.5)) * cellSize;
  let center = gridStart + cellOffset;
  let valid = plan[planOffset] > 0.5 && isFiniteValue(center.x) && isFiniteValue(center.y);
  var total = 0.0;
  if (valid) {
    let paddedIndex = (rasterRow + MARGIN) * PADDED_WIDTH + column + MARGIN;
    total = smoothed[smoothedOffset + paddedIndex] * plan[planOffset + 1u];
  }
  ${getKernelDensityResultWGSL(edgeCorrection, normalization, Boolean(studyWindow))}`
    })
  );

  return {
    planNodes,
    bodyNodes,
    exactGate: gate ? gate.getGate(GATE_SLOT.gather) : undefined
  };
}

/**
 * Running-sum box blur of one line with zero extension. The window sum of `2R + 1` cells plus the
 * fractional end taps `edge` is updated by two differences per step and recomputed directly every
 * `interval` cells, which bounds the chain of roundoff to `2R + 3 + interval` terms (library
 * rule 7: no naive running sum over thousands of cells). All terms are non-negative, so the
 * relative error of a direct sum of `n` terms is at most `n * 2^-24`.
 */
const BOX_BLUR_DECLARATIONS = /* wgsl */ `
fn readTap(base: u32, position: i32) -> f32 {
  if (position < 0 || position >= i32(LINE_LENGTH)) { return 0.0; }
  return source[sourceOffset + base + u32(position) * LINE_STEP];
}`;

const BOX_BLUR_BODY = `let base = index * LINE_STRIDE;
  let radius = i32(plan[planOffset + PLAN_BASE]);
  let edge = plan[planOffset + PLAN_BASE + 1u];
  let normalization = plan[planOffset + PLAN_BASE + 2u];
  let interval = i32(plan[planOffset + PLAN_BASE + 3u]);
  let lineLength = i32(LINE_LENGTH);
  var window = 0.0;
  for (var position = 0; position < lineLength; position++) {
    if (position % interval == 0) {
      var sum = 0.0;
      let last = min(position + radius, lineLength - 1);
      for (var tap = max(position - radius, 0); tap <= last; tap++) {
        sum += readTap(base, tap);
      }
      window = sum + edge * (readTap(base, position - radius - 1) + readTap(base, position + radius + 1));
    } else {
      window += (1.0 - edge) * (readTap(base, position + radius) - readTap(base, position - radius - 1)) +
        edge * (readTap(base, position + radius + 1) - readTap(base, position - radius - 2));
    }
    destination[destinationOffset + base + u32(position) * LINE_STEP] = max(window, 0.0) * normalization;
  }`;

/**
 * Two-sided exponential (discrete Laplace) filter of one line: a causal pass
 * `y[i] = (1 - a) x[i] + a y[i - 1]` and an anticausal pass `z[i] = (1 - a) y[i] + a z[i + 1]`
 * compose to the kernel `(1 - a) / (1 + a) a^|k|`. The anticausal pass starts from the closed form
 * `z[last] = y[last] / (1 + a)`, which accounts for the decaying tail the causal filter would
 * produce past the end of the line, so the zero extension is exact. Roundoff does not grow along
 * the line: an error decays by `a` per cell, so it stays below `2^-24 / (1 - a)` of the local peak.
 */
const EXPONENTIAL_BLUR_BODY = `let base = index * LINE_STRIDE;
  let pole = plan[planOffset + PLAN_BASE + 4u];
  var state = 0.0;
  for (var position = 0u; position < LINE_LENGTH; position++) {
    let slot = destinationOffset + base + position * LINE_STEP;
    state = (1.0 - pole) * source[sourceOffset + base + position * LINE_STEP] + pole * state;
    destination[slot] = state;
  }
  var next = state / (1.0 + pole);
  destination[destinationOffset + base + (LINE_LENGTH - 1u) * LINE_STEP] = next;
  for (var position = LINE_LENGTH - 1u; position > 0u; position--) {
    let slot = destinationOffset + base + (position - 1u) * LINE_STEP;
    next = (1.0 - pole) * destination[slot] + pole * next;
    destination[slot] = next;
  }`;
