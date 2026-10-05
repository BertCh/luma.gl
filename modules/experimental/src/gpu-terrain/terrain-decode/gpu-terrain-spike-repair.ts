// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUGraphConnectedComponents} from '@luma.gl/gpgpu/gpu-graph';
import type {GPURasterBand} from '../../gpu-raster/index';
import {createWGSLKernelNode, getWGSLFloatLiteral} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  captureGraphCommandNodes,
  importGraphBuffer,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  createNetworkAnalyticsTopology,
  getGraphViewGPUVector
} from '../../gpu-network/network-analysis/network-analytics-topology';
import {
  getTerrainBandViews,
  getTerrainElevationNodes,
  TERRAIN_WGSL_HELPERS,
  validateTerrainBandBelongsToGraph,
  validateTerrainBuffersDistinct,
  validateTerrainGrid
} from '../terrain-analysis/terrain-analysis-utils';

/** Word indices of the optional `statistics` output of {@link GPUTerrainSpikeRepair}. */
export const GPU_TERRAIN_SPIKE_REPAIR_STATISTICS = {
  /** Adjacent valid pixel pairs (right and down neighbours, each unordered pair once) with `|dh| > jump` before repair. */
  jumpCount: 0,
  /** Pixels whose height was shifted. */
  repairedPixelCount: 1,
  /** Components that were shifted. */
  shiftedComponentCount: 2,
  /** Jump pairs remaining after repair (equals `jumpCount` when nothing was repaired). */
  remainingJumpCount: 3,
  /** 1 when component labelling reached a fixed point, 0 when repair failed closed. */
  converged: 4
} as const;

/** Number of `uint32` words written to `GPUTerrainSpikeRepairProps.statistics`. */
export const GPU_TERRAIN_SPIKE_REPAIR_STATISTICS_LENGTH = 5;

/**
 * Default `componentIterations`. Each iteration is one relaxation plus pointer-jumping round of
 * `GPUGraphConnectedComponents`; see {@link GPUTerrainSpikeRepairProps.componentIterations}.
 */
export const GPU_TERRAIN_SPIKE_REPAIR_DEFAULT_COMPONENT_ITERATIONS = 32;

/** Largest magnitude of the Terrarium-step multiple `k` that is voted on. */
const MAXIMUM_STEP_MULTIPLE = 127;
const COMPONENT_BIT_COUNT = 8;
const SCALAR_MAXIMUM_SIZE = 0;
const SCALAR_MAIN_LABEL = 1;
const SCALAR_VALID_COUNT = 2;
const SCALAR_LENGTH = 4;
const NO_VERTEX = '0xffffffffu';

/**
 * Properties for {@link GPUTerrainSpikeRepair}.
 *
 * The contributor is cell-size independent: it compares heights only, never horizontal distances.
 * Topology: grid size, `elevation` format and every numeric setting (all baked into WGSL),
 * `componentIterations`, and which optional outputs exist. Per-frame: elevation contents.
 */
export type GPUTerrainSpikeRepairProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-spike-repair'`. */
  id?: string;
  /** Grid width in pixels. */
  width: number;
  /** Grid height in pixels. */
  height: number;
  /**
   * Decoded full-resolution heights. Canonicalised with calibration and nodata applied; non-finite
   * values are nodata. Invalid pixels are never repaired and never contribute a height.
   */
  elevation: GPURasterBand;
  /** Height of one encoding step of the faulty channel. Default 256 (Terrarium R byte). */
  step?: number;
  /** Neighbour jump `|dh| <= jump` that keeps pixels in one component. Default 200. */
  jump?: number;
  /** Allowed deviation of a seam jump from a multiple of `step`; must be `< step / 2`. Default 40. */
  tolerance?: number;
  /**
   * Fraction of a component's border seams that must agree on one multiple of `step`. Must be a
   * multiple of 0.01 in `(0.5, 1]`; the test is exact integer arithmetic. Default 0.8.
   */
  agreement?: number;
  /**
   * Components larger than this fraction of the valid pixels are never shifted. Must be a multiple
   * of 0.01 in `(0, 1]`. Default 0.25.
   */
  maximumComponentFraction?: number;
  /**
   * Compiled `GPUGraphConnectedComponents` iterations (1 to 1024). Default
   * {@link GPU_TERRAIN_SPIKE_REPAIR_DEFAULT_COMPONENT_ITERATIONS}. If labelling has not converged
   * the repair fails closed: the input is copied unchanged and `converged` is 0.
   */
  componentIterations?: number;
  /** Repaired heights, one float32 per pixel. Invalid pixels hold canonical NaN. */
  values: GraphDataView<'float32'>;
  /** Validity (0 or 1) per pixel, copied from the canonical input. */
  validity: GraphDataView<'uint32'>;
  /** Optional component label per pixel: the lowest pixel index of its component. Invalid pixels label themselves. */
  labels?: GraphDataView<'uint32'>;
  /** Optional `uint32` view of at least 5 words, see {@link GPU_TERRAIN_SPIKE_REPAIR_STATISTICS}. */
  statistics?: GraphDataView<'uint32'>;
};

/** Returns `value * 100` when it is a whole number of hundredths, otherwise `undefined`. */
function getHundredths(value: number | undefined, fallback: number): number | undefined {
  const hundredths = Math.round((value ?? fallback) * 100);
  return Math.abs((value ?? fallback) * 100 - hundredths) < 1e-9 ? hundredths : undefined;
}

/**
 * Repairs +/-1 errors in the high byte of a Terrarium-style elevation encoding, an opt-in port of
 * mt-image `validateTile` without its out-of-range fill.
 *
 * A +/-1 error in the Terrarium R byte (canvas anti-fingerprinting noise, failed decodes) is a
 * +/-256 m spike or plateau. Valid pixels are split into 4-connected components across edges with
 * `|dh| <= jump`. Any component other than the largest whose border seams agree (at least
 * `agreement`) on a jump of `step * k +/- tolerance` (`k != 0`) is shifted by `-step * k`.
 *
 * Opt-in only. In noised realms a real, enclosed butte with 216 to 296 m walls looks exactly like a
 * +256 m error and is shifted when its seams agree. Real cliffs that are not a consistent multiple
 * of the step, and components larger than `maximumComponentFraction`, are left alone.
 *
 * Run the repair on decoded full-resolution heights BEFORE any filtering or resampling: a filtered
 * spike is no longer a clean `step * k` jump. Out-of-range pixels are nodata from the decode step
 * (`GPUTerrainRGBDecode` `validRange`) and stay nodata; mt-image fills them from a neighbour
 * median, which is a separate, explicit choice that this contributor does not make. Nodata never bleeds:
 * invalid pixels are isolated, never seams, and their value slot is never read as a height.
 *
 * Components use `GPUGraphConnectedComponents` over a fixed-stride forward CSR (two slots per
 * pixel: right and down neighbour, `0xffffffff` unless both pixels are valid and
 * `|dh| <= jump`). The gpu-raster `GPURasterConnectedComponents` only labels a binary foreground
 * mask and cannot express an edge predicate, so the graph primitive is used. Labels are the lowest
 * member pixel index. Everything after labelling is exact integer atomics and deterministic.
 *
 * Main component: the maximum size, ties to the lowest label. This equals mt-image, whose BFS
 * discovers components in ascending seed order and keeps the first strict maximum. mt-image skips
 * components with `size > N / 4` where `N` counts all pixels; this contributor uses the valid-pixel
 * count instead, since nodata pixels are not part of any component.
 *
 * Seam votes. For every ordered pair `(a, b)` of 4-adjacent valid pixels in different components,
 * `seams[label(a)]++`, `dh = h_a - h_b`, `k = round(dh / step)`, and the pair votes for `k` iff
 * `k != 0` and `|dh - step * k| < tolerance`. mt-image takes the mode `k` per component and shifts
 * iff `votes(mode) >= agreement * seams`. A per-component histogram is too large on the GPU, so
 * this contributor uses an exact equivalent. Keep per component `voteCount` and eight bit counts of
 * `k + 128`. If some `k*` has `votes(k*) >= agreement * seams` with `agreement > 0.5`, then
 * `votes(k*) >= agreement * voteCount > voteCount / 2`, so `k*` is a strict majority of the votes
 * and each bit of `k* + 128` is the majority bit: the bit counts `> voteCount / 2` reproduce
 * `k*` exactly. A second pass counts votes equal to that candidate (`agreeCount`) and the shift
 * fires iff `agreeCount * 100 >= seams * agreementPercent`. If no `k*` reaches the threshold, the
 * candidate (whatever it is) has `agreeCount < agreement * seams`, so nothing shifts; both
 * rules therefore agree whenever the mode rule fires. Votes with `|k| > 127` do not vote (they
 * count as non-agreeing seams), so a component whose true mode needs `|k| > 127` (a height span
 * above 127 steps, 32 km at step 256, beyond any `validRange`) is left alone.
 *
 * Rounding: WGSL `round()` is ties-to-even and JavaScript `Math.round` is half-up. They differ
 * only at `dh = step * (j + 1/2)`, where `|dh - step * k| = step / 2 >= tolerance`, so the pair
 * does not vote either way; `tolerance < step / 2` is validated for exactly that reason.
 * Terrarium heights are multiples of 1/256 below 2^15, so `dh` is exact in f32; with a
 * power-of-two `step` (256, 25.6 is not) the reciprocal and `step * k` are exact too, making the
 * kernels bit-identical to a float64 oracle. For other steps (Mapbox R byte 6553.6 m, G byte
 * 25.6 m) `k` comes from `dh * fround(1 / step)` and `step * k` is rounded to f32, which can only
 * change a vote when `|dh - step * k|` is within about 1e-3 m of `tolerance`.
 *
 * Memory: the per-component bit counts need `8 * width * height` words in one storage binding, so
 * tiles are limited by `maxStorageBufferBindingSize` (2048 x 2048 at 128 MiB).
 *
 * Owns a CSR, label buffer and small scalars released by {@link destroy}, created on first use.
 * Add the contributor to one graph at a time; destroy compiled graphs first.
 */
export class GPUTerrainSpikeRepair implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTerrainSpikeRepairProps;
  /** Component iterations in effect. */
  readonly componentIterations: number;
  private readonly agreementPercent: number;
  private readonly maximumComponentPercent: number;
  private ownedBuffers?: {
    offsets: Buffer;
    neighbors: Buffer;
    labels: Buffer;
    converged: Buffer;
    status: Buffer;
  };

  constructor(props: GPUTerrainSpikeRepairProps) {
    this.id = props.id ?? 'terrain-spike-repair';
    this.props = props;
    const {id} = this;
    const pixelCount = validateTerrainGrid(id, props.width, props.height);
    const step = props.step ?? 256;
    const jump = props.jump ?? 200;
    const tolerance = props.tolerance ?? 40;
    if (!Number.isFinite(step) || step <= 0) {
      throw new Error(`${id} step must be finite and positive`);
    }
    if (!Number.isFinite(jump) || jump <= 0) {
      throw new Error(`${id} jump must be finite and positive`);
    }
    if (!Number.isFinite(tolerance) || tolerance <= 0 || tolerance >= step / 2) {
      throw new Error(`${id} tolerance must be positive and less than step / 2`);
    }
    const agreementPercent = getHundredths(props.agreement, 0.8);
    if (agreementPercent === undefined || agreementPercent <= 50 || agreementPercent > 100) {
      throw new Error(`${id} agreement must be a multiple of 0.01 in (0.5, 1]`);
    }
    const maximumPercent = getHundredths(props.maximumComponentFraction, 0.25);
    if (maximumPercent === undefined || maximumPercent < 1 || maximumPercent > 100) {
      throw new Error(`${id} maximumComponentFraction must be a multiple of 0.01 in (0, 1]`);
    }
    this.agreementPercent = agreementPercent;
    this.maximumComponentPercent = maximumPercent;
    this.componentIterations =
      props.componentIterations ?? GPU_TERRAIN_SPIKE_REPAIR_DEFAULT_COMPONENT_ITERATIONS;
    if (
      !Number.isSafeInteger(this.componentIterations) ||
      this.componentIterations < 1 ||
      this.componentIterations > 1024
    ) {
      throw new Error(`${id} componentIterations must be an integer between 1 and 1024`);
    }
    validatePackedView(props.values, ['float32'], `${id} values`);
    validatePackedUint32View(props.validity, `${id} validity`);
    for (const [name, view] of [
      ['values', props.values],
      ['validity', props.validity],
      ['labels', props.labels]
    ] as const) {
      if (view) {
        if (name === 'labels') {
          validatePackedUint32View(view, `${id} labels`);
        }
        if (view.length !== pixelCount) {
          throw new Error(`${id} ${name} must contain one value per pixel`);
        }
      }
    }
    if (props.statistics) {
      validatePackedUint32View(props.statistics, `${id} statistics`);
      if (props.statistics.length < GPU_TERRAIN_SPIKE_REPAIR_STATISTICS_LENGTH) {
        throw new Error(
          `${id} statistics must contain at least ${GPU_TERRAIN_SPIKE_REPAIR_STATISTICS_LENGTH} uint32 rows`
        );
      }
    }
    validateTerrainBuffersDistinct(
      id,
      [props.values, props.validity, props.labels, props.statistics],
      getTerrainBandViews(props.elevation)
    );
  }

  /** Returns canonical elevation, graph construction, components, vote, decision and shift nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {width, height} = props;
    const pixelCount = width * height;
    validateTerrainBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, [
      props.values,
      props.validity,
      props.labels,
      props.statistics
    ]);
    const limits = graph.device.limits;
    const bitWords = COMPONENT_BIT_COUNT * pixelCount;
    if (bitWords * 4 > limits.maxStorageBufferBindingSize) {
      throw new Error(
        `${id} needs a ${bitWords * 4} byte vote buffer, more than the device storage binding limit`
      );
    }
    const operation = 'GPUTerrainSpikeRepair';
    const step = props.step ?? 256;
    const jump = props.jump ?? 200;
    const tolerance = props.tolerance ?? 40;

    if (!this.ownedBuffers) {
      const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
      const create = (name: string, words: number) =>
        graph.device.createBuffer({id: `${id}-${name}`, byteLength: Math.max(words, 1) * 4, usage});
      this.ownedBuffers = {
        offsets: create('offsets', pixelCount + 1),
        neighbors: create('neighbors', 2 * pixelCount),
        labels: create('labels', pixelCount),
        converged: create('converged', 1),
        status: create('status', 4)
      };
    }
    const owned = this.ownedBuffers;
    const offsets = importGraphBuffer(
      graph,
      `${id}-offsets`,
      owned.offsets,
      'uint32',
      pixelCount + 1
    );
    const neighbors = importGraphBuffer(
      graph,
      `${id}-neighbors`,
      owned.neighbors,
      'uint32',
      2 * pixelCount
    );
    const labels = importGraphBuffer(graph, `${id}-labels`, owned.labels, 'uint32', pixelCount);
    const converged = importGraphBuffer(graph, `${id}-converged`, owned.converged, 'uint32', 1);

    const source = getTerrainElevationNodes(graph, id, props.elevation, width, height, true);
    const nodes: GPUCommandNode<Parameters>[] = [...source.nodes];
    const sourceValues = source.band.storage.values as GraphDataView<'float32'>;
    const sourceValidity = source.band.validity as GraphDataView<'uint32'>;

    const transient = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);
    const componentSize = transient('component-size', pixelCount);
    const seamCount = transient('seam-count', pixelCount);
    const voteCount = transient('vote-count', pixelCount);
    const agreeCount = transient('agree-count', pixelCount);
    const bitCount = transient('bit-count', bitWords);
    const scalars = transient('scalars', SCALAR_LENGTH);
    const shiftKey = transient('shift-key', pixelCount);
    const statistics =
      props.statistics ?? transient('statistics', GPU_TERRAIN_SPIKE_REPAIR_STATISTICS_LENGTH);
    const word = GPU_TERRAIN_SPIKE_REPAIR_STATISTICS;

    const declarations = /* wgsl */ `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const PIXEL_COUNT: u32 = ${pixelCount}u;
const NO_VERTEX: u32 = ${NO_VERTEX};
const JUMP: f32 = ${getWGSLFloatLiteral(jump)};
const STEP: f32 = ${getWGSLFloatLiteral(step)};
const INVERSE_STEP: f32 = ${getWGSLFloatLiteral(1 / step)};
const TOLERANCE: f32 = ${getWGSLFloatLiteral(tolerance)};
const BIT_COUNT: u32 = ${COMPONENT_BIT_COUNT}u;
${TERRAIN_WGSL_HELPERS}`;
    const pixelHelpers = /* wgsl */ `
fn isValidPixel(pixel: u32) -> bool {
  return validity[validityOffset + pixel] != 0u && isFiniteValue(values[valuesOffset + pixel]);
}
fn getNeighbor(pixel: u32, direction: u32) -> u32 {
  let column = pixel % WIDTH;
  let row = pixel / WIDTH;
  if (direction == 0u) { return select(NO_VERTEX, pixel + 1u, column + 1u < WIDTH); }
  if (direction == 1u) { return select(NO_VERTEX, pixel - 1u, column > 0u); }
  if (direction == 2u) { return select(NO_VERTEX, pixel + WIDTH, row + 1u < HEIGHT); }
  return select(NO_VERTEX, pixel - WIDTH, row > 0u);
}`;
    const voteHelpers = /* wgsl */ `
// Biased step multiple k + 128 in 1..255 when the height difference votes, else 0.
fn getVoteKey(difference: f32) -> u32 {
  let multiple = round(difference * INVERSE_STEP);
  if (multiple == 0.0 || abs(multiple) > ${MAXIMUM_STEP_MULTIPLE}.0) { return 0u; }
  if (!(abs(difference - multiple * STEP) < TOLERANCE)) { return 0u; }
  return u32(i32(multiple) + 128);
}`;

    // Zero all accumulators, scalars and statistics.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-clear`,
        operation,
        variant: 'clear',
        bindings: [
          {name: 'componentSize', view: componentSize, type: 'u32', access: 'read_write'},
          {name: 'seamCount', view: seamCount, type: 'u32', access: 'read_write'},
          {name: 'voteCount', view: voteCount, type: 'u32', access: 'read_write'},
          {name: 'agreeCount', view: agreeCount, type: 'u32', access: 'read_write'},
          {name: 'bitCount', view: bitCount, type: 'u32', access: 'read_write'},
          {name: 'scalars', view: scalars, type: 'u32', access: 'read_write'},
          {name: 'statistics', view: statistics, type: 'u32', access: 'read_write'}
        ],
        invocationCount: bitWords,
        declarations: `const PIXEL_COUNT: u32 = ${pixelCount}u;`,
        body: `bitCount[bitCountOffset + index] = 0u;
  if (index < PIXEL_COUNT) {
    componentSize[componentSizeOffset + index] = 0u;
    seamCount[seamCountOffset + index] = 0u;
    voteCount[voteCountOffset + index] = 0u;
    agreeCount[agreeCountOffset + index] = 0u;
  }
  if (index < ${SCALAR_LENGTH}u) {
    scalars[scalarsOffset + index] = select(0u, 0xffffffffu, index == ${SCALAR_MAIN_LABEL}u);
  }
  if (index < ${GPU_TERRAIN_SPIKE_REPAIR_STATISTICS_LENGTH}u) {
    statistics[statisticsOffset + index] = 0u;
  }`
      })
    );

    // Fixed-stride forward CSR: slot 2i is the right neighbour, 2i + 1 the down neighbour.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-graph`,
        operation,
        variant: 'graph',
        bindings: [
          {name: 'values', view: sourceValues, type: 'f32', access: 'read'},
          {name: 'validity', view: sourceValidity, type: 'u32', access: 'read'},
          {name: 'offsets', view: offsets, type: 'u32', access: 'read_write'},
          {name: 'neighbors', view: neighbors, type: 'u32', access: 'read_write'}
        ],
        invocationCount: pixelCount + 1,
        declarations: `${declarations}${pixelHelpers}
fn getEdge(pixel: u32, direction: u32) -> u32 {
  let neighbor = getNeighbor(pixel, direction);
  if (neighbor == NO_VERTEX || !isValidPixel(pixel) || !isValidPixel(neighbor)) {
    return NO_VERTEX;
  }
  let difference = values[valuesOffset + pixel] - values[valuesOffset + neighbor];
  return select(NO_VERTEX, neighbor, abs(difference) <= JUMP);
}`,
        body: `offsets[offsetsOffset + index] = 2u * index;
  if (index < PIXEL_COUNT) {
    neighbors[neighborsOffset + 2u * index] = getEdge(index, 0u);
    neighbors[neighborsOffset + 2u * index + 1u] = getEdge(index, 2u);
  }`
      })
    );

    nodes.push(
      ...captureGraphCommandNodes(graph, () => {
        const topology = createNetworkAnalyticsTopology({
          id,
          nodeCount: pixelCount,
          offsets,
          neighbors,
          status: owned.status
        });
        new GPUGraphConnectedComponents({
          id: `${id}-components`,
          topology,
          output: getGraphViewGPUVector(id, 'component labels', labels),
          iterations: this.componentIterations,
          converged: getGraphViewGPUVector(id, 'component converged', converged)
        }).addToGraph(graph);
      })
    );

    // Component sizes, valid pixel count and the pre-repair jump count.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-sizes`,
        operation,
        variant: 'sizes',
        bindings: [
          {name: 'values', view: sourceValues, type: 'f32', access: 'read'},
          {name: 'validity', view: sourceValidity, type: 'u32', access: 'read'},
          {name: 'labels', view: labels, type: 'u32', access: 'read'},
          {name: 'componentSize', view: componentSize, type: 'atomic<u32>', access: 'read_write'},
          {name: 'scalars', view: scalars, type: 'atomic<u32>', access: 'read_write'},
          {name: 'statistics', view: statistics, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: pixelCount,
        declarations: `${declarations}${pixelHelpers}`,
        body: `if (!isValidPixel(index)) {
    return;
  }
  let label = labels[labelsOffset + index];
  if (label < PIXEL_COUNT) {
    atomicAdd(&componentSize[componentSizeOffset + label], 1u);
  }
  atomicAdd(&scalars[scalarsOffset + ${SCALAR_VALID_COUNT}u], 1u);
  let value = values[valuesOffset + index];
  var count = 0u;
  for (var direction = 0u; direction < 4u; direction += 2u) {
    let neighbor = getNeighbor(index, direction);
    if (neighbor != NO_VERTEX && isValidPixel(neighbor) &&
        abs(value - values[valuesOffset + neighbor]) > JUMP) {
      count++;
    }
  }
  if (count > 0u) {
    atomicAdd(&statistics[statisticsOffset + ${word.jumpCount}u], count);
  }`
      })
    );

    // Main component: maximum size, then the lowest label of that size.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-main-size`,
        operation,
        variant: 'main-size',
        bindings: [
          {name: 'componentSize', view: componentSize, type: 'u32', access: 'read'},
          {name: 'scalars', view: scalars, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: pixelCount,
        body: `let size = componentSize[componentSizeOffset + index];
  if (size > 0u) {
    atomicMax(&scalars[scalarsOffset + ${SCALAR_MAXIMUM_SIZE}u], size);
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-main-label`,
        operation,
        variant: 'main-label',
        bindings: [
          {name: 'componentSize', view: componentSize, type: 'u32', access: 'read'},
          {name: 'scalars', view: scalars, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: pixelCount,
        body: `let size = componentSize[componentSizeOffset + index];
  if (size > 0u && size == atomicLoad(&scalars[scalarsOffset + ${SCALAR_MAXIMUM_SIZE}u])) {
    atomicMin(&scalars[scalarsOffset + ${SCALAR_MAIN_LABEL}u], index);
  }`
      })
    );

    // Seam votes over ordered pairs of 4-adjacent valid pixels in different components.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-votes`,
        operation,
        variant: 'votes',
        bindings: [
          {name: 'values', view: sourceValues, type: 'f32', access: 'read'},
          {name: 'validity', view: sourceValidity, type: 'u32', access: 'read'},
          {name: 'labels', view: labels, type: 'u32', access: 'read'},
          {name: 'seamCount', view: seamCount, type: 'atomic<u32>', access: 'read_write'},
          {name: 'voteCount', view: voteCount, type: 'atomic<u32>', access: 'read_write'},
          {name: 'bitCount', view: bitCount, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: pixelCount,
        declarations: `${declarations}${pixelHelpers}${voteHelpers}`,
        body: `if (!isValidPixel(index)) {
    return;
  }
  let label = labels[labelsOffset + index];
  if (label >= PIXEL_COUNT) {
    return;
  }
  let value = values[valuesOffset + index];
  for (var direction = 0u; direction < 4u; direction++) {
    let neighbor = getNeighbor(index, direction);
    if (neighbor == NO_VERTEX || !isValidPixel(neighbor) ||
        labels[labelsOffset + neighbor] == label) {
      continue;
    }
    atomicAdd(&seamCount[seamCountOffset + label], 1u);
    let key = getVoteKey(value - values[valuesOffset + neighbor]);
    if (key == 0u) {
      continue;
    }
    atomicAdd(&voteCount[voteCountOffset + label], 1u);
    for (var bit = 0u; bit < BIT_COUNT; bit++) {
      if (((key >> bit) & 1u) != 0u) {
        atomicAdd(&bitCount[bitCountOffset + label * BIT_COUNT + bit], 1u);
      }
    }
  }`
      })
    );

    const candidateHelper = /* wgsl */ `
// Majority-bit candidate key of a component, 0 when no bit has a majority.
fn getCandidateKey(label: u32) -> u32 {
  let votes = voteCount[voteCountOffset + label];
  var key = 0u;
  for (var bit = 0u; bit < BIT_COUNT; bit++) {
    if (bitCount[bitCountOffset + label * BIT_COUNT + bit] * 2u > votes) {
      key |= 1u << bit;
    }
  }
  return key;
}`;

    // Count votes that equal the candidate.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-agree`,
        operation,
        variant: 'agree',
        bindings: [
          {name: 'values', view: sourceValues, type: 'f32', access: 'read'},
          {name: 'validity', view: sourceValidity, type: 'u32', access: 'read'},
          {name: 'labels', view: labels, type: 'u32', access: 'read'},
          {name: 'voteCount', view: voteCount, type: 'u32', access: 'read'},
          {name: 'bitCount', view: bitCount, type: 'u32', access: 'read'},
          {name: 'agreeCount', view: agreeCount, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: pixelCount,
        declarations: `${declarations}${pixelHelpers}${voteHelpers}${candidateHelper}`,
        body: `if (!isValidPixel(index)) {
    return;
  }
  let label = labels[labelsOffset + index];
  if (label >= PIXEL_COUNT) {
    return;
  }
  let candidate = getCandidateKey(label);
  if (candidate == 0u) {
    return;
  }
  let value = values[valuesOffset + index];
  for (var direction = 0u; direction < 4u; direction++) {
    let neighbor = getNeighbor(index, direction);
    if (neighbor == NO_VERTEX || !isValidPixel(neighbor) ||
        labels[labelsOffset + neighbor] == label) {
      continue;
    }
    if (getVoteKey(value - values[valuesOffset + neighbor]) == candidate) {
      atomicAdd(&agreeCount[agreeCountOffset + label], 1u);
    }
  }`
      })
    );

    // Per component decision: biased shift key, 0 for no shift.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-decide`,
        operation,
        variant: 'decide',
        bindings: [
          {name: 'componentSize', view: componentSize, type: 'u32', access: 'read'},
          {name: 'seamCount', view: seamCount, type: 'u32', access: 'read'},
          {name: 'voteCount', view: voteCount, type: 'u32', access: 'read'},
          {name: 'agreeCount', view: agreeCount, type: 'u32', access: 'read'},
          {name: 'bitCount', view: bitCount, type: 'u32', access: 'read'},
          {name: 'scalars', view: scalars, type: 'u32', access: 'read'},
          {name: 'converged', view: converged, type: 'u32', access: 'read'},
          {name: 'shiftKey', view: shiftKey, type: 'u32', access: 'read_write'}
        ],
        invocationCount: pixelCount,
        declarations: `const PIXEL_COUNT: u32 = ${pixelCount}u;
const BIT_COUNT: u32 = ${COMPONENT_BIT_COUNT}u;
const AGREEMENT_PERCENT: u32 = ${this.agreementPercent}u;
const MAXIMUM_COMPONENT_PERCENT: u32 = ${this.maximumComponentPercent}u;${candidateHelper}`,
        body: `let size = componentSize[componentSizeOffset + index];
  let seams = seamCount[seamCountOffset + index];
  let key = getCandidateKey(index);
  let validCount = scalars[scalarsOffset + ${SCALAR_VALID_COUNT}u];
  let mainLabel = scalars[scalarsOffset + ${SCALAR_MAIN_LABEL}u];
  let shift = converged[convergedOffset] != 0u && size > 0u && index != mainLabel &&
    size * 100u <= validCount * MAXIMUM_COMPONENT_PERCENT && seams > 0u && key != 0u &&
    agreeCount[agreeCountOffset + index] * 100u >= seams * AGREEMENT_PERCENT;
  shiftKey[shiftKeyOffset + index] = select(0u, key, shift);`
      })
    );

    // Apply shifts, publish validity and copy; mark invalid pixels with canonical NaN.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-shift`,
        operation,
        variant: 'shift',
        bindings: [
          {name: 'values', view: sourceValues, type: 'f32', access: 'read'},
          {name: 'validity', view: sourceValidity, type: 'u32', access: 'read'},
          {name: 'labels', view: labels, type: 'u32', access: 'read'},
          {name: 'shiftKey', view: shiftKey, type: 'u32', access: 'read'},
          {name: 'converged', view: converged, type: 'u32', access: 'read'},
          {name: 'valuesOut', view: props.values, type: 'f32', access: 'read_write'},
          {name: 'validityOut', view: props.validity, type: 'u32', access: 'read_write'},
          {name: 'statistics', view: statistics, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: pixelCount,
        declarations: `${declarations}${pixelHelpers}`,
        body: `if (index == 0u) {
    atomicStore(&statistics[statisticsOffset + ${word.converged}u],
      select(0u, 1u, converged[convergedOffset] != 0u));
  }
  if (!isValidPixel(index)) {
    valuesOut[valuesOutOffset + index] = bitcast<f32>(0x7fc00000u | (index & 0u));
    validityOut[validityOutOffset + index] = 0u;
    return;
  }
  validityOut[validityOutOffset + index] = 1u;
  var value = values[valuesOffset + index];
  let label = labels[labelsOffset + index];
  var key = 0u;
  if (label < PIXEL_COUNT) {
    key = shiftKey[shiftKeyOffset + label];
  }
  if (key != 0u) {
    value = value - STEP * f32(i32(key) - 128);
    atomicAdd(&statistics[statisticsOffset + ${word.repairedPixelCount}u], 1u);
    if (label == index) {
      atomicAdd(&statistics[statisticsOffset + ${word.shiftedComponentCount}u], 1u);
    }
  }
  valuesOut[valuesOutOffset + index] = value;`
      })
    );

    if (props.labels) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-publish-labels`,
          operation,
          variant: 'labels',
          bindings: [
            {name: 'labels', view: labels, type: 'u32', access: 'read'},
            {name: 'labelsOut', view: props.labels, type: 'u32', access: 'read_write'}
          ],
          invocationCount: pixelCount,
          body: 'labelsOut[labelsOutOffset + index] = labels[labelsOffset + index];'
        })
      );
    }

    // Remaining jumps on the repaired output.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-recount`,
        operation,
        variant: 'recount',
        bindings: [
          {name: 'values', view: props.values, type: 'f32', access: 'read'},
          {name: 'validity', view: props.validity, type: 'u32', access: 'read'},
          {name: 'statistics', view: statistics, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: pixelCount,
        declarations: `${declarations}${pixelHelpers}`,
        body: `if (!isValidPixel(index)) {
    return;
  }
  let value = values[valuesOffset + index];
  var count = 0u;
  for (var direction = 0u; direction < 4u; direction += 2u) {
    let neighbor = getNeighbor(index, direction);
    if (neighbor != NO_VERTEX && isValidPixel(neighbor) &&
        abs(value - values[valuesOffset + neighbor]) > JUMP) {
      count++;
    }
  }
  if (count > 0u) {
    atomicAdd(&statistics[statisticsOffset + ${word.remainingJumpCount}u], count);
  }`
      })
    );
    return nodes;
  }

  /** Releases the owned CSR, label, convergence and status buffers. Destroy compiled graphs first. */
  destroy(): void {
    if (this.ownedBuffers) {
      for (const buffer of Object.values(this.ownedBuffers)) {
        buffer.destroy();
      }
      this.ownedBuffers = undefined;
    }
  }
}
