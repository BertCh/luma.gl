// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPURasterBand, GPURasterMetadata} from '../../gpu-raster/index';
import {assertCompatibleGPUVerticalDatums} from '../../analysis-core/contracts/vertical-datum';
import {createFillNode, createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  getStorageBindingElementLimit,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  getRasterGridGeoreference,
  getRasterGroundCellSizeWGSL,
  type GPURasterCellSizeMode,
  validateRasterCellSizeMode,
  writeRasterGridParameterValues
} from '../../analysis-core/raster-grid';
import {
  getRasterBandViews,
  getRasterBufferBandNodes,
  RASTER_WGSL_HELPERS,
  validateRasterBandBelongsToGraph,
  validateParameterTarget,
  validateRasterBuffersDistinct,
  validateRasterGrid,
  validateRasterGridAxisAligned,
  validateRasterParameters
} from '../../gpu-raster/raster-band-utils';
import {
  TERRAIN_FEATURES_DISC_WGSL,
  TERRAIN_FEATURES_WGSL_CONSTANTS,
  validateTerrainFeaturesCellSize,
  validateTerrainFeaturesMaximumRadiusPixels,
  validateTerrainFeaturesScalar
} from './terrain-features-utils';

/**
 * Status codes written to `GPUTerrainPeakSnapOutput.snapStatus`, in precedence order:
 *
 * - `unchanged` (0): the best disc pixel is the pixel nearest the candidate. The position is kept.
 * - `snapped` (1): moved to the pixel centre of the best disc pixel.
 * - `onRing` (2): the interior rule is on and the best pixel lies on the disc ring, so it is a
 *   flank rather than a summit. The position is kept.
 * - `rejectedMove` (3): the move exceeds `maximumMove`. The position is kept.
 * - `rejectedHeight` (4): `|best - reference|` exceeds `maximumHeightChange`. The position is kept.
 * - `noData` (5): no valid pixel in the disc (or the cell size or radius is unusable).
 * - `outside` (6): the candidate is non-finite or outside `[0, width - 1] x [0, height - 1]`.
 *
 * Evaluation order is outside, noData, onRing, unchanged, rejectedMove, rejectedHeight, snapped.
 */
export const GPU_TERRAIN_PEAK_SNAP_STATUS = {
  unchanged: 0,
  snapped: 1,
  onRing: 2,
  rejectedMove: 3,
  rejectedHeight: 4,
  noData: 5,
  outside: 6
} as const;

/** Number of float32 values read from `GPUTerrainPeakSnapProps.parameters`. */
export const GPU_TERRAIN_PEAK_SNAP_PARAMETER_LENGTH = 8;

/**
 * Largest `candidates` row count of {@link GPUTerrainPeakSnap} on `device`: the kernel keeps five
 * `uint32` result words (20 bytes) per candidate in one storage binding, so 6710886 on a default
 * device (128 MiB bindings).
 */
export function getGPUTerrainPeakSnapMaximumCandidateCount(device: Device): number {
  return getStorageBindingElementLimit(device, 20);
}

/** Cell size and row-edge slots of the peak snap parameter layout. */
const TERRAIN_PEAK_SNAP_GRID_SLOTS = {cellSizeIndex: 3, northEdgeIndex: 5} as const;

/**
 * CPU-side description packed by {@link getGPUTerrainPeakSnapParameterValues}. The cell size and
 * latitude band come from the grid passed next to it and follow
 * `GPUTerrainPeakSnapProps.cellSizeMode`. `radius`, `maximumMove`, and `maximumHeightChange` are
 * ground metres or elevation units.
 */
export type GPUTerrainPeakSnapSettings = {
  /** Default disc radius in ground metres, used for candidates without a radius. Defaults to 150. */
  radius?: number;
  /** Largest accepted move in ground metres. Defaults to 300. */
  maximumMove?: number;
  /** Largest accepted `|best - reference|` in elevation units. Defaults to 80. */
  maximumHeightChange?: number;
  /** Keep the candidate when the best pixel lies on the disc ring (a flank). Defaults to true. */
  interior?: boolean;
};

/**
 * Packs peak snap settings into
 * `[radius, maximumMove, maximumHeightChange, cellSizeX, cellSizeY, northEdge, southEdge, interior]`,
 * with the cell size and the row-edge world y taken from `grid`.
 *
 * @throws If a radius, move, or height limit is not finite and positive (`maximumHeightChange`
 * may be 0), the grid cell size is not finite and positive, or `target` is too short.
 */
export function getGPUTerrainPeakSnapParameterValues(
  settings: GPUTerrainPeakSnapSettings,
  grid: GPURasterMetadata,
  target: Float32Array = new Float32Array(GPU_TERRAIN_PEAK_SNAP_PARAMETER_LENGTH)
): Float32Array {
  const name = 'getGPUTerrainPeakSnapParameterValues';
  validateParameterTarget(name, target, GPU_TERRAIN_PEAK_SNAP_PARAMETER_LENGTH);
  const radius = settings.radius ?? 150;
  const maximumMove = settings.maximumMove ?? 300;
  const maximumHeightChange = settings.maximumHeightChange ?? 80;
  validateTerrainFeaturesScalar(name, 'radius', radius, 0, true);
  validateTerrainFeaturesScalar(name, 'maximumMove', maximumMove, 0);
  validateTerrainFeaturesScalar(name, 'maximumHeightChange', maximumHeightChange, 0);
  validateTerrainFeaturesCellSize(name, getRasterGridGeoreference(grid).cellSize);
  target[0] = radius;
  target[1] = maximumMove;
  target[2] = maximumHeightChange;
  target[7] = settings.interior === false ? 0 : 1;
  writeRasterGridParameterValues(target, grid, TERRAIN_PEAK_SNAP_GRID_SLOTS);
  return target;
}

/** Status words of {@link GPUTerrainPeakSnap}. */
export type GPUTerrainPeakSnapStatus = {
  /** One-row `uint32` set to 1 when any candidate's radius was clamped, else 0. */
  radiusClamped?: GraphDataView<'uint32'>;
};

/** Caller-owned outputs of {@link GPUTerrainPeakSnap}, one row per candidate. */
export type GPUTerrainPeakSnapOutput = {
  /**
   * Positions `[column, row]` per candidate: the best pixel centre when `snapStatus` is `snapped`,
   * otherwise the original candidate position.
   */
  positions: GraphDataView<'float32x2'>;
  /**
   * DEM heights per candidate: the height of the best pixel when `snapStatus` is `snapped`,
   * otherwise the bilinear DEM height at the candidate (NaN when the candidate is outside or any of
   * its four surrounding pixels is invalid). The caller decides how to combine it with a catalogue
   * elevation, for example `max(height, catalogueHeight)`.
   */
  heights: GraphDataView<'float32'>;
  /** {@link GPU_TERRAIN_PEAK_SNAP_STATUS} code per candidate. */
  snapStatus: GraphDataView<'uint32'>;
  /** Optional move length in ground metres: positive when `snapped`, otherwise 0. */
  snapDistance?: GraphDataView<'float32'>;
  /** Optional status words. */
  status?: GPUTerrainPeakSnapStatus;
};

/**
 * Properties for {@link GPUTerrainPeakSnap}.
 *
 * Topology (needs a new graph): grid size, elevation format, `cellSizeMode`,
 * `maximumRadiusPixels`, candidate count, and which optional views exist. Per-frame:
 * `parameters`, candidate contents, and elevation contents.
 */
export type GPUTerrainPeakSnapProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-peak-snap'`. */
  id?: string;
  /** Raster grid: its width and height size the kernels. */
  grid: GPURasterMetadata;
  /** Elevation band, buffer or texture, canonicalised to float32 plus validity. */
  elevation: GPURasterBand;
  /** How the grid cell size becomes ground metres per pixel per row. Defaults to `'uniform'`. */
  cellSizeMode?: GPURasterCellSizeMode;
  /**
   * Compile-time loop bound: the disc may reach this many pixels from the candidate on each axis.
   * Integer in `[1, 64]`, default 16. A larger per-frame radius is clamped to the largest radius
   * that fits and raises `output.status.radiusClamped`.
   */
  maximumRadiusPixels?: number;
  /**
   * Candidate positions `[column, row]` in pixel-centre index space: pixel `(c, r)` has its centre
   * at `(c, r)`, like `GPUTerrainViewshed.observer`. Fractional values are allowed. At most
   * {@link getGPUTerrainPeakSnapMaximumCandidateCount} rows (6710886 on a default device).
   */
  candidates: GraphDataView<'float32x2'>;
  /**
   * Optional known candidate heights in elevation units (for example a catalogue elevation), one per
   * candidate. NaN or infinity means unknown. Used only as the reference of the height check.
   */
  candidateHeights?: GraphDataView<'float32'>;
  /**
   * Vertical datum of `candidateHeights`, for example a catalogue in a national datum. Requires
   * `candidateHeights` and is checked against `elevation.verticalDatum` when both are declared;
   * the library never converts. Geoid offsets up to about ±100 m exceed the default
   * `maximumHeightChange` of 80.
   */
  candidateVerticalDatum?: string;
  /**
   * Optional per-candidate search radius in ground metres, one per candidate. Non-finite or
   * non-positive entries fall back to the parameters radius. A distance-dependent rule such as
   * `min(250, 60 + 0.004 * distance)` metres (60 m plus 0.4 % of the distance to the viewer,
   * capped at 250 m, used for peaks drawn from coarser tiles in the mt-image skyline renderer) is
   * a caller contributor: compute it per candidate and upload it here.
   */
  candidateRadii?: GraphDataView<'float32'>;
  /** Per-frame parameters with at least 8 float32 values, see {@link getGPUTerrainPeakSnapParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /** Caller-owned outputs. */
  output: GPUTerrainPeakSnapOutput;
};

/**
 * Snaps caller-supplied candidate points (for example catalogue peaks) to the DEM summit near them.
 *
 * Generalises the mt-image `localMaxOf` and `snapPeaks` rule with a metric disc instead of a
 * latitude/longitude square: a square grid reaches `1.41 * radius` at its corners and, measured on
 * a peak catalogue, 22 % of candidates snapped to the square's outer ring, 84 % of which were still
 * climbing at twice the radius (a flank, not a summit).
 *
 * For each candidate the disc is centred on its continuous position: pixel `q` belongs when
 * `((q - candidate) * cell)` has length at most `radius`, with `cell` evaluated at the row of the
 * pixel nearest the candidate. The best pixel is the strict `(height, -index)` maximum over the
 * valid disc pixels, so the lowest row-major index wins ties. Invalid pixels are skipped and their
 * value slots never read, so nodata cannot bleed into a snap. With the interior rule on, a best
 * pixel on the disc ring (a pixel with an 8-neighbour outside the disc) means the maximum is still
 * rising beyond the disc: the candidate keeps its position (`onRing`). Unlike mt-image this rule
 * applies even when the candidate has no DEM height.
 *
 * The move is accepted when its ground length is at most `maximumMove` and, if a reference height
 * is known, `|best - reference| <= maximumHeightChange`. The reference is the candidate's height if
 * finite, otherwise the bilinear DEM height at the candidate (requires all four corners valid;
 * otherwise unknown and the height check is skipped). See {@link GPU_TERRAIN_PEAK_SNAP_STATUS} for
 * the precedence of outcomes.
 *
 * Disc membership and the best pixel are pure float32 comparisons, so indices and status are
 * deterministic. Two platform sensitivities remain: a pixel whose squared metric distance lies
 * within one ULP of `radius^2` may fall on either side of a fused multiply-add, and the bilinear
 * reference/height may differ by a few ULP between GPUs.
 */
export class GPUTerrainPeakSnap implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTerrainPeakSnapProps;

  constructor(props: GPUTerrainPeakSnapProps) {
    this.id = props.id ?? 'terrain-peak-snap';
    this.props = props;
    const {id} = this;
    if (props.candidateVerticalDatum !== undefined && !props.candidateHeights) {
      throw new Error(`${id} candidateVerticalDatum needs candidateHeights`);
    }
    assertCompatibleGPUVerticalDatums(
      id,
      props.candidateVerticalDatum,
      props.elevation.verticalDatum,
      'candidate'
    );
    const {output} = props;
    const radiusClamped = output.status?.radiusClamped;
    validateRasterGrid(id, props.grid);
    validateRasterCellSizeMode(id, props.cellSizeMode ?? 'uniform');
    validateRasterGridAxisAligned(id, props.grid, props.cellSizeMode);
    validateTerrainFeaturesMaximumRadiusPixels(id, props.maximumRadiusPixels);
    validatePackedView(props.candidates, ['float32x2'], `${id} candidates`);
    const candidateCount = props.candidates.length;
    if (candidateCount < 1) {
      throw new Error(`${id} candidates must contain at least one row`);
    }
    validateRasterParameters(id, props.parameters, GPU_TERRAIN_PEAK_SNAP_PARAMETER_LENGTH);
    for (const [name, view] of [
      ['candidateHeights', props.candidateHeights],
      ['candidateRadii', props.candidateRadii],
      ['output.heights', output.heights],
      ['output.snapDistance', output.snapDistance]
    ] as const) {
      if (!view) {
        continue;
      }
      validatePackedView(view, ['float32'], `${id} ${name}`);
      if (view.length !== candidateCount) {
        throw new Error(`${id} ${name} must contain one value per candidate`);
      }
    }
    validatePackedView(output.positions, ['float32x2'], `${id} output.positions`);
    if (output.positions.length !== candidateCount) {
      throw new Error(`${id} output.positions must contain one value per candidate`);
    }
    validatePackedView(output.snapStatus, ['uint32'], `${id} output.snapStatus`);
    if (output.snapStatus.length !== candidateCount) {
      throw new Error(`${id} output.snapStatus must contain one value per candidate`);
    }
    if (radiusClamped) {
      validatePackedView(radiusClamped, ['uint32'], `${id} output.status.radiusClamped`);
      if (radiusClamped.length < 1) {
        throw new Error(`${id} output.status.radiusClamped must contain one row`);
      }
    }
    validateRasterBuffersDistinct(
      id,
      [output.positions, output.heights, output.snapStatus, output.snapDistance, radiusClamped],
      [
        ...getRasterBandViews(props.elevation),
        props.parameters,
        props.candidates,
        props.candidateHeights,
        props.candidateRadii
      ]
    );
  }

  /** Returns canonical elevation, search, and unpack nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    const {width, height} = props.grid;
    const radiusClamped = output.status?.radiusClamped;
    validateRasterBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, [
      props.parameters,
      props.candidates,
      props.candidateHeights,
      props.candidateRadii,
      output.positions,
      output.heights,
      output.snapStatus,
      output.snapDistance,
      radiusClamped
    ]);
    const candidateCount = props.candidates.length;
    validateMaximumCount(
      id,
      'candidates rows',
      candidateCount,
      getGPUTerrainPeakSnapMaximumCandidateCount(graph.device)
    );
    const cellSizeMode = props.cellSizeMode ?? 'uniform';
    const maximumRadiusPixels = validateTerrainFeaturesMaximumRadiusPixels(
      id,
      props.maximumRadiusPixels
    );
    const source = getRasterBufferBandNodes(graph, id, props.elevation, props.grid, true);
    const nodes: GPUCommandNode<Parameters>[] = [...source.nodes];
    const validity = source.band.validity as GraphDataView<'uint32'>;
    // Per candidate: status, position column bits, position row bits, height bits, distance bits.
    const results = createTransientView(graph, `${id}-results`, 'uint32', candidateCount * 5);
    if (radiusClamped) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-overflow-clear`,
          operation: 'GPUTerrainPeakSnap',
          view: radiusClamped,
          type: 'u32',
          value: '0u',
          componentCount: 1
        })
      );
    }
    const searchBindings = [
      {
        name: 'elevationValues',
        view: source.band.storage.values as GraphDataView,
        type: 'f32' as const,
        access: 'read' as const
      },
      {name: 'elevationValidity', view: validity, type: 'u32' as const, access: 'read' as const},
      {name: 'settings', view: props.parameters, type: 'f32' as const, access: 'read' as const},
      {
        name: 'candidates',
        view: props.candidates as GraphDataView,
        type: 'f32' as const,
        access: 'read' as const
      },
      ...(props.candidateHeights
        ? [
            {
              name: 'candidateHeights',
              view: props.candidateHeights as GraphDataView,
              type: 'f32' as const,
              access: 'read' as const
            }
          ]
        : []),
      ...(props.candidateRadii
        ? [
            {
              name: 'candidateRadii',
              view: props.candidateRadii as GraphDataView,
              type: 'f32' as const,
              access: 'read' as const
            }
          ]
        : []),
      {name: 'results', view: results, type: 'u32' as const, access: 'read_write' as const},
      ...(radiusClamped
        ? [
            {
              name: 'overflowValues',
              view: radiusClamped,
              type: 'atomic<u32>' as const,
              access: 'read_write' as const
            }
          ]
        : [])
    ];
    const codes = GPU_TERRAIN_PEAK_SNAP_STATUS;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-search`,
        operation: 'GPUTerrainPeakSnap',
        variant: `disc-${cellSizeMode}`,
        bindings: searchBindings,
        // One 256-thread workgroup per candidate: a disc holds up to (2 * 64 + 1)^2 pixels.
        invocationCount: candidateCount * 256,
        workgroupSize: 256,
        guardIndex: false,
        declarations: `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const CANDIDATE_COUNT: u32 = ${candidateCount}u;
var<workgroup> reduceIndex: array<u32, 256>;
var<workgroup> reduceHeight: array<f32, 256>;
// True when (height, index) beats the incumbent: higher first, lower index on ties, any pixel beats none.
fn isBetterPixel(height: f32, pixelIndex: u32, incumbentHeight: f32, incumbentIndex: u32) -> bool {
  if (pixelIndex == NO_INDEX) { return false; }
  return incumbentIndex == NO_INDEX || height > incumbentHeight ||
    (height == incumbentHeight && pixelIndex < incumbentIndex);
}
const MAXIMUM_RADIUS_PIXELS: i32 = ${maximumRadiusPixels};
const NO_INDEX: u32 = 0xffffffffu;
const STATUS_UNCHANGED: u32 = ${codes.unchanged}u;
const STATUS_SNAPPED: u32 = ${codes.snapped}u;
const STATUS_ON_RING: u32 = ${codes.onRing}u;
const STATUS_REJECTED_MOVE: u32 = ${codes.rejectedMove}u;
const STATUS_REJECTED_HEIGHT: u32 = ${codes.rejectedHeight}u;
const STATUS_NO_DATA: u32 = ${codes.noData}u;
const STATUS_OUTSIDE: u32 = ${codes.outside}u;
${TERRAIN_FEATURES_WGSL_CONSTANTS}
${RASTER_WGSL_HELPERS}
${getRasterGroundCellSizeWGSL(cellSizeMode, TERRAIN_PEAK_SNAP_GRID_SLOTS)}
${TERRAIN_FEATURES_DISC_WGSL}
var<private> snapStatus: u32;
var<private> snapPosition: vec2<f32>;
var<private> snapHeight: f32;
var<private> snapDistanceValue: f32;
fn isValidPixel(column: u32, row: u32) -> bool {
  return elevationValidity[elevationValidityOffset + row * WIDTH + column] != 0u;
}
fn getElevation(column: u32, row: u32) -> f32 {
  return elevationValues[elevationValuesOffset + row * WIDTH + column];
}
// Bilinear DEM at an in-grid position; NaN unless all four surrounding pixels are valid.
fn sampleElevation(position: vec2<f32>, seed: u32) -> f32 {
  let base = vec2<u32>(floor(position));
  let next = min(base + vec2<u32>(1u), vec2<u32>(WIDTH - 1u, HEIGHT - 1u));
  let fraction = position - floor(position);
  if (!isValidPixel(base.x, base.y) || !isValidPixel(next.x, base.y) ||
      !isValidPixel(base.x, next.y) || !isValidPixel(next.x, next.y)) {
    return getNan(seed);
  }
  let top = mix(getElevation(base.x, base.y), getElevation(next.x, base.y), fraction.x);
  let bottom = mix(getElevation(base.x, next.y), getElevation(next.x, next.y), fraction.x);
  return mix(top, bottom, fraction.y);
}
var<private> snapNearest: vec2<i32>;
var<private> snapCell: vec2<f32>;
var<private> snapRadiusSquared: f32;
var<private> snapExtent: vec2<i32>;
var<private> snapDemHeight: f32;
// Validates the candidate and sets up the disc search. Returns false when snapStatus is final.
fn prepareSnap(index: u32, candidate: vec2<f32>, isLeader: bool) -> bool {
  snapStatus = STATUS_OUTSIDE;
  snapPosition = candidate;
  snapHeight = getNan(index);
  snapDistanceValue = 0.0;
  if (!(isFiniteValue(candidate.x) && isFiniteValue(candidate.y) &&
      candidate.x >= 0.0 && candidate.y >= 0.0 &&
      candidate.x <= f32(WIDTH - 1u) && candidate.y <= f32(HEIGHT - 1u))) {
    return false;
  }
  let demHeight = sampleElevation(candidate, index);
  snapHeight = demHeight;
  snapStatus = STATUS_NO_DATA;
  let nearest = vec2<i32>(i32(floor(candidate.x + 0.5)), i32(floor(candidate.y + 0.5)));
  let cell = getGroundCellSize(u32(nearest.y));
  var requested = settings[settingsOffset];
  ${props.candidateRadii ? 'let candidateRadius = candidateRadii[candidateRadiiOffset + index];\n  if (isFiniteValue(candidateRadius) && candidateRadius > 0.0) { requested = candidateRadius; }' : ''}
  if (!(isFiniteValue(cell.x) && isFiniteValue(cell.y) && cell.x > 0.0 && cell.y > 0.0 &&
      isFiniteValue(requested) && requested > 0.0)) {
    return false;
  }
  ${radiusClamped ? 'if (isLeader && isRadiusClamped(requested, cell)) { atomicMax(&overflowValues[overflowValuesOffset], 1u); }' : ''}
  let radius = getEffectiveRadius(requested, cell);
  let radiusSquared = radius * radius;
  let extentX = getDiscExtent(radius, cell.x);
  let extentY = getDiscExtent(radius, cell.y);
  snapNearest = nearest;
  snapCell = cell;
  snapRadiusSquared = radiusSquared;
  snapExtent = vec2<i32>(extentX, extentY);
  snapDemHeight = demHeight;
  return true;
}
// Applies the move, height, and ring policies to the disc maximum found by the cooperative scan.
fn finishSnap(index: u32, candidate: vec2<f32>, bestIndex: u32, bestHeight: f32) {
  let nearest = snapNearest;
  let cell = snapCell;
  let radiusSquared = snapRadiusSquared;
  let demHeight = snapDemHeight;
  if (bestIndex == NO_INDEX) { return; }
  let bestPixel = vec2<i32>(i32(bestIndex % WIDTH), i32(bestIndex / WIDTH));
  if (settings[settingsOffset + 7u] != 0.0 &&
      isOnDiscRing(bestPixel, candidate, cell, radiusSquared)) {
    snapStatus = STATUS_ON_RING;
    return;
  }
  if (bestPixel.x == nearest.x && bestPixel.y == nearest.y) {
    snapStatus = STATUS_UNCHANGED;
    return;
  }
  let moveX = (f32(bestPixel.x) - candidate.x) * cell.x;
  let moveY = (f32(bestPixel.y) - candidate.y) * cell.y;
  let moveSquared = moveX * moveX + moveY * moveY;
  let maximumMove = settings[settingsOffset + 1u];
  if (moveSquared > maximumMove * maximumMove) {
    snapStatus = STATUS_REJECTED_MOVE;
    return;
  }
  var reference = demHeight;
  ${props.candidateHeights ? 'let candidateHeight = candidateHeights[candidateHeightsOffset + index];\n  if (isFiniteValue(candidateHeight)) { reference = candidateHeight; }' : ''}
  if (isFiniteValue(reference) &&
      abs(bestHeight - reference) > settings[settingsOffset + 2u]) {
    snapStatus = STATUS_REJECTED_HEIGHT;
    return;
  }
  snapStatus = STATUS_SNAPPED;
  snapPosition = vec2<f32>(f32(bestPixel.x), f32(bestPixel.y));
  snapHeight = bestHeight;
  snapDistanceValue = sqrt(moveSquared);
}`,
        body: `let candidateIndex = workgroupIndex;
  let isLeader = localInvocationIndex == 0u;
  let inRange = candidateIndex < CANDIDATE_COUNT;
  var candidate = vec2<f32>(0.0, 0.0);
  if (inRange) {
    candidate = vec2<f32>(candidates[candidatesOffset + 2u * candidateIndex], candidates[candidatesOffset + 2u * candidateIndex + 1u]);
  }
  snapStatus = STATUS_OUTSIDE;
  snapPosition = candidate;
  snapHeight = getNan(candidateIndex);
  snapDistanceValue = 0.0;
  let prepared = inRange && prepareSnap(candidateIndex, candidate, isLeader);
  // The whole workgroup scans one candidate's disc; the (height, -index) maximum is order independent.
  var bestIndex = NO_INDEX;
  var bestHeight = 0.0;
  if (prepared) {
    let extent = snapExtent;
    let spanX = u32(2 * extent.x + 1);
    let scanCount = spanX * u32(2 * extent.y + 1);
    for (var scan = localInvocationIndex; scan < scanCount; scan += 256u) {
      let offsetX = i32(scan % spanX) - extent.x;
      let offsetY = i32(scan / spanX) - extent.y;
      let pixel = snapNearest + vec2<i32>(offsetX, offsetY);
      if (pixel.x < 0 || pixel.y < 0 || pixel.x >= i32(WIDTH) || pixel.y >= i32(HEIGHT)) { continue; }
      if (!isInDisc(pixel, candidate, snapCell, snapRadiusSquared)) { continue; }
      if (!isValidPixel(u32(pixel.x), u32(pixel.y))) { continue; }
      let pixelIndex = u32(pixel.y) * WIDTH + u32(pixel.x);
      let pixelHeight = elevationValues[elevationValuesOffset + pixelIndex];
      if (isBetterPixel(pixelHeight, pixelIndex, bestHeight, bestIndex)) {
        bestIndex = pixelIndex;
        bestHeight = pixelHeight;
      }
    }
  }
  reduceIndex[localInvocationIndex] = bestIndex;
  reduceHeight[localInvocationIndex] = bestHeight;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
    if (localInvocationIndex < stride) {
      let otherIndex = reduceIndex[localInvocationIndex + stride];
      let otherHeight = reduceHeight[localInvocationIndex + stride];
      if (isBetterPixel(otherHeight, otherIndex, reduceHeight[localInvocationIndex], reduceIndex[localInvocationIndex])) {
        reduceIndex[localInvocationIndex] = otherIndex;
        reduceHeight[localInvocationIndex] = otherHeight;
      }
    }
    workgroupBarrier();
  }
  if (isLeader && inRange) {
    if (prepared && reduceIndex[0] != NO_INDEX) {
      finishSnap(candidateIndex, candidate, reduceIndex[0], reduceHeight[0]);
    }
    let resultBase = resultsOffset + 5u * candidateIndex;
    results[resultBase] = snapStatus;
    results[resultBase + 1u] = bitcast<u32>(snapPosition.x);
    results[resultBase + 2u] = bitcast<u32>(snapPosition.y);
    results[resultBase + 3u] = bitcast<u32>(snapHeight);
    results[resultBase + 4u] = bitcast<u32>(snapDistanceValue);
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-unpack`,
        operation: 'GPUTerrainPeakSnap',
        variant: 'unpack',
        bindings: [
          {name: 'results', view: results, type: 'u32', access: 'read'},
          {
            name: 'positionValues',
            view: output.positions as GraphDataView,
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'heightValues',
            view: output.heights as GraphDataView,
            type: 'u32',
            access: 'read_write'
          },
          {
            name: 'statusValues',
            view: output.snapStatus as GraphDataView,
            type: 'u32',
            access: 'read_write'
          },
          ...(output.snapDistance
            ? [
                {
                  name: 'distanceValues',
                  view: output.snapDistance as GraphDataView,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: candidateCount,
        body: `let resultBase = resultsOffset + 5u * index;
  statusValues[statusValuesOffset + index] = results[resultBase];
  positionValues[positionValuesOffset + 2u * index] = bitcast<f32>(results[resultBase + 1u]);
  positionValues[positionValuesOffset + 2u * index + 1u] = bitcast<f32>(results[resultBase + 2u]);
  heightValues[heightValuesOffset + index] = results[resultBase + 3u];
  ${output.snapDistance ? 'distanceValues[distanceValuesOffset + index] = results[resultBase + 4u];' : ''}`
      })
    );
    return nodes;
  }
}
