// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPURasterBand} from '../../gpu-raster/index';
import {createFillNode, createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import type {GPUTerrainCellSizeMode} from '../terrain-analysis/gpu-terrain-derivatives';
import {
  getTerrainBandViews,
  getTerrainElevationNodes,
  TERRAIN_WGSL_HELPERS,
  validateTerrainBandBelongsToGraph,
  validateTerrainBuffersDistinct,
  validateTerrainGrid,
  validateTerrainSettings
} from '../terrain-analysis/terrain-analysis-utils';
import {
  getTerrainFeaturesCellSizeWGSL,
  TERRAIN_FEATURES_DISC_WGSL,
  TERRAIN_FEATURES_WGSL_CONSTANTS,
  validateTerrainFeaturesCellSize,
  validateTerrainFeaturesCellSizeMode,
  validateTerrainFeaturesMaximumRadiusPixels,
  validateTerrainFeaturesScalar
} from './terrain-features-utils';

/**
 * Status codes written to `GPUTerrainPeakSnapProps.status`, in precedence order:
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

/** Number of float32 values read from `GPUTerrainPeakSnapProps.settings`. */
export const GPU_TERRAIN_PEAK_SNAP_PARAMETER_LENGTH = 8;

/**
 * CPU-side description packed by {@link getGPUTerrainPeakSnapParameterValues}.
 *
 * Cell-size model: `cellSize` and the edges follow `GPUTerrainPeakSnapProps.cellSizeMode`
 * (`uniform`: projected metres per pixel; `web-mercator`: equatorial Web Mercator metres and
 * normalized y edges; `geographic`: degrees per pixel and latitude-degree edges). `radius`,
 * `maximumMove`, and `maximumHeightChange` are ground metres or elevation units.
 */
export type GPUTerrainPeakSnapSettings = {
  /** Default disc radius in ground metres, used for candidates without a radius. Defaults to 150. */
  radius?: number;
  /** Largest accepted move in ground metres. Defaults to 300. */
  maximumMove?: number;
  /** Largest accepted `|best - reference|` in elevation units. Defaults to 80. */
  maximumHeightChange?: number;
  /** `[x, y]` cell size; see the cell-size model above. Finite and positive. */
  cellSize: readonly [number, number];
  /** Keep the candidate when the best pixel lies on the disc ring (a flank). Defaults to true. */
  interior?: boolean;
  /** Top edge of row 0: normalized Web Mercator y or latitude degrees. Defaults to 0. */
  northEdge?: number;
  /** Bottom edge of the last row, same units as `northEdge`. Defaults to 0. */
  southEdge?: number;
};

/**
 * Packs peak snap settings into
 * `[radius, maximumMove, maximumHeightChange, cellSizeX, cellSizeY, northEdge, southEdge, interior]`.
 *
 * @throws If a radius, move, or height limit is not finite and positive (`maximumHeightChange`
 * may be 0), the cell size is not finite and positive, or `target` is too short.
 */
export function getGPUTerrainPeakSnapParameterValues(
  settings: GPUTerrainPeakSnapSettings,
  target: Float32Array = new Float32Array(GPU_TERRAIN_PEAK_SNAP_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_TERRAIN_PEAK_SNAP_PARAMETER_LENGTH) {
    throw new Error('Terrain peak snap settings target must hold 8 values');
  }
  const radius = settings.radius ?? 150;
  const maximumMove = settings.maximumMove ?? 300;
  const maximumHeightChange = settings.maximumHeightChange ?? 80;
  validateTerrainFeaturesScalar('Terrain peak snap', 'radius', radius, 0, true);
  validateTerrainFeaturesScalar('Terrain peak snap', 'maximumMove', maximumMove, 0);
  validateTerrainFeaturesScalar('Terrain peak snap', 'maximumHeightChange', maximumHeightChange, 0);
  validateTerrainFeaturesCellSize('Terrain peak snap', settings.cellSize);
  target.set([
    radius,
    maximumMove,
    maximumHeightChange,
    settings.cellSize[0],
    settings.cellSize[1],
    settings.northEdge ?? 0,
    settings.southEdge ?? 0,
    settings.interior === false ? 0 : 1
  ]);
  return target;
}

/**
 * Properties for {@link GPUTerrainPeakSnap}.
 *
 * Topology (needs a new graph): grid size, elevation format, `cellSizeMode`,
 * `maximumRadiusPixels`, candidate count, and which optional views exist. Per-frame: `settings`,
 * candidate contents, and elevation contents.
 */
export type GPUTerrainPeakSnapProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-peak-snap'`. */
  id?: string;
  /** Grid width in pixels. */
  width: number;
  /** Grid height in pixels. */
  height: number;
  /** Elevation band, buffer or texture, canonicalised to float32 plus validity. */
  elevation: GPURasterBand;
  /** How `settings.cellSize` becomes ground metres per pixel per row. Defaults to `'uniform'`. */
  cellSizeMode?: GPUTerrainCellSizeMode;
  /**
   * Compile-time loop bound: the disc may reach this many pixels from the candidate on each axis.
   * Integer in `[1, 64]`, default 16. A larger per-frame radius is clamped to the largest radius
   * that fits and raises `overflow`.
   */
  maximumRadiusPixels?: number;
  /**
   * Candidate positions `[column, row]` in pixel-centre index space: pixel `(c, r)` has its centre
   * at `(c, r)`, like `GPUTerrainViewshed.observer`. Fractional values are allowed.
   */
  candidates: GraphDataView<'float32x2'>;
  /**
   * Optional known candidate heights in elevation units (for example a catalogue elevation), one per
   * candidate. NaN or infinity means unknown. Used only as the reference of the height check.
   */
  candidateHeights?: GraphDataView<'float32'>;
  /**
   * Optional per-candidate search radius in ground metres, one per candidate. Non-finite or
   * non-positive entries fall back to the settings radius. A distance-dependent rule such as
   * `min(250, 60 + 0.004 * distance)` metres (60 m plus 0.4 % of the distance to the viewer,
   * capped at 250 m, used for peaks drawn from coarser tiles in the mt-image skyline renderer) is
   * a caller recipe: compute it per candidate and upload it here.
   */
  candidateRadii?: GraphDataView<'float32'>;
  /** Per-frame settings with at least 8 float32 values, see {@link getGPUTerrainPeakSnapParameterValues}. */
  settings: GraphDataView<'float32'>;
  /**
   * Output positions `[column, row]` per candidate: the best pixel centre when status is `snapped`,
   * otherwise the original candidate position.
   */
  positions: GraphDataView<'float32x2'>;
  /**
   * Output DEM heights per candidate: the height of the best pixel when status is `snapped`,
   * otherwise the bilinear DEM height at the candidate (NaN when the candidate is outside or any of
   * its four surrounding pixels is invalid). The caller decides how to combine it with a catalogue
   * elevation, for example `max(height, catalogueHeight)`.
   */
  heights: GraphDataView<'float32'>;
  /** Output {@link GPU_TERRAIN_PEAK_SNAP_STATUS} code per candidate. */
  status: GraphDataView<'uint32'>;
  /** Optional output move length in ground metres: positive when `snapped`, otherwise 0. */
  snapDistance?: GraphDataView<'float32'>;
  /** Optional one-row `uint32` set to 1 when any candidate's radius was clamped, else 0. */
  overflow?: GraphDataView<'uint32'>;
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
    validateTerrainGrid(id, props.width, props.height);
    validateTerrainFeaturesCellSizeMode(id, props.cellSizeMode ?? 'uniform');
    validateTerrainFeaturesMaximumRadiusPixels(id, props.maximumRadiusPixels);
    validatePackedView(props.candidates, ['float32x2'], `${id} candidates`);
    const candidateCount = props.candidates.length;
    if (candidateCount < 1) {
      throw new Error(`${id} candidates must contain at least one row`);
    }
    validateTerrainSettings(id, props.settings, GPU_TERRAIN_PEAK_SNAP_PARAMETER_LENGTH);
    for (const [name, view] of [
      ['candidateHeights', props.candidateHeights],
      ['candidateRadii', props.candidateRadii],
      ['heights', props.heights],
      ['snapDistance', props.snapDistance]
    ] as const) {
      if (!view) {
        continue;
      }
      validatePackedView(view, ['float32'], `${id} ${name}`);
      if (view.length !== candidateCount) {
        throw new Error(`${id} ${name} must contain one value per candidate`);
      }
    }
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (props.positions.length !== candidateCount) {
      throw new Error(`${id} positions must contain one value per candidate`);
    }
    validatePackedUint32View(props.status, `${id} status`);
    if (props.status.length !== candidateCount) {
      throw new Error(`${id} status must contain one value per candidate`);
    }
    if (props.overflow) {
      validatePackedUint32View(props.overflow, `${id} overflow`);
      if (props.overflow.length < 1) {
        throw new Error(`${id} overflow must contain one uint32 row`);
      }
    }
    validateTerrainBuffersDistinct(
      id,
      [props.positions, props.heights, props.status, props.snapDistance, props.overflow],
      [
        ...getTerrainBandViews(props.elevation),
        props.settings,
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
    const {width, height} = props;
    validateTerrainBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, [
      props.settings,
      props.candidates,
      props.candidateHeights,
      props.candidateRadii,
      props.positions,
      props.heights,
      props.status,
      props.snapDistance,
      props.overflow
    ]);
    const candidateCount = props.candidates.length;
    const cellSizeMode = props.cellSizeMode ?? 'uniform';
    const maximumRadiusPixels = validateTerrainFeaturesMaximumRadiusPixels(
      id,
      props.maximumRadiusPixels
    );
    const source = getTerrainElevationNodes(graph, id, props.elevation, width, height, true);
    const nodes: GPUCommandNode<Parameters>[] = [...source.nodes];
    const validity = source.band.validity as GraphDataView<'uint32'>;
    // Per candidate: status, position column bits, position row bits, height bits, distance bits.
    const results = createTransientView(graph, `${id}-results`, 'uint32', candidateCount * 5);
    if (props.overflow) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-overflow-clear`,
          operation: 'GPUTerrainPeakSnap',
          view: props.overflow,
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
      {name: 'settings', view: props.settings, type: 'f32' as const, access: 'read' as const},
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
      ...(props.overflow
        ? [
            {
              name: 'overflowValues',
              view: props.overflow,
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
        invocationCount: candidateCount,
        declarations: `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
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
${TERRAIN_WGSL_HELPERS}
${getTerrainFeaturesCellSizeWGSL(cellSizeMode, {cellSizeIndex: 3, northEdgeIndex: 5, southEdgeIndex: 6})}
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
fn snapCandidate(index: u32, candidate: vec2<f32>) {
  snapStatus = STATUS_OUTSIDE;
  snapPosition = candidate;
  snapHeight = getNan(index);
  snapDistanceValue = 0.0;
  if (!(isFiniteValue(candidate.x) && isFiniteValue(candidate.y) &&
      candidate.x >= 0.0 && candidate.y >= 0.0 &&
      candidate.x <= f32(WIDTH - 1u) && candidate.y <= f32(HEIGHT - 1u))) {
    return;
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
    return;
  }
  ${props.overflow ? 'if (isRadiusClamped(requested, cell)) { atomicMax(&overflowValues[overflowValuesOffset], 1u); }' : ''}
  let radius = getEffectiveRadius(requested, cell);
  let radiusSquared = radius * radius;
  let extentX = getDiscExtent(radius, cell.x);
  let extentY = getDiscExtent(radius, cell.y);
  var bestIndex = NO_INDEX;
  var bestHeight = 0.0;
  for (var offsetY = -extentY; offsetY <= extentY; offsetY++) {
    for (var offsetX = -extentX; offsetX <= extentX; offsetX++) {
      let pixel = nearest + vec2<i32>(offsetX, offsetY);
      if (pixel.x < 0 || pixel.y < 0 || pixel.x >= i32(WIDTH) || pixel.y >= i32(HEIGHT)) { continue; }
      if (!isInDisc(pixel, candidate, cell, radiusSquared)) { continue; }
      if (!isValidPixel(u32(pixel.x), u32(pixel.y))) { continue; }
      let pixelIndex = u32(pixel.y) * WIDTH + u32(pixel.x);
      let pixelHeight = elevationValues[elevationValuesOffset + pixelIndex];
      if (bestIndex == NO_INDEX || pixelHeight > bestHeight ||
          (pixelHeight == bestHeight && pixelIndex < bestIndex)) {
        bestIndex = pixelIndex;
        bestHeight = pixelHeight;
      }
    }
  }
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
        body: `let candidate = vec2<f32>(candidates[candidatesOffset + 2u * index], candidates[candidatesOffset + 2u * index + 1u]);
  snapCandidate(index, candidate);
  let resultBase = resultsOffset + 5u * index;
  results[resultBase] = snapStatus;
  results[resultBase + 1u] = bitcast<u32>(snapPosition.x);
  results[resultBase + 2u] = bitcast<u32>(snapPosition.y);
  results[resultBase + 3u] = bitcast<u32>(snapHeight);
  results[resultBase + 4u] = bitcast<u32>(snapDistanceValue);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-unpack`,
        operation: 'GPUTerrainPeakSnap',
        variant: 'unpack',
        bindings: [
          {name: 'results', view: results, type: 'u32', access: 'read'},
          {
            name: 'positionValues',
            view: props.positions as GraphDataView,
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'heightValues',
            view: props.heights as GraphDataView,
            type: 'u32',
            access: 'read_write'
          },
          {
            name: 'statusValues',
            view: props.status as GraphDataView,
            type: 'u32',
            access: 'read_write'
          },
          ...(props.snapDistance
            ? [
                {
                  name: 'distanceValues',
                  view: props.snapDistance as GraphDataView,
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
  ${props.snapDistance ? 'distanceValues[distanceValuesOffset + index] = results[resultBase + 4u];' : ''}`
      })
    );
    return nodes;
  }
}
