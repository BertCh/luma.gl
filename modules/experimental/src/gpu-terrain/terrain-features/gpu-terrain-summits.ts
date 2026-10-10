// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCompaction,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPURasterBand} from '../../gpu-raster/index';
import {
  createFillNode,
  createWGSLKernelNode,
  createPublishNode
} from '../../utils/wgsl-kernel-nodes';
import type {GPUCompactOutput} from '../../utils/gpu-contributor-types';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphViewsBelongToGraph,
  validateCompactOutput
} from '../../utils/gpu-contributor-utils';
import {
  getTerrainGroundCellSizeWGSL,
  type GPUTerrainCellSizeMode,
  validateTerrainCellSizeMode
} from '../terrain-grid-utils';
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
  TERRAIN_FEATURES_DISC_WGSL,
  TERRAIN_FEATURES_WGSL_CONSTANTS,
  validateTerrainFeaturesCellSize,
  validateTerrainFeaturesMaximumRadiusPixels,
  validateTerrainFeaturesScalar
} from './terrain-features-utils';

/** Number of float32 values read from `GPUTerrainSummitsProps.settings`. */
export const GPU_TERRAIN_SUMMITS_PARAMETER_LENGTH = 8;

/**
 * CPU-side description packed by {@link getGPUTerrainSummitsParameterValues}.
 *
 * Cell-size model: `cellSize` and the edges follow `GPUTerrainSummitsProps.cellSizeMode`
 * (`uniform`: projected metres per pixel; `web-mercator`: equatorial Web Mercator metres and
 * normalized y edges; `geographic`: degrees per pixel and latitude-degree edges). `radius` is
 * always ground metres.
 */
export type GPUTerrainSummitsSettings = {
  /** Disc radius in ground metres. Finite and positive. */
  radius: number;
  /** Smallest accepted drop in elevation units. Finite and at least 0. Defaults to 0. */
  minimumDrop?: number;
  /** `[x, y]` cell size; see the cell-size model above. Finite and positive. */
  cellSize: readonly [number, number];
  /** Top edge of row 0: normalized Web Mercator y or latitude degrees. Defaults to 0. */
  northEdge?: number;
  /** Bottom edge of the last row, same units as `northEdge`. Defaults to 0. */
  southEdge?: number;
};

/**
 * Packs summit settings into
 * `[radius, minimumDrop, cellSizeX, cellSizeY, northEdge, southEdge, 0, 0]`.
 *
 * @throws If the radius or cell size is not finite and positive, the minimum drop is negative or
 * not finite, or `target` is too short.
 */
export function getGPUTerrainSummitsParameterValues(
  settings: GPUTerrainSummitsSettings,
  target: Float32Array = new Float32Array(GPU_TERRAIN_SUMMITS_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_TERRAIN_SUMMITS_PARAMETER_LENGTH) {
    throw new Error('Terrain summit settings target must hold 8 values');
  }
  validateTerrainFeaturesScalar('Terrain summit', 'radius', settings.radius, 0, true);
  validateTerrainFeaturesScalar('Terrain summit', 'minimumDrop', settings.minimumDrop ?? 0, 0);
  validateTerrainFeaturesCellSize('Terrain summit', settings.cellSize);
  target.set([
    settings.radius,
    settings.minimumDrop ?? 0,
    settings.cellSize[0],
    settings.cellSize[1],
    settings.northEdge ?? 0,
    settings.southEdge ?? 0,
    0,
    0
  ]);
  return target;
}

/**
 * Properties for {@link GPUTerrainSummits}.
 *
 * Topology (needs a new graph): grid size, elevation format, `cellSizeMode`,
 * `maximumRadiusPixels`, `incompleteNeighborhood`, and which outputs exist. Per-frame: `settings`
 * (radius, minimum drop, cell size, edges) and elevation contents. At least one of `summitMask`,
 * `drop`, or `output` is required.
 */
export type GPUTerrainSummitsProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-summits'`. */
  id?: string;
  /** Grid width in pixels. */
  width: number;
  /** Grid height in pixels. */
  height: number;
  /** Elevation band, buffer or texture, canonicalised to float32 plus validity. */
  elevation: GPURasterBand;
  /**
   * How `settings.cellSize` becomes ground metres per pixel for each row.
   * Defaults to `'uniform'`. See `GPUTerrainCellSizeMode`.
   */
  cellSizeMode?: GPUTerrainCellSizeMode;
  /**
   * Compile-time loop bound: the disc may reach this many pixels from the centre on each axis.
   * Integer in `[1, 64]`, default 16. A larger per-frame radius is clamped to the largest radius
   * that fits and raises `overflow`.
   */
  maximumRadiusPixels?: number;
  /**
   * `'reject'` (default): a pixel whose disc leaves the grid or touches an invalid pixel is not a
   * summit. `'ignore'`: outside and invalid pixels are simply absent from the disc and ring.
   * Invalid pixels never contribute a height under either policy.
   */
  incompleteNeighborhood?: 'reject' | 'ignore';
  /** Per-frame settings with at least 8 float32 values, see {@link getGPUTerrainSummitsParameterValues}. */
  settings: GraphDataView<'float32'>;
  /** Optional `uint32` per pixel: 1 for a summit, otherwise 0. */
  summitMask?: GraphDataView<'uint32'>;
  /** Optional `float32` per pixel: the drop at a summit, NaN elsewhere. */
  drop?: GraphDataView<'float32'>;
  /**
   * Optional capacity-bounded compact list of summit pixel indices (`row * width + column`),
   * ascending. `count` is clamped and `overflow` is raised when more summits exist.
   */
  output?: GPUCompactOutput;
  /**
   * Optional `float32` column aligned with `output.ids` (at least `output.ids.length` rows) holding
   * each listed summit's drop. Rows at or beyond `output.count` are NaN. Requires `output`.
   */
  outputDrop?: GraphDataView<'float32'>;
  /** Optional one-row `uint32` set to 1 when the radius was clamped to `maximumRadiusPixels`, else 0. */
  overflow?: GraphDataView<'uint32'>;
};

/**
 * Finds DEM summits: pixels that are the highest point of a metric disc and stand out from its
 * boundary ring.
 *
 * Pixel `p` is a summit when all of the following hold:
 *
 * - `p` is valid and its ground cell size is finite and positive.
 * - With `incompleteNeighborhood: 'reject'` every disc pixel is inside the grid and valid; with
 *   `'ignore'` outside and invalid pixels are absent. Nodata never bleeds in either mode: an
 *   invalid pixel's value slot is never read as a height.
 * - `p` is the strict maximum of its disc under the order `(height, -index)`: among equal heights
 *   the lowest row-major index wins, so a plateau yields exactly one summit.
 * - `drop(p) = height(p) - max(height(q))` over `q` in `Ring(p) \ {p}` is at least `minimumDrop`.
 *   If the ring is empty or entirely absent (`'ignore'`), the drop is `+Infinity`.
 *
 * The disc is the set of pixels `q` with `((dx * cx)^2 + (dy * cy)^2) <= radius^2`, evaluated at
 * the centre pixel's row (the error over a few-hundred-metre disc is negligible; in pixel space it
 * is an ellipse). The ring is the disc pixels with at least one of their eight neighbours outside
 * the disc, so every 8-connected path from `p` to outside the disc crosses it.
 *
 * Meaning of drop: any path from `p` to outside the disc crosses the ring, so it descends at least
 * `drop` along the way. `drop` is therefore a radius-limited lower bound on the topographic
 * prominence of `p`. True two-dimensional prominence and isolation need a sequential
 * Priority-Flood or union-find sweep over the whole DEM and are out of scope here.
 *
 * Heights are compared as float32 with no arithmetic except one subtraction, so indices are
 * deterministic. The only platform sensitivity is a pixel whose squared metric distance lies
 * within one ULP of `radius^2`, where a fused multiply-add may change disc membership.
 *
 * Cost: up to `(2 * maximumRadiusPixels + 1)^2` disc tests per pixel plus eight tests per
 * visited disc pixel for the ring.
 */
export class GPUTerrainSummits implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTerrainSummitsProps;

  constructor(props: GPUTerrainSummitsProps) {
    this.id = props.id ?? 'terrain-summits';
    this.props = props;
    const {id} = this;
    const pixelCount = validateTerrainGrid(id, props.width, props.height);
    validateTerrainCellSizeMode(id, props.cellSizeMode ?? 'uniform');
    validateTerrainFeaturesMaximumRadiusPixels(id, props.maximumRadiusPixels);
    if (
      props.incompleteNeighborhood !== undefined &&
      props.incompleteNeighborhood !== 'reject' &&
      props.incompleteNeighborhood !== 'ignore'
    ) {
      throw new Error(`${id} incompleteNeighborhood must be 'reject' or 'ignore'`);
    }
    if (!props.summitMask && !props.drop && !props.output) {
      throw new Error(`${id} requires at least one output`);
    }
    validateTerrainSettings(id, props.settings, GPU_TERRAIN_SUMMITS_PARAMETER_LENGTH);
    if (props.summitMask) {
      validatePackedUint32View(props.summitMask, `${id} summitMask`);
      if (props.summitMask.length !== pixelCount) {
        throw new Error(`${id} summitMask must contain one value per pixel`);
      }
    }
    if (props.drop) {
      validatePackedView(props.drop, ['float32'], `${id} drop`);
      if (props.drop.length !== pixelCount) {
        throw new Error(`${id} drop must contain one value per pixel`);
      }
    }
    if (props.output) {
      validateCompactOutput(id, props.output);
    }
    if (props.outputDrop) {
      if (!props.output) {
        throw new Error(`${id} outputDrop requires output`);
      }
      validatePackedView(props.outputDrop, ['float32'], `${id} outputDrop`);
      if (props.outputDrop.length < props.output.ids.length) {
        throw new Error(`${id} outputDrop must contain at least output.ids.length rows`);
      }
    }
    if (props.overflow) {
      validatePackedUint32View(props.overflow, `${id} overflow`);
      if (props.overflow.length < 1) {
        throw new Error(`${id} overflow must contain one uint32 row`);
      }
    }
    validateTerrainBuffersDistinct(
      id,
      [
        props.summitMask,
        props.drop,
        props.overflow,
        props.outputDrop,
        props.output?.ids,
        props.output?.count,
        props.output?.overflow,
        props.output?.requiredCount
      ],
      [...getTerrainBandViews(props.elevation), props.settings]
    );
  }

  /** Returns canonical elevation, summit, compaction, and optional drop-column nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {width, height, output} = props;
    validateTerrainBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, [
      props.settings,
      props.summitMask,
      props.drop,
      props.overflow,
      props.outputDrop,
      output?.ids,
      output?.count,
      output?.overflow,
      output?.requiredCount
    ]);
    const pixelCount = width * height;
    const cellSizeMode = props.cellSizeMode ?? 'uniform';
    const maximumRadiusPixels = validateTerrainFeaturesMaximumRadiusPixels(
      id,
      props.maximumRadiusPixels
    );
    const source = getTerrainElevationNodes(graph, id, props.elevation, width, height, true);
    const nodes: GPUCommandNode<Parameters>[] = [...source.nodes];
    const validity = source.band.validity as GraphDataView<'uint32'>;
    const mask =
      props.summitMask ??
      (output ? createTransientView(graph, `${id}-summit-mask`, 'uint32', pixelCount) : undefined);
    const drop =
      props.drop ??
      (props.outputDrop
        ? createTransientView(graph, `${id}-drop`, 'float32', pixelCount)
        : undefined);
    const bindings = [
      {
        name: 'elevationValues',
        view: source.band.storage.values as GraphDataView,
        type: 'f32' as const,
        access: 'read' as const
      },
      {name: 'elevationValidity', view: validity, type: 'u32' as const, access: 'read' as const},
      {name: 'settings', view: props.settings, type: 'f32' as const, access: 'read' as const},
      ...(mask
        ? [{name: 'maskValues', view: mask, type: 'u32' as const, access: 'read_write' as const}]
        : []),
      ...(drop
        ? [{name: 'dropValues', view: drop, type: 'f32' as const, access: 'read_write' as const}]
        : []),
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
    if (props.overflow) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-overflow-clear`,
          operation: 'GPUTerrainSummits',
          view: props.overflow,
          type: 'u32',
          value: '0u',
          componentCount: 1
        })
      );
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-summits`,
        operation: 'GPUTerrainSummits',
        variant: `disc-${cellSizeMode}-${props.incompleteNeighborhood ?? 'reject'}`,
        bindings,
        invocationCount: pixelCount,
        declarations: `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const MAXIMUM_RADIUS_PIXELS: i32 = ${maximumRadiusPixels};
const REJECT_INCOMPLETE: bool = ${(props.incompleteNeighborhood ?? 'reject') === 'reject'};
const NO_INDEX: u32 = 0xffffffffu;
${TERRAIN_FEATURES_WGSL_CONSTANTS}
${TERRAIN_WGSL_HELPERS}
${getTerrainGroundCellSizeWGSL(cellSizeMode, {cellSizeIndex: 2, northEdgeIndex: 4})}
${TERRAIN_FEATURES_DISC_WGSL}
var<private> summitDrop: f32;
// Returns whether the pixel is a summit and leaves its drop in summitDrop.
fn evaluateSummit(column: i32, row: i32, index: u32) -> bool {
  let cell = getGroundCellSize(u32(row));
  if (!(isFiniteValue(cell.x) && isFiniteValue(cell.y) && cell.x > 0.0 && cell.y > 0.0)) {
    return false;
  }
  let requested = settings[settingsOffset];
  let minimumDrop = settings[settingsOffset + 1u];
  if (!(isFiniteValue(requested) && requested > 0.0)) { return false; }
  ${props.overflow ? 'if (isRadiusClamped(requested, cell)) { atomicMax(&overflowValues[overflowValuesOffset], 1u); }' : ''}
  if (elevationValidity[elevationValidityOffset + index] == 0u) { return false; }
  let radius = getEffectiveRadius(requested, cell);
  let radiusSquared = radius * radius;
  let centre = vec2<f32>(f32(column), f32(row));
  let extentX = getDiscExtent(radius, cell.x);
  let extentY = getDiscExtent(radius, cell.y);
  let ownHeight = elevationValues[elevationValuesOffset + index];
  // Cheap rejection first: almost every pixel has a higher disc pixel among its 8 neighbors, so the
  // wide scan below only runs for 3x3 local maxima. Every rejection here would also be reached by
  // the full scan (higher pixel, or an incomplete neighborhood under 'reject'), so results match.
  for (var nearY = -1; nearY <= 1; nearY++) {
    for (var nearX = -1; nearX <= 1; nearX++) {
      let nearPixel = vec2<i32>(column + nearX, row + nearY);
      if ((nearX == 0 && nearY == 0) || !isInDisc(nearPixel, centre, cell, radiusSquared)) { continue; }
      if (nearPixel.x < 0 || nearPixel.y < 0 || nearPixel.x >= i32(WIDTH) || nearPixel.y >= i32(HEIGHT)) { continue; }
      let nearIndex = u32(nearPixel.y) * WIDTH + u32(nearPixel.x);
      if (elevationValidity[elevationValidityOffset + nearIndex] == 0u) { continue; }
      let nearHeight = elevationValues[elevationValuesOffset + nearIndex];
      if (nearHeight > ownHeight || (nearHeight == ownHeight && nearIndex < index)) { return false; }
    }
  }
  var ringHeight = 0.0;
  var hasRing = false;
  for (var offsetY = -extentY; offsetY <= extentY; offsetY++) {
    for (var offsetX = -extentX; offsetX <= extentX; offsetX++) {
      let pixel = vec2<i32>(column + offsetX, row + offsetY);
      if (!isInDisc(pixel, centre, cell, radiusSquared)) { continue; }
      if (pixel.x < 0 || pixel.y < 0 || pixel.x >= i32(WIDTH) || pixel.y >= i32(HEIGHT)) {
        if (REJECT_INCOMPLETE) { return false; }
        continue;
      }
      let neighborIndex = u32(pixel.y) * WIDTH + u32(pixel.x);
      if (elevationValidity[elevationValidityOffset + neighborIndex] == 0u) {
        if (REJECT_INCOMPLETE) { return false; }
        continue;
      }
      let neighborHeight = elevationValues[elevationValuesOffset + neighborIndex];
      // Another disc pixel beats this one under (height, -index): not the maximum.
      if (neighborHeight > ownHeight || (neighborHeight == ownHeight && neighborIndex < index)) {
        return false;
      }
      if ((offsetX != 0 || offsetY != 0) && isOnDiscRing(pixel, centre, cell, radiusSquared)) {
        if (!hasRing || neighborHeight > ringHeight) {
          ringHeight = neighborHeight;
          hasRing = true;
        }
      }
    }
  }
  if (hasRing) {
    let dropValue = ownHeight - ringHeight;
    if (dropValue < minimumDrop) { return false; }
    summitDrop = dropValue;
  } else {
    summitDrop = getInfinity(index);
  }
  return true;
}`,
        body: `let column = i32(index % WIDTH);
  let row = i32(index / WIDTH);
  summitDrop = getNan(index);
  let isSummit = evaluateSummit(column, row, index);
  ${mask ? 'maskValues[maskValuesOffset + index] = select(0u, 1u, isSummit);' : ''}
  ${drop ? 'dropValues[dropValuesOffset + index] = select(getNan(index), summitDrop, isSummit);' : ''}`
      })
    );
    if (output && mask) {
      const pixelIds = createTransientView(graph, `${id}-pixel-ids`, 'uint32', pixelCount);
      const compactRows = createTransientView(graph, `${id}-compact-rows`, 'uint32', pixelCount);
      const requiredCount = createTransientView(graph, `${id}-summit-total`, 'uint32', 1);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-pixel-ids`,
          operation: 'GPUTerrainSummits',
          variant: 'pixel-ids',
          bindings: [{name: 'pixelIds', view: pixelIds, type: 'u32', access: 'read_write'}],
          invocationCount: pixelCount,
          body: 'pixelIds[pixelIdsOffset + index] = index;'
        }),
        ...new GPUCompaction({
          id: `${id}-compaction`,
          input: pixelIds,
          flags: mask,
          output: compactRows,
          count: requiredCount
        }).getCommandNodes(graph),
        createPublishNode<Parameters>(graph, {
          id: `${id}-publish`,
          operation: 'GPUTerrainSummits',
          requiredCount,
          compactIds: compactRows,
          output
        })
      );
      if (props.outputDrop && drop) {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-output-drop`,
            operation: 'GPUTerrainSummits',
            variant: 'gather-drop',
            bindings: [
              {name: 'idsIn', view: output.ids, type: 'u32', access: 'read'},
              {name: 'countIn', view: output.count, type: 'u32', access: 'read'},
              {name: 'dropIn', view: drop, type: 'f32', access: 'read'},
              {name: 'dropOut', view: props.outputDrop, type: 'f32', access: 'read_write'}
            ],
            invocationCount: output.ids.length,
            body: `if (index < countIn[countInOffset]) {
    dropOut[dropOutOffset + index] = dropIn[dropInOffset + idsIn[idsInOffset + index]];
  } else {
    dropOut[dropOutOffset + index] = bitcast<f32>(0x7fc00000u | (index & 0u));
  }`
          })
        );
      }
    }
    return nodes;
  }
}
