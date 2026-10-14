// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCompaction,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPURasterBand, GPURasterMetadata} from '../../gpu-raster/index';
import {
  getGPURasterExtremaPyramidLayout,
  getRasterExtremaPyramidWGSL,
  createRasterExtremaPyramidNodes,
  type GPURasterExtremaPyramidLayout,
  type GPURasterExtremaPyramidOutput
} from '../../gpu-raster/raster-pyramid/gpu-raster-extrema-pyramid';
import {assert} from '@luma.gl/core';
import {
  createFillNode,
  createWGSLKernelNode,
  createPublishNode
} from '../../utils/wgsl-kernel-nodes';
import type {GPUCompactOutput} from '../../utils/gpu-contributor-types';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphViewsBelongToGraph,
  validateCompactOutput,
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
  TERRAIN_FEATURES_DISC_ROW_WGSL,
  TERRAIN_FEATURES_DISC_WGSL,
  TERRAIN_FEATURES_WGSL_CONSTANTS,
  validateTerrainFeaturesCellSize,
  validateTerrainFeaturesMaximumRadiusPixels,
  validateTerrainFeaturesScalar
} from './terrain-features-utils';

/** Number of float32 values read from `GPUTerrainSummitsProps.parameters`. */
export const GPU_TERRAIN_SUMMITS_PARAMETER_LENGTH = 8;

/** Cell size and row-edge slots of the summit parameter layout. */
const TERRAIN_SUMMITS_GRID_SLOTS = {cellSizeIndex: 2, northEdgeIndex: 4} as const;

/**
 * CPU-side description packed by {@link getGPUTerrainSummitsParameterValues}. The cell size and
 * latitude band come from the grid passed next to it and follow
 * `GPUTerrainSummitsProps.cellSizeMode`. `radius` is always ground metres.
 */
export type GPUTerrainSummitsSettings = {
  /** Disc radius in ground metres. Finite and positive. */
  radius: number;
  /** Smallest accepted drop in elevation units. Finite and at least 0. Defaults to 0. */
  minimumDrop?: number;
};

/**
 * Packs summit settings into
 * `[radius, minimumDrop, cellSizeX, cellSizeY, northEdge, southEdge, 0, 0]`, with the cell size
 * and the row-edge world y taken from `grid`.
 *
 * @throws If the radius or the grid cell size is not finite and positive, the minimum drop is
 * negative or not finite, or `target` is too short.
 */
export function getGPUTerrainSummitsParameterValues(
  settings: GPUTerrainSummitsSettings,
  grid: GPURasterMetadata,
  target: Float32Array = new Float32Array(GPU_TERRAIN_SUMMITS_PARAMETER_LENGTH)
): Float32Array {
  const name = 'getGPUTerrainSummitsParameterValues';
  validateParameterTarget(name, target, GPU_TERRAIN_SUMMITS_PARAMETER_LENGTH);
  validateTerrainFeaturesScalar(name, 'radius', settings.radius, 0, true);
  validateTerrainFeaturesScalar(name, 'minimumDrop', settings.minimumDrop ?? 0, 0);
  validateTerrainFeaturesCellSize(name, getRasterGridGeoreference(grid).cellSize);
  target[0] = settings.radius;
  target[1] = settings.minimumDrop ?? 0;
  target[6] = 0;
  target[7] = 0;
  writeRasterGridParameterValues(target, grid, TERRAIN_SUMMITS_GRID_SLOTS);
  return target;
}

/** Status words of {@link GPUTerrainSummits}. */
export type GPUTerrainSummitsStatus = {
  /** One-row `uint32` set to 1 when the radius was clamped to `maximumRadiusPixels`, else 0. */
  radiusClamped?: GraphDataView<'uint32'>;
};

/** Caller-owned outputs of {@link GPUTerrainSummits}; at least one of the first three is required. */
export type GPUTerrainSummitsOutput = {
  /** `uint32` per pixel: 1 for a summit, otherwise 0. */
  summitMask?: GraphDataView<'uint32'>;
  /** `float32` per pixel: the drop at a summit, NaN elsewhere. */
  drop?: GraphDataView<'float32'>;
  /**
   * Capacity-bounded compact list of summit pixel indices (`row * width + column`), ascending.
   * `count` is clamped and `overflow` is raised when more summits exist.
   */
  summits?: GPUCompactOutput;
  /**
   * `float32` column aligned with `summits.ids` (at least `summits.ids.length` rows) holding each
   * listed summit's drop. Rows at or beyond `summits.count` are NaN. Requires `summits`.
   */
  summitDrop?: GraphDataView<'float32'>;
  /** Optional status words. */
  status?: GPUTerrainSummitsStatus;
};

/**
 * Properties for {@link GPUTerrainSummits}.
 *
 * Topology (needs a new graph): grid size, elevation format, `cellSizeMode`,
 * `maximumRadiusPixels`, `incompleteNeighborhood`, and which outputs exist. Per-frame:
 * `parameters` (radius, minimum drop, grid cell size and edges) and elevation contents. At least
 * one of `output.summitMask`, `output.drop`, or `output.summits` is required.
 */
export type GPUTerrainSummitsProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-summits'`. */
  id?: string;
  /** Raster grid: its width and height size the kernels. */
  grid: GPURasterMetadata;
  /** Elevation band, buffer or texture, canonicalised to float32 plus validity. */
  elevation: GPURasterBand;
  /**
   * How the grid cell size becomes ground metres per pixel for each row.
   * Defaults to `'uniform'`. See `GPURasterCellSizeMode`.
   */
  cellSizeMode?: GPURasterCellSizeMode;
  /**
   * Compile-time loop bound: the disc may reach this many pixels from the centre on each axis.
   * Integer in `[1, 64]`, default 16. A larger per-frame radius is clamped to the largest radius
   * that fits and raises `output.status.radiusClamped`.
   */
  maximumRadiusPixels?: number;
  /**
   * `'reject'` (default): a pixel whose disc leaves the grid or touches an invalid pixel is not a
   * summit. `'ignore'`: outside and invalid pixels are simply absent from the disc and ring.
   * Invalid pixels never contribute a height under either policy.
   */
  incompleteNeighborhood?: 'reject' | 'ignore';
  /** Per-frame parameters with at least 8 float32 values, see {@link getGPUTerrainSummitsParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /**
   * Optional prebuilt maximum pyramid to read instead of building one. Pass
   * `GPURasterExtremaPyramid.output` of a pyramid created with `combined`, `footprint: 'cell'`
   * and the default level count (the top level is one cell), over exactly the same elevation
   * values and validity as `elevation`. Requires `incompleteNeighborhood: 'ignore'` (a `'reject'`
   * search needs invalid pixels to count as above every height, which a plain pyramid does not
   * encode). Results are bit-identical to the self-built pyramid.
   */
  pyramid?: GPURasterExtremaPyramidOutput;
  /** Caller-owned outputs. */
  output: GPUTerrainSummitsOutput;
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
 * Cost: the maximum test descends a square min/max pyramid (`footprint: 'cell'`, first block 2)
 * over the rows above and below `p`, reading a block's maximum when the block lies wholly inside
 * the disc, so it visits O(R) blocks instead of the O(R^2) disc pixels (R = disc radius in
 * pixels). The row of `p` and the ring (O(R) pixels, found per row from the disc row half-widths)
 * are scanned directly. Almost every pixel is rejected by a 3 x 3 pre-test before any of this.
 * The result is bit-identical to scanning every disc pixel: the disc predicate is monotone in
 * `|dx|` and `|dy|`, which makes block containment and disjointness exact (see
 * `getTerrainSummitsKernelWGSL`).
 */
export class GPUTerrainSummits implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTerrainSummitsProps;
  /** Layout of the maximum pyramid the search reads (`'cell'` footprint). */
  readonly pyramidLayout: GPURasterExtremaPyramidLayout;

  constructor(props: GPUTerrainSummitsProps) {
    this.id = props.id ?? 'terrain-summits';
    this.props = props;
    const {id} = this;
    const {output} = props;
    const radiusClamped = output.status?.radiusClamped;
    const pixelCount = validateRasterGrid(id, props.grid);
    validateRasterCellSizeMode(id, props.cellSizeMode ?? 'uniform');
    validateRasterGridAxisAligned(id, props.grid, props.cellSizeMode);
    validateTerrainFeaturesMaximumRadiusPixels(id, props.maximumRadiusPixels);
    if (
      props.incompleteNeighborhood !== undefined &&
      props.incompleteNeighborhood !== 'reject' &&
      props.incompleteNeighborhood !== 'ignore'
    ) {
      throw new Error(`${id} incompleteNeighborhood must be 'reject' or 'ignore'`);
    }
    if (!output.summitMask && !output.drop && !output.summits) {
      throw new Error(`${id} requires at least one output`);
    }
    validateRasterParameters(id, props.parameters, GPU_TERRAIN_SUMMITS_PARAMETER_LENGTH);
    if (props.pyramid) {
      if (props.incompleteNeighborhood === undefined || props.incompleteNeighborhood === 'reject') {
        throw new Error(`${id} pyramid requires incompleteNeighborhood 'ignore'`);
      }
      this.pyramidLayout = validateSummitsPyramid(id, props.pyramid, props.grid);
    } else {
      this.pyramidLayout = getGPURasterExtremaPyramidLayout(props.grid, {
        firstBlockSize: SUMMITS_FIRST_BLOCK_SIZE,
        footprint: 'cell'
      });
    }
    if (output.summitMask) {
      validatePackedView(output.summitMask, ['uint32'], `${id} output.summitMask`);
      if (output.summitMask.length !== pixelCount) {
        throw new Error(`${id} output.summitMask must contain one value per pixel`);
      }
    }
    if (output.drop) {
      validatePackedView(output.drop, ['float32'], `${id} output.drop`);
      if (output.drop.length !== pixelCount) {
        throw new Error(`${id} output.drop must contain one value per pixel`);
      }
    }
    if (output.summits) {
      validateCompactOutput(id, output.summits);
    }
    if (output.summitDrop) {
      if (!output.summits) {
        throw new Error(`${id} output.summitDrop requires output.summits`);
      }
      validatePackedView(output.summitDrop, ['float32'], `${id} output.summitDrop`);
      if (output.summitDrop.length < output.summits.ids.length) {
        throw new Error(`${id} output.summitDrop must contain at least summits.ids.length rows`);
      }
    }
    if (radiusClamped) {
      validatePackedView(radiusClamped, ['uint32'], `${id} output.status.radiusClamped`);
      if (radiusClamped.length < 1) {
        throw new Error(`${id} output.status.radiusClamped must contain one row`);
      }
    }
    validateRasterBuffersDistinct(
      id,
      [
        output.summitMask,
        output.drop,
        radiusClamped,
        output.summitDrop,
        output.summits?.ids,
        output.summits?.status.count,
        output.summits?.status.overflow,
        output.summits?.status.requiredCount
      ],
      [
        ...getRasterBandViews(props.elevation),
        props.parameters,
        ...(props.pyramid ? [props.pyramid.combined] : [])
      ]
    );
  }

  /** Returns canonical elevation, summit, compaction, and optional drop-column nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {width, height} = props.grid;
    const summits = props.output.summits;
    const radiusClamped = props.output.status?.radiusClamped;
    validateRasterBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, [
      props.parameters,
      props.pyramid?.combined,
      props.output.summitMask,
      props.output.drop,
      radiusClamped,
      props.output.summitDrop,
      summits?.ids,
      summits?.status.count,
      summits?.status.overflow,
      summits?.status.requiredCount
    ]);
    const pixelCount = width * height;
    const cellSizeMode = props.cellSizeMode ?? 'uniform';
    const maximumRadiusPixels = validateTerrainFeaturesMaximumRadiusPixels(
      id,
      props.maximumRadiusPixels
    );
    const source = getRasterBufferBandNodes(graph, id, props.elevation, props.grid, true);
    const nodes: GPUCommandNode<Parameters>[] = [...source.nodes];
    const validity = source.band.validity as GraphDataView<'uint32'>;
    const mask =
      props.output.summitMask ??
      (summits ? createTransientView(graph, `${id}-summit-mask`, 'uint32', pixelCount) : undefined);
    const drop =
      props.output.drop ??
      (props.output.summitDrop
        ? createTransientView(graph, `${id}-drop`, 'float32', pixelCount)
        : undefined);
    const {pyramidLayout} = this;
    const rejectIncomplete = (props.incompleteNeighborhood ?? 'reject') === 'reject';
    let pyramidView: GraphDataView<'float32'>;
    if (props.pyramid) {
      pyramidView = props.pyramid.combined;
    } else {
      pyramidView = createTransientView(
        graph,
        `${id}-pyramid-maximum`,
        'float32',
        pyramidLayout.length
      );
      let pyramidValues = source.band.storage.values as GraphDataView<'float32'>;
      let pyramidValidity = validity;
      if (rejectIncomplete) {
        // Invalid pixels become the largest finite height, valid for the pyramid, so that any
        // block holding one has a maximum no height can beat (the kernel handles the exact
        // +/-FLT_MAX heights without reading block maxima).
        pyramidValues = createTransientView(graph, `${id}-pyramid-values`, 'float32', pixelCount);
        pyramidValidity = createTransientView(
          graph,
          `${id}-pyramid-validity`,
          'uint32',
          pixelCount
        );
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-pyramid-input`,
            operation: 'GPUTerrainSummits',
            variant: 'pyramid-input',
            bindings: [
              {
                name: 'sourceValues',
                view: source.band.storage.values as GraphDataView,
                type: 'f32',
                access: 'read'
              },
              {name: 'sourceValidity', view: validity, type: 'u32', access: 'read'},
              {name: 'blockValues', view: pyramidValues, type: 'f32', access: 'read_write'},
              {name: 'blockValidity', view: pyramidValidity, type: 'u32', access: 'read_write'}
            ],
            invocationCount: pixelCount,
            body: `let isValid = sourceValidity[sourceValidityOffset + index] != 0u;
  blockValues[blockValuesOffset + index] = select(${FLOAT_MAXIMUM_LITERAL}, sourceValues[sourceValuesOffset + index], isValid);
  blockValidity[blockValidityOffset + index] = 1u;`
          })
        );
      }
      nodes.push(
        ...createRasterExtremaPyramidNodes(graph, {
          id: `${id}-pyramid`,
          layout: pyramidLayout,
          values: pyramidValues,
          validity: pyramidValidity,
          maximum: pyramidView
        })
      );
    }
    const bindings = [
      {
        name: 'elevationValues',
        view: source.band.storage.values as GraphDataView,
        type: 'f32' as const,
        access: 'read' as const
      },
      {name: 'elevationValidity', view: validity, type: 'u32' as const, access: 'read' as const},
      {name: 'settings', view: props.parameters, type: 'f32' as const, access: 'read' as const},
      {name: 'pyramid', view: pyramidView, type: 'f32' as const, access: 'read' as const},
      ...(mask
        ? [{name: 'maskValues', view: mask, type: 'u32' as const, access: 'read_write' as const}]
        : []),
      ...(drop
        ? [{name: 'dropValues', view: drop, type: 'f32' as const, access: 'read_write' as const}]
        : []),
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
    if (radiusClamped) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-overflow-clear`,
          operation: 'GPUTerrainSummits',
          view: radiusClamped,
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
        declarations: getTerrainSummitsKernelWGSL({
          width,
          height,
          maximumRadiusPixels,
          cellSizeMode,
          rejectIncomplete: (props.incompleteNeighborhood ?? 'reject') === 'reject',
          reportsClamp: Boolean(radiusClamped),
          layout: pyramidLayout
        }),
        body: `let column = i32(index % WIDTH);
  let row = i32(index / WIDTH);
  summitDrop = getNan(index);
  let isSummit = evaluateSummit(column, row, index);
  ${mask ? 'maskValues[maskValuesOffset + index] = select(0u, 1u, isSummit);' : ''}
  ${drop ? 'dropValues[dropValuesOffset + index] = select(getNan(index), summitDrop, isSummit);' : ''}`
      })
    );
    if (summits && mask) {
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
          output: summits
        })
      );
      if (props.output.summitDrop && drop) {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-output-drop`,
            operation: 'GPUTerrainSummits',
            variant: 'gather-drop',
            bindings: [
              {name: 'idsIn', view: summits.ids, type: 'u32', access: 'read'},
              {name: 'countIn', view: summits.status.count, type: 'u32', access: 'read'},
              {name: 'dropIn', view: drop, type: 'f32', access: 'read'},
              {name: 'dropOut', view: props.output.summitDrop, type: 'f32', access: 'read_write'}
            ],
            invocationCount: summits.ids.length,
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

/** Edge in pixels of a level-0 pyramid cell: 2 keeps the pyramid at a third of a pixel count. */
const SUMMITS_FIRST_BLOCK_SIZE = 2;

/** WGSL literal of the largest finite float32, also the value of invalid pixels in a reject pyramid. */
const FLOAT_MAXIMUM_LITERAL = '3.4028234663852886e38';

/** Validates a caller-supplied pyramid for the summit search and returns its layout. */
function validateSummitsPyramid(
  id: string,
  pyramid: GPURasterExtremaPyramidOutput,
  grid: GPURasterMetadata
): GPURasterExtremaPyramidLayout {
  const {layout, combined} = pyramid;
  if (!layout || !combined) {
    throw new Error(`${id} pyramid must provide a layout and a combined view`);
  }
  if (layout.width !== grid.width || layout.height !== grid.height) {
    throw new Error(
      `${id} pyramid covers ${layout.width}x${layout.height} pixels but the grid is ${grid.width}x${grid.height}`
    );
  }
  if (layout.footprint !== 'cell') {
    throw new Error(`${id} pyramid must use the 'cell' footprint`);
  }
  const top = layout.levels[layout.levels.length - 1];
  if (!top || top.width !== 1 || top.height !== 1) {
    throw new Error(`${id} pyramid must reach a single top cell`);
  }
  validatePackedView(combined, ['float32'], `${id} pyramid combined`);
  if (combined.length !== 2 * layout.length) {
    throw new Error(`${id} pyramid combined must contain ${2 * layout.length} values`);
  }
  return layout;
}

/**
 * Private memory in bytes of the search stack for a pyramid layout. Each popped block pushes at
 * most four children, so the stack never holds more than `3 * levels + 4` entries of three words.
 * @internal
 */
export function getTerrainSummitsStackBytes(layout: GPURasterExtremaPyramidLayout): number {
  return (3 * layout.levels.length + 4) * 3 * 4;
}

/** Options of {@link getTerrainSummitsKernelWGSL}. @internal */
export type TerrainSummitsKernelWGSLOptions = {
  width: number;
  height: number;
  maximumRadiusPixels: number;
  cellSizeMode: GPURasterCellSizeMode;
  rejectIncomplete: boolean;
  /** Whether the kernel raises the `radiusClamped` word (binding `overflowValues`). */
  reportsClamp: boolean;
  layout: GPURasterExtremaPyramidLayout;
};

/**
 * Declarations of the summit kernel: constants, helpers and `evaluateSummit`.
 *
 * Algorithm (bit-identical to scanning every disc pixel, which it replaces):
 *
 * 1. `p` loses if any disc pixel `q` beats it under `(height, -index)`. Split by index order: rows
 *    above `p` and `p`'s columns to the left have smaller indices, so equal heights beat `p` there
 *    (violation `>=`); rows below and columns to the right need a strictly higher pixel (`>`).
 * 2. The row of `p` is scanned directly; every `|dx| <= getDiscRowHalfWidth(0)` pixel is in the disc.
 * 3. Each half (rows above, rows below) is a depth-first descent of a max pyramid built with the
 *    `'cell'` footprint. For a block clipped to the half's rows: its pixel nearest the centre (least
 *    `|dx|` and `|dy|`) failing `isInDisc` means the block is disjoint from the disc; its pixel
 *    farthest from the centre (greatest `|dx|` and `|dy|`, one of the four corners) passing means
 *    every pixel passes. Both claims follow because `isInDisc` is monotone non-increasing in `|dx|`
 *    and `|dy|` even under float32 rounding (see `TERRAIN_FEATURES_DISC_ROW_WGSL`), so they equal the
 *    per-pixel predicate exactly. Contained blocks (also inside the half's rows) are decided by one
 *    maximum read; others split into four children; level 0 tests pixels. Blocks along the circle
 *    and the cut line number O(R) in total.
 * 4. Empty cells hold -FLT_MAX and, under 'reject', invalid pixels are +FLT_MAX in the pyramid, so a
 *    block maximum equals the sentinel for a valid pixel of that exact height. When `|height|` is
 *    FLT_MAX the descent therefore never reads maxima and tests pixels (`exact`).
 * 5. The ring is read per disc row: a pixel is interior iff `|dx| <= min(e(dy-1), e(dy), e(dy+1)) - 1`
 *    with `e` the row half-width (-1 outside the disc); only the remaining runs are visited, rows
 *    ascending and columns ascending, matching the scan order so ties keep the first value.
 *
 * @internal
 */
export function getTerrainSummitsKernelWGSL(options: TerrainSummitsKernelWGSLOptions): string {
  const {width, height, maximumRadiusPixels, cellSizeMode, layout} = options;
  const stackSize = 3 * layout.levels.length + 4;
  // The stack must leave room for the kernel's other private words.
  assert(getTerrainSummitsStackBytes(layout) + 256 < 8192);
  const top = layout.levels.length - 1;
  return `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const MAXIMUM_RADIUS_PIXELS: i32 = ${maximumRadiusPixels};
const REJECT_INCOMPLETE: bool = ${options.rejectIncomplete};
const NO_INDEX: u32 = 0xffffffffu;
const FLOAT_MAXIMUM: f32 = ${FLOAT_MAXIMUM_LITERAL};
const STACK_SIZE: u32 = ${stackSize}u;
${TERRAIN_FEATURES_WGSL_CONSTANTS}
${RASTER_WGSL_HELPERS}
${getRasterGroundCellSizeWGSL(cellSizeMode, TERRAIN_SUMMITS_GRID_SLOTS)}
${TERRAIN_FEATURES_DISC_WGSL}
${TERRAIN_FEATURES_DISC_ROW_WGSL}
${getRasterExtremaPyramidWGSL(layout, 'pyramid')}
var<private> summitDrop: f32;
var<private> stackLevel: array<u32, ${stackSize}>;
var<private> stackColumn: array<u32, ${stackSize}>;
var<private> stackRow: array<u32, ${stackSize}>;
// Whether a pixel of height \`other\` beats the centre: >= where it has the smaller index, else >.
fn beatsCentre(other: f32, ownHeight: f32, hasSmallerIndex: bool) -> bool {
  return select((other > ownHeight), (other >= ownHeight), hasSmallerIndex);
}
// Returns true when no disc pixel of rows [rowLow, rowHigh] beats the centre pixel.
fn isHalfDiscDominated(column: i32, row: i32, rowLow: i32, rowHigh: i32, ownHeight: f32,
    hasSmallerIndex: bool, exact: bool, cell: vec2<f32>, radiusSquared: f32) -> bool {
  if (rowLow > rowHigh) { return true; }
  var depth = 1u;
  stackLevel[0] = ${top}u;
  stackColumn[0] = 0u;
  stackRow[0] = 0u;
  loop {
    if (depth == 0u) { break; }
    depth--;
    let level = stackLevel[depth];
    let cellColumn = i32(stackColumn[depth]);
    let cellRow = i32(stackRow[depth]);
    let blockSize = i32(pyramidLevelBlockSize(level));
    let x0 = cellColumn * blockSize;
    let y0 = cellRow * blockSize;
    let x1 = min(x0 + blockSize, i32(WIDTH)) - 1;
    let y1 = min(y0 + blockSize, i32(HEIGHT)) - 1;
    let ya = max(y0, rowLow);
    let yb = min(y1, rowHigh);
    if (ya > yb) { continue; }
    let dxLow = x0 - column;
    let dxHigh = x1 - column;
    let dyLow = ya - row;
    let dyHigh = yb - row;
    // Block pixel nearest the centre: disjoint from the disc when it is outside.
    var nearX = 0;
    if (dxLow > 0) { nearX = dxLow; } else if (dxHigh < 0) { nearX = dxHigh; }
    var nearY = 0;
    if (dyLow > 0) { nearY = dyLow; } else if (dyHigh < 0) { nearY = dyHigh; }
    if (!isOffsetInDisc(nearX, nearY, cell, radiusSquared)) { continue; }
    let farX = max(abs(dxLow), abs(dxHigh));
    let farY = max(abs(dyLow), abs(dyHigh));
    if (!exact && ya == y0 && yb == y1 && isOffsetInDisc(farX, farY, cell, radiusSquared)) {
      let blockMaximum = pyramid[pyramidOffset + pyramidLevelIndex(level, u32(x0), u32(y0))];
      if (beatsCentre(blockMaximum, ownHeight, hasSmallerIndex)) { return false; }
      continue;
    }
    if (level == 0u) {
      for (var y = ya; y <= yb; y++) {
        for (var x = x0; x <= x1; x++) {
          if (!isOffsetInDisc(x - column, y - row, cell, radiusSquared)) { continue; }
          let pixelIndex = u32(y) * WIDTH + u32(x);
          if (elevationValidity[elevationValidityOffset + pixelIndex] == 0u) {
            if (REJECT_INCOMPLETE) { return false; }
            continue;
          }
          if (beatsCentre(elevationValues[elevationValuesOffset + pixelIndex], ownHeight, hasSmallerIndex)) {
            return false;
          }
        }
      }
      continue;
    }
    let childSize = i32(pyramidLevelBlockSize(level - 1u));
    for (var child = 0; child < 4; child++) {
      let childColumn = 2 * cellColumn + (child & 1);
      let childRow = 2 * cellRow + (child >> 1);
      if (childColumn * childSize >= i32(WIDTH) || childRow * childSize >= i32(HEIGHT)) { continue; }
      stackLevel[depth] = level - 1u;
      stackColumn[depth] = u32(childColumn);
      stackRow[depth] = u32(childRow);
      depth++;
    }
  }
  return true;
}
// Returns whether the pixel is a summit and leaves its drop in summitDrop.
fn evaluateSummit(column: i32, row: i32, index: u32) -> bool {
  let cell = getGroundCellSize(u32(row));
  if (!(isFiniteValue(cell.x) && isFiniteValue(cell.y) && cell.x > 0.0 && cell.y > 0.0)) {
    return false;
  }
  let requested = settings[settingsOffset];
  let minimumDrop = settings[settingsOffset + 1u];
  if (!(isFiniteValue(requested) && requested > 0.0)) { return false; }
  ${
    options.reportsClamp
      ? `// The clamp depends only on the row's cell size: one pixel per row reports it, so at most
  // HEIGHT invocations touch the flag word instead of every pixel (review C-17).
  if (column == 0 && isRadiusClamped(requested, cell)) { atomicMax(&overflowValues[overflowValuesOffset], 1u); }`
      : ''
  }
  if (elevationValidity[elevationValidityOffset + index] == 0u) { return false; }
  let radius = getEffectiveRadius(requested, cell);
  let radiusSquared = radius * radius;
  let centre = vec2<f32>(f32(column), f32(row));
  let ownHeight = elevationValues[elevationValuesOffset + index];
  // Cheap rejection first: almost every pixel has a higher disc pixel among its 8 neighbors, so the
  // search below only runs for 3x3 local maxima. Every rejection here would also be reached by
  // the full search (higher pixel, or an incomplete neighborhood under 'reject'), so results match.
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
  let ownRowHalfWidth = getDiscRowHalfWidth(0, cell, radiusSquared);
  let columnHalfHeight = getDiscColumnHalfHeight(cell, radiusSquared);
  // The widest row is dy = 0 and the tallest column is dx = 0, so these decide grid containment.
  if (REJECT_INCOMPLETE &&
      (column - ownRowHalfWidth < 0 || column + ownRowHalfWidth >= i32(WIDTH) ||
       row - columnHalfHeight < 0 || row + columnHalfHeight >= i32(HEIGHT))) {
    return false;
  }
  // Heights of FLT_MAX collide with the empty/invalid sentinels of block maxima.
  let exact = abs(ownHeight) >= FLOAT_MAXIMUM;
  // Own row: every |dx| <= ownRowHalfWidth is in the disc.
  for (var offsetX = -ownRowHalfWidth; offsetX <= ownRowHalfWidth; offsetX++) {
    let pixelColumn = column + offsetX;
    if (offsetX == 0 || pixelColumn < 0 || pixelColumn >= i32(WIDTH)) { continue; }
    let neighborIndex = index + u32(offsetX) ;
    if (elevationValidity[elevationValidityOffset + neighborIndex] == 0u) {
      if (REJECT_INCOMPLETE) { return false; }
      continue;
    }
    if (beatsCentre(elevationValues[elevationValuesOffset + neighborIndex], ownHeight, offsetX < 0)) {
      return false;
    }
  }
  let lastRow = i32(HEIGHT) - 1;
  if (!isHalfDiscDominated(column, row, max(row - columnHalfHeight, 0), row - 1, ownHeight, true, exact, cell, radiusSquared) ||
      !isHalfDiscDominated(column, row, row + 1, min(row + columnHalfHeight, lastRow), ownHeight, false, exact, cell, radiusSquared)) {
    return false;
  }
  var ringHeight = 0.0;
  var hasRing = false;
  var previousHalfWidth = -1;
  var currentHalfWidth = getDiscRowHalfWidth(-columnHalfHeight, cell, radiusSquared);
  for (var offsetY = -columnHalfHeight; offsetY <= columnHalfHeight; offsetY++) {
    var nextHalfWidth = -1;
    if (offsetY < columnHalfHeight) { nextHalfWidth = getDiscRowHalfWidth(offsetY + 1, cell, radiusSquared); }
    let pixelRow = row + offsetY;
    if (pixelRow >= 0 && pixelRow <= lastRow) {
      let interiorHalfWidth = min(previousHalfWidth, min(currentHalfWidth, nextHalfWidth));
      // Interior pixels (all eight neighbors in the disc) have |dx| <= interiorHalfWidth - 1.
      let hasInterior = interiorHalfWidth > 0;
      for (var run = 0; run < 2; run++) {
        if (run == 1 && !hasInterior) { break; }
        var runStart = -currentHalfWidth;
        var runEnd = currentHalfWidth;
        if (hasInterior) {
          if (run == 0) { runEnd = -interiorHalfWidth; } else { runStart = interiorHalfWidth; }
        }
        for (var offsetX = runStart; offsetX <= runEnd; offsetX++) {
          let pixelColumn = column + offsetX;
          if ((offsetX == 0 && offsetY == 0) || pixelColumn < 0 || pixelColumn >= i32(WIDTH)) { continue; }
          let neighborIndex = u32(pixelRow) * WIDTH + u32(pixelColumn);
          if (elevationValidity[elevationValidityOffset + neighborIndex] == 0u) { continue; }
          let neighborHeight = elevationValues[elevationValuesOffset + neighborIndex];
          if (!hasRing || neighborHeight > ringHeight) {
            ringHeight = neighborHeight;
            hasRing = true;
          }
        }
      }
    }
    previousHalfWidth = currentHalfWidth;
    currentHalfWidth = nextHalfWidth;
  }
  if (hasRing) {
    let dropValue = ownHeight - ringHeight;
    if (dropValue < minimumDrop) { return false; }
    summitDrop = dropValue;
  } else {
    summitDrop = getInfinity(index);
  }
  return true;
}`;
}
