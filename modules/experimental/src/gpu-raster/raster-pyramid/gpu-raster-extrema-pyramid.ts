// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPURasterBand} from '../index';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import {
  getTerrainBandViews,
  getTerrainElevationNodes,
  validateTerrainBandBelongsToGraph,
  validateTerrainBuffersDistinct,
  validateTerrainGrid
} from '../../gpu-terrain/terrain-analysis/terrain-analysis-utils';

/**
 * Value of a maximum cell whose footprint holds no valid pixel (-FLT_MAX).
 *
 * Finite on purpose: Metal fast math may assume no infinities.
 */
export const GPU_RASTER_EXTREMA_PYRAMID_EMPTY_MAXIMUM = -3.4028234663852886e38;

/** Value of a minimum cell whose footprint holds no valid pixel (+FLT_MAX). */
export const GPU_RASTER_EXTREMA_PYRAMID_EMPTY_MINIMUM = 3.4028234663852886e38;

/** Largest supported `firstBlockSize`. */
const MAXIMUM_FIRST_BLOCK_SIZE = 256;

/**
 * Cell footprint model of a pyramid.
 *
 * - `'cell'`: a cell covers exactly its `S x S` pixel block.
 * - `'bilinear'`: a cell also covers one extra pixel column and row (clipped to the grid), so every
 *   bilinear sample whose lower corner pixel `floor(position)` lies in the block satisfies
 *   `minimum <= sample <= maximum` with one cell lookup.
 */
export type GPURasterExtremaPyramidFootprint = 'cell' | 'bilinear';

/** One level of a {@link GPURasterExtremaPyramidLayout}. */
export type GPURasterExtremaPyramidLevel = {
  /** 0-based level. */
  level: number;
  /** Block size in pixels, `firstBlockSize * 2^level`. */
  blockSize: number;
  /** Cells per row, `ceil(width / blockSize)`. */
  width: number;
  /** Cell rows, `ceil(height / blockSize)`. */
  height: number;
  /** Index of the level's first cell in a packed view (levels concatenated, level 0 first). */
  offset: number;
};

/** Level sizes and offsets of a pyramid over a `width x height` raster. */
export type GPURasterExtremaPyramidLayout = {
  /** Raster width in pixels. */
  width: number;
  /** Raster height in pixels. */
  height: number;
  /** Block size of level 0. */
  firstBlockSize: number;
  /** Footprint model. */
  footprint: GPURasterExtremaPyramidFootprint;
  /** Levels, finest first. */
  levels: readonly GPURasterExtremaPyramidLevel[];
  /** Total float32 count of one packed extremum view. */
  length: number;
};

/** Options of {@link getGPURasterExtremaPyramidLayout}. */
export type GPURasterExtremaPyramidLayoutOptions = {
  /** Power of two in [1, 256]. Defaults to 4. */
  firstBlockSize?: number;
  /**
   * Maximum number of levels (integer >= 1). Defaults to, and is clamped to, the number of levels
   * until a level is one cell.
   */
  levelCount?: number;
  /** Footprint model. Defaults to `'bilinear'`. */
  footprint?: GPURasterExtremaPyramidFootprint;
};

/**
 * Computes the level layout of a min/max pyramid.
 *
 * Level `L` has block size `firstBlockSize * 2^L` and `ceil(width / block) x ceil(height / block)`
 * cells. The default level count runs until a level is `1 x 1`, inclusive.
 *
 * @throws On a non-positive size, a `firstBlockSize` that is not a power of two in [1, 256], a
 *   `levelCount` that is not an integer >= 1, or an unknown footprint.
 */
export function getGPURasterExtremaPyramidLayout(
  width: number,
  height: number,
  options: GPURasterExtremaPyramidLayoutOptions = {}
): GPURasterExtremaPyramidLayout {
  validateTerrainGrid('GPURasterExtremaPyramid', width, height);
  const firstBlockSize = options.firstBlockSize ?? 4;
  if (
    !Number.isInteger(firstBlockSize) ||
    firstBlockSize < 1 ||
    firstBlockSize > MAXIMUM_FIRST_BLOCK_SIZE ||
    (firstBlockSize & (firstBlockSize - 1)) !== 0
  ) {
    throw new Error('GPURasterExtremaPyramid firstBlockSize must be a power of two in [1, 256]');
  }
  const footprint = options.footprint ?? 'bilinear';
  if (footprint !== 'cell' && footprint !== 'bilinear') {
    throw new Error("GPURasterExtremaPyramid footprint must be 'cell' or 'bilinear'");
  }
  const requestedLevelCount = options.levelCount ?? Infinity;
  if (
    requestedLevelCount !== Infinity &&
    (!Number.isInteger(requestedLevelCount) || requestedLevelCount < 1)
  ) {
    throw new Error('GPURasterExtremaPyramid levelCount must be an integer >= 1');
  }
  const levels: GPURasterExtremaPyramidLevel[] = [];
  let offset = 0;
  for (let level = 0; level < requestedLevelCount; level++) {
    const blockSize = firstBlockSize * 2 ** level;
    const levelWidth = Math.ceil(width / blockSize);
    const levelHeight = Math.ceil(height / blockSize);
    levels.push({level, blockSize, width: levelWidth, height: levelHeight, offset});
    offset += levelWidth * levelHeight;
    if (levelWidth === 1 && levelHeight === 1) {
      break;
    }
  }
  return {width, height, firstBlockSize, footprint, levels, length: offset};
}

/** Properties for {@link GPURasterExtremaPyramid}. */
export type GPURasterExtremaPyramidProps = GPURasterExtremaPyramidLayoutOptions & {
  /** Prefix for node and transient IDs. Defaults to `'raster-extrema-pyramid'`. */
  id?: string;
  /** Raster width in pixels. Cells are in pixel index space, so no cell-size model applies. */
  width: number;
  /** Raster height in pixels. */
  height: number;
  /** Raster band, buffer or texture. Calibration, no-data and validity are applied. */
  input: GPURasterBand;
  /** Receives per-cell maxima; `length === layout.length`. */
  maximum?: GraphDataView<'float32'>;
  /** Receives per-cell minima; `length === layout.length`. */
  minimum?: GraphDataView<'float32'>;
};

function getFloat32Literal(value: number): string {
  return `${Math.fround(value)}`.replace(/^(-?\d+)$/, '$1.0');
}

/** Returns the WGSL identifier stem `POINT_HORIZON` for `pointHorizon`. */
function getConstantStem(prefix: string): string {
  return prefix.replace(/([a-z0-9])([A-Z])/g, '$1_$2').toUpperCase();
}

function validateViewLength(id: string, name: string, view: GraphDataView, length: number): void {
  validatePackedView(view, ['float32'], `${id} ${name}`);
  if (view.length !== length) {
    throw new Error(`${id} ${name} must contain ${length} values`);
  }
}

/**
 * Builds a min/max pyramid ("maximum mipmap", Tevs, Ihrke and Seidel 2008) of a raster band.
 *
 * Level `L` has block size `S = firstBlockSize * 2^L`. The footprint of cell `(cx, cy)` is the pixel
 * range `[cx*S, min(cx*S + S + e, width))` by the same in rows, with `e = 1` for `'bilinear'` and
 * `e = 0` for `'cell'`. `maximum` / `minimum` hold the extremum over valid pixels of the footprint,
 * or {@link GPU_RASTER_EXTREMA_PYRAMID_EMPTY_MAXIMUM} / `_EMPTY_MINIMUM` when none is valid.
 * Levels are concatenated, level 0 first, row-major within a level (see `layout`).
 *
 * Build order and exactness: level 0 scans each footprint rows ascending then columns ascending;
 * level `L > 0` reduces the children `(2cx,2cy), (2cx+1,2cy), (2cx,2cy+1), (2cx+1,2cy+1)` of level
 * `L - 1` that exist, in that order. Both compare from the sentinel with strict `>` / `<`, so the
 * result is bit-identical to the CPU mirror, including the sign of zero.
 *
 * The hierarchical build equals the direct per-pixel definition for both footprints. Along one axis
 * the child footprints of cell `c` at level `L` are `[cS, cS + S/2 + e)` and
 * `[cS + S/2, cS + S + e)`, clipped to the grid; their union is `[cS, cS + S + e)` clipped, which is
 * the parent footprint. If the second child does not exist (`cS + S/2 >= size`), the first child's
 * footprint already reaches `size`, equal to the clipped parent. The first child always exists
 * because `cS < size`. Rows and columns are independent, so the 2D union is the parent rectangle.
 */
export class GPURasterExtremaPyramid implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPURasterExtremaPyramidProps;
  /** Level layout; use it to size and index the output views. */
  readonly layout: GPURasterExtremaPyramidLayout;

  constructor(props: GPURasterExtremaPyramidProps) {
    this.id = props.id ?? 'raster-extrema-pyramid';
    this.props = props;
    const {id} = this;
    validateTerrainGrid(id, props.width, props.height);
    this.layout = getGPURasterExtremaPyramidLayout(props.width, props.height, props);
    if (!props.maximum && !props.minimum) {
      throw new Error(`${id} requires at least one output`);
    }
    for (const [name, view] of [
      ['maximum', props.maximum],
      ['minimum', props.minimum]
    ] as const) {
      if (view) {
        validateViewLength(id, name, view, this.layout.length);
      }
    }
    validateTerrainBuffersDistinct(
      id,
      [props.maximum, props.minimum],
      getTerrainBandViews(props.input)
    );
  }

  /** Returns canonicalization nodes followed by one `${id}-level-${L}` node per level. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, layout} = this;
    validateTerrainBandBelongsToGraph(id, graph, props.input, []);
    validateGraphViewsBelongToGraph(id, graph, [props.maximum, props.minimum]);
    const source = getTerrainElevationNodes(
      graph,
      id,
      props.input,
      props.width,
      props.height,
      true
    );
    return [
      ...source.nodes,
      ...createRasterExtremaPyramidNodes(graph, {
        id,
        layout,
        values: source.band.storage.values as GraphDataView<'float32'>,
        validity: source.band.validity as GraphDataView<'uint32'>,
        maximum: props.maximum,
        minimum: props.minimum
      })
    ];
  }
}

/**
 * Creates the per-level kernel nodes `${id}-level-${L}` of a pyramid from canonical values.
 *
 * `combined` is an alternative to `maximum` / `minimum`: one view of `2 * layout.length` floats with
 * maximum levels at `[0, length)` and minimum levels at `[length, 2 * length)`.
 *
 * @internal
 */
export function createRasterExtremaPyramidNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    layout: GPURasterExtremaPyramidLayout;
    values: GraphDataView<'float32'>;
    validity: GraphDataView<'uint32'>;
    maximum?: GraphDataView<'float32'>;
    minimum?: GraphDataView<'float32'>;
    /**
     * Alternative to maximum/minimum: ONE view of 2 * layout.length floats, maximum levels at
     * [0, length), minimum levels at [length, 2 * length). Consumers use it to spend one storage
     * binding on both.
     */
    combined?: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, layout, values, validity, maximum, minimum, combined} = props;
  if (combined) {
    if (maximum || minimum) {
      throw new Error(`${id} combined cannot be used together with maximum or minimum`);
    }
    validateViewLength(id, 'combined', combined, 2 * layout.length);
  } else if (!maximum && !minimum) {
    throw new Error(`${id} requires at least one output`);
  }
  for (const [name, view] of [
    ['maximum', maximum],
    ['minimum', minimum]
  ] as const) {
    if (view) {
      validateViewLength(id, name, view, layout.length);
    }
  }
  const pixelCount = layout.width * layout.height;
  if (values.length !== pixelCount || validity.length !== pixelCount) {
    throw new Error(`${id} values and validity must contain one value per pixel`);
  }

  const emptyMaximum = getFloat32Literal(GPU_RASTER_EXTREMA_PYRAMID_EMPTY_MAXIMUM);
  const emptyMinimum = getFloat32Literal(GPU_RASTER_EXTREMA_PYRAMID_EMPTY_MINIMUM);
  const extraPixels = layout.footprint === 'bilinear' ? 1 : 0;
  const outputBindings: WGSLKernelBinding[] = combined
    ? [{name: 'combinedLevels', view: combined, type: 'f32', access: 'read_write'}]
    : [
        ...(maximum
          ? [{name: 'maximumLevels', view: maximum, type: 'f32', access: 'read_write'} as const]
          : []),
        ...(minimum
          ? [{name: 'minimumLevels', view: minimum, type: 'f32', access: 'read_write'} as const]
          : [])
      ];
  const hasMaximum = Boolean(combined || maximum);
  const hasMinimum = Boolean(combined || minimum);
  // Element index of a level cell in the maximum / minimum storage, offsets included.
  const maximumArray = combined ? 'combinedLevels' : 'maximumLevels';
  const minimumArray = combined ? 'combinedLevels' : 'minimumLevels';
  const maximumBase = combined ? 'combinedLevelsOffset' : 'maximumLevelsOffset';
  const minimumBase = combined
    ? `(combinedLevelsOffset + ${layout.length}u)`
    : 'minimumLevelsOffset';

  const nodes: GPUCommandNode<Parameters>[] = [];
  for (const level of layout.levels) {
    const cellCount = level.width * level.height;
    const header = `const LEVEL_WIDTH: u32 = ${level.width}u;
const LEVEL_OFFSET: u32 = ${level.offset}u;
const EMPTY_MAXIMUM: f32 = ${emptyMaximum};
const EMPTY_MINIMUM: f32 = ${emptyMinimum};`;
    const write = `${
      hasMaximum ? `${maximumArray}[${maximumBase} + LEVEL_OFFSET + index] = maximumValue;\n  ` : ''
    }${hasMinimum ? `${minimumArray}[${minimumBase} + LEVEL_OFFSET + index] = minimumValue;` : ''}`;
    if (level.level === 0) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-level-0`,
          operation: 'GPURasterExtremaPyramid',
          variant: 'level-0',
          bindings: [
            {name: 'values', view: values, type: 'f32', access: 'read'},
            {name: 'validity', view: validity, type: 'u32', access: 'read'},
            ...outputBindings
          ],
          invocationCount: cellCount,
          declarations: `const WIDTH: u32 = ${layout.width}u;
const HEIGHT: u32 = ${layout.height}u;
const BLOCK_SIZE: u32 = ${level.blockSize}u;
const FOOTPRINT_EXTRA: u32 = ${extraPixels}u;
${header}`,
          body: `let cellColumn = index % LEVEL_WIDTH;
  let cellRow = index / LEVEL_WIDTH;
  let x0 = cellColumn * BLOCK_SIZE;
  let y0 = cellRow * BLOCK_SIZE;
  let x1 = min(x0 + BLOCK_SIZE + FOOTPRINT_EXTRA, WIDTH);
  let y1 = min(y0 + BLOCK_SIZE + FOOTPRINT_EXTRA, HEIGHT);
  var maximumValue = EMPTY_MAXIMUM;
  var minimumValue = EMPTY_MINIMUM;
  for (var y = y0; y < y1; y++) {
    for (var x = x0; x < x1; x++) {
      let pixel = y * WIDTH + x;
      if (validity[validityOffset + pixel] != 0u) {
        let value = values[valuesOffset + pixel];
        if (value > maximumValue) { maximumValue = value; }
        if (value < minimumValue) { minimumValue = value; }
      }
    }
  }
  ${write}`
        })
      );
    } else {
      const below = layout.levels[level.level - 1];
      const reduce = (array: string, base: string, sentinel: string, comparison: string) => `var ${
        comparison === '>' ? 'maximumValue' : 'minimumValue'
      } = ${sentinel};
  for (var child = 0u; child < 4u; child++) {
    let childColumn = 2u * cellColumn + (child & 1u);
    let childRow = 2u * cellRow + (child >> 1u);
    if (childColumn < BELOW_WIDTH && childRow < BELOW_HEIGHT) {
      let value = ${array}[${base} + BELOW_OFFSET + childRow * BELOW_WIDTH + childColumn];
      if (value ${comparison} ${comparison === '>' ? 'maximumValue' : 'minimumValue'}) { ${
        comparison === '>' ? 'maximumValue' : 'minimumValue'
      } = value; }
    }
  }`;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-level-${level.level}`,
          operation: 'GPURasterExtremaPyramid',
          variant: 'reduce',
          bindings: outputBindings,
          invocationCount: cellCount,
          declarations: `const BELOW_WIDTH: u32 = ${below.width}u;
const BELOW_HEIGHT: u32 = ${below.height}u;
const BELOW_OFFSET: u32 = ${below.offset}u;
${header}`,
          body: `let cellColumn = index % LEVEL_WIDTH;
  let cellRow = index / LEVEL_WIDTH;
  ${hasMaximum ? reduce(maximumArray, maximumBase, 'EMPTY_MAXIMUM', '>') : 'let maximumValue = 0.0;'}
  ${hasMinimum ? reduce(minimumArray, minimumBase, 'EMPTY_MINIMUM', '<') : 'let minimumValue = 0.0;'}
  ${write}`
        })
      );
    }
  }
  return nodes;
}

/**
 * Emits WGSL constants and helpers for consumers that read a pyramid, with identifiers derived
 * from `prefix` (camelCase for functions, `UPPER_SNAKE` for constants):
 *
 * - `const ${PREFIX}_LEVEL_COUNT: u32`
 * - `const ${PREFIX}_EMPTY_MAXIMUM: f32`, `${PREFIX}_EMPTY_MINIMUM: f32`
 * - `const ${PREFIX}_MINIMUM_OFFSET: u32 = layout.length`, the float32 offset of the minimum
 *   levels inside a `combined` view
 * - `fn ${prefix}LevelBlockSize(level: u32) -> u32`
 * - `fn ${prefix}LevelIndex(level: u32, column: u32, row: u32) -> u32`: float32 index (without any
 *   binding offset) of the cell containing base pixel `(column, row)` at `level`
 *
 * @internal
 */
export function getRasterExtremaPyramidWGSL(
  layout: GPURasterExtremaPyramidLayout,
  prefix: string = 'pyramid'
): string {
  const stem = getConstantStem(prefix);
  const count = layout.levels.length;
  const list = (select: (level: GPURasterExtremaPyramidLevel) => number) =>
    `array<u32, ${count}>(${layout.levels.map(level => `${select(level)}u`).join(', ')})`;
  return `const ${stem}_LEVEL_COUNT: u32 = ${count}u;
const ${stem}_EMPTY_MAXIMUM: f32 = ${getFloat32Literal(GPU_RASTER_EXTREMA_PYRAMID_EMPTY_MAXIMUM)};
const ${stem}_EMPTY_MINIMUM: f32 = ${getFloat32Literal(GPU_RASTER_EXTREMA_PYRAMID_EMPTY_MINIMUM)};
const ${stem}_MINIMUM_OFFSET: u32 = ${layout.length}u;
const ${stem}_BLOCK_SIZES = ${list(level => level.blockSize)};
const ${stem}_LEVEL_WIDTHS = ${list(level => level.width)};
const ${stem}_LEVEL_OFFSETS = ${list(level => level.offset)};
fn ${prefix}LevelBlockSize(level: u32) -> u32 {
  return ${stem}_BLOCK_SIZES[level];
}
fn ${prefix}LevelIndex(level: u32, column: u32, row: u32) -> u32 {
  let blockSize = ${stem}_BLOCK_SIZES[level];
  return ${stem}_LEVEL_OFFSETS[level] + (row / blockSize) * ${stem}_LEVEL_WIDTHS[level] + column / blockSize;
}`;
}
