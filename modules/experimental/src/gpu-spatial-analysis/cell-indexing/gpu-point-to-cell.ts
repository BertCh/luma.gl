// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GraphVectorView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {dggs} from '@luma.gl/shadertools';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {CELL_KEY_WGSL, getCellKeyLayout, QUADBIN_TILE_WGSL} from '../cell-aggregation/cell-keys';
import {validateCellIndexResolution, type GPUCellIndexFamily} from './cell-index-families';
import {GEOHASH_INDEX_WGSL, QUADKEY_INDEX_WGSL, S2_INDEX_WGSL} from './cell-index-wgsl';
import {H3_INDEX_WGSL} from './h3-index-wgsl';

const OPERATION = 'GPUPointToCell';

/** Caller-owned outputs of {@link GPUPointToCell}. Both views are row-aligned with `positions`. */
export type GPUPointToCellOutput = {
  /** One 64-bit key per row as little-endian `(low, high)` words. Zero for masked or non-finite rows. */
  cells: GraphDataView<'uint32x2'>;
  /** Optional `1` for every row that received a key and `0` for masked, non-finite or unplaceable rows. */
  validity?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUPointToCell}.
 *
 * Per-frame (no recompile): the contents of `positions` and `mask`. Topology (needs a new graph):
 * `family`, `resolution`, the row count and which optional views are present.
 */
export type GPUPointToCellProps = {
  /** Prefix for generated node IDs. Defaults to `'point-to-cell'`. */
  id?: string;
  /** Grid family of the output keys. */
  family: GPUCellIndexFamily;
  /**
   * Resolution of the output cells, fixed at compile time: Quadbin 0-26, H3 0-15, quadkey 1-29,
   * geohash 1-12, S2 0-30.
   */
  resolution: number;
  /** Longitude/latitude degrees per row. */
  positions: GraphDataView<'float32x2'>;
  /** Optional per-row mask; zero produces a zero key. */
  mask?: GraphDataView<'uint32'>;
  /** Caller-owned outputs. */
  output: GPUPointToCellOutput;
};

/**
 * Converts longitude/latitude points to 64-bit discrete global grid cell keys, one row per row.
 *
 * One compute kernel writes `output.cells` (little-endian `(low, high)` words, the Arrow `Uint64`
 * layout that `GPUCellAggregation` accepts as pre-keyed `cells`) and optionally `output.validity`.
 * Rows that are masked out or have a non-finite longitude or latitude get the zero key, the "no
 * cell" sentinel of every family. The contributor never compiles, submits or reads back.
 *
 * Exactness per family, with positions rounded to f32:
 * - `'quadbin'`, `'quadkey'`: integer-only arithmetic (the `GPUCellAggregation` fixed-point
 *   Mercator row, exact column), bit-identical on every device and equal to the BigInt reference.
 *   Longitude 180 wraps to column 0 and latitude is clipped to ±85.051129. Quadkey tiles are the
 *   Quadbin tiles (same clip semantics), packed as documented on {@link GPUCellIndexFamily}.
 *   Above zoom 26 the fixed-point row has less than one bit of margin: against the f64 formula the
 *   row differs by one tile in 1.3% of random points at zoom 26 and 9.6% at zoom 29 (about 2% at
 *   zoom 27, 4.5% at zoom 28); the kernel itself stays deterministic.
 * - `'geohash'`: integer-exact. The bisection of an f32 coordinate is an exact dyadic floor, so
 *   the key equals the reference bisection of the same f32 value. Longitude 180 and latitude 90
 *   land in the last bin; larger inputs clamp.
 * - `'s2'`: f32 face, `uv`, quadratic `st` and `ij`, then the integer Hilbert state machine. The
 *   result equals an f64 reference except for points within the f32 error of a cell edge (about
 *   6e-8 in `st`), so the mismatch rate grows with the level. Measured on 10k equal-area points:
 *   none through level 10, 1e-4 at level 11, 5e-4 at 12, 2.6e-3 at 15, 6.6e-2 at 20, 0.26 at 22,
 *   0.78 at 24 and nearly every point at levels 26-30. Through level 22 a mismatch is always an
 *   edge neighbor of the reference cell; beyond it the error spans several cells (about 9 cells at
 *   level 26 and 140 at level 30), so use levels of 20 or coarser for exact cells.
 * - `'h3'`: f32 `latLngToCell` (see `H3_INDEX_WGSL`). Exact against h3-js at resolutions 0-4;
 *   the mismatch rate on uniform points is about 1e-3 at resolution 9, 1.8% at resolution 12 and
 *   28% at resolution 15, and a mismatch is always a grid neighbor.
 *
 * Inputs must be single packed views. Outputs must not share buffers with the inputs.
 */
export class GPUPointToCell implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUPointToCellProps;

  constructor(props: GPUPointToCellProps) {
    this.id = props.id ?? 'point-to-cell';
    this.props = props;
    const id = this.id;
    validateCellIndexResolution(id, props.family, props.resolution);
    for (const [name, view] of [
      ['positions', props.positions],
      ['mask', props.mask],
      ['output.cells', props.output.cells],
      ['output.validity', props.output.validity]
    ] as const) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    const rows = props.positions.length;
    if (rows < 1) {
      throw new Error(`${id} needs at least one row`);
    }
    if (props.mask) {
      validatePackedUint32View(props.mask, `${id} mask`);
      if (props.mask.length !== rows) {
        throw new Error(`${id} mask length must equal the row count`);
      }
    }
    validatePackedView(props.output.cells, ['uint32x2'], `${id} output.cells`);
    if (props.output.cells.length !== rows) {
      throw new Error(`${id} output.cells length must equal the row count`);
    }
    if (props.output.validity) {
      validatePackedUint32View(props.output.validity, `${id} output.validity`);
      if (props.output.validity.length !== rows) {
        throw new Error(`${id} output.validity length must equal the row count`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.cells, props.output.validity],
      [props.positions, props.mask]
    );
  }

  /** Returns the single key kernel node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {family, resolution, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.mask,
      output.cells,
      output.validity
    ]);
    const bindings: WGSLKernelBinding[] = [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      ...(props.mask
        ? [{name: 'rowMask', view: props.mask, type: 'u32', access: 'read'} as const]
        : []),
      {name: 'cellsOut', view: output.cells, type: 'u32', access: 'read_write'},
      ...(output.validity
        ? [{name: 'validityOut', view: output.validity, type: 'u32', access: 'read_write'} as const]
        : [])
    ];
    const keyExpression = getKeyExpression(family, resolution);
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-keys`,
        operation: OPERATION,
        variant: family,
        bindings,
        invocationCount: props.positions.length,
        declarations: `const RESOLUTION: u32 = ${resolution}u;
${getFamilyDeclarations(family)}
fn isFiniteBits(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}`,
        body: `let longitude = positions[positionsOffset + 2u * index];
  let latitude = positions[positionsOffset + 2u * index + 1u];
  var isValid = isFiniteBits(longitude) && isFiniteBits(latitude);
  ${props.mask ? 'isValid = isValid && rowMask[rowMaskOffset + index] != 0u;' : ''}
  var key = vec2u(0u);
  if (isValid) {
    key = ${keyExpression};
  }
  isValid = isValid && any(key != vec2u(0u));
  cellsOut[cellsOutOffset + 2u * index] = key.y;
  cellsOut[cellsOutOffset + 2u * index + 1u] = key.x;
  ${output.validity ? 'validityOut[validityOutOffset + index] = select(0u, 1u, isValid);' : ''}`
      })
    ];
  }
}

function getFamilyDeclarations(family: GPUCellIndexFamily): string {
  switch (family) {
    case 'quadbin':
      return `${CELL_KEY_WGSL}\n${QUADBIN_TILE_WGSL}`;
    case 'quadkey':
      return QUADKEY_INDEX_WGSL;
    case 'geohash':
      return GEOHASH_INDEX_WGSL;
    case 's2':
      return S2_INDEX_WGSL;
    case 'h3':
      return `${dggs.source}\n${H3_INDEX_WGSL}`;
  }
}

function getKeyExpression(family: GPUCellIndexFamily, resolution: number): string {
  switch (family) {
    case 'quadbin': {
      const layout = getCellKeyLayout('quadbin', resolution);
      return `cellGetKey(
      quadbinGetCompactKey(quadbinGetTileX(longitude, RESOLUTION), quadbinGetTileY(latitude, RESOLUTION)),
      ${layout.headerHigh}u, RESOLUTION, ${layout.lowBit}u)`;
    }
    case 'quadkey':
      return 'cellIndexQuadkeyFromLngLat(longitude, latitude, RESOLUTION)';
    case 'geohash':
      return 'cellIndexGeohashFromLngLat(longitude, latitude, RESOLUTION)';
    case 's2':
      return 'cellIndexS2FromLngLat(longitude, latitude, RESOLUTION)';
    case 'h3':
      return 'cellIndexH3FromLngLat(vec2f(longitude, latitude), RESOLUTION)';
  }
}
