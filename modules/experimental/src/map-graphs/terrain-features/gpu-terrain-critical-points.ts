// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPURasterBand} from '../../gpu-raster/index';
import {createMapGraphFillNode, createMapGraphKernelNode} from '../map-graph-kernels';
import type {GPUMapGraphRecipe} from '../map-graph-types';
import {validateGraphViewsBelongToGraph} from '../map-graph-utils';
import {
  getTerrainBandViews,
  getTerrainElevationNodes,
  validateTerrainBandBelongsToGraph,
  validateTerrainBuffersDistinct,
  validateTerrainGrid
} from '../terrain-analysis/terrain-analysis-utils';

/**
 * Critical-point classes written by {@link GPUTerrainCriticalPoints}.
 *
 * `boundary` marks pixels that are not classified because the ring leaves the grid or touches
 * nodata; it is deliberately not a guess.
 */
export const GPU_TERRAIN_CRITICAL_POINT = {
  regular: 0,
  peak: 1,
  pit: 2,
  saddle: 3,
  boundary: 4,
  noData: 5
} as const;

/** Number of class codes, and the required length of the optional `counts` output. */
export const GPU_TERRAIN_CRITICAL_POINT_CLASS_COUNT = 6;

/**
 * Properties for {@link GPUTerrainCriticalPoints}.
 *
 * Cell-size independent: the classification is made of pure comparisons of heights, so it does not
 * depend on the metric size of a pixel or on the vertical unit.
 */
export type GPUTerrainCriticalPointsProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-critical-points'`. */
  id?: string;
  /** Grid width in pixels. */
  width: number;
  /** Grid height in pixels. */
  height: number;
  /** Elevation band, buffer or texture. It is canonicalised to float32 values plus validity. */
  elevation: GPURasterBand;
  /** Ring connectivity: 8 (Peucker-Douglas, default) or 6 (Freudenthal triangulation). */
  connectivity?: 8 | 6;
  /** One {@link GPU_TERRAIN_CRITICAL_POINT} code per pixel. */
  classes: GraphDataView<'uint32'>;
  /** Optional number of sign changes `c` around the ring, per pixel (0 for unclassified pixels). */
  signChanges?: GraphDataView<'uint32'>;
  /**
   * Optional atomic count per class, 6 values indexed by {@link GPU_TERRAIN_CRITICAL_POINT}. It is
   * cleared at the start of every encoding.
   */
  counts?: GraphDataView<'uint32'>;
};

/**
 * Classifies every pixel as regular, peak, pit, saddle, boundary or nodata by counting sign
 * changes of the height difference around its ring (Peucker and Douglas 1975; Banchoff 1967 for
 * the piecewise-linear critical points).
 *
 * Rule. An invalid pixel is `noData`. A pixel on the grid edge, or with any invalid ring neighbour,
 * is `boundary`. Otherwise the ring is walked in cyclic order and each neighbour `q` of `p` is
 * "higher" iff `h_q > h_p || (h_q == h_p && q > p)` with `q` and `p` row-major indices (simulation
 * of simplicity: no zero signs, so plateaus resolve deterministically as if heights were perturbed
 * by an increasing function of the index). With `c` sign changes around the cycle: `c == 0` with
 * all neighbours lower is a `peak`; `c == 0` with all higher is a `pit`; `c == 2` is `regular`;
 * `c >= 4` is a `saddle` of multiplicity `c / 2 - 1` (read it from `signChanges`).
 *
 * Rings, with rows growing downward. 8: E, NE, N, NW, W, SW, S, SE, that is offsets
 * `(+1,0), (+1,-1), (0,-1), (-1,-1), (-1,0), (-1,+1), (0,+1), (+1,+1)` in (column, row). 6: the
 * Freudenthal triangulation, every cell split along its main diagonal from `(c, r)` to
 * `(c + 1, r + 1)`, giving neighbours `(+1,0), (+1,+1), (0,+1), (-1,0), (-1,-1), (0,-1)` in cyclic
 * order.
 *
 * The 8-ring is the classic Peucker-Douglas classification but is not a consistent triangulation:
 * at diagonal ambiguities it can count spurious saddles (on a periodic egg-crate it reports twice
 * the saddle multiplicity). The 6-ring is the Banchoff, PL-consistent choice, where for a surface
 * without boundary effects `#peaks - #saddles (with multiplicity) + #pits` obeys the Euler
 * relation: it equals the Euler characteristic, 0 on a torus (a periodic egg-crate window) and 1
 * for a disk whose height decreases towards its rim.
 *
 * Nodata never contributes a height: an invalid neighbour makes the pixel `boundary` before any
 * height comparison is used.
 */
export class GPUTerrainCriticalPoints implements GPUMapGraphRecipe {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'terrain-critical-points';
  /** Validated properties. */
  readonly props: GPUTerrainCriticalPointsProps;

  constructor(props: GPUTerrainCriticalPointsProps) {
    this.id = props.id ?? this.recipe;
    this.props = props;
    const {id} = this;
    const pixelCount = validateTerrainGrid(id, props.width, props.height);
    if (props.connectivity !== undefined && props.connectivity !== 8 && props.connectivity !== 6) {
      throw new Error(`${id} connectivity must be 8 or 6`);
    }
    validatePackedUint32View(props.classes, `${id} classes`);
    if (props.classes.length !== pixelCount) {
      throw new Error(`${id} classes must contain one value per pixel`);
    }
    if (props.signChanges) {
      validatePackedUint32View(props.signChanges, `${id} signChanges`);
      if (props.signChanges.length !== pixelCount) {
        throw new Error(`${id} signChanges must contain one value per pixel`);
      }
    }
    if (props.counts) {
      validatePackedUint32View(props.counts, `${id} counts`);
      if (props.counts.length !== GPU_TERRAIN_CRITICAL_POINT_CLASS_COUNT) {
        throw new Error(`${id} counts must contain one value per class (6)`);
      }
    }
    validateTerrainBuffersDistinct(
      id,
      [props.classes, props.signChanges, props.counts],
      getTerrainBandViews(props.elevation)
    );
  }

  /** Returns canonical elevation, optional count clearing, and the classification kernel. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {width, height} = props;
    validateTerrainBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, [props.classes, props.signChanges, props.counts]);
    const source = getTerrainElevationNodes(graph, id, props.elevation, width, height, true);
    const nodes: GPUCommandNode<Parameters>[] = [...source.nodes];
    if (props.counts) {
      nodes.push(
        createMapGraphFillNode<Parameters>(graph, {
          id: `${id}-clear-counts`,
          operation: 'GPUTerrainCriticalPoints',
          view: props.counts,
          type: 'u32',
          value: '0u'
        })
      );
    }
    const ring = (props.connectivity ?? 8) === 8 ? RING_8 : RING_6;
    const bindings = [
      {
        name: 'elevationValues',
        view: source.band.storage.values as GraphDataView,
        type: 'f32' as const,
        access: 'read' as const
      },
      {
        name: 'elevationValidity',
        view: source.band.validity as GraphDataView<'uint32'>,
        type: 'u32' as const,
        access: 'read' as const
      },
      {
        name: 'classValues',
        view: props.classes,
        type: 'u32' as const,
        access: 'read_write' as const
      },
      ...(props.signChanges
        ? [
            {
              name: 'signChangeValues',
              view: props.signChanges,
              type: 'u32' as const,
              access: 'read_write' as const
            }
          ]
        : []),
      ...(props.counts
        ? [
            {
              name: 'classCounts',
              view: props.counts,
              type: 'atomic<u32>' as const,
              access: 'read_write' as const
            }
          ]
        : [])
    ];
    const higherLets = ring
      .map(
        ([columnOffset, rowOffset], ringIndex) =>
          `let side${ringIndex} = getSide(column, row, ${columnOffset}, ${rowOffset}, centerHeight, index);`
      )
      .join('\n  ');
    const invalidCondition = ring.map((_, ringIndex) => `side${ringIndex} == 2u`).join(' || ');
    const changeSum = ring
      .map(
        (_, ringIndex) => `select(0u, 1u, side${ringIndex} != side${(ringIndex + 1) % ring.length})`
      )
      .join(' +\n    ');
    const higherSum = ring.map((_, ringIndex) => `side${ringIndex}`).join(' + ');
    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-classify`,
        operation: 'GPUTerrainCriticalPoints',
        variant: `ring-${ring.length}`,
        bindings,
        invocationCount: width * height,
        declarations: `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const RING_COUNT: u32 = ${ring.length}u;
const REGULAR: u32 = ${GPU_TERRAIN_CRITICAL_POINT.regular}u;
const PEAK: u32 = ${GPU_TERRAIN_CRITICAL_POINT.peak}u;
const PIT: u32 = ${GPU_TERRAIN_CRITICAL_POINT.pit}u;
const SADDLE: u32 = ${GPU_TERRAIN_CRITICAL_POINT.saddle}u;
const BOUNDARY: u32 = ${GPU_TERRAIN_CRITICAL_POINT.boundary}u;
const NO_DATA: u32 = ${GPU_TERRAIN_CRITICAL_POINT.noData}u;
// 1 when the neighbor is higher (simulation of simplicity), 0 when lower, 2 when it is invalid.
// The neighbor is always inside the grid because edge pixels return earlier.
fn getSide(column: u32, row: u32, columnOffset: i32, rowOffset: i32, centerHeight: f32, centerIndex: u32) -> u32 {
  let neighborIndex = u32(i32(row) + rowOffset) * WIDTH + u32(i32(column) + columnOffset);
  if (elevationValidity[elevationValidityOffset + neighborIndex] == 0u) { return 2u; }
  let neighborHeight = elevationValues[elevationValuesOffset + neighborIndex];
  return select(0u, 1u, neighborHeight > centerHeight || (neighborHeight == centerHeight && neighborIndex > centerIndex));
}`,
        body: `let column = index % WIDTH;
  let row = index / WIDTH;
  var classCode = NO_DATA;
  var changeCount = 0u;
  if (elevationValidity[elevationValidityOffset + index] != 0u) {
    if (column == 0u || row == 0u || column == WIDTH - 1u || row == HEIGHT - 1u) {
      classCode = BOUNDARY;
    } else {
      let centerHeight = elevationValues[elevationValuesOffset + index];
      ${higherLets}
      if (${invalidCondition}) {
        classCode = BOUNDARY;
      } else {
        changeCount = ${changeSum};
        let higherCount = ${higherSum};
        classCode = select(SADDLE, REGULAR, changeCount == 2u);
        if (changeCount == 0u) {
          classCode = select(PEAK, PIT, higherCount == RING_COUNT);
        }
      }
    }
  }
  classValues[classValuesOffset + index] = classCode;
  ${props.signChanges ? 'signChangeValues[signChangeValuesOffset + index] = changeCount;' : ''}
  ${props.counts ? 'atomicAdd(&classCounts[classCountsOffset + classCode], 1u);' : ''}`
      })
    );
    return nodes;
  }
}

/** Ring offsets `[column, row]`, cyclic, rows growing downward. */
const RING_8 = [
  [1, 0],
  [1, -1],
  [0, -1],
  [-1, -1],
  [-1, 0],
  [-1, 1],
  [0, 1],
  [1, 1]
] as const;
const RING_6 = [
  [1, 0],
  [1, 1],
  [0, 1],
  [-1, 0],
  [-1, -1],
  [0, -1]
] as const;
