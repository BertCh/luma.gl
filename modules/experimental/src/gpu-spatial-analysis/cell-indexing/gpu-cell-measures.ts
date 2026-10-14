// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {dggs} from '@luma.gl/shadertools';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {GPU_EARTH_AUTHALIC_RADIUS_METERS} from '../../analysis-core/contracts/earth';
import {
  getGPUAreaUnit,
  type GPUMeasureDescription,
  type GPUSpatialUnit
} from '../../analysis-core/contracts/spatial-context';
import {CELL_KEY_WGSL} from '../cell-aggregation/cell-keys';
import type {GPUCellWordOrder} from '../cell-aggregation/index';
import {H3_BOUNDARY_WGSL} from './h3-boundary-wgsl';
import {H3_MEASURES_WGSL} from './h3-measures-wgsl';

const OPERATION = 'GPUCellMeasures';

/** Most boundary edges an H3 cell has (hexagons 6, pentagons 5, distortion vertices up to 10). */
export const GPU_CELL_MEASURES_H3_MAXIMUM_EDGE_COUNT = 10;

/** Caller-owned outputs of {@link GPUCellMeasures}, all row-aligned with the input cells. */
export type GPUCellMeasuresOutput = {
  /**
   * One `float32` area per cell: square meters by default, square kilometers with `units:
   * 'kilometers'` (H3 `cellAreaM2` / `cellAreaKm2`); 0 for invalid cells.
   */
  areas?: GraphDataView<'float32'>;
  /** One `float32` perimeter per cell in meters or kilometers (sum of `edgeLengths`); 0 for invalid cells. */
  perimeters?: GraphDataView<'float32'>;
  /**
   * `rows * maximumEdgeCount` `float32` great-circle edge lengths in meters (or kilometers). Cell `row` owns
   * entries `[row * maximumEdgeCount, (row + 1) * maximumEdgeCount)`; entry `i` joins vertex `i` to
   * vertex `i + 1` (cyclic) of H3 `cellToBoundary`. Entries past the cell's edge count are zero.
   */
  edgeLengths?: GraphDataView<'float32'>;
  /**
   * One `uint32` boundary edge count per cell (6, 5 for pentagons, up to 10 with distortion
   * vertices; 0 for invalid cells), before any truncation to `maximumEdgeCount`.
   */
  edgeCounts?: GraphDataView<'uint32'>;
  /** One `uint32` per cell: 1 for the 12 pentagons of the resolution, 0 for hexagons and invalid cells. */
  pentagons?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUCellMeasures}.
 *
 * Per-frame (no recompile): the contents of `cells`. Topology (needs a new graph): `family`,
 * `wordOrder`, `maximumEdgeCount`, the row count and which outputs are present.
 */
export type GPUCellMeasuresProps = {
  /** Prefix for generated node IDs. Defaults to `'cell-measures'`. */
  id?: string;
  /** Cell family. Only `'h3'` is supported. */
  family: 'h3';
  /** One 64-bit H3 cell per row as two `uint32` words. */
  cells: GraphDataView<'uint32x2'>;
  /** Word order of `cells`. Defaults to `'little-endian'` (`(low, high)`, Arrow layout). */
  wordOrder?: GPUCellWordOrder;
  /**
   * Fixed `edgeLengths` stride per cell, in `[6, GPU_CELL_MEASURES_H3_MAXIMUM_EDGE_COUNT]`.
   * Defaults to 6. A smaller stride truncates the up to 10 edges of cells with distortion vertices.
   */
  maximumEdgeCount?: number;
  /**
   * Unit of `areas`, `perimeters` and `edgeLengths`: `'meters'` (default; square meters for areas)
   * or `'kilometers'` (square kilometers). Before this option the outputs were always kilometers.
   */
  units?: Exclude<GPUSpatialUnit, 'native'>;
  /** Caller-owned outputs; at least one. */
  output: GPUCellMeasuresOutput;
};

/**
 * Exact spherical area, perimeter, edge lengths and pentagon flag of H3 cells (`cellAreaM2`,
 * `edgeLengthM`, `isPentagon`). Outputs are meters and square meters unless `units: 'kilometers'`;
 * `measures` describes the model (exact spherical polygon areas with great-circle edges on the
 * WGS84 authalic sphere, radius {@link GPU_EARTH_AUTHALIC_RADIUS_METERS}).
 *
 * One kernel; each thread decodes one cell. Areas are the sum of the spherical triangles between
 * the cell center and consecutive boundary vertices, as H3 `cellAreaRads2` defines them, scaled by
 * the squared authalic Earth radius (multiply by `(radius / GPU_EARTH_AUTHALIC_RADIUS_METERS) ** 2`
 * for another sphere).
 *
 * Cell identifiers carry no coordinates, so this class takes no `spatialContext`; its measure
 * model is the discrete-grid H3 sphere, reported in `measures`.
 *
 * Precision, from `H3_MEASURES_WGSL`: hexagons whose boundary stays on the center's icosahedron
 * face (all but the 12 pentagons and the thin set of cells straddling an icosahedron edge) are
 * measured from exact integer lattice offsets and stay within 1e-6 relative error at every
 * resolution 0-15. Pentagons and edge-straddling cells fall back to f32 boundary unit vectors,
 * whose relative error is 6e-7 at resolution 0, 2e-5 at 2, 2.5e-4 at 4 and about 1e-2 at 6,
 * growing about 40x per two resolutions: treat them as unreliable beyond resolution 5. Pentagon
 * flags and edge counts are exact. Invalid and zero cells produce all-zero rows.
 */
export class GPUCellMeasures implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUCellMeasuresProps;
  /** `edgeLengths` stride per cell. */
  readonly maximumEdgeCount: number;
  /** Resolved output unit. */
  readonly units: 'meters' | 'kilometers';
  /** Machine-readable length and area models and units of the outputs. */
  readonly measures: GPUMeasureDescription;

  constructor(props: GPUCellMeasuresProps) {
    this.id = props.id ?? 'cell-measures';
    this.props = props;
    const id = this.id;
    if (props.family !== 'h3') {
      throw new Error(`${id} family must be 'h3'`);
    }
    if (props.wordOrder && props.wordOrder !== 'little-endian' && props.wordOrder !== 'high-low') {
      throw new Error(`${id} wordOrder must be 'little-endian' or 'high-low'`);
    }
    this.units = props.units ?? 'meters';
    if (this.units !== 'meters' && this.units !== 'kilometers') {
      throw new Error(`${id} units must be 'meters' or 'kilometers'`);
    }
    this.measures = {
      lengthModel: 'great-circle',
      areaModel: 'sphere-polygon',
      lengthUnits: this.units,
      areaUnits: getGPUAreaUnit(this.units)
    };
    this.maximumEdgeCount = props.maximumEdgeCount ?? 6;
    if (
      !Number.isInteger(this.maximumEdgeCount) ||
      this.maximumEdgeCount < 6 ||
      this.maximumEdgeCount > GPU_CELL_MEASURES_H3_MAXIMUM_EDGE_COUNT
    ) {
      throw new Error(
        `${id} maximumEdgeCount must be an integer in [6, ${GPU_CELL_MEASURES_H3_MAXIMUM_EDGE_COUNT}]`
      );
    }
    const {output} = props;
    const outputViews = [
      ['output.areas', output.areas],
      ['output.perimeters', output.perimeters],
      ['output.edgeLengths', output.edgeLengths],
      ['output.edgeCounts', output.edgeCounts],
      ['output.pentagons', output.pentagons]
    ] as const;
    validatePackedView(props.cells, ['uint32x2'], `${id} cells`);
    const rows = props.cells.length;
    if (rows < 1) {
      throw new Error(`${id} needs at least one row`);
    }
    if (!outputViews.some(([, view]) => view)) {
      throw new Error(`${id} needs at least one output`);
    }
    for (const [name, view] of [
      ['output.areas', output.areas],
      ['output.perimeters', output.perimeters]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
        if (view.length !== rows) {
          throw new Error(`${id} ${name} length must equal the row count`);
        }
      }
    }
    if (output.edgeLengths) {
      validatePackedView(output.edgeLengths, ['float32'], `${id} output.edgeLengths`);
      if (output.edgeLengths.length !== rows * this.maximumEdgeCount) {
        throw new Error(
          `${id} output.edgeLengths length must equal rows * maximumEdgeCount (${rows * this.maximumEdgeCount})`
        );
      }
    }
    for (const [name, view] of [
      ['output.edgeCounts', output.edgeCounts],
      ['output.pentagons', output.pentagons]
    ] as const) {
      if (view) {
        validatePackedView(view, ['uint32'], `${id} ${name}`);
        if (view.length !== rows) {
          throw new Error(`${id} ${name} length must equal the row count`);
        }
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      outputViews.map(([, view]) => view),
      [props.cells]
    );
  }

  /** Returns the single measures node `${id}-measures`. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, maximumEdgeCount} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [props.cells, ...Object.values(output)]);
    const bindings: WGSLKernelBinding[] = [
      {name: 'cells', view: props.cells, type: 'u32', access: 'read'}
    ];
    const addOutput = (name: string, view: GraphDataView | undefined, type: 'f32' | 'u32') => {
      if (view) {
        bindings.push({name, view, type, access: 'read_write'});
      }
    };
    addOutput('areasOut', output.areas, 'f32');
    addOutput('perimetersOut', output.perimeters, 'f32');
    addOutput('edgeLengthsOut', output.edgeLengths, 'f32');
    addOutput('edgeCountsOut', output.edgeCounts, 'u32');
    addOutput('pentagonsOut', output.pentagons, 'u32');

    const needsMeasures = Boolean(
      output.areas || output.perimeters || output.edgeLengths || output.edgeCounts
    );
    const body = `let words = vec2u(cells[cellsOffset + 2u * index], cells[cellsOffset + 2u * index + 1u]);
  let key = ${props.wordOrder === 'high-low' ? 'words' : 'words.yx'};
  ${output.pentagons ? 'pentagonsOut[pentagonsOutOffset + index] = select(0u, 1u, cellMeasuresIsPentagon(key));' : ''}
  ${
    needsMeasures
      ? `let measures = cellMeasuresGetH3(key);
  ${output.areas ? 'areasOut[areasOutOffset + index] = measures.area * EARTH_RADIUS * EARTH_RADIUS;' : ''}
  ${output.perimeters ? 'perimetersOut[perimetersOutOffset + index] = measures.perimeter * EARTH_RADIUS;' : ''}
  ${output.edgeCounts ? 'edgeCountsOut[edgeCountsOutOffset + index] = measures.count;' : ''}
  ${
    output.edgeLengths
      ? `for (var edge = 0u; edge < MAXIMUM_EDGE_COUNT; edge++) {
    edgeLengthsOut[edgeLengthsOutOffset + index * MAXIMUM_EDGE_COUNT + edge] =
      select(0.0, measures.edges[edge] * EARTH_RADIUS, edge < measures.count);
  }`
      : ''
  }`
      : ''
  }`;
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-measures`,
        operation: OPERATION,
        variant: 'h3',
        bindings,
        invocationCount: props.cells.length,
        declarations: `const MAXIMUM_EDGE_COUNT: u32 = ${maximumEdgeCount}u;
const EARTH_RADIUS: f32 = ${getWGSLFloatLiteral(this.units === 'kilometers' ? GPU_EARTH_AUTHALIC_RADIUS_METERS / 1000 : GPU_EARTH_AUTHALIC_RADIUS_METERS)};
${dggs.source}
${CELL_KEY_WGSL}
${H3_BOUNDARY_WGSL}
${H3_MEASURES_WGSL}`,
        body
      })
    ];
  }
}
