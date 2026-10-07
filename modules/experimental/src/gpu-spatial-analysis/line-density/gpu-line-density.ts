// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  GraphVectorView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {getSortedSegmentSumNodes} from '../../utils/sorted-segment-sums';
import {createFillNode} from '../../utils/wgsl-kernel-nodes';
import {GPU_GEODESIC_MEAN_EARTH_RADIUS} from '../geometry-measures/geodesic-wgsl';
import {
  createDensityNode,
  createRecordPublishNode,
  createWalkCountNode,
  createWalkEmitNode,
  type LineDensityGridConstants
} from './line-density-kernels';

const OPERATION = 'GPULineDensity';

/** Number of float32 elements in a {@link GPULineDensity} parameter buffer. */
export const GPU_LINE_DENSITY_PARAMETER_LENGTH = 4;

/** CPU description of the per-frame grid of {@link GPULineDensity}. */
export type GPULineDensityParameters = {
  /** X of the grid's lower-left corner. */
  minX: number;
  /** Y of the grid's lower-left corner. */
  minY: number;
  /** Cell width in position units (degrees for `'spherical'`). */
  cellWidth: number;
  /** Cell height in position units (degrees for `'spherical'`). */
  cellHeight: number;
};

/**
 * Packs {@link GPULineDensityParameters} into the 4-element float32 layout
 * `[minX, minY, cellWidth, cellHeight]`. Write the result into a `GPUParameterBuffer` between
 * encodings to pan or zoom the grid without recompiling.
 *
 * @param parameters Parameters to encode.
 * @param target Optional destination of at least 4 elements.
 * @throws If a value is not finite, a cell size is not positive, or `target` is too short.
 */
export function getGPULineDensityParameterValues(
  parameters: GPULineDensityParameters,
  target: Float32Array = new Float32Array(GPU_LINE_DENSITY_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_LINE_DENSITY_PARAMETER_LENGTH) {
    throw new Error(`Line density target must hold ${GPU_LINE_DENSITY_PARAMETER_LENGTH} elements`);
  }
  const {minX, minY, cellWidth, cellHeight} = parameters;
  if (![minX, minY, cellWidth, cellHeight].every(Number.isFinite)) {
    throw new Error('Line density parameters must be finite');
  }
  if (cellWidth <= 0 || cellHeight <= 0) {
    throw new Error('Line density cellWidth and cellHeight must be positive');
  }
  target.set([minX, minY, cellWidth, cellHeight]);
  return target;
}

/** Caller-owned outputs of {@link GPULineDensity}. */
export type GPULineDensityOutput = {
  /**
   * Total line length inside each cell, `columns * rows` rows in row-major order (cell
   * `row * columns + column`), in position units (`'planar'`) or sphere-radius units
   * (`'spherical'`, meters by default).
   */
  lengths: GraphDataView<'float32'>;
  /**
   * Optional length divided by cell area (`1 / unit`), same layout. Spherical cells use the exact
   * spherical area of the longitude/latitude cell.
   */
  densities?: GraphDataView<'float32'>;
  /** One-row scalar receiving `1` when the piece capacity overflowed (lengths are then low), else `0`. */
  overflow: GraphDataView<'uint32'>;
  /** Optional one-row scalar receiving the unclamped number of segment-cell pieces. */
  totalRecords?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPULineDensity}.
 *
 * Per-frame (no recompile): the grid origin and cell size in `parameters`, and every input
 * buffer. Compile-time: view lengths, `columns`, `rows`, `coordinateSystem`, `radius`,
 * `maximumRecords`, and which outputs are present.
 */
export type GPULineDensityProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'line-density'`. */
  id?: string;
  /** Packed positions sorted by path: planar coordinates, or longitude/latitude degrees. */
  positions: GraphDataView<'float32x2'>;
  /** `pathCount + 1` monotonic row offsets; path `p` owns rows `[pathOffsets[p], pathOffsets[p + 1])`. */
  pathOffsets: GraphDataView<'uint32'>;
  /** Grid columns (compile time). */
  columns: number;
  /** Grid rows (compile time). */
  rows: number;
  /**
   * `'planar'` (default): Euclidean lengths. `'spherical'`: positions and the grid are in
   * longitude/latitude degrees, segments are straight in longitude/latitude, and each clipped
   * piece is measured as the great-circle distance between its endpoints.
   */
  coordinateSystem?: 'planar' | 'spherical';
  /** Sphere radius of `'spherical'`. Defaults to {@link GPU_GEODESIC_MEAN_EARTH_RADIUS}. */
  radius?: number;
  /**
   * Compile-time capacity of segment-cell pieces. A segment crossing `k` grid lines produces
   * `k + 1` pieces. Default `max(1024, 4 * positions.length)`. When exceeded, `overflow` is set
   * and the cells fed by the dropped pieces (the last segments in row order) are low.
   */
  maximumRecords?: number;
  /** Per-frame packed float32 view written with {@link getGPULineDensityParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /** Output columns. */
  output: GPULineDensityOutput;
};

/**
 * Line length per grid cell and line density (QGIS "Line density", "Sum line lengths" per grid
 * cell, PostGIS `ST_Length(ST_Intersection(line, cell))` summed).
 *
 * Every segment is clipped to the grid with Liang-Barsky and walked through the cells it crosses
 * (Amanatides-Woo with integer step counts), emitting one `(cell, length)` piece per crossed
 * cell; pieces sum to the clipped length exactly up to f32 rounding. Pieces are counted, scanned
 * into slots and emitted (the walk code is shared, so count and emit agree), then reduced per
 * cell by a stable sort and a fixed-order segmented sum, so lengths are bitwise reproducible. No
 * float atomics. A segment lying exactly along a cell boundary counts in the cell above or to the
 * right, and one on the grid's upper or right edge in the last row or column. Segments outside the
 * grid contribute nothing; consecutive paths are not joined, and only segments inside a path are
 * counted (single-vertex paths have length 0).
 *
 * For per-polygon line length (QGIS "Sum line lengths") use {@link GPULineLengthPerPolygon}.
 *
 * Precision: f32 grid coordinates relative to the grid origin; piece lengths come from f32
 * parameters along the segment (relative error about `1e-6` of the segment length).
 */
export class GPULineDensity implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPULineDensityProps;
  /** Number of grid cells. */
  readonly cellCount: number;
  /** Resolved coordinate system. */
  readonly coordinateSystem: 'planar' | 'spherical';
  /** Resolved sphere radius. */
  readonly radius: number;
  /** Resolved piece capacity. */
  readonly maximumRecords: number;

  constructor(props: GPULineDensityProps) {
    this.id = props.id ?? 'line-density';
    this.props = props;
    this.coordinateSystem = props.coordinateSystem ?? 'planar';
    this.radius = props.radius ?? GPU_GEODESIC_MEAN_EARTH_RADIUS;
    this.maximumRecords = props.maximumRecords ?? Math.max(1024, 4 * props.positions.length);
    const {id} = this;
    for (const [name, view] of Object.entries({
      positions: props.positions,
      pathOffsets: props.pathOffsets,
      parameters: props.parameters,
      ...props.output
    })) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    for (const [name, value] of [
      ['columns', props.columns],
      ['rows', props.rows]
    ] as const) {
      if (!Number.isSafeInteger(value) || value < 1) {
        throw new Error(`${id} ${name} must be a positive integer`);
      }
    }
    this.cellCount = props.columns * props.rows;
    if (this.cellCount > 0x7fffffff) {
      throw new Error(`${id} grid is too large`);
    }
    if (this.coordinateSystem !== 'planar' && this.coordinateSystem !== 'spherical') {
      throw new Error(`${id} coordinateSystem must be 'planar' or 'spherical'`);
    }
    if (!Number.isFinite(this.radius) || this.radius <= 0) {
      throw new Error(`${id} radius must be a positive finite number`);
    }
    if (!Number.isSafeInteger(this.maximumRecords) || this.maximumRecords < 1) {
      throw new Error(`${id} maximumRecords must be a positive integer`);
    }
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (props.positions.length < 2) {
      throw new Error(`${id} needs at least two positions`);
    }
    validatePackedUint32View(props.pathOffsets, `${id} pathOffsets`);
    if (props.pathOffsets.length < 2) {
      throw new Error(`${id} pathOffsets must contain at least two rows`);
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_LINE_DENSITY_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_LINE_DENSITY_PARAMETER_LENGTH} float32 values`
      );
    }
    const {lengths, densities, overflow, totalRecords} = props.output;
    validatePackedView(lengths, ['float32'], `${id} output.lengths`);
    if (lengths.length !== this.cellCount) {
      throw new Error(`${id} output.lengths must hold ${this.cellCount} rows`);
    }
    if (densities) {
      validatePackedView(densities, ['float32'], `${id} output.densities`);
      if (densities.length !== this.cellCount) {
        throw new Error(`${id} output.densities must hold ${this.cellCount} rows`);
      }
    }
    for (const [name, scalar] of [
      ['overflow', overflow],
      ['totalRecords', totalRecords]
    ] as const) {
      if (scalar) {
        validatePackedUint32View(scalar, `${id} output.${name}`);
        if (scalar.length < 1) {
          throw new Error(`${id} output.${name} must contain one uint32 row`);
        }
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [lengths, densities, overflow, totalRecords],
      [props.positions, props.pathOffsets, props.parameters]
    );
  }

  /**
   * Returns the count, scan, key fill, emit, publish, cell count, sorted sum and optional density
   * nodes.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, cellCount, maximumRecords} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.pathOffsets,
      props.parameters,
      output.lengths,
      output.densities,
      output.overflow,
      output.totalRecords
    ]);
    const rowCount = props.positions.length;
    const constants: LineDensityGridConstants = {
      positions: props.positions,
      pathOffsets: props.pathOffsets,
      columns: props.columns,
      rows: props.rows,
      coordinateSystem: this.coordinateSystem,
      radius: this.radius,
      capacity: maximumRecords
    };
    const counts = createTransientView(graph, `${id}-counts`, 'uint32', rowCount);
    const starts = createTransientView(graph, `${id}-starts`, 'uint32', rowCount);
    const keys = createTransientView(graph, `${id}-keys`, 'uint32', maximumRecords);
    const contributions = createTransientView(
      graph,
      `${id}-contributions`,
      'float32',
      maximumRecords
    );
    return [
      createWalkCountNode<Parameters>(graph, {
        id: `${id}-count`,
        operation: OPERATION,
        constants,
        parameters: props.parameters,
        counts
      }),
      ...new GPUScan({
        id: `${id}-scan`,
        input: counts,
        output: starts,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      // Unwritten slots keep the out-of-range key, which sorts after every cell.
      createFillNode<Parameters>(graph, {
        id: `${id}-fill-keys`,
        operation: OPERATION,
        view: keys,
        type: 'u32',
        value: `${cellCount}u`
      }),
      createWalkEmitNode<Parameters>(graph, {
        id: `${id}-emit`,
        operation: OPERATION,
        constants,
        parameters: props.parameters,
        starts,
        keys,
        contributions
      }),
      createRecordPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        capacity: maximumRecords,
        starts,
        counts,
        overflow: output.overflow,
        totalRecords: output.totalRecords
      }),
      ...getSortedSegmentSumNodes<Parameters>(graph, {
        id: `${id}-sum`,
        operation: OPERATION,
        segmentCount: cellCount,
        segmentKeys: keys,
        sumContributions: contributions,
        sums: output.lengths
      }),
      ...(output.densities
        ? [
            createDensityNode<Parameters>(graph, {
              id: `${id}-density`,
              operation: OPERATION,
              columns: props.columns,
              rows: props.rows,
              coordinateSystem: this.coordinateSystem,
              radius: this.radius,
              parameters: props.parameters,
              lengths: output.lengths,
              densities: output.densities
            })
          ]
        : [])
    ];
  }
}
