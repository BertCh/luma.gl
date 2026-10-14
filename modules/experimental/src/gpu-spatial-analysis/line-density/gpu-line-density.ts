// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPULineGeometryFields,
  GPUOverflowStatusPort
} from '../../analysis-core/contracts/index';
import {
  getGPUAuthalicSphereRadius,
  getGPUSphereRadius,
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContextSupport
} from '../../analysis-core/contracts/spatial-context';
import {createSaturatedScanTotalNodes} from '../../utils/scan-total';
import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {getSortedSegmentSumNodes} from '../../utils/sorted-segment-sums';
import {createFillNode} from '../../utils/wgsl-kernel-nodes';
import {
  defineGPUSpatialParameterSchema,
  packGPUSpatialParameterValues,
  type GPUSpatialContext
} from '../../analysis-core/contracts/index';
import {
  createDensityNode,
  createRecordPublishNode,
  createWalkCountNode,
  createWalkEmitNode,
  type LineDensityGridConstants
} from './line-density-kernels';

const OPERATION = 'GPULineDensity';
const LINE_DENSITY_SUPPORT: GPUSpatialContextSupport = Object.freeze({
  ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  measure: 'area'
});

/** Number of float32 elements in a {@link GPULineDensity} parameter buffer. */
export const GPU_LINE_DENSITY_PARAMETER_LENGTH = 4;

/**
 * Largest `columns * rows` of {@link GPULineDensity}: the per-cell sum runs one 256-thread
 * workgroup per cell and the dispatch holds at most `2^32 - 1` invocations, so
 * `floor((2^32 - 1) / 256)`, `2^24 - 1`.
 */
export const GPU_LINE_DENSITY_MAXIMUM_CELL_COUNT = Math.floor(0xffffffff / 256);

/** Declarative layout of the per-encoding line-density grid. */
export const GPU_LINE_DENSITY_PARAMETER_SCHEMA = defineGPUSpatialParameterSchema({
  id: 'line-density',
  format: 'float32',
  wordLength: GPU_LINE_DENSITY_PARAMETER_LENGTH,
  fields: [
    {name: 'minimumX', format: 'float32', wordOffset: 0, defaultValue: 0, dynamic: true},
    {name: 'minimumY', format: 'float32', wordOffset: 1, defaultValue: 0, dynamic: true},
    {
      name: 'cellWidth',
      format: 'float32',
      wordOffset: 2,
      defaultValue: 1,
      minimum: Number.MIN_VALUE,
      units: 'position-units',
      dynamic: true
    },
    {
      name: 'cellHeight',
      format: 'float32',
      wordOffset: 3,
      defaultValue: 1,
      minimum: Number.MIN_VALUE,
      units: 'position-units',
      dynamic: true
    }
  ]
});

/** CPU description of the per-frame grid of {@link GPULineDensity}. */
export type GPULineDensitySettings = {
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
 * Packs {@link GPULineDensitySettings} into the 4-element float32 layout
 * `[minX, minY, cellWidth, cellHeight]`. Write the result into a `GPUParameterBuffer` between
 * encodings to pan or zoom the grid without recompiling.
 *
 * @param settings Settings to encode.
 * @param target Optional destination of at least 4 elements.
 * @throws If a value is not finite, a cell size is not positive, or `target` is too short.
 */
export function getGPULineDensityParameterValues(
  settings: GPULineDensitySettings,
  target: Float32Array = new Float32Array(GPU_LINE_DENSITY_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_LINE_DENSITY_PARAMETER_LENGTH) {
    throw new Error(`Line density target must hold ${GPU_LINE_DENSITY_PARAMETER_LENGTH} elements`);
  }
  target.set(
    packGPUSpatialParameterValues(GPU_LINE_DENSITY_PARAMETER_SCHEMA, {
      minimumX: settings.minX,
      minimumY: settings.minY,
      cellWidth: settings.cellWidth,
      cellHeight: settings.cellHeight
    })
  );
  return target;
}

/** Caller-owned outputs of {@link GPULineDensity}. */
export type GPULineDensityOutput = {
  /**
   * Total line length inside each cell, `columns * rows` rows in row-major order (cell
   * `row * columns + column`), in the context's length unit (position units for planar,
   * meters by default for longitude/latitude).
   */
  lengths: GraphDataView<'float32'>;
  /**
   * Optional length divided by cell area, in the context's length unit per square length unit,
   * same layout. Spherical cells use the exact spherical area with the authalic radius by default.
   */
  densities?: GraphDataView<'float32'>;
  /**
   * One-row status scalars: `overflow` receives `1` when the piece capacity overflowed (lengths
   * are then low), else `0`, and the optional `requiredCount` receives the unclamped number of
   * segment-cell pieces.
   */
  status: GPUOverflowStatusPort & {requiredCount?: GraphDataView<'uint32'>};
};

/**
 * Properties for {@link GPULineDensity}.
 *
 * Per-frame (no recompile): the grid origin and cell size in `parameters`, and every input
 * buffer. Compile-time: view lengths, `columns`, `rows`, `spatialContext`,
 * `recordCapacity`, and which outputs are present.
 */
export type GPULineDensityProps = GPULineGeometryFields & {
  /** Prefix for generated node and transient IDs. Defaults to `'line-density'`. */
  id?: string;
  /**
   * Native planar when omitted. A `longitude-latitude` context with metric `'great-circle'`
   * measures lengths in the context's length unit and areas in its square (meters by default).
   */
  spatialContext?: GPUSpatialContext;
  /** Grid columns (compile time). `columns * rows` is at most {@link GPU_LINE_DENSITY_MAXIMUM_CELL_COUNT}. */
  columns: number;
  /** Grid rows (compile time). `columns * rows` is at most {@link GPU_LINE_DENSITY_MAXIMUM_CELL_COUNT}. */
  rows: number;
  /**
   * Compile-time capacity of segment-cell pieces. A segment crossing `k` grid lines produces
   * `k + 1` pieces. Default `max(1024, 4 * positions.length)`. When exceeded, `status.overflow` is set
   * and the cells fed by the dropped pieces (the last segments in row order) are low.
   */
  recordCapacity?: number;
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
 * grid contribute nothing; consecutive lines are not joined, and only segments inside a line are
 * counted (single-vertex lines have length 0).
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
  /** Canonical coordinate and metric contract. */
  readonly spatialContext: GPUSpatialContext;
  /** Number of grid cells. */
  readonly cellCount: number;
  /** Resolved coordinate system. */
  readonly coordinateSystem: 'planar' | 'spherical';
  /** Mean sphere radius for lengths, in the context's length unit; 1 for planar. */
  readonly radius: number;
  /** Authalic sphere radius for cell areas, in the context's length unit; 1 for planar. */
  readonly areaRadius: number;
  /** Resolved piece capacity. */
  readonly recordCapacity: number;

  constructor(props: GPULineDensityProps) {
    this.id = props.id ?? 'line-density';
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext,
      LINE_DENSITY_SUPPORT
    );
    const spherical = this.spatialContext.coordinateSpace === 'longitude-latitude';
    this.coordinateSystem = spherical ? 'spherical' : 'planar';
    this.radius = spherical ? getGPUSphereRadius(this.spatialContext) : 1;
    this.areaRadius = spherical ? getGPUAuthalicSphereRadius(this.spatialContext) : 1;
    this.recordCapacity = props.recordCapacity ?? Math.max(1024, 4 * props.positions.length);
    const {id} = this;
    validateCapacity(id, 'columns', props.columns);
    validateCapacity(id, 'rows', props.rows);
    this.cellCount = props.columns * props.rows;
    validateMaximumCount(id, 'columns x rows', this.cellCount, GPU_LINE_DENSITY_MAXIMUM_CELL_COUNT);
    validateCapacity(id, 'recordCapacity', this.recordCapacity);
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (props.positions.length < 2) {
      throw new Error(`${id} needs two positions`);
    }
    validatePackedView(props.lineOffsets, ['uint32'], `${id} lineOffsets`);
    if (props.lineOffsets.length < 2) {
      throw new Error(`${id} lineOffsets needs two rows`);
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_LINE_DENSITY_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_LINE_DENSITY_PARAMETER_LENGTH} float32 values`
      );
    }
    const {lengths, densities, status} = props.output;
    const {overflow, requiredCount} = status;
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
      ['requiredCount', requiredCount]
    ] as const) {
      if (scalar) {
        validatePackedView(scalar, ['uint32'], `${id} output.status.${name}`);
        if (scalar.length < 1) {
          throw new Error(`${id} output.status.${name} must contain one uint32 row`);
        }
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [lengths, densities, overflow, requiredCount],
      [props.positions, props.lineOffsets, props.parameters]
    );
  }

  /**
   * Returns the count, scan, key fill, emit, publish, cell count, sorted sum and optional density
   * nodes.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, cellCount, recordCapacity} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.lineOffsets,
      props.parameters,
      output.lengths,
      output.densities,
      output.status.overflow,
      output.status.requiredCount
    ]);
    const rowCount = props.positions.length;
    const constants: LineDensityGridConstants = {
      positions: props.positions,
      lineOffsets: props.lineOffsets,
      columns: props.columns,
      rows: props.rows,
      coordinateSystem: this.coordinateSystem,
      radius: this.radius,
      capacity: recordCapacity
    };
    const counts = createTransientView(graph, `${id}-counts`, 'uint32', rowCount);
    const starts = createTransientView(graph, `${id}-starts`, 'uint32', rowCount);
    const scanTotal = createTransientView(graph, `${id}-scan-total`, 'uint32', 1);
    const keys = createTransientView(graph, `${id}-keys`, 'uint32', recordCapacity);
    const contributions = createTransientView(
      graph,
      `${id}-contributions`,
      'float32',
      recordCapacity
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
      ...createSaturatedScanTotalNodes<Parameters>(graph, {
        id: id,
        operation: OPERATION,
        counts: counts,
        offsets: starts,
        rowCount: rowCount,
        total: scanTotal
      }),
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
        capacity: recordCapacity,
        total: scanTotal,
        overflow: output.status.overflow,
        totalRecords: output.status.requiredCount
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
              areaRadius: this.areaRadius,
              parameters: props.parameters,
              lengths: output.lengths,
              densities: output.densities
            })
          ]
        : [])
    ];
  }
}
