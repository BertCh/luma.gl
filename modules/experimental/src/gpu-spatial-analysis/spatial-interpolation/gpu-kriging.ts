// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUGridIndex,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {GET_NAN_WGSL} from '../../utils/wgsl-common';
import {validateGPUStatusPort, type GPUSpatialContext} from '../../analysis-core/contracts/index';
import {
  getGPUGridIndexMaximumCellCount,
  validateCapacity,
  validateMaximumCount,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {validateRasterGrid} from '../../gpu-raster/raster-band-utils';
import type {GPURasterMetadata} from '../../gpu-raster/types';
import {GRID_COORDINATE_WGSL} from './gpu-inverse-distance-weighting';
import {getRectangleScanWGSL, getRingScanWGSL} from './grid-neighbor-scan';
import {
  getSeparationWGSL,
  resolveInterpolationSeparation,
  type InterpolationSeparation
} from './separation-wgsl';
import {GPU_KRIGING_PARAMETER_LENGTH} from './kriging-parameters';

const OPERATION = 'GPUKriging';
/** Largest compile-time neighborhood; the per-cell system is `(k + 1) x (k + 1)`. */
const MAXIMUM_NEIGHBOR_CAPACITY = 16;

/**
 * A pivot at or below this fraction of the largest entry of the original system is treated as
 * zero. The normalized system has entries in `[0, 1]` with a border of ones, so the scale is 1 and
 * the bound is about 10 f32 ulps, the rounding noise of one elimination column.
 */
export const KRIGING_RELATIVE_PIVOT_TOLERANCE = 1e-6;
/**
 * A system whose smallest elimination pivot is below this fraction of its largest is rejected as
 * ill conditioned. The ratio tracks `1 / cond` to within a small factor, and an f32 solve loses
 * about `cond * 6e-8` of relative accuracy in the weights. At `1e-4` the weights are good to
 * about `1e-3`; measured on emulated f32 solves, a Gaussian model with nugget 0.001 and 12
 * clustered samples sits at 3e-5 with weight errors of 5e-2, and well spaced neighborhoods sit
 * above 2e-2.
 */
export const KRIGING_PIVOT_RATIO_TOLERANCE = 1e-4;
/**
 * Kriging variance is `mu + sum(w * gamma)` in units of the total sill and is never negative for a
 * valid model. Negatives above `-1e-4` are f32 rounding of terms of order 1 and are clamped to
 * zero; anything below is a numerically unreliable solve and yields nodata.
 */
export const KRIGING_NEGATIVE_VARIANCE_TOLERANCE = 1e-4;

/** Per-cell status codes written by the solve kernel. */
const STATUS_OK = 0;
const STATUS_INVALID = 1;
const STATUS_UNCERTAIN = 2;

/**
 * Caller-owned raster outputs of {@link GPUKriging}.
 *
 * `uncertainCount` and `invalidCount` are single `uint32` rows that the node zeroes and then
 * increments once per output cell, so they describe the whole raster of the last encoding.
 */
export type GPUKrigingOutput = {
  /**
   * Predicted value per output cell, at least `grid.width * grid.height` float32 rows, row
   * `y * grid.width + x` in the grid's row order (row 0 is the grid's first row: the top row of a
   * north-up grid, `minY` when rows advance north). Nodata cells (too few distinct neighbors, an ill-conditioned system, an unreliable
   * variance) are NaN.
   */
  values: GraphDataView<'float32'>;
  /**
   * Optional ordinary kriging variance per cell, same layout, in squared value units. Zero at
   * exact hits, NaN at nodata cells. Tiny negative values within f32 rounding of the normalized
   * system (above `-KRIGING_NEGATIVE_VARIANCE_TOLERANCE` of the total sill) are clamped to zero;
   * a larger negative variance makes the cell nodata and counts it in `uncertainCount`.
   */
  variance?: GraphDataView<'float32'>;
  /**
   * Optional leave-one-out cross-validation prediction per SAMPLE row (at least `sampleCount`
   * float32 rows): each sample is kriged from its `k` nearest OTHER samples with the same
   * per-frame variogram, radius and neighborhood settings as the raster. NaN where the sample
   * is NaN or masked, or the neighborhood is insufficient or ill conditioned. Another sample at
   * exactly the same position counts as an exact hit and is returned with variance zero. The
   * standardized error is `(values[i] - prediction[i]) / sqrt(variance[i])`.
   */
  crossValidationPrediction?: GraphDataView<'float32'>;
  /** Optional leave-one-out kriging variance per sample row, same layout. */
  crossValidationVariance?: GraphDataView<'float32'>;
  /** Optional diagnostic counts. */
  status?: {
    /**
     * Optional count of output cells that are nodata because the settings were invalid or fewer
     * than `minimumNeighborCount` distinct sample positions fell in the neighborhood
     * (`GPUInvalidStatusPort`). Cells outside the sample domain are included.
     */
    invalidCount?: GraphDataView<'uint32'>;
    /**
     * Optional count of output cells made nodata because the kriging system was ill conditioned
     * (pivot ratio below `KRIGING_PIVOT_RATIO_TOLERANCE`) or its variance came out materially
     * negative (`GPUUncertaintyStatusPort`).
     */
    uncertainCount?: GraphDataView<'uint32'>;
  };
};

/**
 * Properties for {@link GPUKriging}.
 *
 * Per-frame (no recompile): the contents of `parameters` (radius, neighborhood size,
 * variogram model and parameters) and of every input buffer. Topology (needs a new graph): the
 * `grid` size, sample count, `indexGridSize`, `indexBounds`, `maximumNeighborCount`, and whether
 * `mask` and each optional output are present.
 */
export type GPUKrigingProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'kriging'`. */
  id?: string;
  /**
   * Coordinate space and metric of `positions`. Omitted means native planar coordinates
   * (Euclidean distances; the search radius and variogram range are in position units).
   *
   * `{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}` uses haversine distances on
   * a sphere (`sphereRadius`, else the IUGG mean Earth radius): positions are absolute
   * `[longitude, latitude]` degrees, the grid index and the output grid stay in degrees, and the
   * search radius and variogram range are in meters (kilometers with `units: 'kilometers'`).
   * Neighbors are ordered by the haversine term; f32 absolute lon/lat limits accuracy to about 1 m
   * at Earth scale, and the local-origin offsets used in planar mode do not apply. The query box
   * is conservative and does not wrap the antimeridian; an unbounded radius scans every sample
   * per cell (no early stop). Other longitude/latitude metrics throw.
   */
  spatialContext?: GPUSpatialContext;
  /** Packed sample positions in the same units as the grid, radius and variogram range. */
  positions: GraphDataView<'float32x2'>;
  /** Packed float32 sample values. NaN values are skipped. */
  values: GraphDataView<'float32'>;
  /** Optional packed `uint32` sample mask; zero skips the sample. */
  mask?: GraphDataView<'uint32'>;
  /** Per-frame float32 view of at least 12 values written with `getGPUKrigingParameterValues`. */
  parameters: GraphDataView<'float32'>;
  /**
   * Output raster grid: axis-aligned, its width and height size the kernels, and its cell centers
   * are predicted. Pack the matching placement with `getGPUKrigingParameterValues`.
   */
  grid: GPURasterMetadata;
  /**
   * Dimensions `[columns, rows]` of the internal `GPUGridIndex` over the samples. At most
   * {@link getGPUGridIndexMaximumCellCount} cells (2^25 - 1 on a default device).
   */
  indexGridSize: readonly [number, number];
  /** Inclusive sample domain `[minX, minY, maxX, maxY]` of the internal grid index. */
  indexBounds: readonly [number, number, number, number];
  /** Compile-time neighborhood capacity in `[1, 16]`. Defaults to 16. */
  maximumNeighborCount?: number;
  /** Caller-owned raster outputs. */
  output: GPUKrigingOutput;
};

/**
 * Local ordinary kriging on a raster: for every output cell the `k` nearest samples within the
 * search radius form a `(k + 1) x (k + 1)` ordinary kriging system under a fitted variogram model,
 * solved in the invocation by Gaussian elimination with partial pivoting (Cressie 1993, eq.
 * 3.2.12 to 3.2.16).
 *
 * Neighbor selection keeps the `k` smallest `(d^2, row)` pairs, a total order, so results are
 * deterministic without sorting the index cells. Cells are visited in rings of growing radius
 * around the cell's own index cell and the walk stops once the `k`-th distance is within the
 * visited block (`O(k)` candidates per cell even with an unbounded radius). The variogram is
 * normalized by its total sill before solving to keep the f32 system well scaled, and the
 * prediction is accumulated relative to the nearest neighbor's value. The variogram follows
 * `fitVariogramModel`: `gamma(0) = 0`, `gamma(h) = nugget + sill * shape(h / range)`.
 *
 * Numerical policy:
 * - Coordinates are local. Distances are differences of offsets from the grid start corner (cells)
 *   or from the index domain minimum (cross-validation), so UTM-sized absolute coordinates do not
 *   quantize them to the f32 ulp at the absolute magnitude, and the exact-hit test compares
 *   offsets.
 * - Duplicate sample positions inside a neighborhood are coalesced: their values are averaged
 *   (summed in `(d^2, row)` order, so deterministic) and they form one kriging point, so repeated
 *   survey coordinates do not make the system singular. Duplicates still occupy neighbor slots, so
 *   the effective neighborhood can hold fewer than `k` points, and `minimumNeighborCount` counts
 *   distinct positions.
 * - An exact hit returns the mean of the samples at that position with variance 0. The nugget is
 *   treated as micro-scale variation (the predictor is exact at the data), not as measurement
 *   error; filtered kriging with a measurement-error nugget is not provided.
 * - The system is rejected (cell becomes NaN, counted in `uncertainCount`) when a pivot is at or
 *   below `KRIGING_RELATIVE_PIVOT_TOLERANCE` of the system scale, or when the smallest pivot is
 *   below `KRIGING_PIVOT_RATIO_TOLERANCE` of the largest (a condition estimate: the f32 solve
 *   would lose most digits of the weights). A Gaussian model with a near-zero nugget and
 *   closely spaced samples is the usual cause; add nugget or drop near-duplicate samples. The
 *   variance is clamped at zero only within `KRIGING_NEGATIVE_VARIANCE_TOLERANCE` of f32
 *   rounding; a larger negative variance also makes the cell NaN and counts it in `uncertainCount`.
 * - `invalidCount` counts cells that are NaN for lack of data or invalid settings.
 * - Optional leave-one-out cross-validation krigs every sample from its other neighbors with the
 *   same kernel code, for choosing the variogram or neighborhood.
 *
 * The kernels bind at most 8 storage buffers: a sample mask is folded into a transient value
 * copy first, and the per-cell status is written to a transient before the counters are
 * accumulated. Matches a float64 CPU oracle to f32 rounding for well-conditioned neighborhoods.
 */
export class GPUKriging implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUKrigingProps;
  /** Resolved spatial context; native planar when none was given. */
  readonly spatialContext: GPUSpatialContext;
  /** Resolved metric (planar or great-circle). */
  private readonly separation: InterpolationSeparation;
  /** `grid.width * grid.height`. */
  readonly cellCount: number;
  /** Compile-time neighborhood capacity. */
  readonly maximumNeighborCount: number;

  constructor(props: GPUKrigingProps) {
    this.id = props.id ?? 'kriging';
    this.props = props;
    const {id} = this;
    this.separation = resolveInterpolationSeparation(id, props.spatialContext);
    this.spatialContext = this.separation.context;
    this.cellCount = validateRasterGrid(id, props.grid);
    if (props.grid.affine[1] !== 0 || props.grid.affine[3] !== 0) {
      throw new Error(`${id} needs an axis-aligned grid`);
    }
    for (const [name, size] of [
      ['indexGridSize[0]', props.indexGridSize[0]],
      ['indexGridSize[1]', props.indexGridSize[1]]
    ] as const) {
      validateCapacity(id, name, size, 1, Number.MAX_SAFE_INTEGER);
    }
    const [minX, minY, maxX, maxY] = props.indexBounds;
    if (
      props.indexBounds.length !== 4 ||
      !props.indexBounds.every(Number.isFinite) ||
      minX > maxX ||
      minY > maxY
    ) {
      throw new Error(`${id} indexBounds must be finite and ordered`);
    }
    this.maximumNeighborCount = props.maximumNeighborCount ?? MAXIMUM_NEIGHBOR_CAPACITY;
    validateCapacity(
      id,
      'maximumNeighborCount',
      this.maximumNeighborCount,
      1,
      MAXIMUM_NEIGHBOR_CAPACITY
    );
    const {output} = props;
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    validatePackedView(props.values, ['float32'], `${id} values`);
    const sampleCount = props.values.length;
    if (sampleCount < 1) {
      throw new Error(`${id} needs at least one sample`);
    }
    if (props.positions.length !== sampleCount) {
      throw new Error(`${id} positions length must equal values length`);
    }
    if (props.mask) {
      validatePackedView(props.mask, ['uint32'], `${id} mask`);
      if (props.mask.length !== sampleCount) {
        throw new Error(`${id} mask length must equal values length`);
      }
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_KRIGING_PARAMETER_LENGTH) {
      throw new Error(`${id} parameters must hold ${GPU_KRIGING_PARAMETER_LENGTH} float32 values`);
    }
    validatePackedView(output.values, ['float32'], `${id} output.values`);
    if (output.values.length < this.cellCount) {
      throw new Error(`${id} output.values must hold grid.width * grid.height rows`);
    }
    if (output.variance) {
      validatePackedView(output.variance, ['float32'], `${id} output.variance`);
      if (output.variance.length < this.cellCount) {
        throw new Error(`${id} output.variance must hold grid.width * grid.height rows`);
      }
      if (output.variance.buffer === output.values.buffer) {
        throw new Error(`${id} outputs must not share buffers`);
      }
    }
    if (output.crossValidationVariance && !output.crossValidationPrediction) {
      throw new Error(
        `${id} output.crossValidationVariance needs output.crossValidationPrediction`
      );
    }
    if (output.crossValidationPrediction) {
      for (const [name, view] of [
        ['crossValidationPrediction', output.crossValidationPrediction],
        ['crossValidationVariance', output.crossValidationVariance]
      ] as const) {
        if (!view) {
          continue;
        }
        validatePackedView(view, ['float32'], `${id} output.${name}`);
        if (view.length < sampleCount) {
          throw new Error(`${id} output.${name} must hold one row per sample`);
        }
      }
      if (
        output.crossValidationVariance &&
        output.crossValidationVariance.buffer === output.crossValidationPrediction.buffer
      ) {
        throw new Error(`${id} outputs must not share buffers`);
      }
    }
    validateGPUStatusPort(id, {
      invalidCount: output.status?.invalidCount,
      uncertainCount: output.status?.uncertainCount
    });
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.values,
        output.variance,
        output.crossValidationPrediction,
        output.crossValidationVariance,
        output.status?.invalidCount,
        output.status?.uncertainCount
      ],
      [props.positions, props.values, props.mask, props.parameters]
    );
  }

  /**
   * Returns the grid index build, the optional mask fold, the per-cell kriging solve, the
   * optional cross-validation solve, and the optional status accumulation, in order.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, cellCount, maximumNeighborCount} = this;
    const {output, grid, indexGridSize, indexBounds} = props;
    const {width, height} = grid;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.values,
      props.mask,
      props.parameters,
      output.values,
      output.variance,
      output.crossValidationPrediction,
      output.crossValidationVariance,
      output.status?.invalidCount,
      output.status?.uncertainCount
    ]);
    validateMaximumCount(
      id,
      'indexGridSize cells',
      indexGridSize[0] * indexGridSize[1],
      getGPUGridIndexMaximumCellCount(graph.device)
    );
    const sampleCount = props.values.length;
    const indexCellCount = indexGridSize[0] * indexGridSize[1];
    const cellOffsets = createTransientView(
      graph,
      `${id}-cell-offsets`,
      'uint32',
      indexCellCount + 1
    );
    const objectIds = createTransientView(graph, `${id}-object-ids`, 'uint32', sampleCount);
    const indexCount = createTransientView(graph, `${id}-index-count`, 'uint32', 1);
    const indexOverflow = createTransientView(graph, `${id}-index-overflow`, 'uint32', 1);
    const nodes: GPUCommandNode<Parameters>[] = [
      ...new GPUGridIndex({
        id: `${id}-index`,
        positions: props.positions,
        gridSize: indexGridSize,
        bounds: indexBounds,
        cellOffsets,
        objectIds,
        count: indexCount,
        overflow: indexOverflow
      }).getCommandNodes(graph)
    ];

    // The sample mask is folded into a value copy (NaN where masked) so the solve kernels do not
    // spend a storage binding on it. At most 8 storage bindings per solve kernel: positions,
    // values, params, offsets, ids, out, [variance], [cell status].
    let sampleValues: GraphDataView<'float32'> = props.values;
    if (props.mask) {
      sampleValues = createTransientView(graph, `${id}-masked-values`, 'float32', sampleCount);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-apply-mask`,
          operation: OPERATION,
          variant: 'apply-mask',
          bindings: [
            {name: 'rawValues', view: props.values, type: 'f32', access: 'read'},
            {name: 'sampleMask', view: props.mask, type: 'u32', access: 'read'},
            {name: 'maskedValues', view: sampleValues, type: 'f32', access: 'read_write'}
          ],
          invocationCount: sampleCount,
          declarations: GET_NAN_WGSL,
          body: `maskedValues[maskedValuesOffset + index] = select(getNaN(), rawValues[rawValuesOffset + index], sampleMask[sampleMaskOffset + index] != 0u);`
        })
      );
    }
    // No in-cell sort: the neighbor list is keyed by the total order (d^2, row), so the selected
    // neighbors do not depend on the order the index cells are visited, and the grid index's
    // unspecified in-cell order is harmless here (unlike an IDW sum).
    const sharedBindings: WGSLKernelBinding[] = [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'sampleValues', view: sampleValues, type: 'f32', access: 'read'},
      {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'cellOffsets', view: cellOffsets, type: 'u32', access: 'read'},
      {name: 'sortedIds', view: objectIds, type: 'u32', access: 'read'}
    ];
    const sharedDeclarations = `const INDEX_WIDTH: u32 = ${indexGridSize[0]}u;
const INDEX_HEIGHT: u32 = ${indexGridSize[1]}u;
const MAX_NEIGHBORS: u32 = ${maximumNeighborCount}u;
const SYSTEM_STRIDE: u32 = ${maximumNeighborCount + 2}u;
const DOMAIN_MIN: vec2f = vec2f(${getWGSLFloatLiteral(indexBounds[0])}, ${getWGSLFloatLiteral(indexBounds[1])});
const DOMAIN_MAX: vec2f = vec2f(${getWGSLFloatLiteral(indexBounds[2])}, ${getWGSLFloatLiteral(indexBounds[3])});
const RELATIVE_PIVOT_TOLERANCE: f32 = ${getWGSLFloatLiteral(KRIGING_RELATIVE_PIVOT_TOLERANCE)};
const PIVOT_RATIO_TOLERANCE: f32 = ${getWGSLFloatLiteral(KRIGING_PIVOT_RATIO_TOLERANCE)};
const NEGATIVE_VARIANCE_TOLERANCE: f32 = ${getWGSLFloatLiteral(KRIGING_NEGATIVE_VARIANCE_TOLERANCE)};
const STATUS_OK: u32 = ${STATUS_OK}u;
const STATUS_INVALID: u32 = ${STATUS_INVALID}u;
const STATUS_UNCERTAIN: u32 = ${STATUS_UNCERTAIN}u;
const ORIGIN: vec2f = ${this.separation.geographic ? 'vec2f(0.0)' : 'DOMAIN_MIN'};
${GRID_COORDINATE_WGSL}
${getKrigingWGSL(this.separation)}`;

    const hasStatus = Boolean(output.status?.invalidCount || output.status?.uncertainCount);
    const cellStatus = hasStatus
      ? createTransientView(graph, `${id}-cell-status`, 'uint32', cellCount)
      : undefined;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-solve`,
        operation: OPERATION,
        variant: 'local-ordinary',
        bindings: [
          ...sharedBindings,
          {name: 'valuesOut', view: output.values, type: 'f32', access: 'read_write'},
          ...(output.variance
            ? [
                {
                  name: 'varianceOut',
                  view: output.variance,
                  type: 'f32',
                  access: 'read_write'
                } as const
              ]
            : []),
          ...(cellStatus
            ? [{name: 'cellStatus', view: cellStatus, type: 'u32', access: 'read_write'} as const]
            : [])
        ],
        invocationCount: cellCount,
        workgroupSize: 64,
        declarations: `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
${sharedDeclarations}`,
        body: `let column = index % WIDTH;
  let rasterRow = index / WIDTH;
  let gridStart = vec2f(params[paramsOffset], params[paramsOffset + 1u]);
  let gridEnd = vec2f(params[paramsOffset + 2u], params[paramsOffset + 3u]);
  let cellSize = (gridEnd - gridStart) / vec2f(f32(WIDTH), f32(HEIGHT));
  // Planar offsets from the grid start corner stay small whatever the absolute coordinates are;
  // great-circle distances take absolute lon/lat, so the origin is zero.
  let origin = ${this.separation.geographic ? 'vec2f(0.0)' : 'gridStart'};
  let cellOffset = (vec2f(f32(column), f32(rasterRow)) + vec2f(0.5)) * cellSize;
  let center = gridStart + cellOffset;
  let estimate = estimateKriging(center, ${this.separation.geographic ? 'center' : 'cellOffset'}, origin, 0xffffffffu);
  valuesOut[valuesOutOffset + index] = estimate.value;
  ${output.variance ? 'varianceOut[varianceOutOffset + index] = estimate.variance;' : ''}
  ${cellStatus ? 'cellStatus[cellStatusOffset + index] = estimate.status;' : ''}`
      })
    );

    if (output.crossValidationPrediction) {
      const {crossValidationPrediction, crossValidationVariance} = output;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-cross-validation`,
          operation: OPERATION,
          variant: 'leave-one-out',
          bindings: [
            ...sharedBindings,
            {
              name: 'predictionOut',
              view: crossValidationPrediction,
              type: 'f32',
              access: 'read_write'
            },
            ...(crossValidationVariance
              ? [
                  {
                    name: 'predictionVarianceOut',
                    view: crossValidationVariance,
                    type: 'f32',
                    access: 'read_write'
                  } as const
                ]
              : [])
          ],
          invocationCount: sampleCount,
          workgroupSize: 64,
          declarations: sharedDeclarations,
          body: `let value = sampleValues[sampleValuesOffset + index];
  let position = vec2f(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
  var prediction = getNaN();
  var predictionVariance = getNaN();
  if (!isNanBits(value) && isFiniteValue(position.x) && isFiniteValue(position.y)) {
    // Kriged from every OTHER sample: the row itself is excluded from the neighbor search.
    let estimate = estimateKriging(position, position - ORIGIN, ORIGIN, index);
    prediction = estimate.value;
    predictionVariance = estimate.variance;
  }
  predictionOut[predictionOutOffset + index] = prediction;
  ${crossValidationVariance ? 'predictionVarianceOut[predictionVarianceOutOffset + index] = predictionVariance;' : ''}`
        })
      );
    }

    if (cellStatus) {
      for (const [name, view] of [
        ['invalid', output.status?.invalidCount],
        ['uncertain', output.status?.uncertainCount]
      ] as const) {
        if (view) {
          nodes.push(
            createFillNode<Parameters>(graph, {
              id: `${id}-clear-${name}`,
              operation: OPERATION,
              view,
              type: 'u32',
              value: '0u',
              componentCount: view.length
            })
          );
        }
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-count-status`,
          operation: OPERATION,
          variant: 'count-status',
          bindings: [
            {name: 'cellStatus', view: cellStatus, type: 'u32', access: 'read'},
            ...(output.status?.invalidCount
              ? [
                  {
                    name: 'invalidCountOut',
                    view: output.status?.invalidCount,
                    type: 'atomic<u32>',
                    access: 'read_write'
                  } as const
                ]
              : []),
            ...(output.status?.uncertainCount
              ? [
                  {
                    name: 'uncertainCountOut',
                    view: output.status?.uncertainCount,
                    type: 'atomic<u32>',
                    access: 'read_write'
                  } as const
                ]
              : [])
          ],
          invocationCount: cellCount,
          declarations: `const STATUS_INVALID: u32 = ${STATUS_INVALID}u;
const STATUS_UNCERTAIN: u32 = ${STATUS_UNCERTAIN}u;`,
          body: `let status = cellStatus[cellStatusOffset + index];
  ${output.status?.invalidCount ? 'if (status == STATUS_INVALID) { atomicAdd(&invalidCountOut[invalidCountOutOffset], 1u); }' : ''}
  ${output.status?.uncertainCount ? 'if (status == STATUS_UNCERTAIN) { atomicAdd(&uncertainCountOut[uncertainCountOutOffset], 1u); }' : ''}`
        })
      );
    }
    return nodes;
  }
}

/**
 * Shared WGSL of the raster and cross-validation kernels. `estimateKriging` reads the kernel
 * bindings `positions`, `sampleValues`, `params`, `cellOffsets` and `sortedIds` directly.
 */
function getKrigingWGSL(separation: InterpolationSeparation): string {
  // Great-circle mode visits the whole query box: the ring scan's early stop compares distances
  // with index-cell extents in degrees, which has no cheap provably conservative meters bound.
  const scan = separation.geographic ? getRectangleScanWGSL : getRingScanWGSL;
  return /* wgsl */ `
${getSeparationWGSL(separation)}
struct KrigingEstimate {
  value: f32,
  variance: f32,
  status: u32,
}

// Variogram normalized by its total sill; gamma(0) = 0.
fn evaluateVariogram(distance: f32, modelCode: u32, nugget: f32, sill: f32, range: f32) -> f32 {
  if (distance <= 0.0) { return 0.0; }
  let scaled = distance / range;
  var shape = 1.0;
  if (modelCode == 0u) {
    shape = select(1.5 * scaled - 0.5 * scaled * scaled * scaled, 1.0, scaled >= 1.0);
  } else if (modelCode == 1u) {
    shape = 1.0 - exp(-3.0 * scaled);
  } else {
    shape = 1.0 - exp(-3.0 * scaled * scaled);
  }
  return (nugget + sill * shape) / (nugget + sill);
}

// Gaussian elimination with partial pivoting on an n x (n + 1) augmented system stored row-major
// with SYSTEM_STRIDE columns; leaves the solution in column n. Returns false when a pivot is
// negligible against the system scale or the pivot ratio (a condition estimate) is too small.
fn solveSystem(system: ptr<function, array<f32, ${(MAXIMUM_NEIGHBOR_CAPACITY + 1) * (MAXIMUM_NEIGHBOR_CAPACITY + 2)}>>, n: u32, scale: f32) -> bool {
  var minimumPivot = 3.0e38;
  var maximumPivot = 0.0;
  for (var column = 0u; column < n; column++) {
    var pivotRow = column;
    var pivotMagnitude = abs((*system)[column * SYSTEM_STRIDE + column]);
    for (var row = column + 1u; row < n; row++) {
      let magnitude = abs((*system)[row * SYSTEM_STRIDE + column]);
      if (magnitude > pivotMagnitude) {
        pivotMagnitude = magnitude;
        pivotRow = row;
      }
    }
    if (!(pivotMagnitude > RELATIVE_PIVOT_TOLERANCE * scale)) { return false; }
    minimumPivot = min(minimumPivot, pivotMagnitude);
    maximumPivot = max(maximumPivot, pivotMagnitude);
    if (pivotRow != column) {
      for (var k = column; k <= n; k++) {
        let held = (*system)[column * SYSTEM_STRIDE + k];
        (*system)[column * SYSTEM_STRIDE + k] = (*system)[pivotRow * SYSTEM_STRIDE + k];
        (*system)[pivotRow * SYSTEM_STRIDE + k] = held;
      }
    }
    let pivot = (*system)[column * SYSTEM_STRIDE + column];
    for (var row = column + 1u; row < n; row++) {
      let factor = (*system)[row * SYSTEM_STRIDE + column] / pivot;
      for (var k = column; k <= n; k++) {
        (*system)[row * SYSTEM_STRIDE + k] -= factor * (*system)[column * SYSTEM_STRIDE + k];
      }
    }
  }
  if (!(minimumPivot >= PIVOT_RATIO_TOLERANCE * maximumPivot)) { return false; }
  for (var step = 0u; step < n; step++) {
    let row = n - 1u - step;
    var sum = (*system)[row * SYSTEM_STRIDE + n];
    for (var k = row + 1u; k < n; k++) {
      sum -= (*system)[row * SYSTEM_STRIDE + k] * (*system)[k * SYSTEM_STRIDE + n];
    }
    (*system)[row * SYSTEM_STRIDE + n] = sum / (*system)[row * SYSTEM_STRIDE + row];
  }
  return true;
}

// Ordinary kriging estimate at one query point. center is the absolute position (used only to
// find index cells); localCenter and origin give the same point as an offset from origin, and
// every distance is a difference of offsets. excludeRow removes one sample (leave-one-out).
fn estimateKriging(center: vec2f, localCenter: vec2f, origin: vec2f, excludeRow: u32) -> KrigingEstimate {
  let radius = params[paramsOffset + 4u];
  let neighborLimitValue = params[paramsOffset + 5u];
  let modelCode = u32(clamp(params[paramsOffset + 6u], 0.0, 2.0));
  let nugget = params[paramsOffset + 7u];
  let sill = params[paramsOffset + 8u];
  let range = params[paramsOffset + 9u];
  let minimumNeighborValue = params[paramsOffset + 10u];
  let neighborLimit = u32(clamp(select(0.0, neighborLimitValue, isFiniteValue(neighborLimitValue)), 0.0, f32(MAX_NEIGHBORS)));
  let minimumNeighbors = max(u32(clamp(select(1.0, minimumNeighborValue, isFiniteValue(minimumNeighborValue)), 0.0, 4294967040.0)), 1u);
  let radiusKey = getRadiusKey(radius);
  let queryBox = getQueryBox(center, radius);
  let queryMin = queryBox.minimum;
  let queryMax = queryBox.maximum;
  let variogramValid = nugget >= 0.0 && sill >= 0.0 && nugget + sill > 0.0 && range > 0.0 && isFiniteValue(range);
  let valid = variogramValid && neighborLimit > 0u && isFiniteValue(center.x) && isFiniteValue(center.y) &&
    radius >= 0.0 && all(queryMax >= DOMAIN_MIN) && all(queryMin <= DOMAIN_MAX);
  var estimate = KrigingEstimate(getNaN(), getNaN(), STATUS_INVALID);
  var nearDistances: array<f32, MAX_NEIGHBORS>;
  var nearRows: array<u32, MAX_NEIGHBORS>;
  var nearCount = 0u;
  if (valid) {
    let clampedMin = max(queryMin, DOMAIN_MIN);
    let clampedMax = min(queryMax, DOMAIN_MAX);
    var columnLow = getCoordinate(clampedMin.x, DOMAIN_MIN.x, DOMAIN_MAX.x, INDEX_WIDTH);
    var rowLow = getCoordinate(clampedMin.y, DOMAIN_MIN.y, DOMAIN_MAX.y, INDEX_HEIGHT);
    columnLow = columnLow - min(columnLow, 1u);
    rowLow = rowLow - min(rowLow, 1u);
    let columnHigh = min(getCoordinate(clampedMax.x, DOMAIN_MIN.x, DOMAIN_MAX.x, INDEX_WIDTH) + 1u, INDEX_WIDTH - 1u);
    let rowHigh = min(getCoordinate(clampedMax.y, DOMAIN_MIN.y, DOMAIN_MAX.y, INDEX_HEIGHT) + 1u, INDEX_HEIGHT - 1u);
    ${scan(`if (sample == excludeRow) {
          continue;
        }
        let value = sampleValues[sampleValuesOffset + sample];
        if (isNanBits(value)) {
          continue;
        }
        let position = vec2f(positions[positionsOffset + 2u * sample], positions[positionsOffset + 2u * sample + 1u]);
        let distanceSquared = getSeparationKey(position - origin, localCenter);
        if (!(distanceSquared <= radiusKey)) {
          continue;
        }
        let last = neighborLimit - 1u;
        let isFull = nearCount >= neighborLimit;
        if (!isFull || distanceSquared < nearDistances[last] ||
            (distanceSquared == nearDistances[last] && sample < nearRows[last])) {
          var slotPosition = select(nearCount, last, isFull);
          loop {
            if (slotPosition == 0u) { break; }
            let previousDistance = nearDistances[slotPosition - 1u];
            let previousRow = nearRows[slotPosition - 1u];
            if (previousDistance < distanceSquared ||
                (previousDistance == distanceSquared && previousRow < sample)) {
              break;
            }
            nearDistances[slotPosition] = previousDistance;
            nearRows[slotPosition] = previousRow;
            slotPosition -= 1u;
          }
          nearDistances[slotPosition] = distanceSquared;
          nearRows[slotPosition] = sample;
          nearCount = min(nearCount + 1u, neighborLimit);
        }`)}
  }
  if (nearCount == 0u) {
    return estimate;
  }

  // Coalesce samples at identical positions into one kriging point carrying their mean value.
  // Identical positions have identical distances, and the list order is deterministic, so the
  // sums are too.
  var uniqueCount = 0u;
  var uniquePositions: array<vec2f, MAX_NEIGHBORS>;
  var uniqueDistances: array<f32, MAX_NEIGHBORS>;
  var uniqueValueSums: array<f32, MAX_NEIGHBORS>;
  var uniqueMultiplicities: array<f32, MAX_NEIGHBORS>;
  for (var i = 0u; i < nearCount; i++) {
    let row = nearRows[i];
    let localPosition = vec2f(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]) - origin;
    let value = sampleValues[sampleValuesOffset + row];
    var matched = false;
    for (var j = 0u; j < uniqueCount; j++) {
      if (all(uniquePositions[j] == localPosition)) {
        uniqueValueSums[j] += value;
        uniqueMultiplicities[j] += 1.0;
        matched = true;
        break;
      }
    }
    if (!matched) {
      uniquePositions[uniqueCount] = localPosition;
      uniqueDistances[uniqueCount] = nearDistances[i];
      uniqueValueSums[uniqueCount] = value;
      uniqueMultiplicities[uniqueCount] = 1.0;
      uniqueCount += 1u;
    }
  }

  // Exact hit: the nearest distinct position coincides with the query.
  if (isExactHit(uniqueDistances[0], uniquePositions[0], localCenter)) {
    return KrigingEstimate(uniqueValueSums[0] / uniqueMultiplicities[0], 0.0, STATUS_OK);
  }
  if (uniqueCount < minimumNeighbors) {
    return estimate;
  }

  let n = uniqueCount + 1u;
  var system: array<f32, ${(MAXIMUM_NEIGHBOR_CAPACITY + 1) * (MAXIMUM_NEIGHBOR_CAPACITY + 2)}>;
  var targetGamma: array<f32, MAX_NEIGHBORS>;
  var systemScale = 1.0;
  for (var i = 0u; i < uniqueCount; i++) {
    system[i * SYSTEM_STRIDE + i] = 0.0;
    // The variogram matrix is symmetric: one evaluation fills both triangles.
    for (var j = i + 1u; j < uniqueCount; j++) {
      let gamma = evaluateVariogram(getPairMeters(uniquePositions[i], uniquePositions[j]), modelCode, nugget, sill, range);
      system[i * SYSTEM_STRIDE + j] = gamma;
      system[j * SYSTEM_STRIDE + i] = gamma;
      systemScale = max(systemScale, abs(gamma));
    }
    system[i * SYSTEM_STRIDE + uniqueCount] = 1.0;
    system[uniqueCount * SYSTEM_STRIDE + i] = 1.0;
    targetGamma[i] = evaluateVariogram(getSeparationMeters(uniqueDistances[i]), modelCode, nugget, sill, range);
    system[i * SYSTEM_STRIDE + n] = targetGamma[i];
  }
  system[uniqueCount * SYSTEM_STRIDE + uniqueCount] = 0.0;
  system[uniqueCount * SYSTEM_STRIDE + n] = 1.0;
  if (!solveSystem(&system, n, systemScale)) {
    estimate.status = STATUS_UNCERTAIN;
    return estimate;
  }
  let baseValue = uniqueValueSums[0] / uniqueMultiplicities[0];
  var prediction = 0.0;
  var predictionVariance = system[uniqueCount * SYSTEM_STRIDE + n];
  for (var i = 0u; i < uniqueCount; i++) {
    let weight = system[i * SYSTEM_STRIDE + n];
    prediction += weight * (uniqueValueSums[i] / uniqueMultiplicities[i] - baseValue);
    predictionVariance += weight * targetGamma[i];
  }
  // Variance is in units of the total sill and nonnegative for a valid model. Only f32 rounding
  // is clamped; a materially negative value means the weights are unreliable.
  if (!(predictionVariance >= -NEGATIVE_VARIANCE_TOLERANCE) || !isFiniteValue(prediction)) {
    estimate.status = STATUS_UNCERTAIN;
    return estimate;
  }
  return KrigingEstimate(baseValue + prediction, max(predictionVariance, 0.0) * (nugget + sill), STATUS_OK);
}`;
}
