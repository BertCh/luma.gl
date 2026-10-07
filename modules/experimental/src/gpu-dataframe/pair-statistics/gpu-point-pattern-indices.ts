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
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  createPairStatisticsClearNode,
  getPairStatisticsInputNodes,
  getPairStatisticsSharedWGSL,
  getPairStatisticsSortedPointsNodes,
  getPairStatisticsTotalSumNodes,
  PAIR_STATISTICS_FLOAT_WGSL,
  validatePairStatisticsInputs
} from './pair-statistics-grid';
import {
  GPU_CLARK_EVANS_LENGTH,
  GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH,
  GPU_QUADRAT_MAXIMUM_COUNT,
  GPU_QUADRAT_STATISTICS_LENGTH
} from './point-pattern-indices-parameters';

const OPERATION = 'GPUPointPatternIndices';

/** Id written to `nearestNeighborIds` when a row has no nearest neighbor. */
export const GPU_NO_NEAREST_NEIGHBOR = 0xffffffff;

/**
 * Properties for {@link GPUPointPatternIndices}. At least one output is required.
 *
 * Per-frame (no rebuild or recompile): the contents of `positions`, `mask` and `parameters`
 * (window bounds and search-cell hint). Compile-time: the row count, `gridSize`, `quadratGrid` and
 * which outputs are present.
 */
export type GPUPointPatternIndicesProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'point-pattern-indices'`. */
  id?: string;
  /** Packed planar points, one row per event. At least one and fewer than 2^24 rows. */
  positions: GraphDataView<'float32x2'>;
  /**
   * Per-frame parameters: packed float32 view of at least
   * `GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH` elements written with
   * `getGPUPointPatternIndicesParameterValues`.
   */
  parameters: GraphDataView<'float32'>;
  /**
   * Maximum `[columns, rows]` of the neighbor-search cell lattice. Compile-time. Results never
   * depend on it, only speed.
   */
  gridSize: readonly [number, number];
  /** Optional row selection: nonzero includes the row. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Compile-time `[columns, rows]` of the quadrat tiling of the per-frame bounds. Required with
   * `quadratCounts` or `quadratStatistics`; `columns * rows` is at most 65536.
   */
  quadratGrid?: readonly [number, number];
  /**
   * Optional caller-owned distance to the nearest included neighbor, one row per input row. Quiet
   * NaN for excluded rows and when fewer than two rows are included.
   */
  nearestNeighborDistances?: GraphDataView<'float32'>;
  /**
   * Optional caller-owned row index of the nearest included neighbor (smallest row index among
   * ties), one row per input row. `GPU_NO_NEAREST_NEIGHBOR` when there is none.
   */
  nearestNeighborIds?: GraphDataView<'uint32'>;
  /**
   * Optional caller-owned Clark-Evans summary `[n, observedMeanDistance, expectedMeanDistance,
   * ratio, standardError, zScore]` (`GPU_CLARK_EVANS_LENGTH` rows).
   */
  clarkEvans?: GraphDataView<'float32'>;
  /** Optional caller-owned quadrat counts, `columns * rows` rows, row-major (`row * columns + column`). */
  quadratCounts?: GraphDataView<'uint32'>;
  /**
   * Optional caller-owned quadrat summary `[quadratCount, mean, variance, varianceToMeanRatio,
   * chiSquare, degreesOfFreedom]` (`GPU_QUADRAT_STATISTICS_LENGTH` rows).
   */
  quadratStatistics?: GraphDataView<'float32'>;
};

/**
 * Clark-Evans nearest-neighbor index and quadrat-count dispersion statistics of a planar point
 * pattern in a rectangular window (pointpats `PointPattern` / ArcGIS Average Nearest Neighbor).
 *
 * Definition, which the GPU result matches within the bounds below:
 * - The window is the per-frame bounds rectangle with area `A`. Included rows have a nonzero mask
 *   (when given) and a finite position inside the window. `n` is the included row count.
 * - Nearest neighbor: for each included row, the included row with the smallest f32 squared
 *   distance `dx * dx + dy * dy`, ties to the smallest row index, found by an expanding ring search
 *   over the shared lattice cells that stops once the ring's smallest possible distance exceeds
 *   the best found. It is exact for any distance (isolated points search the whole lattice) and
 *   independent of `maximumDistance`, which only sizes the cells.
 * - Clark-Evans: `observed = mean of nearest-neighbor distances`, `expected = 0.5 / sqrt(n / A)`,
 *   `ratio R = observed / expected` (below 1 clustered, above 1 dispersed),
 *   `standardError = 0.26136 / sqrt(n^2 / A)`, `z = (observed - expected) / standardError`. All but
 *   `n` are NaN when `n < 2` or the window has no area. The mean is a deterministic fixed-order
 *   tree sum of f32 distances.
 * - Quadrats: the window splits into `columns x rows` equal rectangles (a point on the maximum
 *   edge belongs to the last one). With `m` quadrats and counts `c_q` (sum `n`): `mean = n / m`,
 *   sample `variance = sum (c_q - mean)^2 / (m - 1)`, `varianceToMeanRatio = variance / mean`
 *   (above 1 clustered), `chiSquare = (m - 1) * varianceToMeanRatio`, `degreesOfFreedom = m - 1`.
 *   Undefined values are NaN. The summary is one serial fixed-order loop over the counts, so it is
 *   bitwise reproducible. p-values are not computed.
 *
 * Counts, ids and `n` are exact. Distances have f32 rounding only; quadrat counts are exact.
 */
export class GPUPointPatternIndices implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUPointPatternIndicesProps;

  constructor(props: GPUPointPatternIndicesProps) {
    this.id = props.id ?? 'point-pattern-indices';
    this.props = props;
    const id = this.id;
    const rows = validatePairStatisticsInputs(id, {
      ...props,
      parameterLength: GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH
    });
    const wantsNearest =
      props.nearestNeighborDistances || props.nearestNeighborIds || props.clarkEvans;
    const wantsQuadrats = props.quadratCounts || props.quadratStatistics;
    if (!wantsNearest && !wantsQuadrats) {
      throw new Error(`${id} needs at least one output`);
    }
    if (props.nearestNeighborDistances) {
      validatePackedView(
        props.nearestNeighborDistances,
        ['float32'],
        `${id} nearestNeighborDistances`
      );
      if (props.nearestNeighborDistances.length !== rows) {
        throw new Error(`${id} nearestNeighborDistances length must equal positions length`);
      }
    }
    if (props.nearestNeighborIds) {
      validatePackedUint32View(props.nearestNeighborIds, `${id} nearestNeighborIds`);
      if (props.nearestNeighborIds.length !== rows) {
        throw new Error(`${id} nearestNeighborIds length must equal positions length`);
      }
    }
    if (props.clarkEvans) {
      validatePackedView(props.clarkEvans, ['float32'], `${id} clarkEvans`);
      if (props.clarkEvans.length < GPU_CLARK_EVANS_LENGTH) {
        throw new Error(`${id} clarkEvans must hold ${GPU_CLARK_EVANS_LENGTH} rows`);
      }
    }
    if (wantsQuadrats) {
      const quadratGrid = props.quadratGrid;
      if (
        !quadratGrid ||
        !Number.isInteger(quadratGrid[0]) ||
        !Number.isInteger(quadratGrid[1]) ||
        quadratGrid[0] < 1 ||
        quadratGrid[1] < 1 ||
        quadratGrid[0] * quadratGrid[1] > GPU_QUADRAT_MAXIMUM_COUNT
      ) {
        throw new Error(
          `${id} quadratGrid must be two positive integers with columns * rows <= ${GPU_QUADRAT_MAXIMUM_COUNT}`
        );
      }
      if (props.quadratCounts) {
        validatePackedUint32View(props.quadratCounts, `${id} quadratCounts`);
        if (props.quadratCounts.length !== quadratGrid[0] * quadratGrid[1]) {
          throw new Error(`${id} quadratCounts must hold columns * rows rows`);
        }
      }
      if (props.quadratStatistics) {
        validatePackedView(props.quadratStatistics, ['float32'], `${id} quadratStatistics`);
        if (props.quadratStatistics.length < GPU_QUADRAT_STATISTICS_LENGTH) {
          throw new Error(
            `${id} quadratStatistics must hold ${GPU_QUADRAT_STATISTICS_LENGTH} rows`
          );
        }
      }
    } else if (props.quadratGrid) {
      throw new Error(`${id} quadratGrid needs quadratCounts or quadratStatistics`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        props.nearestNeighborDistances,
        props.nearestNeighborIds,
        props.clarkEvans,
        props.quadratCounts,
        props.quadratStatistics
      ],
      [props.positions, props.parameters, props.mask]
    );
  }

  /** Returns the point-pattern nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {positions, parameters, gridSize} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      parameters,
      props.mask,
      props.nearestNeighborDistances,
      props.nearestNeighborIds,
      props.clarkEvans,
      props.quadratCounts,
      props.quadratStatistics
    ]);
    const rows = positions.length;
    const cellCount = gridSize[0] * gridSize[1];
    const inputs = getPairStatisticsInputNodes<Parameters>(graph, {
      id,
      operation: OPERATION,
      positions,
      parameters,
      gridSize,
      mask: props.mask
    });
    const nodes = inputs.nodes;
    const sharedWGSL = getPairStatisticsSharedWGSL(gridSize);

    if (props.nearestNeighborDistances || props.nearestNeighborIds || props.clarkEvans) {
      const distances =
        props.nearestNeighborDistances ??
        createTransientView(graph, `${id}-nearest-distances`, 'float32', rows);
      const ids =
        props.nearestNeighborIds ?? createTransientView(graph, `${id}-nearest-ids`, 'uint32', rows);
      const sortedPoints = getPairStatisticsSortedPointsNodes<Parameters>(graph, {
        id,
        operation: OPERATION,
        positions,
        sortedRows: inputs.sortedRows,
        cellOffsets: inputs.cellOffsets,
        gridSize
      });
      nodes.push(
        ...sortedPoints.nodes,
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-nearest`,
          operation: OPERATION,
          variant: 'nearest-neighbor',
          bindings: [
            {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
            {name: 'sortedRows', view: inputs.sortedRows, type: 'u32', access: 'read'},
            {name: 'sortedPoints', view: sortedPoints.sortedPoints, type: 'u32', access: 'read'},
            {name: 'cellOffsets', view: inputs.cellOffsets, type: 'u32', access: 'read'},
            {name: 'distances', view: distances, type: 'f32', access: 'read_write'},
            {name: 'ids', view: ids, type: 'u32', access: 'read_write'}
          ],
          invocationCount: rows,
          declarations: sharedWGSL,
          // Thread `index` searches for the row at cell-order slot `index` (threads of a workgroup
          // share cells: coalesced candidate reads, coherent ring loops) and writes the result at
          // that row. Excluded rows sort last and only get the "no neighbor" result.
          body: `let lattice = readLattice();
  let focus = sortedRows[sortedRowsOffset + index];
  var nearestDistance = getQuietNaN(index);
  var nearestId = 0xffffffffu;
  if (index < cellOffsets[cellOffsetsOffset + CELL_COUNT] && lattice.valid) {
    let x = bitcast<f32>(sortedPoints[sortedPointsOffset + index * 3u]);
    let y = bitcast<f32>(sortedPoints[sortedPointsOffset + index * 3u + 1u]);
    let queryColumn = i32(getCellColumn(lattice, x));
    let queryRow = i32(getCellRow(lattice, y));
    let lastColumn = i32(lattice.columns) - 1;
    let lastRow = i32(lattice.rows) - 1;
    var bestSquared = 3.0e38;
    var bestId = 0xffffffffu;
    let ringCount = max(lattice.columns, lattice.rows);
    // Absolute slack covering f32 rounding of the cell assignment (as in the kNN search).
    let slack = (lattice.cellWidth + lattice.cellHeight) * 0.0009765625 +
      (abs(lattice.minimumX) + abs(lattice.maximumX) + abs(lattice.minimumY) + abs(lattice.maximumY)) * 0.00000095367431640625;
    for (var ringIndex = 0u; ringIndex < ringCount; ringIndex++) {
      let ring = i32(ringIndex);
      let firstColumn = max(queryColumn - ring, 0);
      let endColumn = min(queryColumn + ring, lastColumn);
      let firstRow = max(queryRow - ring, 0);
      let endRow = min(queryRow + ring, lastRow);
      for (var cellRow = firstRow; cellRow <= endRow; cellRow++) {
        let rowBase = u32(cellRow) * lattice.columns;
        // The top and bottom rows of the ring are full (one contiguous slot range, as cells are
        // row major); the others only hold its two end cells.
        let isFullRow = abs(cellRow - queryRow) == ring;
        let step = select(max(2 * ring, 1), 1, isFullRow);
        var columnRangeEnd = endColumn;
        if (!isFullRow) {
          columnRangeEnd = queryColumn + ring;
        }
        var cellColumn = select(queryColumn - ring, firstColumn, isFullRow);
        loop {
          if (cellColumn > columnRangeEnd) { break; }
          if (cellColumn >= 0 && cellColumn <= lastColumn) {
            var begin = cellOffsets[cellOffsetsOffset + rowBase + u32(cellColumn)];
            var end = cellOffsets[cellOffsetsOffset + rowBase + u32(cellColumn) + 1u];
            if (isFullRow) {
              end = cellOffsets[cellOffsetsOffset + rowBase + u32(endColumn) + 1u];
            }
            for (var slot = begin; slot < end; slot++) {
              let neighbor = sortedPoints[sortedPointsOffset + slot * 3u + 2u];
              if (neighbor == focus) {
                continue;
              }
              let deltaX = bitcast<f32>(sortedPoints[sortedPointsOffset + slot * 3u]) - x;
              let deltaY = bitcast<f32>(sortedPoints[sortedPointsOffset + slot * 3u + 1u]) - y;
              let distanceSquared = deltaX * deltaX + deltaY * deltaY;
              if (distanceSquared < bestSquared || (distanceSquared == bestSquared && neighbor < bestId)) {
                bestSquared = distanceSquared;
                bestId = neighbor;
              }
            }
          }
          if (isFullRow) { break; }
          cellColumn += step;
        }
      }
      // Clearance of the visited box on each side; a side at the lattice edge has nothing beyond
      // it. Every unvisited point is at least that far away, so stop once the best is strictly
      // closer (ties at the best distance therefore always resolve to the lowest ID).
      var clearance = 3.0e38;
      var open = false;
      if (queryColumn - ring > 0) {
        clearance = min(clearance, x - (lattice.minimumX + f32(queryColumn - ring) * lattice.cellWidth));
        open = true;
      }
      if (queryColumn + ring < lastColumn) {
        clearance = min(clearance, lattice.minimumX + f32(queryColumn + ring + 1) * lattice.cellWidth - x);
        open = true;
      }
      if (queryRow - ring > 0) {
        clearance = min(clearance, y - (lattice.minimumY + f32(queryRow - ring) * lattice.cellHeight));
        open = true;
      }
      if (queryRow + ring < lastRow) {
        clearance = min(clearance, lattice.minimumY + f32(queryRow + ring + 1) * lattice.cellHeight - y);
        open = true;
      }
      if (!open) {
        break;
      }
      let safeClearance = max(clearance - slack, 0.0);
      if (bestId != 0xffffffffu && bestSquared < safeClearance * safeClearance) {
        break;
      }
    }
    if (bestId != 0xffffffffu) {
      nearestDistance = sqrt(bestSquared);
      nearestId = bestId;
    }
  }
  distances[distancesOffset + focus] = nearestDistance;
  ids[idsOffset + focus] = nearestId;`
        })
      );

      if (props.clarkEvans) {
        const terms = createTransientView(graph, `${id}-nearest-terms`, 'float32', rows);
        const total = createTransientView(graph, `${id}-nearest-total`, 'float32', 1);
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-nearest-terms`,
            operation: OPERATION,
            variant: 'nearest-terms',
            bindings: [
              {name: 'distances', view: distances, type: 'f32', access: 'read'},
              {name: 'terms', view: terms, type: 'f32', access: 'read_write'}
            ],
            invocationCount: rows,
            declarations: PAIR_STATISTICS_FLOAT_WGSL,
            body: `let distance = distances[distancesOffset + index];
  terms[termsOffset + index] = select(0.0, distance, isFiniteFloat(distance));`
          }),
          ...getPairStatisticsTotalSumNodes<Parameters>(graph, {
            id: `${id}-nearest-sum`,
            operation: OPERATION,
            input: terms,
            output: total
          }),
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-clark-evans`,
            operation: OPERATION,
            variant: 'clark-evans',
            bindings: [
              {name: 'total', view: total, type: 'f32', access: 'read'},
              {name: 'cellOffsets', view: inputs.cellOffsets, type: 'u32', access: 'read'},
              {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
              {name: 'clarkEvans', view: props.clarkEvans, type: 'f32', access: 'read_write'}
            ],
            invocationCount: 1,
            declarations: `${PAIR_STATISTICS_FLOAT_WGSL}
const CELL_COUNT: u32 = ${cellCount}u;`,
            body: `let n = f32(cellOffsets[cellOffsetsOffset + CELL_COUNT]);
  let area = (parameters[parametersOffset + 2u] - parameters[parametersOffset]) *
    (parameters[parametersOffset + 3u] - parameters[parametersOffset + 1u]);
  let nan = getQuietNaN(index);
  var observed = nan;
  var expected = nan;
  var ratio = nan;
  var standardError = nan;
  var zScore = nan;
  if (n >= 2.0 && isFiniteFloat(area) && area > 0.0) {
    observed = total[totalOffset] / n;
    expected = 0.5 * sqrt(area / n);
    ratio = observed / expected;
    standardError = 0.26136 * sqrt(area) / n;
    zScore = (observed - expected) / standardError;
  }
  clarkEvans[clarkEvansOffset] = n;
  clarkEvans[clarkEvansOffset + 1u] = observed;
  clarkEvans[clarkEvansOffset + 2u] = expected;
  clarkEvans[clarkEvansOffset + 3u] = ratio;
  clarkEvans[clarkEvansOffset + 4u] = standardError;
  clarkEvans[clarkEvansOffset + 5u] = zScore;`
          })
        );
      }
    }

    if (props.quadratCounts || props.quadratStatistics) {
      const [quadratColumns, quadratRows] = props.quadratGrid!;
      const quadratCount = quadratColumns * quadratRows;
      const counts =
        props.quadratCounts ??
        createTransientView(graph, `${id}-quadrat-counts`, 'uint32', quadratCount);
      nodes.push(
        createPairStatisticsClearNode<Parameters>(graph, `${id}-quadrat-clear`, OPERATION, counts),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-quadrat-counts`,
          operation: OPERATION,
          variant: 'quadrat-counts',
          bindings: [
            {name: 'positions', view: positions, type: 'f32', access: 'read'},
            {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
            {name: 'cellKeys', view: inputs.cellKeys, type: 'u32', access: 'read'},
            {name: 'quadratCounts', view: counts, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: rows,
          declarations: `${sharedWGSL}
const QUADRAT_COLUMNS: u32 = ${quadratColumns}u;
const QUADRAT_ROWS: u32 = ${quadratRows}u;`,
          body: `if (cellKeys[cellKeysOffset + index] < CELL_COUNT) {
    let x = positions[positionsOffset + index * 2u];
    let y = positions[positionsOffset + index * 2u + 1u];
    let width = readParameter(2u) - readParameter(0u);
    let height = readParameter(3u) - readParameter(1u);
    var column = 0u;
    var row = 0u;
    if (width > 0.0) {
      column = min(u32(floor((x - readParameter(0u)) / width * f32(QUADRAT_COLUMNS))), QUADRAT_COLUMNS - 1u);
    }
    if (height > 0.0) {
      row = min(u32(floor((y - readParameter(1u)) / height * f32(QUADRAT_ROWS))), QUADRAT_ROWS - 1u);
    }
    atomicAdd(&quadratCounts[quadratCountsOffset + row * QUADRAT_COLUMNS + column], 1u);
  }`
        })
      );
      if (props.quadratStatistics) {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-quadrat-statistics`,
            operation: OPERATION,
            variant: 'quadrat-statistics',
            bindings: [
              {name: 'quadratCounts', view: counts, type: 'u32', access: 'read'},
              {name: 'cellOffsets', view: inputs.cellOffsets, type: 'u32', access: 'read'},
              {
                name: 'quadratStatistics',
                view: props.quadratStatistics,
                type: 'f32',
                access: 'read_write'
              }
            ],
            invocationCount: 1,
            declarations: `${PAIR_STATISTICS_FLOAT_WGSL}
const CELL_COUNT: u32 = ${cellCount}u;
const QUADRAT_COUNT: u32 = ${quadratCount}u;`,
            body: `let nan = getQuietNaN(index);
  let m = f32(QUADRAT_COUNT);
  let n = f32(cellOffsets[cellOffsetsOffset + CELL_COUNT]);
  let mean = n / m;
  // One serial fixed-order loop: bitwise reproducible.
  var squares = 0.0;
  for (var quadrat = 0u; quadrat < QUADRAT_COUNT; quadrat++) {
    let centered = f32(quadratCounts[quadratCountsOffset + quadrat]) - mean;
    squares += centered * centered;
  }
  let hasVariance = QUADRAT_COUNT > 1u;
  let variance = select(nan, squares / (m - 1.0), hasVariance);
  let ratio = select(nan, variance / mean, hasVariance && mean > 0.0);
  quadratStatistics[quadratStatisticsOffset] = m;
  quadratStatistics[quadratStatisticsOffset + 1u] = mean;
  quadratStatistics[quadratStatisticsOffset + 2u] = variance;
  quadratStatistics[quadratStatisticsOffset + 3u] = ratio;
  quadratStatistics[quadratStatisticsOffset + 4u] = select(nan, (m - 1.0) * ratio, isFiniteFloat(ratio));
  quadratStatistics[quadratStatisticsOffset + 5u] = m - 1.0;`
          })
        );
      }
    }
    return nodes;
  }
}
