// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {ERFC_VALUE_WGSL} from '../../utils/wgsl-common';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  createPairStatisticsClearNode,
  getPairStatisticsInputNodes,
  getPairStatisticsWindowParameterNodes,
  getPairStatisticsSharedWGSL,
  getPairStatisticsSortedPointsNodes,
  getPairStatisticsTotalSumNodes,
  PAIR_STATISTICS_FLOAT_WGSL,
  validatePairStatisticsInputs,
  validatePairStatisticsRowCount
} from './pair-statistics-grid';
import {
  GPU_CLARK_EVANS_LENGTH,
  GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH,
  GPU_QUADRAT_MAXIMUM_COUNT,
  GPU_QUADRAT_STATISTICS_LENGTH
} from './point-pattern-indices-parameters';

import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';

import {
  resolveGPUStudyWindow,
  type GPUStudyWindow
} from '../../analysis-core/contracts/study-window';
import {EXACT_ORIENTATION_WGSL} from '../../utils/exact-orientation-wgsl';
import {EXACT_GEOMETRY_WGSL} from '../../utils/exact-geometry-wgsl';
import {
  getStudyWindowBinding,
  getStudyWindowWGSL,
  getStudyWindowWGSLOptions
} from '../../utils/study-window-wgsl';

const OPERATION = 'GPUPointPatternIndices';

/**
 * `getNormalPValue(z, alternative)`: p-value of a standard-normal statistic. Alternative codes
 * 0 two-sided, 1 greater (upper tail), 2 lesser (lower tail). Needs `erfcValue`.
 */
const NORMAL_P_VALUE_WGSL = /* wgsl */ `
fn getNormalPValue(z: f32, alternative: u32) -> f32 {
  let root = 0.70710678118655;
  if (alternative == 1u) {
    return 0.5 * erfcValue(z * root);
  }
  if (alternative == 2u) {
    return 0.5 * erfcValue(-z * root);
  }
  return erfcValue(abs(z) * root);
}
`;

/**
 * `getChiSquarePValue(x, degreesOfFreedom, alternative)`: chi-square p-value. Exact regularized
 * incomplete gamma function (series below `x < a + 1`, Lentz continued fraction above, with
 * `a = df / 2`) for `df <= 100`; the Wilson-Hilferty cube-root normal approximation above, where
 * f32 evaluation of the gamma prefactor is no longer accurate. The tail that is computed directly
 * keeps full relative precision; its complement is `1 - tail`. Alternative codes as in
 * `getNormalPValue`. Needs `erfcValue` and `getNormalPValue`.
 */
const CHI_SQUARE_P_VALUE_WGSL = /* wgsl */ `
fn getLogGamma(value: f32) -> f32 {
  var x = value;
  var shift = 0.0;
  loop {
    if (x >= 8.0) { break; }
    shift += log(x);
    x += 1.0;
  }
  let inverse = 1.0 / x;
  let inverseSquared = inverse * inverse;
  return (x - 0.5) * log(x) - x + 0.91893853320467 +
    inverse * (0.08333333333333 - inverseSquared * (0.00277777777778 - inverseSquared * 0.00079365079365)) - shift;
}

// Returns (lower tail P(a, x), upper tail Q(a, x)) of the regularized incomplete gamma function.
fn getGammaTails(a: f32, x: f32) -> vec2<f32> {
  if (x <= 0.0) {
    return vec2<f32>(0.0, 1.0);
  }
  let prefix = exp(a * log(x) - x - getLogGamma(a));
  if (x < a + 1.0) {
    var term = 1.0 / a;
    var sum = term;
    var denominator = a;
    for (var iteration = 0; iteration < 500; iteration++) {
      denominator += 1.0;
      term *= x / denominator;
      sum += term;
      if (abs(term) < abs(sum) * 1.0e-7) { break; }
    }
    let lower = clamp(sum * prefix, 0.0, 1.0);
    return vec2<f32>(lower, 1.0 - lower);
  }
  var b = x + 1.0 - a;
  var c = 1.0e30;
  var d = 1.0 / b;
  var h = d;
  for (var iteration = 1; iteration < 500; iteration++) {
    let an = -f32(iteration) * (f32(iteration) - a);
    b += 2.0;
    d = an * d + b;
    if (abs(d) < 1.0e-30) { d = 1.0e-30; }
    c = b + an / c;
    if (abs(c) < 1.0e-30) { c = 1.0e-30; }
    d = 1.0 / d;
    let delta = d * c;
    h *= delta;
    if (abs(delta - 1.0) < 1.0e-7) { break; }
  }
  let upper = clamp(prefix * h, 0.0, 1.0);
  return vec2<f32>(1.0 - upper, upper);
}

fn getChiSquarePValue(x: f32, degreesOfFreedom: f32, alternative: u32) -> f32 {
  var lower: f32;
  var upper: f32;
  if (degreesOfFreedom <= 100.0) {
    let tails = getGammaTails(0.5 * degreesOfFreedom, 0.5 * x);
    lower = tails.x;
    upper = tails.y;
  } else {
    let scale = 2.0 / (9.0 * degreesOfFreedom);
    let z = (pow(x / degreesOfFreedom, 1.0 / 3.0) - (1.0 - scale)) / sqrt(scale);
    lower = getNormalPValue(z, 2u);
    upper = getNormalPValue(z, 1u);
  }
  if (alternative == 1u) { return upper; }
  if (alternative == 2u) { return lower; }
  return min(1.0, 2.0 * min(lower, upper));
}
`;

/** Id written to `nearestNeighborIds` when a row has no nearest neighbor. */
export const GPU_NO_NEAREST_NEIGHBOR = 0xffffffff;

/** Caller-owned output views of {@link GPUPointPatternIndices}. */
export type GPUPointPatternIndicesOutput = {
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
   * ratio, standardError, zScore, pValue]` (`GPU_CLARK_EVANS_LENGTH` rows).
   */
  clarkEvans?: GraphDataView<'float32'>;
  /** Optional caller-owned quadrat counts, `columns * rows` rows, row-major (`row * columns + column`). */
  quadratCounts?: GraphDataView<'uint32'>;
  /** Window area per row-major quadrat, rescaled to the polygon area; plain cell area for rectangles. */
  quadratAreas?: GraphDataView<'float32'>;
  /**
   * Optional caller-owned quadrat summary `[quadratCount, mean, variance, varianceToMeanRatio,
   * chiSquare, degreesOfFreedom, pValue]` (`GPU_QUADRAT_STATISTICS_LENGTH` rows).
   */
  quadratStatistics?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUPointPatternIndices}. At least one output is required.
 *
 * Per-frame (no rebuild or recompile): the contents of `positions`, `mask` and `parameters`
 * (window bounds and search-cell hint). Compile-time: the row count, `gridSize`, `quadratGrid` and
 * which `output` views are present.
 */
export type GPUPointPatternIndicesProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'point-pattern-indices'`. */
  id?: string;
  /**
   * Coordinate and metric descriptor of `positions`. Omitted means native planar coordinates.
   * Only planar contexts are supported: distances and areas are Euclidean, so a
   * `longitude-latitude` context throws at construction (project positions to a planar frame
   * first).
   * Measures both distance and window area, so neither local frame is exact: use an
   * azimuthal-equidistant frame for study areas small enough that its area distortion is
   * negligible, or a projected CRS.
   */
  spatialContext?: GPUSpatialContext;
  /** Prepared valid polygon in the same planar frame, or 'rectangle' (default).
   * Polygon parameters require bounds: 'study-window' and an explicit 'guard' or 'none' correction.
   */
  studyWindow?: GPUStudyWindow;
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
   * `output.quadratCounts`, `output.quadratAreas` or `output.quadratStatistics`; `columns * rows` is at most 65536.
   */
  quadratGrid?: readonly [number, number];
  /** Midpoint samples per quadrat axis (default 16). Positive integer; count * samples² <= 2^24.
   * Polygon quadrat areas are lattice approximations; increase this for narrow boundary features.
   */
  quadratSamples?: number;
  /** Caller-owned output views. At least one is required. */
  output: GPUPointPatternIndicesOutput;
};

/**
 * Clark-Evans nearest-neighbor index and quadrat-count dispersion statistics of a planar point
 * pattern in a rectangle or prepared polygon window (pointpats `PointPattern` / ArcGIS Average Nearest Neighbor).
 *
 * Definition, which the GPU result matches within the bounds below:
 * - The window is the per-frame bounds rectangle or prepared polygon with area `A`. Included rows have a nonzero mask
 *   (when given) and a finite position inside the window. `n` is the included row count.
 * - Nearest neighbor: for each included row, the included row with the smallest f32 squared
 *   distance `dx * dx + dy * dy`, ties to the smallest row index, found by an expanding ring search
 *   over the shared lattice cells that stops once the ring's smallest possible distance exceeds
 *   the best found. It is exact for any distance (isolated points search the whole lattice) and
 *   independent of `maximumDistance`, which only sizes the cells.
 * - Clark-Evans: `observed = mean of nearest-neighbor distances`, `ratio R = observed / expected`
 *   (below 1 clustered, above 1 dispersed), `z = (observed - expected) / standardError`. All but
 *   `n` are NaN when `n < 2` or the window has no area. The mean is a deterministic fixed-order
 *   tree sum of f32 distances. The expectation depends on the per-frame `clarkEvansCorrection`:
 *   `'donnelly'` (default) is Donnelly (1978, in Hodder (ed.), Simulation Studies in Archaeology,
 *   Cambridge University Press, pp. 91-95) for a rectangular window of area `A` and perimeter `P`:
 *   `expected = 0.5 sqrt(A / n) + (0.0514 + 0.041 / sqrt(n)) P / n`,
 *   `standardError = sqrt(0.0703 A / n^2 + 0.037 P sqrt(A / n^5))`. It removes the bias that makes
 *   the plain index look clustered when the window is small or elongated, because nearest
 *   neighbors of boundary points lie farther away than the infinite-plane theory assumes. (spatstat
 *   `clarkevans(correction = "Donnelly")` uses the constant 0.0412 in place of 0.041; Donnelly's
 *   published value is used here, the difference is at most 0.5% of the correction term.)
 *   `'none'` is the infinite-plane `expected = 0.5 sqrt(A / n)` with
 *   `standardError = 0.26136 sqrt(A) / n` (pointpats, ArcGIS Average Nearest Neighbor). The
 *   `pValue` is the normal-theory p-value of `z` for the per-frame `alternative` (`'two-sided'`
 *   default; `'greater'` tests for regularity, `'lesser'` for clustering). The rectangular
 *   window is assumed to be the study area. Polygon windows require explicit 'none' or 'guard'.
 *   'guard' averages nearest-neighbor distances only over included points at least guardDistance
 *   from any boundary; neighbors still include every included point. Its expectation is
 *   `0.5 sqrt(A / n)` and standard error `0.26136 sqrt(A / (n * focusCount))`. Zero focus
 *   points give NaN statistics. Recommend `guardDistance >= 2 / sqrt(n / A)`.
 * - Quadrats: the window splits into `columns x rows` equal rectangles (a point on the maximum
 *   edge belongs to the last one). With `m` quadrats and counts `c_q` (sum `n`): `mean = n / m`,
 *   sample `variance = sum (c_q - mean)^2 / (m - 1)`, `varianceToMeanRatio = variance / mean`
 *   (above 1 clustered), `chiSquare = (m - 1) * varianceToMeanRatio`, `degreesOfFreedom = m - 1`.
 *   Undefined values are NaN. The summary is one serial fixed-order loop over the counts, so it is
 *   bitwise reproducible. `pValue` is the chi-square p-value with `m - 1` degrees of freedom for
 *   the per-frame `alternative`: `'two-sided'` (default) is `min(1, 2 min(P(X <= x), P(X >= x)))`,
 *   `'greater'` the upper tail (clustered, spatstat `quadrat.test` alternative `"clustered"`),
 *   `'lesser'` the lower tail (regular). It uses the exact regularized incomplete gamma function
 *   up to 100 degrees of freedom and the Wilson-Hilferty approximation (absolute error below 2e-4)
 *   above. The test assumes expected counts of about 5 or more per quadrat.
 *
 * Polygon quadrats sample a quadratSamples² midpoint lattice, normalize the sampled areas to A,
 * drop zero-area quadrats and use expected counts n * area / A. The variance row is the sum
 * of squared residuals / df; the ratio row is chiSquare / df. Equal areas retain the rectangle formula.
 * Counts, ids and `n` are exact. Distances have f32 rounding only; quadrat counts are exact.
 */
export class GPUPointPatternIndices implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUPointPatternIndicesProps;
  /** Resolved spatial context (native planar unless the props say otherwise). */
  readonly spatialContext: GPUSpatialContext;
  /** Resolved compile-time study window. */
  readonly studyWindow: GPUStudyWindow;

  constructor(props: GPUPointPatternIndicesProps) {
    this.id = props.id ?? 'point-pattern-indices';
    this.props = props;
    const id = this.id;
    this.spatialContext = resolveGPUSpatialContext(id, props.spatialContext);
    this.studyWindow = resolveGPUStudyWindow(id, props.studyWindow, this.spatialContext);
    const {output} = props;
    const rows = validatePairStatisticsInputs(id, {
      ...props,
      parameterLength: GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH
    });
    const wantsNearest =
      output.nearestNeighborDistances || output.nearestNeighborIds || output.clarkEvans;
    const wantsQuadrats = output.quadratCounts || output.quadratStatistics || output.quadratAreas;
    if (!wantsNearest && !wantsQuadrats) {
      throw new Error(`${id} output needs at least one view`);
    }
    if (output.nearestNeighborDistances) {
      validatePackedView(
        output.nearestNeighborDistances,
        ['float32'],
        `${id} output.nearestNeighborDistances`
      );
      if (output.nearestNeighborDistances.length !== rows) {
        throw new Error(`${id} output.nearestNeighborDistances length must equal positions length`);
      }
    }
    if (output.nearestNeighborIds) {
      validatePackedUint32View(output.nearestNeighborIds, `${id} output.nearestNeighborIds`);
      if (output.nearestNeighborIds.length !== rows) {
        throw new Error(`${id} output.nearestNeighborIds length must equal positions length`);
      }
    }
    if (output.clarkEvans) {
      validatePackedView(output.clarkEvans, ['float32'], `${id} output.clarkEvans`);
      if (output.clarkEvans.length < GPU_CLARK_EVANS_LENGTH) {
        throw new Error(`${id} output.clarkEvans must hold ${GPU_CLARK_EVANS_LENGTH} rows`);
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
      const quadratSamples = props.quadratSamples ?? 16;
      if (
        !Number.isInteger(quadratSamples) ||
        quadratSamples < 1 ||
        quadratSamples ** 2 * quadratGrid[0] * quadratGrid[1] > 16777216
      ) {
        throw new Error(
          `${id} quadratSamples squared times the quadrat count must be at most 16777216`
        );
      }
      if (output.quadratAreas) {
        validatePackedView(output.quadratAreas, ['float32'], `${id} output.quadratAreas`);
        if (output.quadratAreas.length !== quadratGrid[0] * quadratGrid[1]) {
          throw new Error(`${id} output.quadratAreas must hold columns * rows rows`);
        }
      }
      if (output.quadratCounts) {
        validatePackedUint32View(output.quadratCounts, `${id} output.quadratCounts`);
        if (output.quadratCounts.length !== quadratGrid[0] * quadratGrid[1]) {
          throw new Error(`${id} output.quadratCounts must hold columns * rows rows`);
        }
      }
      if (output.quadratStatistics) {
        validatePackedView(output.quadratStatistics, ['float32'], `${id} output.quadratStatistics`);
        if (output.quadratStatistics.length < GPU_QUADRAT_STATISTICS_LENGTH) {
          throw new Error(
            `${id} output.quadratStatistics must hold ${GPU_QUADRAT_STATISTICS_LENGTH} rows`
          );
        }
      }
    } else if (props.quadratGrid) {
      throw new Error(
        `${id} quadratGrid needs output.quadratCounts, output.quadratAreas or output.quadratStatistics`
      );
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.nearestNeighborDistances,
        output.nearestNeighborIds,
        output.clarkEvans,
        output.quadratCounts,
        output.quadratAreas,
        output.quadratStatistics
      ],
      [props.positions, props.parameters, props.mask]
    );
  }

  /** Returns the point-pattern nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {positions, parameters: callerParameters, gridSize, output} = props;
    validatePairStatisticsRowCount(id, graph.device, positions.length);
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      callerParameters,
      props.mask,
      output.nearestNeighborDistances,
      output.nearestNeighborIds,
      output.clarkEvans,
      output.quadratCounts,
      output.quadratAreas,
      output.quadratStatistics
    ]);
    const studyWindow =
      this.studyWindow === 'rectangle'
        ? undefined
        : this.studyWindow.getViews(graph as unknown as GPUCommandGraph<unknown>);
    if (studyWindow) {
      validateGraphViewsBelongToGraph(id, graph, [
        studyWindow.edges,
        studyWindow.summary,
        studyWindow.slabs.params,
        studyWindow.slabs.slabOffsets,
        studyWindow.slabs.slabEntries,
        studyWindow.sampleCells,
        studyWindow.sampleCellCount,
        studyWindow.packed
      ]);
    }
    const windowParameters = studyWindow
      ? getPairStatisticsWindowParameterNodes(graph, {
          id,
          operation: OPERATION,
          parameters: callerParameters,
          parameterLength: callerParameters.length,
          window: studyWindow
        })
      : undefined;
    const parameters = windowParameters?.parameters ?? callerParameters;
    const windowWGSL = studyWindow
      ? `${EXACT_ORIENTATION_WGSL}
${EXACT_GEOMETRY_WGSL}
${getStudyWindowWGSL(getStudyWindowWGSLOptions(studyWindow))}`
      : '';
    const rows = positions.length;
    const cellCount = gridSize[0] * gridSize[1];
    const inputs = getPairStatisticsInputNodes<Parameters>(graph, {
      id,
      operation: OPERATION,
      positions,
      parameters,
      gridSize,
      mask: props.mask,
      studyWindow
    });
    const nodes = [...(windowParameters?.nodes ?? []), ...inputs.nodes];
    const sharedWGSL = getPairStatisticsSharedWGSL(gridSize);

    if (output.nearestNeighborDistances || output.nearestNeighborIds || output.clarkEvans) {
      const distances =
        output.nearestNeighborDistances ??
        createTransientView(graph, `${id}-nearest-distances`, 'float32', rows);
      const ids =
        output.nearestNeighborIds ??
        createTransientView(graph, `${id}-nearest-ids`, 'uint32', rows);
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

      if (output.clarkEvans) {
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
              ...(studyWindow ? [getStudyWindowBinding(studyWindow)] : []),
              {name: 'total', view: total, type: 'f32', access: 'read'},
              {name: 'cellOffsets', view: inputs.cellOffsets, type: 'u32', access: 'read'},
              {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
              {name: 'clarkEvans', view: output.clarkEvans, type: 'f32', access: 'read_write'}
            ],
            invocationCount: 1,
            declarations: `${PAIR_STATISTICS_FLOAT_WGSL}${windowWGSL}
${ERFC_VALUE_WGSL}
${NORMAL_P_VALUE_WGSL}
const CELL_COUNT: u32 = ${cellCount}u;`,
            body: `let n = f32(cellOffsets[cellOffsetsOffset + CELL_COUNT]);
  let width = parameters[parametersOffset + 2u] - parameters[parametersOffset];
  let height = parameters[parametersOffset + 3u] - parameters[parametersOffset + 1u];
  let area = ${studyWindow ? 'studyWindowSummaryValue(4u)' : 'width * height'};
  let correction = u32(parameters[parametersOffset + 5u] + 0.5);
  let alternative = u32(parameters[parametersOffset + 6u] + 0.5);
  let nan = getQuietNaN(index);
  var observed = nan;
  var expected = nan;
  var ratio = nan;
  var standardError = nan;
  var zScore = nan;
  var pValue = nan;
  if (n >= 2.0 && isFiniteFloat(area) && area > 0.0) {
    observed = total[totalOffset] / n;
    expected = 0.5 * sqrt(area / n);
    standardError = 0.26136 * sqrt(area) / n;
    if (correction == 1u) {
      // Donnelly (1978) edge correction for a rectangle of perimeter P. sqrt(A / n^5) is
      // evaluated as sqrt(A) / (n^2 sqrt(n)) to stay inside the f32 range for large n.
      let perimeter = 2.0 * (width + height);
      expected += (0.0514 + 0.041 / sqrt(n)) * perimeter / n;
      standardError = sqrt(0.0703 * area / (n * n) + 0.037 * perimeter * sqrt(area) / (n * n * sqrt(n)));
    }
    ${studyWindow ? 'if (correction == 1u) { expected = nan; standardError = nan; }' : ''}${studyWindow ? '\n    ' : ''}ratio = observed / expected;
    zScore = (observed - expected) / standardError;
    pValue = getNormalPValue(zScore, alternative);
  }
  clarkEvans[clarkEvansOffset] = n;
  clarkEvans[clarkEvansOffset + 1u] = observed;
  clarkEvans[clarkEvansOffset + 2u] = expected;
  clarkEvans[clarkEvansOffset + 3u] = ratio;
  clarkEvans[clarkEvansOffset + 4u] = standardError;
  clarkEvans[clarkEvansOffset + 5u] = zScore;
  clarkEvans[clarkEvansOffset + 6u] = pValue;`
          })
        );
        // Keep legacy rectangle kernels byte-identical. The per-frame guard pass overwrites
        // the summary only for code 2, and never changes nearest-neighbor outputs.
        const guardTerms = createTransientView(graph, `${id}-guard-terms`, 'float32', rows);
        const guardFocus = createTransientView(graph, `${id}-guard-focus`, 'float32', rows);
        const guardTotal = createTransientView(graph, `${id}-guard-total`, 'float32', 1);
        const guardCount = createTransientView(graph, `${id}-guard-count`, 'float32', 1);
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-guard-terms`,
            operation: OPERATION,
            variant: 'guard-terms',
            bindings: [
              {name: 'positions', view: positions, type: 'f32', access: 'read'},
              {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
              {name: 'distances', view: distances, type: 'f32', access: 'read'},
              {name: 'guardTerms', view: guardTerms, type: 'f32', access: 'read_write'},
              {name: 'guardFocus', view: guardFocus, type: 'f32', access: 'read_write'},
              ...(studyWindow ? [getStudyWindowBinding(studyWindow)] : [])
            ],
            invocationCount: rows,
            declarations: `${PAIR_STATISTICS_FLOAT_WGSL}${windowWGSL}`,
            body: `let distance = distances[distancesOffset + index];
  let guardDistance = parameters[parametersOffset + 7u];
  var focus = false;
  if (parameters[parametersOffset + 5u] == 2.0 && isFiniteFloat(distance) &&
      isFiniteFloat(guardDistance) && guardDistance > 0.0) {
    let point = vec2f(positions[positionsOffset + index * 2u], positions[positionsOffset + index * 2u + 1u]);
    let boundaryDistance = ${
      studyWindow
        ? 'getStudyWindowBoundaryDistance(point, guardDistance)'
        : `min(min(point.x - parameters[parametersOffset], parameters[parametersOffset + 2u] - point.x),
      min(point.y - parameters[parametersOffset + 1u], parameters[parametersOffset + 3u] - point.y))`
    };
    focus = boundaryDistance >= guardDistance;
  }
  guardTerms[guardTermsOffset + index] = select(0.0, distance, focus);
  guardFocus[guardFocusOffset + index] = select(0.0, 1.0, focus);`
          }),
          ...getPairStatisticsTotalSumNodes<Parameters>(graph, {
            id: `${id}-guard-sum`,
            operation: OPERATION,
            input: guardTerms,
            output: guardTotal
          }),
          ...getPairStatisticsTotalSumNodes<Parameters>(graph, {
            id: `${id}-guard-count-sum`,
            operation: OPERATION,
            input: guardFocus,
            output: guardCount
          }),
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-guard-clark-evans`,
            operation: OPERATION,
            variant: 'guard-clark-evans',
            bindings: [
              {name: 'guardTotal', view: guardTotal, type: 'f32', access: 'read'},
              {name: 'guardCount', view: guardCount, type: 'f32', access: 'read'},
              {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
              {name: 'clarkEvans', view: output.clarkEvans, type: 'f32', access: 'read_write'},
              ...(studyWindow ? [getStudyWindowBinding(studyWindow)] : [])
            ],
            invocationCount: 1,
            declarations: `${PAIR_STATISTICS_FLOAT_WGSL}${windowWGSL}
${ERFC_VALUE_WGSL}
${NORMAL_P_VALUE_WGSL}`,
            body: `if (parameters[parametersOffset + 5u] != 2.0) { return; }
  let count = clarkEvans[clarkEvansOffset];
  let focusCount = guardCount[guardCountOffset];
  let area = ${studyWindow ? 'studyWindowSummaryValue(4u)' : '(parameters[parametersOffset + 2u] - parameters[parametersOffset]) * (parameters[parametersOffset + 3u] - parameters[parametersOffset + 1u])'};
  let nan = getQuietNaN(index);
  for (var row = 1u; row < 7u; row++) { clarkEvans[clarkEvansOffset + row] = nan; }
  if (count >= 2.0 && focusCount > 0.0 && isFiniteFloat(area) && area > 0.0) {
    let observed = guardTotal[guardTotalOffset] / focusCount;
    let expected = 0.5 * sqrt(area / count);
    let standardError = 0.26136 * sqrt(area / (count * focusCount));
    let zScore = (observed - expected) / standardError;
    clarkEvans[clarkEvansOffset + 1u] = observed;
    clarkEvans[clarkEvansOffset + 2u] = expected;
    clarkEvans[clarkEvansOffset + 3u] = observed / expected;
    clarkEvans[clarkEvansOffset + 4u] = standardError;
    clarkEvans[clarkEvansOffset + 5u] = zScore;
    clarkEvans[clarkEvansOffset + 6u] = getNormalPValue(zScore, u32(parameters[parametersOffset + 6u] + 0.5));
  }`
          })
        );
      }
    }

    if (output.quadratCounts || output.quadratStatistics || output.quadratAreas) {
      const [quadratColumns, quadratRows] = props.quadratGrid!;
      const quadratCount = quadratColumns * quadratRows;
      const counts =
        output.quadratCounts ??
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
      const areaSamples = studyWindow
        ? createTransientView(graph, `${id}-quadrat-area-samples`, 'uint32', quadratCount)
        : undefined;
      const areaTotals = studyWindow
        ? createTransientView(graph, `${id}-quadrat-area-totals`, 'uint32', 3)
        : undefined;
      if (studyWindow && areaSamples && areaTotals) {
        const quadratSamples = props.quadratSamples ?? 16;
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-quadrat-area-samples`,
            operation: OPERATION,
            variant: 'quadrat-area-samples',
            bindings: [
              getStudyWindowBinding(studyWindow),
              {name: 'areaSamples', view: areaSamples, type: 'u32', access: 'read_write'}
            ],
            invocationCount: quadratCount,
            declarations: windowWGSL,
            body: `let minimum = vec2f(studyWindowSummaryValue(0u), studyWindowSummaryValue(1u));
  let size = (vec2f(studyWindowSummaryValue(2u), studyWindowSummaryValue(3u)) - minimum) / vec2f(${quadratColumns}.0, ${quadratRows}.0);
  let cell = vec2f(f32(index % ${quadratColumns}u), f32(index / ${quadratColumns}u));
  var count = 0u;
  for (var sample = 0u; sample < ${quadratSamples ** 2}u; sample++) {
    let fraction = (vec2f(f32(sample % ${quadratSamples}u), f32(sample / ${quadratSamples}u)) + vec2f(0.5)) / ${quadratSamples}.0;
    let point = minimum + (cell + fraction) * size;
    if (isInsideStudyWindow(point)) { count++; }
  }
  areaSamples[areaSamplesOffset + index] = count;`
          }),
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-quadrat-area-totals`,
            operation: OPERATION,
            variant: 'quadrat-area-totals',
            bindings: [
              {name: 'areaSamples', view: areaSamples, type: 'u32', access: 'read'},
              {name: 'areaTotals', view: areaTotals, type: 'u32', access: 'read_write'}
            ],
            invocationCount: 1,
            body: `var total = 0u;
  var count = 0u;
  var equal = true;
  for (var quadrat = 0u; quadrat < ${quadratCount}u; quadrat++) {
    let samples = areaSamples[areaSamplesOffset + quadrat];
    total += samples;
    count += select(0u, 1u, samples > 0u);
    equal = equal && samples == areaSamples[areaSamplesOffset];
  }
  areaTotals[areaTotalsOffset] = total;
  areaTotals[areaTotalsOffset + 1u] = count;
  areaTotals[areaTotalsOffset + 2u] = select(0u, 1u, equal);`
          })
        );
      }
      if (output.quadratAreas) {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-quadrat-areas`,
            operation: OPERATION,
            variant: 'quadrat-areas',
            bindings: [
              {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
              {name: 'quadratAreas', view: output.quadratAreas, type: 'f32', access: 'read_write'},
              ...(studyWindow && areaSamples && areaTotals
                ? [
                    getStudyWindowBinding(studyWindow),
                    {
                      name: 'areaSamples',
                      view: areaSamples,
                      type: 'u32' as const,
                      access: 'read' as const
                    },
                    {
                      name: 'areaTotals',
                      view: areaTotals,
                      type: 'u32' as const,
                      access: 'read' as const
                    }
                  ]
                : [])
            ],
            invocationCount: quadratCount,
            declarations: `${PAIR_STATISTICS_FLOAT_WGSL}${windowWGSL}`,
            body: studyWindow
              ? `let total = areaTotals[areaTotalsOffset];
  var area = getQuietNaN(index);
  if (total > 0u && isStudyWindowUsable()) {
    area = f32(areaSamples[areaSamplesOffset + index]) * studyWindowSummaryValue(4u) / f32(total);
  }
  quadratAreas[quadratAreasOffset + index] = area;`
              : `quadratAreas[quadratAreasOffset + index] =
    (parameters[parametersOffset + 2u] - parameters[parametersOffset]) *
    (parameters[parametersOffset + 3u] - parameters[parametersOffset + 1u]) / ${quadratCount}.0;`
          })
        );
      }
      if (output.quadratStatistics) {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-quadrat-statistics`,
            operation: OPERATION,
            variant: 'quadrat-statistics',
            bindings: [
              {name: 'quadratCounts', view: counts, type: 'u32', access: 'read'},
              ...(areaSamples && areaTotals
                ? [
                    {
                      name: 'areaSamples',
                      view: areaSamples,
                      type: 'u32' as const,
                      access: 'read' as const
                    },
                    {
                      name: 'areaTotals',
                      view: areaTotals,
                      type: 'u32' as const,
                      access: 'read' as const
                    }
                  ]
                : []),
              {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
              {name: 'cellOffsets', view: inputs.cellOffsets, type: 'u32', access: 'read'},
              {
                name: 'quadratStatistics',
                view: output.quadratStatistics,
                type: 'f32',
                access: 'read_write'
              }
            ],
            invocationCount: 1,
            declarations: `${PAIR_STATISTICS_FLOAT_WGSL}
${ERFC_VALUE_WGSL}
${NORMAL_P_VALUE_WGSL}
${CHI_SQUARE_P_VALUE_WGSL}
const CELL_COUNT: u32 = ${cellCount}u;
const QUADRAT_COUNT: u32 = ${quadratCount}u;`,
            body: studyWindow
              ? `let nan = getQuietNaN(index);
  let m = f32(areaTotals[areaTotalsOffset + 1u]);
  let total = f32(areaTotals[areaTotalsOffset]);
  let equalAreas = areaTotals[areaTotalsOffset + 2u] != 0u;
  let n = f32(cellOffsets[cellOffsetsOffset + CELL_COUNT]);
  let mean = select(nan, n / m, m > 0.0);
  var squares = 0.0;
  var chiSquare = 0.0;
  for (var quadrat = 0u; quadrat < QUADRAT_COUNT; quadrat++) {
    let samples = areaSamples[areaSamplesOffset + quadrat];
    if (samples > 0u) {
      let expected = select(n * f32(samples) / total, mean, equalAreas);
      let centered = f32(quadratCounts[quadratCountsOffset + quadrat]) - expected;
      squares += centered * centered;
      chiSquare += centered * centered / expected;
    }
  }
  let variance = select(nan, squares / (m - 1.0), m > 1.0);
  var ratio = select(nan, chiSquare / (m - 1.0), m > 1.0 && n > 0.0);
  if (equalAreas) {
    // Preserve the legacy operations exactly when every quadrat has the same positive area.
    ratio = select(nan, variance / mean, m > 1.0 && mean > 0.0);
    chiSquare = (m - 1.0) * ratio;
  }
  if (!(m > 1.0 && n > 0.0)) { chiSquare = nan; }
  quadratStatistics[quadratStatisticsOffset] = m;
  quadratStatistics[quadratStatisticsOffset + 1u] = mean;
  quadratStatistics[quadratStatisticsOffset + 2u] = variance;
  quadratStatistics[quadratStatisticsOffset + 3u] = ratio;
  quadratStatistics[quadratStatisticsOffset + 4u] = chiSquare;
  quadratStatistics[quadratStatisticsOffset + 5u] = m - 1.0;
  quadratStatistics[quadratStatisticsOffset + 6u] = select(nan,
    getChiSquarePValue(chiSquare, m - 1.0, u32(parameters[parametersOffset + 6u] + 0.5)),
    isFiniteFloat(chiSquare));`
              : `let nan = getQuietNaN(index);
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
  let chiSquare = select(nan, (m - 1.0) * ratio, isFiniteFloat(ratio));
  quadratStatistics[quadratStatisticsOffset + 4u] = chiSquare;
  quadratStatistics[quadratStatisticsOffset + 5u] = m - 1.0;
  quadratStatistics[quadratStatisticsOffset + 6u] = select(
    nan,
    getChiSquarePValue(chiSquare, m - 1.0, u32(parameters[parametersOffset + 6u] + 0.5)),
    isFiniteFloat(chiSquare));`
          })
        );
      }
    }
    return nodes;
  }
}
