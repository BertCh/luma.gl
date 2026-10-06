// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Recipe scenes over planar points and rasters: cluster and outline, spatial regression, drive-time
 * catchments and straight-line catchments, all on the New York trips, roads and points of interest.
 */

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import {
  getGPUNetworkIsochroneParameterValues,
  GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-network';
import {getGPUDistanceFieldParameterValues} from '@luma.gl/experimental/gpu-raster';
import {
  addClusterAndOutlineRecipe,
  addDriveTimeCatchmentRecipe,
  addSpatialRegressionRecipe,
  addStraightLineCatchmentsRecipe,
  getGPUGeographicallyWeightedRegressionParameterValues,
  getGPUGeographicDistributionParameterValues,
  getGPUOrdinaryLeastSquaresParameterValues,
  getGPUSpatialAutocorrelationParameterValues,
  getGPUSpatialClusteringParameterValues,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_R_SQUARED,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_SIGMA_SQUARED,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_MORAN_I,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_MORAN_Z,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_LM_ERROR,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_LM_LAG,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_STRIDE,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {LocalMetricProjection} from '../spatial-analysis-data';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer
} from '../spatial-analysis-layers';
import {formatCount} from '../spatial-analysis-resources';
import {formatCompact} from './classification-layers';
import {
  addKernelPass,
  DirtyEncoder,
  getCoreBounds,
  getQuantile,
  getTripEndpoints,
  RecipeKit,
  SummaryReader,
  sliceSummary,
  type RecipeParameter,
  type RecipeSceneBuilder
} from './recipes-kit';
import {PolylineLayer} from './contours-layers';
import {CellGridLayer} from './regression-layers';
import {findNearestNode, sortEdgesBySource, writeEdgeTravelSeconds} from './road-network-utils';

const CLUSTER_MAXIMUM_COUNT = 128;
const CLUSTER_MAXIMUM_HULL_VERTICES = 48;
const CLUSTER_HULL_CAPACITY = CLUSTER_MAXIMUM_COUNT * CLUSTER_MAXIMUM_HULL_VERTICES;
const CLUSTER_PALETTE = [
  [78, 201, 255, 255],
  [255, 148, 72, 255],
  [189, 122, 255, 255],
  [87, 235, 168, 255],
  [255, 105, 168, 255],
  [245, 220, 87, 255],
  [107, 158, 255, 255],
  [255, 92, 92, 255]
] as const;

/**
 * `addClusterAndOutlineRecipe`: DBSCAN of taxi pickups and drop-offs, per-cluster summaries, convex
 * hulls and their area. Epsilon and the minimum neighborhood size are per-frame parameters.
 */
export const buildClusterAndOutlineScene: RecipeSceneBuilder = async host => {
  const {context} = host;
  const trips = await context.data.getNewYorkTrips();
  context.signal.throwIfAborted();
  const positions = getTripEndpoints(trips.vertexPositions, trips.tripOffsets);
  const pointCount = positions.length / 2;
  const bounds = getCoreBounds(positions, 0);
  const paddedBounds = [bounds[0] - 10, bounds[1] - 10, bounds[2] + 10, bounds[3] + 10] as const;

  let epsilon = 150;
  let minimumPoints = 5;
  const getParameters = () =>
    getGPUSpatialClusteringParameterValues({bounds: paddedBounds, epsilon, minimumPoints});

  const kit = new RecipeKit(context.device, 'recipe-cluster-outline');
  const pointsInput = kit.input('positions', positions, 'float32x2', pointCount);
  const clustering = kit.parameter('clustering-parameters', 'float32', getParameters());
  const geometry = kit.parameter(
    'geometry-parameters',
    'float32',
    getGPUGeographicDistributionParameterValues()
  );
  const labels = kit.output('labels', 'uint32', pointCount);
  const clusterCount = kit.output('cluster-count', 'uint32', 1);
  const counts = kit.output('counts', 'uint32', CLUSTER_MAXIMUM_COUNT);
  const hullOffsets = kit.output('hull-offsets', 'uint32', CLUSTER_MAXIMUM_COUNT + 1);
  const hullCounts = kit.output('hull-counts', 'uint32', CLUSTER_MAXIMUM_COUNT);
  const hullPositions = kit.output('hull-positions', 'float32x2', CLUSTER_HULL_CAPACITY);
  const hullOverflow = kit.output('hull-overflow', 'uint32', 1);
  const areas = kit.output('areas', 'float32', CLUSTER_MAXIMUM_COUNT);
  const perimeters = kit.output('perimeters', 'float32', CLUSTER_MAXIMUM_COUNT);
  const recipe = addClusterAndOutlineRecipe(kit.graph, {
    positions: pointsInput.view,
    clusteringParameters: clustering.view,
    gridSize: [256, 256],
    geometryParameters: geometry.view,
    maximumClusterCount: CLUSTER_MAXIMUM_COUNT,
    maximumVerticesPerHull: CLUSTER_MAXIMUM_HULL_VERTICES,
    hullCapacity: CLUSTER_HULL_CAPACITY,
    labels: labels.view,
    clusterCount: clusterCount.view,
    counts: counts.view,
    hullOffsets: hullOffsets.view,
    hullCounts: hullCounts.view,
    hullPositions: hullPositions.view,
    hullOverflow: hullOverflow.view,
    areas: areas.view,
    perimeters: perimeters.view
  });
  // Display adapter: one drawable edge per hull slot (NaN past the cluster's last vertex).
  const hullEdges = kit.output('hull-edges', 'float32x4', CLUSTER_HULL_CAPACITY);
  const hullEdgeClusters = kit.output('hull-edge-clusters', 'uint32', CLUSTER_HULL_CAPACITY);
  addKernelPass(kit.graph, {
    id: 'cluster-hull-edges',
    count: CLUSTER_HULL_CAPACITY,
    bindings: [
      {name: 'offsets', view: hullOffsets.view, access: 'read', type: 'u32'},
      {name: 'hullCounts', view: hullCounts.view, access: 'read', type: 'u32'},
      {name: 'hullPositions', view: hullPositions.view, access: 'read', type: 'vec2<f32>'},
      {name: 'edges', view: hullEdges.view, access: 'read_write', type: 'vec4<f32>'},
      {name: 'edgeClusters', view: hullEdgeClusters.view, access: 'read_write', type: 'u32'}
    ],
    body: `let nan = bitcast<f32>(0x7fc00000u | (hullCounts[hullCountsOffset] >> 31u));
  var edge = vec4<f32>(nan);
  var owner = 0xffffffffu;
  for (var cluster = 0u; cluster < ${CLUSTER_MAXIMUM_COUNT}u; cluster = cluster + 1u) {
    let first = offsets[offsetsOffset + cluster];
    let count = hullCounts[hullCountsOffset + cluster];
    if (count >= 2u && index >= first && index < first + count) {
      var next = index + 1u;
      if (next >= first + count) {
        next = first;
      }
      edge = vec4<f32>(hullPositions[hullPositionsOffset + index], hullPositions[hullPositionsOffset + next]);
      owner = cluster;
    }
  }
  edges[edgesOffset + index] = edge;
  edgeClusters[edgeClustersOffset + index] = owner;`
  });
  const compiled = kit.compile();

  const clusterBytes = CLUSTER_MAXIMUM_COUNT * 4;
  const summarySizes = [4, clusterBytes, clusterBytes, clusterBytes, 4];
  const reader = new SummaryReader(
    kit.resources,
    'cluster-outline',
    [
      {buffer: clusterCount.buffer, size: 4},
      {buffer: counts.buffer, size: clusterBytes},
      {buffer: areas.buffer, size: clusterBytes},
      {buffer: perimeters.buffer, size: clusterBytes},
      {buffer: hullOverflow.buffer, size: 4}
    ],
    bytes => {
      const [count, memberRows, areaRows, perimeterRows, overflow] = sliceSummary(
        bytes,
        summarySizes
      );
      const shown = Math.min(count.u32[0], CLUSTER_MAXIMUM_COUNT);
      let members = 0;
      let largest = 0;
      let totalArea = 0;
      let longestPerimeter = 0;
      for (let cluster = 0; cluster < shown; cluster++) {
        members += memberRows.u32[cluster];
        largest = Math.max(largest, memberRows.u32[cluster]);
        totalArea += areaRows.f32[cluster];
        longestPerimeter = Math.max(longestPerimeter, perimeterRows.f32[cluster]);
      }
      host.setOutputs([
        [
          'Clusters',
          `${formatCount(count.u32[0])}${count.u32[0] > CLUSTER_MAXIMUM_COUNT ? ` (first ${CLUSTER_MAXIMUM_COUNT} outlined)` : ''}`
        ],
        [
          'Clustered / noise points',
          `${formatCount(members)} / ${formatCount(pointCount - members)}`
        ],
        ['Largest cluster', `${formatCount(largest)} points`],
        ['Total hull area', `${(totalArea / 1e6).toFixed(2)} km2`],
        ['Longest hull perimeter', `${(longestPerimeter / 1000).toFixed(2)} km`],
        ['Hull overflow', overflow.u32[0] ? 'YES' : 'no']
      ]);
    }
  );
  const encoder = new DirtyEncoder(compiled, reader);
  const writeParameters = () => {
    clustering.parameters.write(getParameters());
    encoder.markDirty();
  };
  const parameters: RecipeParameter[] = [
    {
      kind: 'slider',
      label: 'Epsilon (neighbor radius)',
      minimum: 40,
      maximum: 400,
      step: 10,
      value: epsilon,
      format: value => `${value} m`,
      onChange: value => {
        epsilon = value;
        writeParameters();
      }
    },
    {
      kind: 'slider',
      label: 'Minimum points per neighborhood',
      minimum: 2,
      maximum: 20,
      step: 1,
      value: minimumPoints,
      format: value => `${value} points`,
      onChange: value => {
        minimumPoints = value;
        writeParameters();
      }
    }
  ];
  const origin = trips.origin;
  return {
    compiled,
    contributorCount: recipe.contributors.length,
    chain: [
      'GPUSpatialClustering (DBSCAN labels)',
      'GPUGroupGeometry (counts, centers per cluster)',
      'GPUGroupConvexHull (hull per cluster)',
      'GPUGeometryMeasures (hull area, perimeter)',
      'hull edge adapter (display only)'
    ],
    parameters,
    legend: 'Points and convex hulls share a color per cluster; gray points are noise.',
    dataNote: `${trips.attribution}, ${formatCount(pointCount)} pickups and drop-offs`,
    encode: commandEncoder => encoder.encode(commandEncoder),
    getLayers: (): Layer[] => [
      new SpatialAnalysisPointLayer({
        id: 'recipe-cluster-points',
        coordinateOrigin: [origin[0], origin[1], 0],
        positions: pointsInput.buffer,
        instanceCount: pointCount,
        values: labels.buffer,
        valueFormat: 'uint32',
        colormap: 'category',
        palette: CLUSTER_PALETTE,
        noDataColor: [128, 138, 156, 120],
        radiusPixels: 2.2,
        opacity: 0.9
      }),
      new SpatialAnalysisSegmentLayer({
        id: 'recipe-cluster-hulls',
        coordinateOrigin: [origin[0], origin[1], 0],
        segments: hullEdges.buffer,
        instanceCount: CLUSTER_HULL_CAPACITY,
        values: hullEdgeClusters.buffer,
        valueFormat: 'uint32',
        colormap: 'category',
        palette: CLUSTER_PALETTE,
        noDataColor: [0, 0, 0, 0],
        widthPixels: 2.4
      })
    ],
    destroy: () => {
      reader.stop();
      kit.resources.destroy();
    }
  };
};

const REGRESSION_CELL_METERS = 450;
const REGRESSION_LADDER_CAPACITY = 4;
const REGRESSION_MAPS = [
  'OLS residual',
  'Residual local Moran z',
  'GWR coefficient of trip vertices',
  'GWR local R2'
] as const;
const TIMES_SQUARE: readonly [number, number] = [-73.9855, 40.758];

/**
 * `addSpatialRegressionRecipe`: log points of interest around each cell regressed on log taxi-trip
 * vertices and distance to Times Square. OLS, spatial diagnostics (LM tests, residual Moran), a
 * residual local Moran map and geographically weighted fits. The ridge penalty, the Moran
 * significance level and the GWR bandwidth are per-frame parameters; the mapped output is a layer
 * choice.
 */
export const buildSpatialRegressionScene: RecipeSceneBuilder = async host => {
  const {context} = host;
  const [trips, pois] = await Promise.all([
    context.data.getNewYorkTrips(),
    context.data.getNewYorkPointsOfInterest()
  ]);
  context.signal.throwIfAborted();
  const projection = new LocalMetricProjection(trips.origin);
  const bounds = getCoreBounds(pois.positions, 0.03);
  const columns = Math.ceil((bounds[2] - bounds[0]) / REGRESSION_CELL_METERS);
  const rowsOfCells = Math.ceil((bounds[3] - bounds[1]) / REGRESSION_CELL_METERS);
  const cellCount = columns * rowsOfCells;
  const countInCells = (positions: Float32Array) => {
    const counts = new Float32Array(cellCount);
    for (let index = 0; index < positions.length; index += 2) {
      const column = Math.floor((positions[index] - bounds[0]) / REGRESSION_CELL_METERS);
      const row = Math.floor((positions[index + 1] - bounds[1]) / REGRESSION_CELL_METERS);
      if (column >= 0 && column < columns && row >= 0 && row < rowsOfCells) {
        counts[row * columns + column]++;
      }
    }
    return counts;
  };
  const vertexCounts = countInCells(trips.vertexPositions);
  const ownPoiCounts = countInCells(pois.positions);
  // Points of interest in the cell and its eight neighbors, so the predictor varies smoothly.
  const poiCounts = new Float32Array(cellCount);
  for (let cell = 0; cell < cellCount; cell++) {
    const column = cell % columns;
    const row = Math.floor(cell / columns);
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const neighborColumn = column + dx;
        const neighborRow = row + dy;
        if (neighborColumn < 0 || neighborColumn >= columns) continue;
        if (neighborRow < 0 || neighborRow >= rowsOfCells) continue;
        poiCounts[cell] += ownPoiCounts[neighborRow * columns + neighborColumn];
      }
    }
  }

  // Study cells: enough activity, and at least one included queen neighbor (no islands).
  const active = new Uint8Array(cellCount);
  for (let cell = 0; cell < cellCount; cell++) {
    active[cell] = vertexCounts[cell] >= 3 || ownPoiCounts[cell] >= 1 ? 1 : 0;
  }
  const getNeighbors = (cell: number, include: Uint8Array): number[] => {
    const column = cell % columns;
    const row = Math.floor(cell / columns);
    const neighbors: number[] = [];
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        if (dx === 0 && dy === 0) continue;
        const neighborColumn = column + dx;
        const neighborRow = row + dy;
        if (neighborColumn < 0 || neighborColumn >= columns) continue;
        if (neighborRow < 0 || neighborRow >= rowsOfCells) continue;
        const neighbor = neighborRow * columns + neighborColumn;
        if (include[neighbor]) neighbors.push(neighbor);
      }
    }
    return neighbors;
  };
  const included = new Uint8Array(cellCount);
  for (let cell = 0; cell < cellCount; cell++) {
    included[cell] = active[cell] && getNeighbors(cell, active).length > 0 ? 1 : 0;
  }
  const rowOfCell = new Uint32Array(cellCount).fill(0xffffffff);
  const cellOfRow: number[] = [];
  for (let cell = 0; cell < cellCount; cell++) {
    if (included[cell]) {
      rowOfCell[cell] = cellOfRow.length;
      cellOfRow.push(cell);
    }
  }
  const rowCount = cellOfRow.length;
  const predictorCount = 2;
  const response = new Float32Array(rowCount);
  const predictors = new Float32Array(rowCount * predictorCount);
  const centers = new Float32Array(rowCount * 2);
  const [squareX, squareY] = projection.project(...TIMES_SQUARE);
  const offsets = new Uint32Array(rowCount + 1);
  const neighborList: number[] = [];
  const weightList: number[] = [];
  cellOfRow.forEach((cell, row) => {
    const centerX = bounds[0] + ((cell % columns) + 0.5) * REGRESSION_CELL_METERS;
    const centerY = bounds[1] + (Math.floor(cell / columns) + 0.5) * REGRESSION_CELL_METERS;
    centers.set([centerX, centerY], row * 2);
    response[row] = Math.log1p(poiCounts[cell]);
    predictors[row * 2] = Math.log1p(vertexCounts[cell]);
    predictors[row * 2 + 1] = Math.hypot(centerX - squareX, centerY - squareY) / 1000;
    const neighbors = getNeighbors(cell, included);
    for (const neighbor of neighbors) {
      neighborList.push(rowOfCell[neighbor]);
      weightList.push(1 / neighbors.length);
    }
    offsets[row + 1] = neighborList.length;
  });

  let ridge = 0;
  let significanceLevel = 0.05;
  let neighborCount = 40;
  let mapIndex = 0;
  let sigma = 1;
  let coefficientRange = 1;
  const getGwrParameters = () =>
    getGPUGeographicallyWeightedRegressionParameterValues(
      {kernel: 'bisquare', bandwidthMode: 'adaptive', bandwidths: [neighborCount]},
      REGRESSION_LADDER_CAPACITY
    );

  const kit = new RecipeKit(context.device, 'recipe-regression');
  const predictorInput = kit.input('predictors', predictors, 'float32', rowCount * predictorCount);
  const responseInput = kit.input('response', response, 'float32', rowCount);
  const weightOffsets = kit.input('weight-offsets', offsets, 'uint32', rowCount + 1);
  const weightNeighbors = kit.input(
    'weight-neighbors',
    Uint32Array.from(neighborList),
    'uint32',
    neighborList.length
  );
  const weightValues = kit.input(
    'weight-values',
    Float32Array.from(weightList),
    'float32',
    weightList.length
  );
  const centersInput = kit.input('centers', centers, 'float32x2', rowCount);
  const autocorrelation = kit.parameter(
    'autocorrelation-parameters',
    'float32',
    getGPUSpatialAutocorrelationParameterValues({significanceLevel})
  );
  const olsParameters = kit.parameter(
    'ols-parameters',
    'float32',
    getGPUOrdinaryLeastSquaresParameterValues(ridge)
  );
  const gwrParameters = kit.parameter('gwr-parameters', 'float32', getGwrParameters());
  const olsCoefficients = kit.output('ols-coefficients', 'float32', predictorCount + 1);
  const olsSummary = kit.output(
    'ols-summary',
    'float32',
    GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH
  );
  const olsStatus = kit.output('ols-status', 'uint32', 1);
  const residuals = kit.output('residuals', 'float32', rowCount);
  const tests = kit.output('tests', 'float32', GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH);
  const diagnosticsSummary = kit.output(
    'diagnostics-summary',
    'float32',
    GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH
  );
  const diagnosticsStatus = kit.output('diagnostics-status', 'uint32', 1);
  const moranZ = kit.output('moran-z', 'float32', rowCount);
  const localCoefficients = kit.output(
    'local-coefficients',
    'float32',
    rowCount * (predictorCount + 1)
  );
  const localR2 = kit.output('local-r2', 'float32', rowCount);
  const gwrSummary = kit.output(
    'gwr-summary',
    'float32',
    GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH
  );
  const localStatus = kit.output('local-status', 'uint32', rowCount);
  const recipe = addSpatialRegressionRecipe(kit.graph, {
    predictors: predictorInput.view,
    response: responseInput.view,
    predictorCount,
    weights: {
      offsets: weightOffsets.view,
      neighbors: weightNeighbors.view,
      weights: weightValues.view
    },
    parameters: autocorrelation.view,
    olsParameters: olsParameters.view,
    ols: {
      coefficients: olsCoefficients.view,
      summary: olsSummary.view,
      status: olsStatus.view,
      residuals: residuals.view
    },
    diagnostics: {
      tests: tests.view,
      summary: diagnosticsSummary.view,
      status: diagnosticsStatus.view
    },
    residualMoran: {zScores: moranZ.view},
    localFits: {
      positions: centersInput.view,
      parameters: gwrParameters.view,
      maximumBandwidthCount: REGRESSION_LADDER_CAPACITY,
      coefficients: localCoefficients.view,
      localR2: localR2.view,
      localStatus: localStatus.view,
      summary: gwrSummary.view
    }
  });
  const compiled = kit.compile();
  const rowOfCellBuffer = kit.resources.createBuffer('row-of-cell', rowOfCell);

  const sizes = [
    (predictorCount + 1) * 4,
    GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH * 4,
    4,
    GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH * 4,
    GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH * 4,
    4,
    GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH * 4,
    rowCount * (predictorCount + 1) * 4,
    rowCount * 4
  ];
  const reader = new SummaryReader(
    kit.resources,
    'regression',
    [
      {buffer: olsCoefficients.buffer, size: sizes[0]},
      {buffer: olsSummary.buffer, size: sizes[1]},
      {buffer: olsStatus.buffer, size: sizes[2]},
      {buffer: tests.buffer, size: sizes[3]},
      {buffer: diagnosticsSummary.buffer, size: sizes[4]},
      {buffer: diagnosticsStatus.buffer, size: sizes[5]},
      {buffer: gwrSummary.buffer, size: sizes[6]},
      {buffer: localCoefficients.buffer, size: sizes[7]},
      {buffer: localStatus.buffer, size: sizes[8]}
    ],
    bytes => {
      const [
        coefficients,
        summary,
        olsFlag,
        testRows,
        diagnostics,
        diagnosticsFlag,
        gwr,
        local,
        status
      ] = sliceSummary(bytes, sizes);
      sigma = Math.sqrt(
        Math.max(1e-6, summary.f32[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_SIGMA_SQUARED])
      );
      const coefficientValues: number[] = [];
      for (let row = 0; row < rowCount; row++) {
        const value = local.f32[row * (predictorCount + 1) + 1];
        if (Number.isFinite(value)) coefficientValues.push(value);
      }
      const nextRange = Math.max(
        0.05,
        coefficientValues.length ? getQuantile(coefficientValues.map(Math.abs), 0.9) : 1
      );
      let singular = 0;
      for (let row = 0; row < rowCount; row++) if (status.u32[row] !== 0) singular++;
      const lmLag =
        GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_LM_LAG *
        GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_STRIDE;
      const lmError =
        GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_LM_ERROR *
        GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_STRIDE;
      host.setOutputs([
        [
          'OLS fit (R2, n)',
          `${summary.f32[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_R_SQUARED].toFixed(3)}, ${formatCount(rowCount)} cells${olsFlag.u32[0] ? ' (status ' + olsFlag.u32[0] + ')' : ''}`
        ],
        [
          'Coefficients (const, trips, km)',
          Array.from(coefficients.f32.subarray(0, predictorCount + 1), value =>
            value.toFixed(2)
          ).join(' / ')
        ],
        [
          'LM lag / LM error (p)',
          `${formatCompact(testRows.f32[lmLag])} (${testRows.f32[lmLag + 2].toFixed(3)}) / ${formatCompact(testRows.f32[lmError])} (${testRows.f32[lmError + 2].toFixed(3)})`
        ],
        [
          'Residual Moran I (z)',
          `${diagnostics.f32[GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_MORAN_I].toFixed(3)} (${diagnostics.f32[GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_MORAN_Z].toFixed(1)})${diagnosticsFlag.u32[0] ? ' status ' + diagnosticsFlag.u32[0] : ''}`
        ],
        [
          'GWR R2 / AICc',
          Number.isFinite(gwr.f32[GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY.AICC])
            ? `${gwr.f32[GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY.R_SQUARED].toFixed(3)} / ${gwr.f32[GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY.AICC].toFixed(1)}`
            : 'undefined (a local fit is singular)'
        ],
        ['GWR rows without a local fit', `${singular} of ${rowCount}`]
      ]);
      if (Math.abs(nextRange - coefficientRange) > coefficientRange * 0.1) {
        coefficientRange = nextRange;
      }
      host.updateLayers();
    }
  );
  const encoder = new DirtyEncoder(compiled, reader);
  const parameters: RecipeParameter[] = [
    {
      kind: 'slider',
      label: 'Ridge penalty (OLS)',
      minimum: 0,
      maximum: 200,
      step: 10,
      value: ridge,
      format: value => (value === 0 ? 'none' : `lambda ${value}`),
      onChange: value => {
        ridge = value;
        olsParameters.parameters.write(getGPUOrdinaryLeastSquaresParameterValues(ridge));
        encoder.markDirty();
      }
    },
    {
      kind: 'slider',
      label: 'Moran significance level',
      minimum: 0.01,
      maximum: 0.2,
      step: 0.01,
      value: significanceLevel,
      format: value => `p <= ${value.toFixed(2)}`,
      onChange: value => {
        significanceLevel = value;
        autocorrelation.parameters.write(
          getGPUSpatialAutocorrelationParameterValues({significanceLevel})
        );
        encoder.markDirty();
      }
    },
    {
      kind: 'slider',
      label: 'GWR bandwidth (nearest cells, bisquare)',
      minimum: 15,
      maximum: 120,
      step: 5,
      value: neighborCount,
      format: value => `${value} cells`,
      onChange: value => {
        neighborCount = value;
        gwrParameters.parameters.write(getGwrParameters());
        encoder.markDirty();
      }
    },
    {
      kind: 'slider',
      label: 'Map (layer choice)',
      minimum: 0,
      maximum: REGRESSION_MAPS.length - 1,
      step: 1,
      value: mapIndex,
      format: value => REGRESSION_MAPS[value],
      onChange: value => {
        mapIndex = value;
        host.updateLayers();
      }
    }
  ];
  const origin = trips.origin;
  return {
    compiled,
    contributorCount: recipe.contributors.length,
    chain: [
      'GPUOrdinaryLeastSquares (coefficients, residuals)',
      'GPUSpatialRegressionDiagnostics (LM tests, Moran)',
      'GPULocalMoran (residual map)',
      'GPUGeographicallyWeightedRegression (local fits)'
    ],
    parameters,
    legend:
      'Cells of 450 m with taxi activity. Red positive, blue negative (diverging) for residuals, Moran z and the local POI coefficient; viridis for local R2.',
    dataNote:
      `${trips.attribution}; ${pois.attribution}. Model: log(1 + POIs in the 3 x 3 cell block) ~ log(1 + trip vertices) + km ` +
      'from Times Square, queen contiguity of included cells',
    encode: commandEncoder => encoder.encode(commandEncoder),
    getLayers: (): Layer[] => {
      const common = {
        coordinateOrigin: [origin[0], origin[1], 0] as [number, number, number],
        positionOffset: [bounds[0], bounds[1]] as [number, number],
        cellSize: REGRESSION_CELL_METERS,
        columns,
        cellCount,
        indices: rowOfCellBuffer,
        color: [255, 255, 255, 215] as [number, number, number, number]
      };
      if (mapIndex === 0) {
        return [
          new CellGridLayer({
            id: 'recipe-regression-residual',
            ...common,
            values: residuals.buffer,
            colormap: 'diverging',
            valueRange: [-2 * sigma, 2 * sigma]
          })
        ];
      }
      if (mapIndex === 1) {
        return [
          new CellGridLayer({
            id: 'recipe-regression-moran',
            ...common,
            values: moranZ.buffer,
            colormap: 'diverging',
            valueRange: [-3, 3]
          })
        ];
      }
      if (mapIndex === 2) {
        return [
          new CellGridLayer({
            id: 'recipe-regression-coefficient',
            ...common,
            values: localCoefficients.buffer,
            valueStride: predictorCount + 1,
            valueOffset: 1,
            colormap: 'diverging',
            valueRange: [-coefficientRange, coefficientRange]
          })
        ];
      }
      return [
        new CellGridLayer({
          id: 'recipe-regression-local-r2',
          ...common,
          values: localR2.buffer,
          colormap: 'viridis',
          valueRange: [0, 1]
        })
      ];
    },
    destroy: () => {
      reader.stop();
      kit.resources.destroy();
    }
  };
};

const DRIVE_BAND_COUNT = 4;
const DRIVE_FACILITY_COUNT = 3;
const DRIVE_CELL_RESOLUTION = 16;
const DRIVE_SEGMENT_CAPACITY = 200_000;
const DRIVE_RING_CAPACITY = 2048;
const DRIVE_RING_VERTEX_CAPACITY = 65_536;
const DRIVE_SHELL_COLOR = [255, 255, 255, 255] as const;
const DRIVE_HOLE_COLOR = [255, 60, 200, 255] as const;
const DRIVE_BAND_COLORS = [
  [70, 205, 150, 255],
  [255, 214, 80, 255],
  [255, 140, 60, 255],
  [220, 70, 70, 255]
] as const;

/**
 * `addDriveTimeCatchmentRecipe`: three facilities and taxi pickups and drop-offs as demand are
 * snapped to the New York street graph, multi-source drive times are relaxed on the GPU and each
 * demand point lands in a drive-time band. Click the map to move the next facility; the time
 * budget scales the four band breaks. The recipe also builds the isochrone of the whole budget as
 * closed rings (Quadbin tiles that hold reached nodes, outlined and chained by the GPU); shells
 * are white and holes magenta. Cell outlines need longitude/latitude, so the whole recipe runs in
 * degrees: snapping measures distance in degrees (about 24% narrower east-west at this latitude),
 * which can pick a neighboring street over the nearest one but does not change drive times.
 */
export const buildDriveTimeScene: RecipeSceneBuilder = async host => {
  const {context} = host;
  const [roads, trips] = await Promise.all([
    context.data.getNewYorkRoads(),
    context.data.getNewYorkTrips()
  ]);
  context.signal.throwIfAborted();
  const projection = new LocalMetricProjection(roads.origin);
  const nodeCount = roads.nodePositions.length / 2;
  const edges = sortEdgesBySource(roads);
  const edgeCount = edges.sources.length;
  const offsets = new Uint32Array(nodeCount + 1);
  for (let edge = 0; edge < edgeCount; edge++) offsets[edges.sources[edge] + 1]++;
  for (let node = 0; node < nodeCount; node++) offsets[node + 1] += offsets[node];
  const weights = new Float32Array(edgeCount);
  writeEdgeTravelSeconds(edges, 'drive', weights);
  const demandPositions = getTripEndpoints(trips.vertexPositions, trips.tripOffsets);
  const demandCount = demandPositions.length / 2;
  const demandValues = new Float32Array(demandCount);
  for (let trip = 0; trip < trips.vendors.length; trip++) {
    const length = trips.tripOffsets[trip + 1] - trips.tripOffsets[trip];
    demandValues[trip * 2] = length;
    demandValues[trip * 2 + 1] = length;
  }

  let budgetMinutes = 12;
  const facilityPositions = new Float32Array(DRIVE_FACILITY_COUNT * 2);
  const defaults: readonly (readonly [number, number])[] = [
    [-73.9855, 40.758],
    [-73.9442, 40.7831],
    [-74.0099, 40.7118]
  ];
  defaults.forEach(([longitude, latitude], index) => {
    const [x, y] = projection.project(longitude, latitude);
    facilityPositions.set([x, y], index * 2);
  });
  let nextFacility = 0;

  // The recipe runs in longitude/latitude so the isochrone cell outline and rings are geographic.
  const nodeLngLat = new Float32Array(nodeCount * 2);
  for (let node = 0; node < nodeCount; node++) {
    nodeLngLat.set(
      projection.unproject(roads.nodePositions[node * 2], roads.nodePositions[node * 2 + 1]),
      node * 2
    );
  }
  const demandLngLat = new Float32Array(demandCount * 2);
  for (let point = 0; point < demandCount; point++) {
    demandLngLat.set(
      projection.unproject(demandPositions[point * 2], demandPositions[point * 2 + 1]),
      point * 2
    );
  }
  const facilityLngLat = new Float32Array(DRIVE_FACILITY_COUNT * 2);
  facilityPositions.forEach((_, index) => {
    if (index % 2 === 0) {
      facilityLngLat.set(
        projection.unproject(facilityPositions[index], facilityPositions[index + 1]),
        index
      );
    }
  });

  const kit = new RecipeKit(context.device, 'recipe-drive-time');
  const csrOffsets = kit.input('csr-offsets', offsets, 'uint32', nodeCount + 1);
  const csrNeighbors = kit.input('csr-neighbors', edges.targets, 'uint32', edgeCount);
  const csrWeights = kit.input('csr-weights', weights, 'float32', edgeCount);
  const nodePositions = kit.input('node-positions', nodeLngLat, 'float32x2', nodeCount);
  const facilities = kit.input('facilities', facilityLngLat, 'float32x2', DRIVE_FACILITY_COUNT);
  const demand = kit.input('demand', demandLngLat, 'float32x2', demandCount);
  const demandValueInput = kit.input('demand-values', demandValues, 'float32', demandCount);
  const bandBreaks = kit.parameter('band-breaks', 'float32', new Float32Array(DRIVE_BAND_COUNT));
  const costLimit = kit.parameter('cost-limit', 'float32', new Float32Array(1));
  const nodeCosts = kit.output('node-costs', 'float32', nodeCount);
  const demandTimes = kit.output('demand-times', 'float32', demandCount);
  const demandBands = kit.output('demand-bands', 'uint32', demandCount);
  const bandKeys = kit.output('band-keys', 'uint32', 8);
  const bandCounts = kit.output('band-counts', 'uint32', 8);
  const bandCount = kit.output('band-count', 'uint32', 1);
  const bandOverflow = kit.output('band-overflow', 'uint32', 1);
  const bandSums = kit.output('band-sums', 'float32', 8);
  const isochroneBreaks = kit.parameter(
    'isochrone-breaks',
    'float32',
    new Float32Array(DRIVE_BAND_COUNT)
  );
  const isochroneParameters = kit.parameter(
    'isochrone-parameters',
    'float32',
    new Float32Array(GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH)
  );
  const tableCells = kit.output('isochrone-table-cells', 'uint32x2', nodeCount);
  const tableCounts = kit.output('isochrone-table-counts', 'uint32', nodeCount);
  const tableCount = kit.output('isochrone-table-count', 'uint32', 1);
  const tableOverflow = kit.output('isochrone-table-overflow', 'uint32', 1);
  const outlineRows = kit.output('outline-rows', 'uint32', DRIVE_SEGMENT_CAPACITY);
  const outlineCells = kit.output('outline-cells', 'uint32x2', DRIVE_SEGMENT_CAPACITY);
  const outlineEdges = kit.output('outline-edges', 'uint32', DRIVE_SEGMENT_CAPACITY);
  const outlineEndpoints = kit.output('outline-endpoints', 'float32x4', DRIVE_SEGMENT_CAPACITY);
  const outlineCount = kit.output('outline-count', 'uint32', 1);
  const outlineOverflow = kit.output('outline-overflow', 'uint32', 1);
  const ringOffsets = kit.output('ring-offsets', 'uint32', DRIVE_RING_CAPACITY + 1);
  const ringPositions = kit.output('ring-positions', 'float32x2', DRIVE_RING_VERTEX_CAPACITY);
  const ringIsHole = kit.output('ring-is-hole', 'uint32', DRIVE_RING_CAPACITY);
  const ringCount = kit.output('ring-count', 'uint32', 1);
  const ringOverflow = kit.output('ring-overflow', 'uint32', 1);
  const ringOpen = kit.output('ring-open', 'uint32', 1);
  const ringStyle = kit.resources.createBuffer('ring-style', Uint32Array.of(0, 1));
  const recipe = addDriveTimeCatchmentRecipe(kit.graph, {
    network: {
      offsets: csrOffsets.view,
      neighbors: csrNeighbors.view,
      weights: csrWeights.view,
      nodePositions: nodePositions.view
    },
    facilities: facilities.view,
    demand: demand.view,
    demandValues: demandValueInput.view,
    costLimit: costLimit.view,
    maxIterations: 64,
    bandBreaks: bandBreaks.view,
    bandCapacity: 8,
    nodeCosts: nodeCosts.view,
    demandTimes: demandTimes.view,
    demandBands: demandBands.view,
    bands: {
      keys: bandKeys.view,
      counts: bandCounts.view,
      count: bandCount.view,
      overflow: bandOverflow.view,
      sumValues: bandSums.view
    },
    isochrones: {
      breaks: isochroneBreaks.view,
      parameters: isochroneParameters.view,
      cellOutline: {
        family: 'quadbin',
        resolution: DRIVE_CELL_RESOLUTION,
        table: {
          cells: tableCells.view,
          counts: tableCounts.view,
          count: tableCount.view,
          overflow: tableOverflow.view
        },
        output: {
          rows: outlineRows.view,
          cells: outlineCells.view,
          edgeIndices: outlineEdges.view,
          endpoints: outlineEndpoints.view,
          count: outlineCount.view,
          overflow: outlineOverflow.view
        },
        rings: {
          normalizeWinding: true,
          output: {
            ringOffsets: ringOffsets.view,
            positions: ringPositions.view,
            ringIsHole: ringIsHole.view,
            count: ringCount.view,
            overflow: ringOverflow.view,
            openSegmentCount: ringOpen.view
          }
        }
      }
    }
  });
  const compiled = kit.compile();
  const segments = kit.resources.createBuffer('segments', roads.segments);
  const segmentNodes = kit.resources.createBuffer('segment-nodes', roads.segmentNodes);

  const writeBudget = () => {
    const limit = budgetMinutes * 60;
    bandBreaks.parameters.write(
      Float32Array.from(
        {length: DRIVE_BAND_COUNT},
        (_, band) => ((band + 1) / DRIVE_BAND_COUNT) * limit
      )
    );
    costLimit.parameters.write(Float32Array.of(limit * 1.05));
    isochroneBreaks.parameters.write(
      Float32Array.from(
        {length: DRIVE_BAND_COUNT},
        (_, band) => ((band + 1) / DRIVE_BAND_COUNT) * limit
      )
    );
    isochroneParameters.parameters.write(
      getGPUNetworkIsochroneParameterValues({
        breakCount: DRIVE_BAND_COUNT,
        extent: [-180, -90, 180, 90],
        cellCostLimit: limit
      })
    );
    encoder.markDirty();
  };
  const bandBytes = 8 * 4;
  const summarySizes = [bandBytes, bandBytes, 4, bandBytes, 4, 4, 4, 4, 4];
  const reader = new SummaryReader(
    kit.resources,
    'drive-time',
    [
      {buffer: bandKeys.buffer, size: bandBytes},
      {buffer: bandCounts.buffer, size: bandBytes},
      {buffer: bandCount.buffer, size: 4},
      {buffer: bandSums.buffer, size: bandBytes},
      {buffer: bandOverflow.buffer, size: 4},
      {buffer: ringCount.buffer, size: 4},
      {buffer: ringOverflow.buffer, size: 4},
      {buffer: ringOpen.buffer, size: 4},
      {buffer: outlineOverflow.buffer, size: 4}
    ],
    bytes => {
      const [keys, counts, count, sums, overflow, rings, ringFlag, open, outlineFlag] =
        sliceSummary(bytes, summarySizes);
      const perBand = new Array(DRIVE_BAND_COUNT).fill(0);
      const valuePerBand = new Array(DRIVE_BAND_COUNT).fill(0);
      let inside = 0;
      for (let row = 0; row < Math.min(count.u32[0], 8); row++) {
        if (keys.u32[row] < DRIVE_BAND_COUNT) {
          perBand[keys.u32[row]] = counts.u32[row];
          valuePerBand[keys.u32[row]] = sums.f32[row];
          inside += counts.u32[row];
        }
      }
      const band = (index: number) =>
        `${Math.round(((index + 1) / DRIVE_BAND_COUNT) * budgetMinutes * 10) / 10} min`;
      host.setOutputs([
        ['Demand within the budget', `${formatCount(inside)} of ${formatCount(demandCount)}`],
        ...perBand
          .slice(0, 4)
          .map(
            (value, index) =>
              [
                `Band ${index + 1} (< ${band(index)})`,
                `${formatCount(value)} demand, ${formatCompact(valuePerBand[index])} value`
              ] as const
          ),
        ['Band table overflow', overflow.u32[0] ? 'YES' : 'no'],
        [
          'Isochrone rings (whole budget)',
          `${formatCount(rings.u32[0])}, ${formatCount(open.u32[0])} open segments`
        ],
        [
          'Isochrone overflow (outline / rings)',
          `${outlineFlag.u32[0] ? 'YES' : 'no'} / ${ringFlag.u32[0] ? 'YES' : 'no'}`
        ]
      ]);
    }
  );
  const encoder = new DirtyEncoder(compiled, reader);
  writeBudget();
  const origin = roads.origin;
  const parameters: RecipeParameter[] = [
    {
      kind: 'slider',
      label: 'Drive-time budget (scales the four bands)',
      minimum: 2,
      maximum: 30,
      step: 1,
      value: budgetMinutes,
      format: value => `${value} min`,
      onChange: value => {
        budgetMinutes = value;
        writeBudget();
        host.updateLayers();
      }
    }
  ];
  return {
    compiled,
    contributorCount: recipe.contributors.length,
    chain: [
      'GPUNetworkSnapping (facilities to edges)',
      'GPUNetworkServiceAreas (multi-source costs)',
      'GPUNetworkIsochrones (Quadbin cell outline, ring assembly)',
      'GPUNetworkSnapping (demand to edges)',
      'demand time and band adapters',
      'GPUGroupStatistics (demand per band)'
    ],
    parameters,
    legend:
      'Roads colored by drive time to the nearest facility (A, B, C); demand points by band: green fastest to red slowest, gray beyond the budget. White outlines (magenta holes) bound the whole-budget isochrone. Click the map to move the next facility.',
    dataNote:
      `${roads.attribution}; ${trips.attribution} as demand (${formatCount(demandCount)} pickups ` +
      'and drop-offs). Facilities start at Times Square, Upper East Side and Wall Street',
    encode: commandEncoder => encoder.encode(commandEncoder),
    onClick: event => {
      if (!event.coordinate) return false;
      const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
      const node = findNearestNode(roads.nodePositions, [x, y]);
      const row = nextFacility * 2;
      facilityPositions.set(
        [roads.nodePositions[node * 2], roads.nodePositions[node * 2 + 1]],
        row
      );
      facilityLngLat.set([nodeLngLat[node * 2], nodeLngLat[node * 2 + 1]], row);
      nextFacility = (nextFacility + 1) % DRIVE_FACILITY_COUNT;
      facilities.buffer.write(facilityLngLat);
      encoder.markDirty();
      host.updateLayers();
      return true;
    },
    getLayers: (): Layer[] => [
      new SpatialAnalysisSegmentLayer({
        id: 'recipe-drive-roads',
        coordinateOrigin: [origin[0], origin[1], 0],
        segments,
        instanceCount: roads.segmentNodes.length,
        values: nodeCosts.buffer,
        valueFormat: 'float32',
        valueIndices: segmentNodes,
        colormap: 'inferno',
        valueRange: [0, budgetMinutes * 60],
        noDataColor: [110, 115, 130, 60],
        widthPixels: 1.6,
        opacity: 0.9
      }),
      new SpatialAnalysisPointLayer({
        id: 'recipe-drive-demand',
        coordinateOrigin: [origin[0], origin[1], 0],
        coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
        positions: demand.buffer,
        instanceCount: demandCount,
        values: demandBands.buffer,
        valueFormat: 'uint32',
        colormap: 'category',
        palette: DRIVE_BAND_COLORS,
        noDataColor: [150, 150, 160, 90],
        radiusPixels: 2.4
      }),
      new SpatialAnalysisPointLayer({
        id: 'recipe-drive-facilities',
        coordinateOrigin: [origin[0], origin[1], 0],
        coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
        positions: facilities.buffer,
        instanceCount: DRIVE_FACILITY_COUNT,
        color: [255, 255, 255, 255],
        radiusPixels: 9
      }),
      new PolylineLayer({
        id: 'recipe-drive-isochrone-rings',
        coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
        segments: ringPositions.buffer,
        polylineOffsets: ringOffsets.buffer,
        valueIndices: ringIsHole.buffer,
        extent: ringCount.buffer,
        values: ringStyle,
        valueFormat: 'uint32',
        colormap: 'category',
        palette: [DRIVE_SHELL_COLOR, DRIVE_HOLE_COLOR],
        instanceCount: DRIVE_RING_VERTEX_CAPACITY,
        widthPixels: 3
      })
    ],
    getTooltip: () => null,
    destroy: () => {
      reader.stop();
      kit.resources.destroy();
    }
  };
};

const CATCHMENT_WIDTH = 200;
const CATCHMENT_HEIGHT = 240;
const CATCHMENT_CAPACITY = 32;
const CATCHMENT_CATEGORIES = ['hospital', 'fire_station', 'police', 'school', 'library', 'bank'];

/**
 * `addStraightLineCatchmentsRecipe`: nearest-facility zones (raster Voronoi) of points of interest
 * and the taxi-trip density each zone holds. The number of active facilities and the maximum
 * distance are per-frame; clicking the map adds a facility.
 */
export const buildStraightLineScene: RecipeSceneBuilder = async host => {
  const {context} = host;
  const [pois, trips] = await Promise.all([
    context.data.getNewYorkPointsOfInterest(),
    context.data.getNewYorkTrips()
  ]);
  context.signal.throwIfAborted();
  const projection = new LocalMetricProjection(trips.origin);
  const bounds = getCoreBounds(pois.positions, 0.03);
  const cellWidth = (bounds[2] - bounds[0]) / CATCHMENT_WIDTH;
  const cellHeight = (bounds[3] - bounds[1]) / CATCHMENT_HEIGHT;

  const categoryCounts = new Map<string, number>();
  for (const category of pois.categories) {
    const name = pois.categoryNames[category];
    categoryCounts.set(name, (categoryCounts.get(name) ?? 0) + 1);
  }
  const category =
    CATCHMENT_CATEGORIES.find(name => (categoryCounts.get(name) ?? 0) >= 8) ??
    Array.from(categoryCounts.entries()).sort((a, b) => b[1] - a[1])[0][0];
  const seedPositions = new Float32Array(CATCHMENT_CAPACITY * 2);
  let activeSeeds = 0;
  for (let index = 0; index < pois.categories.length && activeSeeds < 24; index++) {
    if (pois.categoryNames[pois.categories[index]] !== category) continue;
    const x = pois.positions[index * 2];
    const y = pois.positions[index * 2 + 1];
    if (x < bounds[0] || x > bounds[2] || y < bounds[1] || y > bounds[3]) continue;
    seedPositions.set([x, y], activeSeeds * 2);
    activeSeeds++;
  }
  // Trip density raster: the value summarized per catchment.
  const density = new Float32Array(CATCHMENT_WIDTH * CATCHMENT_HEIGHT);
  for (let index = 0; index < trips.vertexPositions.length; index += 2) {
    const column = Math.floor((trips.vertexPositions[index] - bounds[0]) / cellWidth);
    const row = Math.floor((trips.vertexPositions[index + 1] - bounds[1]) / cellHeight);
    if (column >= 0 && column < CATCHMENT_WIDTH && row >= 0 && row < CATCHMENT_HEIGHT) {
      density[row * CATCHMENT_WIDTH + column]++;
    }
  }

  let maximumDistance = 6000;
  const getSettings = () =>
    getGPUDistanceFieldParameterValues({
      origin: [bounds[0], bounds[1]],
      cellSize: [cellWidth, cellHeight],
      maxDistance: maximumDistance
    });

  const kit = new RecipeKit(context.device, 'recipe-straight-line');
  const settings = kit.parameter('distance-settings', 'float32', getSettings());
  const seeds = kit.input('seeds', seedPositions, 'float32x2', CATCHMENT_CAPACITY);
  const seedCount = kit.parameter('seed-count', 'uint32', Uint32Array.of(activeSeeds));
  const values = kit.input('values', density, 'float32', density.length);
  const allocation = kit.output('allocation', 'uint32', density.length);
  const distances = kit.output('distances', 'float32', density.length);
  const cellCounts = kit.output('cell-counts', 'uint32', CATCHMENT_CAPACITY);
  const valueCounts = kit.output('value-counts', 'uint32', CATCHMENT_CAPACITY);
  const sums = kit.output('sums', 'float32', CATCHMENT_CAPACITY);
  const means = kit.output('means', 'float32', CATCHMENT_CAPACITY);
  const minimums = kit.output('minimums', 'float32', CATCHMENT_CAPACITY);
  const maximums = kit.output('maximums', 'float32', CATCHMENT_CAPACITY);
  const recipe = addStraightLineCatchmentsRecipe(kit.graph, {
    width: CATCHMENT_WIDTH,
    height: CATCHMENT_HEIGHT,
    settings: settings.view,
    seedPositions: seeds.view,
    seedCount: seedCount.view,
    values: values.view,
    allocation: allocation.view,
    distances: distances.view,
    statistics: {
      cellCounts: cellCounts.view,
      valueCounts: valueCounts.view,
      sums: sums.view,
      means: means.view,
      minimums: minimums.view,
      maximums: maximums.view
    }
  });
  const compiled = kit.compile();

  const rowBytes = CATCHMENT_CAPACITY * 4;
  const summarySizes = [rowBytes, rowBytes];
  const reader = new SummaryReader(
    kit.resources,
    'straight-line',
    [
      {buffer: cellCounts.buffer, size: rowBytes},
      {buffer: sums.buffer, size: rowBytes}
    ],
    bytes => {
      const [cellRows, sumRows] = sliceSummary(bytes, summarySizes);
      let covered = 0;
      let largestCells = 0;
      let busiest = 0;
      let busiestSum = 0;
      let totalSum = 0;
      for (let seed = 0; seed < activeSeeds; seed++) {
        covered += cellRows.u32[seed];
        largestCells = Math.max(largestCells, cellRows.u32[seed]);
        totalSum += sumRows.f32[seed];
        if (sumRows.f32[seed] > busiestSum) {
          busiestSum = sumRows.f32[seed];
          busiest = seed;
        }
      }
      host.setOutputs([
        ['Facilities', `${activeSeeds} (${category.replace('_', ' ')})`],
        [
          'Cells allocated',
          `${formatCount(covered)} of ${formatCount(density.length)} (${((100 * covered) / density.length).toFixed(0)}%)`
        ],
        ['Largest catchment', `${formatCount(largestCells)} cells`],
        [
          'Busiest catchment',
          `#${busiest + 1}: ${formatCompact(busiestSum)} of ${formatCompact(totalSum)} trip vertices`
        ]
      ]);
    }
  );
  const encoder = new DirtyEncoder(compiled, reader);
  const parameters: RecipeParameter[] = [
    {
      kind: 'slider',
      label: 'Active facilities',
      minimum: 1,
      maximum: CATCHMENT_CAPACITY,
      step: 1,
      value: activeSeeds,
      format: value => `${value}`,
      onChange: value => {
        activeSeeds = value;
        seedCount.parameters.write(Uint32Array.of(activeSeeds));
        encoder.markDirty();
        host.updateLayers();
      }
    },
    {
      kind: 'slider',
      label: 'Maximum distance',
      minimum: 500,
      maximum: 8000,
      step: 250,
      value: maximumDistance,
      format: value => `${(value / 1000).toFixed(2)} km`,
      onChange: value => {
        maximumDistance = value;
        settings.parameters.write(getSettings());
        encoder.markDirty();
      }
    }
  ];
  const origin = trips.origin;
  return {
    compiled,
    contributorCount: recipe.contributors.length,
    chain: [
      'GPUDistanceField (Euclidean allocation, raster Voronoi)',
      'GPURasterZonalStatistics (trip density per catchment)'
    ],
    parameters,
    legend:
      'Cells colored by their nearest facility; white dots are the facilities. Click the map to add one.',
    dataNote:
      `${pois.attribution} (${category.replace('_', ' ')} facilities); ${trips.attribution} for the ` +
      `demand raster (trip vertices per ${Math.round(cellWidth)} m cell)`,
    encode: commandEncoder => encoder.encode(commandEncoder),
    onClick: event => {
      if (!event.coordinate || activeSeeds >= CATCHMENT_CAPACITY) return false;
      const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
      seedPositions.set([x, y], activeSeeds * 2);
      activeSeeds++;
      seeds.buffer.write(seedPositions);
      seedCount.parameters.write(Uint32Array.of(activeSeeds));
      encoder.markDirty();
      host.updateLayers();
      return true;
    },
    getLayers: (): Layer[] => [
      new SpatialAnalysisRasterLayer({
        id: 'recipe-straight-line-zones',
        coordinateOrigin: [origin[0], origin[1], 0],
        gridSize: [CATCHMENT_WIDTH, CATCHMENT_HEIGHT],
        bounds,
        rowOrigin: 'south',
        values: allocation.buffer,
        valueFormat: 'uint32',
        colormap: 'category',
        noDataValue: 0xffffffff,
        noDataColor: [0, 0, 0, 0],
        opacity: 0.45
      }),
      new SpatialAnalysisPointLayer({
        id: 'recipe-straight-line-seeds',
        coordinateOrigin: [origin[0], origin[1], 0],
        positions: seeds.buffer,
        instanceCount: activeSeeds,
        color: [255, 255, 255, 255],
        radiusPixels: 5
      })
    ],
    destroy: () => {
      reader.stop();
      kit.resources.destroy();
    }
  };
};
