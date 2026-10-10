// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUSpatialAnalysisCapability,
  GPUSpatialAnalysisCapabilityConnections,
  GPUSpatialAnalysisCapabilityEvidence,
  GPUSpatialAnalysisCapabilityQuery
} from './spatial-analysis-ontology';

const AVAILABLE = 'available' as const;
const OPPORTUNITY = 'opportunity' as const;

/**
 * Runtime catalogue over the flat public exports. It intentionally stores export names rather than
 * constructors so importing discovery metadata does not pull every contributor into an application.
 */
const CAPABILITY_DEFINITIONS: readonly GPUSpatialAnalysisCapability[] = [
  {
    id: 'dynamic-parameters',
    title: 'Dynamic graph parameters',
    status: AVAILABLE,
    domain: 'execution',
    stage: 'bridge',
    summary: 'Writes small per-frame values into one graph view without changing graph topology.',
    inputs: [],
    outputs: ['parameters'],
    coordinateModels: ['none'],
    outputBound: 'fixed-sized',
    operators: ['GPUParameterBuffer'],
    relatedCapabilities: ['spatial-neighborhoods', 'workflow-recipes']
  },
  {
    id: 'geometry-measurement',
    title: 'Geometry measurement',
    status: AVAILABLE,
    domain: 'geometry',
    stage: 'analyze',
    summary: 'Measures feature area, length, perimeter, bounds, centroids and shape descriptors.',
    inputs: ['points', 'lines', 'polygons', 'attributes'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['planar', 'spherical'],
    outputBound: 'source-sized',
    operators: [
      'GPUGeometryMeasures',
      'GPUShapeDescriptors',
      'GPUMinimumBounds',
      'GPUMinimumClearance',
      'GPULabelPoint',
      'GPUGeographicDistribution'
    ],
    relatedCapabilities: ['geometry-validation', 'group-and-regional-analysis']
  },
  {
    id: 'geodesic-operations',
    title: 'Geodesic and rhumb operations',
    status: AVAILABLE,
    domain: 'geometry',
    stage: 'analyze',
    summary:
      'Computes distance, bearing, midpoint and destination pairs on spherical or ellipsoidal models.',
    inputs: ['points', 'parameters'],
    outputs: ['points', 'attributes'],
    coordinateModels: ['spherical', 'ellipsoidal'],
    outputBound: 'source-sized',
    operators: ['GPUGeodesicPairs', 'GPUGeodesicDestination'],
    relatedCapabilities: ['line-processing', 'spatial-neighborhoods']
  },
  {
    id: 'geometry-editing',
    title: 'Geometry editing',
    status: AVAILABLE,
    domain: 'geometry',
    stage: 'transform',
    summary: 'Transforms, reorients, cleans, snaps and offsets existing feature coordinates.',
    inputs: ['points', 'lines', 'polygons', 'parameters'],
    outputs: ['points', 'lines', 'polygons'],
    coordinateModels: ['planar'],
    outputBound: 'mixed',
    operators: [
      'GPUAffineTransform',
      'GPUGeometryOrientation',
      'GPUGeometryCleanup',
      'GPUVertexSnap',
      'GPUOffsetCurve'
    ],
    relatedCapabilities: ['geometry-validation', 'line-processing'],
    limitation:
      'Offset curves are not general polygon buffers and cleanup is not a make-valid operation.'
  },
  {
    id: 'geometry-construction',
    title: 'Geometry construction and sampling',
    status: AVAILABLE,
    domain: 'geometry',
    stage: 'construct',
    summary:
      'Builds grids, analytic shapes, random samples, polygon triangles and draw-oriented outlines.',
    inputs: ['points', 'lines', 'polygons', 'parameters', 'attributes'],
    outputs: ['points', 'lines', 'polygons', 'render-geometry'],
    coordinateModels: ['planar', 'spherical'],
    outputBound: 'capacity-bounded',
    operators: [
      'GPUGridGenerator',
      'GPUShapeGenerator',
      'GPUDotDensity',
      'GPURandomPointsInPolygon',
      'GPURandomPointsOnLine',
      'GPUPolygonTriangulation',
      'GPUOutlineGeometry'
    ],
    relatedCapabilities: ['density-and-interpolation', 'geometry-overlay'],
    limitation: 'Generated variable-size geometry requires caller-selected capacities.'
  },
  {
    id: 'geometry-validation',
    title: 'Geometry predicates and validation',
    status: AVAILABLE,
    domain: 'topology',
    stage: 'analyze',
    summary: 'Checks structural predicates, validity, clearance and polygon-coverage topology.',
    inputs: ['points', 'lines', 'polygons'],
    outputs: ['attributes', 'lines', 'summary'],
    coordinateModels: ['planar', 'topological'],
    outputBound: 'mixed',
    operators: ['GPUGeometryPredicates', 'GPUGeometryValidity', 'GPUCoverageValidity'],
    relatedCapabilities: ['segment-and-coverage-topology', 'geometry-repair'],
    limitation: 'Reports invalidity but does not construct a repaired polygon.'
  },
  {
    id: 'line-processing',
    title: 'Line processing and linear referencing',
    status: AVAILABLE,
    domain: 'geometry',
    stage: 'transform',
    summary:
      'Densifies, chunks, smooths, simplifies, clips, merges, splits and locates along paths.',
    inputs: ['points', 'lines', 'polygons', 'parameters'],
    outputs: ['points', 'lines', 'attributes'],
    coordinateModels: ['planar', 'spherical'],
    outputBound: 'capacity-bounded',
    operators: [
      'GPUGreatCircleArcs',
      'GPULineChunk',
      'GPULineSegmentize',
      'GPULineSmooth',
      'GPULineSimplification',
      'GPULineSplit',
      'GPULineMerge',
      'GPULineClipByPolygon',
      'GPUSharedPaths',
      'GPURectangleClip',
      'GPULinearReferencing',
      'GPULineLocate'
    ],
    relatedCapabilities: ['segment-and-coverage-topology', 'movement-analysis']
  },
  {
    id: 'segment-and-coverage-topology',
    title: 'Segment and coverage topology',
    status: AVAILABLE,
    domain: 'topology',
    stage: 'relate',
    summary:
      'Finds segment intersections and preserves shared boundaries through coverage operations.',
    inputs: ['lines', 'polygons', 'attributes'],
    outputs: ['points', 'lines', 'polygons', 'pairs'],
    coordinateModels: ['planar', 'topological'],
    outputBound: 'capacity-bounded',
    operators: [
      'GPUSegmentIntersection',
      'GPULineNoding',
      'GPUSegmentRingAssembly',
      'GPUCoverageDissolve',
      'GPUCoverageSimplification'
    ],
    relatedCapabilities: ['line-processing', 'geometry-overlay', 'geometry-repair'],
    limitation:
      'General polygonization uses bounded intermediate and output capacities with explicit incomplete status.'
  },
  {
    id: 'spatial-ordering',
    title: 'Spatial ordering',
    status: AVAILABLE,
    domain: 'proximity',
    stage: 'index',
    summary: 'Maps planar coordinates to Hilbert keys and a locality-preserving permutation.',
    inputs: ['points', 'parameters'],
    outputs: ['attributes'],
    coordinateModels: ['planar'],
    outputBound: 'source-sized',
    operators: ['GPUHilbertKeys'],
    relatedCapabilities: ['spatial-joins', 'spatial-neighborhoods']
  },
  {
    id: 'spatial-joins',
    title: 'Spatial joins and pair materialization',
    status: AVAILABLE,
    domain: 'proximity',
    stage: 'relate',
    summary: 'Builds candidate or exact feature pairs and gathers pair-aligned source values.',
    inputs: ['points', 'lines', 'polygons', 'attributes', 'parameters'],
    outputs: ['pairs', 'attributes', 'spatial-weights'],
    coordinateModels: ['planar'],
    outputBound: 'capacity-bounded',
    operators: [
      'GPUSpatialJoinCandidates',
      'GPUSpatialJoinPrepared',
      'GPUSpatialPredicateJoin',
      'GPUPointInPolygonJoin',
      'GPUNearestFeatureJoin',
      'GPUBufferSelection',
      'GPUPairGather',
      'GPUOffsetExpansion',
      'GPUBoundsFilter'
    ],
    relatedCapabilities: ['spatial-neighborhoods', 'zonal-and-region-statistics']
  },
  {
    id: 'spatial-neighborhoods',
    title: 'Spatial neighborhoods and weights',
    status: AVAILABLE,
    domain: 'spatial-statistics',
    stage: 'relate',
    summary:
      'Constructs, transforms, combines, summarizes and applies CSR spatial-weight matrices.',
    inputs: ['points', 'polygons', 'surface', 'attributes', 'spatial-weights', 'parameters'],
    outputs: ['spatial-weights', 'attributes', 'summary'],
    coordinateModels: ['planar', 'grid', 'topological'],
    outputBound: 'capacity-bounded',
    operators: [
      'GPUNeighborSearch',
      'GPUContiguityWeights',
      'GPULatticeWeights',
      'GPUNearestFeatureWeights',
      'GPUSpatialWeightsTransform',
      'GPUSpatialWeightsAlgebra',
      'GPUSpatialWeightsSummary',
      'GPUSpatialWeightsTranspose',
      'GPUSpatialLag',
      'GPUMapColoring'
    ],
    relatedCapabilities: [
      'spatial-autocorrelation-and-inference',
      'spatial-regression',
      'group-and-regional-analysis'
    ]
  },
  {
    id: 'dggs-indexing-and-topology',
    title: 'DGGS indexing and topology',
    status: AVAILABLE,
    domain: 'discrete-global-grids',
    stage: 'index',
    summary:
      'Converts between points, H3 or Quadbin cells, boundaries, parents, neighbors and paths.',
    inputs: ['points', 'polygons', 'cells', 'parameters'],
    outputs: ['cells', 'points', 'lines', 'polygons', 'attributes'],
    coordinateModels: ['spherical', 'grid', 'topological'],
    outputBound: 'capacity-bounded',
    operators: [
      'GPUPointToCell',
      'GPUCellGeometry',
      'GPUCellTopology',
      'GPUCellCompaction',
      'GPUCellGridPath',
      'GPUCellMeasures',
      'GPUCellCover'
    ],
    relatedCapabilities: ['dggs-aggregation', 'dggs-outlines-and-change']
  },
  {
    id: 'dggs-aggregation',
    title: 'DGGS aggregation and pyramids',
    status: AVAILABLE,
    domain: 'discrete-global-grids',
    stage: 'aggregate',
    summary: 'Aggregates points into sparse cell tables and rolls those tables across resolutions.',
    inputs: ['points', 'cells', 'attributes'],
    outputs: ['cells', 'summary'],
    coordinateModels: ['grid'],
    outputBound: 'capacity-bounded',
    operators: ['GPUCellAggregation', 'GPUCellRollup', 'GPUCellPyramid', 'GPUCellLevelSelection'],
    relatedCapabilities: ['dggs-indexing-and-topology', 'dggs-outlines-and-change']
  },
  {
    id: 'dggs-outlines-and-change',
    title: 'DGGS outlines and change',
    status: AVAILABLE,
    domain: 'discrete-global-grids',
    stage: 'transform',
    summary: 'Builds cell-set outlines and compares two sparse cell tables.',
    inputs: ['cells'],
    outputs: ['lines', 'polygons', 'summary'],
    coordinateModels: ['grid', 'topological'],
    outputBound: 'capacity-bounded',
    operators: ['GPUCellSetOutline', 'GPUCellTableCompare'],
    relatedCapabilities: ['dggs-aggregation', 'segment-and-coverage-topology']
  },
  {
    id: 'density-and-interpolation',
    title: 'Density and interpolation',
    status: AVAILABLE,
    domain: 'density-and-interpolation',
    stage: 'aggregate',
    summary:
      'Turns points or lines into density fields and interpolates sampled values onto surfaces.',
    inputs: ['points', 'lines', 'polygons', 'attributes', 'parameters'],
    outputs: ['surface', 'attributes', 'summary'],
    coordinateModels: ['planar', 'spherical', 'grid'],
    outputBound: 'fixed-sized',
    operators: [
      'GPUPointDensity',
      'GPULineDensity',
      'GPULineLengthPerPolygon',
      'GPUInverseDistanceWeighting',
      'GPUKriging',
      'GPUFocalStatistics'
    ],
    relatedCapabilities: ['zonal-and-region-statistics', 'spatial-neighborhoods']
  },
  {
    id: 'zonal-and-region-statistics',
    title: 'Zonal and interactive region statistics',
    status: AVAILABLE,
    domain: 'regional-analysis',
    stage: 'aggregate',
    summary: 'Masks, picks and summarizes values inside vector or interactive regions.',
    inputs: ['points', 'polygons', 'surface', 'attributes', 'regions', 'parameters'],
    outputs: ['regions', 'summary', 'attributes'],
    coordinateModels: ['planar', 'grid'],
    outputBound: 'mixed',
    operators: [
      'GPUZonalStatistics',
      'GPURegionMask',
      'GPUPickRegionMask',
      'GPURegionStatistics',
      'GPURegionStatisticsReadback'
    ],
    relatedCapabilities: ['spatial-joins', 'density-and-interpolation'],
    limitation:
      'GPURegionStatisticsReadback is an explicit application bridge; normal contributors remain GPU-resident.'
  },
  {
    id: 'group-and-regional-analysis',
    title: 'Clustering, groups and regionalization',
    status: AVAILABLE,
    domain: 'regional-analysis',
    stage: 'analyze',
    summary: 'Clusters rows, constructs connected regions and measures or outlines labeled groups.',
    inputs: ['points', 'lines', 'polygons', 'attributes', 'spatial-weights', 'parameters'],
    outputs: ['regions', 'summary', 'points', 'polygons'],
    coordinateModels: ['planar', 'topological'],
    outputBound: 'capacity-bounded',
    operators: [
      'GPUSpatialClustering',
      'GPUKMeans',
      'GPUSpatialWeightsMinimumSpanningTree',
      'GPUSkaterRegions',
      'GPURegionPartitionEvaluation',
      'GPUGroupGeometry',
      'GPUGroupConvexHull',
      'GPULocalOutlierFactor',
      'GPUSimilarLocations'
    ],
    relatedCapabilities: ['spatial-neighborhoods', 'region-optimization']
  },
  {
    id: 'spatial-autocorrelation-and-inference',
    title: 'Spatial autocorrelation and inference',
    status: AVAILABLE,
    domain: 'spatial-statistics',
    stage: 'infer',
    summary:
      'Computes global and local spatial statistics, permutation tests, scan statistics and rate smoothing.',
    inputs: ['attributes', 'spatial-weights', 'points', 'parameters'],
    outputs: ['attributes', 'regions', 'summary'],
    coordinateModels: ['planar', 'topological'],
    outputBound: 'mixed',
    operators: [
      'GPUGlobalSpatialStatistics',
      'GPUHotSpotAnalysis',
      'GPULocalMoran',
      'GPULocalPermutationTest',
      'GPUGlobalPermutationTest',
      'GPUNeighborhoodSummary',
      'GPUEmpiricalBayesRates',
      'GPUSpatialEmpiricalBayesRates',
      'GPUSpatialScanStatistic',
      'GPUKnoxTest',
      'GPUMantelTest'
    ],
    relatedCapabilities: ['spatial-neighborhoods', 'distribution-dynamics', 'local-statistics']
  },
  {
    id: 'spatial-regression',
    title: 'Spatial regression',
    status: AVAILABLE,
    domain: 'spatial-models',
    stage: 'infer',
    summary:
      'Fits global, lag/error and geographically weighted regression models with diagnostics.',
    inputs: ['attributes', 'points', 'spatial-weights', 'parameters'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['planar', 'topological'],
    outputBound: 'fixed-sized',
    operators: [
      'GPUOrdinaryLeastSquares',
      'GPUSpatialRegressionDiagnostics',
      'GPUSpatialTwoStageLeastSquares',
      'GPUSpatialErrorGM',
      'GPUGeographicallyWeightedRegression',
      'GPUGeographicallyWeightedRegressionNonstationarityTest'
    ],
    relatedCapabilities: ['spatial-neighborhoods', 'advanced-spatial-models']
  },
  {
    id: 'distribution-dynamics',
    title: 'Classification and distribution dynamics',
    status: AVAILABLE,
    domain: 'spatial-statistics',
    stage: 'analyze',
    summary: 'Assigns classes and describes temporal or spatial transitions between distributions.',
    inputs: ['attributes', 'spatial-weights', 'parameters'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['none', 'topological'],
    outputBound: 'fixed-sized',
    operators: [
      'GPUClassAssignment',
      'GPUClassificationFit',
      'GPUTransitionMatrix',
      'GPUSpatialMarkov',
      'GPULISAMarkov',
      'GPUEmergingHotSpots'
    ],
    relatedCapabilities: ['spatial-autocorrelation-and-inference', 'workflow-recipes']
  },
  {
    id: 'accessibility-and-segregation',
    title: 'Accessibility and segregation',
    status: AVAILABLE,
    domain: 'spatial-models',
    stage: 'analyze',
    summary: 'Models catchment access, probabilistic trade areas and population segregation.',
    inputs: ['attributes', 'pairs', 'spatial-weights', 'parameters'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['planar', 'topological'],
    outputBound: 'source-sized',
    operators: ['GPUCatchmentAccessibility', 'GPUHuffTradeAreas', 'GPUSegregation'],
    relatedCapabilities: ['spatial-neighborhoods', 'workflow-recipes']
  },
  {
    id: 'change-of-support',
    title: 'Change of support',
    status: AVAILABLE,
    domain: 'spatial-models',
    stage: 'transform',
    summary:
      'Transfers values between polygon systems and smooths extensive totals over a surface.',
    inputs: ['polygons', 'surface', 'attributes', 'spatial-weights'],
    outputs: ['attributes', 'surface', 'spatial-weights'],
    coordinateModels: ['planar', 'grid'],
    outputBound: 'capacity-bounded',
    operators: ['GPUArealInterpolation', 'GPUPycnophylactic'],
    relatedCapabilities: ['spatial-neighborhoods', 'workflow-recipes']
  },
  {
    id: 'movement-analysis',
    title: 'Trajectory and movement analysis',
    status: AVAILABLE,
    domain: 'movement',
    stage: 'analyze',
    summary:
      'Measures, resamples and compares tracks and finds time-aligned encounters or zone events.',
    inputs: ['trajectories', 'polygons', 'parameters'],
    outputs: ['trajectories', 'points', 'pairs', 'attributes', 'summary'],
    coordinateModels: ['planar', 'spherical'],
    outputBound: 'capacity-bounded',
    operators: [
      'GPUTrajectoryMetrics',
      'GPUTrajectoryPlayhead',
      'GPUTrajectoryResample',
      'GPUZoneEvents',
      'GPUTrajectoryEncounters',
      'GPUTrackSimilarity',
      'addClockEncounters'
    ],
    relatedCapabilities: ['line-processing', 'spatial-joins', 'workflow-recipes']
  },
  {
    id: 'workflow-recipes',
    title: 'Prebuilt analysis workflows',
    status: AVAILABLE,
    domain: 'workflow',
    stage: 'orchestrate',
    summary:
      'Connects contributors from spatial, dataframe, raster and network entry points in one command graph.',
    inputs: [
      'points',
      'lines',
      'polygons',
      'trajectories',
      'cells',
      'network',
      'attributes',
      'parameters'
    ],
    outputs: ['attributes', 'regions', 'surface', 'summary', 'render-geometry'],
    coordinateModels: ['mixed'],
    outputBound: 'mixed',
    operators: [
      'addChangeOfSupportRecipe',
      'addClusterAndOutlineRecipe',
      'addDriveTimeCatchmentRecipe',
      'addFleetDwellRecipe',
      'addFleetDwellZoneEventsRecipe',
      'addHotSpotAnalysisRecipe',
      'addPeriodComparisonRecipe',
      'addPointsInPolygonsChoroplethRecipe',
      'addRateClusterMapRecipe',
      'addSpaceTimeHotSpotsRecipe',
      'addSpatialRegressionRecipe',
      'addStraightLineCatchmentsRecipe'
    ],
    relatedCapabilities: [
      'spatial-neighborhoods',
      'spatial-autocorrelation-and-inference',
      'movement-analysis'
    ]
  },
  {
    id: 'geometry-overlay',
    title: 'General polygon overlay and buffering',
    status: AVAILABLE,
    domain: 'topology',
    stage: 'transform',
    summary:
      'Union, intersection, difference, general dissolve and polygon buffer with new polygon topology.',
    inputs: ['polygons', 'parameters'],
    outputs: ['polygons'],
    coordinateModels: ['planar', 'topological'],
    outputBound: 'capacity-bounded',
    operators: ['GPUPolygonOverlay', 'GPUBufferSurface'],
    relatedCapabilities: ['segment-and-coverage-topology', 'geometry-repair'],
    limitation:
      'Planar float32 side classification follows the explicit vertex-tolerance policy; geodesic overlay is out of scope.'
  },
  {
    id: 'tessellation',
    title: 'Voronoi and Delaunay tessellation',
    status: AVAILABLE,
    domain: 'geometry',
    stage: 'construct',
    summary: 'Constructs bounded planar Delaunay triangles and clipped vector Voronoi edges.',
    inputs: ['points', 'parameters'],
    outputs: ['lines', 'polygons'],
    coordinateModels: ['planar'],
    outputBound: 'capacity-bounded',
    operators: ['GPUDelaunayTessellation', 'GPUVoronoiDiagram'],
    relatedCapabilities: ['geometry-construction', 'spatial-neighborhoods'],
    limitation:
      'Finite planar float32 sites are supported; constrained Delaunay and concave construction remain separate opportunities.'
  },
  {
    id: 'geometry-repair',
    title: 'Geometry repair and general polygonization',
    status: AVAILABLE,
    domain: 'topology',
    stage: 'transform',
    summary:
      'Nodes arbitrary linework, repairs invalid polygons and reports unused polygonize edges.',
    inputs: ['lines', 'polygons'],
    outputs: ['lines', 'polygons', 'summary'],
    coordinateModels: ['planar', 'topological'],
    outputBound: 'capacity-bounded',
    operators: ['GPUPolygonize', 'GPUMakeValid'],
    relatedCapabilities: ['geometry-validation', 'segment-and-coverage-topology'],
    limitation:
      'Repair currently rebuilds planar polygon boundaries; dimensional collapse and non-polygon repair modes are not emitted.'
  },
  {
    id: 'local-statistics',
    title: 'Additional local spatial statistics',
    status: AVAILABLE,
    domain: 'spatial-statistics',
    stage: 'infer',
    summary: 'Adds local Geary, spatial Pearson, Gamma and directional distribution tests.',
    inputs: ['attributes', 'spatial-weights', 'parameters'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['planar', 'topological'],
    outputBound: 'source-sized',
    operators: ['GPULocalGeary', 'GPUSpatialPearson', 'GPUGammaStatistic', 'GPURoseStatistic'],
    relatedCapabilities: ['spatial-autocorrelation-and-inference']
  },
  {
    id: 'region-optimization',
    title: 'Region optimization',
    status: AVAILABLE,
    domain: 'regional-analysis',
    stage: 'infer',
    summary:
      'Runs bounded AZP, Max-P, Ward and location-allocation searches with explicit completion evidence.',
    inputs: ['attributes', 'spatial-weights', 'network', 'parameters'],
    outputs: ['regions', 'summary'],
    coordinateModels: ['topological'],
    outputBound: 'source-sized',
    operators: ['GPUAZPRegions', 'GPUMaxPRegions', 'GPUWardRegions', 'GPULocationAllocation'],
    relatedCapabilities: ['group-and-regional-analysis'],
    limitation:
      'Max-P and facility selection are deterministic greedy heuristics rather than randomized multi-start or integer-programming solvers.'
  },
  {
    id: 'advanced-spatial-models',
    title: 'Likelihood, panel and multiscale spatial models',
    status: OPPORTUNITY,
    domain: 'spatial-models',
    stage: 'infer',
    summary:
      'Extends regression to maximum-likelihood lag/error, panel, regimes, SUR, probit and MGWR.',
    inputs: ['attributes', 'points', 'spatial-weights', 'parameters'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['planar', 'topological'],
    outputBound: 'fixed-sized',
    operators: [],
    relatedCapabilities: ['spatial-regression'],
    limitation:
      'A structural solver-status port is shared by three regression families, but likelihood, panel, regimes, SUR, probit and MGWR remain evidence-gated.'
  }
];

const CAPABILITY_EVIDENCE: Readonly<Record<string, GPUSpatialAnalysisCapabilityEvidence>> = {
  'dynamic-parameters': {
    representativeSourceRows: 16,
    transientMemory: 'constant',
    dispatch: 'single-pass',
    correctnessOracle:
      'modules/experimental/test/gpu-spatial-analysis/contracts/contracts.node.spec.ts'
  },
  'geometry-measurement': {
    representativeSourceRows: 1_000_000,
    transientMemory: 'linear-source',
    dispatch: 'workgroup-reduction',
    correctnessOracle:
      'modules/experimental/test/gpu-spatial-analysis/geometry-measures/geometry-measures-oracle.ts'
  },
  'geodesic-operations': {
    representativeSourceRows: 100_000,
    transientMemory: 'constant',
    dispatch: 'source-parallel',
    correctnessOracle:
      'modules/experimental/test/gpu-spatial-analysis/geometry-measures/geodesic-oracle.ts'
  },
  'geometry-editing': {
    representativeSourceRows: 100_000,
    representativeOutputRows: 200_000,
    transientMemory: 'linear-source-and-output',
    dispatch: 'multi-pass',
    correctnessOracle:
      'modules/experimental/test/gpu-spatial-analysis/geometry-edit/geometry-edit-fixtures.ts'
  },
  'geometry-construction': {
    representativeSourceRows: 100_000,
    representativeOutputRows: 1_000_000,
    transientMemory: 'linear-output',
    dispatch: 'multi-pass',
    correctnessOracle:
      'modules/experimental/test/gpu-spatial-analysis/grid-generators/grid-generator-oracle.ts'
  },
  'geometry-validation': {
    representativeSourceRows: 100_000,
    transientMemory: 'linear-source',
    dispatch: 'multi-pass',
    correctnessOracle:
      'modules/experimental/test/gpu-spatial-analysis/geometry-validity/shapely-validity-fixture.ts'
  },
  'line-processing': {
    representativeSourceRows: 100_000,
    representativeOutputRows: 1_000_000,
    transientMemory: 'linear-source-and-output',
    dispatch: 'multi-pass',
    correctnessOracle:
      'modules/experimental/test/gpu-spatial-analysis/line-segmentize/line-segmentize-oracle.ts'
  },
  'segment-and-coverage-topology': {
    representativeSourceRows: 50_000,
    representativeOutputRows: 500_000,
    transientMemory: 'linear-source-and-output',
    dispatch: 'bounded-iterative',
    correctnessOracle:
      'modules/experimental/test/gpu-spatial-analysis/geometry-topology/gpu-geometry-topology.spec.ts'
  },
  'spatial-ordering': {
    representativeSourceRows: 1_000_000,
    transientMemory: 'linear-source',
    dispatch: 'multi-pass',
    correctnessOracle:
      'modules/experimental/test/gpu-spatial-analysis/hilbert-keys/hilbert-oracle.ts'
  },
  'spatial-joins': {
    representativeSourceRows: 100_000,
    representativeOutputRows: 1_000_000,
    transientMemory: 'linear-source-and-output',
    dispatch: 'multi-pass',
    correctnessOracle:
      'modules/experimental/test/gpu-spatial-analysis/spatial-join/spatial-join-oracle.ts'
  },
  'spatial-neighborhoods': {
    representativeSourceRows: 100_000,
    representativeOutputRows: 1_600_000,
    transientMemory: 'linear-source-and-output',
    dispatch: 'multi-pass',
    correctnessOracle:
      'modules/experimental/test/gpu-spatial-analysis/neighbor-search/neighbor-search-oracle.ts'
  },
  'dggs-indexing-and-topology': {
    representativeSourceRows: 1_000_000,
    representativeOutputRows: 1_000_000,
    transientMemory: 'linear-source',
    dispatch: 'source-parallel',
    correctnessOracle:
      'modules/experimental/test/gpu-spatial-analysis/cell-indexing/cell-indexing-oracle.ts'
  },
  'dggs-aggregation': {
    representativeSourceRows: 1_000_000,
    representativeOutputRows: 250_000,
    transientMemory: 'linear-source-and-output',
    dispatch: 'multi-pass',
    correctnessOracle:
      'modules/experimental/test/gpu-spatial-analysis/cell-aggregation/cell-aggregation-oracle.ts'
  },
  'dggs-outlines-and-change': {
    representativeSourceRows: 250_000,
    representativeOutputRows: 1_000_000,
    transientMemory: 'linear-source-and-output',
    dispatch: 'multi-pass',
    correctnessOracle:
      'modules/experimental/test/gpu-spatial-analysis/cell-set-outline/cell-set-outline-oracle.ts'
  },
  'density-and-interpolation': {
    representativeSourceRows: 1_000_000,
    representativeOutputRows: 262_144,
    transientMemory: 'linear-source-and-output',
    dispatch: 'multi-pass',
    correctnessOracle:
      'modules/experimental/test/gpu-spatial-analysis/point-density/point-density-oracle.ts'
  },
  'zonal-and-region-statistics': {
    representativeSourceRows: 1_000_000,
    transientMemory: 'linear-source',
    dispatch: 'workgroup-reduction',
    correctnessOracle:
      'modules/experimental/test/gpu-spatial-analysis/region-statistics/gpu-region-mask.spec.ts'
  },
  'group-and-regional-analysis': {
    representativeSourceRows: 100_000,
    representativeOutputRows: 500_000,
    transientMemory: 'linear-source-and-output',
    dispatch: 'bounded-iterative',
    correctnessOracle:
      'modules/experimental/test/gpu-spatial-analysis/group-geometry/group-geometry-oracle.ts'
  },
  'spatial-autocorrelation-and-inference': {
    representativeSourceRows: 100_000,
    representativeOutputRows: 1_600_000,
    transientMemory: 'linear-source-and-output',
    dispatch: 'bounded-iterative',
    correctnessOracle:
      'modules/experimental/test/gpu-spatial-analysis/global-spatial-statistics/global-spatial-statistics-oracle.ts'
  },
  'spatial-regression': {
    representativeSourceRows: 100_000,
    transientMemory: 'solver-matrix',
    dispatch: 'bounded-iterative',
    correctnessOracle:
      'modules/experimental/test/gpu-spatial-analysis/spatial-regression/ordinary-least-squares-oracle.ts'
  },
  'distribution-dynamics': {
    representativeSourceRows: 1_000_000,
    transientMemory: 'linear-source',
    dispatch: 'multi-pass',
    correctnessOracle:
      'modules/experimental/test/gpu-spatial-analysis/distribution-dynamics/distribution-oracle.ts'
  },
  'accessibility-and-segregation': {
    representativeSourceRows: 100_000,
    representativeOutputRows: 1_000_000,
    transientMemory: 'linear-source-and-output',
    dispatch: 'workgroup-reduction',
    correctnessOracle:
      'modules/experimental/test/gpu-spatial-analysis/catchment-accessibility/catchment-oracle.ts'
  },
  'change-of-support': {
    representativeSourceRows: 100_000,
    representativeOutputRows: 1_000_000,
    transientMemory: 'linear-source-and-output',
    dispatch: 'bounded-iterative',
    correctnessOracle:
      'modules/experimental/test/gpu-spatial-analysis/areal-interpolation/areal-interpolation-oracle.ts'
  },
  'movement-analysis': {
    representativeSourceRows: 1_000_000,
    representativeOutputRows: 250_000,
    transientMemory: 'linear-source-and-output',
    dispatch: 'multi-pass',
    correctnessOracle:
      'modules/experimental/test/gpu-spatial-analysis/trajectory-analysis/trajectory-metrics-oracle.ts'
  },
  'workflow-recipes': {
    representativeSourceRows: 100_000,
    representativeOutputRows: 1_000_000,
    transientMemory: 'linear-source-and-output',
    dispatch: 'orchestrated',
    correctnessOracle: 'modules/experimental/test/gpu-spatial-analysis/recipes/recipe-harness.ts'
  },
  'geometry-overlay': {
    representativeSourceRows: 10_000,
    representativeOutputRows: 250_000,
    transientMemory: 'linear-source-and-output',
    dispatch: 'bounded-iterative',
    correctnessOracle:
      'modules/experimental/test/gpu-spatial-analysis/polygon-overlay/polygon-overlay.spec.ts'
  },
  tessellation: {
    representativeSourceRows: 10_000,
    representativeOutputRows: 20_000,
    transientMemory: 'quadratic-source',
    dispatch: 'multi-pass',
    correctnessOracle:
      'modules/experimental/test/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation.spec.ts'
  },
  'geometry-repair': {
    representativeSourceRows: 10_000,
    representativeOutputRows: 250_000,
    transientMemory: 'linear-source-and-output',
    dispatch: 'bounded-iterative',
    correctnessOracle:
      'modules/experimental/test/gpu-spatial-analysis/geometry-topology/gpu-geometry-topology.spec.ts'
  },
  'local-statistics': {
    representativeSourceRows: 100_000,
    representativeOutputRows: 1_600_000,
    transientMemory: 'linear-source-and-output',
    dispatch: 'workgroup-reduction',
    correctnessOracle:
      'modules/experimental/test/gpu-spatial-analysis/spatial-autocorrelation/additional-spatial-statistics.spec.ts'
  },
  'region-optimization': {
    representativeSourceRows: 10_000,
    transientMemory: 'linear-source',
    dispatch: 'bounded-iterative',
    correctnessOracle:
      'modules/experimental/test/gpu-spatial-analysis/spatial-regionalization/gpu-azp-regions.spec.ts'
  }
};

/** Capability catalogue with executable evidence attached to every available family. */
export const GPU_SPATIAL_ANALYSIS_CAPABILITIES: readonly GPUSpatialAnalysisCapability[] =
  CAPABILITY_DEFINITIONS.map(capability =>
    capability.status === AVAILABLE
      ? {...capability, evidence: CAPABILITY_EVIDENCE[capability.id]}
      : capability
  );

const CAPABILITY_BY_ID = new Map(
  GPU_SPATIAL_ANALYSIS_CAPABILITIES.map(capability => [capability.id, capability])
);

/** Returns one capability descriptor, or `undefined` for an unknown ID. */
export function getGPUSpatialAnalysisCapability(
  id: string
): GPUSpatialAnalysisCapability | undefined {
  return CAPABILITY_BY_ID.get(id);
}

/** Returns capabilities matching all supplied filters, preserving catalogue order. */
export function queryGPUSpatialAnalysisCapabilities(
  query: GPUSpatialAnalysisCapabilityQuery = {}
): readonly GPUSpatialAnalysisCapability[] {
  const search = query.search?.trim().toLowerCase();
  return GPU_SPATIAL_ANALYSIS_CAPABILITIES.filter(capability => {
    if (query.status && capability.status !== query.status) return false;
    if (query.domain && capability.domain !== query.domain) return false;
    if (query.stage && capability.stage !== query.stage) return false;
    if (query.input && !capability.inputs.includes(query.input)) return false;
    if (query.output && !capability.outputs.includes(query.output)) return false;
    if (query.coordinateModel && !capability.coordinateModels.includes(query.coordinateModel)) {
      return false;
    }
    if (!search) return true;
    const text = [
      capability.id,
      capability.title,
      capability.summary,
      capability.limitation ?? '',
      ...capability.operators
    ]
      .join(' ')
      .toLowerCase();
    return text.includes(search);
  });
}

/** Resolves the curated connective tissue around one capability. */
export function getGPUSpatialAnalysisCapabilityConnections(
  id: string
): GPUSpatialAnalysisCapabilityConnections | undefined {
  const capability = CAPABILITY_BY_ID.get(id);
  if (!capability) return undefined;
  return {
    capability,
    related: (capability.relatedCapabilities ?? [])
      .map(relatedId => CAPABILITY_BY_ID.get(relatedId))
      .filter((related): related is GPUSpatialAnalysisCapability => Boolean(related))
  };
}

/** Returns the capability family that owns a public contributor or recipe export. */
export function getGPUSpatialAnalysisCapabilityForOperator(
  operator: string
): GPUSpatialAnalysisCapability | undefined {
  return GPU_SPATIAL_ANALYSIS_CAPABILITIES.find(capability =>
    capability.operators.includes(operator)
  );
}
