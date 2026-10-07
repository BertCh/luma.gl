// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  addClusterAndOutlineRecipe,
  getGPUGeographicDistributionParameterValues,
  getGPUSpatialClusteringParameterValues,
  GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH,
  GPU_GROUP_CONVEX_HULL_GROUP_OVERFLOW,
  GPU_GROUP_CONVEX_HULL_TOTAL_OVERFLOW,
  GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH,
  GPUGeographicDistribution,
  GPUGroupConvexHull,
  GPUGroupGeometry,
  GPUKMeans,
  GPUSpatialClustering,
  type GPUParameterBuffer
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getClassTableLayerProps} from '../../cartography/class-table';
import {CHICAGO, findPlace, nearestPlace, nearestPlaceLabel} from '../../cartography/gazetteer';
import {
  formatArea,
  formatCount,
  formatDistance,
  formatPercent,
  formatRate
} from '../../cartography/live-text';
import type {StableHuePrevious} from '../../cartography/stable-hues';
import type {LngLat, MapAnnotation, MapHighlight} from '../../cartography/types';
import {DENSE_POINT_RADIUS_STOPS} from '../../cartography/zoom';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisPolygonLayer,
  SpatialAnalysisSegmentLayer
} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {createSeededRandom, LocalMetricProjection} from '../../engine/projection';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {formatCompiledGraphTiming, measureCompiledGraph} from '../../engine/vector-timing';
import type {ChartData, SceneContext, SceneInstance, TooltipContent, TooltipRow} from '../scene';
import {loadCityGeometry, type CityGeometry} from './b1-city-geometry';
import {BoundsOutlineLayer, EllipseOutlineLayer, GroupMarkerLayer} from './b1-cluster-layers';
import {readNatureColumns, type NatureColumns} from './b1-nature-data';
import {getParameterHalo, getSubjectColor} from './b1-points-look';
import {
  analyzeLabels,
  attachHulls,
  buildHullMesh,
  colourPartition,
  findHullAt,
  getShareOutsideCity,
  pickEpsilonSamples,
  type ClusterAnalysis,
  type EpsilonSample,
  type HullShape
} from './nature-clusters-analysis';
import {
  getInk,
  getNoiseColor,
  getPartitionPalette,
  getSignal,
  ITERATION_LADDER,
  makeSizeTable,
  PARTITION_HUES,
  SIZE_CLASS_LABELS,
  SWEEP_EPSILONS
} from './nature-clusters-classes';

/** Option state of the nature-clusters scene. */
export type NatureClusterOptions = {
  category: string;
  view: 'hotspots' | 'records';
  method: 'dbscan' | 'kmeans';
  pipeline: 'assembled' | 'recipe';
  epsilon: number;
  minimumPoints: number;
  denseBoxShortcut: boolean;
  sumOrder: 'sorted' | 'atomic';
  k: number;
  initialization: 'first-valid' | 'kmeans++';
  seed: number;
  /** Index into {@link ITERATION_LADDER}: the compiled iteration cap. */
  iterationStep: number;
  tolerance: number;
  prefilterLevels: string;
  weight: 'none' | 'researchGrade' | 'introduced';
  standardDeviations: '1' | '2' | '3';
  ellipseConvention: 'arcgis' | 'standard';
  showHulls: boolean;
  showClusterEllipses: boolean;
  showBounds: boolean;
  showMeans: boolean;
  showWeighted: boolean;
  showMedoids: boolean;
  showCore: boolean;
  showNoise: boolean;
  showDistribution: boolean;
  showEpsilonDiscs: boolean;
  labelHotSpots: boolean;
  zoomToWinner: boolean;
};

const GRID_SIZE: readonly [number, number] = [256, 256];
const CLUSTER_CAPACITY = 2048;
const BOUNDS_PADDING_METERS = 10;
const MEDOID_MAXIMUM_GROUP_SIZE = 1024;
const HULL_MAXIMUM_VERTICES = 256;
const HULL_CAPACITY = 1 << 14;
const REBUILD_DEBOUNCE_MILLISECONDS = 250;
const RING_VERTICES = 64;
const ELLIPSE_SEGMENTS = 48;
/** Radius around Montrose Point (metres) in which the epsilon discs pick their records. */
const DISC_WINDOW_METERS = 3500;
const MONTE_CARLO_SAMPLES = 2000;
const MAXIMUM_BARS = 60;
const MAXIMUM_WINNER_ZOOM = 14.5;
const PARTITION_OPACITY = 0.2;

type SetBuffer = Buffer;

/** Typed views over one detail readback, by source name. */
type DetailView = {
  u32: (name: string) => Uint32Array;
  f32: (name: string) => Float32Array;
};

/** The scratch graph that runs the epsilon sweep on its own buffers (compiled once). */
type SweepGraph = {
  graph: CompiledGPUCommandGraph<void>;
  parameters: GPUParameterBuffer<'float32'>;
  clusterCount: SetBuffer;
  reader: SummaryReader;
};

type ClusterSet = {
  serial: number;
  key: string;
  config: NatureClusterOptions;
  resources: SpatialAnalysisResources;
  pointCount: number;
  positions: Float32Array;
  bounds: [number, number, number, number];
  clusteringParameters: GPUParameterBuffer<'float32'>;
  geometryParameters: GPUParameterBuffer<'float32'>;
  distributionParameters: GPUParameterBuffer<'float32'>;
  clusterGraph: CompiledGPUCommandGraph<void>;
  shapeGraph: CompiledGPUCommandGraph<void> | null;
  distributionGraph: CompiledGPUCommandGraph<void>;
  groupCount: number;
  weights: SetBuffer;
  positionsBuffer: SetBuffer;
  labels: SetBuffer;
  coreFlags: SetBuffer;
  clusterCount: SetBuffer;
  clusterStoredCount: SetBuffer;
  clusterOverflow: SetBuffer;
  convergence: SetBuffer;
  sizes: SetBuffer;
  centroids: SetBuffer;
  geometryCounts: SetBuffer;
  geometryBounds: SetBuffer;
  meanCenters: SetBuffer;
  weightedCenters: SetBuffer;
  medoids: SetBuffer;
  ellipses: SetBuffer;
  geometryOverflow: SetBuffer;
  hullOffsets: SetBuffer;
  hullCounts: SetBuffer;
  hullPositions: SetBuffer;
  hullOverflow: SetBuffer;
  areas: SetBuffer;
  perimeters: SetBuffer;
  distributionCounts: SetBuffer;
  distributionMean: SetBuffer;
  distributionMedian: SetBuffer;
  distributionStandardDistance: SetBuffer;
  distributionEllipse: SetBuffer;
  ellipseSegments: SetBuffer;
  circleSegments: SetBuffer;
  /** Drawing buffers the CPU fills from each readback (class colours, id lists, hull meshes). */
  pointSizes: SetBuffer;
  pointSlots: SetBuffer;
  groupSizes: SetBuffer;
  groupSlots: SetBuffer;
  noiseIds: SetBuffer;
  memberIds: SetBuffer;
  coreIds: SetBuffer;
  borderIds: SetBuffer;
  hullTriangles: SetBuffer;
  hullTriangleFeatures: SetBuffer;
  hullOutline: SetBuffer;
  hullOutlineFeatures: SetBuffer;
  drawCounts: {
    noise: number;
    members: number;
    core: number;
    border: number;
    hullVertices: number;
    hullSegments: number;
  };
  /** Share of records that sit exactly on another record (stacked dots). */
  stackedShare: number;
  analysis: ClusterAnalysis | null;
  detail: DetailView | null;
  /** Colour slot of each k-means group. */
  partitionSlots: number[];
  discs: {key: string; samples: EpsilonSample[]} | null;
  water: {key: string; share: number} | null;
  sweep: SweepGraph | null;
  reader: SummaryReader;
  dirty: boolean;
  destroyed: boolean;
};

/**
 * Clusters one Chicago observation group on the GPU and tells the story of what the clusters mean.
 *
 * Compile-time choices (group subset, method, k-means settings, iteration cap, dense-box shortcut,
 * summation order, hull prefilter, the one-call recipe) rebuild the graphs after a short debounce;
 * DBSCAN epsilon and minimum points, the weight attribute and the distribution settings are
 * parameter-buffer writes that only re-run the graphs. Every time the parameters settle the labels,
 * core flags and hulls are read back once: the CPU turns them into the per-record size class
 * (colour), the hull meshes, the epsilon discs, the tooltips and the readouts. A second graph
 * compiled once sweeps eight epsilon values as parameter writes for the scale chart.
 */
export async function createNatureClusters(
  ctx: SceneContext<NatureClusterOptions>
): Promise<SceneInstance<NatureClusterOptions>> {
  const {device} = ctx;
  const observations = ctx.datasets.get('chicago-nature');
  const columns: NatureColumns = readNatureColumns(observations);
  const origin = columns.origin;
  const projection = new LocalMetricProjection(origin);
  const montrose = findPlace(CHICAGO, 'montrose-point');
  const montroseMeters: [number, number] = montrose
    ? projection.project(montrose.lngLat[0], montrose.lngLat[1])
    : [0, 0];
  let serial = 0;
  let destroyed = false;
  let measuring = false;
  let rebuildTimer: ReturnType<typeof setTimeout> | undefined;
  let loadToken = 0;
  const retired: ClusterSet[] = [];
  let appliedEpsilon = Number.NaN;
  let appliedMinimumPoints = Number.NaN;
  /** Class indices an interactive legend isolates; `null` shows everything. */
  let legendClasses: number[] | null = null;
  /** The city limit, loaded in the background for the hull-over-water estimate. */
  let city: CityGeometry | null = null;
  /** Identity of the previous k-means partition, so hues survive a change of k or seed. */
  let previousPartition: StableHuePrevious[] = [];
  let pendingFit = false;
  let furnitureKey = '';
  /** The hull under the pointer (so the tooltip only re-highlights on change). */
  let hoverGroup = -1;

  const sweepState = {
    serial: -1,
    running: false,
    awaiting: false,
    index: 0,
    stale: false,
    minimumPoints: 0,
    counts: [] as number[],
    result: null as {minimumPoints: number; counts: number[]} | null
  };

  const getKey = (options: NatureClusterOptions) =>
    [
      options.category,
      options.method,
      options.pipeline,
      options.denseBoxShortcut,
      options.sumOrder,
      options.k,
      options.initialization,
      options.seed,
      options.iterationStep,
      options.tolerance,
      options.prefilterLevels
    ].join('|');

  const getIterationCap = (options: NatureClusterOptions) =>
    ITERATION_LADDER[Math.min(Math.max(Math.round(options.iterationStep), 0), 5)];

  const writeGeometryParameters = (set: ClusterSet) => {
    const parameters = {
      origin: [0, 0] as [number, number],
      standardDeviations: Number(ctx.options.standardDeviations),
      ellipseConvention: ctx.options.ellipseConvention
    };
    set.geometryParameters.write(getGPUGeographicDistributionParameterValues(parameters));
    set.distributionParameters.write(getGPUGeographicDistributionParameterValues(parameters));
  };

  const subsetCache = new Map<string, Uint32Array>();
  const getSubset = (category: string) => {
    let subset = subsetCache.get(category);
    if (!subset) {
      const wanted = columns.categoryNames.indexOf(category);
      const indexes: number[] = [];
      for (let index = 0; index < columns.count; index++) {
        if (columns.category[index] === wanted) indexes.push(index);
      }
      subset = Uint32Array.from(indexes);
      subsetCache.set(category, subset);
    }
    return subset;
  };

  const makeWeights = (indexes: Uint32Array): Float32Array => {
    const flag =
      ctx.options.weight === 'researchGrade'
        ? columns.researchGrade
        : ctx.options.weight === 'introduced'
          ? columns.introduced
          : null;
    const weights = new Float32Array(indexes.length);
    for (let row = 0; row < indexes.length; row++)
      weights[row] = 1 + (flag ? 4 * flag[indexes[row]] : 0);
    return weights;
  };

  /** Share of records that share an exact coordinate with another record. */
  const getStackedShare = (positions: Float32Array): number => {
    const seen = new Map<string, number>();
    for (let row = 0; row < positions.length / 2; row++) {
      const key = `${positions[row * 2]},${positions[row * 2 + 1]}`;
      seen.set(key, (seen.get(key) ?? 0) + 1);
    }
    let stacked = 0;
    for (const count of seen.values()) if (count > 1) stacked += count;
    return positions.length ? stacked / (positions.length / 2) : 0;
  };

  function buildSet(options: NatureClusterOptions): ClusterSet {
    const setSerial = ++serial;
    const resources = new SpatialAnalysisResources(device, `nature-clusters-${setSerial}`);
    const indexes = getSubset(options.category);
    const pointCount = indexes.length;
    const positions = new Float32Array(pointCount * 2);
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (let row = 0; row < pointCount; row++) {
      const x = columns.positions[indexes[row] * 2];
      const y = columns.positions[indexes[row] * 2 + 1];
      positions[row * 2] = x;
      positions[row * 2 + 1] = y;
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }
    const bounds: [number, number, number, number] = [
      minX - BOUNDS_PADDING_METERS,
      minY - BOUNDS_PADDING_METERS,
      maxX + BOUNDS_PADDING_METERS,
      maxY + BOUNDS_PADDING_METERS
    ];
    const isDbscan = options.method === 'dbscan';
    const isRecipe = isDbscan && options.pipeline === 'recipe';
    const groupCount = isDbscan ? CLUSTER_CAPACITY : options.k;

    const positionsBuffer = resources.createBuffer('positions', positions);
    const weights = resources.createBuffer('weights', makeWeights(indexes));
    const clusteringParameters = resources.createParameterBuffer(
      'clustering-parameters',
      'float32',
      GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH
    );
    const geometryParameters = resources.createParameterBuffer(
      'geometry-parameters',
      'float32',
      GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH
    );
    const distributionParameters = resources.createParameterBuffer(
      'distribution-parameters',
      'float32',
      GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH
    );
    const labels = resources.createBuffer('labels', pointCount * 4);
    const coreFlags = resources.createBuffer('core-flags', pointCount * 4);
    const clusterCount = resources.createBuffer('cluster-count', 4);
    const clusterStoredCount = resources.createBuffer('cluster-stored-count', 4);
    const clusterOverflow = resources.createBuffer('cluster-overflow', 4);
    const convergence = resources.createBuffer('convergence', 8);
    const sizes = resources.createBuffer('sizes', groupCount * 4);
    const centroids = resources.createBuffer('centroids', groupCount * 8);
    const geometryCounts = resources.createBuffer('geometry-counts', groupCount * 4);
    const geometryBounds = resources.createBuffer('geometry-bounds', groupCount * 16);
    const meanCenters = resources.createBuffer('mean-centers', groupCount * 8);
    const weightedCenters = resources.createBuffer('weighted-centers', groupCount * 8);
    const medoids = resources.createBuffer('medoids', groupCount * 4);
    const ellipses = resources.createBuffer('ellipses', groupCount * 12);
    const geometryOverflow = resources.createBuffer('geometry-overflow', 4);
    const hullIndices = resources.createBuffer('hull-indices', HULL_CAPACITY * 4);
    const hullPositions = resources.createBuffer('hull-positions', HULL_CAPACITY * 8);
    const hullOffsets = resources.createBuffer('hull-offsets', (groupCount + 1) * 4);
    const hullCounts = resources.createBuffer('hull-counts', groupCount * 4);
    const hullOverflow = resources.createBuffer('hull-overflow', 4);
    const areas = resources.createBuffer('areas', groupCount * 4);
    const perimeters = resources.createBuffer('perimeters', groupCount * 4);
    const hullCentroids = resources.createBuffer('hull-centroids', groupCount * 8);

    // Drawing buffers the CPU fills at every settle.
    const pointSizes = resources.createBuffer('point-sizes', pointCount * 4);
    const pointSlots = resources.createBuffer('point-slots', pointCount * 4);
    const groupSizes = resources.createBuffer('group-sizes', groupCount * 4);
    const groupSlots = resources.createBuffer('group-slots', groupCount * 4);
    const noiseIds = resources.createBuffer('noise-ids', pointCount * 4);
    const memberIds = resources.createBuffer('member-ids', pointCount * 4);
    const coreIds = resources.createBuffer('core-ids', pointCount * 4);
    const borderIds = resources.createBuffer('border-ids', pointCount * 4);
    const hullTriangles = resources.createBuffer('hull-triangles', HULL_CAPACITY * 3 * 8);
    const hullTriangleFeatures = resources.createBuffer(
      'hull-triangle-features',
      HULL_CAPACITY * 3 * 4
    );
    const hullOutline = resources.createBuffer('hull-outline', HULL_CAPACITY * 16);
    const hullOutlineFeatures = resources.createBuffer('hull-outline-features', HULL_CAPACITY * 4);

    // ---- Clustering (and, for the recipe, the whole outline pipeline) ----
    let clusterGraph: CompiledGPUCommandGraph<void>;
    let shapeGraph: CompiledGPUCommandGraph<void> | null = null;
    if (isRecipe) {
      const graph = new GPUCommandGraph<void>(device, {id: `nature-recipe-${setSerial}`});
      const view = <F extends 'float32' | 'uint32' | 'float32x2' | 'float32x4'>(
        name: string,
        buffer: SetBuffer,
        format: F,
        length: number
      ) => importGraphBuffer(graph, name, buffer, format, length);
      addClusterAndOutlineRecipe(graph, {
        id: 'cluster-outline',
        positions: view('positions', positionsBuffer, 'float32x2', pointCount),
        clusteringParameters: clusteringParameters.importToGraph(graph),
        gridSize: GRID_SIZE,
        geometryParameters: geometryParameters.importToGraph(graph),
        maximumClusterCount: CLUSTER_CAPACITY,
        maximumVerticesPerHull: HULL_MAXIMUM_VERTICES,
        hullCapacity: HULL_CAPACITY,
        labels: view('labels', labels, 'uint32', pointCount),
        clusterCount: view('cluster-count', clusterCount, 'uint32', 1),
        counts: view('counts', geometryCounts, 'uint32', groupCount),
        bounds: view('bounds', geometryBounds, 'float32x4', groupCount),
        meanCenters: view('mean-centers', meanCenters, 'float32x2', groupCount),
        hullVertexIndices: view('hull-indices', hullIndices, 'uint32', HULL_CAPACITY),
        hullPositions: view('hull-positions', hullPositions, 'float32x2', HULL_CAPACITY),
        hullOffsets: view('hull-offsets', hullOffsets, 'uint32', groupCount + 1),
        hullCounts: view('hull-counts', hullCounts, 'uint32', groupCount),
        hullOverflow: view('hull-overflow', hullOverflow, 'uint32', 1),
        areas: view('areas', areas, 'float32', groupCount),
        perimeters: view('perimeters', perimeters, 'float32', groupCount),
        centroids: view('hull-centroids', hullCentroids, 'float32x2', groupCount)
      });
      clusterGraph = resources.track(graph.compile());
    } else {
      if (isDbscan) {
        const graph = new GPUCommandGraph<void>(device, {id: `nature-dbscan-${setSerial}`});
        const clusterIds = resources.createBuffer('cluster-ids', CLUSTER_CAPACITY * 4);
        graph.add(
          new GPUSpatialClustering({
            id: 'clusters',
            positions: importGraphBuffer(
              graph,
              'positions',
              positionsBuffer,
              'float32x2',
              pointCount
            ),
            parameters: clusteringParameters.importToGraph(graph),
            gridSize: GRID_SIZE,
            denseBoxShortcut: options.denseBoxShortcut,
            sumOrder: options.sumOrder,
            labels: importGraphBuffer(graph, 'labels', labels, 'uint32', pointCount),
            coreFlags: importGraphBuffer(graph, 'core-flags', coreFlags, 'uint32', pointCount),
            clusterCount: importGraphBuffer(graph, 'cluster-count', clusterCount, 'uint32', 1),
            clusters: {
              ids: importGraphBuffer(graph, 'cluster-ids', clusterIds, 'uint32', CLUSTER_CAPACITY),
              count: importGraphBuffer(
                graph,
                'cluster-stored-count',
                clusterStoredCount,
                'uint32',
                1
              ),
              overflow: importGraphBuffer(graph, 'cluster-overflow', clusterOverflow, 'uint32', 1)
            },
            clusterSizes: importGraphBuffer(
              graph,
              'cluster-sizes',
              sizes,
              'uint32',
              CLUSTER_CAPACITY
            ),
            clusterCentroids: importGraphBuffer(
              graph,
              'cluster-centroids',
              centroids,
              'float32x2',
              CLUSTER_CAPACITY
            )
          })
        );
        clusterGraph = resources.track(graph.compile());
      } else {
        const graph = new GPUCommandGraph<void>(device, {id: `nature-kmeans-${setSerial}`});
        graph.add(
          new GPUKMeans({
            id: 'kmeans',
            positions: importGraphBuffer(
              graph,
              'positions',
              positionsBuffer,
              'float32x2',
              pointCount
            ),
            k: options.k,
            iterations: getIterationCap(options),
            tolerance: options.tolerance,
            convergence: importGraphBuffer(graph, 'convergence', convergence, 'uint32', 2),
            initialization: options.initialization,
            seed: options.seed,
            labels: importGraphBuffer(graph, 'labels', labels, 'uint32', pointCount),
            centers: importGraphBuffer(graph, 'centers', centroids, 'float32x2', options.k),
            sizes: importGraphBuffer(graph, 'sizes', sizes, 'uint32', options.k)
          })
        );
        clusterGraph = resources.track(graph.compile());
      }

      // ---- Description graph: GPUGroupGeometry + GPUGroupConvexHull ----
      const shapes = new GPUCommandGraph<void>(device, {id: `nature-shapes-${setSerial}`});
      const shapePositions = importGraphBuffer(
        shapes,
        'positions',
        positionsBuffer,
        'float32x2',
        pointCount
      );
      const shapeLabels = importGraphBuffer(shapes, 'labels', labels, 'uint32', pointCount);
      shapes.add(
        new GPUGroupGeometry({
          id: 'geometry',
          positions: shapePositions,
          labels: shapeLabels,
          groupCount,
          weights: importGraphBuffer(shapes, 'weights', weights, 'float32', pointCount),
          parameters: geometryParameters.importToGraph(shapes),
          medoidMaximumGroupSize: MEDOID_MAXIMUM_GROUP_SIZE,
          output: {
            counts: importGraphBuffer(
              shapes,
              'geometry-counts',
              geometryCounts,
              'uint32',
              groupCount
            ),
            bounds: importGraphBuffer(
              shapes,
              'geometry-bounds',
              geometryBounds,
              'float32x4',
              groupCount
            ),
            meanCenters: importGraphBuffer(
              shapes,
              'mean-centers',
              meanCenters,
              'float32x2',
              groupCount
            ),
            weightedCenters: importGraphBuffer(
              shapes,
              'weighted-centers',
              weightedCenters,
              'float32x2',
              groupCount
            ),
            medoidIndices: importGraphBuffer(shapes, 'medoids', medoids, 'uint32', groupCount),
            ellipses: importGraphBuffer(shapes, 'ellipses', ellipses, 'float32', groupCount * 3),
            overflow: importGraphBuffer(shapes, 'geometry-overflow', geometryOverflow, 'uint32', 1)
          }
        })
      );
      shapes.add(
        new GPUGroupConvexHull({
          id: 'hulls',
          positions: shapePositions,
          labels: shapeLabels,
          groupCount,
          maximumVerticesPerGroup: HULL_MAXIMUM_VERTICES,
          totalCapacity: HULL_CAPACITY,
          prefilterLevels: Number(options.prefilterLevels),
          output: {
            vertexIndices: importGraphBuffer(
              shapes,
              'hull-indices',
              hullIndices,
              'uint32',
              HULL_CAPACITY
            ),
            vertexPositions: importGraphBuffer(
              shapes,
              'hull-positions',
              hullPositions,
              'float32x2',
              HULL_CAPACITY
            ),
            offsets: importGraphBuffer(
              shapes,
              'hull-offsets',
              hullOffsets,
              'uint32',
              groupCount + 1
            ),
            counts: importGraphBuffer(shapes, 'hull-counts', hullCounts, 'uint32', groupCount),
            overflow: importGraphBuffer(shapes, 'hull-overflow', hullOverflow, 'uint32', 1)
          }
        })
      );
      shapeGraph = resources.track(shapes.compile());
    }

    // ---- Whole-pattern distribution: GPUGeographicDistribution over the selected points ----
    const distributionCounts = resources.createBuffer('distribution-counts', 4);
    const distributionMean = resources.createBuffer('distribution-mean', 8);
    const distributionMedian = resources.createBuffer('distribution-median', 8);
    const distributionStandardDistance = resources.createBuffer(
      'distribution-standard-distance',
      4
    );
    const distributionEllipse = resources.createBuffer('distribution-ellipse', 12);
    const ellipseVertices = resources.createBuffer('ellipse-vertices', RING_VERTICES * 8);
    const circleVertices = resources.createBuffer('circle-vertices', RING_VERTICES * 8);
    const ellipseSegments = resources.createBuffer('ellipse-segments', RING_VERTICES * 16);
    const circleSegments = resources.createBuffer('circle-segments', RING_VERTICES * 16);
    const distribution = new GPUCommandGraph<void>(device, {
      id: `nature-distribution-${setSerial}`
    });
    {
      const view = <F extends 'float32' | 'uint32' | 'float32x2'>(
        name: string,
        buffer: SetBuffer,
        format: F,
        length: number
      ) => importGraphBuffer(distribution, name, buffer, format, length);
      const ellipseView = view('ellipse-vertices', ellipseVertices, 'float32x2', RING_VERTICES);
      const circleView = view('circle-vertices', circleVertices, 'float32x2', RING_VERTICES);
      distribution.add(
        new GPUGeographicDistribution({
          id: 'distribution',
          positions: view('positions', positionsBuffer, 'float32x2', pointCount),
          weights: view('weights', weights, 'float32', pointCount),
          polygonVertexCount: RING_VERTICES,
          parameters: distributionParameters.importToGraph(distribution),
          output: {
            counts: view('counts', distributionCounts, 'uint32', 1),
            meanCenters: view('mean', distributionMean, 'float32x2', 1),
            medianCenters: view('median', distributionMedian, 'float32x2', 1),
            standardDistances: view(
              'standard-distance',
              distributionStandardDistance,
              'float32',
              1
            ),
            ellipses: view('ellipse', distributionEllipse, 'float32', 3),
            ellipseVertices: ellipseView,
            circleVertices: circleView
          }
        })
      );
      for (const [name, vertices, segments] of [
        ['ellipse', ellipseView, ellipseSegments],
        ['circle', circleView, circleSegments]
      ] as const) {
        addKernelPass(distribution, {
          id: `${name}-segments`,
          invocationCount: RING_VERTICES,
          bindings: [
            {name: 'vertices', view: vertices, type: 'f32', access: 'read'},
            {
              name: 'segments',
              view: importGraphBuffer(
                distribution,
                `${name}-segments`,
                segments,
                'float32',
                RING_VERTICES * 4
              ),
              type: 'f32',
              access: 'read_write'
            }
          ],
          declarations: `const RING: u32 = ${RING_VERTICES}u;`,
          body: /* wgsl */ `
  let next = (index + 1u) % RING;
  segments[segmentsOffset + index * 4u] = vertices[verticesOffset + index * 2u];
  segments[segmentsOffset + index * 4u + 1u] = vertices[verticesOffset + index * 2u + 1u];
  segments[segmentsOffset + index * 4u + 2u] = vertices[verticesOffset + next * 2u];
  segments[segmentsOffset + index * 4u + 3u] = vertices[verticesOffset + next * 2u + 1u];`
        });
      }
    }
    const distributionGraph = resources.track(distribution.compile());

    // ---- One detail read when the parameters settle: labels, core flags, hulls, centres ----
    const detailSources: {name: string; buffer: SetBuffer; size: number}[] = [
      {name: 'labels', buffer: labels, size: pointCount * 4},
      {name: 'coreFlags', buffer: coreFlags, size: pointCount * 4},
      {name: 'clusterCount', buffer: clusterCount, size: 4},
      {name: 'clusterOverflow', buffer: clusterOverflow, size: 4},
      {name: 'geometryOverflow', buffer: geometryOverflow, size: 4},
      {name: 'hullOverflow', buffer: hullOverflow, size: 4},
      {name: 'convergence', buffer: convergence, size: 8},
      {name: 'hullOffsets', buffer: hullOffsets, size: (groupCount + 1) * 4},
      {name: 'hullCounts', buffer: hullCounts, size: groupCount * 4},
      {name: 'hullPositions', buffer: hullPositions, size: HULL_CAPACITY * 8},
      {name: 'geometryBounds', buffer: geometryBounds, size: groupCount * 16},
      {name: 'meanCenters', buffer: meanCenters, size: groupCount * 8},
      {name: 'weightedCenters', buffer: weightedCenters, size: groupCount * 8},
      {name: 'medoids', buffer: medoids, size: groupCount * 4},
      {name: 'areas', buffer: areas, size: groupCount * 4},
      {name: 'perimeters', buffer: perimeters, size: groupCount * 4},
      {name: 'distributionCounts', buffer: distributionCounts, size: 4},
      {name: 'distributionMean', buffer: distributionMean, size: 8},
      {name: 'distributionMedian', buffer: distributionMedian, size: 8},
      {name: 'distributionStandardDistance', buffer: distributionStandardDistance, size: 4},
      {name: 'distributionEllipse', buffer: distributionEllipse, size: 12}
    ];
    const layout = new Map<string, {offset: number; size: number}>();
    let offset = 0;
    for (const source of detailSources) {
      layout.set(source.name, {offset, size: source.size});
      offset += source.size;
    }
    const reader = new SummaryReader(
      resources,
      `detail-${setSerial}`,
      detailSources.map(({buffer, size}) => ({buffer, size})),
      bytes =>
        handleDetail(set, {
          u32: name => {
            const entry = layout.get(name)!;
            return new Uint32Array(bytes, entry.offset, entry.size / 4);
          },
          f32: name => {
            const entry = layout.get(name)!;
            return new Float32Array(bytes, entry.offset, entry.size / 4);
          }
        })
    );

    const set: ClusterSet = {
      serial: setSerial,
      key: getKey(options),
      config: {...options},
      resources,
      pointCount,
      positions,
      bounds,
      clusteringParameters,
      geometryParameters,
      distributionParameters,
      clusterGraph,
      shapeGraph,
      distributionGraph,
      groupCount,
      weights,
      positionsBuffer,
      labels,
      coreFlags,
      clusterCount,
      clusterStoredCount,
      clusterOverflow,
      convergence,
      sizes,
      centroids,
      geometryCounts,
      geometryBounds,
      meanCenters,
      weightedCenters,
      medoids,
      ellipses,
      geometryOverflow,
      hullOffsets,
      hullCounts,
      hullPositions,
      hullOverflow,
      areas,
      perimeters,
      distributionCounts,
      distributionMean,
      distributionMedian,
      distributionStandardDistance,
      distributionEllipse,
      ellipseSegments,
      circleSegments,
      pointSizes,
      pointSlots,
      groupSizes,
      groupSlots,
      noiseIds,
      memberIds,
      coreIds,
      borderIds,
      hullTriangles,
      hullTriangleFeatures,
      hullOutline,
      hullOutlineFeatures,
      drawCounts: {noise: 0, members: 0, core: 0, border: 0, hullVertices: 0, hullSegments: 0},
      stackedShare: getStackedShare(positions),
      analysis: null,
      detail: null,
      partitionSlots: [],
      discs: null,
      water: null,
      sweep: null,
      reader,
      dirty: true,
      destroyed: false
    };
    writeGeometryParameters(set);
    return set;
  }

  /**
   * Compiles the scratch sweep graph of a set: the same clustering contributor on its own labels
   * and parameter buffer, so eight epsilon runs never disturb what the map shows. Compiled once.
   */
  function ensureSweep(set: ClusterSet): void {
    if (set.sweep || set.destroyed || set.config.method !== 'dbscan') return;
    const {resources} = set;
    const graph = new GPUCommandGraph<void>(device, {id: `nature-sweep-${set.serial}`});
    const parameters = resources.createParameterBuffer(
      'sweep-parameters',
      'float32',
      GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH
    );
    const labels = resources.createBuffer('sweep-labels', set.pointCount * 4);
    const count = resources.createBuffer('sweep-count', 4);
    const ids = resources.createBuffer('sweep-ids', CLUSTER_CAPACITY * 4);
    const stored = resources.createBuffer('sweep-stored', 4);
    const overflow = resources.createBuffer('sweep-overflow', 4);
    graph.add(
      new GPUSpatialClustering({
        id: 'sweep',
        positions: importGraphBuffer(
          graph,
          'positions',
          set.positionsBuffer,
          'float32x2',
          set.pointCount
        ),
        parameters: parameters.importToGraph(graph),
        gridSize: GRID_SIZE,
        sumOrder: set.config.sumOrder,
        labels: importGraphBuffer(graph, 'labels', labels, 'uint32', set.pointCount),
        clusterCount: importGraphBuffer(graph, 'count', count, 'uint32', 1),
        clusters: {
          ids: importGraphBuffer(graph, 'ids', ids, 'uint32', CLUSTER_CAPACITY),
          count: importGraphBuffer(graph, 'stored', stored, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1)
        }
      })
    );
    const compiled = resources.track(graph.compile());
    const reader = new SummaryReader(
      resources,
      `sweep-${set.serial}`,
      [{buffer: count, size: 4}],
      bytes => handleSweepRun(set, new Uint32Array(bytes)[0])
    );
    set.sweep = {graph: compiled, parameters, clusterCount: count, reader};
    sweepState.serial = set.serial;
    sweepState.stale = true;
    sweepState.result = null;
  }

  function handleSweepRun(set: ClusterSet, clusters: number): void {
    if (destroyed || set.destroyed || set !== active || !sweepState.running) return;
    sweepState.counts.push(clusters);
    sweepState.awaiting = false;
    sweepState.index++;
    if (sweepState.index >= SWEEP_EPSILONS.length) {
      sweepState.running = false;
      sweepState.result = {minimumPoints: sweepState.minimumPoints, counts: [...sweepState.counts]};
      publishCharts(set);
      publishCost(set);
    }
  }

  // ---- Readback: from labels and hulls to everything the story draws and says ----

  function handleDetail(set: ClusterSet, detail: DetailView): void {
    if (destroyed || set.destroyed || set !== active) return;
    const isDbscan = set.config.method === 'dbscan';
    const isRecipe = isDbscan && set.config.pipeline === 'recipe';
    const analysis = analyzeLabels({
      positions: set.positions,
      labels: detail.u32('labels'),
      coreFlags: isDbscan && !isRecipe ? detail.u32('coreFlags') : null,
      groupCount: set.groupCount,
      hasNoise: isDbscan
    });
    attachHulls(
      analysis,
      detail.u32('hullOffsets'),
      detail.u32('hullCounts'),
      detail.f32('hullPositions')
    );
    set.analysis = analysis;
    set.detail = detail;
    set.discs = null;
    set.water = null;
    set.drawCounts.noise = writeIds(set.noiseIds, analysis.noiseIds);
    set.drawCounts.members = writeIds(set.memberIds, analysis.memberIds);
    set.drawCounts.core = writeIds(set.coreIds, analysis.coreIds);
    set.drawCounts.border = writeIds(set.borderIds, analysis.borderIds);
    set.pointSizes.write(analysis.pointSizes);
    set.groupSizes.write(analysis.groupSizes);
    if (!isDbscan) {
      const slots = colourPartition(
        analysis.groupCentres,
        set.groupCount,
        previousPartition,
        PARTITION_HUES
      );
      set.partitionSlots = slots;
      previousPartition = [];
      slots.forEach((slot, group) => {
        if (analysis.sizes[group] > 0) {
          previousPartition.push({
            center: [analysis.groupCentres[group * 2], analysis.groupCentres[group * 2 + 1]],
            color: slot
          });
        }
      });
      const labels = detail.u32('labels');
      const pointSlots = new Uint32Array(set.pointCount);
      for (let row = 0; row < set.pointCount; row++) {
        pointSlots[row] = slots[labels[row]] ?? 0;
      }
      set.pointSlots.write(pointSlots);
      set.groupSlots.write(Uint32Array.from(slots));
    }
    const mesh = buildHullMesh(analysis.hulls);
    set.drawCounts.hullVertices = mesh.triangleFeatures.length;
    set.drawCounts.hullSegments = mesh.outlineFeatures.length;
    if (mesh.triangleFeatures.length > 0) {
      set.hullTriangles.write(mesh.triangles);
      set.hullTriangleFeatures.write(mesh.triangleFeatures);
      set.hullOutline.write(mesh.outline);
      set.hullOutlineFeatures.write(mesh.outlineFeatures);
    }
    publishAll(set);
    ctx.requestLayers();
  }

  /** Writes an id list (skipping an empty one, which a buffer write rejects); returns its length. */
  function writeIds(buffer: SetBuffer, ids: Uint32Array): number {
    if (ids.length > 0) buffer.write(ids);
    return ids.length;
  }

  const toLngLat = (x: number, y: number): LngLat => projection.unproject(x, y);

  const getRankedHull = (set: ClusterSet, group: number): HullShape | undefined =>
    set.analysis?.hulls.find(hull => hull.group === group);

  function publishAll(set: ClusterSet): void {
    publishReadouts(set);
    publishDerived(set);
    publishCharts(set);
    publishLegend(set);
    publishCost(set);
    publishFurniture(set);
  }

  /** Everything that follows from the options and the stored analysis (cheap, no GPU). */
  function publishDerived(set: ClusterSet): void {
    publishAnnotations(set);
    publishWinner(set);
    publishFurniture(set);
  }

  function publishReadouts(set: ClusterSet): void {
    const analysis = set.analysis;
    const detail = set.detail;
    if (!analysis || !detail) return;
    const isDbscan = set.config.method === 'dbscan';
    const isRecipe = isDbscan && set.config.pipeline === 'recipe';
    const total = set.pointCount;
    const largest = analysis.ranked.length ? analysis.sizes[analysis.ranked[0]] : 0;
    ctx.setReadout('points', formatCount(total));
    ctx.setReadout('stacked', formatPercent(set.stackedShare, 0));
    ctx.setReadout('clusters', formatCount(analysis.clusterCount));
    ctx.setReadout(
      'clusteredShare',
      isDbscan ? formatPercent(total ? analysis.memberCount / total : 0, 0) : '100%'
    );
    ctx.setReadout(
      'clustered',
      `${formatCount(analysis.memberCount)} / ${formatCount(analysis.noiseCount)}`
    );
    ctx.setReadout('largest', formatCount(largest));
    ctx.setReadout(
      'meanSize',
      analysis.clusterCount > 0 ? (analysis.memberCount / analysis.clusterCount).toFixed(1) : null
    );
    ctx.setReadout('coreCount', isRecipe ? 'n/a (recipe)' : formatCount(analysis.coreCount));
    ctx.setReadout('borderCount', isRecipe ? 'n/a (recipe)' : formatCount(analysis.borderCount));
    ctx.setReadout('noiseCount', formatCount(analysis.noiseCount));
    ctx.setReadout(
      'parameters',
      isDbscan
        ? `DBSCAN ${ctx.options.epsilon} m / ${ctx.options.minimumPoints} points${set.config.denseBoxShortcut ? ', dense-box' : ''}${isRecipe ? ', recipe' : ''}`
        : `K-means k = ${set.config.k}, ${set.config.initialization}${set.config.initialization === 'kmeans++' ? ` seed ${set.config.seed}` : ''}, cap ${getIterationCap(set.config)}`
    );
    const convergence = detail.u32('convergence');
    ctx.setReadout(
      'convergence',
      isDbscan
        ? 'n/a (DBSCAN)'
        : convergence[1]
          ? `converged after ${convergence[0]} of ${getIterationCap(set.config)} iterations`
          : `not converged: stopped at the ${convergence[0]} iteration cap`
    );
    if (!isDbscan) {
      let farthest = 0;
      const labels = detail.u32('labels');
      for (let row = 0; row < set.pointCount; row++) {
        const group = labels[row];
        farthest = Math.max(
          farthest,
          Math.hypot(
            set.positions[row * 2] - analysis.groupCentres[group * 2],
            set.positions[row * 2 + 1] - analysis.groupCentres[group * 2 + 1]
          )
        );
      }
      ctx.setReadout('farthest', `${formatDistance(farthest)} from its group centre`);
    } else {
      ctx.setReadout('farthest', 'n/a (DBSCAN leaves noise)');
    }
    const hullFlags = [
      detail.u32('hullOverflow')[0] & GPU_GROUP_CONVEX_HULL_GROUP_OVERFLOW
        ? 'a hull over 256 vertices (dropped)'
        : '',
      detail.u32('hullOverflow')[0] & GPU_GROUP_CONVEX_HULL_TOTAL_OVERFLOW ? 'total capacity' : ''
    ].filter(Boolean);
    ctx.setReadout(
      'hulls',
      `${formatCount(analysis.hulls.length)} hulls${hullFlags.length ? `, OVERFLOW: ${hullFlags.join(', ')}` : ''}`
    );
    const winner = analysis.ranked[0];
    if (winner !== undefined) {
      const meanCenters = detail.f32('meanCenters');
      const meanX = meanCenters[winner * 2];
      const meanY = meanCenters[winner * 2 + 1];
      if (isRecipe) {
        ctx.setReadout('weightedShift', 'n/a (recipe)');
        ctx.setReadout('medoid', 'n/a (recipe)');
        const gpuAreas = detail.f32('areas');
        let areaTotal = 0;
        for (const group of analysis.ranked) areaTotal += gpuAreas[group];
        ctx.setReadout(
          'recipeHulls',
          `GPU areas: largest ${formatArea(gpuAreas[winner])}, all ${formatArea(areaTotal)}; largest perimeter ${formatDistance(detail.f32('perimeters')[winner])}`
        );
      } else {
        const weighted = detail.f32('weightedCenters');
        ctx.setReadout(
          'weightedShift',
          `${Math.hypot(weighted[winner * 2] - meanX, weighted[winner * 2 + 1] - meanY).toFixed(1)} m`
        );
        const medoidRow = detail.u32('medoids')[winner];
        ctx.setReadout(
          'medoid',
          medoidRow === 0xffffffff
            ? `none (cluster over ${MEDOID_MAXIMUM_GROUP_SIZE} members)`
            : `${Math.hypot(set.positions[medoidRow * 2] - meanX, set.positions[medoidRow * 2 + 1] - meanY).toFixed(0)} m from the mean centre`
        );
        ctx.setReadout('recipeHulls', 'recipe pipeline only');
      }
    }
    // Distribution of the whole selected pattern.
    const mean = detail.f32('distributionMean');
    const median = detail.f32('distributionMedian');
    ctx.setReadout(
      'medianGap',
      `${formatDistance(Math.hypot(median[0] - mean[0], median[1] - mean[1]))} apart`
    );
    ctx.setReadout(
      'standardDistance',
      formatDistance(detail.f32('distributionStandardDistance')[0])
    );
    const ellipse = detail.f32('distributionEllipse');
    ctx.setReadout(
      'ellipse',
      `${formatDistance(Math.max(ellipse[1], ellipse[2]))} by ${formatDistance(Math.min(ellipse[1], ellipse[2]))}, rotated ${((ellipse[0] * 180) / Math.PI).toFixed(0)} deg`
    );
    ctx.setReadout(
      'numerics',
      `Hull cap ${HULL_MAXIMUM_VERTICES} vertices per hot spot, ${formatCount(HULL_CAPACITY)} in all; sums ${set.config.sumOrder}; dense-box ${set.config.denseBoxShortcut ? 'on' : 'off'} (3 to 9 % at 200,000 points, rarely pays at this size)`
    );
  }

  /** Hull area, records per km², and the share of the biggest hull that is not city land. */
  function publishWinner(set: ClusterSet): void {
    const analysis = set.analysis;
    if (!analysis || set.config.method !== 'dbscan' || analysis.ranked.length === 0) {
      for (const id of ['winnerSize', 'hullArea', 'hullWaterShare', 'density']) {
        ctx.setReadout(id, null);
      }
      return;
    }
    const winner = analysis.ranked[0];
    const size = analysis.sizes[winner];
    ctx.setReadout('winnerSize', formatCount(size));
    const hull = getRankedHull(set, winner);
    if (!hull) {
      ctx.setReadout('hullArea', 'hull over the vertex cap');
      ctx.setReadout('hullWaterShare', null);
      ctx.setReadout('density', null);
      return;
    }
    ctx.setReadout('hullArea', formatArea(hull.areaSquareMeters));
    ctx.setReadout('density', formatRate(size / (hull.areaSquareMeters / 1e6), 'km²'));
    if (ctx.options.zoomToWinner && city) {
      const key = `${set.serial}|${hull.group}|${hull.size}|${hull.areaSquareMeters.toFixed(0)}`;
      if (set.water?.key !== key) {
        set.water = {
          key,
          share: getShareOutsideCity(
            hull,
            city.containsMeters,
            createSeededRandom(1),
            MONTE_CARLO_SAMPLES
          )
        };
      }
      ctx.setReadout('hullWaterShare', formatPercent(set.water.share, 0));
    } else {
      ctx.setReadout('hullWaterShare', ctx.options.zoomToWinner ? 'measuring...' : null);
    }
    if (pendingFit) {
      pendingFit = false;
      const bounds = getGroupBounds(set, winner);
      if (bounds) {
        // Let the step's own camera flight start first; this refines it to the data.
        setTimeout(() => {
          if (!destroyed) ctx.fitBounds(bounds, {transitionMs: 1200, maxZoom: MAXIMUM_WINNER_ZOOM});
        }, 450);
      }
    }
  }

  /** Geographic bounds `[west, south, east, north]` of one cluster's members. */
  function getGroupBounds(set: ClusterSet, group: number): [number, number, number, number] | null {
    const bounds = set.detail?.f32('geometryBounds');
    if (!bounds || !((set.analysis?.sizes[group] ?? 0) > 0)) return null;
    const [west, south] = toLngLat(bounds[group * 4], bounds[group * 4 + 1]);
    const [east, north] = toLngLat(bounds[group * 4 + 2], bounds[group * 4 + 3]);
    return [west, south, east, north];
  }

  function publishAnnotations(set: ClusterSet): void {
    const analysis = set.analysis;
    const detail = set.detail;
    const options = ctx.options;
    const isDbscan = set.config.method === 'dbscan';
    const hotspots = options.view === 'hotspots' && isDbscan;
    // The three epsilon discs: one core, one border, one noise record, counted on the CPU.
    if (
      analysis &&
      detail &&
      hotspots &&
      options.showEpsilonDiscs &&
      set.config.pipeline === 'assembled'
    ) {
      const key = `${set.serial}|${options.epsilon}|${options.minimumPoints}`;
      if (set.discs?.key !== key) {
        set.discs = {
          key,
          samples: pickEpsilonSamples({
            positions: set.positions,
            labels: detail.u32('labels'),
            coreFlags: detail.u32('coreFlags'),
            epsilon: options.epsilon,
            minimumPoints: options.minimumPoints,
            centre: montroseMeters,
            windowMeters: DISC_WINDOW_METERS
          })
        };
      }
      ctx.setAnnotations(
        'epsilon-discs',
        set.discs.samples.map(
          (sample): MapAnnotation => ({
            kind: 'ring',
            id: `disc-${sample.role}`,
            coordinate: toLngLat(sample.x, sample.y),
            radiusMeters: options.epsilon,
            dashed: true,
            text: `${sample.neighbours} within ε: ${sample.role}`
          })
        )
      );
    } else {
      ctx.setAnnotations('epsilon-discs', null);
    }
    // The biggest hot spots, named from the gazetteer.
    if (analysis && hotspots && options.labelHotSpots) {
      ctx.setAnnotations(
        'top-spots',
        analysis.ranked.slice(0, 3).map((group, rank): MapAnnotation => {
          const centre = toLngLat(
            analysis.groupCentres[group * 2],
            analysis.groupCentres[group * 2 + 1]
          );
          return {
            kind: 'note',
            id: `hot-spot-${rank}`,
            coordinate: centre,
            title: `${formatCount(analysis.sizes[group])} records`,
            text: nearestPlaceLabel(CHICAGO, centre) ?? undefined,
            tone: 'accent'
          };
        })
      );
    } else {
      ctx.setAnnotations('top-spots', null);
    }
    // Mean and median centre of the whole pattern, with the gap between them.
    if (detail && options.showDistribution) {
      const mean = detail.f32('distributionMean');
      const median = detail.f32('distributionMedian');
      const meanPoint = toLngLat(mean[0], mean[1]);
      const medianPoint = toLngLat(median[0], median[1]);
      ctx.setAnnotations('distribution', [
        {
          kind: 'point',
          id: 'mean-centre',
          coordinate: meanPoint,
          text: 'Mean centre',
          detail: nearestPlaceLabel(CHICAGO, meanPoint) ?? undefined,
          marker: 'none',
          rank: 'subject',
          priority: 3
        },
        {
          kind: 'point',
          id: 'median-centre',
          coordinate: medianPoint,
          text: 'Median centre',
          detail: nearestPlaceLabel(CHICAGO, medianPoint) ?? undefined,
          marker: 'none',
          rank: 'subject',
          tone: 'signal',
          priority: 3
        },
        {
          kind: 'dimension',
          id: 'median-gap',
          from: meanPoint,
          to: medianPoint,
          text: formatDistance(Math.hypot(median[0] - mean[0], median[1] - mean[1]))
        }
      ]);
    } else {
      ctx.setAnnotations('distribution', null);
    }
  }

  function publishCharts(set: ClusterSet): void {
    const analysis = set.analysis;
    const table = makeSizeTable(ctx.ground(), analysis?.noiseCount);
    if (analysis && set.config.method === 'dbscan' && analysis.ranked.length > 0) {
      const ranked = analysis.ranked.slice(0, MAXIMUM_BARS);
      const bars: ChartData = {
        kind: 'bars',
        values: ranked.map(group => analysis.sizes[group]),
        colors: ranked.map(group => {
          const color = table.colors[getClassIndexForSize(analysis.sizes[group])];
          return [color[0], color[1], color[2], 255] as const;
        }),
        yScale: 'log',
        xLabel: 'Hot spots, largest first',
        yLabel: 'Records',
        description: `Sizes of the ${ranked.length} largest hot spots, largest first, on a log scale, coloured by size class.`,
        onBarClick: index => focusHotSpot(set, ranked[index])
      };
      ctx.setChart('sizeBars', bars);
    } else {
      ctx.setChart('sizeBars', null);
    }
    const result = sweepState.result;
    if (result && set.config.method === 'dbscan') {
      ctx.setChart('epsilonCurve', {
        kind: 'line',
        series: [
          {
            label: `Hot spots at ${result.minimumPoints} points`,
            x: SWEEP_EPSILONS,
            y: result.counts,
            points: true,
            directLabel: false
          }
        ],
        xLabel: 'Neighbour radius ε (m)',
        yLabel: 'Hot spots',
        yDomain: [0, Math.max(...result.counts, 1) * 1.1],
        link: {option: 'epsilon', label: value => `ε = ${value} m`},
        description:
          'Number of hot spots against the neighbour radius epsilon: it rises while small piles appear, then falls as hot spots merge.',
        table: true
      });
    } else {
      ctx.setChart('epsilonCurve', null);
    }
  }

  function getClassIndexForSize(size: number): number {
    return size >= 1000 ? 3 : size >= 200 ? 2 : size >= 50 ? 1 : 0;
  }

  /** Flies to a hot spot and outlines it (a click on a size bar). */
  function focusHotSpot(set: ClusterSet, group: number): void {
    const bounds = getGroupBounds(set, group);
    const hull = getRankedHull(set, group);
    if (hull) ctx.setHighlight({kind: 'polygon', rings: [getHullRing(hull)], pulse: true});
    if (bounds) ctx.fitBounds(bounds, {transitionMs: 1200, maxZoom: MAXIMUM_WINNER_ZOOM});
  }

  /** A hull as a closed `[lng, lat]` ring. */
  function getHullRing(hull: HullShape): LngLat[] {
    const ring: LngLat[] = [];
    for (let i = 0; i < hull.ring.length / 2; i++) {
      ring.push(toLngLat(hull.ring[i * 2], hull.ring[i * 2 + 1]));
    }
    ring.push(ring[0]);
    return ring;
  }

  function publishLegend(set: ClusterSet): void {
    const analysis = set.analysis;
    ctx.setLegendData('ground', ctx.ground());
    if (!analysis) return;
    ctx.setLegendData('table', makeSizeTable(ctx.ground(), analysis.noiseCount));
    ctx.setLegendData('classCounts', analysis.classCounts);
  }

  function publishCost(set: ClusterSet): void {
    const isDbscan = set.config.method === 'dbscan';
    ctx.setCost({
      records: set.pointCount,
      passes: isDbscan ? 5 : getIterationCap(set.config),
      note: isDbscan
        ? sweepState.result
          ? `compiled once, ${SWEEP_EPSILONS.length} runs`
          : 'epsilon is a parameter write'
        : 'compiled per iteration cap'
    });
  }

  /** The cartouche line that names the live parameters, and the scale-bar tick at epsilon. */
  function publishFurniture(set: ClusterSet): void {
    const options = ctx.options;
    const isDbscan = set.config.method === 'dbscan';
    const hotspots = options.view === 'hotspots';
    const group = options.category.toLowerCase();
    const subtitle = !hotspots
      ? undefined
      : isDbscan
        ? `DBSCAN, ε ${options.epsilon} m, at least ${options.minimumPoints} records · ${group}`
        : `k-means, k ${set.config.k}, up to ${getIterationCap(set.config)} iterations · ${group}`;
    const tick = hotspots && isDbscan ? options.epsilon : null;
    const key = `${subtitle}|${tick}|${set.pointCount}`;
    if (key === furnitureKey) return;
    furnitureKey = key;
    ctx.setFurniture({
      title: {
        ...(subtitle ? {subtitle} : {}),
        sample: `${formatCount(set.pointCount)} iNaturalist ${group} records, Chicago, 2023`
      },
      scaleBar: {units: 'metric', ...(tick ? {ticks: [tick]} : {})}
    });
  }

  const destroySet = (set: ClusterSet) => {
    set.destroyed = true;
    set.reader.stop();
    set.sweep?.reader.stop();
    set.resources.destroy();
  };

  let active: ClusterSet = buildSet(ctx.options);
  furnitureKey = '';

  const scheduleSweepCompile = (set: ClusterSet) => {
    setTimeout(() => {
      if (destroyed || set.destroyed || set !== active) return;
      try {
        ensureSweep(set);
      } catch {
        // Without the sweep graph the scale chart stays empty; the map is unaffected.
      }
    }, 150);
  };
  scheduleSweepCompile(active);

  // The city limit is only needed for the hull-over-water estimate; load it after the first paint.
  void loadCityGeometry(ctx, origin)
    .then(geometry => {
      city = geometry;
      if (!destroyed) publishWinner(active);
    })
    .catch(() => {
      // Without the city limit the hull-over-water readout stays empty.
    });

  const scheduleRebuild = () => {
    clearTimeout(rebuildTimer);
    rebuildTimer = setTimeout(() => {
      if (destroyed) return;
      const token = ++loadToken;
      ctx.setStatus('Rebuilding the cluster graphs…');
      // Let the status paint before the synchronous compile.
      setTimeout(() => {
        if (destroyed || token !== loadToken) return;
        try {
          const next = buildSet(ctx.options);
          const previous = active;
          active = next;
          appliedEpsilon = Number.NaN;
          appliedMinimumPoints = Number.NaN;
          furnitureKey = '';
          sweepState.running = false;
          sweepState.awaiting = false;
          sweepState.result = null;
          previous.destroyed = true;
          previous.reader.stop();
          previous.sweep?.reader.stop();
          retired.push(previous);
          setTimeout(() => {
            const index = retired.indexOf(previous);
            if (index >= 0) {
              retired.splice(index, 1);
              previous.resources.destroy();
            }
          }, 500);
          ctx.setStatus('');
          scheduleSweepCompile(next);
          ctx.requestLayers();
        } catch (error) {
          ctx.setStatus(`Could not rebuild: ${String(error)}`);
        }
      }, 30);
    }, REBUILD_DEBOUNCE_MILLISECONDS);
  };

  /** Times DBSCAN with and without the dense-box shortcut on scratch buffers. */
  async function measureShortcut(): Promise<void> {
    const set = active;
    if (measuring || destroyed) return;
    if (set.config.method !== 'dbscan' || set.config.pipeline === 'recipe') {
      ctx.setReadout('timing', 'select DBSCAN with the assembled pipeline');
      return;
    }
    measuring = true;
    ctx.setReadout('timing', 'measuring…');
    const scratch = new SpatialAnalysisResources(device, `nature-timing-${set.serial}`);
    try {
      set.clusteringParameters.write(
        getGPUSpatialClusteringParameterValues({
          bounds: set.bounds,
          epsilon: ctx.options.epsilon,
          minimumPoints: ctx.options.minimumPoints
        })
      );
      const results: {milliseconds: number; label: string}[] = [];
      for (const denseBoxShortcut of [false, true]) {
        const graph = new GPUCommandGraph<void>(device, {id: `nature-timing-${denseBoxShortcut}`});
        const labels = scratch.createBuffer(`labels-${denseBoxShortcut}`, set.pointCount * 4);
        const count = scratch.createBuffer(`count-${denseBoxShortcut}`, 4);
        const ids = scratch.createBuffer(`ids-${denseBoxShortcut}`, CLUSTER_CAPACITY * 4);
        const stored = scratch.createBuffer(`stored-${denseBoxShortcut}`, 4);
        const overflow = scratch.createBuffer(`overflow-${denseBoxShortcut}`, 4);
        graph.add(
          new GPUSpatialClustering({
            id: 'clusters',
            positions: importGraphBuffer(
              graph,
              'positions',
              set.positionsBuffer,
              'float32x2',
              set.pointCount
            ),
            parameters: set.clusteringParameters.importToGraph(graph),
            gridSize: GRID_SIZE,
            denseBoxShortcut,
            sumOrder: set.config.sumOrder,
            labels: importGraphBuffer(graph, 'labels', labels, 'uint32', set.pointCount),
            clusterCount: importGraphBuffer(graph, 'count', count, 'uint32', 1),
            clusters: {
              ids: importGraphBuffer(graph, 'ids', ids, 'uint32', CLUSTER_CAPACITY),
              count: importGraphBuffer(graph, 'stored', stored, 'uint32', 1),
              overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1)
            }
          })
        );
        const compiled = scratch.track(graph.compile());
        const timing = await measureCompiledGraph(device, compiled, {
          parameters: undefined,
          completionBuffer: count,
          signal: ctx.signal
        });
        results.push({
          milliseconds: timing.milliseconds,
          label: formatCompiledGraphTiming(timing).split(' · ')[0]
        });
      }
      if (destroyed || set.destroyed) return;
      const [plain, dense] = results;
      const ratio = plain.milliseconds / Math.max(dense.milliseconds, 1e-6);
      ctx.setReadout(
        'timing',
        `plain ${plain.label}, dense-box ${dense.label} (${ratio.toFixed(2)}x) at ${ctx.options.epsilon} m / ${ctx.options.minimumPoints}`
      );
    } catch {
      // Interrupted by a rebuild or destroy.
    } finally {
      scratch.destroy();
      measuring = false;
    }
  }

  async function measureShapes(): Promise<void> {
    const set = active;
    if (measuring || destroyed) return;
    measuring = true;
    ctx.setReadout('shapeTiming', 'measuring…');
    try {
      const target = set.shapeGraph ?? set.clusterGraph;
      const timing = await measureCompiledGraph(device, target, {
        parameters: undefined,
        completionBuffer: set.hullOverflow,
        signal: ctx.signal
      });
      if (!destroyed && !set.destroyed) {
        ctx.setReadout(
          'shapeTiming',
          `${formatCompiledGraphTiming(timing).split(' · ')[0]} (${set.shapeGraph ? 'geometry + hulls' : 'whole recipe'}, prefilter ${set.config.prefilterLevels})`
        );
      }
    } catch {
      // Interrupted.
    } finally {
      measuring = false;
    }
  }

  // ---- Tooltips ----

  function getHullTooltip(set: ClusterSet, hull: HullShape): TooltipContent {
    const analysis = set.analysis!;
    const table = makeSizeTable(ctx.ground(), analysis.noiseCount);
    const isDbscan = set.config.method === 'dbscan';
    const centre = toLngLat(
      analysis.groupCentres[hull.group * 2],
      analysis.groupCentres[hull.group * 2 + 1]
    );
    const rank = analysis.ranked.indexOf(hull.group) + 1;
    const areaKm2 = hull.areaSquareMeters / 1e6;
    const swatch = isDbscan
      ? table.colors[hull.sizeClass]
      : getPartitionPalette(ctx.ground())[set.partitionSlots[hull.group] ?? 0];
    const rows: TooltipRow[] = [
      {
        label: 'Records',
        value: formatCount(hull.size),
        swatch,
        emphasis: true
      },
      {label: 'Hull area', value: formatArea(hull.areaSquareMeters)},
      {label: 'Density', value: formatRate(hull.size / Math.max(areaKm2, 1e-9), 'km²')},
      {label: 'Share of group', value: formatPercent(hull.size / set.pointCount, 1)}
    ];
    if (isDbscan)
      rows.splice(1, 0, {label: 'Size class', value: SIZE_CLASS_LABELS[hull.sizeClass]});
    return {
      title: nearestPlace(CHICAGO, centre)?.name ?? 'Hot spot',
      subtitle: isDbscan
        ? `Hot spot ${rank} of ${analysis.clusterCount}, ${nearestPlaceLabel(CHICAGO, centre) ?? ''}`.replace(
            /, $/,
            ''
          )
        : `Group ${rank} of ${analysis.clusterCount}`,
      rows,
      note: 'The convex hull encloses the records; it is not the footprint of the place.',
      highlight: {kind: 'polygon', rings: [getHullRing(hull)]} satisfies MapHighlight
    };
  }

  const clearHover = () => {
    if (hoverGroup !== -1) hoverGroup = -1;
    return null;
  };

  return {
    getCompiledGraphs: () =>
      [active.clusterGraph, active.shapeGraph, active.distributionGraph].filter(
        Boolean
      ) as unknown as CompiledGPUCommandGraph<never>[],

    setOption(id, value, state) {
      const compileIds = [
        'category',
        'method',
        'pipeline',
        'denseBoxShortcut',
        'sumOrder',
        'k',
        'initialization',
        'seed',
        'iterationStep',
        'tolerance',
        'prefilterLevels'
      ];
      if (compileIds.includes(id)) {
        if (getKey(state) !== active.key) scheduleRebuild();
        return;
      }
      if (id === 'weight') {
        active.weights.write(makeWeights(getSubset(state.category)));
        active.dirty = true;
      } else if (id === 'standardDeviations' || id === 'ellipseConvention') {
        writeGeometryParameters(active);
        active.dirty = true;
      } else if (id === 'epsilon') {
        // Applied in encode.
      } else if (id === 'minimumPoints') {
        sweepState.stale = true;
      } else {
        if (id === 'zoomToWinner' && value) pendingFit = true;
        publishDerived(active);
        ctx.requestLayers();
      }
    },

    onAction(id) {
      if (id === 'measureShortcut') void measureShortcut();
      else if (id === 'measureShapes') void measureShapes();
    },

    onThemeChange() {
      publishCharts(active);
      publishLegend(active);
      ctx.requestLayers();
    },

    onGroundChange() {
      publishCharts(active);
      publishLegend(active);
      ctx.requestLayers();
    },

    onLegendFilter(id, classes) {
      if (id !== 'hot-spot-size') return;
      legendClasses = classes === null ? null : [...classes];
      ctx.requestLayers();
    },

    encode(commandEncoder) {
      const set = active;
      const {epsilon, minimumPoints} = ctx.options;
      if (
        set.config.method === 'dbscan' &&
        (epsilon !== appliedEpsilon || minimumPoints !== appliedMinimumPoints)
      ) {
        appliedEpsilon = epsilon;
        appliedMinimumPoints = minimumPoints;
        set.dirty = true;
      }
      if (set.dirty) {
        set.dirty = false;
        if (set.config.method === 'dbscan') {
          set.clusteringParameters.write(
            getGPUSpatialClusteringParameterValues({bounds: set.bounds, epsilon, minimumPoints})
          );
        }
        set.clusterGraph.encode(commandEncoder, {parameters: undefined});
        set.shapeGraph?.encode(commandEncoder, {parameters: undefined});
        set.distributionGraph.encode(commandEncoder, {parameters: undefined});
        set.reader.markStale();
      }
      // The epsilon sweep: eight parameter writes of one scratch graph, one run per frame.
      const sweep = set.sweep;
      if (sweep && sweepState.serial === set.serial) {
        if (sweepState.stale && !sweepState.awaiting) {
          sweepState.stale = false;
          sweepState.running = true;
          sweepState.index = 0;
          sweepState.counts = [];
          sweepState.minimumPoints = ctx.options.minimumPoints;
        }
        if (sweepState.running && !sweepState.awaiting && !sweep.reader.isPending) {
          sweep.parameters.write(
            getGPUSpatialClusteringParameterValues({
              bounds: set.bounds,
              epsilon: SWEEP_EPSILONS[sweepState.index],
              minimumPoints: sweepState.minimumPoints
            })
          );
          sweep.graph.encode(commandEncoder, {parameters: undefined});
          sweepState.awaiting = true;
          sweep.reader.request(commandEncoder);
        } else {
          sweep.reader.flush(commandEncoder);
        }
      }
      set.reader.flush(commandEncoder);
    },

    getTooltip(event) {
      const set = active;
      const analysis = set.analysis;
      const options = ctx.options;
      if (!analysis || options.view !== 'hotspots' || !event.coordinate) return clearHover();
      const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
      const hull = options.showHulls ? findHullAt(analysis.hulls, x, y) : null;
      if (hull) {
        hoverGroup = hull.group;
        return getHullTooltip(set, hull);
      }
      if (set.config.method === 'dbscan' && options.showNoise) {
        const reach = Math.max(6 * ctx.getMetersPerPixel(), 15);
        let nearest = Infinity;
        for (const row of analysis.noiseIds) {
          const dx = set.positions[row * 2] - x;
          const dy = set.positions[row * 2 + 1] - y;
          nearest = Math.min(nearest, dx * dx + dy * dy);
        }
        if (Math.sqrt(nearest) <= reach) {
          return {
            title: 'Not in a hot spot',
            subtitle: 'Noise',
            note: `Fewer than ${options.minimumPoints} neighbours within ${options.epsilon} m, and no core point among them.`
          };
        }
      }
      return clearHover();
    },

    getLayers() {
      const set = active;
      const options = ctx.options;
      const ground = ctx.ground();
      const analysis = set.analysis;
      const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];
      const isDbscan = set.config.method === 'dbscan';
      const isRecipe = isDbscan && set.config.pipeline === 'recipe';
      const layers: Layer[] = [];
      const id = (name: string) => `nature-clusters-${name}-${set.serial}`;
      const noiseRadius = DENSE_POINT_RADIUS_STOPS.map(
        ([zoom, radius]) => [zoom, radius * 0.8] as const
      );
      const table = makeSizeTable(ground, analysis?.noiseCount);
      const tableProps = getClassTableLayerProps(table);
      const halo = getParameterHalo(ground, 120);

      if (options.view === 'records' || !analysis) {
        // The unanalysed record map (and the frame before the first readback): one ink dot each.
        layers.push(
          new SpatialAnalysisPointLayer({
            id: id('records'),
            coordinateOrigin,
            positions: set.positionsBuffer,
            instanceCount: set.pointCount,
            colormap: 'uniform',
            color: getSubjectColor(ground, options.view === 'records' ? 130 : 60),
            radiusPixels: DENSE_POINT_RADIUS_STOPS
          })
        );
      } else if (isDbscan) {
        if (options.showHulls && set.drawCounts.hullVertices > 0) {
          layers.push(
            new SpatialAnalysisPolygonLayer({
              id: id('hull-fill'),
              coordinateOrigin,
              triangles: set.hullTriangles,
              features: set.hullTriangleFeatures,
              vertexCount: set.drawCounts.hullVertices,
              values: set.groupSizes,
              valueFormat: 'float32',
              colormap: 'uniform',
              ...tableProps,
              highlightClasses: legendClasses,
              dimOpacity: 0.15,
              opacity: ground === 'dark' ? 0.18 : 0.14
            })
          );
        }
        if (options.showNoise && set.drawCounts.noise > 0) {
          layers.push(
            new SpatialAnalysisPointLayer({
              id: id('noise'),
              coordinateOrigin,
              positions: set.positionsBuffer,
              ids: set.noiseIds,
              instanceCount: set.drawCounts.noise,
              colormap: 'uniform',
              color: getNoiseColor(ground, 77),
              radiusPixels: noiseRadius
            })
          );
        }
        const memberRadius = options.showCore
          ? DENSE_POINT_RADIUS_STOPS.map(([zoom, radius]) => [zoom, radius * 1.35] as const)
          : DENSE_POINT_RADIUS_STOPS;
        const memberStyle = {
          coordinateOrigin,
          positions: set.positionsBuffer,
          values: set.pointSizes,
          valueFormat: 'float32' as const,
          colormap: 'uniform' as const,
          ...tableProps,
          highlightClasses: legendClasses,
          dimOpacity: 0.15
        };
        if (options.showCore && !isRecipe) {
          // Core: full class colour, filled. Border: the same colour, a faint fill and a 1 px ring.
          if (set.drawCounts.border > 0) {
            layers.push(
              new SpatialAnalysisPointLayer({
                ...memberStyle,
                id: id('border-fill'),
                ids: set.borderIds,
                instanceCount: set.drawCounts.border,
                radiusPixels: memberRadius,
                opacity: 0.25
              }),
              new SpatialAnalysisPointLayer({
                ...memberStyle,
                id: id('border-ring'),
                ids: set.borderIds,
                instanceCount: set.drawCounts.border,
                shape: 'ring',
                outlineWidthPixels: 1,
                radiusPixels: memberRadius
              })
            );
          }
          if (set.drawCounts.core > 0) {
            layers.push(
              new SpatialAnalysisPointLayer({
                ...memberStyle,
                id: id('core'),
                ids: set.coreIds,
                instanceCount: set.drawCounts.core,
                radiusPixels: memberRadius,
                outlineColor: halo,
                outlineWidthPixels: 0.5
              })
            );
          }
        } else if (set.drawCounts.members > 0) {
          layers.push(
            new SpatialAnalysisPointLayer({
              ...memberStyle,
              id: id('members'),
              ids: set.memberIds,
              instanceCount: set.drawCounts.members,
              radiusPixels: memberRadius,
              outlineColor: halo,
              outlineWidthPixels: 0.5
            })
          );
        }
        if (options.showHulls && set.drawCounts.hullSegments > 0) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: id('hull-outline'),
              coordinateOrigin,
              segments: set.hullOutline,
              instanceCount: set.drawCounts.hullSegments,
              valueIndices: set.hullOutlineFeatures,
              values: set.groupSizes,
              valueFormat: 'float32',
              colormap: 'uniform',
              ...tableProps,
              widthPixels: 1.4,
              cap: 'round',
              highlightClasses: legendClasses,
              dimOpacity: 0.15,
              opacity: 0.9
            })
          );
        }
      } else {
        // K-means: nominal groups, Okabe-Ito hues assigned by map colouring.
        const palette = getPartitionPalette(ground);
        if (options.showHulls && set.drawCounts.hullVertices > 0) {
          layers.push(
            new SpatialAnalysisPolygonLayer({
              id: id('hull-fill'),
              coordinateOrigin,
              triangles: set.hullTriangles,
              features: set.hullTriangleFeatures,
              vertexCount: set.drawCounts.hullVertices,
              values: set.groupSlots,
              valueFormat: 'uint32',
              colormap: 'category',
              palette,
              opacity: PARTITION_OPACITY
            })
          );
        }
        layers.push(
          new SpatialAnalysisPointLayer({
            id: id('partition-points'),
            coordinateOrigin,
            positions: set.positionsBuffer,
            instanceCount: set.pointCount,
            values: set.pointSlots,
            valueFormat: 'uint32',
            colormap: 'category',
            palette,
            radiusPixels: DENSE_POINT_RADIUS_STOPS,
            outlineColor: halo,
            outlineWidthPixels: 0.5
          })
        );
        if (options.showHulls && set.drawCounts.hullSegments > 0) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: id('hull-outline'),
              coordinateOrigin,
              segments: set.hullOutline,
              instanceCount: set.drawCounts.hullSegments,
              valueIndices: set.hullOutlineFeatures,
              values: set.groupSlots,
              valueFormat: 'uint32',
              colormap: 'category',
              palette,
              widthPixels: 1.2,
              cap: 'round',
              opacity: 0.9
            })
          );
        }
      }

      if (options.view === 'hotspots' && analysis) {
        const ink = getInk(ground);
        const inkPalette = [ink];
        const groupCount = set.groupCount;
        const markerProps = {
          coordinateOrigin,
          counts: set.geometryCounts,
          centers: set.meanCenters,
          medoidRows: set.medoids,
          positions: set.positionsBuffer,
          groupCount
        };
        if (options.showBounds) {
          layers.push(
            new BoundsOutlineLayer({
              id: id('bounds'),
              coordinateOrigin,
              counts: set.geometryCounts,
              bounds: set.geometryBounds,
              groupCount,
              palette: inkPalette,
              widthPixels: 1,
              opacity: 0.6
            })
          );
        }
        if (options.showClusterEllipses && !isRecipe) {
          layers.push(
            new EllipseOutlineLayer({
              id: id('ellipses'),
              coordinateOrigin,
              counts: set.geometryCounts,
              centers: set.weightedCenters,
              ellipses: set.ellipses,
              groupCount,
              segments: ELLIPSE_SEGMENTS,
              palette: inkPalette,
              widthPixels: 1.4,
              opacity: 0.8
            })
          );
        }
        if (options.showMeans) {
          layers.push(
            new GroupMarkerLayer({
              ...markerProps,
              id: id('mean-centers'),
              source: 'centers',
              shape: 'ring',
              sizePixels: 7,
              palette: inkPalette
            })
          );
        }
        if (options.showMedoids && !isRecipe) {
          layers.push(
            new GroupMarkerLayer({
              ...markerProps,
              id: id('medoids'),
              source: 'medoids',
              shape: 'disc',
              sizePixels: 3.5,
              palette: inkPalette
            })
          );
        }
        if (options.showWeighted && !isRecipe && options.weight !== 'none') {
          layers.push(
            new GroupMarkerLayer({
              ...markerProps,
              id: id('weighted-centers'),
              centers: set.weightedCenters,
              source: 'centers',
              shape: 'diamond',
              sizePixels: 6,
              palette: [getSignal(ground)]
            })
          );
        }
      }

      if (options.showDistribution) {
        const ink = getInk(ground);
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: id('distribution-circle'),
            coordinateOrigin,
            segments: set.circleSegments,
            instanceCount: RING_VERTICES,
            widthPixels: 1,
            dashArray: [6, 4],
            color: [ink[0], ink[1], ink[2], 200]
          }),
          new SpatialAnalysisSegmentLayer({
            id: id('distribution-ellipse'),
            coordinateOrigin,
            segments: set.ellipseSegments,
            instanceCount: RING_VERTICES,
            widthPixels: 2,
            color: ink
          }),
          new GroupMarkerLayer({
            id: id('distribution-mean'),
            coordinateOrigin,
            counts: set.distributionCounts,
            centers: set.distributionMean,
            medoidRows: set.distributionCounts,
            positions: set.positionsBuffer,
            groupCount: 1,
            source: 'centers',
            shape: 'ring',
            sizePixels: 10,
            palette: [ink]
          }),
          new GroupMarkerLayer({
            id: id('distribution-median'),
            coordinateOrigin,
            counts: set.distributionCounts,
            centers: set.distributionMedian,
            medoidRows: set.distributionCounts,
            positions: set.positionsBuffer,
            groupCount: 1,
            source: 'centers',
            shape: 'diamond',
            sizePixels: 8,
            palette: [getSignal(ground)]
          })
        );
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      loadToken++;
      clearTimeout(rebuildTimer);
      destroySet(active);
      for (const set of retired.splice(0)) set.resources.destroy();
    }
  };
}
