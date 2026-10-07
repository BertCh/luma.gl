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
  GPU_SPATIAL_CLUSTERING_NOISE,
  GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH,
  GPUGeographicDistribution,
  GPUGroupConvexHull,
  GPUGroupGeometry,
  GPUKMeans,
  GPUSpatialClustering,
  type GPUParameterBuffer
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {formatCompiledGraphTiming, measureCompiledGraph} from '../../engine/vector-timing';
import type {SceneContext, SceneInstance} from '../scene';
import {readNatureColumns, type NatureColumns} from './b1-nature-data';
import {
  BoundsOutlineLayer,
  ClusterCentroidLayer,
  EllipseOutlineLayer,
  GroupMarkerLayer,
  HullOutlineLayer
} from './b1-cluster-layers';

/** Option state of the nature-clusters scene. */
export type NatureClusterOptions = {
  category: string;
  method: 'dbscan' | 'kmeans';
  pipeline: 'assembled' | 'recipe';
  epsilon: number;
  minimumPoints: number;
  denseBoxShortcut: boolean;
  sumOrder: 'sorted' | 'atomic';
  k: number;
  initialization: 'first-valid' | 'kmeans++';
  seed: number;
  iterations: number;
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
};

/** DBSCAN starting parameters per group, chosen with a CPU sweep so each reads as distinct places. */
export const CLUSTER_DEFAULTS: Record<string, {epsilon: number; minimumPoints: number}> = {
  Plants: {epsilon: 200, minimumPoints: 25},
  Birds: {epsilon: 150, minimumPoints: 20},
  Insects: {epsilon: 200, minimumPoints: 25},
  Fungi: {epsilon: 150, minimumPoints: 15},
  Mammals: {epsilon: 400, minimumPoints: 8},
  'Spiders and kin': {epsilon: 400, minimumPoints: 8},
  'Amphibians and reptiles': {epsilon: 800, minimumPoints: 5}
};

const GRID_SIZE: readonly [number, number] = [256, 256];
const CLUSTER_CAPACITY = 2048;
const BOUNDS_PADDING_METERS = 10;
const MEDOID_MAXIMUM_GROUP_SIZE = 1024;
const HULL_MAXIMUM_VERTICES = 256;
const HULL_CAPACITY = 1 << 14;
const REBUILD_DEBOUNCE_MILLISECONDS = 250;
const READBACK_INTERVAL_FRAMES = 20;
const RING_VERTICES = 64;
const ELLIPSE_SEGMENTS = 48;

export const CLUSTER_PALETTE = [
  [78, 201, 255, 255],
  [255, 148, 72, 255],
  [189, 122, 255, 255],
  [87, 235, 168, 255],
  [255, 105, 168, 255],
  [245, 220, 87, 255],
  [107, 158, 255, 255],
  [255, 92, 92, 255]
] as const;
const NOISE_COLOR = [128, 138, 156, 120] as const;
const HIDDEN_COLOR = [0, 0, 0, 0] as const;
const MEAN_MARKER_COLOR = [255, 255, 255, 255] as const;
const WEIGHTED_MARKER_COLOR = [255, 214, 64, 255] as const;
const MEDOID_MARKER_COLOR = [20, 20, 30, 255] as const;

type SetBuffer = ReturnType<SpatialAnalysisResources['createBuffer']>;

type ClusterSet = {
  serial: number;
  key: string;
  config: NatureClusterOptions;
  resources: SpatialAnalysisResources;
  pointCount: number;
  positions: Float32Array;
  bounds: [number, number, number, number];
  kmeansTolerance: number;
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
  drawCommands: DrawCommandBuffer;
  reader: SummaryReader;
  dirty: boolean;
  destroyed: boolean;
};

/**
 * Clusters one Chicago observation group on the GPU, then describes every cluster and the whole pattern.
 *
 * Compile-time choices (group subset, method, k-means settings, dense-box shortcut, summation
 * order, hull prefilter, and the one-call recipe) rebuild the graphs after a short debounce; DBSCAN
 * epsilon and minimum points, the weight attribute and the distribution settings are parameter-buffer
 * writes that only re-run the graphs.
 */
export async function createNatureClusters(
  ctx: SceneContext<NatureClusterOptions>
): Promise<SceneInstance<NatureClusterOptions>> {
  const {device} = ctx;
  const observations = ctx.datasets.get('chicago-nature');
  const columns: NatureColumns = readNatureColumns(observations);
  const origin = columns.origin;
  let serial = 0;
  let destroyed = false;
  let measuring = false;
  let frameCount = 0;
  let wantReadback = true;
  let rebuildTimer: ReturnType<typeof setTimeout> | undefined;
  let loadToken = 0;
  const retired: ClusterSet[] = [];
  let appliedEpsilon = Number.NaN;
  let appliedMinimumPoints = Number.NaN;

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
      options.iterations,
      options.tolerance,
      options.prefilterLevels
    ].join('|');

  const writeGeometryParameters = (set: ClusterSet) => {
    const parameters = {
      origin: [0, 0] as [number, number],
      standardDeviations: Number(ctx.options.standardDeviations),
      ellipseConvention: ctx.options.ellipseConvention
    };
    set.geometryParameters.write(getGPUGeographicDistributionParameterValues(parameters));
    set.distributionParameters.write(getGPUGeographicDistributionParameterValues(parameters));
  };

  const subsetIndexes = (category: string): Uint32Array => {
    const wanted = columns.categoryNames.indexOf(category);
    const indexes: number[] = [];
    for (let index = 0; index < columns.count; index++) {
      if (columns.category[index] === wanted) indexes.push(index);
    }
    return Uint32Array.from(indexes);
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

  const subsetCache = new Map<string, Uint32Array>();
  const getSubset = (category: string) => {
    let subset = subsetCache.get(category);
    if (!subset) {
      subset = subsetIndexes(category);
      subsetCache.set(category, subset);
    }
    return subset;
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
    const kmeansTolerance = options.tolerance;
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
    const drawCommands = resources.track(
      new DrawCommandBuffer(device, {
        id: `nature-clusters-${setSerial}-draw`,
        type: 'draw',
        commands: [{vertexCount: 6, instanceCount: isDbscan ? 0 : options.k}]
      })
    );

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
            drawInstanceCount: graph.importGPUData(
              'draw-instance-count',
              drawCommands.getInstanceCountData(0)
            ),
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
            iterations: options.iterations,
            tolerance: kmeansTolerance,
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

    // ---- One summary read for every readout ----
    const reader = new SummaryReader(
      resources,
      `summary-${setSerial}`,
      [
        {buffer: clusterCount, size: 4},
        {buffer: clusterStoredCount, size: 4},
        {buffer: clusterOverflow, size: 4},
        {buffer: geometryOverflow, size: 4},
        {buffer: hullOverflow, size: 4},
        {buffer: convergence, size: 8},
        {buffer: sizes, size: groupCount * 4},
        {buffer: geometryCounts, size: groupCount * 4},
        {buffer: hullCounts, size: groupCount * 4},
        {buffer: meanCenters, size: groupCount * 8},
        {buffer: weightedCenters, size: groupCount * 8},
        {buffer: medoids, size: groupCount * 4},
        {buffer: areas, size: groupCount * 4},
        {buffer: perimeters, size: groupCount * 4},
        {buffer: distributionCounts, size: 4},
        {buffer: distributionMean, size: 8},
        {buffer: distributionMedian, size: 8},
        {buffer: distributionStandardDistance, size: 4},
        {buffer: distributionEllipse, size: 12}
      ],
      bytes => handleSummary(set, bytes)
    );

    const set: ClusterSet = {
      serial: setSerial,
      key: getKey(options),
      config: {...options},
      resources,
      pointCount,
      positions,
      bounds,
      kmeansTolerance,
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
      drawCommands,
      reader,
      dirty: true,
      destroyed: false
    };
    writeGeometryParameters(set);
    return set;
  }

  const formatMeters = (value: number) =>
    value >= 1000 ? `${(value / 1000).toFixed(2)} km` : `${value.toFixed(0)} m`;

  function handleSummary(set: ClusterSet, bytes: ArrayBuffer): void {
    if (destroyed || set.destroyed || set !== active) return;
    const words = new Uint32Array(bytes);
    const floats = new Float32Array(bytes);
    const groups = set.groupCount;
    const isDbscan = set.config.method === 'dbscan';
    const isRecipe = isDbscan && set.config.pipeline === 'recipe';
    const clusterTotal = isDbscan ? words[0] : set.config.k;
    const stored = isDbscan ? Math.min(words[1] || words[0], CLUSTER_CAPACITY) : set.config.k;
    const sizesStart = 8;
    const geometryStart = sizesStart + groups;
    const hullStart = geometryStart + groups;
    const meanStart = hullStart + groups;
    const weightedStart = meanStart + groups * 2;
    const medoidStart = weightedStart + groups * 2;
    const areaStart = medoidStart + groups;
    const perimeterStart = areaStart + groups;
    const distributionStart = perimeterStart + groups;
    // In recipe mode there is no stored-count, so count the non-empty groups.
    let clustered = 0;
    let largest = 0;
    let largestCluster = -1;
    let groupsWithMembers = 0;
    let hullTotal = 0;
    let hullLargest = 0;
    let areaTotal = 0;
    const limit = isRecipe ? Math.min(clusterTotal, CLUSTER_CAPACITY) : stored;
    for (let cluster = 0; cluster < limit; cluster++) {
      const size = isRecipe ? words[geometryStart + cluster] : words[sizesStart + cluster];
      if (size > 0) groupsWithMembers++;
      clustered += size;
      if (size > largest) {
        largest = size;
        largestCluster = cluster;
      }
      const hullVertices = words[hullStart + cluster];
      if (hullVertices > 0) hullTotal++;
      hullLargest = Math.max(hullLargest, hullVertices);
      if (isRecipe) areaTotal += floats[areaStart + cluster];
    }
    const noise = set.pointCount - clustered;
    ctx.setReadout(
      'points',
      `${formatCount(set.pointCount)} ${formatCategory(set.config.category)}`
    );
    ctx.setReadout(
      'clusters',
      `${formatCount(clusterTotal)}${isDbscan && words[2] ? ' (capacity overflow)' : ''}`
    );
    ctx.setReadout('clustered', `${formatCount(clustered)} / ${formatCount(noise)}`);
    ctx.setReadout('largest', formatCount(largest));
    ctx.setReadout(
      'meanSize',
      groupsWithMembers > 0 ? (clustered / groupsWithMembers).toFixed(1) : null
    );
    ctx.setReadout(
      'parameters',
      isDbscan
        ? `DBSCAN ${ctx.options.epsilon} m / ${ctx.options.minimumPoints} points${set.config.denseBoxShortcut ? ', dense-box' : ''}${isRecipe ? ', recipe' : ''}`
        : `K-means k = ${set.config.k}, ${set.config.initialization}${set.config.initialization === 'kmeans++' ? ` seed ${set.config.seed}` : ''}`
    );
    ctx.setReadout(
      'convergence',
      isDbscan
        ? 'n/a (DBSCAN)'
        : words[6]
          ? `converged after ${words[5]} of ${set.config.iterations} iterations`
          : `not converged: stopped at the ${words[5]} iteration cap`
    );
    const hullFlags = [
      words[4] & GPU_GROUP_CONVEX_HULL_GROUP_OVERFLOW ? 'a hull over 256 vertices' : '',
      words[4] & GPU_GROUP_CONVEX_HULL_TOTAL_OVERFLOW ? 'total capacity' : ''
    ].filter(Boolean);
    ctx.setReadout(
      'hulls',
      `${formatCount(hullTotal)} hulls, largest ${hullLargest} vertices${hullFlags.length ? `, OVERFLOW: ${hullFlags.join(', ')}` : ''}`
    );
    if (largestCluster >= 0) {
      const meanX = floats[meanStart + largestCluster * 2];
      const meanY = floats[meanStart + largestCluster * 2 + 1];
      if (isRecipe) {
        ctx.setReadout('weightedShift', 'n/a (recipe)');
        ctx.setReadout('medoid', 'n/a (recipe)');
        const area = floats[areaStart + largestCluster];
        ctx.setReadout(
          'hullArea',
          `largest ${(area / 1e6).toFixed(3)} km², all hulls ${(areaTotal / 1e6).toFixed(2)} km²`
        );
        ctx.setReadout('hullPerimeter', formatMeters(floats[perimeterStart + largestCluster]));
      } else {
        const weightedX = floats[weightedStart + largestCluster * 2];
        const weightedY = floats[weightedStart + largestCluster * 2 + 1];
        ctx.setReadout(
          'weightedShift',
          `${Math.hypot(weightedX - meanX, weightedY - meanY).toFixed(1)} m (${(weightedX - meanX).toFixed(1)} east, ${(weightedY - meanY).toFixed(1)} north)`
        );
        const medoidRow = words[medoidStart + largestCluster];
        if (medoidRow === 0xffffffff) {
          ctx.setReadout('medoid', `none (cluster over ${MEDOID_MAXIMUM_GROUP_SIZE} members)`);
        } else {
          const medoidX = set.positions[medoidRow * 2];
          const medoidY = set.positions[medoidRow * 2 + 1];
          ctx.setReadout(
            'medoid',
            `${Math.hypot(medoidX - meanX, medoidY - meanY).toFixed(0)} m from the mean centre`
          );
        }
        ctx.setReadout('hullArea', 'recipe pipeline only');
        ctx.setReadout('hullPerimeter', 'recipe pipeline only');
      }
    }
    // Distribution of the whole selected pattern.
    const meanDistributionX = floats[distributionStart + 1];
    const meanDistributionY = floats[distributionStart + 2];
    const medianX = floats[distributionStart + 3];
    const medianY = floats[distributionStart + 4];
    ctx.setReadout(
      'medianGap',
      `${formatMeters(Math.hypot(medianX - meanDistributionX, medianY - meanDistributionY))} apart`
    );
    ctx.setReadout('standardDistance', formatMeters(floats[distributionStart + 5]));
    const angle = floats[distributionStart + 6];
    const sigmaX = floats[distributionStart + 7];
    const sigmaY = floats[distributionStart + 8];
    ctx.setReadout(
      'ellipse',
      `${formatMeters(Math.max(sigmaX, sigmaY))} by ${formatMeters(Math.min(sigmaX, sigmaY))}, rotated ${((angle * 180) / Math.PI).toFixed(0)} deg`
    );
  }

  function formatCategory(name: string): string {
    const lower = name.toLowerCase();
    return lower.charAt(0).toUpperCase() + lower.slice(1);
  }

  const destroySet = (set: ClusterSet) => {
    set.destroyed = true;
    set.reader.stop();
    set.resources.destroy();
  };

  let active: ClusterSet = buildSet(ctx.options);

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
          previous.destroyed = true;
          previous.reader.stop();
          retired.push(previous);
          setTimeout(() => {
            const index = retired.indexOf(previous);
            if (index >= 0) {
              retired.splice(index, 1);
              previous.resources.destroy();
            }
          }, 500);
          wantReadback = true;
          ctx.setStatus('');
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

  return {
    getCompiledGraphs: () =>
      [active.clusterGraph, active.shapeGraph, active.distributionGraph].filter(
        Boolean
      ) as unknown as CompiledGPUCommandGraph<never>[],

    setOption(id, _value, state) {
      const compileIds = [
        'category',
        'method',
        'pipeline',
        'denseBoxShortcut',
        'sumOrder',
        'k',
        'initialization',
        'seed',
        'iterations',
        'tolerance',
        'prefilterLevels'
      ];
      if (compileIds.includes(id)) {
        if (id === 'category') {
          const defaults = CLUSTER_DEFAULTS[state.category];
          if (defaults) {
            // The step or panel sets epsilon too when it matters; keep the current values otherwise.
            void defaults;
          }
        }
        if (getKey(state) !== active.key) scheduleRebuild();
      } else if (id === 'weight') {
        active.weights.write(makeWeights(getSubset(state.category)));
        active.dirty = true;
      } else if (id === 'standardDeviations' || id === 'ellipseConvention') {
        writeGeometryParameters(active);
        active.dirty = true;
      } else if (id === 'epsilon' || id === 'minimumPoints') {
        // Applied in encode.
      } else {
        ctx.requestLayers();
      }
      wantReadback = true;
    },

    onAction(id) {
      if (id === 'measureShortcut') void measureShortcut();
      else if (id === 'measureShapes') void measureShapes();
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    encode(commandEncoder) {
      const set = active;
      frameCount++;
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
        wantReadback = true;
      }
      if (wantReadback || frameCount % READBACK_INTERVAL_FRAMES === 0) {
        if (!set.reader.isPending) {
          wantReadback = false;
          set.reader.request(commandEncoder);
          return;
        }
      }
      set.reader.flush(commandEncoder);
    },

    getLayers() {
      const set = active;
      const options = ctx.options;
      const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];
      const isDbscan = set.config.method === 'dbscan';
      const isRecipe = isDbscan && set.config.pipeline === 'recipe';
      const dark = ctx.theme() === 'dark';
      const radiusPixels = set.pointCount > 10000 ? 1.8 : 3;
      const id = (name: string) => `nature-clusters-${name}-${set.serial}`;
      const layers: Layer[] = [];
      if (options.showCore && isDbscan && !isRecipe) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: id('core'),
            coordinateOrigin,
            positions: set.positionsBuffer,
            instanceCount: set.pointCount,
            values: set.coreFlags,
            valueFormat: 'uint32',
            colormap: 'mask',
            color: dark ? [255, 255, 255, 110] : [20, 20, 40, 110],
            noDataColor: HIDDEN_COLOR,
            radiusPixels: radiusPixels + 2.5
          })
        );
      }
      layers.push(
        new SpatialAnalysisPointLayer({
          id: id('points'),
          coordinateOrigin,
          positions: set.positionsBuffer,
          instanceCount: set.pointCount,
          values: set.labels,
          valueFormat: 'uint32',
          colormap: 'category',
          palette: CLUSTER_PALETTE,
          noDataValue: GPU_SPATIAL_CLUSTERING_NOISE,
          noDataColor: options.showNoise ? NOISE_COLOR : HIDDEN_COLOR,
          radiusPixels
        })
      );
      const groupCount = set.groupCount;
      if (options.showBounds) {
        layers.push(
          new BoundsOutlineLayer({
            id: id('bounds'),
            coordinateOrigin,
            counts: set.geometryCounts,
            bounds: set.geometryBounds,
            groupCount,
            palette: CLUSTER_PALETTE,
            widthPixels: 1,
            opacity: 0.8
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
            palette: CLUSTER_PALETTE,
            widthPixels: 1.6
          })
        );
      }
      if (options.showHulls) {
        layers.push(
          new HullOutlineLayer({
            id: id('hulls'),
            coordinateOrigin,
            hullOffsets: set.hullOffsets,
            hullCounts: set.hullCounts,
            hullPositions: set.hullPositions,
            slotCount: HULL_CAPACITY,
            groupCount,
            palette: CLUSTER_PALETTE,
            widthPixels: 2.2
          })
        );
      }
      if (!isRecipe) {
        layers.push(
          new ClusterCentroidLayer({
            id: id('centroids'),
            coordinateOrigin,
            centroids: set.centroids,
            sizes: set.sizes,
            drawCommands: set.drawCommands,
            palette: CLUSTER_PALETTE,
            basePixels: 4,
            pixelsPerSqrtMember: 0.18,
            maximumPixels: 22,
            fillOpacity: 0.16
          })
        );
      }
      const markerProps = {
        coordinateOrigin,
        counts: set.geometryCounts,
        centers: set.meanCenters,
        medoidRows: set.medoids,
        positions: set.positionsBuffer,
        groupCount
      };
      if (options.showMeans) {
        layers.push(
          new GroupMarkerLayer({
            ...markerProps,
            id: id('mean-centers'),
            source: 'centers',
            shape: 'ring',
            sizePixels: 7,
            palette: [dark ? MEAN_MARKER_COLOR : [20, 20, 40, 255]]
          })
        );
      }
      if (options.showWeighted && !isRecipe) {
        layers.push(
          new GroupMarkerLayer({
            ...markerProps,
            id: id('weighted-centers'),
            centers: set.weightedCenters,
            source: 'centers',
            shape: 'diamond',
            sizePixels: 5,
            palette: [WEIGHTED_MARKER_COLOR]
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
            palette: [dark ? [240, 240, 250, 255] : MEDOID_MARKER_COLOR]
          })
        );
      }
      if (options.showDistribution) {
        const distributionColor = dark ? [255, 255, 255, 235] : [10, 10, 30, 235];
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: id('distribution-circle'),
            coordinateOrigin,
            segments: set.circleSegments,
            instanceCount: RING_VERTICES,
            widthPixels: 2.2,
            color: [distributionColor[0], distributionColor[1], distributionColor[2], 150]
          }),
          new SpatialAnalysisSegmentLayer({
            id: id('distribution-ellipse'),
            coordinateOrigin,
            segments: set.ellipseSegments,
            instanceCount: RING_VERTICES,
            widthPixels: 3,
            color: [255, 214, 64, 255]
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
            sizePixels: 12,
            widthPixels: 3,
            palette: [[255, 214, 64, 255]]
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
            sizePixels: 9,
            palette: [[255, 70, 70, 255]]
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

export type {Buffer};
