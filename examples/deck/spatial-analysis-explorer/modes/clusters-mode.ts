// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  GPUReadbackRing,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUGeographicDistributionParameterValues,
  getGPUSpatialClusteringParameterValues,
  GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH,
  GPU_GROUP_CONVEX_HULL_GROUP_OVERFLOW,
  GPU_GROUP_CONVEX_HULL_TOTAL_OVERFLOW,
  GPU_SPATIAL_CLUSTERING_NOISE,
  GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH,
  GPUGroupConvexHull,
  GPUGroupGeometry,
  GPUKMeans,
  GPUSpatialClustering,
  type GPUKMeansInitialization,
  type GPUParameterBuffer
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {importGraphBuffer} from '../graph-buffers';
import {SpatialAnalysisPointLayer} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {
  BoundsOutlineLayer,
  ClusterCentroidLayer,
  EllipseOutlineLayer,
  GroupMarkerLayer,
  HullOutlineLayer
} from './clusters-layers';
import {formatCompiledGraphTiming, measureCompiledGraph} from './vector-timing';

/** Maximum `[columns, rows]` of the neighbor-search lattice (compile-time). */
const GRID_SIZE: readonly [number, number] = [256, 256];
/** Compile-time cluster capacity; clusters beyond it are labeled but have no size or centroid. */
const CLUSTER_CAPACITY = 2048;
const READBACK_INTERVAL_FRAMES = 12;
/** Padding added around the data extent so every point is inside the inclusive bounds. */
const BOUNDS_PADDING_METERS = 10;
/** Lloyd iterations of `GPUKMeans` (compile-time, fixed). */
const KMEANS_ITERATIONS = 8;
/** Compile-time `k` range of the k-means slider (the contributor allows up to 256). */
const KMEANS_MAXIMUM_K = 64;
const KMEANS_DEFAULT_K = 16;
const KMEANS_MAXIMUM_SEED = 31;
/** Groups larger than this get no medoid (the search is quadratic per group). */
const MEDOID_MAXIMUM_GROUP_SIZE = 1024;
/** Largest convex hull emitted per group, and the total hull vertex capacity. */
const HULL_MAXIMUM_VERTICES = 256;
const HULL_CAPACITY = 1 << 14;
const REBUILD_DEBOUNCE_MILLISECONDS = 250;
const AUTO_MEASURE_FRAME = 40;
const SUMMARY_HEADER_WORDS = 5;
const ELLIPSE_SEGMENTS = 48;
/** Period in meters of the demo weight sawtooth along x. */
const WEIGHT_PERIOD_METERS = 250;

/**
 * Per-point-set starting parameters, chosen with a CPU DBSCAN sweep so each set reads as many
 * distinct neighborhoods (30-150 clusters, none above a few percent of the points) instead of one
 * giant connected component.
 */
const POINT_SET_DEFAULTS: Record<PointSetId, {epsilon: number; minimumPoints: number}> = {
  pickups: {epsilon: 130, minimumPoints: 5},
  pois: {epsilon: 70, minimumPoints: 8},
  vertices: {epsilon: 40, minimumPoints: 50}
};

/** Cluster `i` is drawn with `CLUSTER_PALETTE[i % 8]` by both the points and the centroids. */
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
const NOISE_COLOR = [128, 138, 156, 150] as const;
const MEAN_MARKER_COLOR = [255, 255, 255, 255] as const;
const WEIGHTED_MARKER_COLOR = [255, 214, 64, 255] as const;
const MEDOID_MARKER_COLOR = [20, 20, 30, 255] as const;

type PointSetId = 'pickups' | 'pois' | 'vertices';
type ClusterMethod = 'dbscan' | 'kmeans';

const POINT_SET_OPTIONS: readonly {value: PointSetId; label: string}[] = [
  {value: 'pickups', label: 'Trip pickups & drop-offs'},
  {value: 'pois', label: 'Points of interest'},
  {value: 'vertices', label: 'All trip vertices'}
];

const METHOD_OPTIONS: readonly {value: ClusterMethod; label: string}[] = [
  {value: 'dbscan', label: 'DBSCAN (GPUSpatialClustering)'},
  {value: 'kmeans', label: 'K-means (GPUKMeans)'}
];

const INITIALIZATION_OPTIONS: readonly {value: GPUKMeansInitialization; label: string}[] = [
  {value: 'first-valid', label: 'First k valid points'},
  {value: 'kmeans++', label: 'Seeded k-means++'}
];

type PointSetData = {
  positions: Float32Array;
  origin: readonly [number, number];
  attribution: string;
  label: string;
};

/** Compile-time choices; any change rebuilds the graphs. */
type ClusterConfig = {
  pointSet: PointSetId;
  method: ClusterMethod;
  k: number;
  initialization: GPUKMeansInitialization;
  seed: number;
  denseBoxShortcut: boolean;
};

type ClusterBuffer = ReturnType<SpatialAnalysisResources['createBuffer']>;

/** Caller-owned buffers of one `GPUSpatialClustering` instance. */
type DbscanOutputs = {
  labels: ClusterBuffer;
  coreFlags: ClusterBuffer;
  clusterCount: ClusterBuffer;
  clusterStoredCount: ClusterBuffer;
  clusterOverflow: ClusterBuffer;
  clusterSizes: ClusterBuffer;
  clusterCentroids: ClusterBuffer;
};

/** Everything that depends on the compile-time choices; destroyed as one unit. */
type ClusterSet = {
  serial: number;
  config: ClusterConfig;
  data: PointSetData;
  pointCount: number;
  resources: SpatialAnalysisResources;
  bounds: [number, number, number, number];
  parameters: GPUParameterBuffer<'float32'>;
  clusterGraph: CompiledGPUCommandGraph<void>;
  shapeGraph: CompiledGPUCommandGraph<void>;
  /** Number of groups the shape contributors summarize (cluster capacity or `k`). */
  groupCount: number;
  positionsBuffer: ClusterBuffer;
  labels: ClusterBuffer;
  /** DBSCAN only. */
  coreFlags: ClusterBuffer | null;
  clusterCount: ClusterBuffer | null;
  clusterStoredCount: ClusterBuffer | null;
  clusterOverflow: ClusterBuffer | null;
  sizes: ClusterBuffer;
  centroids: ClusterBuffer;
  geometryCounts: ClusterBuffer;
  geometryBounds: ClusterBuffer;
  meanCenters: ClusterBuffer;
  weightedCenters: ClusterBuffer;
  medoids: ClusterBuffer;
  ellipses: ClusterBuffer;
  geometryOverflow: ClusterBuffer;
  hullOffsets: ClusterBuffer;
  hullCounts: ClusterBuffer;
  hullPositions: ClusterBuffer;
  hullOverflow: ClusterBuffer;
  drawCommands: DrawCommandBuffer;
  readbackRing: GPUReadbackRing;
  summaryByteLength: number;
  /** Encode the clustering and shape graphs on the next frame. */
  dirty: boolean;
  destroyed: boolean;
};

/**
 * Clustering of New York points on the GPU, then a description of every cluster.
 *
 * - Method: `GPUSpatialClustering` (DBSCAN) or `GPUKMeans`. The point set, method, `k`, k-means
 *   initialization and seed and the DBSCAN dense-box shortcut are compile-time choices (they
 *   rebuild the graphs, labeled in the controls); DBSCAN `epsilon` and `minimumPoints` are written
 *   into a parameter buffer, so dragging them never recompiles, and the graphs run only when a
 *   value changed.
 * - Description graph: `GPUGroupGeometry` (bounds, mean and weighted centers, medoids, standard
 *   deviational ellipses) and `GPUGroupConvexHull` read the labels, and bespoke layers draw their
 *   outputs straight from the storage buffers.
 * - Only a small summary (counts, hull sizes, one cluster's centers and medoid) is read back.
 */
export const clustersMode: SpatialAnalysisModeDefinition = {
  id: 'clusters',
  title: 'Clusters',
  contributors: ['GPUSpatialClustering', 'GPUKMeans', 'GPUGroupGeometry', 'GPUGroupConvexHull'],
  description:
    'DBSCAN or k-means of taxi points on the GPU, then per-cluster convex hulls, standard ' +
    'ellipses, bounds, mean and weighted centers and medoids. Drag epsilon and the minimum ' +
    'neighborhood size, or switch method and k; toggle the shapes to compare them.',
  initialViewState: {longitude: -73.985, latitude: 40.722, zoom: 12.2},

  async create(context) {
    const {device} = context;
    const config: ClusterConfig = {
      pointSet: 'pickups',
      method: 'dbscan',
      k: KMEANS_DEFAULT_K,
      initialization: 'first-valid',
      seed: 0,
      denseBoxShortcut: false
    };
    let epsilon = POINT_SET_DEFAULTS[config.pointSet].epsilon;
    let minimumPoints = POINT_SET_DEFAULTS[config.pointSet].minimumPoints;
    let appliedEpsilon = Number.NaN;
    let appliedMinimumPoints = Number.NaN;
    let highlightCore = false;
    let showHulls = true;
    let showEllipses = true;
    let showBounds = false;
    let showMeans = true;
    let showWeighted = true;
    let showMedoids = true;
    let readbackPending = false;
    let readbackWanted = true;
    let destroyed = false;
    let loadToken = 0;
    let serial = 0;
    let encodedFrames = 0;
    let measuring = false;
    let measuredSerial = -1;
    let rebuildTimer: ReturnType<typeof setTimeout> | undefined;
    const retired: ClusterSet[] = [];
    const pointSetCache = new Map<PointSetId, Promise<PointSetData>>();

    const loadPointSet = (id: PointSetId): Promise<PointSetData> => {
      let cached = pointSetCache.get(id);
      if (!cached) {
        cached = loadPointSetUncached(id);
        pointSetCache.set(id, cached);
      }
      return cached;
    };
    const loadPointSetUncached = async (id: PointSetId): Promise<PointSetData> => {
      if (id === 'pois') {
        const pois = await context.data.getNewYorkPointsOfInterest();
        return {
          positions: pois.positions,
          origin: pois.origin,
          attribution: pois.attribution,
          label: 'points of interest'
        };
      }
      const trips = await context.data.getNewYorkTrips();
      if (id === 'vertices') {
        return {
          positions: trips.vertexPositions,
          origin: trips.origin,
          attribution: trips.attribution,
          label: 'trip vertices'
        };
      }
      // First and last vertex of every trip: its pickup and drop-off.
      const tripCount = trips.vendors.length;
      const positions = new Float32Array(tripCount * 4);
      for (let trip = 0; trip < tripCount; trip++) {
        const first = trips.tripOffsets[trip];
        const last = trips.tripOffsets[trip + 1] - 1;
        positions.set(trips.vertexPositions.subarray(first * 2, first * 2 + 2), trip * 4);
        positions.set(trips.vertexPositions.subarray(last * 2, last * 2 + 2), trip * 4 + 2);
      }
      return {
        positions,
        origin: trips.origin,
        attribution: trips.attribution,
        label: 'pickups and drop-offs'
      };
    };

    /** Creates the caller-owned DBSCAN buffers inside `resources`. */
    const createDbscanOutputs = (
      resources: SpatialAnalysisResources,
      pointCount: number
    ): DbscanOutputs => ({
      labels: resources.createBuffer('labels', pointCount * 4),
      coreFlags: resources.createBuffer('core-flags', pointCount * 4),
      clusterCount: resources.createBuffer('cluster-count', 4),
      clusterStoredCount: resources.createBuffer('cluster-stored-count', 4),
      clusterOverflow: resources.createBuffer('cluster-overflow', 4),
      clusterSizes: resources.createBuffer('cluster-sizes', CLUSTER_CAPACITY * 4),
      clusterCentroids: resources.createBuffer('cluster-centroids', CLUSTER_CAPACITY * 8)
    });

    /** Compiles one DBSCAN graph; the dense-box shortcut is the only compile-time difference. */
    const compileDbscan = (
      resources: SpatialAnalysisResources,
      id: string,
      positionsBuffer: Buffer,
      pointCount: number,
      parameters: GPUParameterBuffer<'float32'>,
      outputs: DbscanOutputs,
      denseBoxShortcut: boolean
    ): CompiledGPUCommandGraph<void> => {
      const graph = new GPUCommandGraph<void>(device, {id});
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
          parameters: parameters.importToGraph(graph),
          gridSize: GRID_SIZE,
          denseBoxShortcut,
          labels: importGraphBuffer(graph, 'labels', outputs.labels, 'uint32', pointCount),
          coreFlags: importGraphBuffer(
            graph,
            'core-flags',
            outputs.coreFlags,
            'uint32',
            pointCount
          ),
          clusterCount: importGraphBuffer(
            graph,
            'cluster-count',
            outputs.clusterCount,
            'uint32',
            1
          ),
          clusters: {
            ids: importGraphBuffer(graph, 'cluster-ids', clusterIds, 'uint32', CLUSTER_CAPACITY),
            count: importGraphBuffer(
              graph,
              'cluster-stored-count',
              outputs.clusterStoredCount,
              'uint32',
              1
            ),
            overflow: importGraphBuffer(
              graph,
              'cluster-overflow',
              outputs.clusterOverflow,
              'uint32',
              1
            )
          },
          clusterSizes: importGraphBuffer(
            graph,
            'cluster-sizes',
            outputs.clusterSizes,
            'uint32',
            CLUSTER_CAPACITY
          ),
          clusterCentroids: importGraphBuffer(
            graph,
            'cluster-centroids',
            outputs.clusterCentroids,
            'float32x2',
            CLUSTER_CAPACITY
          )
        })
      );
      return resources.track(graph.compile());
    };

    const buildSet = async (setConfig: ClusterConfig): Promise<ClusterSet> => {
      const data = await loadPointSet(setConfig.pointSet);
      context.signal.throwIfAborted();
      const pointCount = data.positions.length / 2;
      const setSerial = ++serial;
      const resources = new SpatialAnalysisResources(
        device,
        `clusters-${setConfig.pointSet}-${setSerial}`
      );
      let minX = Infinity;
      let minY = Infinity;
      let maxX = -Infinity;
      let maxY = -Infinity;
      for (let index = 0; index < pointCount; index++) {
        const x = data.positions[index * 2];
        const y = data.positions[index * 2 + 1];
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
      // Demo weights: a sawtooth in x (1 to 4 over WEIGHT_PERIOD_METERS), so within one
      // neighborhood the weighted center visibly separates from the plain mean center.
      const weightValues = new Float32Array(pointCount);
      for (let index = 0; index < pointCount; index++) {
        const phase = data.positions[index * 2] / WEIGHT_PERIOD_METERS;
        weightValues[index] = 1 + 3 * (phase - Math.floor(phase));
      }
      const positionsBuffer = resources.createBuffer('positions', data.positions);
      const weightsBuffer = resources.createBuffer('weights', weightValues);
      const parameters = resources.createParameterBuffer(
        'parameters',
        'float32',
        GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH
      );
      const geometryParameters = resources.createParameterBuffer(
        'geometry-parameters',
        'float32',
        GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH,
        getGPUGeographicDistributionParameterValues({
          origin: [0, 0],
          standardDeviations: 1,
          ellipseConvention: 'standard'
        })
      );

      // Clustering graph.
      const isDbscan = setConfig.method === 'dbscan';
      const groupCount = isDbscan ? CLUSTER_CAPACITY : setConfig.k;
      let labels: ClusterBuffer;
      let sizes: ClusterBuffer;
      let centroids: ClusterBuffer;
      let clusterGraph: CompiledGPUCommandGraph<void>;
      let dbscan: DbscanOutputs | null = null;
      if (isDbscan) {
        dbscan = createDbscanOutputs(resources, pointCount);
        ({labels, clusterSizes: sizes, clusterCentroids: centroids} = dbscan);
        clusterGraph = compileDbscan(
          resources,
          `clusters-dbscan-${setSerial}`,
          positionsBuffer,
          pointCount,
          parameters,
          dbscan,
          setConfig.denseBoxShortcut
        );
      } else {
        labels = resources.createBuffer('labels', pointCount * 4);
        sizes = resources.createBuffer('sizes', setConfig.k * 4);
        centroids = resources.createBuffer('centers', setConfig.k * 8);
        const graph = new GPUCommandGraph<void>(device, {id: `clusters-kmeans-${setSerial}`});
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
            k: setConfig.k,
            iterations: KMEANS_ITERATIONS,
            initialization: setConfig.initialization,
            seed: setConfig.seed,
            labels: importGraphBuffer(graph, 'labels', labels, 'uint32', pointCount),
            centers: importGraphBuffer(graph, 'centers', centroids, 'float32x2', setConfig.k),
            sizes: importGraphBuffer(graph, 'sizes', sizes, 'uint32', setConfig.k)
          })
        );
        clusterGraph = resources.track(graph.compile());
      }

      // Description graph: geometry and hulls per label.
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
      const shapes = new GPUCommandGraph<void>(device, {id: `clusters-shapes-${setSerial}`});
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
          weights: importGraphBuffer(shapes, 'weights', weightsBuffer, 'float32', pointCount),
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
      const shapeGraph = resources.track(shapes.compile());

      const drawCommands = resources.track(
        new DrawCommandBuffer(device, {
          id: `clusters-${setSerial}-draw`,
          type: 'draw',
          commands: [{vertexCount: 6, instanceCount: isDbscan ? 0 : setConfig.k}]
        })
      );
      // Summary: header words, then sizes, geometry counts, hull counts, mean and weighted centers
      // and medoids of every group.
      const summaryByteLength = SUMMARY_HEADER_WORDS * 4 + groupCount * (4 + 4 + 4 + 8 + 8 + 4);
      const readbackRing = resources.track(
        new GPUReadbackRing(device, {
          id: `clusters-${setSerial}-summary`,
          byteLength: summaryByteLength
        })
      );
      return {
        serial: setSerial,
        config: {...setConfig},
        data,
        pointCount,
        resources,
        bounds,
        parameters,
        clusterGraph,
        shapeGraph,
        groupCount,
        positionsBuffer,
        labels,
        coreFlags: dbscan?.coreFlags ?? null,
        clusterCount: dbscan?.clusterCount ?? null,
        clusterStoredCount: dbscan?.clusterStoredCount ?? null,
        clusterOverflow: dbscan?.clusterOverflow ?? null,
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
        drawCommands,
        readbackRing,
        summaryByteLength,
        dirty: true,
        destroyed: false
      };
    };

    const destroySet = (set: ClusterSet) => {
      set.destroyed = true;
      set.resources.destroy();
    };

    let active = await buildSet(config);
    context.signal.throwIfAborted();

    // --- Applying compile-time choices -----------------------------------------------------------
    const applyConfig = (resetParameters: boolean) => {
      const token = ++loadToken;
      context.setStatus(`Building ${config.method} on ${config.pointSet}…`);
      buildSet(config)
        .then(next => {
          if (destroyed || token !== loadToken) {
            destroySet(next);
            return;
          }
          const previous = active;
          active = next;
          appliedEpsilon = Number.NaN;
          appliedMinimumPoints = Number.NaN;
          encodedFrames = 0;
          // Keep the old buffers alive for a few frames: Deck may still hold them in layers.
          previous.destroyed = true;
          retired.push(previous);
          setTimeout(() => {
            if (retired.includes(previous)) {
              retired.splice(retired.indexOf(previous), 1);
              previous.resources.destroy();
            }
          }, 400);
          if (resetParameters) {
            epsilon = POINT_SET_DEFAULTS[config.pointSet].epsilon;
            minimumPoints = POINT_SET_DEFAULTS[config.pointSet].minimumPoints;
            epsilonControl.setValue(epsilon);
            minimumPointsControl.setValue(minimumPoints);
          }
          context.setStatus('');
          pointsReadout.setValue(`${formatCount(next.pointCount)} ${next.data.label}`);
          attributionReadout.setValue(next.data.attribution);
          updateControlAvailability();
          if (config.method !== 'dbscan') timingReadout.setValue('n/a for k-means');
          context.updateLayers();
        })
        .catch(error => {
          if (destroyed || context.signal.aborted) return;
          context.setStatus(`Failed to build ${config.method}: ${String(error)}`);
        });
    };
    const scheduleRebuild = () => {
      clearTimeout(rebuildTimer);
      rebuildTimer = setTimeout(() => {
        if (!destroyed) applyConfig(false);
      }, REBUILD_DEBOUNCE_MILLISECONDS);
    };

    // --- Controls ------------------------------------------------------------------------------
    context.controls.addSelect<PointSetId>({
      label: 'Point set (compile-time, rebuilds the graph)',
      options: POINT_SET_OPTIONS,
      value: config.pointSet,
      onChange: value => {
        config.pointSet = value;
        applyConfig(true);
      }
    });
    context.controls.addSelect<ClusterMethod>({
      label: 'Method (compile-time, rebuilds the graph)',
      options: METHOD_OPTIONS,
      value: config.method,
      onChange: value => {
        config.method = value;
        updateControlAvailability();
        scheduleRebuild();
      }
    });
    const epsilonControl = context.controls.addSlider({
      label: 'DBSCAN epsilon (neighbor radius, per-frame)',
      min: 20,
      max: 400,
      step: 5,
      value: epsilon,
      format: value => `${value} m`,
      onChange: value => {
        epsilon = value;
      }
    });
    const minimumPointsControl = context.controls.addSlider({
      label: 'DBSCAN minimum points (core threshold, per-frame)',
      min: 2,
      max: 50,
      step: 1,
      value: minimumPoints,
      format: value => `${value}`,
      onChange: value => {
        minimumPoints = value;
      }
    });
    const denseBoxControl = context.controls.addToggle({
      label: 'DBSCAN dense-box shortcut (compile-time, same labels)',
      value: config.denseBoxShortcut,
      onChange: value => {
        config.denseBoxShortcut = value;
        scheduleRebuild();
      }
    });
    context.controls.addButton({
      label: 'Time DBSCAN plain vs dense-box (outside frame)',
      onClick: () => void measureShortcut()
    });
    const kControl = context.controls.addSlider({
      label: 'K-means k (compile-time, rebuilds the graph)',
      min: 2,
      max: KMEANS_MAXIMUM_K,
      step: 1,
      value: config.k,
      format: value => `k = ${value}`,
      onChange: value => {
        config.k = value;
        scheduleRebuild();
      }
    });
    const initializationControl = context.controls.addSelect<GPUKMeansInitialization>({
      label: 'K-means initialization (compile-time)',
      options: INITIALIZATION_OPTIONS,
      value: config.initialization,
      onChange: value => {
        config.initialization = value;
        updateControlAvailability();
        scheduleRebuild();
      }
    });
    const seedControl = context.controls.addSlider({
      label: 'K-means++ seed (compile-time, rebuilds the graph)',
      min: 0,
      max: KMEANS_MAXIMUM_SEED,
      step: 1,
      value: config.seed,
      format: value => `seed ${value}`,
      onChange: value => {
        config.seed = value;
        scheduleRebuild();
      }
    });
    const updateControlAvailability = () => {
      const isDbscan = config.method === 'dbscan';
      epsilonControl.setDisabled(!isDbscan);
      minimumPointsControl.setDisabled(!isDbscan);
      denseBoxControl.setDisabled(!isDbscan);
      kControl.setDisabled(isDbscan);
      initializationControl.setDisabled(isDbscan);
      seedControl.setDisabled(isDbscan || config.initialization !== 'kmeans++');
    };
    updateControlAvailability();
    const addShapeToggle = (label: string, value: boolean, onChange: (value: boolean) => void) =>
      context.controls.addToggle({
        label,
        value,
        onChange: next => {
          onChange(next);
          context.updateLayers();
        }
      });
    addShapeToggle('Convex hulls (GPUGroupConvexHull)', showHulls, value => (showHulls = value));
    addShapeToggle(
      'Standard ellipses (GPUGroupGeometry, weighted)',
      showEllipses,
      value => (showEllipses = value)
    );
    addShapeToggle('Bounding boxes', showBounds, value => (showBounds = value));
    addShapeToggle('Mean centers (white rings)', showMeans, value => (showMeans = value));
    addShapeToggle(
      'Weighted centers (yellow diamonds)',
      showWeighted,
      value => (showWeighted = value)
    );
    addShapeToggle('Medoids (dark discs)', showMedoids, value => (showMedoids = value));
    context.controls.addToggle({
      label: 'Highlight DBSCAN core points',
      value: highlightCore,
      onChange: value => {
        highlightCore = value;
        context.updateLayers();
      }
    });
    context.controls.addLegend({
      title: 'Cluster colors (compact ID mod 8, discs are centroids)',
      entries: [
        ...CLUSTER_PALETTE.map((color, index) => ({
          color,
          label: index === 7 ? '7, 15, ...' : String(index)
        })),
        {color: NOISE_COLOR, label: 'noise'}
      ]
    });
    context.controls.addNote(
      'Hulls, ellipses and bounds take their cluster color. Weights are a demo sawtooth along x ' +
        '(1 to 4 every 250 m), so the weighted center (and the weighted ellipse) shifts away from the ' +
        'plain mean. Medoids are the member closest to all others; clusters over 1,024 members ' +
        'have none. DBSCAN labels follow the smallest core row of each cluster, so colors are ' +
        'stable while parameters change; k-means is deterministic per seed.'
    );
    const pointsReadout = context.controls.addReadout(
      'Points',
      `${formatCount(active.pointCount)} ${active.data.label}`
    );
    const parametersReadout = context.controls.addReadout('Method / parameters');
    const clustersReadout = context.controls.addReadout('Clusters');
    const clusteredReadout = context.controls.addReadout('Clustered / noise');
    const largestReadout = context.controls.addReadout('Largest cluster');
    const meanReadout = context.controls.addReadout('Mean cluster size');
    const geometryReadout = context.controls.addReadout('Group counts vs cluster sizes');
    const hullReadout = context.controls.addReadout('Hulls (largest, overflow)');
    const centerReadout = context.controls.addReadout('Largest cluster: weighted - mean');
    const medoidReadout = context.controls.addReadout('Largest cluster: medoid - mean');
    const timingReadout = context.controls.addReadout('DBSCAN graph timing');
    const attributionReadout = context.controls.addReadout('Data', active.data.attribution);

    // --- Summary readback ----------------------------------------------------------------------
    const readSummary = async (
      set: ClusterSet,
      commandEncoder: Parameters<SpatialAnalysisModeInstance['encode']>[0]
    ) => {
      const ticket = set.readbackRing.tryAcquire();
      if (!ticket) return;
      let offset = 0;
      const copy = (source: Buffer | null, size: number) => {
        if (source) {
          commandEncoder.copyBufferToBuffer({
            sourceBuffer: source,
            sourceOffset: 0,
            destinationBuffer: ticket.buffer,
            destinationOffset: offset,
            size
          });
        }
        offset += size;
      };
      const groups = set.groupCount;
      copy(set.clusterCount, 4);
      copy(set.clusterStoredCount, 4);
      copy(set.clusterOverflow, 4);
      copy(set.geometryOverflow, 4);
      copy(set.hullOverflow, 4);
      copy(set.sizes, groups * 4);
      copy(set.geometryCounts, groups * 4);
      copy(set.hullCounts, groups * 4);
      copy(set.meanCenters, groups * 8);
      copy(set.weightedCenters, groups * 8);
      copy(set.medoids, groups * 4);
      ticket.markEncoded({byteOffset: 0, byteLength: set.summaryByteLength});
      readbackPending = true;
      const echoedEpsilon = epsilon;
      const echoedMinimumPoints = minimumPoints;
      try {
        const bytes = await ticket.read();
        if (destroyed || set.destroyed) return;
        const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
        const words = new Uint32Array(buffer);
        const floats = new Float32Array(buffer);
        const isDbscan = set.config.method === 'dbscan';
        const clusterTotal = isDbscan ? words[0] : set.config.k;
        const stored = isDbscan ? Math.min(words[1], CLUSTER_CAPACITY) : set.config.k;
        const sizesStart = SUMMARY_HEADER_WORDS;
        const geometryStart = sizesStart + groups;
        const hullStart = geometryStart + groups;
        const meanStart = hullStart + groups;
        const weightedStart = meanStart + groups * 2;
        const medoidStart = weightedStart + groups * 2;
        let clustered = 0;
        let largest = 0;
        let largestCluster = -1;
        let geometryMatches = true;
        let hullTotal = 0;
        let hullLargest = 0;
        for (let cluster = 0; cluster < stored; cluster++) {
          const size = words[sizesStart + cluster];
          clustered += size;
          if (size > largest) {
            largest = size;
            largestCluster = cluster;
          }
          if (words[geometryStart + cluster] !== size) geometryMatches = false;
          const hullVertices = words[hullStart + cluster];
          if (hullVertices > 0) hullTotal++;
          hullLargest = Math.max(hullLargest, hullVertices);
        }
        const noise = set.pointCount - clustered;
        clustersReadout.setValue(
          `${formatCount(clusterTotal)}${isDbscan && words[2] ? ' (capacity overflow)' : ''}`
        );
        clusteredReadout.setValue(`${formatCount(clustered)} / ${formatCount(noise)}`);
        largestReadout.setValue(formatCount(largest));
        meanReadout.setValue(stored > 0 ? (clustered / stored).toFixed(1) : '–');
        parametersReadout.setValue(
          isDbscan
            ? `DBSCAN ${echoedEpsilon} m / ${echoedMinimumPoints}${
                set.config.denseBoxShortcut ? ' (dense-box)' : ''
              }`
            : `K-means k = ${set.config.k}, ${set.config.initialization}${
                set.config.initialization === 'kmeans++' ? ` seed ${set.config.seed}` : ''
              }, ${KMEANS_ITERATIONS} iterations`
        );
        geometryReadout.setValue(
          geometryMatches ? 'equal for every cluster' : 'MISMATCH (a group count differs)'
        );
        const hullOverflow = words[4];
        const hullFlags = [
          hullOverflow & GPU_GROUP_CONVEX_HULL_GROUP_OVERFLOW ? 'a hull over 256 vertices' : '',
          hullOverflow & GPU_GROUP_CONVEX_HULL_TOTAL_OVERFLOW ? 'total capacity' : ''
        ].filter(Boolean);
        hullReadout.setValue(
          `${formatCount(hullTotal)} hulls, largest ${hullLargest} vertices${
            hullFlags.length ? `, OVERFLOW: ${hullFlags.join(', ')}` : ''
          }`
        );
        if (largestCluster >= 0) {
          const meanX = floats[meanStart + largestCluster * 2];
          const meanY = floats[meanStart + largestCluster * 2 + 1];
          const weightedX = floats[weightedStart + largestCluster * 2];
          const weightedY = floats[weightedStart + largestCluster * 2 + 1];
          centerReadout.setValue(
            `cluster ${largestCluster}: ${(weightedX - meanX).toFixed(1)} m east, ${(
              weightedY - meanY
            ).toFixed(1)} m north`
          );
          const medoidRow = words[medoidStart + largestCluster];
          if (medoidRow === 0xffffffff) {
            medoidReadout.setValue(
              `cluster ${largestCluster}: none (over ${MEDOID_MAXIMUM_GROUP_SIZE} members${
                words[3] ? ', overflow flag set' : ''
              })`
            );
          } else {
            const medoidX = set.data.positions[medoidRow * 2];
            const medoidY = set.data.positions[medoidRow * 2 + 1];
            medoidReadout.setValue(
              `cluster ${largestCluster}: row ${formatCount(medoidRow)}, ${(
                medoidX - meanX
              ).toFixed(1)} m east, ${(medoidY - meanY).toFixed(1)} m north`
            );
          }
        } else {
          centerReadout.setValue('–');
          medoidReadout.setValue('–');
        }
      } catch {
        // The ring or device was destroyed while the read was in flight.
      } finally {
        readbackPending = false;
      }
    };

    /**
     * Times DBSCAN with and without the dense-box shortcut. Both variants are compiled into
     * scratch buffers, run between frames at the current parameters, and destroyed again.
     */
    const measureShortcut = async () => {
      const set = active;
      if (measuring || destroyed || set.config.method !== 'dbscan') {
        if (set.config.method !== 'dbscan') timingReadout.setValue('select DBSCAN to time it');
        return;
      }
      measuring = true;
      measuredSerial = set.serial;
      timingReadout.setValue('measuring...');
      const scratch = new SpatialAnalysisResources(device, `clusters-timing-${set.serial}`);
      try {
        set.parameters.write(
          getGPUSpatialClusteringParameterValues({bounds: set.bounds, epsilon, minimumPoints})
        );
        const outputs = createDbscanOutputs(scratch, set.pointCount);
        const results: {milliseconds: number; label: string}[] = [];
        for (const denseBoxShortcut of [false, true]) {
          const graph = compileDbscan(
            scratch,
            `clusters-timing-${denseBoxShortcut ? 'dense' : 'plain'}`,
            set.positionsBuffer,
            set.pointCount,
            set.parameters,
            outputs,
            denseBoxShortcut
          );
          const timing = await measureCompiledGraph(device, graph, {
            parameters: undefined,
            completionBuffer: outputs.clusterCount,
            signal: context.signal
          });
          results.push({
            milliseconds: timing.milliseconds,
            label: formatCompiledGraphTiming(timing).split(' · ')[0]
          });
        }
        if (destroyed || set.destroyed) return;
        const [plain, dense] = results;
        const ratio = plain.milliseconds / Math.max(dense.milliseconds, 1e-6);
        timingReadout.setValue(
          `plain ${plain.label}, dense-box ${dense.label} (${ratio.toFixed(2)}x) at ${epsilon} m / ${minimumPoints}`
        );
      } catch {
        // Interrupted by a rebuild or destroy; the next measurement replaces this one.
      } finally {
        scratch.destroy();
        measuring = false;
      }
    };

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [active.clusterGraph, active.shapeGraph],
      encode(commandEncoder, frame) {
        const set = active;
        encodedFrames++;
        if (set.config.method === 'dbscan') {
          if (epsilon !== appliedEpsilon || minimumPoints !== appliedMinimumPoints) {
            appliedEpsilon = epsilon;
            appliedMinimumPoints = minimumPoints;
            set.dirty = true;
          }
        }
        if (set.dirty) {
          set.dirty = false;
          if (set.config.method === 'dbscan') {
            set.parameters.write(
              getGPUSpatialClusteringParameterValues({
                bounds: set.bounds,
                epsilon,
                minimumPoints
              })
            );
          }
          set.clusterGraph.encode(commandEncoder, {parameters: undefined});
          if (set.clusterStoredCount) {
            // The contributor has no drawInstanceCount: copy the stored cluster count into the
            // indirect record's instance-count word (second uint32 of the 16-byte record).
            commandEncoder.copyBufferToBuffer({
              sourceBuffer: set.clusterStoredCount,
              sourceOffset: 0,
              destinationBuffer: set.drawCommands.buffer,
              destinationOffset: 4,
              size: 4
            });
          }
          set.shapeGraph.encode(commandEncoder, {parameters: undefined});
          readbackWanted = true;
        }
        if (
          (readbackWanted || frame.frameIndex % READBACK_INTERVAL_FRAMES === 0) &&
          !readbackPending
        ) {
          readbackWanted = false;
          void readSummary(set, commandEncoder);
        }
        if (
          set.config.method === 'dbscan' &&
          measuredSerial !== set.serial &&
          encodedFrames >= AUTO_MEASURE_FRAME
        ) {
          void measureShortcut();
        }
      },
      getLayers(): Layer[] {
        const set = active;
        const coordinateOrigin: [number, number, number] = [
          set.data.origin[0],
          set.data.origin[1],
          0
        ];
        const dense = set.pointCount > 10000;
        const radiusPixels = dense ? 1.8 : 3.2;
        const groupCount = set.groupCount;
        const id = (name: string) => `clusters-${name}-${set.serial}`;
        const layers: Layer[] = [];
        if (highlightCore && set.coreFlags) {
          layers.push(
            new SpatialAnalysisPointLayer({
              id: id('core'),
              coordinateOrigin,
              positions: set.positionsBuffer,
              instanceCount: set.pointCount,
              values: set.coreFlags,
              valueFormat: 'uint32',
              colormap: 'mask',
              color: [255, 255, 255, 120],
              noDataColor: [0, 0, 0, 0],
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
            noDataColor: NOISE_COLOR,
            radiusPixels
          })
        );
        if (showBounds) {
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
        if (showEllipses) {
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
        if (showHulls) {
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
        layers.push(
          new ClusterCentroidLayer({
            id: id('centroids'),
            coordinateOrigin,
            centroids: set.centroids,
            sizes: set.sizes,
            drawCommands: set.drawCommands,
            palette: CLUSTER_PALETTE
          })
        );
        const markerProps = {
          coordinateOrigin,
          counts: set.geometryCounts,
          centers: set.meanCenters,
          medoidRows: set.medoids,
          positions: set.positionsBuffer,
          groupCount
        };
        if (showMeans) {
          layers.push(
            new GroupMarkerLayer({
              ...markerProps,
              id: id('mean-centers'),
              source: 'centers',
              shape: 'ring',
              sizePixels: 7,
              palette: [MEAN_MARKER_COLOR]
            })
          );
        }
        if (showWeighted) {
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
        if (showMedoids) {
          layers.push(
            new GroupMarkerLayer({
              ...markerProps,
              id: id('medoids'),
              source: 'medoids',
              shape: 'disc',
              sizePixels: 3.5,
              palette: [MEDOID_MARKER_COLOR]
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
    return instance;
  }
};
