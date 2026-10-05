// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  GPUReadbackRing,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {getGPUSpatialClusteringParameterValues, GPU_SPATIAL_CLUSTERING_NOISE, GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH, GPUSpatialClustering} from '@luma.gl/experimental/geospatial';
import {importGraphBuffer} from '@luma.gl/experimental/UNRESOLVED';
import {type GPUParameterBuffer} from '@luma.gl/experimental/geospatial';
import {MapGraphsPointLayer} from '../map-graphs-layers';
import type {MapGraphsModeDefinition, MapGraphsModeInstance} from '../map-graphs-mode';
import {formatCount, MapGraphsResources} from '../map-graphs-resources';
import {ClusterCentroidLayer} from './clusters-layers';

/** Maximum `[columns, rows]` of the neighbor-search lattice (compile-time). */
const GRID_SIZE: readonly [number, number] = [256, 256];
/** Compile-time cluster capacity; clusters beyond it are labeled but have no size or centroid. */
const CLUSTER_CAPACITY = 2048;
const READBACK_INTERVAL_FRAMES = 12;
/** Summary ring layout in bytes: cluster count, stored count, overflow flag, then the sizes. */
const SUMMARY_HEADER_BYTES = 12;
const SUMMARY_BYTE_LENGTH = SUMMARY_HEADER_BYTES + CLUSTER_CAPACITY * 4;
/** Padding added around the data extent so every point is inside the inclusive bounds. */
const BOUNDS_PADDING_METERS = 10;

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

type PointSetId = 'pickups' | 'pois' | 'vertices';

const POINT_SET_OPTIONS: readonly {value: PointSetId; label: string}[] = [
  {value: 'pickups', label: 'Trip pickups & drop-offs'},
  {value: 'pois', label: 'Points of interest'},
  {value: 'vertices', label: 'All trip vertices'}
];

type PointSetData = {
  positions: Float32Array;
  origin: readonly [number, number];
  attribution: string;
  label: string;
};

/** Everything that depends on the compile-time point set; destroyed as one unit. */
type ClusterSet = {
  pointSet: PointSetId;
  data: PointSetData;
  pointCount: number;
  resources: MapGraphsResources;
  bounds: [number, number, number, number];
  parameters: GPUParameterBuffer<'float32'>;
  compiled: CompiledGPUCommandGraph<void>;
  labels: ReturnType<MapGraphsResources['createBuffer']>;
  coreFlags: ReturnType<MapGraphsResources['createBuffer']>;
  clusterCount: ReturnType<MapGraphsResources['createBuffer']>;
  clusterOverflow: ReturnType<MapGraphsResources['createBuffer']>;
  clusterStoredCount: ReturnType<MapGraphsResources['createBuffer']>;
  clusterSizes: ReturnType<MapGraphsResources['createBuffer']>;
  clusterCentroids: ReturnType<MapGraphsResources['createBuffer']>;
  positionsBuffer: ReturnType<MapGraphsResources['createBuffer']>;
  drawCommands: DrawCommandBuffer;
  readbackRing: GPUReadbackRing;
  destroyed: boolean;
};

/**
 * DBSCAN density clustering of New York points on the GPU. The point set is a compile-time choice
 * (it rebuilds the graph); the neighbor radius `epsilon` (meters) and `minimumPoints` are written
 * into a parameter buffer every frame, so dragging them never recompiles. Points are colored by
 * their GPU cluster label (noise grey), cluster centroids and sizes come straight from the recipe
 * outputs, and the number of centroid discs is copied GPU to GPU into an indirect draw record.
 * Only a small summary (counts and the per-cluster sizes) is read back for the panel.
 */
export const clustersMode: MapGraphsModeDefinition = {
  id: 'clusters',
  title: 'Clusters',
  recipes: ['GPUSpatialClustering'],
  description:
    'DBSCAN of taxi pickups and drop-offs on the GPU. Drag epsilon and the minimum neighborhood ' +
    'size: labels, centroids and sizes are recomputed every frame from a parameter buffer.',
  initialViewState: {longitude: -73.985, latitude: 40.722, zoom: 12.2},

  async create(context) {
    const {device} = context;
    let pointSetId: PointSetId = 'pickups';
    let epsilon = POINT_SET_DEFAULTS[pointSetId].epsilon;
    let minimumPoints = POINT_SET_DEFAULTS[pointSetId].minimumPoints;
    let highlightCore = false;
    let readbackPending = false;
    let destroyed = false;
    let loadToken = 0;
    const retired: ClusterSet[] = [];

    const loadPointSet = async (id: PointSetId): Promise<PointSetData> => {
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

    const buildSet = async (id: PointSetId): Promise<ClusterSet> => {
      const data = await loadPointSet(id);
      context.signal.throwIfAborted();
      const pointCount = data.positions.length / 2;
      const resources = new MapGraphsResources(device, `clusters-${id}`);
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
      const positionsBuffer = resources.createBuffer('positions', data.positions);
      const parameters = resources.createParameterBuffer(
        'parameters',
        'float32',
        GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH
      );
      const labels = resources.createBuffer('labels', pointCount * 4);
      const coreFlags = resources.createBuffer('core-flags', pointCount * 4);
      const clusterCount = resources.createBuffer('cluster-count', 4);
      const clusterIds = resources.createBuffer('cluster-ids', CLUSTER_CAPACITY * 4);
      const clusterStoredCount = resources.createBuffer('cluster-stored-count', 4);
      const clusterOverflow = resources.createBuffer('cluster-overflow', 4);
      const clusterSizes = resources.createBuffer('cluster-sizes', CLUSTER_CAPACITY * 4);
      const clusterCentroids = resources.createBuffer('cluster-centroids', CLUSTER_CAPACITY * 8);
      const drawCommands = resources.track(
        new DrawCommandBuffer(device, {
          id: `clusters-${id}-draw`,
          type: 'draw',
          commands: [{vertexCount: 6, instanceCount: 0}]
        })
      );
      const readbackRing = resources.track(
        new GPUReadbackRing(device, {id: `clusters-${id}-summary`, byteLength: SUMMARY_BYTE_LENGTH})
      );

      const graph = new GPUCommandGraph<void>(device, {id: `clusters-${id}`});
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
            clusterSizes,
            'uint32',
            CLUSTER_CAPACITY
          ),
          clusterCentroids: importGraphBuffer(
            graph,
            'cluster-centroids',
            clusterCentroids,
            'float32x2',
            CLUSTER_CAPACITY
          )
        })
      );
      const compiled = resources.track(graph.compile());
      return {
        pointSet: id,
        data,
        pointCount,
        resources,
        bounds,
        parameters,
        compiled,
        labels,
        coreFlags,
        clusterCount,
        clusterOverflow,
        clusterStoredCount,
        clusterSizes,
        clusterCentroids,
        positionsBuffer,
        drawCommands,
        readbackRing,
        destroyed: false
      };
    };

    const destroySet = (set: ClusterSet) => {
      set.destroyed = true;
      set.resources.destroy();
    };

    let active = await buildSet(pointSetId);
    context.signal.throwIfAborted();

    // --- Controls ------------------------------------------------------------------------------
    context.controls.addSelect<PointSetId>({
      label: 'Point set (compile-time, rebuilds the graph)',
      options: POINT_SET_OPTIONS,
      value: pointSetId,
      onChange: value => {
        pointSetId = value;
        const token = ++loadToken;
        context.setStatus(`Loading ${value}…`);
        buildSet(value)
          .then(next => {
            if (destroyed || token !== loadToken) {
              destroySet(next);
              return;
            }
            const previous = active;
            active = next;
            // Keep the old buffers alive for a few frames: Deck may still hold them in layers.
            previous.destroyed = true;
            retired.push(previous);
            setTimeout(() => {
              if (retired.includes(previous)) {
                retired.splice(retired.indexOf(previous), 1);
                previous.resources.destroy();
              }
            }, 400);
            epsilon = POINT_SET_DEFAULTS[value].epsilon;
            minimumPoints = POINT_SET_DEFAULTS[value].minimumPoints;
            epsilonControl.setValue(epsilon);
            minimumPointsControl.setValue(minimumPoints);
            context.setStatus('');
            pointsReadout.setValue(`${formatCount(next.pointCount)} ${next.data.label}`);
            attributionReadout.setValue(next.data.attribution);
            context.updateLayers();
          })
          .catch(error => {
            if (destroyed || context.signal.aborted) return;
            context.setStatus(`Failed to load ${value}: ${String(error)}`);
          });
      }
    });
    const epsilonControl = context.controls.addSlider({
      label: 'Epsilon (neighbor radius, per-frame)',
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
      label: 'Minimum points (core threshold, per-frame)',
      min: 2,
      max: 50,
      step: 1,
      value: minimumPoints,
      format: value => `${value}`,
      onChange: value => {
        minimumPoints = value;
      }
    });
    context.controls.addToggle({
      label: 'Highlight core points',
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
      'Disc radius grows with the square root of the cluster size. Labels follow the smallest core ' +
        'row of each cluster, so colors are stable while parameters change.'
    );
    const pointsReadout = context.controls.addReadout(
      'Points',
      `${formatCount(active.pointCount)} ${active.data.label}`
    );
    const parametersReadout = context.controls.addReadout('Epsilon / min points');
    const clustersReadout = context.controls.addReadout('Clusters');
    const clusteredReadout = context.controls.addReadout('Clustered / noise');
    const largestReadout = context.controls.addReadout('Largest cluster');
    const meanReadout = context.controls.addReadout('Mean cluster size');
    const attributionReadout = context.controls.addReadout('Data', active.data.attribution);

    // --- Summary readback ----------------------------------------------------------------------
    const readSummary = async (
      set: ClusterSet,
      commandEncoder: Parameters<MapGraphsModeInstance['encode']>[0]
    ) => {
      const ticket = set.readbackRing.tryAcquire();
      if (!ticket) return;
      const copy = (source: typeof set.clusterCount, offset: number, size: number) =>
        commandEncoder.copyBufferToBuffer({
          sourceBuffer: source,
          sourceOffset: 0,
          destinationBuffer: ticket.buffer,
          destinationOffset: offset,
          size
        });
      copy(set.clusterCount, 0, 4);
      copy(set.clusterStoredCount, 4, 4);
      copy(set.clusterOverflow, 8, 4);
      copy(set.clusterSizes, SUMMARY_HEADER_BYTES, CLUSTER_CAPACITY * 4);
      ticket.markEncoded({byteOffset: 0, byteLength: SUMMARY_BYTE_LENGTH});
      readbackPending = true;
      const echoedEpsilon = epsilon;
      const echoedMinimumPoints = minimumPoints;
      try {
        const bytes = await ticket.read();
        if (destroyed || set.destroyed) return;
        const words = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
        const clusterTotal = words[0];
        const stored = Math.min(words[1], CLUSTER_CAPACITY);
        let clustered = 0;
        let largest = 0;
        for (let cluster = 0; cluster < stored; cluster++) {
          const size = words[3 + cluster];
          clustered += size;
          largest = Math.max(largest, size);
        }
        const noise = set.pointCount - clustered;
        clustersReadout.setValue(
          `${formatCount(clusterTotal)}${words[2] ? ' (capacity overflow)' : ''}`
        );
        clusteredReadout.setValue(`${formatCount(clustered)} / ${formatCount(noise)}`);
        largestReadout.setValue(formatCount(largest));
        meanReadout.setValue(stored > 0 ? (clustered / stored).toFixed(1) : '–');
        parametersReadout.setValue(`${echoedEpsilon} m / ${echoedMinimumPoints}`);
      } catch {
        // The ring or device was destroyed while the read was in flight.
      } finally {
        readbackPending = false;
      }
    };

    const instance: MapGraphsModeInstance = {
      getCompiledGraphs: () => [active.compiled],
      encode(commandEncoder, frame) {
        const set = active;
        set.parameters.write(
          getGPUSpatialClusteringParameterValues({
            bounds: set.bounds,
            epsilon,
            minimumPoints
          })
        );
        set.compiled.encode(commandEncoder, {parameters: undefined});
        // The recipe has no drawInstanceCount: copy the stored cluster count into the indirect
        // record's instance-count word (second uint32 of the 16-byte record).
        commandEncoder.copyBufferToBuffer({
          sourceBuffer: set.clusterStoredCount,
          sourceOffset: 0,
          destinationBuffer: set.drawCommands.buffer,
          destinationOffset: 4,
          size: 4
        });
        if (frame.frameIndex % READBACK_INTERVAL_FRAMES === 0 && !readbackPending) {
          void readSummary(set, commandEncoder);
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
        const layers: Layer[] = [];
        if (highlightCore) {
          layers.push(
            new MapGraphsPointLayer({
              id: `clusters-core-${set.pointSet}`,
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
          new MapGraphsPointLayer({
            id: `clusters-points-${set.pointSet}`,
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
          }),
          new ClusterCentroidLayer({
            id: `clusters-centroids-${set.pointSet}`,
            coordinateOrigin,
            centroids: set.clusterCentroids,
            sizes: set.clusterSizes,
            drawCommands: set.drawCommands,
            palette: CLUSTER_PALETTE
          })
        );
        return layers;
      },
      destroy() {
        destroyed = true;
        loadToken++;
        destroySet(active);
        for (const set of retired.splice(0)) set.resources.destroy();
      }
    };
    return instance;
  }
};
