// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Isochrones over the New York road network. One compiled graph converts the directed edge list to
 * CSR on the GPU (`GPUCOOToCSR`) and runs `GPUNetworkReachability` from a clicked node. The source
 * node, the time budget and the edge travel times (walk vs drive) are buffer contents, so changing
 * any of them rewrites a buffer and re-encodes the same compiled graph.
 */

import type {Layer} from '@deck.gl/core';
import {
  createTransientView,
  GPUCommandGraph,
  GPUCOOToCSR,
  GPUReadbackRing,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUNetworkReachability,
  GPU_NETWORK_REACHABILITY_NONE
} from '@luma.gl/experimental/gpu-network';
import {importGraphBuffer} from '../graph-buffers';
import {LocalMetricProjection} from '../spatial-analysis-data';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {
  findNearestNode,
  sortEdgesBySource,
  writeEdgeTravelSeconds,
  type Transport
} from './road-network-utils';

/**
 * Compile-time relaxation round limit. Each round is one graph node covering up to 16 hops
 * (`localIterations`), so Manhattan's longest drive (about 380 hops) needs about 25 rounds; the
 * graph stops early on the GPU once costs converge and reports `converged` honestly otherwise.
 */
const MAXIMUM_ITERATIONS = 64;
const BAND_COUNT = 4;
const BAND_FRACTIONS = [0.25, 0.5, 0.75, 1] as const;
const READBACK_INTERVAL_FRAMES = 15;
/** Times Square. */
const DEFAULT_SOURCE: readonly [number, number] = [-73.9855, 40.758];
const BAND_COLORS = [
  [255, 70, 50, 255],
  [255, 170, 0, 255],
  [70, 205, 150, 255],
  [70, 120, 255, 255]
] as const;
const UNREACHED_COLOR = [90, 92, 105, 110] as const;

type View = 'bands' | 'continuous';

export const reachabilityMode: SpatialAnalysisModeDefinition = {
  id: 'reachability',
  title: 'Reachability',
  contributors: ['GPUNetworkReachability', 'GPUCOOToCSR'],
  description:
    'Travel-time isochrones on the New York street graph, computed on the GPU. Click the map to ' +
    'move the source; switch walk/drive and the time budget without recompiling.',
  initialViewState: {longitude: -73.985, latitude: 40.755, zoom: 13.3},

  async create(context) {
    const roads = await context.data.getNewYorkRoads();
    context.signal.throwIfAborted();
    const {device} = context;
    const projection = new LocalMetricProjection(roads.origin);
    const resources = new SpatialAnalysisResources(device, 'reachability');
    const nodeCount = roads.nodePositions.length / 2;
    const segmentCount = roads.segmentNodes.length;
    const edges = sortEdgesBySource(roads);
    const edgeCount = edges.sources.length;
    const weights = new Float32Array(edgeCount);

    let transport: Transport = 'walk';
    let budgetMinutes = 15;
    let view: View = 'bands';
    let sourceNode = findNearestNode(roads.nodePositions, projection.project(...DEFAULT_SOURCE));
    let dirty = true;
    let destroyed = false;
    let readbackPending = false;

    const costLimitSeconds = () => budgetMinutes * 60;
    const writeWeights = () => {
      writeEdgeTravelSeconds(edges, transport, weights);
      weightsBuffer.write(weights);
    };

    const cooRows = resources.createBuffer('coo-rows', edges.sources);
    const cooColumns = resources.createBuffer('coo-columns', edges.targets);
    // Travel time per COO edge in seconds. Rewritten when the transport mode changes.
    const weightsBuffer = resources.createBuffer('coo-weights', weights);
    const costs = resources.createBuffer('costs', nodeCount * 4);
    const bands = resources.createBuffer('bands', nodeCount * 4);
    const bandCounts = resources.createBuffer('band-counts', BAND_COUNT * 4);
    const converged = resources.createBuffer('converged', 4);
    const iterationCount = resources.createBuffer('iteration-count', 4);
    const segmentsBuffer = resources.createBuffer('segments', roads.segments);
    const segmentNodes = resources.createBuffer('segment-nodes', roads.segmentNodes);
    const sourcePosition = resources.createBuffer('source-position', 8);
    const sources = resources.createParameterBuffer(
      'sources',
      'uint32',
      1,
      Uint32Array.of(sourceNode)
    );
    const costLimit = resources.createParameterBuffer('cost-limit', 'float32', 1);
    const bandThresholds = resources.createParameterBuffer(
      'band-thresholds',
      'float32',
      BAND_COUNT
    );
    const readbackRing = resources.track(
      new GPUReadbackRing(device, {id: 'reachability-summary', byteLength: (2 + BAND_COUNT) * 4})
    );

    const graph = new GPUCommandGraph<void>(device, {id: 'reachability'});
    const csrOffsets = createTransientView(graph, 'csr-offsets', 'uint32', nodeCount + 1);
    const csrNeighbors = createTransientView(graph, 'csr-neighbors', 'uint32', edgeCount);
    const csrWeights = createTransientView(graph, 'csr-weights', 'float32', edgeCount);
    graph.add(
      new GPUCOOToCSR({
        id: 'reachability-csr',
        rows: nodeCount,
        rowIndices: importGraphBuffer(graph, 'coo-rows', cooRows, 'uint32', edgeCount),
        columnIndices: importGraphBuffer(graph, 'coo-columns', cooColumns, 'uint32', edgeCount),
        values: importGraphBuffer(graph, 'coo-weights', weightsBuffer, 'float32', edgeCount),
        rowOffsets: csrOffsets,
        outputColumnIndices: csrNeighbors,
        outputValues: csrWeights
      })
    );
    graph.add(
      new GPUNetworkReachability({
        id: 'reachability',
        offsets: csrOffsets,
        neighbors: csrNeighbors,
        weights: csrWeights,
        sources: sources.importToGraph(graph),
        costLimit: costLimit.importToGraph(graph),
        bandThresholds: bandThresholds.importToGraph(graph),
        maxIterations: MAXIMUM_ITERATIONS,
        costs: importGraphBuffer(graph, 'costs', costs, 'float32', nodeCount),
        bands: importGraphBuffer(graph, 'bands', bands, 'uint32', nodeCount),
        bandCounts: importGraphBuffer(graph, 'band-counts', bandCounts, 'uint32', BAND_COUNT),
        converged: importGraphBuffer(graph, 'converged', converged, 'uint32', 1),
        iterationCount: importGraphBuffer(graph, 'iteration-count', iterationCount, 'uint32', 1)
      })
    );
    const compiled: CompiledGPUCommandGraph<void> = resources.track(graph.compile());

    const sourceReadout = context.controls.addReadout('Source node');
    const writeSource = (node: number) => {
      sourceNode = node;
      sources.write(Uint32Array.of(node));
      sourcePosition.write(roads.nodePositions.subarray(node * 2, node * 2 + 2));
      sourceReadout.setValue(`#${formatCount(node)}`);
      dirty = true;
    };
    const writeBudget = () => {
      const limit = costLimitSeconds();
      costLimit.write(Float32Array.of(limit));
      bandThresholds.write(Float32Array.from(BAND_FRACTIONS, fraction => fraction * limit));
      dirty = true;
    };

    context.controls.addSelect<Transport>({
      label: 'Transport (rewrites edge weights)',
      options: [
        {value: 'walk', label: 'Walk (1.4 m/s)'},
        {value: 'drive', label: 'Drive (7-20 m/s by road class)'}
      ],
      value: transport,
      onChange: value => {
        transport = value;
        writeWeights();
        dirty = true;
      }
    });
    context.controls.addSlider({
      label: 'Time budget (per-frame cost limit)',
      min: 2,
      max: 30,
      step: 1,
      value: budgetMinutes,
      format: value => `${value} min`,
      onChange: value => {
        budgetMinutes = value;
        writeBudget();
        context.updateLayers();
      }
    });
    context.controls.addSelect<View>({
      label: 'Color by',
      options: [
        {value: 'bands', label: 'Isochrone bands'},
        {value: 'continuous', label: 'Continuous travel time'}
      ],
      value: view,
      onChange: value => {
        view = value;
        context.updateLayers();
      }
    });
    context.controls.addLegend({
      title: 'Travel time bands (share of the budget)',
      entries: [
        ...BAND_COLORS.map((color, band) => ({
          color,
          label: `${band === 0 ? '0' : BAND_FRACTIONS[band - 1] * 100}-${BAND_FRACTIONS[band] * 100}%`
        })),
        {color: UNREACHED_COLOR, label: 'Not reached'}
      ]
    });
    context.controls.addNote('Click the map to move the source to the nearest intersection.');
    context.controls.addReadout('Nodes', formatCount(nodeCount));
    context.controls.addReadout('Directed edges', formatCount(edgeCount));
    const convergedReadout = context.controls.addReadout('Converged');
    const bandReadout = context.controls.addReadout('Nodes per band');
    context.controls.addReadout('Data', roads.attribution);

    writeWeights();
    writeBudget();
    writeSource(sourceNode);

    const readSummary = async (
      commandEncoder: Parameters<SpatialAnalysisModeInstance['encode']>[0]
    ) => {
      const ticket = readbackRing.tryAcquire();
      if (!ticket) return;
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: converged,
        destinationBuffer: ticket.buffer,
        destinationOffset: 0,
        size: 4
      });
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: iterationCount,
        destinationBuffer: ticket.buffer,
        destinationOffset: 4,
        size: 4
      });
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: bandCounts,
        destinationBuffer: ticket.buffer,
        destinationOffset: 8,
        size: BAND_COUNT * 4
      });
      ticket.markEncoded({byteOffset: 0, byteLength: (2 + BAND_COUNT) * 4});
      readbackPending = true;
      try {
        const bytes = await ticket.read();
        if (destroyed) return;
        const words = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
        convergedReadout.setValue(
          `${words[0] ? 'yes' : 'no (iteration limit)'} after ${words[1]} of ${MAXIMUM_ITERATIONS} rounds`
        );
        bandReadout.setValue(Array.from(words.subarray(2), formatCount).join(' / '));
      } catch {
        // The ring or device was destroyed while the read was in flight.
      } finally {
        readbackPending = false;
      }
    };

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [compiled],
      encode(commandEncoder, frame) {
        // Costs persist in their buffers, so the graph only re-encodes when an input changed.
        if (dirty || frame.frameIndex < 2) {
          compiled.encode(commandEncoder, {parameters: undefined});
          dirty = false;
        }
        if (frame.frameIndex % READBACK_INTERVAL_FRAMES === 1 && !readbackPending) {
          void readSummary(commandEncoder);
        }
      },
      getLayers() {
        const coordinateOrigin: [number, number, number] = [roads.origin[0], roads.origin[1], 0];
        const continuous = view === 'continuous';
        const layers: Layer[] = [
          new SpatialAnalysisSegmentLayer({
            id: 'reachability-roads',
            coordinateOrigin,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            widthPixels: 2.5,
            values: continuous ? costs : bands,
            valueFormat: continuous ? 'float32' : 'uint32',
            valueIndices: segmentNodes,
            colormap: continuous ? 'viridis' : 'category',
            valueRange: [0, costLimitSeconds()],
            palette: BAND_COLORS,
            noDataValue: GPU_NETWORK_REACHABILITY_NONE,
            noDataColor: UNREACHED_COLOR
          }),
          new SpatialAnalysisPointLayer({
            id: 'reachability-source',
            coordinateOrigin,
            positions: sourcePosition,
            instanceCount: 1,
            radiusPixels: 7,
            color: [255, 255, 255, 255]
          })
        ];
        return layers;
      },
      onClick(event) {
        if (!event.coordinate) return false;
        const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
        writeSource(findNearestNode(roads.nodePositions, [x, y]));
        return true;
      },
      destroy() {
        destroyed = true;
        resources.destroy();
      }
    };
    return instance;
  }
};
