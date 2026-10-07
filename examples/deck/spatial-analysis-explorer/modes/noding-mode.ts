// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Noding: the New York road polylines become a routable planar network on the GPU.
 * `GPUNetworkNoding` (on `GPULineSplit` and `GPUSegmentIntersection`) splits the lines at every
 * intersection, snaps piece end points within a per-frame tolerance, numbers the nodes and emits
 * the edge list plus an undirected CSR. `GPUNetworkReachability` then routes over that CSR from a
 * clicked node, which proves the result is a connected, weighted network. The tolerance and the
 * travel budget are buffer writes; the graph compiles once.
 */

import type {Buffer} from '@luma.gl/core';
import type {Layer} from '@deck.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUNetworkReachability} from '@luma.gl/experimental/gpu-network';
import {GPUNetworkNoding} from '@luma.gl/experimental/gpu-network';
import {importGraphBuffer} from '../graph-buffers';
import {LocalMetricProjection} from '../spatial-analysis-data';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {addKernelPass} from './mode-kernels';
import {addNodingDisplayPasses, NODING_HIDDEN} from './noding-layers';
import {SummaryReader} from './summary-reader';

/** Times Square; the reachability source starts at the nearest node. */
const DEFAULT_SOURCE: readonly [number, number] = [-73.9855, 40.758];
const MAXIMUM_ITERATIONS = 48;
const LOCAL_ITERATIONS = 16;
const NODE_DEGREE_PALETTE = [
  [90, 92, 105, 0],
  [255, 70, 50, 255],
  [150, 150, 165, 255],
  [255, 200, 40, 255],
  [70, 205, 150, 255],
  [70, 150, 255, 255],
  [190, 120, 255, 255],
  [255, 255, 255, 255]
] as const;
const PIECE_PALETTE = [
  [78, 201, 255, 255],
  [255, 148, 72, 255],
  [189, 122, 255, 255],
  [87, 235, 168, 255],
  [255, 105, 168, 255],
  [245, 220, 87, 255],
  [107, 158, 255, 255],
  [255, 92, 92, 255]
] as const;
const UNREACHED_COLOR = [90, 92, 105, 120] as const;

type View = 'pieces' | 'cost';

/** Chains consecutive road segments that continue one another into polylines. */
function buildLines(segments: Float32Array, classes: Uint32Array) {
  const positions: number[] = [];
  const lineOffsets = [0];
  const segmentCount = segments.length / 4;
  for (let segment = 0; segment < segmentCount; segment++) {
    const x0 = segments[segment * 4];
    const y0 = segments[segment * 4 + 1];
    const continues =
      segment > 0 &&
      classes[segment] === classes[segment - 1] &&
      segments[(segment - 1) * 4 + 2] === x0 &&
      segments[(segment - 1) * 4 + 3] === y0;
    if (!continues) {
      if (segment > 0) lineOffsets.push(positions.length / 2);
      positions.push(x0, y0);
    }
    positions.push(segments[segment * 4 + 2], segments[segment * 4 + 3]);
  }
  lineOffsets.push(positions.length / 2);
  return {positions: Float32Array.from(positions), lineOffsets: Uint32Array.from(lineOffsets)};
}

export const nodingMode: SpatialAnalysisModeDefinition = {
  id: 'noding',
  title: 'Noding',
  contributors: [
    'GPUNetworkNoding',
    'GPULineSplit',
    'GPUSegmentIntersection',
    'GPUNetworkReachability'
  ],
  description:
    'Road polylines split at every junction into edges (pieces coloured), with nodes drawn by ' +
    'degree (red = dead end). Raise the snap tolerance to merge near-coincident end points, and ' +
    'click the map to route from a node over the freshly built network.',
  initialViewState: {longitude: -73.985, latitude: 40.755, zoom: 13.3},

  async create(context) {
    const roads = await context.data.getNewYorkRoads();
    context.signal.throwIfAborted();
    const {device} = context;
    const projection = new LocalMetricProjection(roads.origin);
    const resources = new SpatialAnalysisResources(device, 'noding');
    const lines = buildLines(roads.segments, roads.segmentClasses);
    const segmentCount = roads.segments.length / 4;
    const lineCount = lines.lineOffsets.length - 1;
    const vertexTotal = lines.positions.length / 2;

    // Capacities: junction touches bring about two pairs per incident segment, and a piece spans
    // at least one segment. Overflow is reported, never silent.
    const intersectionCapacity = Math.max(4096, segmentCount * 3);
    const pieceCapacity = segmentCount + 1024;
    const vertexCapacity = vertexTotal + pieceCapacity * 2;
    const nodeCapacity = vertexTotal + 1024;

    let toleranceMeters = 0;
    let budgetMeters = 1500;
    let view: View = 'pieces';
    let dirty = true;
    let destroyed = false;

    const linePositions = resources.createBuffer('line-positions', lines.positions);
    const lineOffsetsBuffer = resources.createBuffer('line-offsets', lines.lineOffsets);
    const tolerance = resources.createParameterBuffer('tolerance', 'float32', 1);
    const costLimit = resources.createParameterBuffer('cost-limit', 'float32', 1);
    const clickPoint = resources.createParameterBuffer('click-point', 'float32', 2);
    const pieceLines = resources.createBuffer('piece-lines', pieceCapacity * 4);
    const pieceOffsets = resources.createBuffer('piece-offsets', (pieceCapacity + 1) * 4);
    const piecePositions = resources.createBuffer('piece-positions', vertexCapacity * 8);
    const pieceCount = resources.createBuffer('piece-count', 4);
    const pieceTotal = resources.createBuffer('piece-total', 4);
    const nodePositions = resources.createBuffer('node-positions', nodeCapacity * 8);
    const nodeCount = resources.createBuffer('node-count', 4);
    const nodeTotal = resources.createBuffer('node-total', 4);
    const fromNodes = resources.createBuffer('from-nodes', pieceCapacity * 4);
    const toNodes = resources.createBuffer('to-nodes', pieceCapacity * 4);
    const lengths = resources.createBuffer('lengths', pieceCapacity * 4);
    const csrOffsets = resources.createBuffer('csr-offsets', (nodeCapacity + 1) * 4);
    const csrNeighbors = resources.createBuffer('csr-neighbors', pieceCapacity * 8);
    const csrWeights = resources.createBuffer('csr-weights', pieceCapacity * 8);
    const overflow = resources.createBuffer('overflow', 4);
    const costs = resources.createBuffer('costs', nodeCapacity * 4);
    const converged = resources.createBuffer('converged', 4);
    const source = resources.createBuffer('source', 4);
    const sourcePosition = resources.createBuffer('source-position', 8);
    const segmentRows = resources.createBuffer('segment-rows', vertexCapacity * 16);
    const rowPieces = resources.createBuffer('row-pieces', vertexCapacity * 4);
    const rowCosts = resources.createBuffer('row-costs', vertexCapacity * 4);
    const nodeDegrees = resources.createBuffer('node-degrees', nodeCapacity * 4);

    const graph = new GPUCommandGraph<void>(device, {id: 'noding'});
    const scalarView = <Format extends 'uint32' | 'float32'>(
      name: string,
      buffer: Buffer,
      format: Format,
      length: number
    ) => importGraphBuffer(graph, name, buffer, format, length);
    const pairView = (name: string, buffer: Buffer, length: number) =>
      importGraphBuffer(graph, name, buffer, 'float32x2', length);
    const views = {
      pieceLines: scalarView('piece-lines', pieceLines, 'uint32', pieceCapacity),
      pieceOffsets: scalarView('piece-offsets', pieceOffsets, 'uint32', pieceCapacity + 1),
      piecePositions: pairView('piece-positions', piecePositions, vertexCapacity),
      pieceCount: scalarView('piece-count', pieceCount, 'uint32', 1),
      nodePositions: pairView('node-positions', nodePositions, nodeCapacity),
      nodeCount: scalarView('node-count', nodeCount, 'uint32', 1),
      fromNodes: scalarView('from-nodes', fromNodes, 'uint32', pieceCapacity),
      toNodes: scalarView('to-nodes', toNodes, 'uint32', pieceCapacity),
      csrOffsets: scalarView('csr-offsets', csrOffsets, 'uint32', nodeCapacity + 1),
      costs: scalarView('costs', costs, 'float32', nodeCapacity),
      source: scalarView('source', source, 'uint32', 1)
    };
    const csrNeighborsView = scalarView('csr-neighbors', csrNeighbors, 'uint32', pieceCapacity * 2);
    const csrWeightsView = scalarView('csr-weights', csrWeights, 'float32', pieceCapacity * 2);
    graph.add(
      new GPUNetworkNoding({
        id: 'noding',
        lines: {
          kind: 'lines',
          positions: pairView('line-positions', linePositions, vertexTotal),
          lineOffsets: scalarView(
            'line-offsets',
            lineOffsetsBuffer,
            'uint32',
            lines.lineOffsets.length
          )
        },
        intersectionCapacity,
        tolerance: tolerance.importToGraph(graph),
        pieces: {
          lineIds: views.pieceLines,
          offsets: views.pieceOffsets,
          positions: views.piecePositions,
          count: views.pieceCount,
          totalCount: scalarView('piece-total', pieceTotal, 'uint32', 1)
        },
        nodes: {
          positions: views.nodePositions,
          count: views.nodeCount,
          totalCount: scalarView('node-total', nodeTotal, 'uint32', 1)
        },
        edges: {
          fromNodes: views.fromNodes,
          toNodes: views.toNodes,
          lengths: scalarView('lengths', lengths, 'float32', pieceCapacity)
        },
        csr: {offsets: views.csrOffsets, neighbors: csrNeighborsView, weights: csrWeightsView},
        overflow: scalarView('overflow', overflow, 'uint32', 1)
      })
    );
    // Nearest node to the clicked point: one thread scans the nodes (a click is rare).
    addKernelPass(graph, {
      id: 'noding-nearest-node',
      bindings: [
        {name: 'nodePositions', view: views.nodePositions, type: 'f32', access: 'read'},
        {name: 'nodeCount', view: views.nodeCount, type: 'u32', access: 'read'},
        {name: 'clickPoint', view: clickPoint.importToGraph(graph), type: 'f32', access: 'read'},
        {name: 'source', view: views.source, type: 'u32', access: 'read_write'},
        {
          name: 'sourcePosition',
          view: pairView('source-position', sourcePosition, 1),
          type: 'f32',
          access: 'read_write'
        }
      ],
      invocationCount: 1,
      body: `var best = 0u;
  var bestDistance = 3.4e38;
  let click = vec2f(clickPoint[clickPointOffset], clickPoint[clickPointOffset + 1u]);
  for (var node = 0u; node < nodeCount[nodeCountOffset]; node++) {
    let position = vec2f(nodePositions[nodePositionsOffset + node * 2u], nodePositions[nodePositionsOffset + node * 2u + 1u]);
    let delta = position - click;
    let distanceSquared = dot(delta, delta);
    if (distanceSquared < bestDistance) { bestDistance = distanceSquared; best = node; }
  }
  source[sourceOffset] = best;
  sourcePosition[sourcePositionOffset] = nodePositions[nodePositionsOffset + best * 2u];
  sourcePosition[sourcePositionOffset + 1u] = nodePositions[nodePositionsOffset + best * 2u + 1u];`
    });
    graph.add(
      new GPUNetworkReachability({
        id: 'noding-reachability',
        offsets: views.csrOffsets,
        neighbors: csrNeighborsView,
        weights: csrWeightsView,
        sources: views.source,
        costLimit: costLimit.importToGraph(graph),
        maxIterations: MAXIMUM_ITERATIONS,
        localIterations: LOCAL_ITERATIONS,
        costs: views.costs,
        converged: scalarView('converged', converged, 'uint32', 1)
      })
    );
    addNodingDisplayPasses(graph, {
      pieceOffsets: views.pieceOffsets,
      piecePositions: views.piecePositions,
      pieceCount: views.pieceCount,
      fromNodes: views.fromNodes,
      toNodes: views.toNodes,
      csrOffsets: views.csrOffsets,
      nodeCount: views.nodeCount,
      costs: views.costs,
      segmentRows: pairView('segment-rows', segmentRows, vertexCapacity * 2),
      rowPieces: scalarView('row-pieces', rowPieces, 'uint32', vertexCapacity),
      rowCosts: scalarView('row-costs', rowCosts, 'float32', vertexCapacity),
      nodeDegrees: scalarView('node-degrees', nodeDegrees, 'uint32', nodeCapacity)
    });
    const compiled: CompiledGPUCommandGraph<void> = resources.track(graph.compile());

    const toleranceReadout = context.controls.addReadout('Snap tolerance');
    const writeTolerance = () => {
      tolerance.write(Float32Array.of(toleranceMeters));
      toleranceReadout.setValue(toleranceMeters === 0 ? 'exact (0 m)' : `${toleranceMeters} m`);
      dirty = true;
    };
    const writeBudget = () => {
      costLimit.write(Float32Array.of(budgetMeters));
      dirty = true;
    };
    const writeClick = (x: number, y: number) => {
      clickPoint.write(Float32Array.of(x, y));
      dirty = true;
    };

    context.controls.addSlider({
      label: 'Snap tolerance (per-frame parameter)',
      min: 0,
      max: 12,
      step: 0.5,
      value: toleranceMeters,
      format: value => (value === 0 ? 'exact' : `${value} m`),
      onChange: value => {
        toleranceMeters = value;
        writeTolerance();
      }
    });
    context.controls.addSlider({
      label: 'Route budget (cost limit)',
      min: 250,
      max: 6000,
      step: 250,
      value: budgetMeters,
      format: value => `${value} m`,
      onChange: value => {
        budgetMeters = value;
        writeBudget();
        context.updateLayers();
      }
    });
    context.controls.addSelect<View>({
      label: 'Color edges by',
      options: [
        {value: 'pieces', label: 'Piece (edge)'},
        {value: 'cost', label: 'Route cost from the clicked node'}
      ],
      value: view,
      onChange: value => {
        view = value;
        context.updateLayers();
      }
    });
    context.controls.addLegend({
      title: 'Node degree',
      entries: [
        {color: NODE_DEGREE_PALETTE[1], label: '1 (dead end)'},
        {color: NODE_DEGREE_PALETTE[3], label: '3'},
        {color: NODE_DEGREE_PALETTE[4], label: '4'},
        {color: NODE_DEGREE_PALETTE[5], label: '5 or more'}
      ]
    });
    context.controls.addNote('Click the map to route from the nearest node.');
    context.controls.addReadout(
      'Input segments / polylines',
      `${formatCount(segmentCount)} / ${formatCount(lineCount)}`
    );
    const pieceReadout = context.controls.addReadout('Edges (pieces)');
    const nodeReadout = context.controls.addReadout('Nodes (original graph)');
    const overflowReadout = context.controls.addReadout('Overflow');
    const convergedReadout = context.controls.addReadout('Routing converged');
    context.controls.addReadout('Data', roads.attribution);

    const originalNodeCount = roads.nodePositions.length / 2;
    const reader = new SummaryReader(
      resources,
      'noding',
      [
        {buffer: pieceCount, size: 4},
        {buffer: pieceTotal, size: 4},
        {buffer: nodeCount, size: 4},
        {buffer: nodeTotal, size: 4},
        {buffer: overflow, size: 4},
        {buffer: converged, size: 4}
      ],
      bytes => {
        if (destroyed) return;
        const words = new Uint32Array(bytes);
        pieceReadout.setValue(
          `${formatCount(words[0])}${words[1] > words[0] ? ` of ${formatCount(words[1])}` : ''}`
        );
        nodeReadout.setValue(`${formatCount(words[2])} (${formatCount(originalNodeCount)})`);
        overflowReadout.setValue(words[4] ? 'yes: raise a capacity' : 'no');
        convergedReadout.setValue(words[5] ? 'yes' : 'no (iteration limit)');
      }
    );

    writeTolerance();
    writeBudget();
    writeClick(...projection.project(...DEFAULT_SOURCE));

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [compiled],
      encode(commandEncoder, frame) {
        if (dirty || frame.frameIndex < 2) {
          compiled.encode(commandEncoder, {parameters: undefined});
          dirty = false;
          reader.markStale();
        }
        reader.flush(commandEncoder);
      },
      getLayers() {
        const coordinateOrigin: [number, number, number] = [roads.origin[0], roads.origin[1], 0];
        const byCost = view === 'cost';
        const layers: Layer[] = [
          new SpatialAnalysisSegmentLayer({
            id: 'noding-edges',
            coordinateOrigin,
            segments: segmentRows,
            instanceCount: vertexCapacity,
            widthPixels: 2.5,
            values: byCost ? rowCosts : rowPieces,
            valueFormat: byCost ? 'float32' : 'uint32',
            colormap: byCost ? 'viridis' : 'category',
            valueRange: [0, budgetMeters],
            palette: PIECE_PALETTE,
            noDataValue: NODING_HIDDEN,
            noDataColor: byCost ? UNREACHED_COLOR : [0, 0, 0, 0],
            discardAtOrBelow: byCost ? -0.5 : undefined
          }),
          new SpatialAnalysisPointLayer({
            id: 'noding-nodes',
            coordinateOrigin,
            positions: nodePositions,
            instanceCount: nodeCapacity,
            radiusPixels: 3,
            values: nodeDegrees,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: NODE_DEGREE_PALETTE,
            noDataValue: NODING_HIDDEN,
            noDataColor: [0, 0, 0, 0]
          }),
          new SpatialAnalysisPointLayer({
            id: 'noding-source',
            coordinateOrigin,
            positions: sourcePosition,
            instanceCount: 1,
            radiusPixels: 8,
            color: [255, 255, 255, 255]
          })
        ];
        return layers;
      },
      onClick(event) {
        if (!event.coordinate) return false;
        writeClick(...projection.project(event.coordinate[0], event.coordinate[1]));
        return true;
      },
      destroy() {
        destroyed = true;
        reader.stop();
        resources.destroy();
      }
    };
    return instance;
  }
};
