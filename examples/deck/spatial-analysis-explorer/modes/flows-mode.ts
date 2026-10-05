// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer, CommandEncoder} from '@luma.gl/core';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  GPUReadbackRing,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUTimeWindowParameterValues,
  GPU_TIME_WINDOW_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-dataframe';
import {GPUFlowAggregation} from '@luma.gl/experimental/gpu-network';
import {importGraphBuffer} from '../graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisRasterLayer} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {ARC_SEGMENTS, FlowArcLayer} from './flows-layers';

type ZoneKind = 'grid' | 'hexagon';
type WeightKind = 'trips' | 'distance';
type ZoneTotals = 'departures' | 'arrivals';
type Bounds = [number, number, number, number];

const SQRT3 = Math.sqrt(3);
const TOP_FLOW_COUNT = 256;
const PAIR_CAPACITY = 2048;
const MAXIMUM_ZONE_COUNT = 65535;
/** Padding around the origin/destination extent so every endpoint is inside the lattice. */
const BOUNDS_PADDING_METERS = 800;
const READBACK_INTERVAL_FRAMES = 15;
/** Header words in the summary readback before the zone totals. */
const SUMMARY_HEADER_WORDS = 8;
/** Frames a replaced graph's buffers stay alive so in-flight frames never touch destroyed buffers. */
const RETIRE_FRAMES = 4;
const ORIGIN_COLOR = [255, 150, 60, 255] as const;
const DESTINATION_COLOR = [60, 220, 255, 255] as const;
const VIRIDIS_STOPS = [
  [68, 1, 84],
  [59, 82, 139],
  [33, 145, 140],
  [94, 201, 98],
  [253, 231, 37]
] as const;

type FlowGraph = {
  resources: SpatialAnalysisResources;
  compiled: CompiledGPUCommandGraph<void>;
  zoneKind: ZoneKind;
  zoneSize: number;
  gridSize: [number, number];
  zoneCount: number;
  flowOriginZoneIds: Buffer;
  flowDestinationZoneIds: Buffer;
  flowWeights: Buffer;
  zoneOutCounts: Buffer;
  zoneInCounts: Buffer;
  drawCommands: DrawCommandBuffer;
  count: Buffer;
  totalCount: Buffer;
  overflow: Buffer;
  pairOverflow: Buffer;
  readbackRing: GPUReadbackRing;
  serial: number;
};

/**
 * Origin-destination flows between map zones. Every New York trip is one flow row (first vertex to
 * last vertex). One compiled `GPUFlowAggregation` assigns hexagon or grid zones, gates rows by a
 * per-frame time window, counts (or sums distance for) every origin-destination pair in a GPU hash
 * table, ranks the pairs, and writes the top 256 flows, per-zone departure and arrival totals, and
 * the indirect arc count. Window, weights and totals selection change per frame; zone kind, zone
 * size and self-flow exclusion are compile-time and rebuild the graph.
 */
export const flowsMode: SpatialAnalysisModeDefinition = {
  id: 'flows',
  title: 'Flows',
  contributors: ['GPUFlowAggregation'],
  description:
    'Taxi origin-destination flows: trips are binned into hexagons or grid cells on the GPU, ' +
    'paired, ranked and drawn as arcs from origin (orange) to destination (cyan). Slide or play ' +
    'the time window: rows outside it are gated on the GPU without recompiling.',
  initialViewState: {longitude: -74.015, latitude: 40.72, zoom: 12.4},

  async create(context) {
    const trips = await context.data.getNewYorkTrips();
    context.signal.throwIfAborted();
    const {device} = context;
    const resources = new SpatialAnalysisResources(device, 'flows');

    // One flow row per trip: origin = first vertex, destination = last vertex.
    const tripCount = trips.vendors.length;
    const origins = new Float32Array(tripCount * 2);
    const destinations = new Float32Array(tripCount * 2);
    // Departure times are rebased to a local epoch because the contributor compares float32 values.
    const [timeStart, timeEnd] = trips.timeRange;
    const departureTimes = new Float32Array(tripCount);
    const tripWeights = new Float32Array(tripCount).fill(1);
    const distanceWeights = new Float32Array(tripCount);
    for (let trip = 0; trip < tripCount; trip++) {
      const first = trips.tripOffsets[trip];
      const last = trips.tripOffsets[trip + 1] - 1;
      origins.set(trips.vertexPositions.subarray(first * 2, first * 2 + 2), trip * 2);
      destinations.set(trips.vertexPositions.subarray(last * 2, last * 2 + 2), trip * 2);
      departureTimes[trip] = trips.vertexTimestamps[first] - timeStart;
      let meters = 0;
      for (let vertex = first; vertex < last; vertex++) {
        meters += Math.hypot(
          trips.vertexPositions[vertex * 2 + 2] - trips.vertexPositions[vertex * 2],
          trips.vertexPositions[vertex * 2 + 3] - trips.vertexPositions[vertex * 2 + 1]
        );
      }
      distanceWeights[trip] = meters / 1000;
    }
    // Literal bounds computed once from the padded extent of all origins and destinations. The
    // dataset has a few far-away outlier endpoints (other cities), so the extent uses the 0.5th to
    // 99.5th percentile per axis; trips outside it are rejected by the contributor (counted below).
    const bounds = getRobustBounds([origins, destinations]);
    let outsideCount = 0;
    for (let trip = 0; trip < tripCount; trip++) {
      for (const array of [origins, destinations]) {
        const x = array[trip * 2];
        const y = array[trip * 2 + 1];
        if (x < bounds[0] || x > bounds[2] || y < bounds[1] || y > bounds[3]) {
          outsideCount++;
          break;
        }
      }
    }

    const originsBuffer = resources.createBuffer('origins', origins);
    const destinationsBuffer = resources.createBuffer('destinations', destinations);
    const timestampsBuffer = resources.createBuffer('departure-times', departureTimes);
    const weightsBuffer = resources.createBuffer('weights', tripWeights);
    const windowBuffer = resources.createParameterBuffer(
      'window',
      'float32',
      GPU_TIME_WINDOW_PARAMETER_LENGTH
    );

    let zoneKind: ZoneKind = 'hexagon';
    let zoneSize = 700;
    let excludeSelfFlows = true;
    let weightKind: WeightKind = 'trips';
    let zoneTotals: ZoneTotals = 'departures';
    let showEndpoints = false;
    let playing = false;
    let playbackSpeed = 60;
    const span = Math.max(1, timeEnd - timeStart);
    let windowStart = 0;
    let windowLength = Math.ceil(span) + 1;
    let zoneValueMaximum = 20;

    let graph: FlowGraph | null = null;
    let serial = 0;
    let readbackPending = false;
    let destroyed = false;
    let framesSinceRebuild = 0;
    const retired: {resources: SpatialAnalysisResources; frames: number}[] = [];

    function getGridSize(kind: ZoneKind, size: number): [number, number] {
      const width = bounds[2] - bounds[0];
      const height = bounds[3] - bounds[1];
      return kind === 'hexagon'
        ? [Math.ceil(width / (SQRT3 * size)) + 1, Math.ceil(height / (1.5 * size)) + 1]
        : [Math.ceil(width / size), Math.ceil(height / size)];
    }

    function buildGraph(): void {
      const gridSize = getGridSize(zoneKind, zoneSize);
      const zoneCount = gridSize[0] * gridSize[1];
      if (zoneCount > MAXIMUM_ZONE_COUNT) {
        throw new Error(`Zone count ${zoneCount} exceeds ${MAXIMUM_ZONE_COUNT}`);
      }
      if (graph) {
        retired.push({resources: graph.resources, frames: 0});
      }
      const graphResources = new SpatialAnalysisResources(device, `flows-${++serial}`);
      const idsBuffer = graphResources.createBuffer('ids', TOP_FLOW_COUNT * 4);
      const count = graphResources.createBuffer('count', 4);
      const totalCount = graphResources.createBuffer('total-count', 4);
      const overflow = graphResources.createBuffer('overflow', 4);
      const pairOverflow = graphResources.createBuffer('pair-overflow', 4);
      const flowOriginZoneIds = graphResources.createBuffer('flow-origins', TOP_FLOW_COUNT * 4);
      const flowDestinationZoneIds = graphResources.createBuffer(
        'flow-destinations',
        TOP_FLOW_COUNT * 4
      );
      const flowCounts = graphResources.createBuffer('flow-counts', TOP_FLOW_COUNT * 4);
      const flowWeights = graphResources.createBuffer('flow-weights', TOP_FLOW_COUNT * 4);
      const zoneOutCounts = graphResources.createBuffer('zone-out', zoneCount * 4);
      const zoneInCounts = graphResources.createBuffer('zone-in', zoneCount * 4);
      const drawCommands = graphResources.track(
        new DrawCommandBuffer(device, {
          id: `flows-draw-${serial}`,
          type: 'draw',
          commands: [{vertexCount: ARC_SEGMENTS * 6, instanceCount: 0}]
        })
      );
      const readbackRing = graphResources.track(
        new GPUReadbackRing(device, {
          id: `flows-summary-${serial}`,
          byteLength: (SUMMARY_HEADER_WORDS + zoneCount * 2) * 4
        })
      );

      const commandGraph = new GPUCommandGraph<void>(device, {id: `flows-${serial}`});
      commandGraph.add(
        new GPUFlowAggregation({
          id: 'flows',
          zones:
            zoneKind === 'hexagon'
              ? {kind: 'hexagon', bounds, gridSize, radius: zoneSize}
              : {kind: 'grid', bounds, gridSize},
          origins: importGraphBuffer(
            commandGraph,
            'origins',
            originsBuffer,
            'float32x2',
            tripCount
          ),
          destinations: importGraphBuffer(
            commandGraph,
            'destinations',
            destinationsBuffer,
            'float32x2',
            tripCount
          ),
          weights: importGraphBuffer(commandGraph, 'weights', weightsBuffer, 'float32', tripCount),
          timeWindow: {
            timestamps: importGraphBuffer(
              commandGraph,
              'timestamps',
              timestampsBuffer,
              'float32',
              tripCount
            ),
            window: windowBuffer.importToGraph(commandGraph)
          },
          excludeSelfFlows,
          pairCapacity: PAIR_CAPACITY,
          output: {
            ids: importGraphBuffer(commandGraph, 'ids', idsBuffer, 'uint32', TOP_FLOW_COUNT),
            count: importGraphBuffer(commandGraph, 'count', count, 'uint32', 1),
            overflow: importGraphBuffer(commandGraph, 'overflow', overflow, 'uint32', 1),
            totalCount: importGraphBuffer(commandGraph, 'total-count', totalCount, 'uint32', 1)
          },
          flowOriginZoneIds: importGraphBuffer(
            commandGraph,
            'flow-origins',
            flowOriginZoneIds,
            'uint32',
            TOP_FLOW_COUNT
          ),
          flowDestinationZoneIds: importGraphBuffer(
            commandGraph,
            'flow-destinations',
            flowDestinationZoneIds,
            'uint32',
            TOP_FLOW_COUNT
          ),
          flowCounts: importGraphBuffer(
            commandGraph,
            'flow-counts',
            flowCounts,
            'uint32',
            TOP_FLOW_COUNT
          ),
          flowWeights: importGraphBuffer(
            commandGraph,
            'flow-weights',
            flowWeights,
            'float32',
            TOP_FLOW_COUNT
          ),
          pairOverflow: importGraphBuffer(commandGraph, 'pair-overflow', pairOverflow, 'uint32', 1),
          zoneOutCounts: importGraphBuffer(
            commandGraph,
            'zone-out',
            zoneOutCounts,
            'uint32',
            zoneCount
          ),
          zoneInCounts: importGraphBuffer(
            commandGraph,
            'zone-in',
            zoneInCounts,
            'uint32',
            zoneCount
          ),
          drawInstanceCount: commandGraph.importGPUData(
            'arcs',
            drawCommands.getInstanceCountData(0)
          )
        })
      );
      const compiled = graphResources.track(commandGraph.compile());
      graph = {
        resources: graphResources,
        compiled,
        zoneKind,
        zoneSize,
        gridSize,
        zoneCount,
        flowOriginZoneIds,
        flowDestinationZoneIds,
        flowWeights,
        zoneOutCounts,
        zoneInCounts,
        drawCommands,
        count,
        totalCount,
        overflow,
        pairOverflow,
        readbackRing,
        serial
      };
      framesSinceRebuild = READBACK_INTERVAL_FRAMES; // read the new graph's totals immediately
      zoneReadout.setValue(
        `${gridSize[0]} × ${gridSize[1]} ${zoneKind === 'hexagon' ? 'hexagons' : 'cells'} ` +
          `(${formatCount(zoneCount)} zones)`
      );
    }

    context.controls.addSelect<ZoneKind>({
      label: 'Zones (compile-time)',
      options: [
        {value: 'hexagon', label: 'Hexagons'},
        {value: 'grid', label: 'Square grid'}
      ],
      value: zoneKind,
      onChange: value => {
        zoneKind = value;
        buildGraph();
        context.updateLayers();
      }
    });
    context.controls.addSelect<string>({
      label: 'Zone size (compile-time; hexagon radius / cell size)',
      options: [
        {value: '250', label: '250 m'},
        {value: '400', label: '400 m'},
        {value: '700', label: '700 m'}
      ],
      value: String(zoneSize),
      onChange: value => {
        zoneSize = Number(value);
        buildGraph();
        context.updateLayers();
      }
    });
    context.controls.addToggle({
      label: 'Exclude same-zone trips (compile-time)',
      value: excludeSelfFlows,
      onChange: value => {
        excludeSelfFlows = value;
        buildGraph();
        context.updateLayers();
      }
    });
    context.controls.addSelect<WeightKind>({
      label: 'Flow weight (per-frame buffer)',
      options: [
        {value: 'trips', label: 'Trips'},
        {value: 'distance', label: 'Distance (km)'}
      ],
      value: weightKind,
      onChange: value => {
        weightKind = value;
        weightsBuffer.write(weightKind === 'trips' ? tripWeights : distanceWeights);
        flowUnit = weightKind === 'trips' ? 'trips' : 'km';
      }
    });
    context.controls.addSelect<ZoneTotals>({
      label: 'Zone totals (layer prop)',
      options: [
        {value: 'departures', label: 'Departures'},
        {value: 'arrivals', label: 'Arrivals'}
      ],
      value: zoneTotals,
      onChange: value => {
        zoneTotals = value;
        context.updateLayers();
      }
    });
    const startSlider = context.controls.addSlider({
      label: 'Window start (per-frame)',
      min: 0,
      max: Math.ceil(span),
      step: 1,
      value: windowStart,
      format: formatSeconds,
      onChange: value => {
        windowStart = value;
      }
    });
    context.controls.addSlider({
      label: 'Window length (per-frame)',
      min: 60,
      max: Math.ceil(span) + 1,
      step: 10,
      value: windowLength,
      format: value => (value >= span ? 'full range' : formatSeconds(value)),
      onChange: value => {
        windowLength = value;
      }
    });
    context.controls.addToggle({
      label: 'Play (advance window start)',
      value: playing,
      onChange: value => {
        playing = value;
      }
    });
    context.controls.addSlider({
      label: 'Play speed (simulated seconds per second)',
      min: 10,
      max: 300,
      step: 10,
      value: playbackSpeed,
      format: value => `${value}×`,
      onChange: value => {
        playbackSpeed = value;
      }
    });
    context.controls.addToggle({
      label: 'Show trip endpoints',
      value: showEndpoints,
      onChange: value => {
        showEndpoints = value;
        context.updateLayers();
      }
    });
    context.controls.addLegend({
      title: 'Arcs: origin → destination (width = flow weight)',
      gradient: {
        colors: [ORIGIN_COLOR, DESTINATION_COLOR],
        minimumLabel: 'origin',
        maximumLabel: 'destination'
      }
    });
    context.controls.addLegend({
      title: 'Zone totals (sqrt scale)',
      gradient: {colors: VIRIDIS_STOPS, minimumLabel: '0', maximumLabel: 'busiest zone'}
    });
    const zoneReadout = context.controls.addReadout('Zones');
    const windowReadout = context.controls.addReadout('Window');
    const tripsReadout = context.controls.addReadout('Trips in window');
    const pairsReadout = context.controls.addReadout('Distinct pairs');
    const arcsReadout = context.controls.addReadout('Arcs drawn');
    const truncatedReadout = context.controls.addReadout('Top-256 truncated');
    const pairOverflowReadout = context.controls.addReadout('Pair table overflow');
    const busiestOriginReadout = context.controls.addReadout('Busiest origin zone');
    const busiestDestinationReadout = context.controls.addReadout('Busiest destination zone');
    const largestFlowReadout = context.controls.addReadout('Largest flow');
    context.controls.addNote(
      'Departure times are rebased to the first trip (t - ' +
        `${timeStart.toFixed(0)} s) before upload. Zone totals and readouts come from a ` +
        'small readback (2 × zones u32) every 15 frames; the zone color range follows it.'
    );
    context.controls.addReadout('Trips outside lattice', formatCount(outsideCount));
    context.controls.addReadout('Data', trips.attribution);
    let flowUnit = 'trips';

    function formatSeconds(seconds: number): string {
      const whole = Math.max(0, Math.round(seconds));
      return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
    }

    buildGraph();

    const readSummary = async (current: FlowGraph, commandEncoder: CommandEncoder) => {
      const ticket = current.readbackRing.tryAcquire();
      if (!ticket) return;
      const zoneBytes = current.zoneCount * 4;
      const copy = (source: Buffer, destinationOffset: number, size: number) =>
        commandEncoder.copyBufferToBuffer({
          sourceBuffer: source,
          sourceOffset: 0,
          destinationBuffer: ticket.buffer,
          destinationOffset,
          size
        });
      copy(current.count, 0, 4);
      copy(current.totalCount, 4, 4);
      copy(current.overflow, 8, 4);
      copy(current.pairOverflow, 12, 4);
      copy(current.flowWeights, 16, 4);
      copy(current.zoneOutCounts, SUMMARY_HEADER_WORDS * 4, zoneBytes);
      copy(current.zoneInCounts, SUMMARY_HEADER_WORDS * 4 + zoneBytes, zoneBytes);
      const byteLength = SUMMARY_HEADER_WORDS * 4 + zoneBytes * 2;
      ticket.markEncoded({byteOffset: 0, byteLength});
      readbackPending = true;
      try {
        const bytes = await ticket.read();
        if (destroyed || graph !== current) return;
        const words = new Uint32Array(
          bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + byteLength)
        );
        const largestWeight = new Float32Array(words.buffer, 16, 1)[0];
        let tripsInWindow = 0;
        let busiestOrigin = 0;
        let busiestDestination = 0;
        for (let zone = 0; zone < current.zoneCount; zone++) {
          const out = words[SUMMARY_HEADER_WORDS + zone];
          const incoming = words[SUMMARY_HEADER_WORDS + current.zoneCount + zone];
          tripsInWindow += out;
          busiestOrigin = Math.max(busiestOrigin, out);
          busiestDestination = Math.max(busiestDestination, incoming);
        }
        const [arcCount, pairCount, overflow, pairOverflow] = words;
        tripsReadout.setValue(formatCount(tripsInWindow));
        pairsReadout.setValue(formatCount(pairCount));
        arcsReadout.setValue(formatCount(arcCount));
        truncatedReadout.setValue(pairCount > TOP_FLOW_COUNT ? 'yes' : 'no');
        pairOverflowReadout.setValue(pairOverflow ? 'yes (incomplete)' : overflow ? 'no' : 'no');
        busiestOriginReadout.setValue(`${formatCount(busiestOrigin)} departures`);
        busiestDestinationReadout.setValue(`${formatCount(busiestDestination)} arrivals`);
        largestFlowReadout.setValue(
          `${flowUnit === 'trips' ? formatCount(largestWeight) : largestWeight.toFixed(1)} ${flowUnit}`
        );
        // The contributor has no extent output, so the zone color range comes from this readback.
        const maximum = Math.max(
          1,
          zoneTotals === 'departures' ? busiestOrigin : busiestDestination
        );
        if (maximum !== zoneValueMaximum) {
          zoneValueMaximum = maximum;
          context.updateLayers();
        }
      } catch {
        // The ring or device was destroyed while the read was in flight.
      } finally {
        readbackPending = false;
      }
    };

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => (graph ? [graph.compiled] : []),
      encode(commandEncoder, frame) {
        if (!graph) return;
        for (let index = retired.length - 1; index >= 0; index--) {
          if (++retired[index].frames > RETIRE_FRAMES) {
            retired[index].resources.destroy();
            retired.splice(index, 1);
          }
        }
        if (playing) {
          windowStart += frame.deltaSeconds * playbackSpeed;
          if (windowStart > span) windowStart -= span;
        }
        windowBuffer.write(
          getGPUTimeWindowParameterValues({start: windowStart, end: windowStart + windowLength})
        );
        graph.compiled.encode(commandEncoder, {parameters: undefined});
        if (frame.frameIndex % 10 === 0) {
          if (playing) startSlider.setValue(Math.round(windowStart));
          windowReadout.setValue(
            windowLength >= span
              ? 'full range'
              : `${formatSeconds(windowStart)} – ${formatSeconds(windowStart + windowLength)}`
          );
        }
        framesSinceRebuild++;
        if (framesSinceRebuild >= READBACK_INTERVAL_FRAMES && !readbackPending) {
          framesSinceRebuild = 0;
          void readSummary(graph, commandEncoder);
        }
      },
      getLayers(): Layer[] {
        if (!graph) return [];
        const coordinateOrigin: [number, number, number] = [trips.origin[0], trips.origin[1], 0];
        const layers: Layer[] = [
          new SpatialAnalysisRasterLayer({
            id: `flows-zones-${graph.serial}`,
            coordinateOrigin,
            gridSize: graph.gridSize,
            bounds,
            binning: graph.zoneKind,
            hexagonRadius: graph.zoneSize,
            values: zoneTotals === 'departures' ? graph.zoneOutCounts : graph.zoneInCounts,
            valueFormat: 'uint32',
            colormap: 'viridis',
            valueRange: [0, zoneValueMaximum],
            sqrtScale: true,
            discardAtOrBelow: 0,
            color: [255, 255, 255, 120]
          }),
          new FlowArcLayer({
            id: `flows-arcs-${graph.serial}`,
            coordinateOrigin,
            flowOriginZoneIds: graph.flowOriginZoneIds,
            flowDestinationZoneIds: graph.flowDestinationZoneIds,
            flowWeights: graph.flowWeights,
            flowCount: graph.count,
            drawCommands: graph.drawCommands,
            zoneKind: graph.zoneKind,
            gridSize: graph.gridSize,
            bounds,
            hexagonRadius: graph.zoneSize,
            originColor: [...ORIGIN_COLOR],
            destinationColor: [...DESTINATION_COLOR]
          })
        ];
        if (showEndpoints) {
          layers.push(
            new SpatialAnalysisPointLayer({
              id: 'flows-origins',
              coordinateOrigin,
              positions: originsBuffer,
              instanceCount: tripCount,
              radiusPixels: 2.5,
              color: [...ORIGIN_COLOR]
            }),
            new SpatialAnalysisPointLayer({
              id: 'flows-destinations',
              coordinateOrigin,
              positions: destinationsBuffer,
              instanceCount: tripCount,
              radiusPixels: 2.5,
              color: [...DESTINATION_COLOR]
            })
          );
        }
        return layers;
      },
      destroy() {
        destroyed = true;
        graph?.resources.destroy();
        for (const entry of retired) entry.resources.destroy();
        resources.destroy();
      }
    };
    return instance;
  }
};

/** Padded `[minX, minY, maxX, maxY]` of the 0.5th to 99.5th percentile of all pooled positions. */
function getRobustBounds(positionArrays: readonly Float32Array[]): Bounds {
  const xs: number[] = [];
  const ys: number[] = [];
  for (const positions of positionArrays) {
    for (let index = 0; index < positions.length; index += 2) {
      xs.push(positions[index]);
      ys.push(positions[index + 1]);
    }
  }
  xs.sort((a, b) => a - b);
  ys.sort((a, b) => a - b);
  const quantile = (values: number[], fraction: number) =>
    values[Math.round(fraction * (values.length - 1))];
  return [
    quantile(xs, 0.005) - BOUNDS_PADDING_METERS,
    quantile(ys, 0.005) - BOUNDS_PADDING_METERS,
    quantile(xs, 0.995) + BOUNDS_PADDING_METERS,
    quantile(ys, 0.995) + BOUNDS_PADDING_METERS
  ];
}
