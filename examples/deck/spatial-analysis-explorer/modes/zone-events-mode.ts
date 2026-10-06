// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Trajectories against zones. One compiled graph (`addFleetDwellZoneEventsRecipe`) runs
 * `GPUZoneEvents` over every New York trip against jittered zip-code-like zones, then
 * `GPUGroupStatistics` rolls the dense `(track, zone)` dwell and visit matrices up to one table row
 * per zone. Enter and exit events are drawn by a custom layer that interpolates each crossing
 * position on the GPU from the event row and time; a playhead (a parameter write, never a
 * recompile) makes the markers pulse as the trips cross zone borders. The zone fill reads the
 * statistics table directly. Only the per-track event cap rebuilds the graph.
 */

import type {Layer} from '@deck.gl/core';
import type {CommandEncoder} from '@luma.gl/core';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {addFleetDwellZoneEventsRecipe} from '@luma.gl/experimental/gpu-spatial-analysis';
import {importGraphBuffer} from '../graph-buffers';
import {createSeededRandom, LocalMetricProjection} from '../spatial-analysis-data';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {SummaryReader} from './summary-reader';
import {ZoneChoroplethLayer, ZoneEventMarkerLayer} from './zone-events-layers';

const ZONE_COLUMNS = 8;
const ZONE_ROWS = 6;
const ZONE_COUNT = ZONE_COLUMNS * ZONE_ROWS;
/** Largest per-track event cap the buffers are sized for. */
const MAXIMUM_EVENTS_PER_TRACK = 8;
const CANDIDATE_CAPACITY = 1 << 20;
const RECORD_MARKERS = 0;
const RECORD_ZONES = 1;
const RECORD_BYTE_LENGTH = 16;
const NO_ZONE = 0xffffffff;

type Metric = 'dwell' | 'mean' | 'visitors';
const METRIC_INDEX: Record<Metric, number> = {dwell: 0, mean: 1, visitors: 2};

/** Zones as boundary edges, fan triangles and CPU rings (for picking). */
type ZoneGrid = {
  edgeStarts: Float32Array;
  edgeEnds: Float32Array;
  edgeZones: Uint32Array;
  /** Interleaved `x0, y0, x1, y1` per edge for drawing. */
  outlineSegments: Float32Array;
  triangles: Float32Array;
  triangleZones: Uint32Array;
  rings: Float32Array[];
};

/**
 * Builds a jittered lattice of zones over `bounds`. Corner and edge-midpoint vertices are shared by
 * neighbors, so borders match exactly while the zones stay irregular like postal areas.
 */
function createZoneGrid(bounds: readonly [number, number, number, number]): ZoneGrid {
  const random = createSeededRandom(4242);
  const [minX, minY, maxX, maxY] = bounds;
  const cellWidth = (maxX - minX) / ZONE_COLUMNS;
  const cellHeight = (maxY - minY) / ZONE_ROWS;
  const jitter = (amount: number) => (random() - 0.5) * 2 * amount;
  const corners: [number, number][][] = [];
  for (let row = 0; row <= ZONE_ROWS; row++) {
    corners.push([]);
    for (let column = 0; column <= ZONE_COLUMNS; column++) {
      const border = row === 0 || row === ZONE_ROWS || column === 0 || column === ZONE_COLUMNS;
      corners[row].push([
        minX + column * cellWidth + (border ? 0 : jitter(cellWidth * 0.22)),
        minY + row * cellHeight + (border ? 0 : jitter(cellHeight * 0.22))
      ]);
    }
  }
  // Midpoints of horizontal edges (row, column to column + 1) and vertical edges.
  const horizontal = new Map<string, [number, number]>();
  const vertical = new Map<string, [number, number]>();
  const midpoint = (
    cache: Map<string, [number, number]>,
    row: number,
    column: number,
    from: [number, number],
    to: [number, number],
    isHorizontal: boolean
  ): [number, number] => {
    const key = `${row},${column}`;
    let result = cache.get(key);
    if (!result) {
      const offset = jitter(0.12);
      result = [
        (from[0] + to[0]) / 2 + (isHorizontal ? 0 : offset * cellWidth),
        (from[1] + to[1]) / 2 + (isHorizontal ? offset * cellHeight : 0)
      ];
      cache.set(key, result);
    }
    return result;
  };
  const starts: number[] = [];
  const ends: number[] = [];
  const zones: number[] = [];
  const triangles: number[] = [];
  const triangleZones: number[] = [];
  const rings: Float32Array[] = [];
  for (let row = 0; row < ZONE_ROWS; row++) {
    for (let column = 0; column < ZONE_COLUMNS; column++) {
      const zone = row * ZONE_COLUMNS + column;
      const bottomLeft = corners[row][column];
      const bottomRight = corners[row][column + 1];
      const topRight = corners[row + 1][column + 1];
      const topLeft = corners[row + 1][column];
      const ring = [
        bottomLeft,
        midpoint(horizontal, row, column, bottomLeft, bottomRight, true),
        bottomRight,
        midpoint(vertical, row, column + 1, bottomRight, topRight, false),
        topRight,
        midpoint(horizontal, row + 1, column, topLeft, topRight, true),
        topLeft,
        midpoint(vertical, row, column, bottomLeft, topLeft, false)
      ];
      const centerX = ring.reduce((sum, point) => sum + point[0], 0) / ring.length;
      const centerY = ring.reduce((sum, point) => sum + point[1], 0) / ring.length;
      rings.push(Float32Array.from(ring.flat()));
      ring.forEach((point, index) => {
        const next = ring[(index + 1) % ring.length];
        starts.push(point[0], point[1]);
        ends.push(next[0], next[1]);
        zones.push(zone);
        triangles.push(centerX, centerY, point[0], point[1], next[0], next[1]);
        triangleZones.push(zone);
      });
    }
  }
  const outlineSegments = new Float32Array(starts.length * 2);
  for (let edge = 0; edge < zones.length; edge++) {
    outlineSegments.set(starts.slice(edge * 2, edge * 2 + 2), edge * 4);
    outlineSegments.set(ends.slice(edge * 2, edge * 2 + 2), edge * 4 + 2);
  }
  return {
    edgeStarts: Float32Array.from(starts),
    edgeEnds: Float32Array.from(ends),
    edgeZones: Uint32Array.from(zones),
    outlineSegments,
    triangles: Float32Array.from(triangles),
    triangleZones: Uint32Array.from(triangleZones),
    rings
  };
}

/** Returns the zone whose ring contains the point, or `NO_ZONE`. */
function findZone(grid: ZoneGrid, x: number, y: number): number {
  for (let zone = 0; zone < grid.rings.length; zone++) {
    const ring = grid.rings[zone];
    const vertexCount = ring.length / 2;
    let inside = false;
    for (let index = 0, previous = vertexCount - 1; index < vertexCount; previous = index++) {
      const x0 = ring[index * 2];
      const y0 = ring[index * 2 + 1];
      const x1 = ring[previous * 2];
      const y1 = ring[previous * 2 + 1];
      if (y0 > y !== y1 > y && x < ((x1 - x0) * (y - y0)) / (y1 - y0) + x0) inside = !inside;
    }
    if (inside) return zone;
  }
  return NO_ZONE;
}

/** Returns the `[low, high]` of the central 90 percent of a strided coordinate sample. */
function getCentralRange(values: Float32Array, offset: number): [number, number] {
  const sample: number[] = [];
  for (let index = offset; index < values.length; index += 2 * 5) sample.push(values[index]);
  sample.sort((a, b) => a - b);
  return [sample[Math.floor(sample.length * 0.05)], sample[Math.floor(sample.length * 0.95)]];
}

function formatDuration(seconds: number): string {
  if (seconds >= 3600) return `${(seconds / 3600).toFixed(1)} h`;
  if (seconds >= 60) return `${(seconds / 60).toFixed(1)} min`;
  return `${seconds.toFixed(0)} s`;
}

function formatClock(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`;
}

/** Enter and exit events of trips against zones, with per-zone dwell and visit statistics. */
export const zoneEventsMode: SpatialAnalysisModeDefinition = {
  id: 'zone-events',
  title: 'Zone events',
  contributors: ['GPUZoneEvents', 'GPUGroupStatistics'],
  description:
    'Taxi trips crossing zip-code-like zones: green pulses are enter events, orange exits, ' +
    'placed on the GPU at the interpolated crossing. Play or scrub the playhead; the zone fill ' +
    'shows dwell and visits per zone, click a zone for its numbers.',
  initialViewState: {longitude: -73.985, latitude: 40.735, zoom: 12},

  async create(context) {
    const trips = await context.data.getNewYorkTrips();
    context.signal.throwIfAborted();
    const {device} = context;
    const projection = new LocalMetricProjection(trips.origin);
    const resources = new SpatialAnalysisResources(device, 'zone-events');

    const trackCount = trips.vendors.length;
    const rowCount = trips.vertexTimestamps.length;
    const epoch = trips.timeRange[0];
    const duration = Math.max(1, trips.timeRange[1] - epoch);
    const rebasedTimestamps = new Float32Array(rowCount);
    for (let row = 0; row < rowCount; row++) {
      rebasedTimestamps[row] = trips.vertexTimestamps[row] - epoch;
    }
    const [minX, maxX] = getCentralRange(trips.vertexPositions, 0);
    const [minY, maxY] = getCentralRange(trips.vertexPositions, 1);
    const grid = createZoneGrid([minX, minY, maxX, maxY]);
    const edgeCount = grid.edgeZones.length;
    const cellCount = trackCount * ZONE_COUNT;
    const maximumCapacity = trackCount * MAXIMUM_EVENTS_PER_TRACK;

    let maxEventsPerTrack = 4;
    let metric: Metric = 'dwell';
    let playing = true;
    let playhead = 0;
    let speed = 60;
    let fadeSeconds = 60;
    let showAll = true;
    let selectedZone = NO_ZONE;
    let dirty = true;
    let destroyed = false;
    let compiled: CompiledGPUCommandGraph<void> | null = null;
    const metricMaximum: Record<Metric, number> = {dwell: 1, mean: 1, visitors: 1};
    let table = {
      keys: new Uint32Array(ZONE_COUNT),
      counts: new Uint32Array(ZONE_COUNT),
      sums: new Float32Array(ZONE_COUNT),
      means: new Float32Array(ZONE_COUNT)
    };

    const positionsBuffer = resources.createBuffer('positions', trips.vertexPositions);
    const timestampsBuffer = resources.createBuffer('timestamps', rebasedTimestamps);
    const offsetsBuffer = resources.createBuffer('track-offsets', trips.tripOffsets);
    const edgeStartsBuffer = resources.createBuffer('edge-starts', grid.edgeStarts);
    const edgeEndsBuffer = resources.createBuffer('edge-ends', grid.edgeEnds);
    const edgeZonesBuffer = resources.createBuffer('edge-zones', grid.edgeZones);
    const outlineBuffer = resources.createBuffer('outline', grid.outlineSegments);
    const triangleBuffer = resources.createBuffer('zone-triangles', grid.triangles);
    const selectedBuffer = resources.createBuffer('selected-outline', 8 * 16);
    const selectedPoint = resources.createBuffer('selected-point', 8);
    const eventTracks = resources.createBuffer('event-tracks', maximumCapacity * 4);
    const eventZones = resources.createBuffer('event-zones', maximumCapacity * 4);
    const eventTypes = resources.createBuffer('event-types', maximumCapacity * 4);
    const eventTimes = resources.createBuffer('event-times', maximumCapacity * 4);
    const eventRows = resources.createBuffer('event-rows', maximumCapacity * 4);
    const eventCount = resources.createBuffer('event-count', 4);
    const eventOverflow = resources.createBuffer('event-overflow', 4);
    const dwellTimes = resources.createBuffer('dwell-times', cellCount * 4);
    const visitCounts = resources.createBuffer('visit-counts', cellCount * 4);
    const trackEventCounts = resources.createBuffer('track-event-counts', trackCount * 4);
    const tableKeys = resources.createBuffer('table-keys', ZONE_COUNT * 4);
    const tableCounts = resources.createBuffer('table-counts', ZONE_COUNT * 4);
    const tableCount = resources.createBuffer('table-count', 4);
    const tableOverflow = resources.createBuffer('table-overflow', 4);
    const tableSums = resources.createBuffer('table-sums', ZONE_COUNT * 4);
    const tableMeans = resources.createBuffer('table-means', ZONE_COUNT * 4);
    const tableMaximums = resources.createBuffer('table-maximums', ZONE_COUNT * 4);
    const playheadParameters = resources.createParameterBuffer('playhead', 'float32', 4);
    const choroplethParameters = resources.createParameterBuffer('choropleth', 'float32', 4);
    const drawCommands = resources.track(
      new DrawCommandBuffer(device, {
        id: 'zone-events-draw',
        type: 'draw',
        commands: [
          {vertexCount: 6, instanceCount: 0},
          {vertexCount: grid.triangleZones.length * 3, instanceCount: 1}
        ]
      })
    );

    function buildGraph(): void {
      if (compiled) resources.release(compiled);
      const eventCapacity = trackCount * maxEventsPerTrack;
      const graph = new GPUCommandGraph<void>(device, {id: `zone-events-${maxEventsPerTrack}`});
      const view = <Format extends 'uint32' | 'float32'>(
        name: string,
        buffer: typeof eventCount,
        format: Format,
        length: number
      ) => importGraphBuffer(graph, name, buffer, format, length);
      addFleetDwellZoneEventsRecipe(graph, {
        id: 'zone-events',
        positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', rowCount),
        timestamps: view('timestamps', timestampsBuffer, 'float32', rowCount),
        trackOffsets: view('track-offsets', offsetsBuffer, 'uint32', trackCount + 1),
        edgeStarts: importGraphBuffer(
          graph,
          'edge-starts',
          edgeStartsBuffer,
          'float32x2',
          edgeCount
        ),
        edgeEnds: importGraphBuffer(graph, 'edge-ends', edgeEndsBuffer, 'float32x2', edgeCount),
        edgeZones: view('edge-zones', edgeZonesBuffer, 'uint32', edgeCount),
        zoneCount: ZONE_COUNT,
        candidateCapacity: CANDIDATE_CAPACITY,
        maxEventsPerTrack,
        eventCapacity,
        events: {
          output: {
            ids: view('event-tracks', eventTracks, 'uint32', eventCapacity),
            count: view('event-count', eventCount, 'uint32', 1),
            overflow: view('event-overflow', eventOverflow, 'uint32', 1)
          },
          eventZones: view('event-zones', eventZones, 'uint32', eventCapacity),
          eventTypes: view('event-types', eventTypes, 'uint32', eventCapacity),
          eventTimes: view('event-times', eventTimes, 'float32', eventCapacity),
          eventRows: view('event-rows', eventRows, 'uint32', eventCapacity)
        },
        dwellTimes: view('dwell-times', dwellTimes, 'float32', cellCount),
        visitCounts: view('visit-counts', visitCounts, 'uint32', cellCount),
        trackEventCounts: view('track-event-counts', trackEventCounts, 'uint32', trackCount),
        table: {
          keys: view('table-keys', tableKeys, 'uint32', ZONE_COUNT),
          counts: view('table-counts', tableCounts, 'uint32', ZONE_COUNT),
          count: view('table-count', tableCount, 'uint32', 1),
          overflow: view('table-overflow', tableOverflow, 'uint32', 1),
          sumValues: view('table-sums', tableSums, 'float32', ZONE_COUNT),
          means: view('table-means', tableMeans, 'float32', ZONE_COUNT),
          maximums: view('table-maximums', tableMaximums, 'float32', ZONE_COUNT)
        }
      });
      compiled = resources.track(graph.compile());
      dirty = true;
    }

    const writeChoropleth = () => {
      choroplethParameters.write(
        Float32Array.of(METRIC_INDEX[metric], metricMaximum[metric], 0, 0)
      );
    };
    const describeZone = (zone: number): string => {
      if (zone === NO_ZONE) return 'none';
      const row = table.keys.indexOf(zone);
      const name = `Zone ${zone + 1}`;
      if (row < 0 || table.counts[row] === 0) return `${name}: no visits`;
      return (
        `${name}: ${formatCount(table.counts[row])} visiting trips, ` +
        `${formatDuration(table.sums[row])} total, ${formatDuration(table.means[row])} mean dwell`
      );
    };
    const selectedReadout = context.controls.addReadout('Selected zone', 'none');
    const writeSelection = () => {
      if (selectedZone === NO_ZONE) {
        selectedBuffer.write(new Float32Array(8 * 4).fill(Number.NaN));
        selectedReadout.setValue('none');
        return;
      }
      const ring = grid.rings[selectedZone];
      const segments = new Float32Array(8 * 4);
      for (let edge = 0; edge < 8; edge++) {
        const next = (edge + 1) % 8;
        segments.set(
          [ring[edge * 2], ring[edge * 2 + 1], ring[next * 2], ring[next * 2 + 1]],
          edge * 4
        );
      }
      selectedBuffer.write(segments);
      selectedReadout.setValue(describeZone(selectedZone));
    };

    context.controls.addSelect<Metric>({
      label: 'Zone fill',
      options: [
        {value: 'dwell', label: 'Total dwell time (all trips)'},
        {value: 'mean', label: 'Mean dwell per visiting trip'},
        {value: 'visitors', label: 'Visiting trips'}
      ],
      value: metric,
      onChange: value => {
        metric = value;
        writeChoropleth();
      }
    });
    context.controls.addToggle({
      label: 'Play',
      value: playing,
      onChange: value => {
        playing = value;
      }
    });
    const playheadSlider = context.controls.addSlider({
      label: 'Playhead',
      min: 0,
      max: Math.round(duration),
      step: 1,
      value: 0,
      format: formatClock,
      onChange: value => {
        playhead = value;
      }
    });
    context.controls.addSlider({
      label: 'Playback speed',
      min: 10,
      max: 300,
      step: 10,
      value: speed,
      format: value => `${value}x`,
      onChange: value => {
        speed = value;
      }
    });
    context.controls.addSlider({
      label: 'Event pulse duration (playhead seconds)',
      min: 10,
      max: 300,
      step: 10,
      value: fadeSeconds,
      format: value => `${value} s`,
      onChange: value => {
        fadeSeconds = value;
      }
    });
    context.controls.addToggle({
      label: 'Show every event as a faint dot',
      value: showAll,
      onChange: value => {
        showAll = value;
      }
    });
    context.controls.addSelect<string>({
      label: 'Events kept per trip (compile-time, rebuilds the graph)',
      options: [2, 4, 8].map(value => ({value: String(value), label: `${value} events`})),
      value: String(maxEventsPerTrack),
      onChange: value => {
        maxEventsPerTrack = Number(value);
        buildGraph();
        context.updateLayers();
      }
    });
    context.controls.addLegend({
      title: 'Zone fill: viridis from none to the busiest zone (sqrt scale)',
      gradient: {
        colors: [
          [68, 1, 84],
          [59, 82, 139],
          [33, 145, 140],
          [94, 201, 98],
          [253, 231, 37]
        ],
        minimumLabel: 'low',
        maximumLabel: 'high'
      }
    });
    context.controls.addLegend({
      title: 'Events',
      entries: [
        {color: [77, 230, 140, 255], label: 'Enter a zone'},
        {color: [255, 115, 64, 255], label: 'Exit a zone'}
      ]
    });
    context.controls.addNote(
      'Click a zone to read its visits and dwell. Events are computed once.'
    );
    context.controls.addReadout('Trips', formatCount(trackCount));
    context.controls.addReadout('Zones', `${ZONE_COUNT} (${formatCount(edgeCount)} edges)`);
    const eventReadout = context.controls.addReadout('Events');
    const visitReadout = context.controls.addReadout('Zone visits / total dwell');
    const clockReadout = context.controls.addReadout('Playhead');
    context.controls.addReadout('Data', `${trips.attribution}; zones synthetic jittered grid`);

    const summary = new SummaryReader(
      resources,
      'zone-events',
      [
        {buffer: eventCount, size: 4},
        {buffer: eventOverflow, size: 4},
        {buffer: tableCount, size: 4},
        {buffer: tableOverflow, size: 4},
        {buffer: tableKeys, size: ZONE_COUNT * 4},
        {buffer: tableCounts, size: ZONE_COUNT * 4},
        {buffer: tableSums, size: ZONE_COUNT * 4},
        {buffer: tableMeans, size: ZONE_COUNT * 4}
      ],
      bytes => {
        if (destroyed) return;
        const words = new Uint32Array(bytes);
        const floats = new Float32Array(bytes);
        const zone = (index: number) => 4 + index * ZONE_COUNT;
        table = {
          keys: words.slice(zone(0), zone(1)),
          counts: words.slice(zone(1), zone(2)),
          sums: floats.slice(zone(2), zone(3)),
          means: floats.slice(zone(3), zone(4))
        };
        const occupied = Math.min(words[2], ZONE_COUNT);
        let visits = 0;
        let dwell = 0;
        const maxima = {dwell: 0, mean: 0, visitors: 0};
        for (let row = 0; row < occupied; row++) {
          visits += table.counts[row];
          dwell += table.sums[row];
          maxima.dwell = Math.max(maxima.dwell, table.sums[row]);
          maxima.mean = Math.max(maxima.mean, table.means[row]);
          maxima.visitors = Math.max(maxima.visitors, table.counts[row]);
        }
        Object.assign(metricMaximum, {
          dwell: Math.max(maxima.dwell, 1),
          mean: Math.max(maxima.mean, 1),
          visitors: Math.max(maxima.visitors, 1)
        });
        writeChoropleth();
        eventReadout.setValue(
          `${formatCount(words[0])} kept (${maxEventsPerTrack} per trip max)` +
            `${words[1] ? ', OVERFLOW (a trip exceeds the cap, or candidates ran out)' : ', no overflow'}`
        );
        visitReadout.setValue(
          `${formatCount(visits)} (${occupied} of ${ZONE_COUNT} zones) / ${formatDuration(dwell)}` +
            `${words[3] ? ' (table OVERFLOW)' : ''}`
        );
        writeSelection();
      }
    );

    writeChoropleth();
    writeSelection();
    buildGraph();

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => (compiled ? [compiled] : []),
      encode(commandEncoder: CommandEncoder, frame) {
        if (playing) {
          playhead = (playhead + frame.deltaSeconds * speed) % duration;
          playheadSlider.setValue(Math.round(playhead));
        }
        playheadParameters.write(Float32Array.of(playhead, fadeSeconds, showAll ? 1 : 0, 0));
        clockReadout.setValue(`${formatClock(playhead)} of ${formatClock(duration)}`);
        if (dirty && compiled) {
          // Events and statistics depend only on the trips and zones: encode once per build.
          compiled.encode(commandEncoder, {parameters: undefined});
          commandEncoder.copyBufferToBuffer({
            sourceBuffer: eventCount,
            destinationBuffer: drawCommands.buffer,
            destinationOffset: RECORD_MARKERS * RECORD_BYTE_LENGTH + 4,
            size: 4
          });
          dirty = false;
          summary.markStale();
        }
        summary.flush(commandEncoder);
      },
      getLayers() {
        const coordinateOrigin: [number, number, number] = [trips.origin[0], trips.origin[1], 0];
        const layers: Layer[] = [
          new ZoneChoroplethLayer({
            id: 'zone-events-fill',
            coordinateOrigin,
            gridSize: [1, 1],
            bounds: [0, 0, 1, 1],
            triangles: triangleBuffer,
            tableKeys,
            tableSums,
            tableCounts,
            extent: choroplethParameters.buffer,
            drawCommands,
            drawCommandIndex: RECORD_ZONES
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'zone-events-outline',
            coordinateOrigin,
            segments: outlineBuffer,
            instanceCount: edgeCount,
            widthPixels: 1.2,
            color: [255, 255, 255, 120]
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'zone-events-selected',
            coordinateOrigin,
            segments: selectedBuffer,
            instanceCount: 8,
            widthPixels: 3.5,
            color: [255, 255, 255, 255]
          }),
          new ZoneEventMarkerLayer({
            id: 'zone-events-markers',
            coordinateOrigin,
            positions: positionsBuffer,
            timestamps: timestampsBuffer,
            trackOffsets: offsetsBuffer,
            eventTracks,
            eventRows,
            eventTimes,
            eventTypes,
            extent: playheadParameters.buffer,
            radiusPixels: 2.6,
            drawCommands,
            drawCommandIndex: RECORD_MARKERS
          })
        ];
        if (selectedZone !== NO_ZONE) {
          layers.push(
            new SpatialAnalysisPointLayer({
              id: 'zone-events-selected-point',
              coordinateOrigin,
              positions: selectedPoint,
              instanceCount: 1,
              radiusPixels: 4,
              color: [255, 255, 255, 255]
            })
          );
        }
        return layers;
      },
      onClick(event) {
        if (!event.coordinate) return false;
        const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
        const zone = findZone(grid, x, y);
        selectedZone = zone === selectedZone ? NO_ZONE : zone;
        if (selectedZone !== NO_ZONE) selectedPoint.write(Float32Array.of(x, y));
        writeSelection();
        context.updateLayers();
        return true;
      },
      getTooltip(event) {
        if (!event.coordinate) return null;
        const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
        const zone = findZone(grid, x, y);
        return zone === NO_ZONE ? null : describeZone(zone);
      },
      destroy() {
        destroyed = true;
        summary.stop();
        resources.destroy();
      }
    };
    return instance;
  }
};
