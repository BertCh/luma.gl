// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPULineDensityParameterValues,
  GPU_LINE_DENSITY_PARAMETER_LENGTH,
  GPU_ZONE_EVENT_TYPE,
  GPULineDensity,
  GPUZoneEvents
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisSegmentLayer} from '../../engine/layers';
import type {RampName} from '../../engine/ramps';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {SatelliteCellLayer} from './satellite-layers';
import {
  createTrackSubset,
  loadSatelliteTracks,
  SATELLITE_DATASET_ID,
  SATELLITE_GROUPS,
  selectTracksByGroup,
  type SatelliteTrackSet,
  type TrackSubset
} from './satellite-tracks';

/** Option state of the ground-track scene. */
export type SatelliteGroundTracksOptions = {
  group: string;
  ramp: 'inferno' | 'magma' | 'viridis' | 'cividis';
  scale: 'sqrt' | 'linear';
  showTracks: boolean;
  trackOpacity: number;
  showBand: boolean;
  band: number;
  perSatellite: boolean;
};

/** Grid of the density map: 2 degree cells from 180 W to 180 E and 85 S to 85 N. */
export const DENSITY_GRID = {minLongitude: -180, minLatitude: -85, cell: 2, columns: 180, rows: 85};
/** Latitude bands of the pass counter: 5 degrees from 85 S to 85 N. */
export const BAND_HEIGHT = 5;
export const BAND_COUNT = 34;
const BAND_MIN_LATITUDE = -85;
const EVENT_CAPACITY = 1 << 18;
const MAX_EVENTS_PER_TRACK = 128;
const CANDIDATE_CAPACITY = 1 << 19;

/** Results read back once per family: the graphs are static, so they run once. */
type GroundResult = {
  densities: Float32Array;
  totalLengthKm: number;
  maximumDensity: number;
  maximumDensityLatitude: number;
  touchedFraction: number;
  /** Enter events per latitude band. */
  enters: Float64Array;
  eventCount: number;
  candidateCount: number;
  densityOverflow: boolean;
  eventOverflow: boolean;
  trackOverflow: boolean;
};

type GroundGraph = {
  key: string;
  subset: TrackSubset;
  satelliteCount: number;
  compiled: CompiledGPUCommandGraph<void>;
  densities: Buffer;
  reader: SummaryReader;
  encoded: boolean;
  result: GroundResult | null;
};

export function formatLatitude(latitude: number): string {
  return `${Math.abs(latitude).toFixed(Number.isInteger(latitude) ? 0 : 1)}°${latitude < 0 ? 'S' : 'N'}`;
}

/**
 * Ground-track density and pass counts. `GPULineDensity` (spherical) sums the track length per
 * 2-degree cell; `GPUZoneEvents` against 34 latitude-band polygons counts every entry into a band.
 * Both graphs are static (the tracks do not change), so each family compiles one graph, encodes it
 * once and reads the summary back once.
 */
export async function createSatelliteGroundTracks(
  ctx: SceneContext<SatelliteGroundTracksOptions>
): Promise<SceneInstance<SatelliteGroundTracksOptions>> {
  const tracks = loadSatelliteTracks(ctx.datasets.get(SATELLITE_DATASET_ID));
  const resources = new SpatialAnalysisResources(ctx.device, 'satellite-ground');
  const hours = tracks.durationSeconds / 3600;
  const cellCount = DENSITY_GRID.columns * DENSITY_GRID.rows;
  let destroyed = false;

  // Latitude bands as rectangles wider than the world, so only their horizontal edges are crossed.
  const edgeStarts = new Float32Array(BAND_COUNT * 4 * 2);
  const edgeEnds = new Float32Array(BAND_COUNT * 4 * 2);
  const edgeZones = new Uint32Array(BAND_COUNT * 4);
  for (let band = 0; band < BAND_COUNT; band++) {
    const south = BAND_MIN_LATITUDE + band * BAND_HEIGHT;
    const north = south + BAND_HEIGHT;
    const west = -190;
    const east = 190;
    const corners = [
      [west, south],
      [east, south],
      [east, north],
      [west, north]
    ];
    for (let edge = 0; edge < 4; edge++) {
      const row = band * 4 + edge;
      edgeStarts.set(corners[edge], row * 2);
      edgeEnds.set(corners[(edge + 1) % 4], row * 2);
      edgeZones[row] = band;
    }
  }
  const edgeStartBuffer = resources.createBuffer('edge-starts', edgeStarts);
  const edgeEndBuffer = resources.createBuffer('edge-ends', edgeEnds);
  const edgeZoneBuffer = resources.createBuffer('edge-zones', edgeZones);
  const densityParameters = resources.createParameterBuffer(
    'density-grid',
    'float32',
    GPU_LINE_DENSITY_PARAMETER_LENGTH,
    getGPULineDensityParameterValues({
      minX: DENSITY_GRID.minLongitude,
      minY: DENSITY_GRID.minLatitude,
      cellWidth: DENSITY_GRID.cell,
      cellHeight: DENSITY_GRID.cell
    })
  );

  // ---- Backdrop segments (all tracks, masked by family) -----------------------------------------
  const segmentsBuffer = resources.createBuffer('segments', tracks.segments);
  const segmentTracksBuffer = resources.createBuffer('segment-tracks', tracks.segmentTracks);
  const trackMaskBuffer = resources.createBuffer(
    'track-mask',
    new Uint32Array(tracks.trackCount).fill(1)
  );
  const bandValueBuffer = resources.createBuffer('band-value', new Float32Array([1]));

  const groupKey = (value: string) => (value === 'all' ? null : Number(value));

  function writeTrackMask(): void {
    const group = groupKey(ctx.options.group);
    const mask = new Uint32Array(tracks.trackCount);
    for (let track = 0; track < tracks.trackCount; track++) {
      mask[track] = group === null || tracks.group[track] === group ? 1 : 0;
    }
    trackMaskBuffer.write(mask);
  }
  writeTrackMask();

  // ---- One graph per family ---------------------------------------------------------------------
  const graphs = new Map<string, GroundGraph>();
  let active: GroundGraph;

  function getGraph(key: string): GroundGraph {
    const existing = graphs.get(key);
    if (existing) return existing;
    const group = groupKey(key);
    const subset = createTrackSubset(
      tracks,
      selectTracksByGroup(tracks, group === null ? null : [group])
    );
    const satelliteCount = new Set(subset.satelliteIndex).size;
    const prefix = `ground-${key}`;
    const positions = resources.createBuffer(`${prefix}-positions`, subset.positions);
    const timestamps = resources.createBuffer(`${prefix}-timestamps`, subset.timestamps);
    const offsets = resources.createBuffer(`${prefix}-offsets`, subset.offsets);
    const lengths = resources.createBuffer(`${prefix}-lengths`, cellCount * 4);
    const densities = resources.createBuffer(`${prefix}-densities`, cellCount * 4);
    const densityOverflow = resources.createBuffer(`${prefix}-density-overflow`, 4);
    const densityTotal = resources.createBuffer(`${prefix}-density-total`, 4);
    const eventIds = resources.createBuffer(`${prefix}-event-ids`, EVENT_CAPACITY * 4);
    const eventCount = resources.createBuffer(`${prefix}-event-count`, 4);
    const eventOverflow = resources.createBuffer(`${prefix}-event-overflow`, 4);
    const eventZones = resources.createBuffer(`${prefix}-event-zones`, EVENT_CAPACITY * 4);
    const eventTypes = resources.createBuffer(`${prefix}-event-types`, EVENT_CAPACITY * 4);
    const candidateCount = resources.createBuffer(`${prefix}-candidate-count`, 4);
    const candidateOverflow = resources.createBuffer(`${prefix}-candidate-overflow`, 4);
    const trackOverflow = resources.createBuffer(`${prefix}-track-overflow`, 4);
    const eventOverflowDiagnostic = resources.createBuffer(`${prefix}-event-diagnostic`, 4);

    const graph = new GPUCommandGraph<void>(ctx.device, {id: `satellite-ground-${key}`});
    const positionsView = importGraphBuffer(
      graph,
      'positions',
      positions,
      'float32x2',
      subset.vertexCount
    );
    const offsetsView = importGraphBuffer(
      graph,
      'offsets',
      offsets,
      'uint32',
      subset.trackCount + 1
    );
    graph.add(
      new GPULineDensity({
        id: 'density',
        positions: positionsView,
        pathOffsets: offsetsView,
        columns: DENSITY_GRID.columns,
        rows: DENSITY_GRID.rows,
        coordinateSystem: 'spherical',
        maximumRecords: Math.max(1024, subset.vertexCount * 2),
        parameters: densityParameters.importToGraph(graph),
        output: {
          lengths: importGraphBuffer(graph, 'lengths', lengths, 'float32', cellCount),
          densities: importGraphBuffer(graph, 'densities', densities, 'float32', cellCount),
          overflow: importGraphBuffer(graph, 'density-overflow', densityOverflow, 'uint32', 1),
          totalRecords: importGraphBuffer(graph, 'density-total', densityTotal, 'uint32', 1)
        }
      })
    );
    graph.add(
      new GPUZoneEvents({
        id: 'bands',
        positions: positionsView,
        timestamps: importGraphBuffer(
          graph,
          'timestamps',
          timestamps,
          'float32',
          subset.vertexCount
        ),
        trackOffsets: offsetsView,
        edgeStarts: importGraphBuffer(
          graph,
          'edge-starts',
          edgeStartBuffer,
          'float32x2',
          edgeZones.length
        ),
        edgeEnds: importGraphBuffer(
          graph,
          'edge-ends',
          edgeEndBuffer,
          'float32x2',
          edgeZones.length
        ),
        edgeZones: importGraphBuffer(
          graph,
          'edge-zones',
          edgeZoneBuffer,
          'uint32',
          edgeZones.length
        ),
        zoneCount: BAND_COUNT,
        candidateCapacity: CANDIDATE_CAPACITY,
        maxEventsPerTrack: MAX_EVENTS_PER_TRACK,
        events: {
          output: {
            ids: importGraphBuffer(graph, 'event-ids', eventIds, 'uint32', EVENT_CAPACITY),
            count: importGraphBuffer(graph, 'event-count', eventCount, 'uint32', 1),
            overflow: importGraphBuffer(graph, 'event-overflow', eventOverflow, 'uint32', 1)
          },
          eventZones: importGraphBuffer(graph, 'event-zones', eventZones, 'uint32', EVENT_CAPACITY),
          eventTypes: importGraphBuffer(graph, 'event-types', eventTypes, 'uint32', EVENT_CAPACITY)
        },
        diagnostics: {
          candidateCount: importGraphBuffer(graph, 'candidate-count', candidateCount, 'uint32', 1),
          candidateOverflow: importGraphBuffer(
            graph,
            'candidate-overflow',
            candidateOverflow,
            'uint32',
            1
          ),
          trackOverflow: importGraphBuffer(graph, 'track-overflow', trackOverflow, 'uint32', 1),
          eventOverflow: importGraphBuffer(
            graph,
            'event-diagnostic',
            eventOverflowDiagnostic,
            'uint32',
            1
          )
        }
      })
    );
    const compiled = resources.track(graph.compile());

    const reader = new SummaryReader(
      resources,
      `ground-${key}`,
      [
        {buffer: densityOverflow, size: 4},
        {buffer: densityTotal, size: 4},
        {buffer: eventCount, size: 4},
        {buffer: eventOverflow, size: 4},
        {buffer: candidateCount, size: 4},
        {buffer: candidateOverflow, size: 4},
        {buffer: trackOverflow, size: 4},
        {buffer: eventOverflowDiagnostic, size: 4},
        {buffer: lengths, size: cellCount * 4},
        {buffer: densities, size: cellCount * 4},
        {buffer: eventZones, size: EVENT_CAPACITY * 4},
        {buffer: eventTypes, size: EVENT_CAPACITY * 4}
      ],
      bytes => {
        if (destroyed) return;
        const words = new Uint32Array(bytes);
        const floats = new Float32Array(bytes);
        const lengthStart = 8;
        const densityStart = lengthStart + cellCount;
        const zoneStart = densityStart + cellCount;
        const typeStart = zoneStart + EVENT_CAPACITY;
        const kept = Math.min(words[2], EVENT_CAPACITY);
        const enters = new Float64Array(BAND_COUNT);
        for (let event = 0; event < kept; event++) {
          const zone = words[zoneStart + event];
          if (words[typeStart + event] === GPU_ZONE_EVENT_TYPE.enter && zone < BAND_COUNT) {
            enters[zone]++;
          }
        }
        let totalLength = 0;
        let touched = 0;
        for (let cell = 0; cell < cellCount; cell++) {
          const length = floats[lengthStart + cell];
          totalLength += length;
          if (length > 0) touched++;
        }
        const cellDensities = floats.slice(densityStart, zoneStart);
        let maximum = 0;
        let maximumCell = 0;
        for (let cell = 0; cell < cellCount; cell++) {
          if (cellDensities[cell] > maximum) {
            maximum = cellDensities[cell];
            maximumCell = cell;
          }
        }
        graph_.result = {
          densities: cellDensities,
          totalLengthKm: totalLength / 1000,
          maximumDensity: maximum,
          maximumDensityLatitude:
            DENSITY_GRID.minLatitude +
            (Math.floor(maximumCell / DENSITY_GRID.columns) + 0.5) * DENSITY_GRID.cell,
          touchedFraction: touched / cellCount,
          enters,
          eventCount: words[2],
          candidateCount: words[4],
          densityOverflow: words[0] !== 0,
          eventOverflow: words[3] !== 0 || words[7] !== 0,
          trackOverflow: words[6] !== 0
        };
        if (graph_ === active) applyResult();
      }
    );
    const graph_: GroundGraph = {
      key,
      subset,
      satelliteCount,
      compiled,
      densities,
      reader,
      encoded: false,
      result: null
    };
    graphs.set(key, graph_);
    return graph_;
  }

  active = getGraph(ctx.options.group);
  ctx.setReadout(
    'tracks',
    `${formatCount(tracks.satelliteCount)} satellites, ${formatCount(tracks.vertexCount)} positions`
  );

  // ---- Results to readouts, legend and chart -----------------------------------------------------
  const valueScale = 1e6 / hours;

  function bandIndex(latitude: number): number {
    return Math.min(
      BAND_COUNT - 1,
      Math.max(0, Math.floor((latitude - BAND_MIN_LATITUDE) / BAND_HEIGHT))
    );
  }

  function applyResult(): void {
    const result = active.result;
    if (!result) return;
    const options = ctx.options;
    ctx.setLegendExtent('density', [0, result.maximumDensity * valueScale]);
    ctx.setReadout('trackLength', `${formatCount(Math.round(result.totalLengthKm))} km`);
    ctx.setReadout('touched', `${(result.touchedFraction * 100).toFixed(0)}% of cells`);
    ctx.setReadout(
      'peak',
      `${(result.maximumDensity * valueScale).toFixed(1)} km per 1,000 km² per h near ${formatLatitude(result.maximumDensityLatitude)}`
    );
    ctx.setReadout(
      'events',
      `${formatCount(result.eventCount)} band crossings, ${formatCount(result.candidateCount)} candidates${
        result.eventOverflow || result.densityOverflow || result.trackOverflow ? ' (OVERFLOW)' : ''
      }`
    );
    const divisor = hours * (options.perSatellite ? Math.max(1, active.satelliteCount) : 1);
    const centers = Array.from(
      {length: BAND_COUNT},
      (_, band) => BAND_MIN_LATITUDE + (band + 0.5) * BAND_HEIGHT
    );
    const passes = Float64Array.from(result.enters, count => count / divisor);
    const bandCenter = options.band;
    ctx.setChart('passChart', {
      kind: 'line',
      xLabel: 'latitude (degrees north)',
      yLabel: options.perSatellite ? 'passes per satellite per hour' : 'passes per hour',
      xDomain: [-85, 85],
      height: 130,
      series: [
        {
          label: SATELLITE_GROUPS[Number(active.key)] ?? 'All satellites',
          x: centers,
          y: passes,
          area: true
        }
      ],
      markers: options.showBand ? [{x: bandCenter, label: formatLatitude(bandCenter)}] : [],
      formatX: value => `${Math.round(value)}`,
      formatY: value => (value >= 10 ? value.toFixed(0) : value.toFixed(1)),
      description:
        'Entries into each 5-degree latitude band per hour, counted by GPUZoneEvents. One pass is one entry (ascending or descending).'
    });
    const band = bandIndex(bandCenter);
    const perHour = result.enters[band] / hours;
    ctx.setReadout(
      'bandPasses',
      `${perHour.toFixed(1)} per hour in the ${formatLatitude(BAND_MIN_LATITUDE + band * BAND_HEIGHT)} to ${formatLatitude(BAND_MIN_LATITUDE + (band + 1) * BAND_HEIGHT)} band`
    );
    ctx.setReadout(
      'bandGap',
      perHour > 0
        ? `${formatGap(3600 / perHour)} between passes of any satellite${
            active.satelliteCount > 0
              ? `; one satellite every ${formatGap((3600 * active.satelliteCount) / perHour)}`
              : ''
          }`
        : 'no pass in 3 h'
    );
    ctx.refreshTooltip();
  }

  function formatGap(seconds: number): string {
    if (seconds < 90) return `${seconds.toFixed(0)} s`;
    if (seconds < 5400) return `${(seconds / 60).toFixed(0)} min`;
    return `${(seconds / 3600).toFixed(1)} h`;
  }

  return {
    getCompiledGraphs: () => [active.compiled],

    setOption(id, _value, state) {
      switch (id) {
        case 'group':
          active = getGraph(state.group);
          writeTrackMask();
          if (active.result) applyResult();
          ctx.requestLayers();
          break;
        case 'band':
        case 'perSatellite':
        case 'showBand':
          applyResult();
          ctx.requestLayers();
          break;
        default:
          ctx.requestLayers();
      }
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    encode(commandEncoder) {
      if (!active.encoded) {
        active.compiled.encode(commandEncoder, {parameters: undefined});
        active.encoded = true;
        active.reader.request(commandEncoder);
      } else {
        active.reader.flush(commandEncoder);
      }
    },

    getLayers() {
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const result = active.result;
      const layers: Layer[] = [
        new SatelliteCellLayer({
          id: 'satellite-density',
          values: active.densities,
          gridSize: [DENSITY_GRID.columns, DENSITY_GRID.rows],
          grid: [
            DENSITY_GRID.minLongitude,
            DENSITY_GRID.minLatitude,
            DENSITY_GRID.cell,
            DENSITY_GRID.cell
          ],
          colormap: options.ramp as RampName,
          valueRange: [0, (result?.maximumDensity ?? 1) * valueScale],
          valueScale,
          sqrtScale: options.scale === 'sqrt',
          discardAtOrBelow: 0,
          opacity: 0.88
        })
      ];
      if (options.showBand) {
        const south = BAND_MIN_LATITUDE + bandIndex(options.band) * BAND_HEIGHT;
        layers.push(
          new SatelliteCellLayer({
            id: 'satellite-band',
            values: bandValueBuffer,
            gridSize: [1, 1],
            grid: [-180, south, 360, BAND_HEIGHT],
            color: dark ? [255, 255, 255, 46] : [20, 30, 60, 46],
            opacity: 1
          })
        );
      }
      if (options.showTracks) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'satellite-ground-lines',
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            segments: segmentsBuffer,
            instanceCount: tracks.segmentCount,
            values: trackMaskBuffer,
            valueFormat: 'uint32',
            valueIndices: segmentTracksBuffer,
            colormap: 'mask',
            color: dark ? [220, 232, 255, 255] : [30, 40, 70, 255],
            noDataColor: [0, 0, 0, 0],
            widthPixels: 0.8,
            opacity: options.trackOpacity
          })
        );
      }
      return layers;
    },

    getTooltip(event) {
      const result = active.result;
      if (!result || !event.coordinate) return null;
      const [longitude, latitude] = event.coordinate;
      const column = Math.floor((longitude - DENSITY_GRID.minLongitude) / DENSITY_GRID.cell);
      const row = Math.floor((latitude - DENSITY_GRID.minLatitude) / DENSITY_GRID.cell);
      if (column < 0 || column >= DENSITY_GRID.columns || row < 0 || row >= DENSITY_GRID.rows) {
        return null;
      }
      const value = result.densities[row * DENSITY_GRID.columns + column] * valueScale;
      const band = bandIndex(latitude);
      return `${formatLatitude(latitude)}: ${value.toFixed(1)} km of track per 1,000 km² per hour, ${(result.enters[band] / hours).toFixed(1)} passes per hour in this band`;
    },

    onClick(event) {
      if (!event.coordinate) return false;
      const band = bandIndex(event.coordinate[1]);
      ctx.setOptions(
        {band: BAND_MIN_LATITUDE + (band + 0.5) * BAND_HEIGHT, showBand: true},
        {notify: true}
      );
      return true;
    },

    destroy() {
      destroyed = true;
      for (const graph of graphs.values()) graph.reader.stop();
      resources.destroy();
    }
  };
}

export type {SatelliteTrackSet};
