// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUTimeWindowParameterValues,
  GPU_TIME_WINDOW_PARAMETER_LENGTH,
  GPUTimeWindowFilter
} from '@luma.gl/experimental/gpu-dataframe';
import {
  getGPUTrajectoryPlayheadParameterValues,
  GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH,
  GPU_TRAJECTORY_PLAYHEAD_STATUS,
  GPUTrajectoryPlayhead
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {DrawCommandBuffer, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {createPlaybackClock} from '../../engine/playback';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {findNearestTrack} from './b12-tracks';
import {
  DAYS_IN_YEAR,
  describeMigrationTrack,
  formatYearDay,
  loadMigrationTracks,
  MIGRATION_DATASET_ID,
  MIGRATION_SPECIES_COLORS,
  MIGRATION_SPECIES_LABELS,
  SECONDS_PER_DAY
} from './migration-shared';
import {getMigrationSeasonComparison} from './migration-season-comparison';

/** Option state of the migration season scene. */
export type MigrationSeasonOptions = {
  play: boolean;
  day: number;
  playSpeed: number;
  loop: boolean;
  maxGapHours: number;
  markerSize: number;
  showTrails: boolean;
  trailDays: number;
  tailFade: number;
  trailOpacity: number;
  showFlyways: boolean;
  followSelected: boolean;
};

const STATUS_INTERVAL_FRAMES = 10;
const NO_TRACK = 0xffffffff;
/** Birds south of this latitude are counted as being in Africa. */
const AFRICA_LATITUDE = 35;

/**
 * Migration season: the whole population plays through one folded year. `GPUTrajectoryPlayhead`
 * interpolates every animal-year at the clock on the GPU (positions in longitude/latitude degrees,
 * which is fine for two-hour fixes), `GPUTimeWindowFilter` selects the trail segments of the last
 * days with a fade, and a small readback of the interpolated positions gives the readouts. The
 * chart of the share of birds south of 35 N is computed once from the tracks.
 */
export async function createMigrationSeason(
  ctx: SceneContext<MigrationSeasonOptions>
): Promise<SceneInstance<MigrationSeasonOptions>> {
  const tracks = loadMigrationTracks(ctx.datasets.get(MIGRATION_DATASET_ID));
  const {device} = ctx;
  const {trackCount, vertexCount, segmentCount} = tracks;
  const resources = new SpatialAnalysisResources(device, 'season');
  const drawProps = {coordinateSystem: COORDINATE_SYSTEM.LNGLAT} as const;
  const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    graph: GPUCommandGraph<void>,
    name: string,
    buffer: Buffer,
    format: Format,
    length: number
  ) => importGraphBuffer(graph, name, buffer, format, length);

  // ---- Static inputs ----------------------------------------------------------------------------
  const lngLatBuffer = resources.createBuffer('lng-lat', tracks.lngLat);
  const timestampsBuffer = resources.createBuffer('timestamps', tracks.timestamps);
  const offsetsBuffer = resources.createBuffer('offsets', tracks.offsets);
  const segmentsBuffer = resources.createBuffer('segments', tracks.segments);
  const segmentTracksBuffer = resources.createBuffer('segment-tracks', tracks.segmentTracks);
  const segmentStartTimesBuffer = resources.createBuffer(
    'segment-start-times',
    tracks.segmentStartTimes
  );
  const segmentEndTimesBuffer = resources.createBuffer('segment-end-times', tracks.segmentEndTimes);
  const speciesBuffer = resources.createBuffer('species', Uint32Array.from(tracks.species));

  // ---- Playhead graph ---------------------------------------------------------------------------
  const currentPositions = resources.createBuffer('current-positions', trackCount * 8);
  const status = resources.createBuffer('status', trackCount * 4);
  const activeIds = resources.createBuffer('active-ids', trackCount * 4);
  const activeCount = resources.createBuffer('active-count', 4);
  const activeOverflow = resources.createBuffer('active-overflow', 4);
  const playheadParameters = resources.createParameterBuffer(
    'playhead',
    'float32',
    GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH
  );
  const markerDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'season-marker-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const playheadGraph = new GPUCommandGraph<void>(device, {id: 'season-playhead'});
  playheadGraph.add(
    new GPUTrajectoryPlayhead({
      id: 'playhead',
      positions: view(playheadGraph, 'lng-lat', lngLatBuffer, 'float32x2', vertexCount),
      timestamps: view(playheadGraph, 'timestamps', timestampsBuffer, 'float32', vertexCount),
      trackOffsets: view(playheadGraph, 'offsets', offsetsBuffer, 'uint32', trackCount + 1),
      parameters: playheadParameters.importToGraph(playheadGraph),
      currentPositions: view(
        playheadGraph,
        'current-positions',
        currentPositions,
        'float32x2',
        trackCount
      ),
      status: view(playheadGraph, 'status', status, 'uint32', trackCount),
      activeTracks: {
        ids: view(playheadGraph, 'active-ids', activeIds, 'uint32', trackCount),
        count: view(playheadGraph, 'active-count', activeCount, 'uint32', 1),
        overflow: view(playheadGraph, 'active-overflow', activeOverflow, 'uint32', 1)
      },
      drawInstanceCount: playheadGraph.importGPUData(
        'marker-draw-count',
        markerDraw.getInstanceCountData(0)
      )
    })
  );
  const playheadCompiled = resources.track(playheadGraph.compile());

  // ---- Trail (time window) graph ----------------------------------------------------------------
  const trailIds = resources.createBuffer('trail-ids', segmentCount * 4);
  const trailCount = resources.createBuffer('trail-count', 4);
  const trailOverflow = resources.createBuffer('trail-overflow', 4);
  const fadeWeights = resources.createBuffer('fade-weights', segmentCount * 4);
  const clipFractions = resources.createBuffer('clip-fractions', segmentCount * 8);
  const windowParameters = resources.createParameterBuffer(
    'window',
    'float32',
    GPU_TIME_WINDOW_PARAMETER_LENGTH
  );
  const trailDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'season-trail-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const trailGraph = new GPUCommandGraph<void>(device, {id: 'season-trails'});
  trailGraph.add(
    new GPUTimeWindowFilter({
      id: 'trail-window',
      timestamps: view(
        trailGraph,
        'segment-start-times',
        segmentStartTimesBuffer,
        'float32',
        segmentCount
      ),
      endTimestamps: view(
        trailGraph,
        'segment-end-times',
        segmentEndTimesBuffer,
        'float32',
        segmentCount
      ),
      window: windowParameters.importToGraph(trailGraph),
      output: {
        ids: view(trailGraph, 'trail-ids', trailIds, 'uint32', segmentCount),
        count: view(trailGraph, 'trail-count', trailCount, 'uint32', 1),
        overflow: view(trailGraph, 'trail-overflow', trailOverflow, 'uint32', 1)
      },
      fadeWeights: view(trailGraph, 'fade-weights', fadeWeights, 'float32', segmentCount),
      clipFractions: view(trailGraph, 'clip-fractions', clipFractions, 'float32x2', segmentCount),
      drawInstanceCount: trailGraph.importGPUData(
        'trail-draw-count',
        trailDraw.getInstanceCountData(0)
      )
    })
  );
  const trailCompiled = resources.track(trailGraph.compile());

  // ---- Selected bird (CPU interpolation for the follow camera) ----------------------------------
  const selectedSegments = resources.createBuffer('selected-segments', tracks.longestTrack * 16);
  let selectedTrack = NO_TRACK;

  /** Position of a track at a time (seconds of the year) by binary search, or `null` outside it. */
  function positionAt(track: number, seconds: number): [number, number] | null {
    let low = tracks.offsets[track];
    let high = tracks.offsets[track + 1] - 1;
    if (high < low || seconds < tracks.timestamps[low] || seconds > tracks.timestamps[high]) {
      return null;
    }
    while (low < high) {
      const middle = (low + high + 1) >> 1;
      if (tracks.timestamps[middle] <= seconds) low = middle;
      else high = middle - 1;
    }
    const next = Math.min(low + 1, tracks.offsets[track + 1] - 1);
    const span = tracks.timestamps[next] - tracks.timestamps[low];
    const fraction = span > 0 ? (seconds - tracks.timestamps[low]) / span : 0;
    return [
      tracks.lngLat[low * 2] + (tracks.lngLat[next * 2] - tracks.lngLat[low * 2]) * fraction,
      tracks.lngLat[low * 2 + 1] +
        (tracks.lngLat[next * 2 + 1] - tracks.lngLat[low * 2 + 1]) * fraction
    ];
  }

  function writeSelection(): void {
    const segments = new Float32Array(tracks.longestTrack * 4).fill(Number.NaN);
    if (selectedTrack !== NO_TRACK) {
      const first = tracks.offsets[selectedTrack];
      const last = tracks.offsets[selectedTrack + 1] - 1;
      for (let vertex = first; vertex < last; vertex++) {
        segments.set(tracks.lngLat.subarray(vertex * 2, vertex * 2 + 4), (vertex - first) * 4);
      }
    }
    selectedSegments.write(segments);
  }

  function describeSelected(): void {
    ctx.setReadout(
      'selected',
      selectedTrack === NO_TRACK
        ? 'click a bird'
        : `${describeMigrationTrack(tracks, selectedTrack)}, ${tracks.coverageDays[selectedTrack].toFixed(0)} days of data`
    );
  }

  // ---- Static chart: share of birds south of 35 N, per species, per day ---------------------------
  const staticChartData = getMigrationSeasonComparison(tracks, AFRICA_LATITUDE);

  function setSeasonChart(day: number): void {
    ctx.setChart('africaChart', {
      kind: 'line',
      series: staticChartData.shares.map((y, species) => ({
        label: MIGRATION_SPECIES_LABELS[species],
        x: staticChartData.days,
        y,
        color: species
      })),
      xDomain: [0, DAYS_IN_YEAR],
      yDomain: [0, 100],
      markers: [{x: day, label: 'now'}],
      xLabel: 'day of the year',
      yLabel: `% south of ${AFRICA_LATITUDE} N`,
      height: 130,
      formatX: value => formatYearDay(Math.min(365, value)),
      formatY: value => `${Math.round(value)}`,
      description: `Share of each species' tagged birds that are south of ${AFRICA_LATITUDE} degrees North (Africa) on each day of the folded year. Marsh harriers and Montagu's harriers spend the winter there; spoonbills never go.`
    });
    const activePanel = staticChartData.panels.findIndex(
      panel => day >= panel.start && day < panel.end
    );
    ctx.setChart('seasonMultiples', {
      kind: 'multiples',
      columns: 2,
      shareDomains: false,
      highlight: Math.max(0, activePanel),
      titles: staticChartData.panels.map(panel => panel.label),
      description:
        'The same data split into the five folded-calendar seasons. The outlined panel contains the playhead, so the map, the full-year chart and this comparison always refer to the same day.',
      charts: staticChartData.panels.map(panel => ({
        kind: 'line' as const,
        series: staticChartData.shares.map((values, species) => ({
          label: MIGRATION_SPECIES_LABELS[species],
          x: staticChartData.days,
          y: values,
          color: species
        })),
        xDomain: [panel.start, panel.end],
        yDomain: [0, 100],
        markers: [{x: day, label: day >= panel.start && day < panel.end ? 'now' : ''}],
        xLabel: 'day',
        yLabel: '% south',
        height: 88,
        table: false,
        formatX: value => formatYearDay(Math.min(DAYS_IN_YEAR - 1, value)),
        formatY: value => `${Math.round(value)}`
      }))
    });
    ctx.setAnnotations('season-threshold', [
      {
        kind: 'line',
        id: 'africa-threshold',
        coordinates: [
          [-20, AFRICA_LATITUDE],
          [35, AFRICA_LATITUDE]
        ],
        text: `${AFRICA_LATITUDE} N counting rule`,
        dashed: true,
        tone: 'signal',
        priority: 4,
        minZoom: 2.6
      }
    ]);
  }

  // ---- State ------------------------------------------------------------------------------------
  let destroyed = false;
  let statusStale = true;
  let chartDay = -1;
  let snapshot: {
    active: number;
    status: Uint32Array;
    positions: Float32Array;
    trails: number;
  } | null = null;
  const clock = createPlaybackClock(
    ctx,
    {time: 'day', play: 'play', speed: 'playSpeed', loop: 'loop'},
    {range: [0, DAYS_IN_YEAR - 1], rate: 1, step: 1}
  );
  let playhead = ctx.options.day * SECONDS_PER_DAY;

  const reader = new SummaryReader(
    resources,
    'season-status',
    [
      {buffer: activeCount, size: 4},
      {buffer: trailCount, size: 4},
      {buffer: status, size: trackCount * 4},
      {buffer: currentPositions, size: trackCount * 8}
    ],
    bytes => {
      if (destroyed) return;
      const words = new Uint32Array(bytes);
      const floats = new Float32Array(bytes);
      snapshot = {
        active: words[0],
        trails: words[1],
        status: words.slice(2, 2 + trackCount),
        positions: floats.slice(2 + trackCount, 2 + trackCount * 3)
      };
      summarize();
    }
  );

  function summarize(): void {
    if (!snapshot) return;
    const latitudes: number[] = [];
    const speciesActive = [0, 0, 0];
    let south = 0;
    for (let track = 0; track < trackCount; track++) {
      if (snapshot.status[track] !== GPU_TRAJECTORY_PLAYHEAD_STATUS.active) continue;
      const latitude = snapshot.positions[track * 2 + 1];
      latitudes.push(latitude);
      speciesActive[tracks.species[track]]++;
      if (latitude < AFRICA_LATITUDE) south++;
    }
    latitudes.sort((a, b) => a - b);
    ctx.setReadout(
      'birds',
      `${snapshot.active} of ${trackCount} animal-years have data (${speciesActive.join(' + ')} by species)`
    );
    ctx.setReadout(
      'africa',
      latitudes.length
        ? `${south} of ${latitudes.length} birds south of ${AFRICA_LATITUDE} N (${Math.round((100 * south) / latitudes.length)}%)`
        : 'n/a'
    );
    ctx.setReadout(
      'latitude',
      latitudes.length
        ? `${latitudes[Math.floor(latitudes.length / 2)].toFixed(1)} N median, ${latitudes[0].toFixed(1)} to ${latitudes[latitudes.length - 1].toFixed(1)} N`
        : 'n/a'
    );
    ctx.setReadout('trailSegments', formatCount(snapshot.trails));
  }

  // Default selection: the longest track that spans most of the year.
  for (let track = 0; track < trackCount; track++) {
    if (tracks.coverageDays[track] < 300) continue;
    if (selectedTrack === NO_TRACK || tracks.pathMeters[track] > tracks.pathMeters[selectedTrack]) {
      selectedTrack = track;
    }
  }
  writeSelection();
  ctx.setReadout('tracks', `${trackCount} animal-years, ${formatCount(vertexCount)} fixes`);
  describeSelected();
  setSeasonChart(ctx.options.day);

  // ---- Instance ---------------------------------------------------------------------------------
  return {
    getCompiledGraphs: () => [playheadCompiled, trailCompiled],

    setOption(id) {
      switch (id) {
        case 'play':
        case 'day':
        case 'playSpeed':
        case 'loop':
        case 'maxGapHours':
        case 'trailDays':
        case 'tailFade':
          statusStale = true;
          break;
        default:
          ctx.requestLayers();
      }
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      const options = ctx.options;
      const day = clock.advance(frame);
      playhead = day * SECONDS_PER_DAY;
      ctx.setReadout(
        'clock',
        `${formatYearDay(day)} (day ${Math.floor(day) + 1} of the folded year)`
      );
      if (Math.floor(day) !== chartDay) {
        chartDay = Math.floor(day);
        setSeasonChart(day);
      }
      playheadParameters.write(
        getGPUTrajectoryPlayheadParameterValues({
          playhead,
          maxGap: options.maxGapHours * 3600
        })
      );
      playheadCompiled.encode(commandEncoder, {parameters: undefined});
      if (options.showTrails) {
        const trailSeconds = options.trailDays * SECONDS_PER_DAY;
        windowParameters.write(
          getGPUTimeWindowParameterValues({
            start: playhead - trailSeconds,
            end: playhead,
            startFadeDuration: trailSeconds * options.tailFade
          })
        );
        trailCompiled.encode(commandEncoder, {parameters: undefined});
      }
      if (options.followSelected && selectedTrack !== NO_TRACK && clock.moved) {
        const position = positionAt(selectedTrack, playhead);
        if (position) ctx.flyTo({longitude: position[0], latitude: position[1]});
      }
      if (statusStale || frame.frameIndex % STATUS_INTERVAL_FRAMES === 0) {
        reader.markStale();
        statusStale = false;
      }
      reader.flush(commandEncoder);
    },

    getLayers() {
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      if (options.showFlyways) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'season-backdrop',
            ...drawProps,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            widthPixels: 0.8,
            color: dark ? [200, 208, 224, 26] : [50, 60, 85, 30]
          })
        );
      }
      if (options.showTrails) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'season-trails',
            ...drawProps,
            segments: segmentsBuffer,
            ids: trailIds,
            drawCommands: trailDraw,
            weights: fadeWeights,
            clipFractions,
            values: speciesBuffer,
            valueFormat: 'uint32',
            valueIndices: segmentTracksBuffer,
            colormap: 'category',
            palette: MIGRATION_SPECIES_COLORS,
            widthPixels: 2.2,
            opacity: options.trailOpacity
          })
        );
      }
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'season-selected',
          ...drawProps,
          segments: selectedSegments,
          instanceCount: tracks.longestTrack,
          widthPixels: 1.6,
          color: dark ? [255, 255, 255, 150] : [15, 20, 30, 150]
        }),
        new SpatialAnalysisPointLayer({
          id: 'season-birds',
          ...drawProps,
          positions: currentPositions,
          ids: activeIds,
          drawCommands: markerDraw,
          values: speciesBuffer,
          valueFormat: 'uint32',
          colormap: 'category',
          palette: MIGRATION_SPECIES_COLORS,
          radiusPixels: options.markerSize,
          opacity: 0.98
        })
      );
      return layers;
    },

    getTooltip(event) {
      const track = pickBird(event.pixel);
      return track < 0 ? null : describeMigrationTrack(tracks, track);
    },

    onClick(event) {
      const track = pickBird(event.pixel);
      if (track < 0) {
        const [x, y] = event.coordinate
          ? tracks.project(event.coordinate[0], event.coordinate[1])
          : [0, 0];
        const nearest = event.coordinate
          ? findNearestTrack(tracks, x, y)
          : {track: -1, distance: 0};
        if (nearest.track < 0 || nearest.distance > 150000) return false;
        selectedTrack = nearest.track;
      } else {
        selectedTrack = track;
      }
      writeSelection();
      describeSelected();
      ctx.requestLayers();
      return true;
    },

    destroy() {
      destroyed = true;
      reader.stop();
      resources.destroy();
    }
  };

  /** Nearest active bird within 14 CSS pixels of the pointer, or -1. */
  function pickBird(pixel: readonly [number, number]): number {
    const viewport = ctx.getViewport();
    const current = snapshot;
    if (!viewport || !current) return -1;
    let best = -1;
    let bestDistance = 14 * 14;
    for (let track = 0; track < trackCount; track++) {
      if (current.status[track] !== GPU_TRAJECTORY_PLAYHEAD_STATUS.active) continue;
      const [x, y] = viewport.project([
        current.positions[track * 2],
        current.positions[track * 2 + 1]
      ]);
      const squared = (x - pixel[0]) ** 2 + (y - pixel[1]) ** 2;
      if (squared < bestDistance) {
        bestDistance = squared;
        best = track;
      }
    }
    return best;
  }
}
