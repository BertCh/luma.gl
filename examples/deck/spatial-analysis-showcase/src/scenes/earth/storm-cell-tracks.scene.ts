// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CREDITS, joinCredits} from '../../cartography/credits';
import {US, labelsFor} from '../../cartography/gazetteer';
import {ground} from '../../cartography/grounds';
import {playbackOptions} from '../../engine/playback';
import {defineScene} from '../scene';
import {
  COMPASS_COLORS,
  COMPASS_SECTORS,
  formatStormClock,
  STORM_EVENT_SECONDS,
  STORM_VIEW
} from './storm-data';
import type {StormCellTracksOptions} from './storm-cell-tracks.compute';

const IOWA_VIEW = {longitude: -94.2, latitude: 42.4, zoom: 6.3};
const PLAINS_VIEW = {longitude: -97.3, latitude: 34.2, zoom: 6.1};
const TRACK_LABELS = labelsFor(US, ['msp', 'ord', 'dfw']);
const SPEED_CLASS_ENTRIES = [
  {color: [255, 237, 160, 255] as const, label: '0–20 km/h'},
  {color: [254, 178, 76, 255] as const, label: '20–40 km/h'},
  {color: [240, 59, 32, 255] as const, label: '40–70 km/h'},
  {color: [189, 0, 38, 255] as const, label: '70–100 km/h'},
  {color: [103, 0, 31, 255] as const, label: '100+ km/h'}
];
const cartouche = (title: string, subtitle: string) => ({
  title,
  subtitle,
  chips: ['radar-derived tracks', 'sampled lightning'] as const
});

export default defineScene<StormCellTracksOptions>({
  id: 'storm-cell-tracks',
  title: 'How fast do storm cells move, and where does it flash?',
  chapter: 'earth',
  order: 6,
  summary:
    'Replay radar-derived storm-cell tracks: separate speed from cyclic heading, inspect a buffered corridor, then compare lightning density at different grid sizes.',
  contributors: [
    'GPUTrajectoryPlayhead',
    'GPUTimeWindowFilter',
    'GPUTrajectoryMetrics',
    'GPUOutlineGeometry',
    'GPUPointDensity'
  ],
  datasets: [
    {id: 'poopdeck-mrms-precip-tracks', role: 'storm-cell tracks (MRMS, 21-22 May 2024)'},
    {id: 'poopdeck-goes-glm-lightning', role: 'lightning flashes (GOES-16 GLM)'}
  ],
  initialView: STORM_VIEW,

  options: [
    ...playbackOptions<StormCellTracksOptions>({
      time: {
        min: 0,
        max: STORM_EVENT_SECONDS,
        step: 300,
        default: 36000,
        label: 'Time (UTC)',
        format: formatStormClock,
        help: 'The window runs from 12:00 UTC on 21 May to 06:00 UTC on 22 May 2024. Central time (the Plains and Midwest) is five hours earlier.'
      },
      playing: false,
      speed: {
        min: 1,
        max: 60,
        step: 1,
        default: 12,
        unit: ' min/s',
        label: 'Playback speed',
        help: 'Simulated minutes per real second. 12 plays the whole 18 hours in 90 seconds; 2 plays one hour in 30 seconds.'
      },
      loop: true
    }),
    {
      kind: 'select',
      id: 'colorBy',
      label: 'Color cells and trails by',
      group: 'Cells',
      apply: 'param',
      default: 'speed',
      help: 'Speed of the 10-minute step the cell is on, or the compass direction it is heading. Both come from the step columns of GPUTrajectoryMetrics, measured on each track’s own tangent plane.',
      options: [
        {value: 'speed', label: 'Speed (km/h)'},
        {value: 'heading', label: 'Heading (compass sector)'}
      ]
    },
    {
      kind: 'slider',
      id: 'markerSize',
      label: 'Arrow size',
      group: 'Cells',
      apply: 'param',
      min: 4,
      max: 16,
      step: 1,
      default: 9,
      unit: 'px',
      help: 'Half-length in screen pixels of the arrowhead that marks each active cell and points along its heading.'
    },
    {
      kind: 'toggle',
      id: 'showBackdrop',
      label: 'Show every track faintly',
      group: 'Cells',
      apply: 'param',
      default: true,
      help: 'Draws the whole 18 hours of tracks as thin gray lines, so you can see where cells will go before they get there.'
    },
    {
      kind: 'toggle',
      id: 'showTrails',
      label: 'Show trails',
      group: 'Cells',
      apply: 'param',
      default: true,
      help: 'Trails are the track segments inside a sliding window ending at the playhead, selected by GPUTimeWindowFilter.'
    },
    {
      kind: 'slider',
      id: 'trailMinutes',
      label: 'Trail length',
      group: 'Cells',
      apply: 'param',
      min: 10,
      max: 240,
      step: 10,
      default: 60,
      unit: 'min',
      disabledWhen: state => !state.showTrails,
      help: 'Length of the time window. A cell moving at 70 km/h draws a trail about 70 km long at 60 minutes.'
    },
    {
      kind: 'slider',
      id: 'tailFade',
      label: 'Tail fade',
      group: 'Cells',
      apply: 'param',
      min: 0,
      max: 1,
      step: 0.1,
      default: 0.6,
      disabledWhen: state => !state.showTrails,
      help: 'Fraction of the window over which the oldest part of a trail fades in. 0 draws solid trails.'
    },
    {
      kind: 'toggle',
      id: 'showArrows',
      label: 'Show motion vectors',
      group: 'Motion',
      apply: 'param',
      default: false,
      help: 'A line from each cell along its current heading, as long as the distance it would cover at its current speed in the vector time. A straight-line extrapolation, not a forecast.'
    },
    {
      kind: 'slider',
      id: 'arrowMinutes',
      label: 'Vector time',
      group: 'Motion',
      apply: 'param',
      min: 5,
      max: 90,
      step: 5,
      default: 30,
      unit: 'min',
      disabledWhen: state => !state.showArrows,
      help: 'How far ahead the motion vector reaches, as minutes of travel at the cell’s current speed. A parameter buffer write read by the lookup kernel.'
    },
    {
      kind: 'toggle',
      id: 'showStalls',
      label: 'Show stalled cells',
      group: 'Motion',
      apply: 'param',
      default: false,
      help: 'Marks every place a cell crawled: a run of slow steps lasting at least the minimum duration (GPUTrajectoryMetrics stop detection). Slow cells are the ones that drop the most rain on one place.'
    },
    {
      kind: 'slider',
      id: 'stallSpeedKmh',
      label: 'Stall speed threshold',
      group: 'Motion',
      apply: 'param',
      min: 5,
      max: 60,
      step: 1,
      default: 30,
      unit: 'km/h',
      disabledWhen: state => !state.showStalls,
      help: 'A step is slow when the cell covers less than this speed times the step time. Parameter buffer: changing it re-runs the metrics graph and never recompiles it.'
    },
    {
      kind: 'slider',
      id: 'stallMinutes',
      label: 'Minimum stall duration',
      group: 'Motion',
      apply: 'param',
      min: 10,
      max: 120,
      step: 5,
      default: 30,
      unit: 'min',
      disabledWhen: state => !state.showStalls,
      help: 'A run of slow steps counts as a stall only when it lasts at least this long.'
    },
    {
      kind: 'toggle',
      id: 'showSwath',
      label: 'Show swaths',
      group: 'Swaths',
      apply: 'param',
      default: false,
      help: 'Buffers every track by the swath half-width (GPUOutlineGeometry in spherical mode: the distance is meters on the sphere) and draws the result under the cells.'
    },
    {
      kind: 'slider',
      id: 'swathKm',
      label: 'Swath half-width',
      group: 'Swaths',
      apply: 'param',
      min: 2,
      max: 40,
      step: 1,
      default: 10,
      unit: 'km',
      disabledWhen: state => !state.showSwath,
      help: 'Distance each side of a track. A parameter buffer write: dragging it re-runs the buffer kernel, never recompiles.'
    },
    {
      kind: 'select',
      id: 'swathMode',
      label: 'Swath extent',
      group: 'Swaths',
      apply: 'param',
      default: 'so-far',
      disabledWhen: state => !state.showSwath,
      help: 'Whole event draws every track’s swath; so far draws only the part each cell has covered by the playhead and grows as time plays.',
      options: [
        {value: 'so-far', label: 'Up to the playhead'},
        {value: 'event', label: 'Whole event'}
      ]
    },
    {
      kind: 'select',
      id: 'swathJoin',
      label: 'Round-join smoothness',
      group: 'Swaths',
      apply: 'compile',
      default: '16',
      disabledWhen: state => !state.showSwath,
      help: 'Triangles in the round join and cap at each vertex (compile-time). 8 is visibly faceted, 32 is smooth; each choice compiles once and is cached.',
      options: [
        {value: '8', label: '8 triangles per join'},
        {value: '16', label: '16 triangles per join'},
        {value: '32', label: '32 triangles per join'}
      ]
    },
    {
      kind: 'slider',
      id: 'swathOpacity',
      label: 'Swath opacity',
      group: 'Swaths',
      apply: 'param',
      min: 0.05,
      max: 0.8,
      step: 0.05,
      default: 0.22,
      disabledWhen: state => !state.showSwath,
      help: 'Overlaps are not merged: where cells passed over the same ground the swath draws darker, which is itself a measure of repeated passes.'
    },
    {
      kind: 'toggle',
      id: 'showLightning',
      label: 'Show lightning density',
      group: 'Lightning',
      apply: 'param',
      default: false,
      help: 'Density of the flashes in the time window, binned to the screen on the GPU and following the camera.'
    },
    {
      kind: 'select',
      id: 'lightningMode',
      label: 'Flash window',
      group: 'Lightning',
      apply: 'param',
      default: 'window',
      disabledWhen: state => !state.showLightning && !state.showFlashes,
      help: 'Recent flashes only (a sliding window), all flashes since 12:00 UTC (accumulating), or the whole event at once. The mask comes from GPUTimeWindowFilter and feeds the density.',
      options: [
        {value: 'window', label: 'Sliding window'},
        {value: 'cumulative', label: 'Accumulating from the start'},
        {value: 'event', label: 'Whole event'}
      ]
    },
    {
      kind: 'slider',
      id: 'lightningMinutes',
      label: 'Window length',
      group: 'Lightning',
      apply: 'param',
      min: 5,
      max: 180,
      step: 5,
      default: 30,
      unit: 'min',
      disabledWhen: state => state.lightningMode !== 'window',
      help: 'Length of the sliding window ending at the playhead.'
    },
    {
      kind: 'select',
      id: 'statistic',
      label: 'Statistic per cell',
      group: 'Lightning',
      apply: 'compile',
      default: 'count',
      help: 'Number of flashes per cell, or the sum of their optical energy (a weights buffer). Compile-time: each choice is a separate cached graph.',
      options: [
        {value: 'count', label: 'Count of flashes'},
        {value: 'energy', label: 'Sum of optical energy'}
      ]
    },
    {
      kind: 'select',
      id: 'resolution',
      label: 'Resolution',
      group: 'Lightning',
      apply: 'compile',
      default: 'medium',
      help: 'Cells across the screen (gridSize, compile-time). Finer grids show individual storms; coarser grids show the regional pattern and are less noisy.',
      options: [
        {value: 'coarse', label: 'Coarse (80 across)'},
        {value: 'medium', label: 'Medium (128 across)'},
        {value: 'fine', label: 'Fine (192 across)'}
      ]
    },
    {
      kind: 'slider',
      id: 'sigma',
      label: 'Smoothing radius (sigma)',
      group: 'Lightning',
      apply: 'param',
      min: 0,
      max: 2.5,
      step: 0.25,
      default: 1,
      unit: 'cells',
      help: 'Standard deviation of a Gaussian blur of the cell field, in cells. 0 shows raw counts. The kernel weights are a parameter buffer, so this never recompiles.'
    },
    {
      kind: 'slider',
      id: 'lightningOpacity',
      label: 'Lightning opacity',
      group: 'Lightning',
      apply: 'param',
      min: 0.2,
      max: 1,
      step: 0.05,
      default: 0.75,
      help: 'Opacity of the density raster over the basemap.'
    },
    {
      kind: 'toggle',
      id: 'showFlashes',
      label: 'Show individual flashes',
      group: 'Lightning',
      apply: 'param',
      default: false,
      help: 'Draws every sampled flash in the window as a dot, gathered by the compact id list of GPUTimeWindowFilter.'
    },
    {
      kind: 'toggle',
      id: 'showHotSpots',
      label: 'Highlight hot spots',
      group: 'Lightning',
      apply: 'param',
      default: false,
      disabledWhen: state => !state.showLightning,
      help: 'Outlines the cells whose density is above the chosen percentile of the active cells in view. A percentile cut, not a significance test.'
    },
    {
      kind: 'slider',
      id: 'hotPercentile',
      label: 'Hot-spot percentile',
      group: 'Lightning',
      apply: 'param',
      min: 80,
      max: 99.5,
      step: 0.5,
      default: 95,
      unit: '%',
      disabledWhen: state => !state.showLightning || !state.showHotSpots,
      help: 'Cells above this percentile of the non-empty cells are hot. The threshold is recomputed from the density cells each time the picture settles.'
    }
  ],

  readouts: [
    {
      id: 'rateChart',
      label: 'Lightning through the event',
      kind: 'chart',
      help: 'Estimated flashes per minute in 15-minute bins, scaled up from the sample. The line is the playhead.'
    },
    {
      id: 'speedChart',
      label: 'How fast do steps go?',
      kind: 'chart',
      help: 'Speed of every 10-minute step of every track, from GPUTrajectoryMetrics. Speeds above about 100 km/h are usually the tracker jumping between merging or splitting cells.'
    },
    {
      id: 'roseChart',
      label: 'Which way do cells head?',
      kind: 'chart',
      help: 'Steps per compass sector from the step headings. Heading is the direction of travel (toward), not where the wind comes from.'
    },
    {id: 'clock', label: 'Playhead', help: 'Simulated UTC time.'},
    {
      id: 'activeCells',
      label: 'Cells tracked now',
      format: 'integer',
      help: 'Tracks whose first and last fixes bracket the playhead: the arrows drawn.'
    },
    {
      id: 'activeSpeed',
      label: 'Mean speed now',
      help: 'Mean speed of the steps the active cells are on.'
    },
    {
      id: 'trailSegments',
      label: 'Trail segments live',
      format: 'integer',
      help: 'Segments inside the trail window, counted by GPUTimeWindowFilter.'
    },
    {
      id: 'flashesNow',
      label: 'Sampled flashes in the window',
      format: 'integer',
      help: 'Flashes of the sample inside the current flash window, counted by GPUTimeWindowFilter.'
    },
    {
      id: 'peakDensity',
      label: 'Peak cell value',
      help: 'Highest cell value in view (flashes, or femtojoules).'
    },
    {id: 'cellSize', label: 'Cell width', help: 'Width of one density cell at the current zoom.'},
    {
      id: 'hotCells',
      label: 'Hot-spot cells',
      help: 'Cells above the percentile, among the non-empty cells in view.'
    },
    {id: 'grid', label: 'Density grid'},
    {
      id: 'medianSpeed',
      label: 'Median step speed',
      help: 'Median speed of every step of every track.'
    },
    {id: 'fastest', label: 'Fastest step'},
    {id: 'stalls', label: 'Stalls detected', help: 'Runs of slow steps at the current thresholds.'},
    {id: 'stalledCells', label: 'Cells that stalled'},
    {id: 'longestStall', label: 'Longest stall'},
    {id: 'tracks', label: 'Tracks'},
    {
      id: 'swathWidth',
      label: 'Corridor width',
      help: 'The visible cross-track bracket is twice the selected half-width.'
    },
    {id: 'flashes', label: 'Flashes'},
    {
      id: 'selected',
      label: 'Selected cell',
      help: 'Click a cell to read its times, peak reflectivity, mean and maximum speed.'
    }
  ],

  legends: state => [
    state.colorBy === 'speed'
      ? {
          kind: 'categories' as const,
          title: 'Cell speed (km/h)',
          entries: SPEED_CLASS_ENTRIES,
          note: 'Fixed five speed classes; this is not a continuous ramp.'
        }
      : {
          kind: 'categories' as const,
          title: 'Heading (direction of travel)',
          entries: COMPASS_SECTORS.map((name, index) => ({
            color: COMPASS_COLORS[index],
            label: name
          })),
          note: 'Compass sector of the step, measured from true north.'
        },
    ...(state.showLightning
      ? [
          {
            kind: 'categories' as const,
            id: 'flashes',
            title: 'Sampled lightning per cell (fixed amber classes)',
            entries: [
              {color: [255, 237, 160, 190] as const, label: '1–5'},
              {color: [254, 178, 76, 205] as const, label: '5–15'},
              {color: [240, 59, 32, 220] as const, label: '15–40'},
              {color: [189, 0, 38, 235] as const, label: '40+'}
            ],
            note: 'The identical sampled flash total is rebinned at coarse and fine resolutions; changing cell size demonstrates MAUP.'
          }
        ]
      : []),
    ...(state.showSwath || state.showStalls || state.showHotSpots
      ? [
          {
            kind: 'categories' as const,
            title: 'Overlays',
            entries: [
              ...(state.showSwath
                ? [{color: [226, 128, 28, 120] as const, label: 'Swath (darker = more passes)'}]
                : []),
              ...(state.showStalls
                ? [{color: [255, 41, 128, 255] as const, label: 'Stalled cell (size = duration)'}]
                : []),
              ...(state.showHotSpots && state.showLightning
                ? [
                    {
                      color: [0, 160, 210, 160] as const,
                      label: `Hot spot (top ${(100 - state.hotPercentile).toFixed(1)}% of cells)`
                    }
                  ]
                : [])
            ]
          }
        ]
      : [])
  ],

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUTrajectoryMetrics, GPUTrajectoryPlayhead, GPUOutlineGeometry, GPUPointDensity,
  getGPUTrajectoryMetricsParameterValues, getGPUTrajectoryPlayheadParameterValues,
  getGPUOutlineGeometryParameterValues
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUTimeWindowFilter, getGPUTimeWindowParameterValues} from '@luma.gl/experimental/gpu-dataframe';

// Each track on its own tangent plane (x east, y north, meters): speeds and headings are true.
const metrics = new GPUCommandGraph(device, {id: 'metrics'});
metrics.add(new GPUTrajectoryMetrics({spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'native'},
  positions: localMeters, timestamps, trackOffsets,
  parameters: stallParameters.importToGraph(metrics),
  stepSpeeds, stepHeadings, trackStopCounts,
  stops: {output: {ids, count, overflow}, startRows, endRows, durations}
}));

// Cells at the playhead: positions in degrees, plus the segment row each cell is on.
const play = new GPUCommandGraph(device, {id: 'playhead'});
play.add(new GPUTrajectoryPlayhead({
  positions: lngLat, timestamps, trackOffsets,
  parameters: playheadParameters.importToGraph(play),
  currentPositions, status, segmentRows, activeTracks: {ids, count, overflow}
}));
${
  state.showSwath
    ? `
// Buffer the tracks by meters on the sphere (drawing only; overlaps are not merged).
swath.add(new GPUOutlineGeometry({
  positions: lngLat, geometryType: 'lines', pathOffsets: trackOffsets,
  coordinateSystem: 'spherical', joinSegments: ${state.swathJoin},      // compile-time
  parameters: swathDistance.importToGraph(swath),                       // per-frame
  output: {positions: triangles}
}));`
    : ''
}
${
  state.showLightning
    ? `
// Flashes in a window feed the density as a mask.
flashes.add(new GPUTimeWindowFilter({
  timestamps: flashTimes, window: flashWindow.importToGraph(flashes),
  output: {ids, count, overflow}, outputMask: mask
}));
density.add(new GPUPointDensity({
  positions: flashMeters, mask, bounds: viewBounds.importToGraph(density),
  gridSize: [${state.resolution === 'coarse' ? 80 : state.resolution === 'fine' ? 192 : 128}, ${state.resolution === 'coarse' ? 52 : state.resolution === 'fine' ? 126 : 84}], statistic: '${state.statistic === 'count' ? 'count' : 'sum'}',${state.statistic === 'energy' ? '\n  weights: flashEnergy,' : ''}
  smoothing: {kernel: gaussian.importToGraph(density), kernelWidth: 17, kernelHeight: 17, strategy: 'direct'},
  output: {values, extent, histogram}
}));`
    : ''
}
// every frame: parameters are plain buffer writes
playheadParameters.write(getGPUTrajectoryPlayheadParameterValues({playhead, maxGap: 0}));
stallParameters.write(getGPUTrajectoryMetricsParameterValues({
  stopSpeedThreshold: ${(state.stallSpeedKmh / 3.6).toFixed(2)},   // m/s (${state.stallSpeedKmh} km/h)
  stopMinimumDuration: ${state.stallMinutes * 60}                   // s
}));
swathDistance.write(getGPUOutlineGeometryParameterValues({distance: ${state.swathKm * 1000}}));
flashWindow.write(getGPUTimeWindowParameterValues({start: playhead - ${state.lightningMinutes * 60}, end: playhead}));
play.encode(commandEncoder, {parameters: undefined});`,

  about: {
    what: '`GPUTrajectoryMetrics` measures every 10-minute step of every storm-cell track (speed and heading) and finds runs of slow steps. `GPUTrajectoryPlayhead` interpolates every cell at the clock and tells which segment each is on. `GPUTimeWindowFilter` keeps the trail segments and the lightning flashes inside a window. `GPUOutlineGeometry` buffers the tracks into swaths, and `GPUPointDensity` bins the flashes of the window into a smoothed density that follows the camera.',
    why: 'Forecasters and emergency planners ask the same questions of a severe day: which way are the cells moving, how fast, where did they pass, and where is the electrical activity concentrated? A cell that crawls (a stall) drops more rain on one place, and a fast-moving cell gives less time to react. Doing the step metrics, the windowing and the density on the GPU keeps the whole event live as you scrub.',
    howToRead:
      'Arrows are cells at the playhead, pointing along the heading of the step they are on and colored by speed or by compass sector. Trails show where they have been. The orange swath is every track buffered by the swath half-width: darker means more cells passed. The inferno raster is lightning density; cyan outlines are the hottest cells. Radar cells are 51 dBZ and above, followed by the MRMS cell tracker; a sudden jump in speed is usually the tracker re-linking a merged or split cell, not real motion.'
  },

  pipeline: [
    {
      id: 'metrics',
      label: 'Track metrics',
      detail: 'Measure each tangent-plane step: speed and heading'
    },
    {id: 'playhead', label: 'Playhead', detail: 'Interpolate one cell position per track'},
    {
      id: 'buffer',
      label: 'Corridor',
      detail: 'Buffer lines in spherical metres; overlaps are not unioned'
    },
    {id: 'density', label: 'Grid density', detail: 'Bin sampled flashes into a view-dependent grid'}
  ],
  basemap: ground('night', {labels: 'above', labelPreset: 'places-only'}),
  furniture: {
    title: cartouche(
      'How did the storm cells move?',
      'Motion · km/h / heading · MRMS + GLM · 21–22 May 2024'
    ),
    scaleBar: {units: 'metric'},
    credit: joinCredits('NOAA MRMS and GOES-16 GLM (public domain)', CREDITS.naturalEarth),
    caveat:
      'Tracks are radar-derived objects, not named storms; lightning is a seeded sample and grid results vary with cell size.'
  },
  annotations: TRACK_LABELS,

  create: async ctx => (await import('./storm-cell-tracks.compute')).createStormCellTracks(ctx),

  story: [
    {
      id: 'playback',
      title: 'Radar cells cross the Plains overnight',
      headline: 'Follow radar-derived cells through time',
      textAlternative:
        'Neutral arrowheads and short trails show live radar-derived cells over a dark central-US map.',
      optionsMode: 'fresh',
      body: 'Press **Play** or drag **Time (UTC)**. Each neutral arrow is one radar-derived cell at the clock; its short trail is the preceding time window. The live sample line and flash-rate chart keep the temporal frame visible.\n\nThis is an object tracker over intense radar echoes, not a catalogue of named storms. Reduced motion starts paused; the clock can always be scrubbed directly.',
      evidence:
        'At **{{clock}}**, **{{activeCells}}** radar-derived cells are active; the fixed 18-hour rate chart keeps that instant anchored in the full event.',
      caveat:
        'Cell identities can end, merge or restart as the radar tracker links echoes; an arrow is not a named storm or an independently observed object.',
      camera: {...IOWA_VIEW, transitionMs: 1400},
      options: {time: 34200, play: false, showTrails: true},
      controls: ['play', 'time', 'speed'],
      readouts: ['clock', 'activeCells', 'rateChart']
    },
    {
      id: 'speed',
      title: 'Speed is distance divided by time',
      headline: 'Measure speed one step at a time',
      textAlternative:
        'Classed warm-to-cool speed trails and arrowheads sit over dim context tracks.',
      optionsMode: 'fresh',
      body: '**`GPUTrajectoryMetrics`** measures each segment on its own tangent plane: distance divided by elapsed time gives speed. The histogram is the full step distribution; the live readouts report the median and fastest step.\n\nSteps above the displayed range are tracker-jump candidates, not evidence of a faster storm. Fixes are about ten minutes apart, so a cell head is an interpolation between observations.',
      evidence:
        'Across **{{tracks}}**, the median is **{{medianSpeed}}** and the largest observed step is **{{fastest}}**; the histogram keeps every 10-minute step in the denominator.',
      caveat:
        'The 0–120 km/h display domain is fixed across the story. Values beyond it remain in the readout but are commonly tracker jumps at cell mergers or splits.',
      camera: {...STORM_VIEW, transitionMs: 1400},
      options: {time: 36000, play: false, showTrails: true, trailMinutes: 120, colorBy: 'speed'},
      controls: ['colorBy', 'trailMinutes'],
      readouts: ['medianSpeed', 'fastest', 'speedChart', 'roseChart']
    },
    {
      id: 'heading',
      title: 'Heading needs a cyclic key',
      headline: 'North wraps around to northwest',
      textAlternative:
        'A cyclic compass palette and rose chart show direction of travel for the active tracks.',
      optionsMode: 'fresh',
      body: 'Direction is circular: north-west and north-east are neighbours, so a linear colour scale would make adjacent headings look unrelated. The compass palette closes at north and the rose chart uses the same order.\n\nThe heading is where the radar object travels, not where wind comes from. Compare it with speed without encoding both variables on one mark.',
      evidence:
        'The compass rose counts every qualifying step, while **{{activeCells}}** cells at the shared UTC playhead average **{{activeSpeed}}**.',
      caveat:
        'Heading is the tracked echo’s direction of travel, not wind direction; steps below 1 km/h are excluded from the directional rose.',
      camera: {...IOWA_VIEW, transitionMs: 1400},
      options: {
        time: 34200,
        play: false,
        colorBy: 'heading',
        showArrows: false,
        showTrails: true,
        trailMinutes: 60
      },
      controls: ['colorBy', 'time'],
      readouts: ['activeSpeed', 'roseChart', 'activeCells']
    },
    {
      id: 'vector',
      title: 'A motion vector projects the next position',
      headline: 'Extrapolate, do not forecast',
      textAlternative:
        'A selected cell has a dashed forward vector and arrival ring over a dark map.',
      optionsMode: 'fresh',
      body: 'A lookup reads the current step’s speed and heading and draws a dashed line for the selected number of minutes. It is a straight-line extrapolation, not a forecast: cells turn, merge, split and decay.\n\nChange **Vector time** to see distance scale directly with time. The active-cell count makes clear how many independent motion vectors are present at this moment.',
      evidence:
        'At **{{clock}}**, **{{activeCells}}** active cells have a mean current-step speed of **{{activeSpeed}}**; vector length is that speed multiplied by the selected duration.',
      caveat:
        'The vector assumes unchanged speed and heading. It contains no atmospheric model, uncertainty cone, cell-growth term or merger logic.',
      camera: {...STORM_VIEW, transitionMs: 1400},
      options: {
        showArrows: true,
        arrowMinutes: 30,
        showSwath: false,
        showTrails: true,
        showBackdrop: true,
        play: false,
        time: 54000
      },
      controls: ['arrowMinutes', 'time'],
      readouts: ['activeSpeed', 'activeCells', 'clock']
    },
    {
      id: 'swath',
      title: 'A buffered track is only a corridor',
      headline: 'A buffer is not a storm footprint',
      textAlternative: 'A translucent violet buffered corridor retains its radar-track centreline.',
      optionsMode: 'fresh',
      body: '**`GPUOutlineGeometry`** buffers each path by a chosen spherical-metre distance. The translucent ribbon is a *buffered track corridor*; it is neither a radar footprint nor a merged coverage area.\n\nOverlaps darken because triangles overlap. Adjust the width as a sensitivity test, and keep the centreline visible so the construction is legible.',
      evidence:
        '**{{tracks}}** are buffered to **{{swathWidth}}**; at **{{clock}}** the same UTC clock controls whether the corridor stops at the playhead or spans the event.',
      caveat:
        'Overlapping ribbons are drawn on top of one another rather than unioned, so darker colour means repeated geometry, not greater storm intensity.',
      camera: {...STORM_VIEW, transitionMs: 1400},
      options: {
        showLightning: false,
        showSwath: true,
        swathMode: 'event',
        swathKm: 10,
        showBackdrop: false,
        showTrails: true,
        play: false,
        time: 41400
      },
      controls: ['swathKm', 'swathMode'],
      readouts: ['clock', 'tracks', 'swathWidth']
    },
    {
      id: 'lightning',
      title: 'Hot spots depend on the grid',
      headline: 'Grid size changes the hot cells',
      textAlternative:
        'Amber sampled lightning points sit over a classed dark density grid with highlighted cells.',
      optionsMode: 'fresh',
      body: '**`GPUPointDensity`** bins the sampled flashes into the current map grid, then smooths the cells. Change **Resolution**: the same flashes can fall into different cells, so the rank and shape of a hot cell change with the aggregation unit. That is the modifiable areal unit problem, not a significance test.\n\nThe count is a sample of GLM detections. Keep the point layer on while comparing grids, and read the density value as sampled flashes rather than an absolute total. Next: do counties crossed by these cells show more outages?',
      evidence:
        'The active window contains **{{flashesNow}}** sampled flashes; each cell is **{{cellSize}}** wide, and **{{hotCells}}** exceed the selected percentile among non-empty cells.',
      caveat:
        '**{{flashes}}** are a seeded sample. Hot cells depend on the view, grid resolution, smoothing and percentile and are not a significance test.',
      camera: {...PLAINS_VIEW, transitionMs: 1600},
      options: {
        showLightning: true,
        showHotSpots: true,
        hotPercentile: 95,
        lightningMode: 'window',
        lightningMinutes: 10,
        showFlashes: true,
        showStalls: false,
        showTrails: false,
        showBackdrop: true,
        play: false,
        time: 64800
      },
      controls: ['resolution', 'hotPercentile', 'lightningMinutes'],
      readouts: ['flashesNow', 'peakDensity', 'cellSize', 'hotCells']
    }
  ]
});
