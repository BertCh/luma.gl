// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {playbackOptions} from '../../engine/playback';
import {defineScene} from '../scene';
import {
  COMPASS_COLORS,
  COMPASS_SECTORS,
  formatStormClock,
  STORM_EVENT_SECONDS,
  STORM_SPEED_RAMP_KMH,
  STORM_VIEW
} from './storm-data';
import type {StormCellTracksOptions} from './storm-cell-tracks.compute';

const IOWA_VIEW = {longitude: -94.2, latitude: 42.4, zoom: 6.3};
const PLAINS_VIEW = {longitude: -97.3, latitude: 34.2, zoom: 6.1};

export default defineScene<StormCellTracksOptions>({
  id: 'storm-cell-tracks',
  title: 'How fast do storm cells move, and where does it flash?',
  chapter: 'earth',
  order: 20,
  summary:
    'Replay 289 radar storm-cell tracks from 21-22 May 2024: speed and heading of every step from GPUTrajectoryMetrics, swaths from buffering the tracks, and a lightning density map with hot spots underneath.',
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
      kind: 'select',
      id: 'ramp',
      label: 'Speed ramp',
      group: 'Cells',
      apply: 'param',
      default: 'viridis',
      disabledWhen: state => state.colorBy !== 'speed',
      help: 'Color ramp of the speed encoding. Viridis, magma, inferno and cividis are perceptually uniform and color-blind safe.',
      options: [
        {value: 'viridis', label: 'Viridis'},
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'cividis', label: 'Cividis'}
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
      kind: 'select',
      id: 'lightningRamp',
      label: 'Lightning ramp',
      group: 'Lightning',
      apply: 'param',
      default: 'inferno',
      help: 'Color ramp of the density.',
      options: [
        {value: 'inferno', label: 'Inferno'},
        {value: 'magma', label: 'Magma'},
        {value: 'viridis', label: 'Viridis'},
        {value: 'cividis', label: 'Cividis'}
      ]
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
          kind: 'ramp' as const,
          title: 'Cell speed',
          ramp: state.ramp,
          extent: [0, STORM_SPEED_RAMP_KMH] as const,
          unit: 'km/h',
          format: (value: number) => value.toFixed(0)
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
            kind: 'ramp' as const,
            id: 'flashes',
            title:
              state.statistic === 'count' ? 'Lightning flashes per cell' : 'Flash energy per cell',
            ramp: state.lightningRamp,
            extent: 'gpu' as const,
            sqrtScale: true,
            unit: state.statistic === 'count' ? 'flashes' : 'fJ',
            format: (value: number) =>
              value >= 100 ? value.toFixed(0) : value >= 10 ? value.toFixed(1) : value.toFixed(2)
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
metrics.add(new GPUTrajectoryMetrics({
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

  create: async ctx => (await import('./storm-cell-tracks.compute')).createStormCellTracks(ctx),

  story: [
    {
      id: 'the-question',
      title: 'Which way, and how fast, did the storms move?',
      body: 'On 21 May 2024 an active severe-weather day ran from the Plains to the Northeast. The NOAA MRMS radar composite followed **289 intense storm cells** (51 dBZ and above) between 12:00 UTC and 06:00 UTC, one fix about every ten minutes; the faint gray lines are all their tracks.\n\nPress **Play** below, or drag **Time (UTC)**: each arrow is a cell now, pointing along its heading and colored by speed, with a one-hour trail behind it. The **Playback speed** slider sets how many simulated minutes pass per real second. The chart below the map counts GOES-16 lightning flashes per minute; its line is the playhead.',
      camera: {...IOWA_VIEW, transitionMs: 1400},
      options: {time: 34200, play: false, showTrails: true},
      controls: ['play', 'time', 'speed'],
      readouts: ['clock', 'activeCells', 'rateChart']
    },
    {
      id: 'speed-and-heading',
      title: 'Speed and heading, step by step',
      body: '**`GPUTrajectoryMetrics`** measures the speed and heading of every step in one pass. Each track is projected on its own tangent plane (x east, y north, in meters), so a track that runs for 300 km still gets true compass headings and speeds within about 3%, which one continental projection cannot give.\n\nSwitch **Color cells and trails by** to *Heading* to color by compass sector, then back to *Speed*. The histogram below is the speed of every step, and the rose chart counts steps per direction: most cells head toward the east and northeast, as storms steered by the jet stream do.',
      camera: {...STORM_VIEW, transitionMs: 1400},
      options: {time: 36000, play: false, showTrails: true, trailMinutes: 120, colorBy: 'speed'},
      controls: ['colorBy', 'ramp', 'trailMinutes'],
      readouts: ['medianSpeed', 'fastest', 'speedChart', 'roseChart']
    },
    {
      id: 'motion-vectors',
      title: 'Where is it heading? A motion vector',
      body: 'The playhead tells every cell which segment it is on. A small kernel then looks up that segment’s speed and heading in the metrics columns and draws a line from the cell along its heading, as long as the distance it covers in the **Vector time**: a straight-line extrapolation, the way a forecaster reads a storm-motion arrow. It is not a forecast; real cells turn, split and decay.\n\nDrag **Vector time** to 60 minutes and watch the vectors fan over the Iowa and Minnesota cells. The **Playback speed** control runs time forward so you can check whether the vector pointed where the cell actually went.',
      camera: {...IOWA_VIEW, transitionMs: 1400},
      options: {
        time: 34200,
        play: false,
        showArrows: true,
        arrowMinutes: 30,
        showTrails: true,
        trailMinutes: 60
      },
      controls: ['showArrows', 'arrowMinutes', 'speed'],
      readouts: ['activeSpeed', 'activeCells']
    },
    {
      id: 'swaths',
      title: 'Swaths: where the cells passed',
      body: '**`GPUOutlineGeometry`** buffers every track by a distance in meters on the sphere and writes the triangles for drawing. Turn on **Show swaths** and set **Swath extent** to *Whole event*: the orange band is everywhere within the half-width of a cell track.\n\nOverlaps are not merged, so ground that several cells crossed draws darker. Slide **Swath half-width** between 5 and 25 km (a parameter-buffer write, no recompile) and watch the swaths merge into corridors. Set the extent back to *Up to the playhead* and press Play to watch them grow.',
      camera: {...STORM_VIEW, transitionMs: 1400},
      options: {
        showSwath: true,
        swathMode: 'event',
        swathKm: 10,
        showTrails: false,
        showBackdrop: false,
        play: false,
        time: 54000
      },
      controls: ['showSwath', 'swathKm', 'swathMode', 'swathOpacity'],
      readouts: ['clock']
    },
    {
      id: 'lightning',
      title: 'Lightning under the storms',
      body: 'GOES-16’s Geostationary Lightning Mapper saw **346,450 flashes** over the US in this window; the scene holds a seeded 43% sample. **`GPUTimeWindowFilter`** selects the flashes in a sliding window and writes a mask; **`GPUPointDensity`** bins them into a grid that follows the camera and smooths them with a Gaussian.\n\nChoose a **Flash window** of *Accumulating from the start* to see the day’s total build, or keep the *Sliding window* and press Play: the hot cores follow the cells. **Smoothing radius** blurs the cells (0 shows raw counts), and **Statistic per cell** switches from flash count to summed optical energy.',
      camera: {...STORM_VIEW, transitionMs: 1400},
      options: {
        showLightning: true,
        lightningMode: 'window',
        lightningMinutes: 30,
        showBackdrop: true,
        showSwath: false,
        showTrails: true,
        trailMinutes: 60,
        play: false,
        time: 41400
      },
      controls: ['lightningMode', 'lightningMinutes', 'sigma', 'statistic'],
      readouts: ['flashesNow', 'peakDensity', 'cellSize']
    },
    {
      id: 'hot-spots',
      title: 'Hot spots, and cells that stalled',
      body: 'Turn on **Highlight hot spots** to outline the cells above the **Hot-spot percentile** of the non-empty cells in view: the electrical cores. This is a percentile cut, not a significance test (the Getis-Ord scene in the weights chapter is the statistical version).\n\nThen switch on **Show stalled cells**: **`GPUTrajectoryMetrics`** finds runs of slow steps (below **Stall speed threshold** for at least **Minimum stall duration**) and marks them, larger for longer stalls. A cell that crawls puts its rain, hail or wind over one place for longer. Try a threshold of 15 km/h and 20 minutes, and compare how many tracks stall.',
      camera: {...PLAINS_VIEW, transitionMs: 1600},
      options: {
        showLightning: true,
        showHotSpots: true,
        hotPercentile: 95,
        lightningMode: 'cumulative',
        showStalls: true,
        stallSpeedKmh: 30,
        stallMinutes: 30,
        showTrails: false,
        showBackdrop: true,
        play: false,
        time: 64800
      },
      controls: ['hotPercentile', 'showStalls', 'stallSpeedKmh', 'stallMinutes'],
      readouts: ['hotCells', 'stalls', 'stalledCells', 'longestStall']
    }
  ]
});
