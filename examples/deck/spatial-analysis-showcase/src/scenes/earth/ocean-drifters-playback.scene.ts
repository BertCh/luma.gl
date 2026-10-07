// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {formatPlaybackTime, playbackOptions} from '../../engine/playback';
import {defineScene, type LegendSpec} from '../scene';
import type {OceanDriftersPlaybackOptions} from './ocean-drifters-playback.compute';

const GLOBAL_VIEW = {longitude: -20, latitude: 15, zoom: 1.5};
/** Unix seconds of 2017-01-01 UTC, the time origin of the dataset. */
const ORIGIN_SECONDS = Date.UTC(2017, 0, 1) / 1000;
const formatDate = (days: number) => formatPlaybackTime.date(days * 86400, ORIGIN_SECONDS);

export default defineScene<OceanDriftersPlaybackOptions>({
  id: 'ocean-drifters-playback',
  title: 'A year of ocean drifters',
  chapter: 'earth',
  order: 2,
  summary:
    'Replay 2,811 satellite-tracked drifters through 2017 with trails colored by sea-surface temperature, then pile all the tracks into a GPU line density to see the Gulf Stream, the Kuroshio and the Agulhas appear.',
  contributors: ['GPUTrajectoryPlayhead', 'GPUTimeWindowFilter', 'GPULineDensity'],
  datasets: [{id: 'poopdeck-drifters', role: 'drifter tracks with temperature (NOAA GDP, 2017)'}],
  initialView: GLOBAL_VIEW,

  options: [
    ...playbackOptions<OceanDriftersPlaybackOptions>({
      time: {
        min: 0,
        max: 365,
        step: 1,
        default: 0,
        label: 'Date (2017)',
        format: formatDate,
        help: 'The playhead. Day 0 is 1 January 2017 (UTC). The window ends on 31 December 2017, so records started late in the year are cut short. While playing the slider follows the clock; drag it to jump.'
      },
      speed: {
        min: 2,
        max: 60,
        step: 1,
        default: 12,
        unit: 'days/s',
        label: 'Playback speed',
        help: 'Simulated days per real second. 12 days/s plays the year in about 35 seconds.'
      },
      loop: true
    }),
    {
      kind: 'slider',
      id: 'trailDays',
      label: 'Trail length',
      group: 'Trails (time window)',
      apply: 'param',
      min: 2,
      max: 60,
      step: 1,
      default: 14,
      unit: 'days',
      disabledWhen: state => !state.showTrails,
      help: 'Width of the window behind the playhead, [playhead - length, playhead]. It is written into the window parameter buffer; GPUTimeWindowFilter selects the live segments on the GPU.'
    },
    {
      kind: 'slider',
      id: 'tailFade',
      label: 'Tail fade',
      group: 'Trails (time window)',
      apply: 'param',
      min: 0,
      max: 1,
      step: 0.05,
      default: 1,
      disabledWhen: state => !state.showTrails,
      format: value => (value === 0 ? 'none' : `${Math.round(value * 100)}% of the trail`),
      help: 'How much of the trail fades out toward its oldest end (the start-fade duration of the window). 0 keeps it solid.'
    },
    {
      kind: 'slider',
      id: 'maxGapDays',
      label: 'Maximum fix gap',
      group: 'Trails (time window)',
      apply: 'param',
      min: 0,
      max: 10,
      step: 0.5,
      default: 0,
      unit: 'days',
      format: value => (value === 0 ? 'off' : `${value} days`),
      help: 'GPUTrajectoryPlayhead hides a drifter whose neighbouring fixes are further apart than this instead of drawing it at a guessed position. 0 turns the test off. The record is 6-hourly, so only real data gaps exceed a day or two.'
    },
    {
      kind: 'select',
      id: 'colorBy',
      label: 'Color tracks by',
      group: 'Color',
      apply: 'param',
      default: 'sst',
      help: 'Sea-surface temperature measured by the drifter, days since its release, or the date.',
      options: [
        {value: 'sst', label: 'Sea-surface temperature'},
        {value: 'age', label: 'Days since release'},
        {value: 'date', label: 'Date'}
      ]
    },
    {
      kind: 'range',
      id: 'sstRange',
      label: 'Temperature range of the ramp',
      group: 'Color',
      apply: 'param',
      min: -2,
      max: 34,
      step: 1,
      default: [0, 30],
      unit: '°C',
      disabledWhen: state => state.colorBy !== 'sst',
      help: 'Temperatures at the two ends of the ramp. Narrow it to bring out gradients inside one ocean.'
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp',
      group: 'Color',
      apply: 'param',
      default: 'inferno',
      help: 'All four are perceptually uniform; cividis is color-blind optimised.',
      options: [
        {value: 'inferno', label: 'Inferno'},
        {value: 'magma', label: 'Magma'},
        {value: 'viridis', label: 'Viridis'},
        {value: 'cividis', label: 'Cividis'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showTrails',
      label: 'Trails',
      group: 'Layers',
      apply: 'param',
      default: true,
      help: 'The part of every track inside the time window.'
    },
    {
      kind: 'toggle',
      id: 'showDots',
      label: 'Drifters now',
      group: 'Layers',
      apply: 'param',
      default: true,
      help: 'A dot at the interpolated position of every drifter that has a fix on both sides of the playhead, colored by the temperature at that moment.'
    },
    {
      kind: 'slider',
      id: 'dotSize',
      label: 'Dot size',
      group: 'Layers',
      apply: 'param',
      min: 2,
      max: 9,
      step: 0.5,
      default: 4,
      unit: 'px',
      disabledWhen: state => !state.showDots,
      help: 'Radius of the drifter dots in screen pixels.'
    },
    {
      kind: 'toggle',
      id: 'showBackdrop',
      label: 'Every track, faintly',
      group: 'Layers',
      apply: 'param',
      default: true,
      help: 'Draws all 292,000 track segments thinly in the same colors, so you see the whole year behind the moving window.'
    },
    {
      kind: 'select',
      id: 'background',
      label: 'Background',
      group: 'Track density',
      apply: 'param',
      default: 'none',
      help: 'Track density: kilometers of drifter track per 1,000 square kilometers, from GPULineDensity over all tracks of the year.',
      options: [
        {value: 'none', label: 'None'},
        {value: 'density', label: 'Track density'}
      ]
    },
    {
      kind: 'select',
      id: 'densityCell',
      label: 'Density cell size',
      group: 'Track density',
      apply: 'compile',
      default: '0.5',
      disabledWhen: state => state.background !== 'density',
      help: 'Grid cell of GPULineDensity. The grid size is a compile-time property, so the three graphs are compiled up front and switched; the panel marks this with a rebuild badge once.',
      options: [
        {value: '1', label: '1° (about 110 km)'},
        {value: '0.5', label: '0.5° (about 55 km)'},
        {value: '0.25', label: '0.25° (about 28 km)'}
      ]
    },
    {
      kind: 'slider',
      id: 'densityMax',
      label: 'Density at the top of the ramp',
      group: 'Track density',
      apply: 'param',
      min: 5,
      max: 400,
      step: 5,
      default: 60,
      unit: 'km / 1000 km²',
      disabledWhen: state => state.background !== 'density',
      help: 'Track length per 1,000 square kilometers that gets the last color. The ramp is a square root, so sparse ocean stays visible. Smaller cells need a higher value.'
    },
    {
      kind: 'slider',
      id: 'backgroundOpacity',
      label: 'Density opacity',
      group: 'Track density',
      apply: 'param',
      min: 0.2,
      max: 1,
      step: 0.05,
      default: 0.85,
      disabledWhen: state => state.background !== 'density',
      help: 'Opacity of the density raster over the basemap.'
    },
    {
      kind: 'select',
      id: 'region',
      label: 'Fly to',
      group: 'Camera',
      apply: 'param',
      default: 'all',
      help: 'Flies the camera to a boundary current or ocean. It does not filter the data.',
      options: [
        {value: 'all', label: 'The whole ocean'},
        {value: 'gulfStream', label: 'Gulf Stream'},
        {value: 'kuroshio', label: 'Kuroshio'},
        {value: 'agulhas', label: 'Agulhas'},
        {value: 'southern', label: 'Southern Ocean'},
        {value: 'tropicalPacific', label: 'Tropical Pacific'}
      ]
    }
  ],

  readouts: [
    {
      id: 'activeChart',
      label: 'Drifters reporting through the year',
      kind: 'chart',
      help: 'Number of drifter records with a position on each day. The vertical rule is the playhead. Records already at sea on 1 January all start on day 0 and are cut after 60 days, which makes the early peak; new deployments follow through the year.'
    },
    {
      id: 'sstChart',
      label: 'Temperatures measured',
      kind: 'chart',
      help: 'Histogram of the sea-surface temperature at every fix, in 1 degree bins.'
    },
    {id: 'date', label: 'Playhead date', help: 'UTC.'},
    {
      id: 'active',
      label: 'Drifters with a position now',
      format: 'integer',
      help: 'Pieces whose first and last fix bracket the playhead (and pass the gap test), counted by GPUTrajectoryPlayhead.'
    },
    {
      id: 'trailSegments',
      label: 'Trail segments live',
      format: 'integer',
      help: 'Segments inside the time window, counted by GPUTimeWindowFilter.'
    },
    {id: 'overflow', label: 'Active-list overflow'},
    {id: 'tracks', label: 'Tracks'},
    {id: 'vertices', label: 'Fixes', help: 'About one every 12 hours (the archive is 6-hourly).'},
    {id: 'sstRange', label: 'Temperature range'},
    {id: 'peakActive', label: 'Peak drifters reporting on one day', format: 'integer'},
    {id: 'density', label: 'Density grids'}
  ],

  legends: state => {
    const legends: LegendSpec[] = [];
    if (state.showTrails || state.showDots || state.showBackdrop) {
      if (state.colorBy === 'sst') {
        legends.push({
          kind: 'ramp',
          title: 'Sea-surface temperature',
          ramp: state.ramp,
          extent: state.sstRange,
          unit: '°C',
          format: value => value.toFixed(0)
        });
      } else if (state.colorBy === 'age') {
        legends.push({
          kind: 'ramp',
          title: 'Days since release',
          ramp: state.ramp,
          extent: [0, 60],
          unit: 'days',
          format: value => value.toFixed(0)
        });
      } else {
        legends.push({
          kind: 'ramp',
          title: 'Date',
          ramp: state.ramp,
          extent: [0, 365],
          labels: ['1 Jan 2017', '31 Dec 2017'],
          format: value => formatDate(value)
        });
      }
    }
    if (state.background === 'density') {
      legends.push({
        kind: 'ramp',
        title: 'Drifter track length per 1,000 km²',
        ramp: 'inferno',
        extent: [0, state.densityMax],
        sqrtScale: true,
        unit: 'km / 1000 km²',
        format: value => Math.round(value).toString()
      });
    }
    return legends;
  },

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUTrajectoryPlayhead, GPULineDensity,
  getGPUTrajectoryPlayheadParameterValues, getGPULineDensityParameterValues
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUTimeWindowFilter, getGPUTimeWindowParameterValues} from '@luma.gl/experimental/gpu-dataframe';

// positions: float32x2 lon/lat, timestamps: float32 days since 1 Jan 2017, trackOffsets: uint32
const playGraph = new GPUCommandGraph(device, {id: 'playback'});
playGraph.add(new GPUTrajectoryPlayhead({
  positions, timestamps, trackOffsets,
  parameters: playheadParameters.importToGraph(playGraph),
  currentPositions, status, segmentRows, segmentFractions,    // rows + fractions give the SST at the playhead
  activeTracks: {ids: activeIds, count: activeCount, overflow: activeOverflow},
  drawInstanceCount
}));

const trailGraph = new GPUCommandGraph(device, {id: 'trails'});
trailGraph.add(new GPUTimeWindowFilter({
  timestamps: segmentStarts, endTimestamps: segmentEnds,      // interval mode: one row per segment
  window: windowParameters.importToGraph(trailGraph),
  output: {ids: trailIds, count: trailCount, overflow: trailOverflow},
  fadeWeights, clipFractions, drawInstanceCount: trailDrawCount
}));

const densityGraph = new GPUCommandGraph(device, {id: 'density'});
densityGraph.add(new GPULineDensity({
  positions, pathOffsets: trackOffsets,
  columns: ${Math.round(360 / Number(state.densityCell))}, rows: ${Math.round(160 / Number(state.densityCell))},       // compile-time: ${state.densityCell} degree cells
  coordinateSystem: 'spherical',                              // lengths are great-circle meters
  parameters: gridParameters.importToGraph(densityGraph),     // [west, south, cellWidth, cellHeight]
  output: {lengths, densities, overflow}
}));

// every frame: parameters are plain buffer writes
playheadParameters.write(getGPUTrajectoryPlayheadParameterValues({playhead, maxGap: ${state.maxGapDays}}));
windowParameters.write(getGPUTimeWindowParameterValues({
  start: playhead - ${state.trailDays}, end: playhead, startFadeDuration: ${(state.trailDays * state.tailFade).toFixed(1)}
}));
play.encode(commandEncoder, {parameters: undefined});`,

  about: {
    what: '`GPUTrajectoryPlayhead` finds, for every drifter piece at once, the two fixes either side of the clock (a binary search per track) and interpolates the position; a small kernel uses its bracketing row and fraction to interpolate the temperature too. `GPUTimeWindowFilter` keeps the track segments that overlap a sliding window and fades them toward the tail. `GPULineDensity` clips every segment to a grid and sums track length per cell.',
    why: "Surface drifters are the ocean's own current meters. Played back, they show the circulation and how it changes with the seasons; piled up, they show where the ocean is crowded with tracks (gyres, convergence zones) and where it is swept clear. The temperature they carry is a free map of the water masses.",
    howToRead:
      'Each dot is a drifter now, each trail the last stretch of its path; the colors are sea-surface temperature (see the legend). The faint lines behind them are every track of the year. In the density map a bright ridge is a current that carries many buoys along the same path: the Gulf Stream and North Atlantic Current, the Kuroshio, the Agulhas and the Antarctic Circumpolar Current. Caveats: only drifters seen in 2017, the first 60 days of each record, at 12-hourly fixes; deployments are not uniform (they follow ship routes), so density measures where buoys were as much as where the water goes; the temperature is from the buoy hull, not a calibrated skin temperature.'
  },

  create: async ctx =>
    (await import('./ocean-drifters-playback.compute')).createOceanDriftersPlayback(ctx),

  story: [
    {
      id: 'a-year-of-drifters',
      title: "Where did the world's drifters go in 2017?",
      body: 'A **drifter** is a buoy with a drogue (a sock) that hangs about 15 meters down and makes it follow the current instead of the wind. It reports its position and sea-surface temperature through a satellite several times a day. The NOAA **Global Drifter Program** keeps a global array of more than a thousand of them afloat.\n\nThis is **2017**: 2,811 drifters, each shown for its first 60 days of the year. Press **Play** below, or drag **Date (2017)**: dots are the drifters now, trails are the last two weeks, and the **Playback speed** control sets how many days pass each second. The chart shows how many report on each day.',
      camera: {...GLOBAL_VIEW, transitionMs: 1200},
      options: {play: true, time: 0, trailDays: 14},
      controls: ['play', 'time', 'speed'],
      readouts: ['date', 'active', 'activeChart']
    },
    {
      id: 'temperature',
      title: 'Read the temperature along the track',
      body: 'Every fix carries the **sea-surface temperature** the buoy measured. The trails are colored by it: the warm tropics glow, the Southern Ocean and the Arctic go dark. A drifter that crosses a color boundary has crossed a front. Narrow **Temperature range of the ramp** to bring out gradients inside one ocean, or switch **Color tracks by** to *Days since release* to see which tracks are fresh.\n\nThe dots take their color from the temperature **at the playhead**: `GPUTrajectoryPlayhead` reports the bracketing fix and the fraction between the two, and a small kernel interpolates between their temperatures.',
      camera: {longitude: -40, latitude: 25, zoom: 2.6, transitionMs: 1400},
      options: {play: false, time: 100, trailDays: 30, showBackdrop: true},
      controls: ['colorBy', 'sstRange', 'ramp'],
      readouts: ['sstRange', 'sstChart']
    },
    {
      id: 'time-window',
      title: 'A sliding window of time, on the GPU',
      body: '**`GPUTimeWindowFilter`** treats every segment between two fixes as a time interval and keeps the ones that overlap the window `[playhead - length, playhead]`. It writes a fade weight (old end transparent) and a clip fraction (the oldest segment is cut part-way), compacts the live ids and writes the count straight into the draw call. Nothing returns to the CPU.\n\nDrag **Trail length** to 60 days for long streaks, or down to 3 days for a swarm of short ones; **Tail fade** sets how much of the trail fades. **Maximum fix gap** hides any drifter whose neighbouring fixes are further apart than the limit, instead of drawing a guess.',
      camera: {longitude: -45, latitude: 35, zoom: 3.2, transitionMs: 1400},
      options: {play: true, time: 120, trailDays: 20, showBackdrop: false, speed: 6},
      controls: ['trailDays', 'tailFade', 'maxGapDays'],
      readouts: ['trailSegments', 'active']
    },
    {
      id: 'density',
      title: 'Where do the tracks pile up?',
      body: '**`GPULineDensity`** clips every segment of every track to a half-degree grid, walks it through the cells it crosses and adds up the great-circle length per cell. The map is **kilometers of track per 1,000 square kilometers**: the bright ridges are currents that carry many buoys along the same path.\n\nChange **Density cell size** to 0.25° for detail or 1° for a smoother picture (the three graphs are compiled up front, so switching is cheap but shows a rebuild badge), and raise **Density at the top of the ramp** when the smaller cells saturate. The ramp is a square root so the sparse ocean stays visible.',
      camera: {...GLOBAL_VIEW, transitionMs: 1400},
      options: {
        play: false,
        time: 0,
        showTrails: false,
        showDots: false,
        showBackdrop: false,
        background: 'density',
        densityCell: '0.5'
      },
      controls: ['densityCell', 'densityMax', 'backgroundOpacity'],
      readouts: ['density']
    },
    {
      id: 'boundary-currents',
      title: 'The Gulf Stream, the Kuroshio and the Agulhas',
      body: 'Zoom in and the density becomes a map of the **western boundary currents**: narrow, fast rivers in the ocean on the west side of each basin. Pick **Fly to** *Gulf Stream* and look for a ridge that follows the US coast and swings out into the North Atlantic; *Kuroshio* is the same current system off Japan, and the *Agulhas* runs down the east coast of southern Africa before it turns back on itself.\n\nA ridge in this map means many tracks share a path, not that the water is fast: drifters were deployed near ship routes and pile up where currents converge. Switch **Density cell size** to 0.25° to see how narrow the streams are.',
      camera: {longitude: -62, latitude: 38, zoom: 4.1, transitionMs: 1500},
      options: {
        play: false,
        time: 0,
        showTrails: false,
        showDots: false,
        showBackdrop: false,
        background: 'density',
        densityCell: '0.25',
        densityMax: 120,
        region: 'gulfStream'
      },
      callout: {coordinate: [-72, 37], text: 'Gulf Stream'},
      controls: ['region', 'densityCell', 'densityMax'],
      readouts: ['density']
    },
    {
      id: 'limits',
      title: 'What to remember, and what to try',
      body: 'Each drifter appears for **its first 60 days of 2017 only**, with fixes about every 12 hours (the archive is 6-hourly), so this is a sample of the year, not the whole Global Drifter Program record. Drifters are not placed uniformly, and some **lose their drogue** and then slide with the wind; this archive does not say which. The temperature is from a sensor on the buoy hull. The **Date** is UTC.\n\n**Try it:** set **Color tracks by** to *Date* and watch the year go by in the colors of one ocean; set **Trail length** to 60 days with **Playback speed** at 30 days/s; fly to the *Southern Ocean* and look for the buoys circling Antarctica.',
      camera: {longitude: 60, latitude: -56, zoom: 2.4, transitionMs: 1500},
      options: {
        play: true,
        time: 0,
        showTrails: true,
        showDots: true,
        showBackdrop: true,
        background: 'none',
        trailDays: 40,
        speed: 20,
        colorBy: 'date',
        region: 'southern'
      },
      controls: ['colorBy', 'trailDays', 'speed'],
      readouts: ['date', 'active']
    }
  ]
});
