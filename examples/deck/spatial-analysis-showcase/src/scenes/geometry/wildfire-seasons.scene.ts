// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {playbackOptions} from '../../engine/playback';
import {CREDITS, joinCredits} from '../../cartography/credits';
import {ground} from '../../cartography/grounds';
import {getSizeLegendEntries} from '../../cartography/proportional';
import {defineScene} from '../scene';
import {storyFromMarkdown} from '../story-markdown';
import narrative from './wildfire-seasons.md?raw';
import type {WildfireSeasonsOptions} from './wildfire-seasons.compute';

/** `12 Jun 2021` from days since 2020-01-01 (kept here so this light file imports no runtime code). */
const formatWildfireDay = (days: number) => {
  const date = new Date(Date.UTC(2020, 0, 1 + Math.round(days)));
  const month = date.toLocaleString('en-GB', {month: 'short', timeZone: 'UTC'});
  return `${date.getUTCDate()} ${month} ${date.getUTCFullYear()}`;
};

export default defineScene<WildfireSeasonsOptions>({
  id: 'wildfire-seasons',
  title: 'Perimeter records through time',
  chapter: 'geometry',
  order: 6,
  summary:
    'Playback of a western-perimeter subset by record date: GPUTimeWindowFilter windows and fades fires while GPUGeometryMeasures checks mapped acreage.',
  contributors: ['GPUTimeWindowFilter', 'GPUGeometryMeasures'],
  datasets: [{id: 'poopdeck-wildfires', role: 'final fire perimeters, 2020 to 2023'}],
  initialView: {longitude: -116.5, latitude: 40, zoom: 4.2},
  basemap: ground('night'),
  furniture: {
    title: {title: 'Wildfire perimeter records', subtitle: 'Record date, not ignition or growth'},
    credit: joinCredits(CREDITS.usgs, 'NIFC perimeter records (public domain)'),
    caveat: 'The record date is an archive field; this subset is not a fire-growth chronology.',
    clock: {
      option: 'time',
      time: {origin: '2020-01-01T00:00:00Z', unit: 'days'},
      zones: ['UTC'],
      show: 'date',
      progress: [140, 1450]
    }
  },

  options: [
    ...playbackOptions<WildfireSeasonsOptions>({
      time: {
        min: 140,
        max: 1450,
        step: 1,
        default: 140,
        label: 'Date',
        format: formatWildfireDay,
        help: 'Day of the NIFC perimeter record, from 20 May 2020 to 3 Jan 2024. Not the ignition date.'
      },
      playing: false,
      speed: {
        kind: 'select',
        options: [
          {value: '0.5', label: '0.5x (48 s per loop)'},
          {value: '1', label: '1x (24 s per loop)'},
          {value: '2', label: '2x'},
          {value: '4', label: '4x'}
        ],
        default: '1',
        help: 'Playhead speed: 55 days per second at 1x.'
      },
      loop: {default: true}
    }),
    {
      kind: 'slider',
      id: 'trailDays',
      label: 'Trail length',
      group: 'Time window (GPUTimeWindowFilter)',
      apply: 'param',
      min: 10,
      max: 1461,
      step: 10,
      default: 365,
      unit: 'days',
      help: 'Fires stay in the window this many days after their perimeter date. 1,461 days keeps everything once it has appeared. A parameter-buffer write.'
    },
    {
      kind: 'slider',
      id: 'tailFade',
      label: 'Fade of the old end',
      group: 'Time window (GPUTimeWindowFilter)',
      apply: 'param',
      min: 0,
      max: 1,
      step: 0.05,
      default: 0.6,
      disabledWhen: state => state.colorBy !== 'age',
      help: 'Share of the window over which the fade weight ramps from 0 (oldest) to 1. The weight colors fires when Color by is Age.'
    },
    {
      kind: 'select',
      id: 'colorBy',
      label: 'Color by',
      group: 'Display',
      apply: 'param',
      default: 'age',
      help: 'Age uses the time-window fade weight on a ramp; year and acreage class use categories. Switching is a parameter write.',
      options: [
        {value: 'age', label: 'Age (fade weight)'},
        {value: 'year', label: 'Perimeter year'},
        {value: 'sizeClass', label: 'Acreage class'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showOutlines',
      label: 'Outlines',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Ring edges of the fires in the window.'
    },
    {
      kind: 'toggle',
      id: 'showMarkers',
      label: 'Centroid dots',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'A dot at each fire in the window, so small fires stay visible when zoomed out.'
    },
    {
      kind: 'toggle',
      id: 'highlightNewest',
      label: 'Outline the newest fire',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'A yellow outline on the last perimeter before the playhead.'
    },
    {
      kind: 'toggle',
      id: 'follow',
      label: 'Camera follows the newest fire',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Flies to each new perimeter, at most once a second. Turn it off to move the map yourself.'
    },
    {
      kind: 'select',
      id: 'areaSystem',
      label: 'Area system',
      group: 'Measures (GPUGeometryMeasures)',
      apply: 'compile',
      default: 'wgs84',
      help: 'Coordinate system of the area measure behind the acres. planar is Web Mercator and overstates areas by about 1.6 times; wgs84 matches the NIFC acres. Rebuilds the measures graph.',
      options: [
        {value: 'planar', label: 'Planar (Web Mercator meters)'},
        {value: 'spherical', label: 'Spherical'},
        {value: 'wgs84', label: 'WGS84 authalic sphere'},
        {value: 'geodesic', label: 'Geodesic edges'}
      ]
    },
    {
      kind: 'select',
      id: 'groupBy',
      label: 'Group fires by',
      group: 'Measures (GPUGeometryMeasures)',
      apply: 'param',
      default: 'year',
      help: 'The group id of each fire for the per-group area and fire count. The ids are a per-frame buffer: changing them rewrites the buffer and re-measures, with no rebuild.',
      options: [
        {value: 'year', label: 'Perimeter year'},
        {value: 'sizeClass', label: 'Acreage class'}
      ]
    }
  ],

  timeline: {
    time: 'time',
    play: 'play',
    speed: 'speed',
    format: formatWildfireDay,
    ticks: [
      {at: 366, label: '2021'},
      {at: 731, label: '2022'},
      {at: 1096, label: '2023'}
    ]
  },

  story: storyFromMarkdown<WildfireSeasonsOptions>(narrative, {
    records: {
      headline: 'These are final perimeter records',
      camera: {longitude: -116.5, latitude: 40, zoom: 4.2, transitionMs: 1400},
      optionsMode: 'fresh',
      options: {play: false, time: 1450, trailDays: 1461, colorBy: 'age'},
      controls: ['trailDays'],
      readouts: ['firesShown', 'totalGpu', 'totalNifc']
    },
    play: {
      headline: 'Record dates arrive through the seasons',
      camera: {longitude: -116.5, latitude: 40, zoom: 4.2, transitionMs: 1400},
      optionsMode: 'fresh',
      options: {play: false, time: 140, speed: '1', trailDays: 365, tailFade: 0.6, colorBy: 'age'},
      controls: ['play', 'time', 'speed', 'loop'],
      readouts: ['date', 'firesShown', 'cumulative', 'cumulativeChart']
    },
    window: {
      headline: 'A fading window is GPU state',
      camera: {longitude: -120, latitude: 41, zoom: 5.2, transitionMs: 1600},
      optionsMode: 'fresh',
      options: {play: false, time: 263, trailDays: 60, tailFade: 0.8, colorBy: 'age'},
      controls: ['trailDays', 'tailFade', 'time'],
      readouts: ['date', 'firesShown', 'gpuCount', 'newest']
    },
    ledger: {
      headline: 'GPU acres reproduce source polygons',
      camera: {longitude: -116.5, latitude: 40, zoom: 4.2, transitionMs: 1400},
      optionsMode: 'fresh',
      options: {play: false, time: 1450, trailDays: 1461, colorBy: 'age', groupBy: 'year'},
      controls: ['areaSystem'],
      readouts: ['totalGpu', 'totalNifc', 'medianRelativeDifference', 'ledgerChart']
    },
    tour: {
      headline: 'A few records dominate this subset',
      camera: {longitude: -120, latitude: 40, zoom: 5.5, transitionMs: 1400},
      optionsMode: 'fresh',
      options: {
        play: false,
        time: 220,
        speed: '2',
        trailDays: 120,
        tailFade: 0.7,
        colorBy: 'age',
        follow: true
      },
      controls: ['play', 'speed', 'highlightNewest'],
      readouts: ['date', 'newest', 'cumulative']
    }
  }),

  legends: state => {
    const sizeLegend = {
      kind: 'size' as const,
      title: 'Agency acres (symbol area)',
      layout: 'nested' as const,
      entries: getSizeLegendEntries(1_000_000, 16, {
        minRadiusPixels: 2,
        format: value => `${Math.round(value / 1000)}k acres`
      })
    };
    if (state.colorBy === 'age') {
      return [
        sizeLegend,
        {
          kind: 'ramp',
          title: 'Age of the perimeter',
          ramp: 'fire',
          extent: [0, 1],
          labels: ['older', 'newly mapped']
        }
      ];
    }
    if (state.colorBy === 'year') {
      return [
        sizeLegend,
        {
          kind: 'categories',
          title: 'Perimeter year',
          entries: [
            {color: [232, 90, 60, 255], label: '2020'},
            {color: [240, 170, 50, 255], label: '2021'},
            {color: [70, 190, 150, 255], label: '2022'},
            {color: [110, 140, 235, 255], label: '2023'}
          ]
        }
      ];
    }
    return [
      sizeLegend,
      {
        kind: 'categories',
        title: 'Acreage class',
        entries: [
          {color: [110, 140, 235, 255], label: 'under 10k acres'},
          {color: [70, 190, 150, 255], label: '11k to 33k'},
          {color: [240, 170, 50, 255], label: '50k to 97k'},
          {color: [232, 90, 60, 255], label: 'over 300k'}
        ]
      }
    ];
  },

  readouts: [
    {
      id: 'date',
      label: 'Playhead date',
      help: 'Date of the playhead; fires appear when it passes their NIFC perimeter date.'
    },
    {
      id: 'firesShown',
      label: 'Fires in the window',
      format: 'integer',
      help: 'Fires the time window accepts (accepted flags read back).'
    },
    {
      id: 'acresShown',
      label: 'Acres in the window',
      format: 'integer',
      help: 'GPU measured area of the fires in the window.'
    },
    {
      id: 'gpuCount',
      label: 'GPU window count',
      format: 'integer',
      help: 'The count the GPUTimeWindowFilter itself writes.'
    },
    {
      id: 'cumulative',
      label: 'Mapped so far',
      help: 'GPU measured acres of every fire dated at or before the playhead.'
    },
    {
      id: 'newest',
      label: 'Newest perimeter',
      layout: 'block',
      help: 'The last perimeter dated at or before the playhead.'
    },
    {
      id: 'totalGpu',
      label: 'GPU acres, all fires',
      format: 'integer',
      help: 'GPUGeometryMeasures area of every loaded fire in the chosen area system.'
    },
    {
      id: 'totalNifc',
      label: 'NIFC acres, all fires',
      format: 'integer',
      help: 'Sum of the acres attribute.'
    },
    {
      id: 'groupFires',
      label: 'Fires per group',
      help: 'GPUGeometryMeasures featureCounts per group (2020 / 2021 / 2022 / 2023, or the four acreage classes).'
    },
    {id: 'cumulativeChart', label: 'Cumulative acres', kind: 'chart'},
    {id: 'yearChart', label: 'Acres per group', kind: 'chart'},
    {id: 'ledgerChart', label: 'GPU minus agency acres', kind: 'chart'},
    {id: 'medianRelativeDifference', label: 'Median relative difference'}
  ],

  snippet: state => `import {
  getGPUTimeWindowParameterValues,
  GPUTimeWindowFilter
} from '@luma.gl/experimental/gpu-dataframe';
import {GPUGeometryMeasures} from '@luma.gl/experimental/gpu-spatial-analysis';

// Perimeter dates are float32 days since 2020-01-01 (exact below 2^24).
graph.add(new GPUTimeWindowFilter({
  timestamps: days, window,
  output: {ids, count, overflow},
  outputMask,                          // 1 for fires in the window
  fadeWeights                          // 0..1, dims toward the old end
}));
graph.add(new GPUGeometryMeasures({
  positions, ringOffsets, featureRingOffsets,
  geometryType: 'polygons', coordinateSystem: '${state.areaSystem}',
  groupIds, groupCount: 4,             // ${state.groupBy}
  output: {areas},
  groupOutput: {areas: groupAreas, featureCounts}
}));
// per frame, no recompile
windowParameters.write(getGPUTimeWindowParameterValues({
  start: playhead - ${state.trailDays}, end: playhead,
  startFadeDuration: ${Math.round(state.trailDays * state.tailFade)}
}));`,

  about: {
    what: '`GPUTimeWindowFilter` keeps the rows whose time falls inside a per-frame window and writes a stable id list, an accepted mask and fade weights. `GPUGeometryMeasures` measures the area of every polygon feature and of each group of features.',
    why: 'Playback and filtering over time should not rebuild anything on the CPU. A window is eight numbers; the GPU decides which of the fires are in view and how faded they are, and the same measure pass gives the acres per year.',
    howToRead:
      'A fire appears on its NIFC perimeter date, which is usually the last mapping before containment, not the ignition. Bright on the age ramp is newly mapped. The cumulative curve steps up by the measured area of each perimeter. The archive does not contain every large fire of these years, and daily progression perimeters are available key-free only for a handful of them, so this is not a growth animation.'
  },

  create: async ctx => (await import('./wildfire-seasons.compute')).createWildfireSeasons(ctx)
});
