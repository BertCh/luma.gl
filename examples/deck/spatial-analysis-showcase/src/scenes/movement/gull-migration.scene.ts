// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {playbackOptions} from '../../engine/playback';
import {defineScene} from '../scene';
import type {GullMigrationOptions} from './gull-migration.compute';

const EUROPE_AFRICA_VIEW = {longitude: -4.5, latitude: 35, zoom: 3.35};

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

/** `15 Jul` for a day count since 15 July 2015. */
const dateLabel = (day: number) => {
  const date = new Date(Date.UTC(2015, 6, 15) + day * 86400000);
  return `${date.getUTCDate()} ${MONTHS[date.getUTCMonth()]}`;
};

/** Log10 of kilometers as a distance label. */
const toleranceLabel = (log: number) => {
  const kilometers = 10 ** log;
  return kilometers < 10 ? `${kilometers.toFixed(1)} km` : `${kilometers.toFixed(0)} km`;
};

export default defineScene<GullMigrationOptions>({
  id: 'gull-migration',
  title: 'Where do the gulls go, and do they all go the same way?',
  chapter: 'movement',
  order: 10,
  summary:
    'Thirty-one lesser black-backed gulls tracked hourly from Belgium and the Netherlands to Iberia and West Africa in autumn 2015: stopover detection and speeds, Douglas-Peucker against time-aware simplification, route families by Frechet distance, and weekly snapshots from a temporal reduction.',
  contributors: [
    'GPUTrajectoryMetrics',
    'GPULineSimplification',
    'GPUTrackSimilarity',
    'GPUTrajectoryResample',
    'GPUTemporalReduction'
  ],
  datasets: [{id: 'gull-migration', role: 'gull tracks (UvA-BiTS GPS, 2015)'}],
  initialView: EUROPE_AFRICA_VIEW,

  options: [
    {
      kind: 'toggle',
      id: 'showStops',
      label: 'Show stopovers',
      group: 'Stopovers (metrics)',
      apply: 'param',
      default: false,
      help: 'Discs where a bird stayed within a small area: a run of slow hourly steps lasting at least the minimum duration. Radius and color grow with the stay.'
    },
    {
      kind: 'slider',
      id: 'stopSpeed',
      label: 'Stopover speed threshold',
      group: 'Stopovers (metrics)',
      apply: 'param',
      min: 0.25,
      max: 4,
      step: 0.25,
      default: 1.5,
      unit: 'm/s',
      help: 'An hourly step is slow when the bird moves less than this speed times one hour (1.5 m/s is 5.4 km in an hour). A parameter-buffer write.'
    },
    {
      kind: 'slider',
      id: 'stopHours',
      label: 'Minimum stopover',
      group: 'Stopovers (metrics)',
      apply: 'param',
      min: 6,
      max: 240,
      step: 6,
      default: 72,
      unit: 'h',
      help: 'A run of slow steps counts as a stopover when it lasts at least this many hours. 72 h separates breeding-colony and wintering stays from resting days.'
    },
    {
      kind: 'toggle',
      id: 'showSimplified',
      label: 'Show simplified tracks',
      group: 'Simplification',
      apply: 'param',
      default: false,
      help: 'Draws only the vertices that survive the tolerance, as a bold line over the original tracks.'
    },
    {
      kind: 'select',
      id: 'simplifyMetric',
      label: 'Distance measure',
      group: 'Simplification',
      apply: 'compile',
      default: 'segment',
      help: 'Compile-time choice of the importance metric; both variants are compiled up front, so switching only changes which one is drawn.',
      options: [
        {
          value: 'segment',
          label: 'Douglas-Peucker (distance to the chord)',
          help: 'Keeps the vertices that deviate most from the straight line between kept neighbours. Ignores when the bird was there.'
        },
        {
          value: 'time-ratio',
          label: 'TD-TR (time-synchronized distance)',
          help: 'Distance to where a bird flying the chord at constant speed would be at that time. Keeps vertices where the bird sped up, slowed down or stopped.'
        }
      ]
    },
    {
      kind: 'slider',
      id: 'toleranceLog',
      label: 'Tolerance',
      group: 'Simplification',
      apply: 'param',
      min: -0.5,
      max: 2.5,
      step: 0.05,
      default: 1,
      format: toleranceLabel,
      help: 'Vertices whose importance is below this distance are dropped. Log scale from 0.3 km to 316 km. The importance rounds run once; moving the tolerance re-runs only the selection.'
    },
    {
      kind: 'select',
      id: 'routeSpacing',
      label: 'Route samples spaced by',
      group: 'Route similarity',
      apply: 'compile',
      default: 'arc-length',
      help: 'Each track is resampled to 96 points before comparison (a different compiled resample graph each). Equal distance compares the shape of the route; equal time also weighs where the bird spent its time.',
      options: [
        {value: 'arc-length', label: 'Equal distance along the path'},
        {value: 'time', label: 'Equal time steps'}
      ]
    },
    {
      kind: 'select',
      id: 'similarityMetric',
      label: 'Route distance',
      group: 'Route similarity',
      apply: 'param',
      default: 'frechet',
      help: 'Both distances are computed for every pair in one pass; this only picks the column that is clustered and colored.',
      options: [
        {
          value: 'frechet',
          label: 'Frechet',
          help: 'Shortest leash for two walkers following the routes in order. Sensitive to timing and direction.'
        },
        {
          value: 'hausdorff',
          label: 'Hausdorff',
          help: 'Farthest any point of one route is from the other. Ignores order.'
        }
      ]
    },
    {
      kind: 'slider',
      id: 'familyCount',
      label: 'Number of route families',
      group: 'Route similarity',
      apply: 'param',
      min: 2,
      max: 6,
      step: 1,
      default: 3,
      help: 'Average-linkage clustering of the route distances, on the CPU (the distance matrix is only 31 x 31). Families are ordered from the most northerly to the most southerly last fix.'
    },
    {
      kind: 'slider',
      id: 'similarityRangeKm',
      label: 'Distance color range',
      group: 'Route similarity',
      apply: 'param',
      min: 200,
      max: 3000,
      step: 100,
      default: 1500,
      unit: 'km',
      disabledWhen: state => state.colorBy !== 'similarity',
      help: 'Route distance at the top of the ramp when tracks are colored by distance to the selected bird (click a track to select).'
    },
    {
      kind: 'toggle',
      id: 'showWeekly',
      label: 'Show weekly snapshots',
      group: 'Time aggregation',
      apply: 'param',
      default: false,
      help: 'One dot per bird per time bucket (its last fix in the bucket), joined by lines, from a GPUTemporalReduction of the hourly fixes. The big ring marks the snapshot bucket.'
    },
    {
      kind: 'slider',
      id: 'bucketDays',
      label: 'Bucket width',
      group: 'Time aggregation',
      apply: 'param',
      min: 3,
      max: 14,
      step: 1,
      default: 7,
      unit: 'days',
      help: 'Width of one time bucket, a parameter-buffer write: 98,000 hourly fixes are reduced to a handful of rows per bird. At most 64 buckets, so the minimum is 3 days.'
    },
    {
      kind: 'range',
      id: 'dayRange',
      label: 'Days shown',
      group: 'Time aggregation',
      apply: 'param',
      min: 0,
      max: 138,
      step: 1,
      default: [0, 138],
      format: value => dateLabel(value),
      help: 'Buckets outside this range are hidden.'
    },
    ...playbackOptions<GullMigrationOptions>({
      ids: {play: 'play', time: 'snapshotDay', speed: 'playSpeed', loop: 'loop'},
      group: 'Time aggregation',
      playing: false,
      time: {
        min: 0,
        max: 138,
        step: 1,
        default: 62,
        label: 'Snapshot day',
        format: dateLabel,
        help: 'The bucket containing this day is ringed and summarized in the readouts: how many birds report, where the median bird is and how fast the fastest hourly step was. Press Play to sweep it through the season.'
      },
      speed: {
        min: 1,
        max: 20,
        step: 1,
        default: 4,
        unit: 'days/s',
        label: 'Play speed',
        help: 'Days of the season per real second. At 4 days/s the whole autumn takes about 35 seconds.'
      },
      loop: true
    }),
    {
      kind: 'select',
      id: 'colorBy',
      label: 'Color tracks by',
      group: 'Display',
      apply: 'param',
      default: 'date',
      help: 'Date shows the migration unfolding; the others are outputs of the contributors below.',
      options: [
        {value: 'date', label: 'Date (15 Jul to 30 Nov)'},
        {value: 'speed', label: 'Ground speed of each step'},
        {value: 'family', label: 'Route family (similarity clusters)'},
        {value: 'similarity', label: 'Route distance to the selected bird'},
        {value: 'sex', label: 'Sex'},
        {value: 'plain', label: 'One color'}
      ]
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp',
      group: 'Display',
      apply: 'param',
      default: 'viridis',
      help: 'Used for dates, speeds, route distance and the weekly snapshots.',
      options: [
        {value: 'viridis', label: 'Viridis'},
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'cividis', label: 'Cividis (color-blind optimised)'}
      ]
    },
    {
      kind: 'slider',
      id: 'trackOpacity',
      label: 'Track opacity',
      group: 'Display',
      apply: 'param',
      min: 0.1,
      max: 1,
      step: 0.05,
      default: 0.85,
      help: 'Lower it to make the overlays stand out.'
    }
  ],

  readouts: [
    {
      id: 'latitudeChart',
      label: 'The migration as one curve',
      kind: 'chart',
      help: 'Median latitude of the birds in each time bucket (line) with the 10th to 90th percentile band, from GPUTemporalReduction. The vertical rule is the snapshot day. The staircase is the autumn push south, with the band widening as birds pick different wintering sites.'
    },
    {
      id: 'speedChart',
      label: 'Ground speed of every fix',
      kind: 'chart',
      help: 'Speed reported by the tag at each hourly fix. The hump near zero is resting and foraging; the second hump at 30 to 50 km/h is flight.'
    },
    {
      id: 'stopChart',
      label: 'How long do stopovers last?',
      kind: 'chart',
      help: 'Duration of every stopover found at the current speed threshold and minimum stay.'
    },
    {
      id: 'familyChart',
      label: 'Route family sizes',
      kind: 'chart',
      help: 'Birds in each Frechet or Hausdorff route family, labelled by the mean latitude where the family ended. Families are numbered from north to south.'
    },
    {id: 'birds', label: 'Birds'},
    {id: 'period', label: 'Period'},
    {
      id: 'distance',
      label: 'Distance flown',
      help: "Sum of every bird's path length, from GPUTrajectoryMetrics in azimuthal-equidistant meters (true distance)."
    },
    {id: 'longestFlight', label: 'Longest journey'},
    {
      id: 'fastest',
      label: 'Fastest step',
      help: 'Highest ground speed between two consecutive hourly fixes.'
    },
    {id: 'stops', label: 'Stopovers', help: 'Number of stopovers at the current thresholds.'},
    {id: 'longestStop', label: 'Longest stopover'},
    {
      id: 'simplified',
      label: 'Vertices kept',
      help: 'Vertices kept by Douglas-Peucker and by TD-TR at the same tolerance. A star means the importance rounds did not converge, so the result is a superset of the exact simplification.'
    },
    {id: 'tolerance', label: 'Tolerance'},
    {
      id: 'rounds',
      label: 'Importance rounds',
      help: 'Level-synchronous rounds used by the selected measure, out of the compile-time cap.'
    },
    {id: 'routes', label: 'Route comparison'},
    {
      id: 'families',
      label: 'Route families',
      help: 'Birds per family and the mean latitude of the last fix of each family.'
    },
    {id: 'selected', label: 'Selected bird'},
    {id: 'reduction', label: 'Temporal reduction'},
    {id: 'snapshot', label: 'Snapshot bucket'},
    {id: 'snapshotBirds', label: 'Birds reporting'},
    {id: 'snapshotLatitude', label: 'Latitude in the bucket'},
    {id: 'snapshotSpeed', label: 'Fastest hourly step in the bucket'}
  ],

  legends: state => [
    ...(state.colorBy === 'date'
      ? [
          {
            kind: 'ramp' as const,
            title: 'Date (autumn 2015)',
            ramp: state.ramp,
            extent: [0, 138] as const,
            labels: ['15 Jul', '30 Nov'] as const
          }
        ]
      : []),
    ...(state.colorBy === 'speed'
      ? [
          {
            kind: 'ramp' as const,
            title: 'Ground speed of each hourly step',
            ramp: state.ramp,
            extent: [0, 60] as const,
            unit: 'km/h',
            format: (value: number) => value.toFixed(0)
          }
        ]
      : []),
    ...(state.colorBy === 'similarity'
      ? [
          {
            kind: 'ramp' as const,
            title: `Route ${state.similarityMetric === 'frechet' ? 'Frechet' : 'Hausdorff'} distance to the selected bird`,
            ramp: state.ramp,
            extent: [0, state.similarityRangeKm] as const,
            unit: 'km',
            format: (value: number) => value.toFixed(0)
          }
        ]
      : []),
    ...(state.colorBy === 'family'
      ? [
          {
            kind: 'categories' as const,
            title: 'Route family (north to south by last fix)',
            entries: [
              [86, 180, 233, 255],
              [240, 150, 30, 255],
              [0, 158, 115, 255],
              [204, 121, 167, 255],
              [240, 228, 66, 255],
              [150, 156, 168, 255]
            ]
              .slice(0, state.familyCount)
              .map((color, index) => ({
                color: color as [number, number, number, number],
                label: `Family ${index + 1}`
              }))
          }
        ]
      : []),
    ...(state.colorBy === 'sex'
      ? [
          {
            kind: 'categories' as const,
            title: 'Sex',
            entries: [
              {color: [230, 120, 160, 255] as const, label: 'Female (19)'},
              {color: [80, 150, 240, 255] as const, label: 'Male (12)'}
            ]
          }
        ]
      : []),
    ...(state.showSimplified
      ? [
          {
            kind: 'categories' as const,
            title: 'Simplified track',
            entries: [
              state.simplifyMetric === 'segment'
                ? {color: [255, 150, 40, 255] as const, label: 'Douglas-Peucker'}
                : {color: [255, 70, 150, 255] as const, label: 'TD-TR (time ratio)'}
            ]
          }
        ]
      : []),
    ...(state.showStops
      ? [
          {
            kind: 'categories' as const,
            title: 'Stopovers (radius and color grow with the stay)',
            entries: [
              {color: [255, 199, 224, 255] as const, label: 'A few days'},
              {color: [255, 41, 128, 255] as const, label: 'About two weeks'},
              {color: [191, 0, 51, 255] as const, label: 'Three weeks or more'}
            ]
          }
        ]
      : []),
    ...(state.showWeekly
      ? [
          {
            kind: 'ramp' as const,
            title: `Weekly snapshots (${state.bucketDays}-day buckets, last fix of each)`,
            ramp: state.ramp,
            extent: [0, 138] as const,
            labels: ['15 Jul', '30 Nov'] as const
          }
        ]
      : [])
  ],

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUTrajectoryMetrics, GPULineSimplification, GPUTrajectoryResample, GPUTrackSimilarity,
  getGPUTrajectoryMetricsParameterValues, getGPULineSimplificationParameterValues
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUTemporalReduction, getGPUTemporalReductionParameterValues} from '@luma.gl/experimental/gpu-dataframe';

// positions: azimuthal-equidistant meters (true distances over 40 degrees of latitude),
// timestamps: float32 seconds since 15 July, trackOffsets: uint32 (31 birds + 1)

// 1. Metrics and stopovers
graph.add(new GPUTrajectoryMetrics({
  positions, timestamps, trackOffsets, parameters: stopParameters.importToGraph(graph),
  trackLengths, averageSpeeds, maximumSpeeds, stepSpeeds, trackStopCounts,
  stops: {output: {ids, count, overflow}, centroids, durations, startRows, endRows}
}));
stopParameters.write(getGPUTrajectoryMetricsParameterValues({
  stopSpeedThreshold: ${state.stopSpeed}, stopMinimumDuration: ${state.stopHours * 3600}   // m/s, s
}));

// 2. Simplification: importance once, selection per tolerance
importanceGraph.add(new GPULineSimplification({
  positions, trackOffsets, importance, metric: '${state.simplifyMetric}',
  timestamps,                                       // needed for 'time-ratio'
  maximumRounds: 96
}));
selectionGraph.add(new GPULineSimplification({
  positions, trackOffsets, importance, metric: '${state.simplifyMetric}', computeImportance: false,
  parameters: toleranceParameters.importToGraph(selectionGraph),
  selection: {output: {ids: keptIds, count: keptCount, overflow}}
}));
toleranceParameters.write(getGPULineSimplificationParameterValues({tolerance: ${Math.round(10 ** state.toleranceLog * 1000)}}));   // meters

// 3. Routes: 96 samples each, then every pair
graph.add(new GPUTrajectoryResample({positions, timestamps, trackOffsets, sampleCount: 96, spacing: '${state.routeSpacing}', samples: routes}));
graph.add(new GPUTrackSimilarity({
  positionsA: routes, offsetsA: routeOffsets, pairA, pairB,   // 465 pairs
  hausdorff, frechet, maxFrechetVertices: 96
}));

// 4. One row per (bird, bucket of ${state.bucketDays} days)
graph.add(new GPUTemporalReduction({
  cellIds: birdOfFix, timestamps, values: latitudeOfFix,
  parameters: bucketParameters.importToGraph(graph), cellCount: 31, bucketCount: 64,
  output: {counts, min, max, first, last, occupiedSlots}
}));
bucketParameters.write(getGPUTemporalReductionParameterValues(0, ${state.bucketDays * 86400}));   // origin, width (s)`,

  about: {
    what: '`GPUTrajectoryMetrics` measures every bird (distance, duration, speeds, stopovers). `GPULineSimplification` computes a per-vertex importance once (Douglas-Peucker, or the time-aware TD-TR) and selects the kept vertices for any tolerance. `GPUTrajectoryResample` and `GPUTrackSimilarity` compare whole routes with the Hausdorff and Frechet distances, and `GPUTemporalReduction` reduces hourly fixes to one row per bird and time bucket.',
    why: 'Animal tracking produces enormous, regular GPS series. Ecologists need journey statistics, a lighter copy of each track that keeps the story, a way to group birds by route, and a view of where the population is at a given week. Doing these on the GPU keeps every slider live on 98,000 fixes.',
    howToRead:
      'Each line is one gull. With date coloring the early purple tracks are the departures from the North Sea coast and the yellow ones are birds still moving in November. Pink discs are multi-day stopovers (colonies, estuaries, wintering sites), bigger for longer stays. The bold orange or pink line is the simplified track. Distances in the readouts are true distances; the map is Web Mercator.'
  },

  create: async ctx => (await import('./gull-migration.compute')).createGullMigration(ctx),

  story: [
    {
      id: 'the-question',
      title: "Where do Zeebrugge's gulls spend the winter?",
      body: 'In summer 2015, 31 adult lesser black-backed gulls from the Belgian and Dutch coast carried GPS tags that logged a fix every hour. From July to the end of November they left the North Sea and spread south: **18 of the 31 reached south of 38 N** (Iberia and Morocco) and four crossed 30 N into the Sahara coast, the furthest to 14.0 N off Senegal.\n\nLines are colored by **date** (**Color tracks by**, below): purple tracks are the first miles in July, yellow the last in November. The thick line is the bird that went furthest. Do they all follow one route, or are there several? The next steps build the tools to answer it.',
      camera: {...EUROPE_AFRICA_VIEW, transitionMs: 1200},
      highlight: {readout: 'selected'},
      controls: ['colorBy'],
      readouts: ['selected', 'speedChart']
    },
    {
      id: 'metrics',
      title: 'Measuring a journey, and finding the stopovers',
      body: '**`GPUTrajectoryMetrics`** gives every bird its path length, duration and average and maximum speed, and a speed for every hourly step, which colors the tracks now: gulls cruise at 30 to 60 km per hour in a migration push and almost stand still the rest of the time.\n\nIt also finds **stopovers**: runs of slow steps that last at least the minimum stay. The discs mark where the birds stayed 72 hours or more within a few kilometers: breeding colonies at the start, estuaries on the way and wintering sites at the end. The numbers are true distances: the analysis runs in azimuthal-equidistant meters, not on flat map units.\n\nChange **Stopover speed threshold** and **Minimum stopover** below to see how the definition decides where the discs appear.',
      options: {colorBy: 'speed', showStops: true},
      highlight: {readout: 'stops'},
      controls: ['colorBy', 'showStops', 'stopSpeed', 'stopHours'],
      readouts: ['stops', 'longestStop', 'fastest', 'stopChart', 'speedChart']
    },
    {
      id: 'douglas-peucker',
      title: 'A lighter copy of every track',
      body: 'Hourly fixes mean about 3,000 vertices per bird. **`GPULineSimplification`** keeps only the vertices that matter. Douglas-Peucker measures how far each vertex lies from the straight chord between its kept neighbours; a vertex is kept when that distance exceeds the **tolerance**. At 10 km the bold orange tracks keep only a few percent of the vertices and are hard to tell from the originals.\n\nThe expensive part, an importance for every vertex, is computed **once**; moving the **Tolerance** slider re-runs only a mask and a compaction on the GPU. Read **Vertices kept** to compare, and watch the readout **Importance rounds** to see whether the result is exact (converged) or an over-estimate (superset).',
      options: {showSimplified: true, simplifyMetric: 'segment', toleranceLog: 1, colorBy: 'plain'},
      highlight: {readout: 'simplified'},
      controls: ['showSimplified', 'toleranceLog'],
      readouts: ['simplified', 'rounds', 'tolerance']
    },
    {
      id: 'time-ratio',
      title: 'Simplify with time in mind',
      body: 'Douglas-Peucker looks only at shape: a bird that sat for a week at one spot adds nothing. **TD-TR** (time-ratio, Meratnia and de By 2004) asks instead **how far the bird is from where a bird flying the chord at constant speed would be at that moment**. A stopover then costs importance, so the simplified track keeps the vertices around it.\n\nSwitch **Distance measure** to *TD-TR*: more vertices survive at the same tolerance, clustered where the bird sped up, slowed or stopped. Turn on **Show stopovers** and see them line up with the kept vertices.',
      options: {
        showSimplified: true,
        simplifyMetric: 'time-ratio',
        toleranceLog: 1,
        colorBy: 'plain',
        showStops: true
      },
      controls: ['simplifyMetric', 'toleranceLog', 'showStops'],
      readouts: ['simplified']
    },
    {
      id: 'families',
      title: 'Do they all take the same route?',
      body: 'To compare routes, **`GPUTrajectoryResample`** rebuilds each track as 96 points, and **`GPUTrackSimilarity`** scores all 465 pairs in one pass with the discrete **Frechet distance**: the shortest leash that lets two walkers follow their routes in order. It follows Shapely `frechet_distance` without densification.\n\nThe 31 x 31 distance matrix is small, so the CPU clusters it into **three route families** (average linkage), drawn in three colors, from the most northerly last fix to the most southerly. Move **Number of route families**, switch **Route distance** to *Hausdorff*, or set **Color tracks by** to *Route distance to the selected bird* and click any track to ask who flies like it.',
      options: {
        colorBy: 'family',
        familyCount: 3,
        similarityMetric: 'frechet',
        showSimplified: false,
        showStops: false
      },
      highlight: {readout: 'families'},
      controls: ['familyCount', 'similarityMetric', 'colorBy'],
      readouts: ['families', 'familyChart', 'routes', 'selected']
    },
    {
      id: 'snapshots',
      title: 'Where is everyone in the second week of September?',
      body: "**`GPUTemporalReduction`** reduces 98,000 hourly fixes to one row per bird and time bucket: first, last, minimum and maximum value and a count. Here the values are longitude, latitude and ground speed, and the bucket is 7 days. Each dot is a bird's last fix of a week; lines join the weeks.\n\nThe ringed dots are the **Snapshot day**, and the readouts summarize the bucket: how many birds are reporting, the median latitude, how many are already south of 40 N, and the fastest hourly step. Press **Play** (or drag **Snapshot day**) from July to November and watch the median bird move south: the ringed dots sweep down the map while the rule on the chart moves along the curve of median latitude. The curve is flat through the breeding season, then drops in steps as the birds leave, and the band widens when families split between Iberia and Africa. The bucket width is a parameter write, so **Bucket width** changes the reduction without a recompile.",
      options: {
        showWeekly: true,
        colorBy: 'plain',
        snapshotDay: 62,
        showSimplified: false,
        showStops: false,
        trackOpacity: 0.35
      },
      highlight: {readout: 'snapshotLatitude'},
      controls: ['play', 'snapshotDay', 'bucketDays', 'dayRange'],
      readouts: ['snapshot', 'snapshotBirds', 'snapshotLatitude', 'latitudeChart']
    },
    {
      id: 'limits',
      title: 'Limits, and what to try',
      body: 'Hourly GPS means short flights and brief stops hide between fixes, and a tag that failed or a bird that died looks like an early arrival. These 31 adults all come from a few colonies, so the families describe this sample, not the species. The clustering is a quick average-linkage on a small matrix, and Frechet on 96 resampled points describes the shape of a route, not every detour. Simplification does not preserve topology, so tracks may touch or cross after simplification. All birds are adults: there are no juveniles in this study.\n\n**Try it:** move **Stopover speed threshold** up to 3 m/s and watch colonies fill with discs; raise **Tolerance** to 100 km and flip **Distance measure** to compare Douglas-Peucker with TD-TR; set **Bucket width** to 3 days and **Snapshot day** to 27 Oct (day 104).',
      options: {colorBy: 'date', showStops: true, showSimplified: true, showWeekly: true},
      camera: {...EUROPE_AFRICA_VIEW, transitionMs: 1200},
      controls: ['stopSpeed', 'toleranceLog', 'simplifyMetric', 'bucketDays', 'snapshotDay']
    }
  ]
});
