// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {formatPlaybackTime, playbackOptions} from '../../engine/playback';
import {joinCredits} from '../../cartography/credits';
import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import {getVesselLegendEntries} from './b12-tracks';
import type {VesselEncountersOptions} from './vessel-encounters.compute';
import {MOVEMENT_CREDITS} from './movement-style';

const UPPER_BAY = {longitude: -74.035, latitude: 40.672, zoom: 11.6};

export default defineScene<VesselEncountersOptions>({
  id: 'vessel-encounters',
  title: 'Which vessels meet, and do they share a route?',
  chapter: 'movement',
  order: 2,
  summary:
    'Put 897 AIS tracks on one clock, list every pair that comes within a distance of each other, and score how alike each pair of whole routes is with Hausdorff and Frechet distances. Click a vessel to colour every track by its route similarity.',
  contributors: [
    'GPUTrajectoryEncounters',
    'addClockEncounters',
    'GPUTrackSimilarity',
    'GPUTrajectoryResample',
    'GPUTrajectoryMetrics'
  ],
  datasets: [{id: 'ais-vessels', role: 'vessel tracks (AIS, 12 June 2024)'}],
  initialView: UPPER_BAY,
  basemap: ground('night', {labels: 'none'}),
  furniture: {
    title: {
      title: 'Who meets whom in New York Harbor?',
      subtitle: 'Space-time close approaches · 12 June 2024',
      chips: ['Close approach is not an encounter']
    },
    scaleBar: {units: 'nautical'},
    credit: joinCredits(MOVEMENT_CREDITS.harborAis),
    clock: {
      option: 'timeOfDay',
      time: {origin: '2024-06-12T00:00:00Z', unit: 'seconds'},
      zones: ['America/New_York', 'UTC'],
      progress: [0, 86400]
    }
  },

  options: [
    {
      kind: 'slider',
      id: 'clockStepSeconds',
      label: 'Clock step',
      group: 'Shared clock',
      apply: 'param',
      min: 30,
      max: 180,
      step: 10,
      default: 120,
      unit: 's',
      help: 'Seconds between the 720 time buckets. Encounters are tested only at the buckets, so a step that is long compared with the distance travelled can miss two vessels passing each other between buckets. 120 s covers 24 hours; 60 s covers 12 hours.'
    },
    {
      kind: 'slider',
      id: 'clockStartHour',
      label: 'Clock starts at',
      group: 'Shared clock',
      apply: 'param',
      min: 0,
      max: 23,
      step: 1,
      default: 0,
      format: value => `${String(value).padStart(2, '0')}:00 UTC`,
      help: 'First instant of the shared clock. With a step under 120 s the clock covers less than a day, so move the start to the hours you care about.'
    },
    {
      kind: 'slider',
      id: 'distance',
      label: 'Encounter distance',
      group: 'Encounters',
      apply: 'param',
      min: 10,
      max: 400,
      step: 5,
      default: 100,
      unit: 'm',
      help: 'Two vessels meet in a time bucket when they are at most this far apart. A parameter buffer, clamped to the lattice cell size (400 m, fixed at compile time).'
    },
    {
      kind: 'slider',
      id: 'minSpeedKnots',
      label: 'Ignore vessels slower than',
      group: 'Encounters',
      apply: 'param',
      min: 0,
      max: 5,
      step: 0.1,
      default: 1,
      unit: 'kn',
      help: 'Moored and anchored vessels sit within meters of each other all day and would swamp the list. A GPU kernel masks tracks whose mean speed (from GPUTrajectoryMetrics) is below this; 0 keeps every track.'
    },
    {
      kind: 'select',
      id: 'similarityMetric',
      label: 'Route distance',
      group: 'Route similarity',
      apply: 'param',
      default: 'hausdorff',
      help: 'Both are computed for every pair in one pass, so switching is only a different column.',
      options: [
        {
          value: 'hausdorff',
          label: 'Hausdorff',
          help: 'The farthest any point of one route is from the other route. Ignores order and direction.'
        },
        {
          value: 'frechet',
          label: 'Frechet',
          help: 'The shortest leash that lets two walkers follow their routes in order. Sensitive to direction.'
        }
      ]
    },
    {
      kind: 'slider',
      id: 'alikeMeters',
      label: 'Routes are alike within',
      group: 'Route similarity',
      apply: 'param',
      min: 250,
      max: 5000,
      step: 250,
      default: 1500,
      unit: 'm',
      help: 'Distance under which two whole routes count as the same route for the readouts and the connector filter.'
    },
    {
      kind: 'select',
      id: 'connectorFilter',
      label: 'Connectors for',
      group: 'Route similarity',
      apply: 'param',
      default: 'all',
      help: 'Keep only the meetings between vessels on alike routes (traveling together), or only those on different routes (crossing paths).',
      options: [
        {value: 'all', label: 'All meetings'},
        {value: 'alike', label: 'Alike routes only (traveling together)'},
        {value: 'different', label: 'Different routes only (crossing paths)'}
      ]
    },
    {
      kind: 'select',
      id: 'trackColor',
      label: 'Color tracks by',
      group: 'Route similarity',
      apply: 'param',
      default: 'category',
      help: 'Vessel type, or the route distance from the selected (white) vessel to every other track. Click any vessel to change the selection.',
      options: [
        {value: 'category', label: 'Vessel type'},
        {value: 'similarity', label: 'Route distance to the selected vessel'},
        {value: 'off', label: 'Hide tracks'}
      ]
    },
    {
      kind: 'slider',
      id: 'colorRangeMeters',
      label: 'Color range',
      group: 'Route similarity',
      apply: 'param',
      min: 1000,
      max: 10000,
      step: 500,
      default: 5000,
      unit: 'm',
      help: 'Route distance at the top of the color ramp, for both the connectors and the tracks.'
    },
    {
      kind: 'button',
      id: 'selectFerry',
      label: 'Reselect the Staten Island ferry',
      group: 'Route similarity',
      help: 'Selects the passenger track that serves both Staten Island Ferry terminals.'
    },
    ...playbackOptions<VesselEncountersOptions>({
      ids: {play: 'play', time: 'timeOfDay', speed: 'playSpeed', loop: 'loop'},
      group: 'View',
      playing: false,
      disabledWhen: state => state.connectorTime === 'first',
      time: {
        min: 0,
        max: 86280,
        step: 120,
        default: 68400,
        label: 'Show connectors at',
        format: formatPlaybackTime.clockUtc,
        help: 'Time bucket at which the connectors and vessel dots are drawn (the nearest bucket of the clock). 68,400 s is 19:00 UTC, 3 pm in New York. Play sweeps through the shared clock window, so meetings appear and dissolve as vessels pass.'
      },
      speed: {
        min: 60,
        max: 1800,
        step: 60,
        default: 300,
        unit: 'x',
        help: 'Simulated seconds per real second. 300x plays an hour in 12 seconds.'
      },
      loop: true
    }),
    {
      kind: 'select',
      id: 'connectorTime',
      label: 'Connectors drawn',
      group: 'View',
      apply: 'param',
      default: 'bucket',
      help: 'Either the pairs that are close at the chosen time, or every pair at the moment it first came within the distance (many more).',
      options: [
        {value: 'bucket', label: 'At the chosen time'},
        {value: 'first', label: "At each pair's first meeting"}
      ]
    },
    {
      kind: 'toggle',
      id: 'showVessels',
      label: 'Show vessels at that time',
      group: 'View',
      apply: 'param',
      default: true,
      help: 'Dots at every vessel position on the shared clock, colored by type. Vessels outside their own track time are absent (NaN) and not drawn.'
    }
  ],

  readouts: [
    {
      id: 'meetingsChart',
      label: 'When do vessels first meet?',
      kind: 'chart',
      help: 'Encounter pairs by the UTC hour in which they first came within the distance. The highlighted bar is the hour of the time on the map; press Play to sweep it.'
    },
    {
      id: 'approachChart',
      label: 'How close do pairs get?',
      kind: 'chart',
      help: 'Smallest distance reached by each pair, from the closest-approach column of GPUTrajectoryEncounters.'
    },
    {
      id: 'similarityChart',
      label: 'Do meeting vessels share a route?',
      kind: 'chart',
      help: 'Whole-route distance for every encounter pair (Hausdorff or Frechet). Pairs left of the alike rule are traveling together; the long tail to the right is independent traffic that merely crossed.'
    },
    {
      id: 'selectedChart',
      label: 'Routes compared with the selected vessel',
      kind: 'chart',
      help: 'Route distance from the selected vessel to all 897 tracks. The spike at small distances is vessels on its lane.'
    },
    {id: 'tracks', label: 'Tracks'},
    {
      id: 'tracksUsed',
      label: 'Tracks that move',
      help: 'Tracks whose mean speed passes the speed filter.'
    },
    {id: 'clock', label: 'Shared clock'},
    {
      id: 'pairs',
      label: 'Encounter pairs',
      help: 'Distinct pairs of tracks that were within the distance in at least one bucket. Capacity is 16,384 pairs.'
    },
    {
      id: 'closest',
      label: 'Closest approach',
      help: 'Smallest distance between any two tracks at a bucket time.'
    },
    {
      id: 'longest',
      label: 'Longest time together',
      help: 'The largest number of buckets a pair stayed within the distance, times the clock step.'
    },
    {id: 'busiestHour', label: 'Busiest hour for meetings'},
    {
      id: 'typePairs',
      label: 'Most common type pairs',
      help: 'Counts of encounter pairs by the types of the two vessels.'
    },
    {
      id: 'alike',
      label: 'Pairs on alike routes',
      help: 'Share of encounter pairs whose whole routes are within the alike distance.'
    },
    {id: 'selected', label: 'Selected vessel'},
    {
      id: 'routesLikeSelected',
      label: 'Routes like the selected one',
      help: 'Tracks whose route is within the alike distance of the selected vessel.'
    }
  ],

  legends: state => [
    {
      kind: 'ramp' as const,
      title: `Route ${state.similarityMetric === 'frechet' ? 'Frechet' : 'Hausdorff'} distance (connectors${state.trackColor === 'similarity' ? ' and tracks' : ''})`,
      ramp: 'magma' as const,
      extent: [0, state.colorRangeMeters] as const,
      unit: 'm',
      format: (value: number) => value.toFixed(0)
    },
    {
      kind: 'categories' as const,
      title: 'Vessel type',
      entries: getVesselLegendEntries(),
      note: 'Dots at the chosen time and, optionally, the tracks.'
    }
  ],

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  addClockEncounters, GPUTrackSimilarity, GPUTrajectoryResample,
  getGPUTrajectoryClockParameterValues
} from '@luma.gl/experimental/gpu-spatial-analysis';

// Whole routes as 64 arc-length samples, once (input of the similarity measures)
routeGraph.add(new GPUTrajectoryResample({
  positions, timestamps, trackOffsets, sampleCount: 64, spacing: 'arc-length', samples: routes
}));

const graph = new GPUCommandGraph(device, {id: 'encounters'});
const pairs = {output: {ids, count, overflow}, partners, firstBuckets, minimumDistances, bucketCounts};
addClockEncounters(graph, {
  positions: maskedPositions, timestamps, trackOffsets, // fixes of vessels slower than ${state.minSpeedKnots} kn set to NaN by a kernel
  clock: clock.importToGraph(graph), bucketCount: 720,  // shared instants start + k * step
  distance: distance.importToGraph(graph),              // per frame, <= cellSize
  cellSize: 400, bounds, hitCapacity: ${1 << 21},       // compile-time lattice
  pairs
});
graph.add(new GPUTrackSimilarity({
  positionsA: routes, offsetsA: routeOffsets,
  pairA: pairs.output.ids, pairB: pairs.partners, activePairCount: pairs.output.count,
  hausdorff, frechet, maxFrechetVertices: 64            // one pass, both measures
}));
const compiled = graph.compile();                       // once

// the sliders are buffer writes
distance.write(Float32Array.of(${state.distance}));
clock.write(getGPUTrajectoryClockParameterValues({start: ${state.clockStartHour * 3600}, step: ${state.clockStepSeconds}}));
compiled.encode(commandEncoder, {parameters: undefined});`,

  about: {
    what: '`addClockEncounters` evaluates every vessel at the same instants (a `GPUTrajectoryResample` with a shared clock; a vessel is absent outside its own time span) and `GPUTrajectoryEncounters` indexes those samples in a space-and-time grid to list the pairs within a distance of each other in the same bucket. `GPUTrackSimilarity` then scores each pair of whole routes with the discrete Hausdorff and Frechet distances.',
    why: 'Meetings are where risk and interaction happen: collision avoidance, tugs and tows rendezvous, ferries sharing a lane, escorts. Knowing *whether the two vessels were on the same route* separates convoys and escorts (alike routes) from vessels that merely crossed paths.',
    howToRead:
      'Colored lines join vessels that are within the distance at the chosen time. A purple connector joins vessels whose whole-day routes nearly coincide; a yellow one joins vessels from very different routes. The white line is the selected vessel; with tracks colored by route distance, the lanes shared with it glow purple.'
  },

  create: async ctx => (await import('./vessel-encounters.compute')).createVesselEncounters(ctx),

  story: [
    {
      id: 'the-question',
      headline: 'Close approaches cluster on ferry lanes',
      textAlternative: 'Pairs of vessels are connected at a shared harbor clock.',
      optionsMode: 'fresh',
      title: 'Which vessels meet in the harbor, and do they share a route?',
      body: 'Tugs meet barges, ferries pass each other mid-channel, a pilot boat comes alongside a tanker. On 12 June 2024 the Upper Bay carried hundreds of tracks at once, and the interesting question is not just *who came close* but *whether they were on the same route*.\n\nThe colored tracks are the whole AIS day by vessel type; the thick white line is a Staten Island ferry. Each colored connector joins two vessels that are **within 100 m of each other at 19:00 UTC (3 pm)** (**Encounter distance** and **Show connectors at**, below). We will build that list in four steps: a shared clock, a distance, a mask for moored boats, and a route similarity score.',
      camera: {...UPPER_BAY, transitionMs: 1200},
      highlight: {readout: 'pairs'},
      controls: ['distance', 'timeOfDay', 'play'],
      readouts: ['pairs', 'closest', 'meetingsChart']
    },
    {
      id: 'clock',
      headline: 'A meeting needs place and time',
      textAlternative: 'A space-time lattice bins vessels at the same minute.',
      optionsMode: 'fresh',
      title: 'Everyone on one clock',
      body: 'Vessels report at different instants, so they cannot be compared directly. **`addClockEncounters`** first resamples every track onto **one shared clock** (`GPUTrajectoryResample` in clock mode): bucket *k* is the instant *start + k x step*, and a vessel is *absent* (NaN) outside its own track time. Then **`GPUTrajectoryEncounters`** looks for pairs in the same bucket.\n\nThe dashed lattice on the map is the actual **3 × 3 neighbourhood** searched for one returned pair: only samples in the same bucket and neighbouring 400 m cells are tested. The clock is a parameter buffer. Set the **Clock step** to 60 s and **Clock starts at** to 12:00 UTC: twice the time resolution, covering the afternoon and evening. A shorter step misses fewer passes between buckets, which matters because a vessel at 12 knots travels 370 m in a minute.',
      options: {clockStepSeconds: 60, clockStartHour: 12},
      highlight: {readout: 'clock'},
      controls: ['clockStepSeconds', 'clockStartHour'],
      readouts: ['clock']
    },
    {
      id: 'distance',
      headline: 'A metric ring is only a few pixels',
      textAlternative: 'A true-distance ring surrounds a close vessel pair.',
      optionsMode: 'fresh',
      title: 'How close is a meeting?',
      body: 'Slide **Encounter distance** up to 300 m: connectors multiply as vessels sailing in the same lane count. The distance is clamped to a compile-time lattice cell (400 m) because the contributor only searches the 3 x 3 cells around each sample, so the cell must be at least as wide as the largest distance you ask for.\n\nMeetings are found in a grid with **three axes: x, y and the time bucket**, so only vessels at the same instant can meet. Tracks that are close in space but at different times never pair. The histogram below shows how close the pairs actually get.',
      options: {distance: 300, clockStepSeconds: 60, clockStartHour: 12},
      controls: ['distance'],
      readouts: ['pairs', 'closest', 'approachChart']
    },
    {
      id: 'moored',
      headline: 'Moored neighbours are crowding',
      textAlternative: 'Slow vessels at piers are separated from moving pairs.',
      optionsMode: 'fresh',
      title: 'Moored vessels do not meet, they just sit',
      body: "Set **Ignore vessels slower than** to 0 and the pair list explodes: the tugs and barges tied up side by side at a pier are within meters of each other for hours. A GPU kernel reads the **average speed** that `GPUTrajectoryMetrics` computed for each track and turns the fixes of slower tracks into NaN, which the shared-clock resample and the encounter search both treat as absent; the default 1 knot keeps only vessels that actually travel.\n\nReturn it to 1 knot and watch the busiest hour for meetings in the readouts: it follows the harbor's working day: the chart shows meetings by hour, and **Play** below sweeps the clock so you can watch connectors appear and dissolve as vessels pass.",
      options: {minSpeedKnots: 0, distance: 100, clockStepSeconds: 60, clockStartHour: 12},
      highlight: {readout: 'pairs'},
      controls: ['minSpeedKnots'],
      readouts: ['pairs', 'tracksUsed', 'busiestHour', 'meetingsChart']
    },
    {
      id: 'similarity',
      headline: 'A shared lane is not a crossing',
      textAlternative: 'Route classes distinguish similar and different paths.',
      optionsMode: 'fresh',
      title: 'Traveling together, or just crossing?',
      body: 'Each connector is now colored by how alike the **whole routes** of the two vessels are, measured by `GPUTrackSimilarity` on 64-point arc-length resamples. The **Hausdorff distance** is the largest distance from any point of one route to the other route: the greater of `max over a of min over b |a - b|` and the same with the roles swapped. It follows Shapely `hausdorff_distance` without densification.\n\nPurple means nearly the same route, yellow means very different routes. Set **Connectors for** to *Alike routes only* to keep the pairs that share a lane (ferries in the same slip, tugs of one tow), or to *Different routes only* to see where independent traffic crosses. The histogram splits the pairs by route distance: a hump near zero is convoys and shared lanes, the long tail is crossings.',
      options: {
        minSpeedKnots: 1,
        connectorFilter: 'alike',
        clockStepSeconds: 60,
        clockStartHour: 12
      },
      highlight: {readout: 'alike'},
      controls: ['similarityMetric', 'connectorFilter', 'alikeMeters'],
      readouts: ['alike', 'similarityChart']
    },
    {
      id: 'selected',
      headline: 'A leash respects route direction',
      textAlternative: 'A selected ferry route is compared against a reverse route.',
      optionsMode: 'fresh',
      title: 'Click a vessel: who sails like it?',
      body: "The same measure answers a different question: **which tracks have a route like this vessel's?** Tracks are now colored by the route distance from the selected vessel (the white line; **Color tracks by**). Click any track to change the selection; the second `GPUTrackSimilarity` instance scores that vessel against all 897 routes in one dispatch. **Reselect the Staten Island ferry** brings the selection back to where it started.\n\nSwitch **Route distance** to *Frechet*: it is the shortest leash that lets two walkers follow their routes **in order**, so a vessel that sails the same lane in the opposite direction is still far away, while Hausdorff calls the two routes identical. The dashed route-sample segment is a concrete leash witness from the displayed encounter pair; it makes the route comparison visible without pretending the connector at this instant is the whole-route distance.",
      options: {
        trackColor: 'similarity',
        similarityMetric: 'frechet',
        connectorFilter: 'all',
        clockStepSeconds: 60,
        clockStartHour: 12
      },
      highlight: {readout: 'routesLikeSelected'},
      controls: ['trackColor', 'similarityMetric', 'selectFerry'],
      readouts: ['selected', 'routesLikeSelected', 'selectedChart']
    },
    {
      id: 'limits',
      headline: 'Sampling decides what can be found',
      textAlternative: 'Encounter limits are summarized with the harbor map.',
      optionsMode: 'fresh',
      title: 'Limits and things to try',
      body: 'Meetings are tested **at the buckets only**: two vessels that pass within the distance between two buckets are missed, so keep the clock step small compared with the distance divided by the speed. Distances are planar meters from AIS positions, which are only about 10 m accurate, so very small distances are noise. Frechet is computed on 64-point routes (the Frechet cap is 256 vertices per track), so it reflects the shape of the route, not every wiggle.\n\n**Try it:** set **Encounter distance** to 25 m and look at the closest approach; drag **Show connectors at** through the afternoon; select a tug and see which routes match; set **Connectors for** to *Different routes only* at the Narrows to see where traffic crosses.',
      options: {
        trackColor: 'category',
        similarityMetric: 'hausdorff',
        clockStepSeconds: 60,
        clockStartHour: 12
      },
      controls: ['distance', 'timeOfDay', 'connectorFilter'],
      readouts: ['closest']
    }
  ]
});
