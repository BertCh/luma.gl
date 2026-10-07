// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {formatPlaybackTime, playbackOptions} from '../../engine/playback';
import {defineScene} from '../scene';
import {ZONE_KIND_COLORS, ZONE_KIND_LABELS, ZONE_KINDS} from './b12-zones';
import type {ZoneDwellOptions} from './zone-dwell.compute';

const HARBOR_VIEW = {longitude: -74.07, latitude: 40.63, zoom: 10.2};

export default defineScene<ZoneDwellOptions>({
  id: 'zone-dwell',
  title: 'How long do vessels spend in each harbor zone?',
  chapter: 'movement',
  order: 3,
  summary:
    'Cross 897 AIS tracks with 63 harbor zones (anchorages, channels, terminals, the Narrows gate, the ferry lane): enter and exit events with their exact crossing positions, and dwell time per zone, counted either as any time inside or as time spent stopped.',
  contributors: [
    'GPUZoneEvents',
    'addFleetDwellZoneEventsRecipe',
    'addFleetDwellRecipe',
    'GPUTrajectoryMetrics',
    'GPUPointInPolygonJoin',
    'GPUGroupStatistics'
  ],
  datasets: [
    {id: 'ais-vessels', role: 'vessel tracks (AIS, 12 June 2024)'},
    {id: 'ais-zones', role: '63 harbor zones'}
  ],
  initialView: HARBOR_VIEW,

  options: [
    {
      kind: 'select',
      id: 'variant',
      label: 'Count dwell as',
      group: 'Dwell',
      apply: 'compile',
      default: 'inside',
      help: 'Compile-time: two different graphs. Any time inside a zone (enter and exit events, moving or not) or only time spent stopped inside it (stops joined to the zones).',
      options: [
        {
          value: 'inside',
          label: 'Any time inside the zone',
          help: 'addFleetDwellZoneEventsRecipe: GPUZoneEvents then a dense per-zone statistic. A vessel passing through a channel counts.'
        },
        {
          value: 'stopped',
          label: 'Time spent stopped in the zone',
          help: 'addFleetDwellRecipe: GPUTrajectoryMetrics stops, a point-in-polygon join to the zones, then the same statistic. A vessel passing through does not count.'
        }
      ]
    },
    {
      kind: 'select',
      id: 'metric',
      label: 'Color zones by',
      group: 'Dwell',
      apply: 'param',
      default: 'total',
      help: 'The per-zone statistic that fills the map. The graph computes sum, mean, maximum and count together; this only picks which one a small kernel writes to the fill buffer.',
      options: [
        {value: 'total', label: 'Total dwell (vessel-hours)'},
        {value: 'mean', label: 'Mean dwell per visit'},
        {value: 'longest', label: 'Longest single stay'},
        {value: 'visits', label: 'Number of visits (or stops)'}
      ]
    },
    {
      kind: 'select',
      id: 'zoneKind',
      label: 'Fill which zones',
      group: 'Dwell',
      apply: 'param',
      default: 'all',
      help: 'Restricts the fill (and the "busiest zone" readout) to one kind of zone. Outlines are always drawn.',
      options: [
        {value: 'all', label: 'All zones'},
        ...ZONE_KINDS.map(kind => ({value: kind, label: ZONE_KIND_LABELS[kind]}))
      ]
    },
    {
      kind: 'select',
      id: 'eventsPerVessel',
      label: 'Events kept per vessel',
      group: 'Events',
      apply: 'compile',
      default: '32',
      disabledWhen: state => state.variant !== 'inside',
      help: 'Compile-time bound on the event list kept for drawing. Dwell and visit statistics always use every event; only the drawn pulses are capped (an overflow flag says when a vessel exceeded the cap).',
      options: [
        {value: '8', label: '8 events'},
        {value: '16', label: '16 events'},
        {value: '32', label: '32 events'},
        {value: '64', label: '64 events'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showEvents',
      label: 'Show enter/exit pulses (or stops)',
      group: 'Events',
      apply: 'param',
      default: true,
      help: 'In the inside variant, a ring pulses where a vessel enters (green) or exits (orange) a zone, at the interpolated crossing position, as the playhead passes. In the stopped variant, discs mark the stops.'
    },
    {
      kind: 'toggle',
      id: 'showRestingEvents',
      label: 'Keep past events as faint dots',
      group: 'Events',
      apply: 'param',
      default: false,
      disabledWhen: state => state.variant !== 'inside',
      help: 'Leaves every event on the map as a small dot instead of only pulsing at its moment.'
    },
    ...playbackOptions<ZoneDwellOptions>({
      ids: {play: 'playing', time: 'time', speed: 'playbackSpeed', loop: 'loop'},
      time: {
        min: 0,
        max: 86400,
        step: 60,
        default: 43200,
        label: 'Time of day (UTC)',
        format: formatPlaybackTime.clockUtc,
        help: 'The clock drives the event pulses only; the dwell statistics cover the whole day.'
      },
      speed: {
        min: 60,
        max: 3600,
        step: 60,
        default: 600,
        unit: 'x',
        help: 'Simulated seconds per real second.'
      },
      loop: true
    }),
    {
      kind: 'slider',
      id: 'pulseMinutes',
      label: 'Pulse duration',
      group: 'Playback',
      apply: 'param',
      min: 2,
      max: 120,
      step: 2,
      default: 20,
      unit: 'min',
      help: 'How long an event stays visible after the clock passes it (simulated minutes).'
    },
    {
      kind: 'slider',
      id: 'stopSpeedKnots',
      label: 'Stop speed threshold',
      group: 'Stops',
      apply: 'param',
      min: 0.1,
      max: 3,
      step: 0.1,
      default: 0.5,
      unit: 'kn',
      disabledWhen: state => state.variant !== 'stopped',
      help: 'A step is slow when the vessel moves less than this speed times the step time. Only used by the stopped variant.'
    },
    {
      kind: 'slider',
      id: 'stopMinutes',
      label: 'Minimum stop duration',
      group: 'Stops',
      apply: 'param',
      min: 2,
      max: 240,
      step: 1,
      default: 15,
      unit: 'min',
      disabledWhen: state => state.variant !== 'stopped',
      help: 'A run of slow steps becomes a stop when it lasts at least this long. Both stop settings are parameter writes; the join and statistics re-run without recompiling.'
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Fill ramp',
      group: 'Display',
      apply: 'param',
      default: 'magma',
      help: 'Perceptually uniform ramps. Zones with nothing to show are left unfilled.',
      options: [
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'viridis', label: 'Viridis'},
        {value: 'cividis', label: 'Cividis (color-blind optimised)'}
      ]
    },
    {
      kind: 'slider',
      id: 'fillOpacity',
      label: 'Fill opacity',
      group: 'Display',
      apply: 'param',
      min: 0.2,
      max: 1,
      step: 0.05,
      default: 0.75,
      help: 'Lower it to read the basemap under the zones.'
    },
    {
      kind: 'toggle',
      id: 'showOutlines',
      label: 'Show zone outlines',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Outlines are colored by zone kind.'
    },
    {
      kind: 'toggle',
      id: 'showTracks',
      label: 'Show vessel tracks faintly',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'All 897 tracks as thin lines, so the shipping lanes show through the zones.'
    }
  ],

  readouts: [
    {
      id: 'rankChart',
      label: 'Top zones',
      kind: 'chart',
      help: 'The eight zones with the highest value of the statistic on the map, in hours (or counts for visits). Follows Count dwell as, Color zones by and Fill which zones.'
    },
    {
      id: 'kindChart',
      label: 'Vessel-hours by kind of zone',
      kind: 'chart',
      help: 'Total time inside each kind of zone. Anchorages and channels hold most of the harbor hours; the thin gate and ferry zones hold few but see many crossings.'
    },
    {id: 'clock', label: 'Playhead'},
    {id: 'tracks', label: 'Tracks'},
    {id: 'zones', label: 'Zones'},
    {
      id: 'events',
      label: 'Events (or stops)',
      help: 'Enter and exit events kept in the list, or the number of stops detected.'
    },
    {
      id: 'candidates',
      label: 'Segment-edge candidates',
      help: 'Bounding-box candidates tested against the zone edges, against the scratch capacity.'
    },
    {id: 'overflow', label: 'Overflow'},
    {id: 'zonesUsed', label: 'Zones with activity'},
    {
      id: 'totalDwell',
      label: 'Total dwell in zones',
      help: 'Sum over every zone, so a vessel in two overlapping zones counts in both.'
    },
    {
      id: 'topZone',
      label: 'Highest zone',
      help: 'The zone with the largest value of the colored statistic.'
    },
    {id: 'stay1', label: 'Longest stay 1'},
    {id: 'stay2', label: 'Longest stay 2'},
    {id: 'stay3', label: 'Longest stay 3'},
    {
      id: 'selectedZone',
      label: 'Selected zone',
      help: 'Click a zone for its numbers; hover for a tooltip.'
    }
  ],

  legends: state => [
    {
      kind: 'ramp' as const,
      id: 'zones',
      title:
        state.metric === 'visits'
          ? state.variant === 'inside'
            ? 'Visits per zone'
            : 'Stops per zone'
          : state.metric === 'total'
            ? 'Total dwell per zone'
            : state.metric === 'mean'
              ? 'Mean dwell per visit'
              : 'Longest single stay',
      ramp: state.ramp,
      extent: 'gpu' as const,
      sqrtScale: true,
      unit: state.metric === 'visits' ? 'count' : 'hours',
      format: (value: number) => (value < 10 ? value.toFixed(1) : value.toFixed(0))
    },
    {
      kind: 'categories' as const,
      title: 'Zone outlines',
      entries: ZONE_KINDS.map((kind, index) => ({
        color: ZONE_KIND_COLORS[index],
        label: ZONE_KIND_LABELS[kind]
      }))
    },
    ...(state.showEvents && state.variant === 'inside'
      ? [
          {
            kind: 'categories' as const,
            title: 'Events (pulse where the track crosses the boundary)',
            entries: [
              {color: [77, 230, 140, 255] as const, label: 'Enter a zone'},
              {color: [255, 115, 64, 255] as const, label: 'Exit a zone'}
            ]
          }
        ]
      : []),
    ...(state.showEvents && state.variant === 'stopped'
      ? [
          {
            kind: 'categories' as const,
            title: 'Stops (radius and color grow with the dwell)',
            entries: [
              {color: [255, 199, 224, 255] as const, label: 'A few minutes'},
              {color: [255, 41, 128, 255] as const, label: 'About 1.5 hours'},
              {color: [191, 0, 51, 255] as const, label: '6 hours or more'}
            ]
          }
        ]
      : [])
  ],

  snippet: state =>
    state.variant === 'inside'
      ? `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {addFleetDwellZoneEventsRecipe} from '@luma.gl/experimental/gpu-spatial-analysis';

// zones: boundary edges tagged with a zone index (holes and extra parts share it)
const graph = new GPUCommandGraph(device, {id: 'zone-dwell'});
addFleetDwellZoneEventsRecipe(graph, {
  positions, timestamps, trackOffsets,                 // planar meters, float32 seconds
  edgeStarts, edgeEnds, edgeZones, zoneCount: 63,
  candidateCapacity: ${1 << 20}, maxEventsPerTrack: ${state.eventsPerVessel},
  events: {output: {ids, count, overflow}, eventTypes, eventTimes, eventPositions},
  visitTable: {output: visitIds, zones: visitZones, dwellTimes},   // sparse (track, zone) rows
  table: {counts, sumValues, means, maximums}          // dense: row k is zone k
});
const compiled = graph.compile();                      // once
compiled.encode(commandEncoder, {parameters: undefined});
// event markers read eventPositions/eventTimes; the instance count is a GPU copy of events.count`
      : `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {addFleetDwellRecipe, getGPUTrajectoryMetricsParameterValues} from '@luma.gl/experimental/gpu-spatial-analysis';

const graph = new GPUCommandGraph(device, {id: 'zone-dwell-stopped'});
addFleetDwellRecipe(graph, {
  positions, timestamps, trackOffsets,
  parameters: stopParameters.importToGraph(graph),     // speed threshold + minimum duration
  stopCapacity: 4096,
  zones: {polygonPositions, featureOffsets, polygonOffsets, ringOffsets, candidateCapacity: ${1 << 18}},
  stops: {ids, count, overflow, centroids, durations},
  stopZones,                                           // zone of each stop (0xffffffff outside)
  table: {counts, sumValues, means, maximums}
});
const compiled = graph.compile();                      // once
stopParameters.write(getGPUTrajectoryMetricsParameterValues({
  stopSpeedThreshold: ${((state.stopSpeedKnots * 1852) / 3600).toFixed(2)},   // m/s (${state.stopSpeedKnots} kn)
  stopMinimumDuration: ${state.stopMinutes * 60}                              // s
}));
compiled.encode(commandEncoder, {parameters: undefined});`,

  about: {
    what: '`GPUZoneEvents` finds every moment a track crosses a zone boundary, interpolating the crossing time and position on the segment, and the dwell time and number of visits per (vessel, zone). `addFleetDwellZoneEventsRecipe` rolls that up to one row per zone with `GPUGroupStatistics`. `addFleetDwellRecipe` does the same from stops instead: `GPUTrajectoryMetrics` stops, `GPUPointInPolygonJoin` to the zones, then the statistic.',
    why: 'Port authorities, pilots and planners ask how busy each anchorage, channel and terminal is, how long vessels wait, and where traffic funnels. Zone dwell turns tracks into per-zone workload, and the two variants separate traffic that passes through from traffic that stays.',
    howToRead:
      'Brighter zones have more of the colored statistic. Green and orange rings pulse where vessels enter and leave zones, at the true crossing point on the track. Where zones overlap (a channel crossing an anchorage), the smaller zone is drawn on top.'
  },

  create: async ctx => (await import('./zone-dwell.compute')).createZoneDwell(ctx),

  story: [
    {
      id: 'the-question',
      title: 'Where does New York Harbor spend its time?',
      body: "A harbor is a set of places with jobs: anchorages where ships wait for a berth, channels they sail through, terminals where they load, a ferry lane that never sleeps. The question for every one of them: **how much vessel time does it hold?**\n\nThe map has the **63 zones** of this dataset (36 official NOAA anchorages, 21 maintained channels and 6 hand-drawn approximate areas) over one UTC day of AIS tracks. Color is the **total dwell** (**Color zones by**, below): the sum, over all vessels, of the time spent inside the zone. The brightest zones are where the harbor's hours go. The bar charts rank them and compare the kinds of zone: anchorages are for waiting, channels for passing, and the two carry most of the day.",
      camera: {...HARBOR_VIEW, transitionMs: 1200},
      highlight: {readout: 'topZone'},
      controls: ['metric', 'zoneKind'],
      readouts: ['topZone', 'totalDwell', 'rankChart', 'kindChart']
    },
    {
      id: 'events',
      title: 'Enter, exit, with exact crossing points',
      body: '**`GPUZoneEvents`** cuts every track segment against the zone edges (through a bounding-volume tree) and writes an **enter** or **exit** event with the interpolated crossing *time* and *position*. Press **Play** below: rings pulse where vessels cross the boundary of the **Verrazzano-Narrows gate** (**Fill which zones** is set to the bridge span gate), the strait every ship to the Upper Bay passes.\n\nThe gate is a thin zone, so a crossing is an enter followed within a minute by an exit. Whether a crossing is an enter or an exit comes from parity (the first sample of each track is tested with a ray), not from polygon orientation, so zones with holes work too.',
      camera: {longitude: -74.045, latitude: 40.606, zoom: 12.4, transitionMs: 1500},
      options: {zoneKind: 'gate', showEvents: true, playing: true, playbackSpeed: 900},
      callout: {coordinate: [-74.044, 40.606], text: 'Verrazzano-Narrows gate'},
      controls: ['playing', 'time', 'playbackSpeed', 'zoneKind'],
      readouts: ['events', 'clock']
    },
    {
      id: 'anchorages',
      title: 'Anchorages: how long does a visit last?',
      body: 'Set **Fill which zones** to *Anchorage (official)* and **Color zones by** to *Mean dwell per visit*. A visit is one maximal stay inside the zone: a track that starts inside counts its first interval, and one still inside at its last fix counts up to that fix.\n\nThe long bars are the waiting rooms: ships at anchor in the Lower Bay and Upper Bay for many hours, while the zones near the piers see short repeated visits. `addFleetDwellZoneEventsRecipe` computed the per-visit dwell for every (vessel, zone) pair, then `GPUGroupStatistics` reduced them to one row per zone. The **Longest stay** readouts rank the individual stays.',
      camera: {longitude: -74.04, latitude: 40.62, zoom: 10.9, transitionMs: 1500},
      options: {zoneKind: 'anchorage', metric: 'mean', showEvents: false, playing: false},
      controls: ['zoneKind', 'metric'],
      readouts: ['stay1', 'stay2', 'stay3', 'rankChart']
    },
    {
      id: 'visits',
      title: 'Where does the traffic funnel?',
      body: 'Set **Color zones by** to *Number of visits* and **Fill which zones** back to *All zones*. Dwell says where vessels *stay*; visits say where they *go*. The Staten Island Ferry lane and the Narrows light up because hundreds of tracks cross them, though no vessel stays there long.\n\nClick any zone to read its visits, total, mean and longest stay in the **Selected zone** readout, or hover for a tooltip. The same events feed both views: only the final statistic changes, and that is a buffer write. The ranking chart reorders as you switch.',
      camera: {longitude: -74.05, latitude: 40.665, zoom: 11, transitionMs: 1500},
      options: {zoneKind: 'all', metric: 'visits', showEvents: false, playing: false},
      controls: ['metric', 'zoneKind'],
      readouts: ['selectedZone', 'rankChart']
    },
    {
      id: 'stopped',
      title: 'Passing through is not waiting',
      body: 'A channel can hold a lot of vessel-hours without a single vessel waiting in it. Switch **Count dwell as** to **Time spent stopped in the zone**. This runs **`addFleetDwellRecipe`**: `GPUTrajectoryMetrics` finds the stops, `GPUPointInPolygonJoin` assigns each stop to the zone that contains it, and the same per-zone statistic follows.\n\nThe channels go dark and the anchorages and terminals stay bright. This is a *compile-time* change (a different graph), so the control shows a rebuild badge. The discs are the individual stops, larger and darker for longer waits.',
      options: {
        variant: 'stopped',
        metric: 'total',
        zoneKind: 'all',
        showEvents: true,
        playing: false
      },
      camera: {longitude: -74.05, latitude: 40.645, zoom: 10.7, transitionMs: 1500},
      callout: {coordinate: [-74.039, 40.66], text: 'Upper Bay anchorage 21B'},
      controls: ['variant', 'showEvents'],
      readouts: ['events', 'totalDwell', 'kindChart']
    },
    {
      id: 'thresholds',
      title: 'What counts as a stop?',
      body: "Raise **Minimum stop duration** to 90 minutes: short waits drop out and only the long stays remain. Raise the **Stop speed threshold** to 1.5 knots instead and drifting craft and slow tows begin to count as stopped.\n\nBoth are parameter writes: the stops, the join and the per-zone statistic re-run in the same compiled graph. Compare the totals with the previous steps to see how much of the harbor's dwell is real waiting and how much is steady slow traffic.",
      options: {
        variant: 'stopped',
        stopMinutes: 90,
        stopSpeedKnots: 0.5,
        showEvents: true,
        playing: false
      },
      controls: ['stopMinutes', 'stopSpeedKnots'],
      readouts: ['totalDwell', 'events', 'rankChart']
    },
    {
      id: 'limits',
      title: 'Caveats, and what to try',
      body: 'Only 36 anchorages and 21 channels are official NOAA zones; the six terminal, gate, ferry and tour areas are **hand-drawn approximations** (their names end in "approx."). Zones overlap, so the total dwell readout counts a vessel in each zone it is in, and the map shows the smaller zone on top. A vessel with a reporting gap of more than 20 minutes is split into separate tracks, which makes a long stay look like two visits. Stops need fixes: moored vessels were thinned to one fix every 5 minutes.\n\n**Try it:** compare *Any time inside* with *Time spent stopped* for the same metric; set **Fill which zones** to *Terminal (approximate)*; set **Events kept per vessel** to 8 and watch the **Overflow** readout; drag **Time of day (UTC)** to 19:00 and raise **Playback speed** to 3600x at the Narrows.',
      options: {
        variant: 'inside',
        metric: 'total',
        zoneKind: 'all',
        showEvents: true,
        playing: true
      },
      controls: ['variant', 'zoneKind', 'eventsPerVessel', 'time', 'playbackSpeed'],
      readouts: ['overflow']
    }
  ]
});
