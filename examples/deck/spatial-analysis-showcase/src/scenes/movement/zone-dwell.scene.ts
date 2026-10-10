// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getClassTableLegend} from '../../cartography/class-table';
import {CREDITS, joinCredits} from '../../cartography/credits';
import {labelsFor, NYC} from '../../cartography/gazetteer';
import {ground} from '../../cartography/grounds';
import type {MapGround} from '../../cartography/hue-registry';
import type {ClassTable} from '../../cartography/types';
import {formatPlaybackTime, playbackOptions} from '../../engine/playback';
import {defineScene, type LegendSpec} from '../scene';
import {MOVEMENT_CREDITS} from './movement-style';
import type {ZoneDwellOptions} from './zone-dwell.compute';
import {getEventLegend, getOutlineLegend, getStopLegend} from './zone-dwell-style';

/** The harbour frame of the opening step; later steps move in on their subject. */
const HARBOR_VIEW = {longitude: -74.07, latitude: 40.62, zoom: 10.3};

const NEXT_STORY_LINE = 'Next: *A day of US coastal shipping*.';

/** Source credits of the whole plate. */
const PLATE_CREDIT = joinCredits(
  MOVEMENT_CREDITS.harborAis,
  MOVEMENT_CREDITS.harborZones,
  CREDITS.colorBrewer
);

/** The cartouche of one step (the standing sample line is counted from the data). */
const cartouche = (title: string, subtitle: string, chips: readonly string[] = []) => ({
  title,
  subtitle,
  chips
});

/** Hand-drawn zones are an honesty chip wherever they carry the numbers. */
const HAND_DRAWN_CHIP = ['Some zones are hand-drawn'] as const;

/** Names of the harbour's waters and landmarks, shown from the zoom of the step that needs them. */
const places = (ids: readonly string[], minZoom = 9) =>
  labelsFor(
    NYC,
    ids,
    Object.fromEntries(ids.map(id => [id, {minZoom}])) as Parameters<typeof labelsFor>[2]
  );

/** The legend inputs the compute module stores with `ctx.setLegendData('zoneLegend', ...)`. */
type ZoneLegendInfo = {
  table: ClassTable;
  counts: number[];
  title: string;
  basis?: string;
  ground: MapGround;
  compare: boolean;
};

const SPEED_CHOICES = [
  {value: '300', label: '300x'},
  {value: '900', label: '900x'},
  {value: '1800', label: '1800x'},
  {value: '3600', label: '3600x'}
] as const;

export default defineScene<ZoneDwellOptions>({
  id: 'zone-dwell',
  title: 'How long do vessels spend in each harbor zone?',
  chapter: 'movement',
  order: 3,
  summary:
    'One day of AIS track segments is intersected on the GPU with anchorage, channel, terminal and hand-drawn zone boundaries to measure visits, vessel-hours, rates or stopped time; small per-zone tables are read to the CPU for units, classes and charts. Overlapping zones count the same vessel separately, and some boundaries are approximate.',
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
    {id: 'ais-zones', role: 'anchorages, channels and hand-drawn harbor zones'}
  ],
  initialView: HARBOR_VIEW,

  options: [
    {
      kind: 'select',
      id: 'zoneKind',
      label: 'Zones shown',
      group: 'Zones',
      display: 'chips',
      apply: 'param',
      default: 'all',
      help: 'Restricts the fill, the ranking, the readouts and the pulses to one kind of zone. Outlines are always drawn.',
      options: [
        {value: 'all', label: 'All zones'},
        {value: 'anchorage', label: 'Anchorages'},
        {value: 'channel', label: 'Channels'},
        {value: 'approximate', label: 'Hand-drawn'}
      ]
    },
    {
      kind: 'select',
      id: 'metric',
      label: 'Colour zones by',
      group: 'Zones',
      apply: 'param',
      default: 'total',
      help: 'The per-zone statistic that fills the map. The graph computes count, sum and maximum together; this only picks which one is drawn, on its own frozen class breaks.',
      options: [
        {value: 'total', label: 'Total time'},
        {value: 'mean', label: 'Mean stay per visit'},
        {value: 'longest', label: 'Longest single stay'},
        {value: 'visits', label: 'Number of visits (or stops)'}
      ]
    },
    {
      kind: 'select',
      id: 'unit',
      label: 'Unit',
      group: 'Zones',
      display: 'segmented',
      apply: 'param',
      default: 'total',
      disabledWhen: state => state.metric !== 'total',
      help: 'Total vessel-hours, vessel-hours per square kilometre, or vessels present on average (vessel-hours over the hours of the day). Present and total differ by a constant, so they share the same breaks; per km² gets its own.',
      options: [
        {value: 'total', label: 'Vessel-hours'},
        {value: 'density', label: 'Per km²'},
        {value: 'present', label: 'Vessels present'}
      ]
    },
    {
      kind: 'select',
      id: 'variant',
      label: 'Count time as',
      group: 'Definition',
      display: 'segmented',
      apply: 'param',
      default: 'inside',
      help: 'Any time inside a zone (enter and exit events, moving or not) or only time spent stopped inside it (stops joined to the zones). Both graphs are compiled once and stay live, so switching is a buffer swap.',
      options: [
        {value: 'inside', label: 'Inside'},
        {value: 'stopped', label: 'Stopped'}
      ]
    },
    {
      kind: 'slider',
      id: 'stopMinutes',
      label: 'Minimum stop duration',
      group: 'Definition',
      apply: 'param',
      min: 2,
      max: 240,
      step: 1,
      default: 15,
      unit: 'min',
      help: 'A run of slow steps becomes a stop when it lasts at least this long. A parameter write: the stops, the join and the statistics re-run without recompiling.'
    },
    {
      kind: 'slider',
      id: 'stopSpeedKnots',
      label: 'Stop speed threshold',
      group: 'Definition',
      apply: 'param',
      min: 0.1,
      max: 3,
      step: 0.1,
      default: 0.5,
      unit: 'kn',
      help: 'A step is slow when the vessel moves less than this speed times the step time. A parameter write.'
    },
    {
      kind: 'toggle',
      id: 'showTracks',
      label: 'Show vessel tracks',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Every track of the day as one thin neutral line, so the shipping lanes show through the zones.'
    },
    {
      kind: 'toggle',
      id: 'showEvents',
      label: 'Show enter and exit pulses',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'A ring pulses where a vessel enters (sky) or exits (orange) a zone in view, at the interpolated crossing position, as the clock passes it.'
    },
    {
      kind: 'toggle',
      id: 'showStops',
      label: 'Show the stops',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'One disc per stop inside the zones in view; size and colour grow with the time stopped.'
    },
    {
      kind: 'toggle',
      id: 'compareVariants',
      label: 'Swipe inside against stopped',
      group: 'Display',
      apply: 'param',
      default: false,
      expert: true,
      help: 'Draws both definitions at once on one frozen class table, with a swipe divider (inside on the left, stopped on the right).'
    },
    {
      kind: 'toggle',
      id: 'showVessel',
      label: 'Follow one vessel',
      group: 'Display',
      apply: 'param',
      default: false,
      expert: true,
      help: 'Outlines the track of one vessel that makes a round trip through the Narrows gate, numbers its crossings and draws its visits as a Gantt chart.'
    },
    {
      kind: 'select',
      id: 'notes',
      label: 'Map notes',
      group: 'Display',
      apply: 'param',
      default: 'none',
      expert: true,
      help: 'Finding notes read from the zone statistics: the largest and the highest zone, or the hand-drawn zones.',
      options: [
        {value: 'none', label: 'None'},
        {value: 'size', label: 'Largest and highest zone'},
        {value: 'approximate', label: 'Hand-drawn zones'}
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
      step: 0.02,
      default: 0.88,
      expert: true,
      help: 'Lower it to read the basemap under the zones.'
    },
    {
      kind: 'select',
      id: 'eventsPerVessel',
      label: 'Events kept per vessel',
      group: 'Engine',
      apply: 'compile',
      default: '64',
      expert: true,
      help: 'Compile-time bound on the event list kept for drawing. Dwell and visit statistics always use every event; only the drawn pulses are capped (an overflow flag says when a vessel exceeded the cap).',
      options: [
        {value: '16', label: '16 events'},
        {value: '32', label: '32 events'},
        {value: '64', label: '64 events'}
      ]
    },
    ...playbackOptions<ZoneDwellOptions>({
      ids: {play: 'playing', time: 'time', speed: 'playbackSpeed', loop: 'loop'},
      playing: false,
      time: {
        min: 0,
        max: 86400,
        step: 60,
        default: 43200,
        label: 'Time of day (UTC)',
        format: formatPlaybackTime.clockUtc,
        help: 'The clock drives the enter and exit pulses only; the dwell statistics cover the whole day.'
      },
      speed: {
        kind: 'select',
        options: SPEED_CHOICES,
        default: '900',
        label: 'Playback speed',
        help: 'Simulated seconds per real second.'
      },
      loop: true
    })
  ],

  readouts: [
    {
      id: 'topZone',
      label: 'Highest zone',
      help: 'The zone with the largest value of the mapped statistic, in the unit on the map.'
    },
    {
      id: 'zonesUsed',
      label: 'Zones with vessel time',
      help: 'Named zones that hold any vessel time in this definition (pieces of one anchorage are added up).'
    },
    {
      id: 'busiestArea',
      label: 'Median area of the ten highest',
      help: 'Median area of the ten highest zones in the unit on the map.'
    },
    {id: 'typicalArea', label: 'Median area of all zones with time'},
    {
      id: 'rankChart',
      label: 'The ten highest zones',
      kind: 'chart',
      help: 'The ten zones with the highest value, in the unit on the map and coloured by class. Click a bar to outline the zone.'
    },
    {
      id: 'events',
      label: 'Crossings of zone edges',
      help: 'Enter and exit events of the zones in view, read back from the GPU list.'
    },
    {
      id: 'followed',
      label: 'The vessel followed',
      help: 'One vessel that makes a round trip through the Narrows gate zone.'
    },
    {
      id: 'visitChart',
      label: 'Visits of the vessel followed',
      kind: 'chart',
      help: 'Each bar is one visit: from the enter event to the exit event of a zone. Times are UTC.'
    },
    {
      id: 'passageShare',
      label: 'Visits that end within the minimum stop',
      help: 'Share of the visits to the zones in view that last less than the minimum stop duration.'
    },
    {id: 'meanVisit', label: 'Mean visit', help: 'Mean length of a visit to the zones in view.'},
    {
      id: 'longestVisit',
      label: 'Longest stay',
      help: 'The longest single visit (or stop) in the zones in view.'
    },
    {
      id: 'kindChart',
      label: 'Vessel-hours by kind of zone',
      kind: 'chart',
      help: 'Total vessel-hours in each kind of zone, counted as any time inside and as time spent stopped.'
    },
    {id: 'insideHours', label: 'Hours inside, all zones'},
    {id: 'stoppedHours', label: 'Hours stopped, all zones'},
    {
      id: 'channelKeep',
      label: 'Channel hours that are stopped time',
      help: 'Stopped vessel-hours over inside vessel-hours, summed over the maintained channels.'
    },
    {
      id: 'anchorageKeep',
      label: 'Anchorage hours that are stopped time',
      help: 'Stopped vessel-hours over inside vessel-hours, summed over the official anchorages.'
    },
    {
      id: 'stopCount',
      label: 'Stops found',
      help: 'Stops found by GPUTrajectoryMetrics, split by whether the join puts them in a zone.'
    },
    {
      id: 'approxShare',
      label: 'Inside hours in hand-drawn zones',
      help: 'Share of all inside vessel-hours that fall in the hand-drawn zones.'
    },
    {id: 'approxZones', label: 'Hand-drawn zones'},
    {id: 'tracks', label: 'Tracks', hood: true},
    {id: 'zones', label: 'Zones', hood: true},
    {id: 'clock', label: 'Playhead', hood: true},
    {
      id: 'eventsKept',
      label: 'Events kept',
      hood: true,
      help: 'Enter and exit events kept in the list, against the per-vessel cap.'
    },
    {
      id: 'candidates',
      label: 'Segment-edge candidates',
      hood: true,
      help: 'Bounding-box candidates tested against the zone edges, against the scratch capacity.'
    },
    {id: 'overflow', label: 'Overflow', hood: true}
  ],

  pipeline: [
    {
      id: 'events',
      label: 'Zone events',
      detail: 'Segment edges are cut against zone edges; parity gives enter and exit'
    },
    {id: 'stops', label: 'Stops', detail: 'Runs of slow steps that last long enough'},
    {id: 'join', label: 'Join to zones', detail: 'Each stop centroid lands in a zone'},
    {id: 'stats', label: 'Zone statistics', detail: 'Count, sum and maximum of the dwell per zone'}
  ],

  timeline: {
    time: 'time',
    play: 'playing',
    speed: 'playbackSpeed',
    format: formatPlaybackTime.clockUtc
  },

  legends: (state, data) => {
    const info = data['zoneLegend'] as ZoneLegendInfo | null | undefined;
    const groundTone: MapGround = info?.ground ?? 'light';
    const legends: LegendSpec[] = [];
    if (info) {
      legends.push(
        getClassTableLegend(info.table, {
          title: info.title,
          id: 'zone-classes',
          basis: info.basis,
          counts: info.counts,
          interactive: true,
          layout: 'list',
          note: info.compare
            ? `${info.table.method}. Left of the divider counts any time inside, right of it time stopped, on this one table.`
            : info.table.method
        })
      );
    }
    legends.push(getOutlineLegend(groundTone));
    if (state.showEvents) legends.push(getEventLegend(groundTone));
    if (state.showStops) legends.push(getStopLegend(groundTone));
    return legends;
  },

  basemap: ground('paperCity', {
    suppressNames: [
      'Upper New York Bay',
      'Lower New York Bay',
      'Raritan Bay',
      'Newark Bay',
      'Kill Van Kull'
    ]
  }),
  furniture: {
    title: cartouche('Where does the harbour spend its time?', 'Vessel-hours in each zone · total'),
    scaleBar: {units: 'nautical'},
    credit: PLATE_CREDIT,
    caveat:
      'Overlapping zones count the same vessel in each polygon; dashed zones are hand-drawn, long AIS gaps split tracks and moored fixes are thinned.',
    clock: false
  },

  snippet: state => {
    const stopped = `import {addFleetDwellRecipe, getGPUTrajectoryMetricsParameterValues} from '@luma.gl/experimental/gpu-spatial-analysis';

const stopsGraph = new GPUCommandGraph(device, {id: 'zone-stops'});
addFleetDwellRecipe(stopsGraph, {
  positions, timestamps, trackOffsets,
  parameters: stopParameters.importToGraph(stopsGraph),     // speed threshold + minimum duration
  stopCapacity: 4096,
  zones: {polygonPositions, featureOffsets, polygonOffsets, ringOffsets, candidateCapacity: ${1 << 18}},
  stops: {ids, count, overflow, centroids, durations},
  stopZones,                                                // zone of each stop (0xffffffff outside)
  table: {counts, sumValues, means, maximums}
});
const stopsCompiled = stopsGraph.compile();                 // once
stopParameters.write(getGPUTrajectoryMetricsParameterValues({
  stopSpeedThreshold: ${((state.stopSpeedKnots * 1852) / 3600).toFixed(2)},        // m/s (${state.stopSpeedKnots} kn)
  stopMinimumDuration: ${state.stopMinutes * 60}                     // s
}));`;
    const events = `import {addFleetDwellZoneEventsRecipe} from '@luma.gl/experimental/gpu-spatial-analysis';

// zones: boundary edges tagged with a zone index (holes and extra parts share it)
const eventsGraph = new GPUCommandGraph(device, {id: 'zone-events'});
addFleetDwellZoneEventsRecipe(eventsGraph, {
  positions, timestamps, trackOffsets,                 // planar meters, float32 seconds
  edgeStarts, edgeEnds, edgeZones, zoneCount,
  candidateCapacity: ${1 << 20}, maxEventsPerTrack: ${state.eventsPerVessel},
  events: {output: {ids, count, overflow}, eventZones, eventTypes, eventTimes, eventPositions},
  visitTable: {output: visitIds, zones: visitZones, dwellTimes},   // one row per (track, zone) visit
  table: {counts, sumValues, means, maximums}          // dense: row k is zone k
});
const eventsCompiled = eventsGraph.compile();          // once`;
    const after = `
// Both graphs encode once per change; the small per-zone tables are read back, the pieces of one
// anchorage are added up, and the unit is applied on the CPU:
//   vessel-hours = sum / 3600;  per km² = vessel-hours / area;  present = vessel-hours / 24
// Class breaks are computed once per unit (${state.unit === 'density' ? 'per km²' : 'vessel-hours'}) and kept.`;
    return `${events}\n\n${stopped}\n${after}`;
  },

  about: {
    what: `Previously: *Which vessels meet, and do they share a route?* ${NEXT_STORY_LINE}

\`GPUZoneEvents\` finds every moment a track crosses a zone edge, interpolating the crossing time and position on the segment, gives each crossing its enter or exit by parity, and writes one visit per (vessel, zone) with its dwell. \`addFleetDwellZoneEventsRecipe\` rolls the visits up to one row per zone with \`GPUGroupStatistics\`. \`addFleetDwellRecipe\` does the same from stops instead: \`GPUTrajectoryMetrics\` stops, \`GPUPointInPolygonJoin\` to the zones, then the statistic. Both graphs are compiled once and stay live.`,
    why: 'Port authorities, pilots and planners ask how busy each anchorage, channel and terminal is, where vessels wait and where traffic funnels. The answer depends on how time is counted (any time inside, or time stopped) and on how big each zone is, so the story shows totals, rates and both definitions on one map.',
    howToRead:
      'Darker zones hold more of the mapped statistic, in classes that the map, the legend and the ranking share. Outline style tells the kind of zone: solid for an official anchorage, lighter for a maintained channel, dashed for a hand-drawn approximation. Where zones overlap a vessel counts in each, and the smaller zone is drawn on top. Pieces of one anchorage are added up and share a colour.'
  },

  create: async ctx => (await import('./zone-dwell.compute')).createZoneDwell(ctx),

  story: [
    {
      id: 'the-question',
      title: 'Where does the harbour spend its time?',
      headline: 'Total vessel-hours rank zones by accumulated time',
      textAlternative:
        'New York Harbor zones classed by total vessel-hours from one day of AIS tracks; solid, light and dashed outlines distinguish official anchorages, channels and hand-drawn zones.',
      body: 'The GPU intersects AIS track segments with zone boundaries, pairs entry and exit events, and sums visit duration by zone. Small per-zone tables are read to the CPU for unit conversion, classification, ranking and charts. The displayed metric is total vessel-hours; use **Zones shown** below to filter zone types.',
      evidence:
        '**{{zonesUsed}}** zones contain recorded vessel time, and **{{topZone}}** has the largest total under the current filter.',
      caveat:
        'Totals increase with polygon area and AIS observations; overlapping zones count the same vessel in each zone.',
      optionsMode: 'fresh',
      options: {showTracks: true},
      controls: ['zoneKind'],
      readouts: ['topZone', 'zonesUsed', 'rankChart'],
      camera: {...HARBOR_VIEW, transitionMs: 1200},
      furniture: {
        title: cartouche(
          'Where does the harbour spend its time?',
          'Vessel-hours in each zone · total · quantile classes',
          HAND_DRAWN_CHIP
        ),
        caveat:
          'Overlapping polygons count the same vessel in each zone; some zones are hand-drawn.',
        clock: false
      },
      annotations: places(
        [
          'lower-bay',
          'upper-bay',
          'raritan-bay',
          'newark-bay',
          'ambrose-channel',
          'the-narrows',
          'kill-van-kull',
          'port-newark'
        ],
        9
      ),
      stage: 'stats'
    },
    {
      id: 'big-zones-win',
      title: 'Big zones win by being big',
      headline: 'Area normalization changes the zone ranking',
      textAlternative:
        'New York Harbor zones classed by total vessel-hours, vessel-hours per square kilometer or mean vessels present; labels and a ranking chart identify changes between units.',
      body: 'The ten highest zones in the current unit have a median area of {{busiestArea}}, against {{typicalArea}} for every zone with time. Flip **Unit** below and the ranking reshuffles: {{topZone}} leads.\n\nVessels present is the total divided by the hours of the day, so it shares the total’s breaks; per km² gets its own.\n\n*Normalise before you compare areas.*',
      optionsMode: 'fresh',
      options: {unit: 'total', notes: 'size'},
      controls: ['unit'],
      readouts: ['topZone', 'busiestArea', 'typicalArea', 'rankChart'],
      camera: {longitude: -74.07, latitude: 40.58, zoom: 10.8, transitionMs: 1400},
      furniture: {
        title: cartouche(
          'Normalise before you compare areas',
          'Total, per km² or vessels present · quantile classes · classes fixed per unit',
          HAND_DRAWN_CHIP
        ),
        caveat:
          'Overlapping polygons count the same vessel in each zone; rate values depend on polygon area.',
        clock: false
      },
      annotations: places(
        ['lower-bay', 'upper-bay', 'raritan-bay', 'newark-bay', 'port-newark'],
        9
      ),
      stage: 'stats'
    },
    {
      id: 'crossing-the-line',
      title: 'Enter and exit decide every visit',
      headline: 'Boundary crossings delimit each zone visit',
      textAlternative:
        'Close map of the Verrazzano-Narrows gate with blue entry rings, orange exit rings, numbered crossings on one AIS track and a chart of the resulting visit intervals.',
      body: 'Press **Play** below. A ring marks each crossing of a zone edge ({{events}}): sky blue enters, orange exits. `GPUZoneEvents` finds the exact point on the segment and decides enter or exit by parity, so the numbered crossings of {{followed}} alternate enter, exit.\n\nA visit is an enter and its exit, as the Gantt chart shows. Drag **Time of day (UTC)** to replay it.\n\n*Parity, not polygon orientation, decides.*',
      optionsMode: 'fresh',
      options: {
        zoneKind: 'approximate',
        showEvents: true,
        showVessel: true,
        showTracks: true,
        fillOpacity: 0.4,
        playing: true,
        playbackSpeed: '900'
      },
      controls: ['playing', 'time'],
      readouts: ['events', 'followed', 'visitChart'],
      camera: {longitude: -74.045, latitude: 40.606, zoom: 12.6, transitionMs: 1500},
      furniture: {
        title: cartouche(
          'Enter and exit decide every visit',
          'Zone crossings · parity of segment-edge hits · 12 June 2024 UTC',
          HAND_DRAWN_CHIP
        ),
        clock: {
          option: 'time',
          time: {origin: '2024-06-12T00:00:00Z', unit: 'seconds'},
          zones: ['America/New_York', 'UTC']
        }
      },
      annotations: places(['verrazzano', 'the-narrows'], 9),
      stage: 'events'
    },
    {
      id: 'waiting-rooms',
      title: 'Waiting rooms',
      headline: 'Most anchorage visits remain below stop duration',
      textAlternative:
        'Upper and Lower Bay anchorages classed by longest recorded visit, with readouts for the share of short passages, mean visit and longest visit.',
      body: 'Anchorages exist for waiting, yet {{passageShare}} of their visits end before a vessel could count as stopped; the average visit lasts {{meanVisit}}. Long stays are the exception: the longest is {{longestVisit}}.\n\nChoose **Colour zones by** to compare the longest stay with the mean visit and the number of visits.\n\n*Time inside is not time waiting.*',
      optionsMode: 'fresh',
      options: {zoneKind: 'anchorage', metric: 'longest'},
      controls: ['metric'],
      readouts: ['passageShare', 'meanVisit', 'longestVisit', 'rankChart'],
      camera: {longitude: -74.04, latitude: 40.6, zoom: 11.2, transitionMs: 1500},
      furniture: {
        title: cartouche(
          'Most anchorage visits are passages',
          'Longest single stay per anchorage · hours · manual classes · 12 June 2024 UTC'
        ),
        caveat:
          'Disconnected polygons with the same anchorage identifier are combined into one value.',
        clock: false
      },
      annotations: places(
        ['lower-bay', 'upper-bay', 'raritan-bay', 'the-narrows', 'newark-bay'],
        9
      ),
      stage: 'stats'
    },
    {
      id: 'passing-is-not-waiting',
      title: 'Passing through is not waiting',
      headline: 'Stopped time is lower than total inside time',
      textAlternative:
        'Swipe comparison of total vessel-hours inside each harbor zone and vessel-hours meeting the stop rule, using one shared class table.',
      body: 'Drag the divider: the left counts any time inside a zone, the right only time spent stopped, on one shared set of classes. Channels keep {{channelKeep}} of their hours as stopped time, anchorages {{anchorageKeep}}.\n\nThe stopped side runs `GPUTrajectoryMetrics` stops, then a point-in-polygon join. Change **Minimum stop duration** below to redraw it.\n\n*A definition is a choice, not a fact.*',
      optionsMode: 'fresh',
      options: {compareVariants: true},
      controls: ['stopMinutes'],
      readouts: ['insideHours', 'stoppedHours', 'kindChart'],
      compare: {mode: 'swipe', labels: ['Inside', 'Stopped'], position: 0.5},
      camera: {longitude: -74.06, latitude: 40.62, zoom: 10.7, transitionMs: 1400},
      furniture: {
        title: cartouche(
          'Passing through is not waiting',
          'Vessel-hours inside against stopped · one shared table · 12 June 2024 UTC',
          HAND_DRAWN_CHIP
        ),
        caveat:
          'Both sides use breaks derived from total inside time, so stopped values occupy lower classes.',
        clock: false
      },
      annotations: places(
        [
          'kill-van-kull',
          'ambrose-channel',
          'upper-bay',
          'lower-bay',
          'port-newark',
          'red-hook-terminal'
        ],
        9
      ),
      stage: 'join'
    },
    {
      id: 'what-counts',
      title: 'What counts as time in a zone?',
      headline: 'Stop thresholds change accumulated zone time',
      textAlternative:
        'Harbor zones classed by stopped vessel-hours, with a point for each qualifying stop and dashed outlines for hand-drawn zones; controls set speed and duration thresholds.',
      body:
        'A stop is a rule: slower than **Stop speed threshold** for at least **Minimum stop duration**. Change either, or **Unit**, and the map redraws on the same classes ({{stopCount}}).\n\nZone edges are a definition too: {{approxShare}} of the hours inside fall in the {{approxZones}}, dashed. Overlaps count a vessel in each zone, a long gap splits a track, moored fixes were thinned.\n\n' +
        NEXT_STORY_LINE,
      optionsMode: 'fresh',
      options: {variant: 'stopped', showStops: true, notes: 'approximate'},
      controls: ['stopMinutes', 'stopSpeedKnots', 'unit'],
      readouts: ['stoppedHours', 'stopCount', 'approxShare', 'approxZones'],
      camera: {longitude: -74.07, latitude: 40.62, zoom: 10.6, transitionMs: 1400},
      furniture: {
        title: cartouche(
          'The definition moves the hours',
          'Stopped vessel-hours · minimum stop and speed adjustable · 12 June 2024 UTC',
          HAND_DRAWN_CHIP
        ),
        caveat: 'Dashed zones are hand-drawn; solid and light outlines use NOAA boundaries.',
        clock: false
      },
      annotations: places(
        [
          'port-newark',
          'red-hook-terminal',
          'statue-of-liberty',
          'st-george-terminal',
          'verrazzano',
          'lower-bay'
        ],
        9
      ),
      stage: 'stops'
    }
  ]
});
