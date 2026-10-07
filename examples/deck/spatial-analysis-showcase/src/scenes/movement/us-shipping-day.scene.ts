// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getClassTableLegend} from '../../cartography/class-table';
import {labelsFor, US, WORLD} from '../../cartography/gazetteer';
import {joinCredits} from '../../cartography/credits';
import {getScaleBarLatitude, mercatorCaveat} from '../../cartography/projection-notes';
import {ground} from '../../cartography/grounds';
import {formatPlaybackTime, playbackOptions} from '../../engine/playback';
import {defineScene, type LegendSpec, type StoryStep} from '../scene';
import {
  getShipSpeedLegend,
  getVesselGroupPalette,
  inkFor,
  MOVEMENT_CREDITS,
  SUBJECT_INK,
  VESSEL_GROUP_LABELS,
  VESSEL_GROUPS,
  ZONE_EVENT_INK
} from './movement-style';
import type {UsShippingOptions} from './us-shipping-day.compute';
import {STRETCH_LABELS, STRETCHES} from './us-shipping-density';
import {US_SHIPPING_PLACES} from './us-shipping-places';
import {
  SHIPPING_FRAMES,
  SHIPPING_GATES,
  SHIPPING_TYPE_LABELS,
  SHIPPING_TYPES
} from './us-shipping.tracks';

/** The night ground in both page themes, so the colours are the dark ground's. */
const GROUND = 'dark' as const;

/** Window of the archive (`poopdeck-ais-us` manifest): the middle latitude sets the scale bar. */
const ARCHIVE_WINDOW = [-130, 24, -65, 49.5] as const;

const CLOCK = {
  option: 'time',
  time: {origin: '2023-01-09T00:00:00Z', unit: 'seconds'},
  zones: ['UTC', 'America/New_York', 'America/Los_Angeles']
} as const;

/** The cartouche of one step: the claim, then variable, unit, method and date. */
const cartouche = (title: string, subtitle: string) => ({title, subtitle});

/** Regional steps measure the scale bar at the map centre and drop the national caveat. */
const REGIONAL_FURNITURE = {scaleBar: {units: 'nautical'}, caveat: ''} as const;

const typeOptions = SHIPPING_TYPES.map(type => ({value: type, label: SHIPPING_TYPE_LABELS[type]}));

/** Marker of the gate midpoints, so a step can label the gate it talks about. */
const getGateLabel = (id: string) => {
  const gate = SHIPPING_GATES.find(candidate => candidate.id === id);
  if (!gate) throw new Error(`Unknown gate ${id}`);
  return {
    kind: 'point' as const,
    id: `gate:${gate.id}`,
    coordinate: [(gate.from[0] + gate.to[0]) / 2, (gate.from[1] + gate.to[1]) / 2] as [
      number,
      number
    ],
    text: gate.name,
    rank: 'subject' as const,
    priority: 3,
    marker: 'none' as const,
    anchor: 'ne' as const
  };
};

const sceneFurniture = {
  title: cartouche(
    "Where do America's ships go in a day?",
    'Vessel positions · interpolated between AIS fixes · 9 January 2023 (UTC)'
  ),
  scaleBar: {
    units: 'nautical' as const,
    latitude: getScaleBarLatitude([
      ARCHIVE_WINDOW[0],
      ARCHIVE_WINDOW[1],
      ARCHIVE_WINDOW[2],
      ARCHIVE_WINDOW[3]
    ]),
    minZoom: 3.5
  },
  credit: joinCredits(MOVEMENT_CREDITS.usAis),
  caveat: mercatorCaveat(),
  clock: CLOCK
};

/** Legend of the vessel marks for the current colour mode. */
function getVesselLegend(
  state: UsShippingOptions,
  data: Readonly<Record<string, unknown>>
): LegendSpec {
  if (state.markerColor === 'speed') return getShipSpeedLegend(GROUND);
  if (state.markerColor === 'group') {
    const kilometers = data.groupKilometers as number[] | undefined;
    const total = kilometers?.reduce((sum, value) => sum + value, 0) ?? 0;
    const colors = getVesselGroupPalette(GROUND);
    return {
      kind: 'categories',
      title: 'Vessel group',
      layout: 'list',
      entries: VESSEL_GROUPS.map((group, index) => ({
        color: colors[index],
        label: VESSEL_GROUP_LABELS[group],
        shape: 'dot' as const,
        ...(kilometers && total > 0
          ? {detail: `${Math.round((kilometers[index] / total) * 100)}% of track in view`}
          : {})
      })),
      note: 'AIS ship-type codes grouped by what the vessel does.'
    };
  }
  const ink = inkFor(SUBJECT_INK, GROUND);
  return {
    kind: 'categories',
    title: 'Vessels',
    entries: [
      {color: ink, label: 'Moving: arrow along the heading', shape: 'dot'},
      {color: ink, label: 'Stopped: square', shape: 'swatch'}
    ],
    note: 'One colour for every vessel, so where ships are reads before what they are.'
  };
}

/** Legend of the gates and the ring colours of their crossings. */
function getGateLegend(state: UsShippingOptions): LegendSpec {
  const gate = SHIPPING_GATES.find(candidate => candidate.id === state.gateFocus);
  return {
    kind: 'categories',
    title: 'Gates',
    entries: [
      {color: [232, 237, 242, 255], label: 'Gate line (approximate)', shape: 'line'},
      {
        color: inkFor(ZONE_EVENT_INK.enter, GROUND),
        label: gate ? gate.directions[0] : 'Crossing in the first direction',
        shape: 'ring'
      },
      {
        color: inkFor(ZONE_EVENT_INK.exit, GROUND),
        label: gate ? gate.directions[1] : 'Crossing in the second direction',
        shape: 'ring'
      }
    ],
    note: 'Each ring marks a crossing and fades with the playhead.'
  };
}

export default defineScene<UsShippingOptions>({
  id: 'us-shipping-day',
  title: 'A day of US coastal shipping',
  chapter: 'movement',
  order: 4,
  summary:
    'One Monday of AIS across the lower 48: vessels played on the GPU, the whole day added up into km of track per km² with line density, the classes that decide whether lanes show at all, and gates that count ships.',
  contributors: ['GPUTrajectoryPlayhead', 'GPUTimeWindowFilter', 'GPULineDensity', 'GPUZoneEvents'],
  datasets: [{id: 'poopdeck-ais-us', role: 'vessel tracks (AIS, 9 January 2023)'}],
  initialView: SHIPPING_FRAMES.country,

  options: [
    ...playbackOptions<UsShippingOptions>({
      time: {
        min: 0,
        max: 86340,
        step: 60,
        default: 36000,
        label: 'Time of day (UTC)',
        format: formatPlaybackTime.clockUtc,
        help: 'The AIS day is 9 January 2023 in UTC. Pacific time is 8 hours earlier, Eastern time 5 hours earlier, so 16:00 UTC is 8 am on the West Coast and 11 am on the East Coast.'
      },
      speed: {
        min: 60,
        max: 3600,
        step: 60,
        default: 900,
        unit: 'x',
        help: 'Simulated seconds per real second. 900x plays the whole day in 96 seconds.'
      },
      loop: true
    }),
    {
      kind: 'slider',
      id: 'maxGapMinutes',
      label: 'Hide vessels in a gap longer than',
      group: 'Playback',
      apply: 'param',
      min: 0,
      max: 120,
      step: 5,
      default: 0,
      unit: 'min',
      format: value => (value === 0 ? 'off' : `${value} min`),
      help: 'A vessel whose two fixes around the playhead are further apart than this is hidden instead of drawn at a guessed place. Parked vessels were thinned in this archive, so they report least often and vanish first. 0 turns the test off.'
    },
    {
      kind: 'select',
      id: 'markerColor',
      label: 'Vessel colour',
      group: 'Layers',
      apply: 'param',
      display: 'segmented',
      default: 'uniform',
      help: 'One colour for every vessel (the default: where ships are comes first), the four vessel groups from the AIS ship-type code, or speed over ground in five classes. Trails follow the choice, except in speed mode.',
      options: [
        {value: 'uniform', label: 'One colour'},
        {value: 'group', label: 'Vessel group'},
        {value: 'speed', label: 'Speed'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showVessels',
      label: 'Vessels',
      group: 'Layers',
      apply: 'param',
      default: true,
      help: 'One mark per vessel at the playhead, interpolated between its two surrounding fixes by GPUTrajectoryPlayhead. Arrows are moving, squares are stopped.'
    },
    {
      kind: 'toggle',
      id: 'showTrails',
      label: 'Trails',
      group: 'Layers',
      apply: 'param',
      default: true,
      help: 'The part of every track inside the sliding window behind the playhead. Which segments are live is decided on the GPU by GPUTimeWindowFilter.'
    },
    {
      kind: 'slider',
      id: 'trailMinutes',
      label: 'Trail length',
      group: 'Layers',
      apply: 'param',
      min: 10,
      max: 240,
      step: 5,
      default: 60,
      unit: 'min',
      disabledWhen: state => !state.showTrails,
      help: 'Window width, written into the window parameter buffer as [playhead - length, playhead]. The trail fades toward its oldest end.'
    },
    {
      kind: 'toggle',
      id: 'showBackdrop',
      label: 'Every track of the day, faintly',
      group: 'Layers',
      apply: 'param',
      default: true,
      help: 'All tracks as hairlines that add their light where they overlap, so busy lanes and the shoreline emerge from the data.'
    },
    {
      kind: 'toggle',
      id: 'showDensity',
      label: 'Track density',
      group: 'Layers',
      apply: 'param',
      default: false,
      help: 'Paints the length of track per grid cell, computed by GPULineDensity in spherical mode: every segment is clipped to the grid and its great-circle length is added to the cells it crosses.'
    },
    {
      kind: 'toggle',
      id: 'showGates',
      label: 'Gates',
      group: 'Layers',
      apply: 'param',
      default: false,
      help: 'Draws the 13 gate lines and a ring at every crossing. Each gate is a thin rectangle in the analysis, so a crossing is an enter event with an interpolated time and position.'
    },
    {
      kind: 'toggle',
      id: 'showReceiverEdge',
      label: 'Mark where receivers stop',
      group: 'Layers',
      apply: 'param',
      default: false,
      help: 'Draws the line west of which 99 percent of the fixes of the Mid-Atlantic latitude band lie, and names the open ocean beyond it.'
    },
    {
      kind: 'select',
      id: 'densityGrid',
      label: 'Grid resolution',
      group: 'Track density',
      apply: 'compile',
      display: 'segmented',
      default: '1024',
      help: 'Columns of the density grid (rows are half as many). The grid size is compile-time, so a new choice rebuilds the graph. The grid covers the lower 48; cell width shrinks with the cosine of the latitude.',
      options: [
        {value: '512', label: 'Coarse'},
        {value: '1024', label: 'Medium'},
        {value: '2048', label: 'Fine'}
      ]
    },
    {
      kind: 'select',
      id: 'densityValue',
      label: 'Cell value',
      group: 'Track density',
      apply: 'param',
      display: 'segmented',
      default: 'density',
      help: 'Track length inside each cell in kilometres, or that length divided by the exact spherical area of the cell (km of track per km²). They differ because cells shrink toward the pole: a cell at 46° N is about a fifth smaller than one at 29° N.',
      options: [
        {value: 'density', label: 'km per km²'},
        {value: 'length', label: 'km per cell'}
      ]
    },
    {
      kind: 'select',
      id: 'stretch',
      label: 'Colour classes',
      group: 'Track density',
      apply: 'param',
      display: 'segmented',
      default: 'quantile',
      help: 'How the five classes divide the occupied cells. Linear, square root and log make equal-width classes on that scale up to the busiest cell; quantile puts a fifth of the occupied cells in each class. Breaks are read back from the GPU grid once per choice.',
      options: STRETCHES.map(value => ({value, label: STRETCH_LABELS[value]}))
    },
    {
      kind: 'select',
      id: 'gateFocus',
      label: 'Gate',
      group: 'Gates',
      apply: 'param',
      default: 'all',
      help: 'The gate whose crossings are ringed on the map and charted per hour. All gates sums them.',
      options: [
        {value: 'all', label: 'All gates'},
        ...SHIPPING_GATES.map(gate => ({value: gate.id, label: gate.name}))
      ]
    },
    {
      kind: 'select',
      id: 'vesselFilter',
      label: 'Show vessel type',
      group: 'Layers',
      apply: 'param',
      default: 'all',
      expert: true,
      help: 'Filters markers and trails to one AIS vessel type. Markers are culled in the vertex shader; trails use a selection mask in the time-window graph.',
      options: [{value: 'all', label: 'All vessels'}, ...typeOptions]
    },
    {
      kind: 'slider',
      id: 'markerSize',
      label: 'Arrow size',
      group: 'Layers',
      apply: 'param',
      min: 0,
      max: 14,
      step: 1,
      default: 0,
      unit: 'px',
      expert: true,
      format: value => (value === 0 ? 'by zoom' : `${value} px`),
      help: 'Half-length of each mark in screen pixels. By zoom grows from 5 px over the country to 7 px at harbour scale.'
    },
    {
      kind: 'slider',
      id: 'densityOpacity',
      label: 'Density opacity',
      group: 'Track density',
      apply: 'param',
      min: 0.2,
      max: 1,
      step: 0.05,
      default: 0.92,
      expert: true,
      help: 'Opacity of the density raster over the basemap.'
    },
    {
      kind: 'slider',
      id: 'gateHalfWidth',
      label: 'Gate half-width',
      group: 'Gates',
      apply: 'param',
      min: 50,
      max: 1500,
      step: 50,
      default: 250,
      unit: 'm',
      expert: true,
      help: 'Half the width of each gate rectangle. Crossing counts barely change (a segment through a thin strip enters and leaves once), but a wide strip also catches vessels that start or stop inside it.'
    },
    {
      kind: 'select',
      id: 'eventsPerTrack',
      label: 'Events kept per track',
      group: 'Gates',
      apply: 'compile',
      default: '16',
      expert: true,
      help: 'Capacity of the event list per vessel track, compile-time. A ferry shuttling through a gate produces many events; when a track needs more, the zone-event capacity readout says so.',
      options: [
        {value: '8', label: '8 events'},
        {value: '16', label: '16 events'},
        {value: '32', label: '32 events'}
      ]
    },
    {
      kind: 'slider',
      id: 'pulseMinutes',
      label: 'Ring fade',
      group: 'Gates',
      apply: 'param',
      min: 1,
      max: 60,
      step: 1,
      default: 20,
      unit: 'min',
      expert: true,
      help: 'How long, in simulated minutes, a crossing ring stays visible after the playhead passes it.'
    }
  ],

  readouts: [
    {
      id: 'active',
      label: 'Vessels with a position now',
      format: 'integer',
      unit: 'vessels',
      emphasis: 'tile',
      help: 'Tracks whose first and last fix bracket the playhead and pass the gap test.'
    },
    {
      id: 'movingShare',
      label: 'Of those, moving',
      format: 'percent',
      help: 'Share of the vessels with a position whose interpolated speed is at least 0.5 knots; the rest are drawn as squares.'
    },
    {
      id: 'inGap',
      label: 'Hidden in a data gap',
      format: 'integer',
      unit: 'vessels',
      help: 'Tracks hidden because the fixes around the playhead are further apart than the gap limit.'
    },
    {
      id: 'cellSize',
      label: 'Cell size here',
      emphasis: 'tile',
      help: 'Width and height of one grid cell at the latitude of the map centre. The width shrinks with the cosine of the latitude.'
    },
    {
      id: 'clipWalk',
      label: 'One segment, cell by cell',
      kind: 'chart',
      help: 'A real segment from the data over the grid: the cells it crosses, the cut points and the length of each piece in kilometres.'
    },
    {
      id: 'pieces',
      label: 'Pieces emitted',
      hood: true,
      help: 'Segment-cell pieces GPULineDensity emitted before the stable sort.'
    },
    {
      id: 'overflow',
      label: 'Piece capacity',
      hood: true,
      help: 'The piece list has a compile-time capacity. If it is exceeded the busiest cells are low and this says so.'
    },
    {
      id: 'portVsLane',
      label: 'Busiest cell vs the median cell',
      emphasis: 'tile',
      help: 'The largest cell value divided by the median of the occupied cells, read back from the GPU grid.'
    },
    {id: 'busiestCell', label: 'Busiest cell', help: 'The largest value in any cell.'},
    {
      id: 'medianCell',
      label: 'Median occupied cell',
      help: 'Half the cells with any track are below this.'
    },
    {id: 'occupiedCells', label: 'Cells with any track'},
    {
      id: 'cellHistogram',
      label: 'Cells by value, with the class breaks',
      kind: 'chart',
      help: 'The occupied cells on a logarithmic axis, coloured by class, with the five-class breaks of the chosen stretch marked.'
    },
    {
      id: 'groupShares',
      label: 'Track length in view, by group',
      kind: 'chart',
      help: 'Kilometres of track recorded over the day inside the map view, by vessel group.'
    },
    {
      id: 'towShare',
      label: 'Tugs and tows, share of track in view',
      format: 'percent',
      emphasis: 'tile'
    },
    {
      id: 'crossings',
      label: 'Crossings today',
      format: 'integer',
      emphasis: 'tile',
      help: 'Enter events of the chosen gate over the day (all gates when none is chosen).'
    },
    {
      id: 'hourly',
      label: 'Crossings per hour',
      kind: 'chart',
      help: 'Enter events of the gate by UTC hour, stacked by direction. The vertical rule is the playhead.'
    },
    {
      id: 'chord',
      label: 'Median time between the fixes at a crossing',
      format: 'decimal',
      unit: 'min',
      help: 'The crossing time is interpolated along the straight chord between the two fixes either side of the gate; this is how long that chord lasts.'
    },
    {
      id: 'medianFixGap',
      label: 'Median time between fixes',
      format: 'decimal',
      unit: 'min',
      emphasis: 'tile',
      help: 'Median interval between consecutive fixes of a track over the whole day. Parked vessels were thinned to about one fix an hour.'
    },
    {
      id: 'gapShare',
      label: 'Intervals longer than the limit',
      format: 'percent',
      help: 'Share of all fix-to-fix intervals of the day longer than the gap limit.'
    },
    {
      id: 'zoneHealth',
      label: 'Zone-event capacity',
      hood: true,
      help: 'Events found, candidate segment-edge pairs, and any overflow flag of GPUZoneEvents.'
    }
  ],

  pipeline: [
    {
      id: 'playhead',
      label: 'Playhead',
      detail: 'Every vessel interpolated between its two fixes at the clock',
      show: {option: 'showDensity', value: false}
    },
    {
      id: 'clip',
      label: 'Clip and walk',
      detail: 'Each segment is cut at every cell edge into (cell, length) pieces',
      show: {option: 'showDensity', value: true}
    },
    {
      id: 'sum',
      label: 'Sort and sum',
      detail: 'A stable sort by cell, then a fixed-order sum and a divide by the exact cell area',
      show: {option: 'showDensity', value: true}
    },
    {
      id: 'classes',
      label: 'Classes',
      detail: 'Breaks from the occupied cells, read back once; colours from one class table',
      show: {option: 'showDensity', value: true}
    },
    {
      id: 'gate',
      label: 'Gate events',
      detail: 'Segments meet the gate rectangles: exact crossings with interpolated times',
      show: {option: 'showGates', value: true}
    }
  ],

  timeline: {
    time: 'time',
    play: 'play',
    speed: 'speed',
    format: formatPlaybackTime.clockUtc,
    ticks: [
      {at: 0, label: '00'},
      {at: 21600, label: '06'},
      {at: 43200, label: '12'},
      {at: 64800, label: '18'}
    ]
  },

  legends: (state, data) => {
    const legends: LegendSpec[] = [];
    if (state.showDensity) {
      const entry = data.densityTable as
        | {
            table: Parameters<typeof getClassTableLegend>[0];
            counts: number[];
            basis: 'length' | 'density';
          }
        | null
        | undefined;
      legends.push(
        entry
          ? getClassTableLegend(entry.table, {
              id: 'density',
              title: entry.basis === 'density' ? 'Track density' : 'Track length per cell',
              counts: entry.counts,
              layout: 'list',
              interactive: true
            })
          : {
              kind: 'categories',
              title: 'Track density',
              entries: [],
              note: 'Reading the grid back from the GPU.'
            }
      );
    }
    if (state.showVessels) legends.push(getVesselLegend(state, data));
    if (state.showGates) legends.push(getGateLegend(state));
    return legends;
  },

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUTrajectoryPlayhead, GPULineDensity, GPUZoneEvents,
  getGPUTrajectoryPlayheadParameterValues, getGPULineDensityParameterValues
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUTimeWindowFilter, getGPUTimeWindowParameterValues} from '@luma.gl/experimental/gpu-dataframe';

// One set of tracks, three coordinate spaces: Web Mercator meters for the playhead (true headings
// on the map), azimuthal-equidistant meters for the gates (true distances), degrees for the grid.
const play = new GPUCommandGraph(device, {id: 'playhead'});
play.add(new GPUTrajectoryPlayhead({positions: mercator, timestamps, trackOffsets, parameters: playheadParameters.importToGraph(play),
  currentPositions, headings, speeds, status, activeTracks: {ids, count, overflow}, drawInstanceCount}));

${
  state.showDensity
    ? `// Line to cells: clip every segment to the grid, walk the cells it crosses, sort the pieces by
// cell (stable) and sum in a fixed order, then divide by the exact spherical cell area.
const density = new GPUCommandGraph(device, {id: 'density'});
density.add(new GPULineDensity({
  positions: lngLat, pathOffsets,                       // float32x2 degrees, one row per fix
  columns: ${state.densityGrid}, rows: ${Number(state.densityGrid) / 2}, coordinateSystem: 'spherical',   // compile-time
  maximumRecords: 6 * fixCount,
  parameters: densityParameters.importToGraph(density), // [minX, minY, cellWidth, cellHeight]
  output: {lengths, densities, overflow, totalRecords}
}));
// Classes are made on the CPU from the occupied cells, once per choice (${STRETCH_LABELS[state.stretch]}):
const breaks = getStretchBreaks(sortedOccupied, '${state.stretch}', 5);
new SpatialAnalysisRasterLayer({values: ${state.densityValue === 'density' ? 'densities' : 'lengths'}, valueScale: ${state.densityValue === 'density' ? 1000 : 0.001},
  classBreaks: breaks, classColors, discardAtOrBelow: 0, gridSize, bounds, tessellation: 64});

`
    : ''
}${
  state.showGates
    ? `const gates = new GPUCommandGraph(device, {id: 'gates'});
gates.add(new GPUZoneEvents({
  positions, timestamps, trackOffsets,             // analysis meters, seconds
  edgeStarts, edgeEnds, edgeZones, zoneCount: ${SHIPPING_GATES.length},
  candidateCapacity: 1 << 18, maxEventsPerTrack: ${state.eventsPerTrack},
  events: {output: {ids: eventTracks, count, overflow}, eventZones, eventTypes, eventTimes, eventPositions}
}));
// crossing time = trackStart[eventTracks[i]] + eventTimes[i], interpolated along the chord

`
    : ''
}// every frame: parameters are plain buffer writes
playheadParameters.write(getGPUTrajectoryPlayheadParameterValues({playhead, maxGap: ${state.maxGapMinutes * 60}}));
windowParameters.write(getGPUTimeWindowParameterValues({start: playhead - ${state.trailMinutes * 60}, end: playhead,
  startFadeDuration: ${state.trailMinutes * 60}}));`,

  about: {
    what: 'Previously: who shares a zone, and for how long, in New York Harbor (*How long do vessels spend in each harbor zone?*). Next: the same idea for birds, a year folded onto one calendar (*Play a year of migration*).\n\n`GPUTrajectoryPlayhead` finds, for every vessel at once, the two AIS fixes either side of the clock and interpolates position, heading and speed; `GPUTimeWindowFilter` keeps the track segments inside a sliding window and fades them toward the tail. `GPULineDensity` clips every segment to a lon/lat grid, walks the cells it crosses, stable-sorts the pieces by cell, sums them in a fixed order (so the result is deterministic) and divides by the exact spherical cell area. `GPUZoneEvents` finds when each track enters and leaves the gate rectangles, with interpolated times and positions (explained in *How long do vessels spend in each harbor zone?*).',
    why: 'A day of tracks becomes a map of where the traffic is. The step from lines to cells is a cartographic choice as well as a computation: a port cell is hundreds of times busier than a lane, so the way the values are classed decides whether lanes are visible at all, and normalising by the exact cell area (km per km², not km per cell) keeps cells at different latitudes comparable. Counting ships through a gate turns the same tracks into a flow.',
    howToRead:
      'Arrows are moving vessels, squares are stopped ones, all one colour until you ask for groups or speed. Trails show the last hour. The density classes are fixed per choice of stretch and the legend counts the occupied cells in each; empty cells are not drawn. Rings mark gate crossings: sky for the first direction, orange for the second. **Coverage:** this is terrestrial AIS, received by shore stations, so open ocean beyond roughly 40 to 60 nautical miles is empty by design, and Class B transponders (many yachts, fishing boats and small tugs) report less often and are the first to drop out in crowded waters. Absence from a map is not absence of ships.'
  },

  // Night ground in both page themes: additive light needs a dark ground. Labels are ours.
  basemap: ground('night', {labels: 'none'}),
  furniture: sceneFurniture,

  create: async ctx => (await import('./us-shipping-day.compute')).createUsShippingDay(ctx),

  story: [
    {
      id: 'one-monday',
      title: "A Monday on America's waters",
      headline: 'Most vessels on the water are not moving',
      textAlternative:
        'Dark map of the lower 48 with thousands of amber arrows and squares along every coast, the Gulf, the Mississippi and the Great Lakes; most marks are squares.',
      body: '**{{active}}** have a position right now, drawn as arrows along their headings; squares are stopped, and only {{movingShare}} are moving. Press **Play** or drag **Time of day (UTC)** and the day crosses the country. Rivers and lakes belong to the network too. *How a vessel is placed between two reports is in [Who is moving in New York Harbor?](#/story/harbor-playback).*',
      optionsMode: 'fresh',
      options: {
        play: true,
        speed: 900,
        time: 36000,
        showVessels: true,
        showTrails: true,
        trailMinutes: 60,
        showBackdrop: true,
        showDensity: false,
        showGates: false
      },
      controls: ['play', 'time'],
      readouts: ['active', 'movingShare'],
      camera: {...SHIPPING_FRAMES.country, transitionMs: 1400},
      furniture: {
        title: cartouche(
          "Where do America's ships go in a day?",
          'Vessel positions · interpolated between AIS fixes · 9 January 2023 (UTC)'
        ),
        scaleBar: sceneFurniture.scaleBar,
        caveat: sceneFurniture.caveat,
        clock: CLOCK
      },
      annotations: [
        ...labelsFor(
          US,
          [
            'port-la',
            'port-long-beach',
            'port-houston',
            'port-new-orleans',
            'port-ny-nj',
            'port-savannah',
            'port-seattle',
            'port-hampton-roads'
          ],
          {
            'port-la': {minZoom: 3},
            'port-long-beach': {minZoom: 3},
            'port-houston': {minZoom: 3},
            'port-new-orleans': {minZoom: 3},
            'port-ny-nj': {minZoom: 3},
            'port-savannah': {minZoom: 3},
            'port-seattle': {minZoom: 3},
            'port-hampton-roads': {minZoom: 3}
          }
        ),
        ...labelsFor(WORLD, ['gulf-of-mexico'], {'gulf-of-mexico': {minZoom: 3}}),
        ...labelsFor(US_SHIPPING_PLACES, [
          'mississippi-river',
          'lake-michigan',
          'atlantic-offshore'
        ])
      ],
      stage: 'playhead'
    },
    {
      id: 'the-lanes',
      title: 'Add the day up',
      headline: 'Add the day up and the lanes appear',
      textAlternative:
        'Density map of the Gulf of Mexico coast in magma colours on a dark ground: bright lanes run along the coast and into Houston, Galveston and the Mississippi, with a boxed segment drawn over it.',
      body: "Every segment is clipped to the grid and walked cell by cell; each piece's length is summed per cell and divided by the cell's exact area. The segment in the diagram is one such walk. Cells are **{{cellSize}}** here: change **Grid resolution** (compile-time, so it rebuilds) and the pieces change. **Cell value** in km per km² corrects for cells that shrink toward the pole.",
      optionsMode: 'fresh',
      options: {
        play: false,
        showDensity: true,
        showVessels: false,
        showTrails: false,
        showBackdrop: false,
        stretch: 'quantile',
        densityValue: 'density',
        densityGrid: '1024'
      },
      controls: ['densityGrid', 'densityValue'],
      readouts: ['clipWalk', 'cellSize', 'pieces', 'overflow'],
      camera: {...SHIPPING_FRAMES.gulf, transitionMs: 2400},
      furniture: {
        title: cartouche(
          'Add the day up and the lanes appear',
          'Track length · km per km² · line to cells · 9 January 2023 (UTC)'
        ),
        ...REGIONAL_FURNITURE,
        clock: false
      },
      annotations: labelsFor(
        US,
        [
          'port-houston',
          'port-new-orleans',
          'port-south-louisiana',
          'port-corpus-christi',
          'port-mobile'
        ],
        {'port-south-louisiana': {minZoom: 5}}
      ).concat(labelsFor(WORLD, ['gulf-of-mexico'])),
      stage: 'clip'
    },
    {
      id: 'port-vs-lane',
      title: 'Port against lane',
      headline: 'The classes decide whether lanes exist',
      textAlternative:
        'The national density map in equal-width classes: only the harbours of New York, Los Angeles and Houston stand out and the lanes are one dim colour; a histogram shows nearly all cells in the lowest class.',
      body: 'The busiest cell carries **{{portVsLane}}** the median occupied cell, so equal-width classes leave most lanes in the lowest class. Try **Colour classes**: square root and log spread the tail, and quantiles give each class a fifth of the occupied cells. *A heavy tail needs a deliberate classification, as in [Classes are a choice](#/story/choropleth-classes).*',
      optionsMode: 'fresh',
      options: {
        play: false,
        showDensity: true,
        showVessels: false,
        showTrails: false,
        showBackdrop: false,
        stretch: 'linear',
        densityValue: 'density',
        densityGrid: '1024'
      },
      controls: ['stretch'],
      readouts: ['portVsLane', 'cellHistogram', 'busiestCell', 'medianCell'],
      camera: {...SHIPPING_FRAMES.country, transitionMs: 2400},
      furniture: {
        title: cartouche(
          'Classes decide whether lanes exist',
          'Track density · km per km² · five classes · 9 January 2023 (UTC)'
        ),
        scaleBar: sceneFurniture.scaleBar,
        caveat: sceneFurniture.caveat,
        clock: false
      },
      annotations: labelsFor(
        US,
        ['port-la', 'port-houston', 'port-new-orleans', 'port-ny-nj', 'port-seattle'],
        {
          'port-la': {minZoom: 3},
          'port-houston': {minZoom: 3},
          'port-new-orleans': {minZoom: 3},
          'port-ny-nj': {minZoom: 3},
          'port-seattle': {minZoom: 3}
        }
      ),
      stage: 'classes'
    },
    {
      id: 'who-is-on-the-water',
      title: 'Who is on the water',
      headline: 'Tugs and tows lead on the lower Mississippi',
      textAlternative:
        'The lower Mississippi and the Louisiana coast on a dark ground; vessels are coloured by group, and the green tug and tow marks line the river while grey and yellow marks dot the Gulf.',
      body: "Colour is opt-in: one colour first, so where ships are reads before what they are. **Vessel colour** now sets four groups. In this view tugs and tows are **{{towShare}}** of the day's track length, the largest share of any group. Switch to *Speed* and the squares at berth stand out from the arrows under way.",
      optionsMode: 'fresh',
      options: {
        play: false,
        time: 61200,
        showVessels: true,
        showTrails: false,
        showBackdrop: true,
        showDensity: false,
        showGates: false,
        markerColor: 'group'
      },
      controls: ['markerColor'],
      readouts: ['groupShares', 'towShare', 'active'],
      camera: {...SHIPPING_FRAMES.lowerMississippi, transitionMs: 2400},
      furniture: {
        title: cartouche(
          'Whose ships are on the water?',
          'Vessel group · AIS ship type · 9 January 2023 (UTC)'
        ),
        ...REGIONAL_FURNITURE,
        clock: CLOCK
      },
      annotations: [
        ...labelsFor(US, ['port-new-orleans', 'port-south-louisiana', 'port-mobile'], {
          'port-south-louisiana': {minZoom: 5}
        }),
        ...labelsFor(US_SHIPPING_PLACES, ['mississippi-river']),
        ...labelsFor(WORLD, ['gulf-of-mexico'])
      ],
      stage: 'playhead'
    },
    {
      id: 'through-a-doorway',
      title: 'Counting through a doorway',
      headline: 'Ships cross this gate all day, in both directions',
      textAlternative:
        'Galveston Bay on a dark ground with a white gate line across the ship channel, amber trails and arrows, and sky and orange rings where vessels cross it.',
      body: 'The gate is a thin rectangle across the Houston Ship Channel in the middle of Galveston Bay; the bay entrance is Bolivar Roads, to the south. Each ring is a crossing, **{{crossings}}** today, timed by interpolating along a chord a median **{{chord}}** long. Press **Play**, or pick another **Gate**. *`GPUZoneEvents` is explained in [How long do vessels spend in each harbor zone?](#/story/zone-dwell).*',
      optionsMode: 'fresh',
      options: {
        play: true,
        speed: 600,
        time: 0,
        showVessels: true,
        showTrails: true,
        trailMinutes: 45,
        showBackdrop: true,
        showDensity: false,
        showGates: true,
        gateFocus: 'houston'
      },
      controls: ['play', 'gateFocus'],
      readouts: ['crossings', 'hourly', 'chord'],
      camera: {...SHIPPING_FRAMES.galvestonBay, transitionMs: 2600},
      furniture: {
        title: cartouche(
          'Counting ships through one doorway',
          'Crossings per hour · UTC · interpolated crossing time'
        ),
        ...REGIONAL_FURNITURE,
        clock: CLOCK
      },
      annotations: [...labelsFor(US, ['bolivar-roads', 'port-galveston']), getGateLabel('houston')],
      stage: 'gate'
    },
    {
      id: 'what-it-cannot-see',
      title: 'What the receivers cannot see',
      headline: 'No dots offshore means no receivers, not no ships',
      textAlternative:
        'The Mid-Atlantic coast on a dark ground: tracks crowd the shore and end along a dashed line, with an empty open ocean beyond it labelled as having no reports.',
      body: 'Shore receivers hear ships within radio range of the coast; beyond the edge nobody was listening, not nobody sailing. The median time between fixes is **{{medianFixGap}}**. Slide **Hide vessels in a gap longer than** and {{gapShare}} of all reporting intervals count as gaps; parked vessels were thinned, so they vanish first. Straight chords between fixes can cut across land. Toggle **Mark where receivers stop**. Next: [Play a year of migration](#/story/migration-season).',
      optionsMode: 'fresh',
      options: {
        play: true,
        speed: 900,
        time: 36000,
        showVessels: true,
        showTrails: true,
        trailMinutes: 60,
        showBackdrop: true,
        showDensity: false,
        showGates: false,
        showReceiverEdge: true,
        maxGapMinutes: 30
      },
      controls: ['maxGapMinutes', 'showReceiverEdge'],
      readouts: ['medianFixGap', 'gapShare', 'inGap'],
      camera: {...SHIPPING_FRAMES.midAtlantic, transitionMs: 2600},
      furniture: {
        title: cartouche(
          'No dots offshore is no receivers',
          'Reporting gaps · terrestrial AIS · 9 January 2023 (UTC)'
        ),
        ...REGIONAL_FURNITURE,
        clock: CLOCK
      },
      annotations: labelsFor(US, ['port-ny-nj', 'port-hampton-roads', 'port-baltimore']),
      stage: 'playhead'
    }
  ] satisfies StoryStep<UsShippingOptions>[]
});
