// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {formatPlaybackTime, playbackOptions} from '../../engine/playback';
import {joinCredits} from '../../cartography/credits';
import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {FlightCorridorsOptions} from './flight-corridors.compute';
import {MOVEMENT_CREDITS} from './movement-style';

const US_VIEW = {longitude: -96, latitude: 38.5, zoom: 3.9};

const ALTITUDE_RAMP_OPTIONS = [
  {value: 'magma', label: 'Magma'},
  {value: 'inferno', label: 'Inferno'}
] as const;

export default defineScene<FlightCorridorsOptions>({
  id: 'flight-corridors',
  title: 'Where do jets fly over America?',
  chapter: 'movement',
  order: 10,
  summary:
    'Replay a full day of US jet traffic on the GPU, then sum every flight into a line-density grid: the airways emerge as bright corridors, and an extruded view lifts every flight to its altitude.',
  contributors: ['GPUTrajectoryPlayhead', 'GPUTimeWindowFilter', 'GPULineDensity'],
  datasets: [{id: 'poopdeck-adsb-paths', role: 'flight trajectories (OpenSky ADS-B, 6 Jan 2020)'}],
  initialView: US_VIEW,
  basemap: ground('night', {labels: 'none'}),
  furniture: {
    title: {
      title: 'A day of US jet traffic',
      subtitle: 'OpenSky ADS-B · 6 January 2020 (UTC)',
      chips: ['Receivers hear what they hear', 'Simplified tracks']
    },
    scaleBar: {units: 'metric'},
    credit: joinCredits(MOVEMENT_CREDITS.openSky),
    clock: {
      option: 'time',
      time: {origin: '2020-01-06T00:00:00Z', unit: 'seconds'},
      zones: ['UTC', 'America/New_York', 'America/Los_Angeles'],
      progress: [0, 86400]
    }
  },

  options: [
    ...playbackOptions<FlightCorridorsOptions>({
      ids: {play: 'playing', time: 'time', speed: 'playbackSpeed', loop: 'loop'},
      time: {
        min: 0,
        max: 86400,
        step: 60,
        default: 64800,
        label: 'Time of day (UTC)',
        format: formatPlaybackTime.clockUtc,
        help: 'Monday 6 January 2020. The US is five to eight hours behind UTC in winter, so 18:00 UTC is 1 pm in New York and 10 am in Los Angeles.'
      },
      speed: {
        min: 60,
        max: 3600,
        step: 60,
        default: 600,
        unit: 'x',
        help: 'Simulated seconds per real second. 600x plays an hour in 6 seconds and the whole day in under 2.5 minutes.'
      },
      loop: true
    }),
    {
      kind: 'slider',
      id: 'maxGapMinutes',
      label: 'Maximum fix gap',
      group: 'Playback',
      apply: 'param',
      min: 0,
      max: 15,
      step: 1,
      default: 0,
      unit: 'min',
      format: value => (value === 0 ? 'off' : `${value} min`),
      help: 'When the two stored positions either side of the playhead are further apart than this, the flight is flagged as in a gap and its marker is hidden. The data is simplified to long straight steps (up to 10 minutes) on purpose, so a limit under 10 hides ordinary cruise; 0 turns the test off.'
    },
    {
      kind: 'select',
      id: 'direction',
      label: 'Flight direction',
      group: 'Flights',
      apply: 'compile',
      default: 'all',
      help: 'Keeps flights whose first-to-last bearing points east, west, north or south (a 90 degree cone each). It filters the trails (an extra predicate mask on the time-window graph) and the corridors (a different track subset feeds a separate line-density graph, built the first time you pick it). Markers and the faint backdrop always show every flight.',
      options: [
        {value: 'all', label: 'All flights'},
        {value: 'east', label: 'Eastbound'},
        {value: 'west', label: 'Westbound'},
        {value: 'north', label: 'Northbound'},
        {value: 'south', label: 'Southbound'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showMarkers',
      label: 'Show aircraft',
      group: 'Aircraft',
      apply: 'param',
      default: true,
      help: 'One disc per airborne flight, placed by GPUTrajectoryPlayhead at the clock. Only the active flights are drawn, through a compact id list and an indirect draw count written on the GPU.'
    },
    {
      kind: 'select',
      id: 'markerColor',
      label: 'Color aircraft by',
      group: 'Aircraft',
      apply: 'param',
      default: 'altitude',
      disabledWhen: state => !state.showMarkers,
      help: 'Interpolated altitude, or the compass direction of the flight (blue eastbound, orange westbound, green northbound, lilac southbound).',
      options: [
        {value: 'altitude', label: 'Altitude'},
        {value: 'direction', label: 'Flight direction'}
      ]
    },
    {
      kind: 'slider',
      id: 'markerSize',
      label: 'Aircraft size',
      group: 'Aircraft',
      apply: 'param',
      min: 1.5,
      max: 8,
      step: 0.5,
      default: 3,
      unit: 'px',
      disabledWhen: state => !state.showMarkers,
      help: 'Disc radius in screen pixels.'
    },
    {
      kind: 'toggle',
      id: 'showTrails',
      label: 'Show trails',
      group: 'Trails (time window)',
      apply: 'param',
      default: true,
      help: 'Draws the part of every flight inside the sliding window behind the playhead. GPUTimeWindowFilter decides which segments are live and how much of the oldest one to clip.'
    },
    {
      kind: 'slider',
      id: 'trailMinutes',
      label: 'Trail length',
      group: 'Trails (time window)',
      apply: 'param',
      min: 2,
      max: 180,
      step: 1,
      default: 30,
      unit: 'min',
      disabledWhen: state => !state.showTrails,
      help: 'Window width, written into the window parameter buffer as [playhead - length, playhead].'
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
      help: 'How much of the trail fades toward its oldest end (the start-fade duration of the window). 0 keeps it solid.'
    },
    {
      kind: 'select',
      id: 'trailColor',
      label: 'Color trails by',
      group: 'Trails (time window)',
      apply: 'param',
      default: 'altitude',
      disabledWhen: state => !state.showTrails,
      help: 'Altitude of the segment, or the compass direction of the step it belongs to.',
      options: [
        {value: 'altitude', label: 'Altitude'},
        {value: 'direction', label: 'Step direction'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showDensity',
      label: 'Show corridors',
      group: 'Corridors (line density)',
      apply: 'param',
      default: false,
      help: 'Paints the GPULineDensity grid: the length of flight track inside each cell for the whole day, per square kilometre.'
    },
    {
      kind: 'select',
      id: 'densityView',
      label: 'Corridor measure',
      group: 'Corridors (line density)',
      apply: 'compile',
      default: 'density',
      disabledWhen: state => !state.showDensity,
      help: 'Density shows total track per area. Direction balance runs independent eastbound and westbound density graphs, then derives (east - west) / (east + west) on the GPU; blue is westbound-heavy and red is eastbound-heavy.',
      options: [
        {value: 'density', label: 'Track density'},
        {value: 'direction-balance', label: 'East / west balance'}
      ]
    },
    {
      kind: 'select',
      id: 'cellDegrees',
      label: 'Cell size',
      group: 'Corridors (line density)',
      apply: 'compile',
      default: '0.2',
      help: 'Compile-time: the grid is columns x rows, so a new size rebuilds the density graph. 0.2 degrees is about 20 km by 16 km; 0.1 degrees resolves parallel airways and the arrival funnels around airports.',
      options: [
        {value: '0.2', label: '0.2 degrees (300 x 125 cells)'},
        {value: '0.1', label: '0.1 degrees (600 x 250 cells)'}
      ]
    },
    {
      kind: 'select',
      id: 'densityRamp',
      label: 'Corridor color ramp',
      group: 'Corridors (line density)',
      apply: 'param',
      default: 'magma',
      help: 'Density is mapped through a square root so faint airways stay visible next to the busiest ones.',
      options: [
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'}
      ]
    },
    {
      kind: 'slider',
      id: 'densityOpacity',
      label: 'Corridor opacity',
      group: 'Corridors (line density)',
      apply: 'param',
      min: 0.1,
      max: 1,
      step: 0.05,
      default: 0.85,
      disabledWhen: state => !state.showDensity,
      help: 'Opacity of the density grid over the basemap.'
    },
    {
      kind: 'toggle',
      id: 'showBackdrop',
      label: 'Show every flight faintly',
      group: 'Altitude view',
      apply: 'param',
      default: true,
      help: 'All 38,135 flights as thin lines: the whole day at once, and the altitude sculpture when extruded.'
    },
    {
      kind: 'slider',
      id: 'altitudeFloor',
      label: 'Hide below altitude',
      group: 'Altitude view',
      apply: 'param',
      min: 0,
      max: 10000,
      step: 500,
      default: 0,
      unit: 'm',
      format: value => (value === 0 ? 'off' : `${value.toLocaleString('en-US')} m`),
      help: 'Hides trail and backdrop segments whose lower end is below this altitude (shader test). 9,000 m keeps cruise and drops climbs and descents. Aircraft markers are unaffected.'
    },
    {
      kind: 'toggle',
      id: 'extrude',
      label: 'Extrude by altitude',
      group: 'Altitude view',
      apply: 'param',
      default: false,
      help: 'Lifts every vertex to its altitude, aircraft and trails alike. Tilt the map (right-drag or Ctrl-drag) to see the layers of traffic.'
    },
    {
      kind: 'slider',
      id: 'exaggeration',
      label: 'Altitude exaggeration',
      group: 'Altitude view',
      apply: 'param',
      min: 1,
      max: 20,
      step: 1,
      default: 6,
      unit: 'x',
      disabledWhen: state => !state.extrude,
      help: 'A cruise altitude of 11 km is tiny next to a 4,000 km country, so heights are multiplied for readability.'
    },
    {
      kind: 'select',
      id: 'altitudeRamp',
      label: 'Altitude color ramp',
      group: 'Altitude view',
      apply: 'param',
      default: 'magma',
      help: 'Ramp over 0 to 13,000 m for every altitude-colored layer.',
      options: ALTITUDE_RAMP_OPTIONS
    }
  ],

  readouts: [
    {
      id: 'airborneChart',
      label: 'Flights through the day',
      kind: 'chart',
      help: 'Flights with a position in each ten-minute bin of the UTC day. The line is the playhead. Flights are only in the data while an OpenSky receiver hears them.'
    },
    {
      id: 'concentrationChart',
      label: 'How concentrated is the traffic?',
      kind: 'chart',
      help: 'Cumulative share of the distance flown against the busiest share of the grid cells that flights cross. A steep start means a few corridors carry most of the traffic.'
    },
    {
      id: 'altitudeChart',
      label: 'Flight levels by direction',
      kind: 'chart',
      help: 'Eastbound and westbound ADS-B positions are normalized separately. The labels mark each direction’s observed modal altitude band.'
    },
    {id: 'clock', label: 'Playhead', help: 'Simulated time on 6 January 2020.'},
    {
      id: 'airborne',
      label: 'Flights with a position now',
      format: 'integer',
      help: 'Flights whose first and last stored position bracket the playhead (and pass the gap test): the discs drawn.'
    },
    {
      id: 'cruising',
      label: 'Above 9,000 m now',
      format: 'integer',
      help: 'Active flights whose interpolated altitude is at least 9,000 m (about 30,000 feet).'
    },
    {
      id: 'notYet',
      label: 'Not yet airborne',
      format: 'integer',
      help: 'Flights that start after the playhead.'
    },
    {
      id: 'landed',
      label: 'Already landed',
      format: 'integer',
      help: 'Flights that ended before the playhead.'
    },
    {
      id: 'inGap',
      label: 'In a data gap',
      format: 'integer',
      help: 'Flights hidden by the maximum fix gap.'
    },
    {
      id: 'trailSegments',
      label: 'Trail segments live',
      format: 'integer',
      help: 'Segments inside the time window and the direction filter, counted by GPUTimeWindowFilter.'
    },
    {
      id: 'flights',
      label: 'Flights',
      help: 'Flights that reach 7,500 m; light aircraft and helicopters are not in the data.'
    },
    {id: 'vertices', label: 'Stored positions'},
    {
      id: 'flown',
      label: 'Distance flown',
      help: 'Sum of the density grid over the area of its cells. Track outside the grid (west of 125 W, east of 65 W) is not counted.'
    },
    {id: 'cells', label: 'Occupied cells'},
    {
      id: 'concentration',
      label: 'Concentration',
      help: 'Read from the chart: the share carried by the busiest tenth of crossed cells.'
    },
    {
      id: 'pieces',
      label: 'Segment-cell pieces',
      help: 'GPULineDensity clips every segment to the cells it crosses; this is the number of pieces against the capacity compiled for this grid. An overflow would make the counts low.'
    },
    {
      id: 'directionCells',
      label: 'Direction cells',
      help: 'Only cells with at least 0.05 km of combined eastbound and westbound track per km². Lower-support cells are transparent rather than neutral.'
    },
    {
      id: 'directionBalance',
      label: 'Direction balance',
      help: 'Distance-weighted eastbound share and the mean of the signed cell balances.'
    },
    {
      id: 'directionPieces',
      label: 'Direction pieces',
      help: 'Per-direction GPULineDensity segment-cell pieces and capacities. Either overflow would make its side of the balance low.'
    },
    {
      id: 'eastLevelPeak',
      label: 'Eastbound modal level',
      help: 'Highest bin of the normalized eastbound stored-altitude distribution.'
    },
    {
      id: 'westLevelPeak',
      label: 'Westbound modal level',
      help: 'Highest bin of the normalized westbound stored-altitude distribution.'
    }
  ],

  legends: state => [
    ...(state.showDensity && state.densityView === 'density'
      ? [
          {
            kind: 'ramp' as const,
            id: 'density',
            title: 'Flight track density',
            ramp: state.densityRamp,
            extent: 'gpu' as const,
            sqrtScale: true,
            unit: 'km per km²',
            format: (value: number) => value.toFixed(value < 10 ? 1 : 0)
          }
        ]
      : []),
    ...(state.showDensity && state.densityView === 'direction-balance'
      ? [
          {
            kind: 'ramp' as const,
            id: 'directionBalance',
            title: 'East / west balance',
            ramp: 'diverging' as const,
            extent: [-1, 1] as const,
            unit: '(east − west) / total',
            format: (value: number) => value.toFixed(1)
          }
        ]
      : []),
    ...((state.showMarkers && state.markerColor === 'altitude') ||
    (state.showTrails && state.trailColor === 'altitude') ||
    state.extrude
      ? [
          {
            kind: 'ramp' as const,
            title: 'Altitude',
            ramp: state.altitudeRamp,
            extent: [0, 13000] as const,
            unit: 'm',
            format: (value: number) => value.toLocaleString('en-US')
          }
        ]
      : []),
    ...((state.showMarkers && state.markerColor === 'direction') ||
    (state.showTrails && state.trailColor === 'direction')
      ? [
          {
            kind: 'categories' as const,
            title: 'Direction',
            entries: [
              {color: [64, 160, 255, 255] as const, label: 'Eastbound'},
              {color: [255, 140, 50, 255] as const, label: 'Westbound'},
              {color: [140, 210, 150, 255] as const, label: 'Northbound'},
              {color: [200, 150, 230, 255] as const, label: 'Southbound'}
            ]
          }
        ]
      : [])
  ],

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUTrajectoryPlayhead, GPULineDensity,
  getGPUTrajectoryPlayheadParameterValues, getGPULineDensityParameterValues
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUTimeWindowFilter, getGPUTimeWindowParameterValues} from '@luma.gl/experimental/gpu-dataframe';

// lngLat: float32x2 degrees, altitude: float32 metres, timestamps: float32 seconds, trackOffsets: uint32
const playGraph = new GPUCommandGraph(device, {id: 'playhead'});
playGraph.add(new GPUTrajectoryPlayhead({
  positions: lngLat, elevations: altitude, timestamps, trackOffsets,   // interpolate degrees directly
  parameters: playheadParameters.importToGraph(playGraph),
  currentPositions, currentElevations, status,
  activeTracks: {ids: activeIds, count: activeCount, overflow: activeOverflow},
  drawInstanceCount                                                    // indirect draw record
}));

const trailGraph = new GPUCommandGraph(device, {id: 'trails'});
trailGraph.add(new GPUTimeWindowFilter({
  timestamps: segmentStarts, endTimestamps: segmentEnds,               // one interval per segment
  window: windowParameters.importToGraph(trailGraph),
  additionalPredicates: [{kind: 'selection', mask: directionMask}],    // ${state.direction} flights
  output: {ids: trailIds, count: trailCount, overflow: trailOverflow},
  fadeWeights, clipFractions, drawInstanceCount: trailDrawCount
}));

const densityGraph = new GPUCommandGraph(device, {id: 'corridors'});
densityGraph.add(new GPULineDensity({
  positions: lngLat, pathOffsets: trackOffsets,
  columns: ${Math.round(60 / Number(state.cellDegrees))}, rows: ${Math.round(25 / Number(state.cellDegrees))},      // ${state.cellDegrees} degree cells, compile-time
  coordinateSystem: 'spherical',                                       // degrees in, great-circle metres out
  maximumRecords,                                                      // compile-time piece capacity
  parameters: densityParameters.importToGraph(densityGraph),
  output: {lengths, densities, overflow, totalRecords}
}));

// every frame: parameters are plain buffer writes
playheadParameters.write(getGPUTrajectoryPlayheadParameterValues({playhead, maxGap: ${state.maxGapMinutes * 60}}));
windowParameters.write(getGPUTimeWindowParameterValues({
  start: playhead - ${state.trailMinutes * 60}, end: playhead, startFadeDuration: ${Math.round(state.trailMinutes * 60 * state.tailFade)}
}));
densityParameters.write(getGPULineDensityParameterValues({minX: -125, minY: 25, cellWidth: ${state.cellDegrees}, cellHeight: ${state.cellDegrees}}));
playCompiled.encode(commandEncoder, {parameters: undefined});`,

  about: {
    what: '`GPUTrajectoryPlayhead` finds, for every flight at once, the two stored positions either side of the clock and interpolates longitude, latitude and altitude. `GPUTimeWindowFilter` keeps the segments that overlap a sliding window behind the playhead and fades their tails. `GPULineDensity` clips every segment of every flight to a longitude/latitude grid and sums the great-circle length per cell, with no atomics.',
    why: 'Airspace planners, airlines and noise or emissions analysts ask where aircraft actually fly, how traffic piles into a few airways, and how that changes by hour and by direction. A density grid answers it for a whole day of flights without a CPU loop, and the same tracks drive the playback and the 3-D view.',
    howToRead:
      'Discs are aircraft now, colored by altitude or direction. Trails are where they have just been. In the corridor grid a brighter cell holds more flight track per square kilometre; straight bright lines are airways, bright knots are airports where arrivals and departures converge. In the extruded view height is altitude, multiplied for readability. The data is ADS-B as heard by OpenSky receivers: gaps and a thin west coast are coverage, not an empty sky.'
  },

  create: async ctx => (await import('./flight-corridors.compute')).createFlightCorridors(ctx),

  story: [
    {
      id: 'the-day',
      headline: 'One day of traffic fills the sky',
      textAlternative: 'Amber aircraft marks animate over a dark map of the United States.',
      optionsMode: 'fresh',
      title: 'What does a day of jet traffic over America look like?',
      body: 'Monday 6 January 2020: **38,135 jet flights** that reached at least 7,500 m crossed the contiguous United States, as heard by the volunteer OpenSky receiver network. Each disc is one aircraft at the playhead, colored by altitude (see the legend): dark discs are climbing or descending near an airport, bright ones are at cruise.\n\nPress **Play** below, or drag **Time of day (UTC)**; the slider follows the clock. **`GPUTrajectoryPlayhead`** runs one thread per flight, finds the two stored positions around the clock with a binary search and interpolates longitude, latitude and altitude, so the whole day stays live as you scrub. The chart counts flights per ten minutes, so you can see the daily rhythm and where the playhead sits in it.',
      camera: {...US_VIEW, transitionMs: 1200},
      options: {
        playing: true,
        time: 64800,
        showMarkers: true,
        showTrails: false,
        showDensity: false,
        showBackdrop: true,
        extrude: false,
        direction: 'all'
      },
      highlight: {readout: 'airborne'},
      controls: ['playing', 'time', 'playbackSpeed'],
      readouts: ['clock', 'airborne', 'cruising', 'airborneChart']
    },
    {
      id: 'trails',
      headline: 'Comet tails retain one hour of flight',
      textAlternative: 'Direction-coloured aircraft trails cross the northeast.',
      optionsMode: 'fresh',
      title: 'Where have they just been? A window in time',
      body: '**`GPUTimeWindowFilter`** treats every segment between two stored positions as a time interval and keeps those that overlap `[playhead - length, playhead]`. It also writes a fade weight and a clip fraction, so the oldest segment is cut part-way, and writes the live count straight into the draw call. Nothing comes back to the CPU.\n\nSlide **Trail length** up to 90 minutes and the northeast corridor fills with long streaks. Lower **Tail fade** to 0 for solid trails, or switch **Color trails by** to *Step direction*: blue streaks head east, orange west.',
      camera: {longitude: -80, latitude: 39.5, zoom: 5.2, transitionMs: 1400},
      options: {
        playing: false,
        time: 64800,
        showTrails: true,
        trailMinutes: 60,
        tailFade: 1,
        showDensity: false
      },
      controls: ['trailMinutes', 'tailFade', 'trailColor'],
      readouts: ['trailSegments', 'airborne']
    },
    {
      id: 'corridors',
      headline: 'Adding tracks reveals airways',
      textAlternative: 'A luminous density grid shows the busiest air corridors.',
      optionsMode: 'fresh',
      title: 'Add up the whole day: the airways emerge',
      body: '**`GPULineDensity`** clips every segment of all 38,135 flights to a 0.2 degree grid and sums the length inside each cell. Line density is what QGIS calls "Line density"; here it runs in a few milliseconds on the GPU, in longitude/latitude with great-circle lengths (`coordinateSystem: \'spherical\'`).\n\nRead the map: bright straight lines are airways, bright knots are airports where arrivals and departures funnel together, dark areas are the empty sky between them. Try **Cell size** at 0.1 degrees (a compile-time option, so the control shows a rebuild badge): parallel airways separate. The curve below says how lopsided the sky is: a small share of cells carries most of the distance.',
      camera: {...US_VIEW, transitionMs: 1400},
      options: {
        showDensity: true,
        showTrails: false,
        showMarkers: false,
        showBackdrop: false,
        cellDegrees: '0.2',
        densityOpacity: 0.9
      },
      highlight: {readout: 'concentration'},
      controls: ['showDensity', 'cellDegrees', 'densityRamp', 'densityOpacity'],
      readouts: ['flown', 'cells', 'concentration', 'concentrationChart']
    },
    {
      id: 'direction',
      headline: 'The same roads carry both directions',
      textAlternative:
        'A diverging direction balance map distinguishes eastbound and westbound traffic.',
      optionsMode: 'fresh',
      title: 'The sky is one-way streets',
      body: 'Pilots fly airways in a set direction and a tailwind makes eastbound flights faster, so eastbound and westbound traffic uses different tracks. This map runs **two independently buffered `GPULineDensity` graphs**, one eastbound and one westbound, then a GPU pass derives `(east - west) / (east + west)` for every cell. Blue is westbound-heavy, red eastbound-heavy, and the pale middle is genuinely balanced rather than low traffic.\n\nSwitch back to *Track density* or set **Flight direction** to *Eastbound* and then *Westbound* to inspect each input. The trails still take the same filter as a predicate mask on the time-window graph. Westbound flights also fly slower over the ground in winter, because of the jet stream: the next scene measures that from the aircraft alone.',
      camera: {longitude: -98, latitude: 39, zoom: 4.3, transitionMs: 1400},
      options: {
        direction: 'all',
        showDensity: true,
        densityView: 'direction-balance',
        showTrails: false,
        showMarkers: false
      },
      controls: ['densityView', 'direction', 'cellDegrees'],
      readouts: ['directionCells', 'directionBalance', 'directionPieces']
    },
    {
      id: 'extruded',
      headline: 'Flight levels separate opposing lanes',
      textAlternative: 'A pitched view lifts aircraft tracks by their true altitude.',
      optionsMode: 'fresh',
      title: 'Lift every flight to its altitude',
      body: 'Flights stack in layers: short hops below 6,000 m, long-haul jets around 10,000 to 12,000 m. Switch on **Extrude by altitude** and tilt the map: every vertex of the backdrop and the aircraft rises to its altitude, exaggerated by **Altitude exaggeration** because 11 km is tiny against a continent. The corridor grid stays on the ground like a shadow.\n\nThe chart compares the normalized eastbound and westbound flight-level distributions; its labels are the observed modal bins, not a claimed assigned flight level. Set **Hide below altitude** to 9,000 m and the climbs and descents vanish, leaving only cruise. Press Play on the clock to watch jets climb out of an airport and level off.',
      // cartography-allow: pitch (flight altitude is real z)
      camera: {
        longitude: -92,
        latitude: 35,
        zoom: 4.6,
        pitch: 58,
        bearing: -18,
        transitionMs: 1800
      },
      options: {
        direction: 'all',
        extrude: true,
        exaggeration: 8,
        showBackdrop: true,
        showDensity: true,
        densityOpacity: 0.5,
        showMarkers: true,
        showTrails: false,
        playing: false
      },
      controls: ['extrude', 'exaggeration', 'altitudeFloor', 'altitudeRamp'],
      readouts: ['cruising', 'eastLevelPeak', 'westLevelPeak', 'altitudeChart']
    },
    {
      id: 'limits',
      headline: 'Receiver coverage limits the map',
      textAlternative: 'Data coverage edges are shown around the US traffic sample.',
      optionsMode: 'fresh',
      title: 'What to remember, and what to try',
      body: 'This is what receivers heard, not every flight: coverage thins over the mountain west and the Gulf, and the data stops at the edge of the contiguous US (so no Atlantic tracks). Positions were simplified to long straight steps (10 km tolerance, 10 minute maximum), light aircraft are removed, and the OpenSky terms allow research and non-commercial use only.\n\n**Try it:** run **Playback speed** at 3,600x to see the day in 24 seconds; set **Flight direction** to *Southbound* to see the Florida and Texas flows; pick **Cell size** 0.1 degrees and look for parallel airways in the busy northeast.',
      camera: {...US_VIEW, transitionMs: 1400},
      options: {
        extrude: false,
        showDensity: false,
        showBackdrop: true,
        showTrails: true,
        trailMinutes: 30,
        showMarkers: true,
        playing: true,
        playbackSpeed: 1200
      },
      controls: ['playbackSpeed', 'direction', 'cellDegrees'],
      readouts: ['airborne', 'flights']
    }
  ]
});
