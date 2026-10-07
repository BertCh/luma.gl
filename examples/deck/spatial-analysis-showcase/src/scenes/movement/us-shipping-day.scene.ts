// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {formatPlaybackTime, playbackOptions} from '../../engine/playback';
import {defineScene} from '../scene';
import type {UsShippingOptions} from './us-shipping-day.compute';
import {
  getShippingLegendEntries,
  SHIPPING_GATES,
  SHIPPING_TYPE_LABELS,
  SHIPPING_TYPES
} from './us-shipping.tracks';

const US_VIEW = {longitude: -96.5, latitude: 38.2, zoom: 3.5};

const typeOptions = SHIPPING_TYPES.map(type => ({value: type, label: SHIPPING_TYPE_LABELS[type]}));

export default defineScene<UsShippingOptions>({
  id: 'us-shipping-day',
  title: 'A day of US coastal shipping',
  chapter: 'movement',
  order: 4,
  summary:
    'Replay 9 January 2023 across the lower 48: 13,000 vessels from the NOAA AIS archive, played on the GPU with fading trails, traffic corridors from line density, hourly crossings of 13 real chokepoints, and anchorages found from stops.',
  contributors: [
    'GPUTrajectoryPlayhead',
    'GPUTimeWindowFilter',
    'GPULineDensity',
    'GPUZoneEvents',
    'GPUTrajectoryMetrics',
    'GPUSpatialClustering'
  ],
  datasets: [{id: 'poopdeck-ais-us', role: 'vessel tracks (AIS, 9 January 2023)'}],
  initialView: US_VIEW,

  options: [
    ...playbackOptions<UsShippingOptions>({
      time: {
        min: 0,
        max: 86340,
        step: 60,
        default: 57600,
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
      label: 'Maximum fix gap',
      group: 'Playback',
      apply: 'param',
      min: 0,
      max: 60,
      step: 5,
      default: 0,
      unit: 'min',
      format: value => (value === 0 ? 'off' : `${value} min`),
      help: 'A vessel whose neighbouring fixes are further apart than this is flagged as in a gap and hidden instead of drawn at a guessed place. This archive reports a vessel about every 12 minutes, so limits under 12 minutes hide almost everything. 0 turns the test off.'
    },
    {
      kind: 'toggle',
      id: 'showVessels',
      label: 'Show vessels',
      group: 'Vessels',
      apply: 'param',
      default: true,
      help: 'Draws one arrow per vessel at the playhead, interpolated between its two surrounding fixes by GPUTrajectoryPlayhead.'
    },
    {
      kind: 'select',
      id: 'vesselFilter',
      label: 'Show vessel type',
      group: 'Vessels',
      apply: 'param',
      default: 'all',
      help: 'Filters markers and trails to one AIS vessel type. The speed charts follow the same choice. Markers are culled in the vertex shader; trails use a selection mask in the time-window graph.',
      options: [{value: 'all', label: 'All vessels'}, ...typeOptions]
    },
    {
      kind: 'select',
      id: 'markerColor',
      label: 'Color arrows by',
      group: 'Vessels',
      apply: 'param',
      default: 'category',
      help: 'Vessel type from the AIS ship-type code, or speed over ground (the playhead speed of the segment each vessel is on, corrected from Mercator to true ground speed).',
      options: [
        {value: 'category', label: 'Vessel type'},
        {value: 'speed', label: 'Speed'}
      ]
    },
    {
      kind: 'slider',
      id: 'markerSize',
      label: 'Arrow size',
      group: 'Vessels',
      apply: 'param',
      min: 3,
      max: 14,
      step: 1,
      default: 6,
      unit: 'px',
      help: 'Half-length of each arrow in screen pixels. Arrows point along the heading of the segment the vessel is on.'
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp',
      group: 'Vessels',
      apply: 'param',
      default: 'viridis',
      help: 'Ramp used wherever a number is the color: speed on arrows or trails, and the traffic-corridor density.',
      options: [
        {value: 'viridis', label: 'Viridis'},
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'cividis', label: 'Cividis (color-blind optimised)'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showBackdrop',
      label: 'Show every track faintly',
      group: 'Vessels',
      apply: 'param',
      default: true,
      help: 'Draws all 15,555 tracks as thin gray lines so the shipping lanes show before any analysis is switched on.'
    },
    {
      kind: 'toggle',
      id: 'showTrails',
      label: 'Show trails',
      group: 'Trails (time window)',
      apply: 'param',
      default: true,
      help: 'Draws the part of every track inside the sliding window behind the playhead. Which segments are live is decided on the GPU by GPUTimeWindowFilter.'
    },
    {
      kind: 'slider',
      id: 'trailMinutes',
      label: 'Trail length',
      group: 'Trails (time window)',
      apply: 'param',
      min: 10,
      max: 240,
      step: 5,
      default: 60,
      unit: 'min',
      disabledWhen: state => !state.showTrails,
      help: 'Window width, written into the window parameter buffer as [playhead - length, playhead]. Fixes are about 12 minutes apart, so trails shorter than that show a single segment.'
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
      help: 'How much of the trail fades out toward its oldest end (the start-fade duration of the window). 100% fades the whole trail; 0 keeps it solid.'
    },
    {
      kind: 'select',
      id: 'trailColor',
      label: 'Color trails by',
      group: 'Trails (time window)',
      apply: 'param',
      default: 'category',
      disabledWhen: state => !state.showTrails,
      help: 'Vessel type, or the speed of each step as measured by GPUTrajectoryMetrics (true distance in the azimuthal-equidistant analysis space).',
      options: [
        {value: 'category', label: 'Vessel type'},
        {value: 'speed', label: 'Step speed'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showDensity',
      label: 'Show traffic corridors',
      group: 'Corridors (line density)',
      apply: 'param',
      default: false,
      help: 'Paints the length of track per grid cell, computed by GPULineDensity in spherical mode: every segment is clipped to the grid and its great-circle length is added to the cells it crosses.'
    },
    {
      kind: 'select',
      id: 'densityType',
      label: 'Corridors of',
      group: 'Corridors (line density)',
      apply: 'compile',
      default: 'all',
      disabledWhen: state => !state.showDensity,
      help: 'The vessel type whose tracks are summed. The contributor takes one packed position table, so each type is its own compiled graph (built on first use).',
      options: [{value: 'all', label: 'All vessels'}, ...typeOptions]
    },
    {
      kind: 'select',
      id: 'densityGrid',
      label: 'Grid resolution',
      group: 'Corridors (line density)',
      apply: 'compile',
      default: '1024',
      disabledWhen: state => !state.showDensity,
      help: 'Columns of the density grid (rows are half as many). The grid size is compile-time, so a new choice rebuilds the graph. Over the whole country 1024 columns are cells of about 6 km; zoom in and use "Current view" for finer cells.',
      options: [
        {value: '512', label: '512 x 256 cells'},
        {value: '1024', label: '1024 x 512 cells'},
        {value: '2048', label: '2048 x 1024 cells'}
      ]
    },
    {
      kind: 'select',
      id: 'densityExtent',
      label: 'Grid covers',
      group: 'Corridors (line density)',
      apply: 'param',
      default: 'us',
      disabledWhen: state => !state.showDensity,
      help: 'The grid origin and cell size are a parameter buffer, so the same compiled graph can cover the lower 48 or just the map you are looking at. "Current view" re-runs the density once the camera has been still for a moment.',
      options: [
        {value: 'us', label: 'Lower 48'},
        {value: 'view', label: 'Current view (finer cells)'}
      ]
    },
    {
      kind: 'select',
      id: 'densityValue',
      label: 'Cell value',
      group: 'Corridors (line density)',
      apply: 'param',
      default: 'length',
      disabledWhen: state => !state.showDensity,
      help: 'Track length inside each cell in kilometers, or that length divided by the exact spherical area of the cell (km of track per square km). They differ only because cells shrink toward the pole.',
      options: [
        {value: 'length', label: 'Track length (km per cell)'},
        {value: 'density', label: 'Line density (km per km2)'}
      ]
    },
    {
      kind: 'slider',
      id: 'densityCeiling',
      label: 'Color ceiling',
      group: 'Corridors (line density)',
      apply: 'param',
      min: 50,
      max: 100,
      step: 1,
      default: 99,
      unit: '%',
      format: value => `${value}th percentile`,
      disabledWhen: state => !state.showDensity,
      help: 'Cells above this percentile of the non-empty cells are drawn at the top color. Harbors are hundreds of times busier than open coast, so a ceiling below 100 keeps the lanes visible.'
    },
    {
      kind: 'slider',
      id: 'densityOpacity',
      label: 'Corridor opacity',
      group: 'Corridors (line density)',
      apply: 'param',
      min: 0.2,
      max: 1,
      step: 0.05,
      default: 0.85,
      disabledWhen: state => !state.showDensity,
      help: 'Opacity of the density raster over the basemap.'
    },
    {
      kind: 'toggle',
      id: 'showGates',
      label: 'Show gates',
      group: 'Gates (zone events)',
      apply: 'param',
      default: false,
      help: 'Draws the 13 gate lines. Each gate is a thin rectangle in the analysis, so a crossing is an enter event followed by an exit event.'
    },
    {
      kind: 'select',
      id: 'gateFocus',
      label: 'Chart gate',
      group: 'Gates (zone events)',
      apply: 'param',
      default: 'all',
      help: 'The gate whose crossings per hour and direction are charted and highlighted on the map. "All gates" charts the sum.',
      options: [
        {value: 'all', label: 'All gates'},
        ...SHIPPING_GATES.map(gate => ({value: gate.id, label: gate.name}))
      ]
    },
    {
      kind: 'slider',
      id: 'gateHalfWidth',
      label: 'Gate half-width',
      group: 'Gates (zone events)',
      apply: 'param',
      min: 50,
      max: 1500,
      step: 50,
      default: 250,
      unit: 'm',
      help: 'Half the width of each gate rectangle. The edge buffer is rewritten and the same compiled graph re-run. Crossing counts barely change (a segment through a thin strip enters and leaves once), but a wide strip also catches vessels that start or stop inside it.'
    },
    {
      kind: 'select',
      id: 'eventsPerTrack',
      label: 'Events kept per track',
      group: 'Gates (zone events)',
      apply: 'compile',
      default: '16',
      help: 'Capacity of the event list per vessel track, compile-time. A ferry shuttling through a gate produces many events; when a track needs more, the overflow readout says so.',
      options: [
        {value: '8', label: '8 events'},
        {value: '16', label: '16 events'},
        {value: '32', label: '32 events'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showCrossings',
      label: 'Flash crossings as they happen',
      group: 'Gates (zone events)',
      apply: 'param',
      default: false,
      help: 'Draws each enter (green) and exit (orange) event at its interpolated crossing position; the ring grows and fades after the playhead passes the crossing time.'
    },
    {
      kind: 'slider',
      id: 'pulseMinutes',
      label: 'Flash length',
      group: 'Gates (zone events)',
      apply: 'param',
      min: 1,
      max: 60,
      step: 1,
      default: 20,
      unit: 'min',
      disabledWhen: state => !state.showCrossings,
      help: 'How long, in simulated minutes, a crossing ring stays visible after the playhead passes it.'
    },
    {
      kind: 'toggle',
      id: 'showRestingCrossings',
      label: 'Show all crossings faintly',
      group: 'Gates (zone events)',
      apply: 'param',
      default: false,
      disabledWhen: state => !state.showCrossings,
      help: 'Also draws every crossing of the day as a faint dot, so the busy gates show up before the clock reaches them.'
    },
    {
      kind: 'toggle',
      id: 'showStops',
      label: 'Show stops',
      group: 'Anchorages (stops and clusters)',
      apply: 'param',
      default: false,
      help: 'Marks every dwell found by GPUTrajectoryMetrics: a run of slow steps lasting at least the minimum duration. Radius and color grow with the duration.'
    },
    {
      kind: 'slider',
      id: 'stopSpeedKnots',
      label: 'Stop speed threshold',
      group: 'Anchorages (stops and clusters)',
      apply: 'param',
      min: 0.1,
      max: 3,
      step: 0.1,
      default: 0.5,
      unit: 'kn',
      help: 'A step is slow when the vessel covers less than this speed times the step time. A parameter-buffer write: the stop list and the anchorage clusters update without any recompile.'
    },
    {
      kind: 'slider',
      id: 'stopMinutes',
      label: 'Minimum stop duration',
      group: 'Anchorages (stops and clusters)',
      apply: 'param',
      min: 15,
      max: 360,
      step: 15,
      default: 120,
      unit: 'min',
      help: 'A run of slow steps becomes a stop only when it lasts at least this long. Moored and anchored vessels stay for hours; a ferry waiting at a slip does not.'
    },
    {
      kind: 'toggle',
      id: 'showAnchorages',
      label: 'Show anchorages',
      group: 'Anchorages (stops and clusters)',
      apply: 'param',
      default: false,
      help: 'Draws the clusters that GPUSpatialClustering (DBSCAN) finds among the stop centroids, one disc per cluster, sized and colored by the total vessel-hours stopped there.'
    },
    {
      kind: 'slider',
      id: 'clusterRadiusKm',
      label: 'Cluster radius (epsilon)',
      group: 'Anchorages (stops and clusters)',
      apply: 'param',
      min: 0.5,
      max: 15,
      step: 0.5,
      default: 3,
      unit: 'km',
      help: 'Two stops are neighbours when their centroids are within this distance (analysis meters). Small radii split a port into its terminals, large ones merge neighbouring ports.'
    },
    {
      kind: 'slider',
      id: 'clusterMinStops',
      label: 'Minimum stops per cluster',
      group: 'Anchorages (stops and clusters)',
      apply: 'param',
      min: 2,
      max: 30,
      step: 1,
      default: 4,
      help: 'A stop is a core point when its neighbourhood holds at least this many stops (itself included). Isolated stops fewer than this are noise: a lone anchored ship is not an anchorage.'
    },
    {
      kind: 'select',
      id: 'rankBy',
      label: 'Rank anchorages by',
      group: 'Anchorages (stops and clusters)',
      apply: 'param',
      default: 'dwell',
      help: 'Total stopped time in vessel-hours, or the number of stops. The ranking is computed on the CPU from the clusters, labels and durations the GPU read back.',
      options: [
        {value: 'dwell', label: 'Vessel-hours stopped'},
        {value: 'stops', label: 'Number of stops'}
      ]
    },
    {
      kind: 'select',
      id: 'speedSource',
      label: 'Speed charts from',
      group: 'Speeds',
      apply: 'param',
      default: 'derived',
      help: 'Derived: the distance between consecutive fixes divided by their time, measured by GPUTrajectoryMetrics. Reported: the speed over ground each vessel transmitted. Derived speeds are lower on turns because a straight chord is shorter than the path.',
      options: [
        {value: 'derived', label: 'Derived from positions (GPU)'},
        {value: 'reported', label: 'Reported by the vessel (AIS)'}
      ]
    }
  ],

  readouts: [
    {
      id: 'clock',
      label: 'Playhead',
      help: 'UTC with the Pacific and Eastern clocks on 9 January 2023 (standard time).'
    },
    {
      id: 'active',
      label: 'Vessels with a position now',
      format: 'integer',
      help: 'Tracks whose first and last fix bracket the playhead and pass the gap test. Most are moored: only a few thousand are moving.'
    },
    {
      id: 'inGap',
      label: 'In a data gap',
      format: 'integer',
      help: 'Tracks hidden because the fixes around the playhead are further apart than the maximum gap.'
    },
    {
      id: 'trailSegments',
      label: 'Trail segments live',
      format: 'integer',
      help: 'Segments inside the time window, counted by GPUTimeWindowFilter.'
    },
    {
      id: 'tracks',
      label: 'Tracks',
      help: 'A vessel has more than one track when it stopped reporting for over 45 minutes.'
    },
    {id: 'fixes', label: 'Fixes'},
    {
      id: 'selected',
      label: 'Selected vessel',
      help: 'Click a vessel to read its type, MMSI, length, distance and top speed.'
    },
    {id: 'density', label: 'Corridor grid'},
    {
      id: 'densityPieces',
      label: 'Line pieces',
      help: 'Segment-cell pieces emitted by GPULineDensity. If the capacity is exceeded the busiest cells are low and this says so.'
    },
    {id: 'densityTotal', label: 'Track length'},
    {id: 'crossings', label: 'Gate crossings today'},
    {id: 'gate', label: 'Chart gate'},
    {
      id: 'zoneHealth',
      label: 'Zone-event capacity',
      help: 'Events found, candidate segment-edge pairs, and any overflow flag of GPUZoneEvents.'
    },
    {
      id: 'hourly',
      label: 'Crossings per hour (UTC)',
      kind: 'chart',
      help: 'Enter events of the gate rectangles by hour. The vertical rule is the playhead.'
    },
    {id: 'gateTotals', label: 'Crossings per gate', kind: 'chart'},
    {id: 'stops', label: 'Stops detected'},
    {id: 'anchorages', label: 'Stopping places found'},
    {
      id: 'anchorageList',
      label: 'Top stopping places',
      help: 'Clusters of stops ranked by the chosen measure, named after the nearest port within 40 km.'
    },
    {id: 'anchorRank', label: 'Rank chart', kind: 'chart'},
    {
      id: 'fastest',
      label: 'Fastest step',
      help: 'Highest speed over any single step between two fixes.'
    },
    {
      id: 'typeTable',
      label: 'Speed and parking by type',
      help: 'Median moving speed, share of vessel time stationary and number of tracks per type.'
    },
    {id: 'typeSpeeds', label: 'Median moving speed by type', kind: 'chart'},
    {
      id: 'speedHist',
      label: 'Speed distribution',
      kind: 'chart',
      help: 'Time-weighted histogram of moving steps (at least 0.5 knots) for the vessel type chosen in Show vessel type; all types when it is All.'
    }
  ],

  legends: state => {
    const usesCategory =
      (state.showVessels && state.markerColor === 'category') ||
      (state.showTrails && state.trailColor === 'category');
    const usesSpeed =
      (state.showVessels && state.markerColor === 'speed') ||
      (state.showTrails && state.trailColor === 'speed');
    return [
      ...(usesCategory
        ? [
            {
              kind: 'categories' as const,
              title: 'Vessel type',
              entries: getShippingLegendEntries(),
              note: 'AIS ship-type code grouped into seven classes; "other" holds yachts, sailing vessels and unknown types.'
            }
          ]
        : []),
      ...(usesSpeed
        ? [
            {
              kind: 'ramp' as const,
              title: 'Speed over ground',
              ramp: state.ramp,
              extent: [0, 25] as const,
              unit: 'kn',
              format: (value: number) => value.toFixed(0)
            }
          ]
        : []),
      ...(state.showDensity
        ? [
            {
              kind: 'ramp' as const,
              id: 'density',
              title:
                state.densityValue === 'length'
                  ? 'Track length per cell'
                  : 'Line density (km of track per km2)',
              ramp: state.ramp,
              extent: 'gpu' as const,
              sqrtScale: true,
              unit: state.densityValue === 'length' ? 'km' : 'km/km2',
              format: (value: number) => (value >= 100 ? value.toFixed(0) : value.toFixed(1))
            }
          ]
        : []),
      ...(state.showGates
        ? [
            {
              kind: 'categories' as const,
              title: 'Gates',
              entries: [
                {color: [255, 255, 255, 255] as const, label: 'Gate line (approximate)'},
                {color: [255, 200, 40, 255] as const, label: 'Gate shown in the chart'},
                ...(state.showCrossings
                  ? [
                      {color: [77, 230, 140, 255] as const, label: 'Vessel enters the gate zone'},
                      {color: [255, 115, 64, 255] as const, label: 'Vessel leaves it'}
                    ]
                  : [])
              ]
            }
          ]
        : []),
      ...(state.showStops || state.showAnchorages
        ? [
            {
              kind: 'size' as const,
              title: state.showAnchorages
                ? 'Anchorage discs (radius grows with vessel-hours)'
                : 'Stops (radius grows with the dwell)',
              entries: state.showAnchorages
                ? [
                    {radiusPixels: 10, label: '24 h'},
                    {radiusPixels: 14, label: '100 h'},
                    {radiusPixels: 21, label: '400 h'}
                  ]
                : [
                    {radiusPixels: 4, label: '2 h'},
                    {radiusPixels: 6, label: '12 h'},
                    {radiusPixels: 9, label: 'a day'}
                  ],
              color: [255, 90, 140, 255] as const
            }
          ]
        : [])
    ];
  },

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUTrajectoryPlayhead, GPUTrajectoryMetrics, GPULineDensity, GPUZoneEvents, GPUSpatialClustering,
  getGPUTrajectoryPlayheadParameterValues, getGPUTrajectoryMetricsParameterValues,
  getGPULineDensityParameterValues, getGPUSpatialClusteringParameterValues
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUTimeWindowFilter, getGPUTimeWindowParameterValues} from '@luma.gl/experimental/gpu-dataframe';

// One set of tracks, three coordinate spaces: Web Mercator meters for the playhead (true headings
// on the map), azimuthal-equidistant meters for the analyses (true distances), degrees to draw.
const play = new GPUCommandGraph(device, {id: 'playhead'});
play.add(new GPUTrajectoryPlayhead({positions: mercator, timestamps, trackOffsets, parameters: playheadParameters.importToGraph(play),
  currentPositions, headings, speeds, status, activeTracks: {ids, count, overflow}, drawInstanceCount}));

${
  state.showDensity
    ? `const density = new GPUCommandGraph(device, {id: 'corridors'});
density.add(new GPULineDensity({
  positions: lngLat, pathOffsets,                  // float32x2 degrees, one row per fix
  columns: ${state.densityGrid}, rows: ${Number(state.densityGrid) / 2}, coordinateSystem: 'spherical',   // compile-time
  maximumRecords: 6 * fixCount,
  parameters: densityParameters.importToGraph(density),   // [minX, minY, cellWidth, cellHeight]
  output: {lengths, densities, overflow, totalRecords}
}));
`
    : ''
}const gates = new GPUCommandGraph(device, {id: 'gates'});
gates.add(new GPUZoneEvents({
  positions, timestamps, trackOffsets,             // analysis meters, seconds
  edgeStarts, edgeEnds, edgeZones, zoneCount: ${SHIPPING_GATES.length},
  candidateCapacity: 1 << 18, maxEventsPerTrack: ${state.eventsPerTrack},
  events: {output: {ids: eventTracks, count, overflow}, eventZones, eventTypes, eventTimes, eventPositions}
}));
// crossing time = trackStart[eventTracks[i]] + eventTimes[i]; enter events are the crossings

const stops = new GPUCommandGraph(device, {id: 'stops'});
stops.add(new GPUTrajectoryMetrics({positions, timestamps, trackOffsets, parameters: metricsParameters.importToGraph(stops),
  stepSpeeds, stops: {output: {ids, count, overflow}, centroids, durations}}));
const anchorages = new GPUCommandGraph(device, {id: 'anchorages'});
anchorages.add(new GPUSpatialClustering({positions: stopCentroids, parameters: clusterParameters.importToGraph(anchorages),
  gridSize: [512, 512], labels, clusters: {ids: clusterIds, count: clusterCount, overflow}, clusterSizes, clusterCentroids}));

// every frame: parameters are plain buffer writes
playheadParameters.write(getGPUTrajectoryPlayheadParameterValues({playhead, maxGap: ${state.maxGapMinutes * 60}}));
windowParameters.write(getGPUTimeWindowParameterValues({start: playhead - ${state.trailMinutes * 60}, end: playhead,
  startFadeDuration: ${Math.round(state.trailMinutes * 60 * state.tailFade)}}));
metricsParameters.write(getGPUTrajectoryMetricsParameterValues({
  stopSpeedThreshold: ${((state.stopSpeedKnots * 1852) / 3600).toFixed(2)}, stopMinimumDuration: ${state.stopMinutes * 60}}));   // m/s, s
clusterParameters.write(getGPUSpatialClusteringParameterValues({bounds, epsilon: ${state.clusterRadiusKm * 1000}, minimumPoints: ${state.clusterMinStops}}));`,

  about: {
    what: '`GPUTrajectoryPlayhead` finds, for every vessel at once, the two AIS fixes either side of the clock and interpolates position, heading and speed; `GPUTimeWindowFilter` keeps the track segments inside a sliding window and fades them toward the tail. `GPULineDensity` clips every segment to a lon/lat grid and sums great-circle length per cell. `GPUZoneEvents` finds when each track enters and leaves the gate rectangles, with interpolated times and positions. `GPUTrajectoryMetrics` measures every step and finds stops, and `GPUSpatialClustering` (DBSCAN) groups the stops into anchorages.',
    why: 'Coastal shipping is a system of lanes, gates and waiting places. Port authorities, pilots, planners and researchers ask where the traffic concentrates, how many vessels pass a chokepoint each hour, and where ships wait and for how long. Doing all of it on the GPU means a whole country-day of tracks stays live while you change thresholds, with parameter writes instead of rebuilds.',
    howToRead:
      'Arrows are vessels now, colored by type; trails show the last hour. The corridor raster is brightest where most track length falls in a cell. Gate lines are white; a yellow one is charted; green and orange rings are vessels entering and leaving a gate. Pink discs are stops or, when anchorages are on, clusters of stops sized by vessel-hours. **Coverage:** this is terrestrial AIS, received by shore stations, so open ocean beyond roughly 40 to 60 nautical miles is empty by design, and Class B transponders (many yachts, fishing boats and small tugs) report less often and are the first to drop out in crowded waters. Absence from a map is not absence of ships.'
  },

  create: async ctx => (await import('./us-shipping-day.compute')).createUsShippingDay(ctx),

  story: [
    {
      id: 'a-day-of-shipping',
      title: 'What does one day of US shipping look like?',
      body: 'On Monday 9 January 2023 the NOAA and Coast Guard AIS archive recorded **13,436 vessels** in US waters, here as **15,555 tracks** and 433,000 fixes. Each arrow is a vessel at the playhead, pointing along its heading and colored by type (see the legend). Press **Play** below and watch the day cross the continent from the Pacific coast to the Gulf and the Atlantic.\n\nMost of what you see is *parked*: at 16:00 UTC about 11,700 vessels have a position and only about 2,400 of them are moving. The faint gray lines are every track of the day; the shipping lanes are already there before any analysis starts.',
      camera: {...US_VIEW, transitionMs: 1400},
      options: {
        play: true,
        speed: 900,
        time: 54000,
        showTrails: false,
        showBackdrop: true,
        showVessels: true,
        showDensity: false,
        showGates: false,
        showCrossings: false,
        showStops: false,
        showAnchorages: false
      },
      controls: ['play', 'time', 'speed'],
      readouts: ['clock', 'active', 'tracks'],
      highlight: {readout: 'active'}
    },
    {
      id: 'trails',
      title: 'Ferries, tugs and cargo ships in Puget Sound',
      body: '**`GPUTimeWindowFilter`** treats every segment between two fixes as a time interval and keeps the ones that overlap `[playhead - length, playhead]`. It also writes a fade weight (old end transparent) and a clip fraction, then compacts the live ids and writes the count straight into the draw call. Nothing returns to the CPU.\n\nPuget Sound has the busiest water in the data: Washington State ferries shuttling between Seattle, Bainbridge and Bremerton (teal), tugs and barges (purple), cargo ships bound for Tacoma (blue). Drag **Trail length** to 180 minutes and the ferry lanes turn into woven bundles. Set **Show vessel type** to *Passenger* to isolate them.',
      camera: {longitude: -122.5, latitude: 47.75, zoom: 8.4, transitionMs: 2600},
      options: {
        play: true,
        speed: 600,
        showTrails: true,
        trailMinutes: 90,
        trailColor: 'category',
        showBackdrop: true,
        showVessels: true,
        vesselFilter: 'all'
      },
      callout: {coordinate: [-122.43, 47.62], text: 'Seattle - Bainbridge ferry lane'},
      controls: ['trailMinutes', 'trailColor', 'tailFade', 'vesselFilter'],
      readouts: ['trailSegments', 'active']
    },
    {
      id: 'corridors',
      title: 'Where are the corridors?',
      body: '**`GPULineDensity`** answers a different question: not who is where now, but where the traffic of the whole day concentrates. Every segment is clipped to the grid and its great-circle length is added to each cell it crosses, so the picture is the sum of 433,000 fixes in one GPU pass. Spherical mode matters here: the grid is in degrees across 25 degrees of latitude.\n\nThe bright threads along the Gulf Coast are the Intracoastal Waterway and the lanes to Houston, Galveston and the Mississippi delta; the Great Lakes, the Ohio and Mississippi towboat routes and the Florida coast show too. Try **Corridors of** *Tanker*, then *Tug or tow*, to see how each type uses a different network. Set **Grid covers** to *Current view* and zoom in for finer cells.',
      camera: {longitude: -91.5, latitude: 29.6, zoom: 5.6, transitionMs: 2800},
      options: {
        play: false,
        time: 57600,
        showDensity: true,
        showTrails: false,
        showBackdrop: false,
        showVessels: false,
        densityType: 'all',
        densityValue: 'length',
        densityExtent: 'us',
        densityGrid: '1024'
      },
      controls: ['densityType', 'densityValue', 'densityCeiling', 'densityExtent'],
      readouts: ['density', 'densityTotal']
    },
    {
      id: 'gates',
      title: 'How many ships pass Galveston Bay each hour?',
      body: '**`GPUZoneEvents`** detects when each track enters and leaves a polygon, with the crossing time and position interpolated along the segment. Each of the 13 gates is a thin rectangle across a real chokepoint (Golden Gate, Angels and Queens Gates, Juan de Fuca, the Houston Ship Channel, New Orleans, Chesapeake Bay, Ambrose Channel and others), so a crossing is an *enter* event; the side it enters from gives the direction.\n\nThe yellow gate is the **Houston Ship Channel** in Galveston Bay. Press play and watch rings flash as vessels cross; the chart shows its crossings per hour, split into northbound and southbound. Change **Chart gate** to compare it with the Seattle ferries or the Mississippi at New Orleans.',
      camera: {longitude: -94.82, latitude: 29.42, zoom: 9.2, transitionMs: 2400},
      options: {
        play: true,
        speed: 900,
        time: 36000,
        showGates: true,
        showCrossings: true,
        showTrails: true,
        trailMinutes: 45,
        showDensity: false,
        showBackdrop: true,
        showVessels: true,
        gateFocus: 'houston'
      },
      callout: {coordinate: [-94.82, 29.42], text: 'Houston Ship Channel gate'},
      controls: ['gateFocus', 'gateHalfWidth', 'pulseMinutes'],
      readouts: ['gate', 'hourly', 'gateTotals']
    },
    {
      id: 'anchorages',
      title: 'Where do ships wait?',
      body: 'A **stop** is a run of slow steps (below the speed threshold) that lasts at least the minimum duration. **`GPUTrajectoryMetrics`** finds them for every track in one pass, then **`GPUSpatialClustering`** (DBSCAN) groups the stop centroids: a cluster needs a minimum number of stops within a radius, and isolated stops are noise. Ranked by summed dwell, the clusters are the places where vessels spend their day.\n\nThe lower Mississippi, from New Orleans past Baton Rouge, is crowded with towboats, barges and tankers at berth and at anchor. Slide **Cluster radius** up to merge terminals into whole ports, or raise **Minimum stops per cluster** to keep only the big ones. This cannot tell an anchorage from a pier: both are a vessel that did not move.',
      camera: {longitude: -90.6, latitude: 29.9, zoom: 8.2, transitionMs: 2400},
      options: {
        play: false,
        time: 57600,
        showStops: true,
        showAnchorages: true,
        showTrails: false,
        showBackdrop: true,
        showVessels: false,
        showGates: false,
        showCrossings: false,
        showDensity: false,
        stopMinutes: 120,
        clusterRadiusKm: 3,
        clusterMinStops: 4,
        rankBy: 'dwell'
      },
      controls: ['clusterRadiusKm', 'clusterMinStops', 'stopMinutes', 'rankBy'],
      readouts: ['anchorages', 'anchorageList', 'anchorRank'],
      highlight: {readout: 'anchorages'}
    },
    {
      id: 'speeds',
      title: 'How fast does each kind of vessel go?',
      body: "Every step between two fixes has a speed, measured on the GPU by `GPUTrajectoryMetrics` in true distance. Weighted by the time each step lasts, the histogram shows how a type spends its moving hours: cargo ships and tankers cruise around 11 to 12 knots (median), passenger vessels and the special-service craft sit near 5, tugs and towboats pushing barges near 4.5, and fishing vessels crawl at about 3.5.\n\nSet **Show vessel type** to compare, and flip **Speed charts from** between positions and *Reported by the vessel* to see where a straight-line chord underestimates a vessel that turns between fixes. The table in the panel adds the share of each type's time spent stationary: most vessel-hours in this data are spent tied up.",
      camera: {...US_VIEW, transitionMs: 2400},
      options: {
        play: true,
        speed: 900,
        showTrails: false,
        showBackdrop: false,
        showVessels: true,
        markerColor: 'speed',
        showDensity: false,
        showGates: false,
        showCrossings: false,
        showStops: false,
        showAnchorages: false
      },
      controls: ['vesselFilter', 'speedSource', 'markerColor'],
      readouts: ['typeSpeeds', 'speedHist', 'typeTable']
    },
    {
      id: 'limits',
      title: 'What to remember, and what to try',
      body: 'This is **terrestrial AIS**: vessels are heard by shore stations, so the open ocean is empty and gaps appear wherever a receiver drops out. Class B transponders, on many yachts, fishing boats and small tugs, report less often and lose the contest for radio slots in crowded waters, so small craft are under-counted, most of all in harbors. The archive thins each vessel to about one fix per 12 minutes, short harbor shuffles under 2 km are dropped, and a parked vessel keeps one fix an hour; Alaska, Hawaii and Puerto Rico are outside the window. Straight lines between fixes cut corners and sometimes cross land, so gate counts are lower bounds and approximate.\n\n**Try it:** open the Chesapeake Bay and Ambrose Channel gates one after another and compare their morning and evening peaks; switch **Corridors of** to *Fishing* on the Pacific coast; set **Gate half-width** to 1,500 m and watch the counts hold.',
      camera: {longitude: -75.2, latitude: 38.6, zoom: 5.9, transitionMs: 2800},
      options: {
        play: true,
        speed: 900,
        showTrails: true,
        trailMinutes: 60,
        showBackdrop: true,
        showVessels: true,
        markerColor: 'category',
        showGates: true,
        showCrossings: true,
        gateFocus: 'chesapeake',
        showDensity: false,
        showStops: false,
        showAnchorages: false,
        vesselFilter: 'all'
      },
      controls: ['gateFocus', 'gateHalfWidth', 'vesselFilter'],
      readouts: ['crossings', 'hourly']
    }
  ]
});
