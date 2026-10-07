// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CREDITS, joinCredits} from '../../cartography/credits';
import {ground} from '../../cartography/grounds';
import {formatPlaybackTime, playbackOptions} from '../../engine/playback';
import {defineScene} from '../scene';
import type {SatellitePlaybackOptions} from './satellite-playback.compute';
import {getGroupLegendEntries, SATELLITE_GROUPS} from './satellite-tracks';

const WORLD_VIEW = {longitude: 10, latitude: 15, zoom: 1.5, pitch: 0, bearing: 0};
// cartography-allow: pitch (the height step renders propagated satellite altitude as real z with stems)

const elapsed = (seconds: number) => formatPlaybackTime.duration(seconds) || '0 min';

export default defineScene<SatellitePlaybackOptions>({
  id: 'satellite-playback',
  title: 'How do propagated satellites move?',
  chapter: 'earth',
  order: 9,
  summary:
    'Replay a three-hour SGP4 model snapshot: family composition, altitude and motion are visible without claiming an observer or footprint.',
  contributors: ['GPUTrajectoryPlayhead', 'GPUTimeWindowFilter'],
  datasets: [{id: 'celestrak-ground-tracks', role: 'SGP4 ground tracks (7 October 2026, 3 h)'}],
  initialView: WORLD_VIEW,

  options: [
    ...playbackOptions<SatellitePlaybackOptions>({
      time: {
        min: 0,
        max: 10800,
        step: 30,
        default: 1800,
        label: 'Time since 00:00 UTC',
        format: elapsed,
        help: 'Time zero is 7 October 2026, 00:00 UTC; the window is three hours.'
      },
      speed: {
        min: 30,
        max: 900,
        step: 30,
        default: 150,
        unit: 'x',
        help: 'Simulated seconds per real second. 150x plays the three hours in 72 seconds, so a 93-minute low orbit takes about 37 seconds.'
      },
      loop: {default: true}
    }),
    {
      kind: 'select',
      id: 'group',
      label: 'Show family',
      group: 'Satellites',
      apply: 'param',
      default: 'all',
      help: 'Limits the markers and the trails to one family. Markers are culled in the vertex shader; trails use an extra predicate mask written once into the time-window graph (no recompile).',
      options: [
        {value: 'all', label: 'All loaded satellites'},
        ...SATELLITE_GROUPS.map((label, index) => ({value: String(index), label}))
      ]
    },
    {
      kind: 'select',
      id: 'colorBy',
      label: 'Color by',
      group: 'Satellites',
      apply: 'param',
      default: 'group',
      help: 'Family, or a fixed ordered altitude scale from 200 km to 25,600 km. There is no user-selectable colour ramp.',
      options: [
        {value: 'group', label: 'Family'},
        {value: 'altitude', label: 'Altitude'}
      ]
    },
    {
      kind: 'slider',
      id: 'markerSize',
      label: 'Marker size',
      group: 'Satellites',
      apply: 'param',
      min: 2,
      max: 12,
      step: 0.5,
      default: 5,
      unit: 'px',
      help: 'Radius of each satellite disc in screen pixels.'
    },
    {
      kind: 'select',
      id: 'altitudeDisplay',
      label: 'Altitude scale',
      group: 'Altitude',
      apply: 'param',
      default: 'compressed',
      help: 'Linear is the true height times the exaggeration. Compressed uses a logarithm, so GPS at 20,200 km stays on screen next to a 550 km Starlink shell; heights are then ordered correctly but not proportional.',
      options: [
        {value: 'compressed', label: 'Compressed (logarithmic)'},
        {value: 'linear', label: 'Linear (true proportions)'}
      ]
    },
    {
      kind: 'slider',
      id: 'altitudeScale',
      label: 'Altitude exaggeration',
      group: 'Altitude',
      apply: 'param',
      min: 1,
      max: 20,
      step: 0.5,
      default: 3,
      unit: 'x',
      help: 'Multiplies the displayed height. At 1x and linear, a 550 km orbit is 1.4 percent of the width of the world map, so low orbits look flat.'
    },
    {
      kind: 'toggle',
      id: 'showStems',
      label: 'Drop lines to the ground',
      group: 'Altitude',
      apply: 'param',
      default: false,
      help: 'Draws a vertical line from each satellite to the point on the ground below it (its sub-satellite point), which makes height readable from a tilted camera.'
    },
    {
      kind: 'toggle',
      id: 'showTrails',
      label: 'Show trails',
      group: 'Trails (time window)',
      apply: 'param',
      default: true,
      help: 'Draws the part of every ground track inside the sliding window behind the playhead, lifted to the satellite altitude. GPUTimeWindowFilter selects the live segments on the GPU.'
    },
    {
      kind: 'slider',
      id: 'trailMinutes',
      label: 'Trail length',
      group: 'Trails (time window)',
      apply: 'param',
      min: 2,
      max: 120,
      step: 1,
      default: 20,
      unit: 'min',
      disabledWhen: state => !state.showTrails,
      help: 'Window width written into the window parameter buffer: [playhead - length, playhead]. One low orbit takes about 93 minutes.'
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
      help: 'How much of the trail fades out toward its oldest end (the start-fade duration of the window). 100 percent fades the whole trail; 0 keeps it solid.'
    }
  ],

  story: [
    {
      id: 'snapshot',
      title: 'This is a propagated orbital snapshot',
      headline: 'A model replay, not surveillance',
      textAlternative:
        'Propagated satellite positions appear over a sparse dark world map at a UTC playhead.',
      optionsMode: 'fresh',
      body: 'Every mark is an SGP4-propagated position at the clock, sampled every 30 seconds and interpolated by **`GPUTrajectoryPlayhead`**. The cartouche and readouts name the model, sample and exact interval.\n\nPress **Play** or scrub the clock. This is neither a visibility calculation nor a record of an observation at a location.',
      camera: {...WORLD_VIEW, transitionMs: 1400},
      options: {play: false, time: 1800, group: 'all', colorBy: 'group', showStems: false},
      controls: ['play', 'time', 'speed'],
      readouts: ['clock', 'active', 'satellites', 'oldestElements']
    },
    {
      id: 'height',
      title: 'Altitude separates orbital regimes',
      headline: 'Tilt only when height is the subject',
      textAlternative:
        'Satellite altitude stems rise from a global map in a deliberately tilted height diagram.',
      optionsMode: 'fresh',
      body: 'Here pitch is intentional: elevation is the subject. Stems connect the displayed altitude to each sub-satellite point; the altitude chart provides the numerical companion.\n\nCompressed display is labelled non-linear: it preserves order but not proportional distance. Switch to linear to see why low orbit otherwise collapses onto the map.',
      camera: {...WORLD_VIEW, pitch: 58, zoom: 1.8, transitionMs: 1400},
      options: {showStems: true, colorBy: 'altitude', showTrails: false},
      controls: ['altitudeDisplay', 'altitudeScale', 'showStems', 'colorBy'],
      readouts: ['highest', 'lowest', 'altitudeChart']
    },
    {
      id: 'trails',
      title: 'Three hours reveal different orbital tempos',
      headline: 'A trail is a time window',
      textAlternative:
        'Family-coloured satellite trails fade behind current positions on a flat world map.',
      optionsMode: 'fresh',
      body: '**`GPUTimeWindowFilter`** keeps segments inside the trailing interval and writes their fade weights. A short display window makes fast low-orbit motion and slower navigation motion comparable without claiming a location-specific pass.\n\nVary trail length, then continue to the ground-track story for the latitude pattern those paths create.',
      camera: {...WORLD_VIEW, zoom: 1.4, transitionMs: 1400},
      options: {showTrails: true, trailMinutes: 45, colorBy: 'group', showStems: false},
      controls: ['showTrails', 'trailMinutes', 'tailFade'],
      readouts: ['trailSegments']
    },
    {
      id: 'families',
      title: 'The sample is deliberately unbalanced',
      headline: 'A sample is not the constellation',
      textAlternative:
        'A selected satellite family is bright while the deliberately sampled Starlink family remains contextual.',
      optionsMode: 'fresh',
      body: 'Family filtering makes the sample composition explicit. Starlink is an 800-object sample, while GPS has 32 records in this manifest; neither count is a claim about every object in orbit.\n\nChoose a family and select a mark for its propagated metadata, including inclination and element age.',
      camera: {...WORLD_VIEW, zoom: 1.6, transitionMs: 1400},
      options: {group: '1', colorBy: 'group', trailMinutes: 20, showTrails: true},
      controls: ['group', 'markerSize', 'colorBy'],
      readouts: ['familyChart', 'active', 'selected']
    },
    {
      id: 'limits',
      title: 'A TLE replay is not surveillance',
      headline: 'Model inputs constrain every claim',
      textAlternative:
        'A static propagated satellite view is paired with element-age and sample readouts.',
      optionsMode: 'fresh',
      body: 'SGP4 positions depend on public element sets and their age. The model window is short, tracks are cut at the antimeridian, and the sample omits most of the catalogue. No footprints, visibility cones or conjunction claims are calculated.\n\nNext: where do these ground tracks accumulate after their length is normalised by area?',
      camera: {...WORLD_VIEW, zoom: 1.5, transitionMs: 1400},
      options: {group: 'all', showStems: false, colorBy: 'group'},
      controls: ['group'],
      readouts: ['oldestElements', 'altitudeChart', 'selected']
    }
  ],

  legends: state => [
    state.colorBy === 'altitude'
      ? {
          kind: 'ramp' as const,
          title: 'Altitude above the ellipsoid',
          ramp: 'mako' as const,
          extent: [200, 25600] as const,
          scale: 'log' as const,
          unit: 'km',
          format: (value: number) => `${Math.round(value).toLocaleString('en-US')}`
        }
      : {
          kind: 'categories' as const,
          title: 'Family',
          entries: getGroupLegendEntries(),
          note: 'Starlink is a bounded 800-object sample; GPS contributes 32 records in this dated snapshot.'
        }
  ],

  readouts: [
    {
      id: 'clock',
      label: 'Playhead',
      help: 'UTC time of the playhead and the time since the start of the window.'
    },
    {
      id: 'active',
      label: 'Satellites shown',
      format: 'integer',
      help: 'Tracks that bracket the playhead, in the selected family. A satellite has several tracks because they are cut at the antimeridian, so only one is active at a time.'
    },
    {
      id: 'highest',
      label: 'Highest',
      help: 'Altitude of the highest satellite shown, read back every few frames.'
    },
    {id: 'lowest', label: 'Lowest', help: 'Altitude of the lowest satellite shown.'},
    {
      id: 'altitudeChart',
      label: 'Satellites by altitude',
      kind: 'chart',
      help: 'Count of satellites shown in each altitude band right now.'
    },
    {
      id: 'familyChart',
      label: 'Loaded records by family',
      kind: 'chart',
      help: 'A count of the loaded metadata records. The Starlink bar is explicitly a bounded sample.'
    },
    {
      id: 'trailSegments',
      label: 'Trail segments live',
      format: 'integer',
      help: 'Segments inside the time window, counted by GPUTimeWindowFilter.'
    },
    {id: 'satellites', label: 'Satellites'},
    {id: 'samples', label: 'Propagated positions'},
    {
      id: 'oldestElements',
      label: 'Oldest orbital elements',
      help: 'Age of the oldest element set at the start of the window. Position error grows with age.'
    },
    {
      id: 'selected',
      label: 'Selected satellite',
      layout: 'block',
      help: 'Click a satellite to read its family, inclination, orbital period, element age and altitude.'
    }
  ],

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUTrajectoryPlayhead, getGPUTrajectoryPlayheadParameterValues
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUTimeWindowFilter, getGPUTimeWindowParameterValues} from '@luma.gl/experimental/gpu-dataframe';

// positions: float32x2 [longitude, latitude] degrees, cut at the antimeridian so no track wraps
// timestamps: float32 seconds since the window start; altitudes: float32 meters per vertex
const graph = new GPUCommandGraph(device, {id: 'satellites'});
graph.add(new GPUTrajectoryPlayhead({
  positions, timestamps, trackOffsets,
  elevations: altitudes,                          // z per vertex, interpolated with the position
  parameters: playheadParameters.importToGraph(graph),
  currentPositions, currentElevations, status,
  activeTracks: {ids: activeIds, count: activeCount, overflow: activeOverflow},
  drawInstanceCount                               // indirect draw: no readback to draw
}));
graph.add(new GPUTimeWindowFilter({
  timestamps: segmentStarts, endTimestamps: segmentEnds,
  window: windowParameters.importToGraph(graph),
  output: {ids: trailIds, count: trailCount, overflow: trailOverflow},
  fadeWeights, clipFractions, drawInstanceCount: trailDrawCount
}));
const compiled = graph.compile();                 // once

// every frame: only parameter writes
playheadParameters.write(getGPUTrajectoryPlayheadParameterValues({playhead, maxGap: 0}));
windowParameters.write(getGPUTimeWindowParameterValues({
  start: playhead - ${state.trailMinutes * 60}, end: playhead, startFadeDuration: ${Math.round(state.trailMinutes * 60 * state.tailFade)}
}));
compiled.encode(commandEncoder, {parameters: undefined});`,

  about: {
    what: 'A replay of 911 satellites over three hours on 7 October 2026. Positions are computed with SGP4 from the dated public CelesTrak TLE snapshot used to generate this archive, at 30-second steps and cut at the antimeridian. `GPUTrajectoryPlayhead` interpolates every track at the playhead with altitude as elevation, and `GPUTimeWindowFilter` picks trail segments behind it.',
    why: 'It explains what an orbital propagation snapshot can show: modelled motion, family composition and altitude. It does not calculate what an observer can see.',
    howToRead:
      'A disc is a satellite; its height above the map is its altitude (compressed unless you pick linear). Trails are where it has just been. Colors are the family or the altitude. These are modeled, not observed, positions: accuracy decays with the age of the orbital elements. Satellite conjunctions are deliberately not shown: `GPUTrajectoryEncounters` is two-dimensional and would invent near misses between satellites that are hundreds of kilometres apart in height.'
  },

  pipeline: [
    {id: 'propagate', label: 'Propagate', detail: 'Read SGP4 positions sampled every 30 seconds'},
    {id: 'playhead', label: 'Playhead', detail: 'Interpolate each antimeridian-cut track'},
    {id: 'window', label: 'Trail window', detail: 'Keep and fade recent path segments'},
    {id: 'draw', label: 'Draw', detail: 'Separate family marks from altitude diagram'}
  ],
  basemap: ground('space', {
    labels: 'none',
    graticule: true,
    referenceLines: ['equator', 'tropics']
  }),
  furniture: {
    title: {
      title: 'How do propagated satellites move?',
      subtitle: 'SGP4 model · 30-second samples · 7 Oct 2026',
      chips: ['propagated, not observed']
    },
    credit: joinCredits('CelesTrak GP elements; propagation with SGP4', CREDITS.naturalEarth),
    caveat: 'No observer, footprint, visibility or conjunction calculation is present.'
  },
  create: async ctx => (await import('./satellite-playback.compute')).createSatellitePlayback(ctx)
});
