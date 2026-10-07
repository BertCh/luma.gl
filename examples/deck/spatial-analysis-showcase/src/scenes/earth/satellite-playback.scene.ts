// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {formatPlaybackTime, playbackOptions} from '../../engine/playback';
import {defineScene} from '../scene';
import type {SatellitePlaybackOptions} from './satellite-playback.compute';
import {getGroupLegendEntries, SATELLITE_GROUPS} from './satellite-tracks';

const WORLD_VIEW = {longitude: 10, latitude: 15, zoom: 1.5, pitch: 45, bearing: 0};

const elapsed = (seconds: number) => formatPlaybackTime.duration(seconds) || '0 min';

export default defineScene<SatellitePlaybackOptions>({
  id: 'satellite-playback',
  title: 'Who is overhead right now?',
  chapter: 'earth',
  order: 30,
  summary:
    'Replay three hours of 911 satellites, from the International Space Station to GPS, propagated from current CelesTrak elements and lifted off the map by their altitude with a GPU playhead.',
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
        {value: 'all', label: 'All 911 satellites'},
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
      help: 'Family, or the altitude of each satellite through a ramp that is logarithmic from 200 km to 25,600 km.',
      options: [
        {value: 'group', label: 'Family'},
        {value: 'altitude', label: 'Altitude'}
      ]
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Altitude ramp',
      group: 'Satellites',
      apply: 'param',
      default: 'viridis',
      disabledWhen: state => state.colorBy !== 'altitude',
      help: 'Color ramp for altitude. Used only when Color by is Altitude.',
      options: [
        {value: 'viridis', label: 'Viridis'},
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'cividis', label: 'Cividis'}
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
      id: 'overhead',
      title: 'Who is overhead right now?',
      body: 'Every disc is a satellite at the playhead: a space station, a GPS vehicle, a weather or Earth observation satellite, or one of 800 Starlink satellites sampled from the roughly 10,600 in orbit. Each position is computed from its public orbital elements with the SGP4 model every 30 seconds; `GPUTrajectoryPlayhead` interpolates all of them at the clock, once per frame.\n\nPress **Play** below, or drag **Time since 00:00 UTC** to any moment in the three hours. The slider follows the clock; **Playback speed** sets how fast simulated time runs.',
      camera: {...WORLD_VIEW, transitionMs: 1400},
      options: {play: true, time: 1800, group: 'all', colorBy: 'group', showStems: false},
      controls: ['play', 'time', 'speed'],
      readouts: ['clock', 'active']
    },
    {
      id: 'height',
      title: 'Height is half the story',
      body: 'The playhead also interpolates an elevation for every satellite, and the layer lifts each disc by it. Low Earth orbit (stations, Starlink, weather and Earth observation satellites) sits between 300 and 900 km; GPS is at about 20,200 km. The map would be useless at true proportions, so **Altitude scale** defaults to a logarithm: heights are ordered, not proportional.\n\nSwitch it to Linear to see how thin low orbit really is, and turn on **Drop lines to the ground** to read each satellite against the point below it.',
      camera: {...WORLD_VIEW, pitch: 58, zoom: 1.8, transitionMs: 1400},
      options: {showStems: true, colorBy: 'altitude', showTrails: false},
      controls: ['altitudeDisplay', 'altitudeScale', 'showStems', 'colorBy'],
      readouts: ['highest', 'lowest', 'altitudeChart']
    },
    {
      id: 'trails',
      title: 'Orbits leave streaks',
      body: 'Each satellite leaves a trail: the segments of its track inside the time window `GPUTimeWindowFilter` keeps. A low orbit is about 93 minutes, so a 45-minute trail is half a lap; the fade makes the newest part brightest. Lines that bend toward the poles come from inclined orbits and are the subject of the next scenes.\n\nLengthen **Trail length** until the streaks of the polar-orbiting Earth observation satellites cross over themselves, then vary **Tail fade**.',
      camera: {...WORLD_VIEW, pitch: 30, zoom: 1.4, transitionMs: 1400},
      options: {showTrails: true, trailMinutes: 45, colorBy: 'group', showStems: false},
      controls: ['showTrails', 'trailMinutes', 'tailFade'],
      readouts: ['trailSegments']
    },
    {
      id: 'families',
      title: 'Pick one family',
      body: 'Choose **Show family** below to isolate Starlink, GPS, the stations, weather or Earth observation satellites. The sampled Starlink satellites form a dense lattice of near-identical orbits; the 32 GPS satellites loiter high above and move slowly, only a quarter of a lap in three hours.\n\nTry **Marker size** with the Starlink family, then click any satellite to read its orbit.',
      camera: {...WORLD_VIEW, zoom: 1.6, transitionMs: 1400},
      options: {group: '1', colorBy: 'group', trailMinutes: 20, showTrails: true},
      controls: ['group', 'markerSize', 'colorBy'],
      readouts: ['active', 'selected']
    },
    {
      id: 'limits',
      title: 'A model, not an observation',
      body: 'These are predictions. SGP4 is only accurate near the epoch of the orbital elements: roughly a kilometre at epoch and growing by a few kilometres per day for low orbits, because drag is unpredictable. The elements here are up to a few weeks old (the oldest is in the readouts), but most are under a day. Debris, military and most commercial satellites are not in the data, and Starlink is a sample.\n\nSet **Show family** back to all and read the altitude chart: most of what is overhead is within 1,000 km of the ground.',
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
          ramp: state.ramp,
          extent: [200, 25600] as const,
          scale: 'log' as const,
          unit: 'km',
          format: (value: number) => `${Math.round(value).toLocaleString('en-US')}`
        }
      : {
          kind: 'categories' as const,
          title: 'Family',
          entries: getGroupLegendEntries(),
          note: 'Starlink is a random sample of 800; GPS is the operational constellation.'
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
    what: 'A replay of 911 satellites over three hours on 7 October 2026. Positions are computed with the SGP4 model from the latest public CelesTrak element sets at 30 second steps and cut at the antimeridian. `GPUTrajectoryPlayhead` interpolates every track at the playhead with the altitude as its elevation, and `GPUTimeWindowFilter` picks the trail segments behind it.',
    why: 'It answers "what is overhead and how high" for the families that matter to an analyst: stations, navigation, weather, Earth observation and a mega-constellation. It also shows planar trajectory contributors working on a global dataset once the antimeridian is handled in the data.',
    howToRead:
      'A disc is a satellite; its height above the map is its altitude (compressed unless you pick linear). Trails are where it has just been. Colors are the family or the altitude. These are modeled, not observed, positions: accuracy decays with the age of the orbital elements. Satellite conjunctions are deliberately not shown: `GPUTrajectoryEncounters` is two-dimensional and would invent near misses between satellites that are hundreds of kilometres apart in height.'
  },

  create: async ctx => (await import('./satellite-playback.compute')).createSatellitePlayback(ctx)
});
