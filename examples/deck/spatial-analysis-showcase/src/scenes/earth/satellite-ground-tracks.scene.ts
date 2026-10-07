// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
import type {SatelliteGroundTracksOptions} from './satellite-ground-tracks.compute';
import {SATELLITE_GROUPS} from './satellite-tracks';

const FLAT_VIEW = {longitude: 0, latitude: 12, zoom: 1.3, pitch: 0, bearing: 0};
const latitudeLabel = (value: number) => `${Math.abs(value)}°${value < 0 ? 'S' : 'N'}`;

export default defineScene<SatelliteGroundTracksOptions>({
  id: 'satellite-ground-tracks',
  title: 'How often does a satellite pass over each latitude?',
  chapter: 'earth',
  order: 31,
  summary:
    'Sum the ground tracks of 911 satellites into a density map on the GPU and count every entry into 34 latitude bands, to see how orbit inclination decides who passes over where, and how often.',
  contributors: ['GPULineDensity', 'GPUZoneEvents'],
  datasets: [{id: 'celestrak-ground-tracks', role: 'SGP4 ground tracks (7 October 2026, 3 h)'}],
  initialView: FLAT_VIEW,

  options: [
    {
      kind: 'select',
      id: 'group',
      label: 'Satellite family',
      group: 'Satellites',
      apply: 'compile',
      default: 'all',
      help: 'The tracks the density and the pass counts are computed from. Each family compiles its own graph the first time you pick it (the tracks are static, so each graph runs once); the panel counts that as a rebuild.',
      options: [
        {value: 'all', label: 'All 911 satellites'},
        ...SATELLITE_GROUPS.map((label, index) => ({value: String(index), label}))
      ]
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp',
      group: 'Density map',
      apply: 'param',
      default: 'inferno',
      help: 'Color ramp of the density map.',
      options: [
        {value: 'inferno', label: 'Inferno'},
        {value: 'magma', label: 'Magma'},
        {value: 'viridis', label: 'Viridis'},
        {value: 'cividis', label: 'Cividis'}
      ]
    },
    {
      kind: 'select',
      id: 'scale',
      label: 'Value scale',
      group: 'Density map',
      apply: 'param',
      default: 'sqrt',
      help: 'Square root lifts the faint cells so low-density regions stay visible; linear shows the peaks at their true contrast.',
      options: [
        {value: 'sqrt', label: 'Square root'},
        {value: 'linear', label: 'Linear'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showTracks',
      label: 'Show ground tracks',
      group: 'Density map',
      apply: 'param',
      default: false,
      help: 'Draws the tracks the density was computed from as thin lines over the map.'
    },
    {
      kind: 'slider',
      id: 'trackOpacity',
      label: 'Track opacity',
      group: 'Density map',
      apply: 'param',
      min: 0.02,
      max: 0.6,
      step: 0.02,
      default: 0.12,
      disabledWhen: state => !state.showTracks,
      help: 'Opacity of the track lines. Hundreds of overlapping tracks need a low value to read as a glow.'
    },
    {
      kind: 'toggle',
      id: 'showBand',
      label: 'Highlight latitude band',
      group: 'Pass counts',
      apply: 'param',
      default: false,
      help: 'Shades the 5-degree band selected below on the map and marks it on the chart.'
    },
    {
      kind: 'slider',
      id: 'band',
      label: 'Latitude band',
      group: 'Pass counts',
      apply: 'param',
      min: -82.5,
      max: 82.5,
      step: 5,
      default: 52.5,
      format: latitudeLabel,
      help: 'Centre of the 5-degree band to read out. Click the map to pick the band under the pointer.'
    },
    {
      kind: 'toggle',
      id: 'perSatellite',
      label: 'Per satellite',
      group: 'Pass counts',
      apply: 'param',
      default: false,
      help: 'Divides the pass counts by the number of satellites in the family, so a dozen stations and 800 Starlinks can be compared as individuals.'
    }
  ],

  story: [
    {
      id: 'density',
      title: 'Where do satellites spend their time?',
      body: 'Each satellite draws a ground track: the point on the Earth directly below it. `GPULineDensity` clips every track to a 2-degree grid on the sphere, walks the cells it crosses and sums the length per cell, then divides by the exact spherical cell area.\n\nBright cells are crossed by more kilometres of track per hour; black cells are never overflown by any of these 911 satellites. Change **Color ramp** or **Value scale** below if the faint cells are hard to see.',
      camera: {...FLAT_VIEW, transitionMs: 1200},
      options: {group: 'all', showTracks: false, showBand: false, perSatellite: false},
      controls: ['ramp', 'scale'],
      readouts: ['trackLength', 'peak']
    },
    {
      id: 'belts',
      title: 'Inclination draws the belts',
      body: 'An orbit tilted by 53 degrees never takes its satellite north of 53 degrees: it turns around there, and a satellite that turns around lingers. Starlink satellites fly at 43, 53, 70 and 97 degrees, so the density piles up in bright belts at those latitudes and thins out toward the equator.\n\nSwitch **Show ground tracks** on below and lower **Track opacity** to see the sinusoids behind the glow.',
      camera: {...FLAT_VIEW, transitionMs: 1200},
      options: {group: '1', showTracks: true, trackOpacity: 0.1},
      controls: ['group', 'showTracks', 'trackOpacity'],
      readouts: ['peak', 'touched']
    },
    {
      id: 'passes',
      title: 'How often is something overhead at a latitude?',
      body: 'Density measures distance, not visits. `GPUZoneEvents` tests every track against 34 latitude-band polygons and records each time a satellite enters one; the chart shows entries per hour (an ascending and a descending crossing are two passes). The curve is nearly flat inside the inclination and falls away beyond it.\n\nPick a **Latitude band** below, or click the map, and read how long you wait for the next pass of any satellite in the family.',
      camera: {...FLAT_VIEW, transitionMs: 1200},
      options: {group: 'all', showTracks: false, showBand: true, band: 52.5, perSatellite: false},
      controls: ['band', 'showBand', 'group'],
      readouts: ['bandPasses', 'bandGap', 'passChart']
    },
    {
      id: 'polar',
      title: 'Equator versus the poles',
      body: 'Earth observation satellites fly near-polar, sun-synchronous orbits at about 98 degrees. One of them enters each band up to about 70 degrees as often as one Starlink satellite (roughly 1.2 times an hour) and still reaches the 80 degree band, which the 43 and 53 degree Starlink shells never do. Totals are very different because the family has 16 members against 800, so turn on **Per satellite** below to compare individuals.\n\nClick near the poles and then the equator and compare the revisit times.',
      camera: {...FLAT_VIEW, transitionMs: 1200},
      options: {
        group: '4',
        showBand: true,
        band: 82.5,
        perSatellite: true,
        showTracks: true,
        trackOpacity: 0.2
      },
      controls: ['group', 'perSatellite', 'band'],
      readouts: ['bandPasses', 'bandGap', 'passChart']
    },
    {
      id: 'limits',
      title: 'Density is not coverage',
      body: 'A ground track is a line, not a footprint. The cells are 2 degrees wide (about 220 km at the equator), the window is only three hours, and nothing here knows what any satellite can see: a sensor swath, a communication beam and a navigation signal all reach hundreds to thousands of kilometres from the track. The next scenes use the swath for Earth observation satellites.\n\nThe readouts show the capacities used and whether any overflowed.',
      camera: {...FLAT_VIEW, transitionMs: 1200},
      options: {group: 'all', showTracks: false, showBand: false, perSatellite: false},
      controls: ['group', 'scale'],
      readouts: ['events', 'tracks']
    }
  ],

  legends: state => [
    {
      kind: 'ramp' as const,
      id: 'density',
      title: 'Ground-track density',
      ramp: state.ramp,
      extent: 'gpu' as const,
      sqrtScale: state.scale === 'sqrt',
      unit: 'km of track per 1,000 km² per hour',
      format: (value: number) => (value >= 10 ? value.toFixed(0) : value.toFixed(1))
    }
  ],

  readouts: [
    {
      id: 'passChart',
      label: 'Passes per hour by latitude',
      kind: 'chart',
      help: 'Entries into each 5-degree latitude band per hour, counted by GPUZoneEvents over the three-hour window.'
    },
    {
      id: 'bandPasses',
      label: 'Passes in the selected band',
      help: 'Entries into the selected band per hour.'
    },
    {
      id: 'bandGap',
      label: 'Time between passes',
      layout: 'block',
      help: 'Mean time between two entries into the band: by any satellite of the family, and by one satellite of it.'
    },
    {
      id: 'trackLength',
      label: 'Ground track length',
      help: 'Sum of the length of every track in the family over three hours, from the density cells.'
    },
    {id: 'peak', label: 'Peak density', help: 'The densest cell, and its latitude.'},
    {
      id: 'touched',
      label: 'Cells touched',
      help: 'Share of the 15,300 cells (2 degrees, 85 S to 85 N) crossed by at least one track.'
    },
    {
      id: 'events',
      label: 'Band crossings',
      layout: 'block',
      help: 'Number of zone events, the candidate segment-edge pairs tested, and OVERFLOW when a capacity (events 262,144, candidates 524,288, 128 events per track) was exceeded.'
    },
    {id: 'tracks', label: 'Dataset'}
  ],

  snippet: state => `import {
  GPULineDensity, GPUZoneEvents, getGPULineDensityParameterValues
} from '@luma.gl/experimental/gpu-spatial-analysis';

// positions: float32x2 [longitude, latitude] degrees; tracks are cut at the antimeridian
graph.add(new GPULineDensity({
  positions, pathOffsets: trackOffsets,
  columns: 180, rows: 85, coordinateSystem: 'spherical',   // 2 degree cells, exact spherical areas
  parameters: gridParameters.importToGraph(graph),
  output: {lengths, densities, overflow}
}));
gridParameters.write(getGPULineDensityParameterValues({minX: -180, minY: -85, cellWidth: 2, cellHeight: 2}));

// 34 latitude bands as rectangles wider than the world: only the horizontal edges are ever crossed
graph.add(new GPUZoneEvents({
  positions, timestamps, trackOffsets, edgeStarts, edgeEnds, edgeZones,
  zoneCount: 34, candidateCapacity: 1 << 19, maxEventsPerTrack: 128,
  events: {output: {ids, count, overflow}, eventZones, eventTypes}
}));
// passes per band = events of type GPU_ZONE_EVENT_TYPE.enter, per band, divided by ${'3'} hours
// family: ${state.group === 'all' ? 'all tracks' : SATELLITE_GROUPS[Number(state.group)]}`,

  about: {
    what: 'The ground tracks of 911 satellites over three hours (SGP4 from current CelesTrak elements, 30 second steps, cut at the antimeridian), summed into a density map by `GPULineDensity` in spherical mode and counted per latitude band by `GPUZoneEvents`.',
    why: 'How often something passes over a latitude is the first question behind revisit time, ground-station planning and sky-watching, and it depends almost entirely on orbit inclination. The two contributors answer different halves: line density measures how much track there is per area, zone events count visits.',
    howToRead:
      'Bright cells have more kilometres of track per area. The chart counts entries into a latitude band, so it also counts a satellite that dips in and out near its maximum latitude. Ground-track density is not sensor coverage: it ignores swath width, field of view and beam size. The window is three hours, so slow orbits (GPS, 12 hours) show only a quarter of a lap; SGP4 accuracy decays with the age of the element sets. Conjunctions are not computed: `GPUTrajectoryEncounters` is two-dimensional and would invent near misses between satellites at different heights.'
  },

  create: async ctx =>
    (await import('./satellite-ground-tracks.compute')).createSatelliteGroundTracks(ctx)
});
