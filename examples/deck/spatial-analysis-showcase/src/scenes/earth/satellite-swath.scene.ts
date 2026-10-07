// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {formatPlaybackTime, playbackOptions} from '../../engine/playback';
import {defineScene} from '../scene';
import type {SatelliteSwathOptions} from './satellite-swath.compute';

const FLAT_VIEW = {longitude: 0, latitude: 15, zoom: 1.3, pitch: 0, bearing: 0};
const elapsed = (seconds: number) => formatPlaybackTime.duration(seconds) || '0 min';

export default defineScene<SatelliteSwathOptions>({
  id: 'satellite-swath',
  title: 'How fast do Earth observation satellites cover the planet?',
  chapter: 'earth',
  order: 33,
  summary:
    'Buffer the ground tracks of 16 Earth observation satellites by their imaging swath on the GPU and find, for every degree of the planet, when a swath first reached it.',
  contributors: ['GPUOutlineGeometry'],
  datasets: [{id: 'celestrak-ground-tracks', role: 'SGP4 ground tracks (7 October 2026, 3 h)'}],
  initialView: FLAT_VIEW,

  options: [
    ...playbackOptions<SatelliteSwathOptions>({
      time: {
        min: 0,
        max: 10800,
        step: 30,
        default: 5400,
        label: 'Time since 00:00 UTC',
        format: elapsed,
        help: 'Swaths and the coverage map show everything up to this time.'
      },
      speed: {
        min: 30,
        max: 900,
        step: 30,
        default: 180,
        unit: 'x',
        help: 'Simulated seconds per real second. 180x plays the three hours in one minute.'
      },
      playing: false,
      loop: {default: true}
    }),
    {
      kind: 'select',
      id: 'satellites',
      label: 'Satellites',
      group: 'Swath',
      apply: 'param',
      default: 'all',
      help: 'Which Earth observation satellites count. The selection is a per-satellite mask: the swath layer culls the others and the coverage kernel skips their segments.',
      options: [
        {value: 'all', label: 'All 16 satellites'},
        {value: 'narrow', label: 'Narrow swaths (under 300 km)'},
        {value: 'wide', label: 'Wide swaths (over 1,000 km)'},
        {value: 'OLI', label: 'Landsat 8 and 9 (OLI, 185 km)'},
        {value: 'C-SAR IW', label: 'Sentinel-1A (radar, 250 km)'},
        {value: 'MSI', label: 'Sentinel-2A, 2B and 2C (MSI, 290 km)'},
        {value: 'OLCI', label: 'Sentinel-3A and 3B (OLCI, 1,270 km)'},
        {value: 'MODIS', label: 'Terra and Aqua (MODIS, 2,330 km)'},
        {value: 'TROPOMI', label: 'Sentinel-5P (TROPOMI, 2,600 km)'},
        {value: 'AVHRR', label: 'Metop-B and C (AVHRR, 2,900 km)'},
        {value: 'VIIRS', label: 'Suomi NPP, NOAA-20 and NOAA-21 (VIIRS, 3,000 km)'}
      ]
    },
    {
      kind: 'select',
      id: 'swathMode',
      label: 'Swath width',
      group: 'Swath',
      apply: 'param',
      default: 'nominal',
      help: "Nominal uses each instrument's published width. Custom gives every selected satellite the same width, to ask what a wider or narrower sensor would do. GPUOutlineGeometry has one distance per node, so nominal mode runs one node per instrument and custom mode one node for all.",
      options: [
        {value: 'nominal', label: 'Nominal, per instrument'},
        {value: 'custom', label: 'Custom, same for all'}
      ]
    },
    {
      kind: 'slider',
      id: 'customWidth',
      label: 'Custom swath width',
      group: 'Swath',
      apply: 'param',
      min: 50,
      max: 3200,
      step: 50,
      default: 500,
      unit: 'km',
      disabledWhen: state => state.swathMode !== 'custom',
      help: 'Full width of the swath centered on the ground track, applied to every selected satellite. The outline distance is half of it, rewritten each change without recompiling.'
    },
    {
      kind: 'toggle',
      id: 'showCoverage',
      label: 'Show first-covered time',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Colors every 1-degree cell by the time a swath first reached it; cells not yet reached at the playhead, or never reached, stay clear.'
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp',
      group: 'Display',
      apply: 'param',
      default: 'viridis',
      disabledWhen: state => !state.showCoverage,
      help: 'Color ramp for the first-covered time, from the start of the window (dark) to the end (bright).',
      options: [
        {value: 'viridis', label: 'Viridis'},
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'cividis', label: 'Cividis'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showSwaths',
      label: 'Show swath outlines',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Draws the triangles of GPUOutlineGeometry for every track vertex up to the playhead. Triangles overlap and are not merged, so darker purple means the swaths overlap.'
    },
    {
      kind: 'slider',
      id: 'swathOpacity',
      label: 'Swath opacity',
      group: 'Display',
      apply: 'param',
      min: 0.02,
      max: 0.5,
      step: 0.02,
      default: 0.12,
      disabledWhen: state => !state.showSwaths,
      help: 'Opacity of each swath triangle. Overlapping triangles add up, so a low value keeps single coverage faint and shows overlaps.'
    }
  ],

  story: [
    {
      id: 'swath',
      title: 'A track is a line; a swath is a footprint',
      body: 'An imaging satellite looks sideways to both sides of the ground track, so what it sees is a ribbon: the track buffered by half the swath width. `GPUOutlineGeometry` builds that ribbon on the sphere for all 16 satellites, one node per instrument because each has its own width.\n\nPress **Play** below and watch the ribbons grow. Landsat sees 185 km across, the VIIRS instruments on Suomi NPP and NOAA-20 and -21 see 3,000 km.',
      camera: {...FLAT_VIEW, transitionMs: 1200},
      options: {
        play: true,
        time: 600,
        satellites: 'all',
        swathMode: 'nominal',
        showCoverage: false,
        showSwaths: true,
        swathOpacity: 0.14
      },
      controls: ['play', 'time', 'satellites'],
      readouts: ['clock', 'satelliteCount']
    },
    {
      id: 'narrow',
      title: 'Narrow swaths need days, not hours',
      body: 'Choose Landsat in **Satellites** below and move **Time since 00:00 UTC** to the end: after three hours two Landsat satellites have imaged about 7 percent of the planet in thin ribbons that never touch. Their orbit shifts a little west every lap and only closes the gaps over a 16-day cycle (8 days with both), which this three-hour window cannot show.\n\nThe coverage map colors each cell by the time a swath first reached it; hover to read it.',
      camera: {...FLAT_VIEW, transitionMs: 1200},
      options: {
        play: false,
        time: 10800,
        satellites: 'OLI',
        swathMode: 'nominal',
        showCoverage: true,
        showSwaths: true,
        swathOpacity: 0.2
      },
      controls: ['satellites', 'time', 'showCoverage'],
      readouts: ['coveredNow', 'covered3h', 'coverageChart']
    },
    {
      id: 'wide',
      title: 'Wide swaths cover most of the world in hours',
      body: 'A 3,000 km swath is as wide as a continent. The wide-swath satellites (MODIS, VIIRS, AVHRR, OLCI and TROPOMI) reach about 88 percent of the planet in three hours, half of it in under 45 minutes. What is left lies in the gaps between the ribbons, which close slowly as the Earth turns under the orbits.\n\nThe curve in the chart rises quickly and then flattens: the last few percent take the longest.',
      camera: {...FLAT_VIEW, transitionMs: 1200},
      options: {
        play: false,
        time: 10800,
        satellites: 'wide',
        swathMode: 'nominal',
        showCoverage: true,
        showSwaths: false
      },
      controls: ['satellites', 'time', 'ramp'],
      readouts: ['t50', 't90', 'covered3h', 'coverageChart']
    },
    {
      id: 'custom',
      title: 'What if the swath were wider?',
      body: 'Give every satellite the same width to ask what a sensor would need. With all 16 satellites at 500 km, about 64 percent of the planet is covered after three hours; at 1,000 km about 84 percent; at 2,000 km about 98 percent and 90 percent of it within about two hours. Doubling the swath roughly halves the time to reach any given share, until the geometry of the orbits (not the width) is the limit.\n\nSwitch **Swath width** to custom and drag **Custom swath width** below; the coverage pass reruns on the GPU each time.',
      camera: {...FLAT_VIEW, transitionMs: 1200},
      options: {
        play: false,
        time: 10800,
        satellites: 'all',
        swathMode: 'custom',
        customWidth: 500,
        showCoverage: true,
        showSwaths: true,
        swathOpacity: 0.1
      },
      controls: ['swathMode', 'customWidth', 'satellites'],
      readouts: ['covered3h', 't50', 't90', 'coverageChart']
    },
    {
      id: 'limits',
      title: 'What this does not know',
      body: 'The widths are nominal values at the ground track; real swaths are narrower at the edges for some instruments and wider for others. Optical instruments that record reflected sunlight (Landsat, Sentinel-2) work only in daylight and cannot see through cloud, while radar images day and night; none of that is modeled. Ground-track density is not coverage, and coverage here is geometric only: a cell counts as covered the moment a swath touches it. Positions are SGP4 predictions from element sets of different ages.\n\nTry the satellites one instrument at a time and compare how many percent of the planet each reaches.',
      camera: {...FLAT_VIEW, transitionMs: 1200},
      options: {
        play: false,
        time: 10800,
        satellites: 'all',
        swathMode: 'nominal',
        showCoverage: true,
        showSwaths: true,
        swathOpacity: 0.1
      },
      controls: ['satellites', 'showSwaths'],
      readouts: ['covered3h', 'satelliteCount']
    }
  ],

  legends: state => [
    ...(state.showCoverage
      ? [
          {
            kind: 'ramp' as const,
            title: 'Time a swath first reached the cell',
            ramp: state.ramp,
            extent: [0, 10800] as const,
            unit: 'h since 00:00 UTC',
            format: (value: number) => (value / 3600).toFixed(1)
          }
        ]
      : []),
    ...(state.showSwaths
      ? [
          {
            kind: 'categories' as const,
            title: 'Swath outline',
            entries: [
              {
                color: [190, 130, 240, 120] as const,
                label: 'Ground track buffered by half the swath'
              }
            ],
            note: 'Overlaps add up and are not merged, so darker means more than one swath.'
          }
        ]
      : [])
  ],

  readouts: [
    {id: 'clock', label: 'Playhead'},
    {id: 'satelliteCount', label: 'Selection'},
    {
      id: 'coveredNow',
      label: 'Covered at the playhead',
      help: 'Share of the Earth between 85 S and 85 N (area weighted, 1-degree cells) inside at least one swath by the playhead.'
    },
    {
      id: 'covered3h',
      label: 'Covered after three hours',
      help: 'Share of the Earth between 85 S and 85 N inside at least one swath by the end of the window.'
    },
    {
      id: 't50',
      label: 'Time to 50% coverage',
      help: 'First 2.5-minute step at which the covered share reaches 50 percent, or that it is not reached in three hours.'
    },
    {id: 't90', label: 'Time to 90% coverage'},
    {id: 't99', label: 'Time to 99% coverage (near full)'},
    {
      id: 'coverageChart',
      label: 'Area covered over time',
      kind: 'chart',
      help: 'Percent of the Earth covered against time, with the playhead marked and guides at 50 and 90 percent.'
    }
  ],

  snippet: state => `import {
  GPUOutlineGeometry, getGPUOutlineGeometryParameterValues, getGPUOutlineGeometryVerticesPerInput
} from '@luma.gl/experimental/gpu-spatial-analysis';

// positions: float32x2 [longitude, latitude]; one node per instrument (one distance per node)
graph.add(new GPUOutlineGeometry({
  positions, geometryType: 'lines', pathOffsets: trackOffsets,
  coordinateSystem: 'spherical',          // distances in meters, local east/north scale per vertex
  joinSegments: 16,
  parameters: distance.importToGraph(graph),
  output: {positions: triangles}          // positions.length * ${'3 * (16 + 2)'} triangle-list vertices
}));
distance.write(getGPUOutlineGeometryParameterValues({
  distance: ${state.swathMode === 'custom' ? state.customWidth : 'swathKm'} * 1000 / 2    // half the swath, in meters
}));

// Coverage: a bespoke compute pass visits every 1-degree cell and every track segment,
// keeping the earliest time a segment passes within half the swath of the cell center.
// firstCover[cell] = min(t0 + f * (t1 - t0)) over segments with distance <= halfWidth`,

  about: {
    what: 'The ground tracks of 16 Earth observation satellites (Landsat, Sentinel, Terra, Aqua, Suomi NPP, NOAA-20 and -21, Metop) over three hours, buffered by half their nominal swath width with `GPUOutlineGeometry` in spherical mode. A bespoke compute pass then measures, for every 1-degree cell, the first time a swath touched it.',
    why: 'Revisit and time to coverage decide whether a sensor can answer a question in hours (weather, smoke, floods) or only in weeks (land cover). The swath width, not the number of satellites, usually dominates.',
    howToRead:
      "Purple ribbons are the buffered tracks; where they overlap they are darker. The colored cells show when each place was first covered, dark early and bright late; empty cells were not reached by the playhead. The chart and readouts give area-weighted coverage between 85 S and 85 N. Nominal swath widths, daylight and cloud, off-nadir pointing and the terrain are ignored. The outline contributor measures meters with each vertex's local east and north scale, which is not geodesic and degrades near the poles and across the antimeridian, so very wide swaths look distorted at high latitude on the Mercator map; the coverage pass uses the same approximation. Satellite conjunctions are deliberately not computed: `GPUTrajectoryEncounters` is two-dimensional."
  },

  create: async ctx => (await import('./satellite-swath.compute')).createSatelliteSwath(ctx)
});
