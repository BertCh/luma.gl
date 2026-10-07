// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CREDITS, joinCredits} from '../../cartography/credits';
import {ground} from '../../cartography/grounds';
import {formatPlaybackTime, playbackOptions} from '../../engine/playback';
import {defineScene} from '../scene';
import type {SatelliteSwathOptions} from './satellite-swath.compute';

const FLAT_VIEW = {longitude: 0, latitude: 15, zoom: 1.3, pitch: 0, bearing: 0};
const elapsed = (seconds: number) => formatPlaybackTime.duration(seconds) || '0 min';

export default defineScene<SatelliteSwathOptions>({
  id: 'satellite-swath',
  title: 'How quickly does a modelled swath reach Earth?',
  chapter: 'earth',
  order: 11,
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
      id: 'compareGeometry',
      label: 'Compare nominal outline',
      group: 'Swath',
      apply: 'param',
      default: false,
      disabledWhen: state => state.swathMode !== 'custom',
      help: 'Draws the selected instruments’ nominal blue outlines under the hypothetical violet custom-width ribbons. The elapsed-time cells always describe the active custom scenario.'
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
      id: 'footprint',
      title: 'A line becomes a nominal footprint',
      headline: 'A buffer is a modelled swath',
      textAlternative:
        'A selected satellite ground track has a violet nominal swath ribbon and local width label.',
      optionsMode: 'fresh',
      body: 'One loaded Earth-observation instrument is the subject: its centreline is buffered by half its nominal width to make a violet ribbon. The map carries a live cross-track bracket; if source width metadata is absent or zero it says unknown rather than inventing a default.\n\nPress **Play** to watch the modelled ribbon grow.',
      camera: {...FLAT_VIEW, transitionMs: 1200},
      options: {
        play: true,
        time: 600,
        satellites: 'OLI',
        swathMode: 'nominal',
        showCoverage: false,
        showSwaths: true,
        swathOpacity: 0.14
      },
      controls: ['play', 'time', 'satellites'],
      readouts: ['clock', 'satelliteCount', 'selectedWidth']
    },
    {
      id: 'narrow',
      title: 'Each cell records its first reach',
      headline: 'First reach is an elapsed-time field',
      textAlternative: 'Elapsed-time coverage cells retain unreached places as explicit no-data.',
      optionsMode: 'fresh',
      body: 'The coverage map colors each cell by the time a swath first reached it; unreached cells remain hatched no-data. The live curve and readouts own the area-weighted findings for this three-hour model window.\n\nHover a cell to read its first-reach UTC.',
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
      title: 'Narrow swaths leave broad gaps',
      headline: 'Width controls first reach',
      textAlternative: 'A narrow nominal swath is paired with area-weighted reach over time.',
      optionsMode: 'fresh',
      body: 'Wide nominal ribbons reach cells sooner than narrow ribbons, but the live area-weighted curve—not prose—reports the share and any threshold crossing. Gaps between adjacent ground tracks remain visible as hatched no-data.\n\nThe curve rises quickly and then flattens as remaining gaps become harder to reach.',
      camera: {...FLAT_VIEW, transitionMs: 1200},
      options: {
        play: false,
        time: 10800,
        satellites: 'wide',
        swathMode: 'nominal',
        showCoverage: true,
        showSwaths: false
      },
      controls: ['satellites', 'time'],
      readouts: ['t50', 't90', 'covered3h', 'coverageChart']
    },
    {
      id: 'compare-widths',
      title: 'Width changes the answer',
      headline: 'A custom width is hypothetical',
      textAlternative:
        'Synchronized nominal and hypothetical custom ribbons are distinct; the active first-reach surface belongs only to the custom scenario.',
      optionsMode: 'fresh',
      body: 'The blue outlines are synchronized nominal geometry; the violet ribbon is the explicitly hypothetical custom width. One first-reach surface is drawn: it belongs to the active violet scenario, not both geometries.\n\nSwitch width and inspect the live curve and threshold readouts; the coverage pass reruns on the GPU each time.',
      camera: {...FLAT_VIEW, transitionMs: 1200},
      options: {
        play: false,
        time: 10800,
        satellites: 'all',
        swathMode: 'custom',
        customWidth: 500,
        compareGeometry: true,
        showCoverage: true,
        showSwaths: true,
        swathOpacity: 0.1
      },
      controls: ['swathMode', 'customWidth', 'compareGeometry', 'satellites'],
      readouts: ['covered3h', 't50', 't90', 'coverageChart']
    },
    {
      id: 'limits',
      title: 'Reach is not usable observation',
      headline: 'A buffered line omits sensor geometry',
      textAlternative:
        'A reach surface remains visible with daylight, cloud, terrain and projection limits stated.',
      optionsMode: 'fresh',
      body: 'The widths are nominal values at the ground track; real swaths vary with viewing geometry. Daylight, cloud, terrain, pointing and edge effects are omitted. A cell is geometrically reached when this planar buffered-track model touches it—not when it yields a usable observation.\n\nTry instruments one at a time and use the live readouts to compare reach.',
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
            kind: 'categories' as const,
            title: 'First reach (elapsed time)',
            entries: [
              {color: [46, 168, 184, 210] as const, label: '0–30 min'},
              {color: [87, 194, 143, 210] as const, label: '30–60 min'},
              {color: [242, 179, 56, 210] as const, label: '60–120 min'},
              {color: [184, 87, 179, 210] as const, label: '120–180 min'},
              {
                color: [160, 175, 200, 180] as const,
                label: 'Not reached by this playhead / window (hatched)'
              }
            ],
            note: 'Classes are elapsed time from 00:00 UTC; cells cover only 85° S to 85° N.'
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
    {id: 'selectedWidth', label: 'Selected nominal width'},
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

  pipeline: [
    {id: 'track', label: 'Ground track', detail: 'Select propagated EO trajectories'},
    {
      id: 'buffer',
      label: 'Nominal swath',
      detail: 'Buffer the line using supplied or hypothetical width'
    },
    {id: 'reach', label: 'First reach', detail: 'Write earliest modelled reach to every grid cell'},
    {id: 'area', label: 'Area curve', detail: 'Weight reached cells by analysed area'}
  ],
  basemap: ground('space', {
    labels: 'none',
    graticule: true,
    referenceLines: ['equator', 'tropics']
  }),
  furniture: {
    title: {
      title: 'How quickly does a modelled swath reach Earth?',
      subtitle: 'Nominal swath · elapsed first reach · SGP4 model'
    },
    credit: joinCredits('CelesTrak GP elements; propagation with SGP4', CREDITS.naturalEarth),
    caveat:
      'Reach omits viewing geometry, pointing, daylight, cloud, terrain and polar Mercator distortion.'
  },

  create: async ctx => (await import('./satellite-swath.compute')).createSatelliteSwath(ctx)
});
