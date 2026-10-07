// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CREDITS, joinCredits} from '../../cartography/credits';
import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {SatelliteGroundTracksOptions} from './satellite-ground-tracks.compute';
import {SATELLITE_GROUPS} from './satellite-tracks';

const FLAT_VIEW = {longitude: 0, latitude: 12, zoom: 1.3, pitch: 0, bearing: 0};
const latitudeLabel = (value: number) => `${Math.abs(value)}°${value < 0 ? 'S' : 'N'}`;

export default defineScene<SatelliteGroundTracksOptions>({
  id: 'satellite-ground-tracks',
  title: 'Where do ground tracks accumulate?',
  chapter: 'earth',
  order: 10,
  summary:
    'Sum the ground tracks of 911 satellites into a density map on the GPU and count every entry into 34 latitude bands, to see how orbit inclination decides who passes over where, and how often.',
  contributors: ['GPULineDensity', 'GPUZoneEvents'],
  datasets: [{id: 'celestrak-ground-tracks', role: 'SGP4 ground tracks (7 October 2026, 3 h)'}],
  initialView: FLAT_VIEW,

  options: [
    {
      kind: 'select',
      id: 'cellSize',
      label: 'Cell size (MAUP)',
      group: 'Density map',
      apply: 'compile',
      default: '2',
      help: 'Recomputes the same spherical-area and satellite-hour denominator at 1°, 2° or 5°. Changing the aggregation unit can change occupied share and peak rank.',
      options: [
        {value: '1', label: '1° cells'},
        {value: '2', label: '2° cells'},
        {value: '5', label: '5° cells'}
      ]
    },
    {
      kind: 'select',
      id: 'densityMode',
      label: 'Density measure',
      group: 'Density map',
      apply: 'param',
      default: 'normalised',
      help: 'Raw is track kilometres in a cell; normalised divides by spherical cell area and satellite-hours.',
      options: [
        {value: 'normalised', label: 'Per 10,000 km² per satellite-hour'},
        {value: 'raw', label: 'Raw track kilometres'}
      ]
    },
    {
      kind: 'toggle',
      id: 'focusOrbit',
      label: 'Show one selected orbit',
      group: 'Density map',
      apply: 'param',
      default: false,
      help: 'Uses one loaded satellite from the selected family; it is a ground-track subject, not a density surface.'
    },
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
      label: 'Denominator: per satellite-hour',
      group: 'Pass counts',
      apply: 'param',
      default: false,
      help: 'Divides the pass counts by the number of satellites in the family, so a dozen stations and 800 Starlinks can be compared as individuals.'
    }
  ],

  story: [
    {
      id: 'one-orbit',
      title: 'Inclination bounds one ground-track belt',
      headline: 'Tilt sets the latitude limit',
      textAlternative:
        'One selected ground track crosses a sparse graticule and turns at its inclination limit.',
      optionsMode: 'fresh',
      body: 'One loaded satellite draws the bright subject track: the point on Earth directly below it. Its inclination bounds the turning latitudes; ascending and descending legs cross the equator in opposite directions. No density surface is drawn in this step.\n\nThe readout identifies its family, inclination, altitude and period from the dated TLE snapshot.',
      camera: {...FLAT_VIEW, transitionMs: 1200},
      options: {
        group: 'all',
        focusOrbit: true,
        showTracks: true,
        showBand: false,
        perSatellite: false
      },
      controls: ['group'],
      readouts: ['selectedOrbit']
    },
    {
      id: 'belts',
      title: 'Families occupy different latitude belts',
      headline: 'Ground tracks inherit orbital tilt',
      textAlternative:
        'Satellite-family tracks trace different latitude belts above a sparse world grid.',
      optionsMode: 'fresh',
      body: 'The selected family’s paths make its inclination envelope visible. The linked latitude chart is derived from those same tracks, so its peak latitude and touched cells remain live rather than typed into the story.\n\nTracks are context here; the belt summary is the subject.',
      camera: {...FLAT_VIEW, transitionMs: 1200},
      options: {group: '1', showTracks: true, trackOpacity: 0.1},
      controls: ['group', 'showTracks', 'trackOpacity'],
      readouts: ['peak', 'touched', 'beltChart']
    },
    {
      id: 'normalise',
      title: 'Equal degrees are not equal areas',
      headline: 'Normalise track length by cell area',
      textAlternative:
        'A global density map explains why equal-degree cells change area with latitude.',
      optionsMode: 'fresh',
      body: 'Equal-degree cells shrink toward the poles. Toggle raw track-kilometres against the normalised measure: the same segment is divided by its exact spherical cell area and the selected satellite-hours. Reference-cell readouts make the equator/high-latitude area contrast explicit.\n\nThe default is normalised because density measures length per area, not visits.',
      camera: {...FLAT_VIEW, transitionMs: 1200},
      options: {group: 'all', showTracks: false, showBand: true, band: 52.5, perSatellite: false},
      controls: ['densityMode', 'group'],
      readouts: ['trackLength', 'rawPeak', 'peak', 'cellAreaRatio']
    },
    {
      id: 'passes',
      title: 'Track length is not a pass count',
      headline: 'A pass is a latitude-band entry',
      textAlternative:
        'A highlighted latitude strip and chart count modelled band entries rather than visibility.',
      optionsMode: 'fresh',
      body: '`GPUZoneEvents` records a band **entry**, not a point pass or revisit at a point. Its live chart distinguishes all selected records from the per-satellite-hour denominator.\n\nClick a latitude band to inspect its entries and mean gap; a sensor footprint is not part of this calculation.',
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
      id: 'resolution',
      title: 'The pattern changes with cell size',
      headline: 'Density is not coverage or revisit',
      textAlternative: 'A global density surface retains its area and sample caveats.',
      optionsMode: 'fresh',
      body: 'This is the modifiable areal unit problem: 1°, 2° and 5° cells aggregate identical tracks with identical spherical-area and satellite-hour normalisation, yet occupied share and the peak cell can change. The 85° cap is intentional: polar cells are excluded rather than presented as equal Mercator area.\n\nA ground track is a line, not a footprint or coverage. The next scene turns a line into a nominal sensor ribbon.',
      camera: {...FLAT_VIEW, transitionMs: 1200},
      options: {group: 'all', showTracks: false, showBand: false, perSatellite: false},
      controls: ['cellSize', 'group', 'perSatellite'],
      readouts: ['occupiedShare', 'peakRank', 'events', 'tracks']
    }
  ],

  legends: () => [
    {
      kind: 'categories' as const,
      id: 'density',
      title: 'Ground-track density',
      entries: [
        {color: [204, 232, 245, 225] as const, label: 'Class 1 · lowest sixth'},
        {color: [155, 204, 232, 225] as const, label: 'Class 2'},
        {color: [115, 168, 214, 225] as const, label: 'Class 3'},
        {color: [110, 133, 194, 225] as const, label: 'Class 4'},
        {color: [122, 94, 173, 225] as const, label: 'Class 5'},
        {color: [87, 51, 133, 225] as const, label: 'Class 6 · highest sixth'}
      ],
      note: 'Equal intervals between zero and this family’s observed maximum; measured values remain in the tooltip.'
    }
  ],

  readouts: [
    {id: 'beltChart', label: 'Family latitude summary', kind: 'chart'},
    {id: 'selectedOrbit', label: 'Selected orbit', layout: 'block'},
    {id: 'rawPeak', label: 'Raw track-km peak'},
    {id: 'cellAreaRatio', label: '1° cell area: equator / 80°'},
    {id: 'occupiedShare', label: 'Occupied share'},
    {id: 'peakRank', label: 'Peak-cell rank change'},
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
    what: 'The ground tracks of loaded satellites over three hours (SGP4 from the dated public CelesTrak TLE snapshot used to generate the 7 October 2026 archive, 30 second steps, cut at the antimeridian), summed into a density map by `GPULineDensity` in spherical mode and counted per latitude band by `GPUZoneEvents`.',
    why: 'How often something passes over a latitude is the first question behind revisit time, ground-station planning and sky-watching, and it depends almost entirely on orbit inclination. The two contributors answer different halves: line density measures how much track there is per area, zone events count visits.',
    howToRead:
      'Bright cells have more kilometres of track per area. The chart counts entries into a latitude band, so it also counts a satellite that dips in and out near its maximum latitude. Ground-track density is not sensor coverage, footprint coverage or visibility at a point. The three-hour SGP4 model window, deliberately unbalanced family sample and Mercator polar cap constrain every pattern.'
  },

  pipeline: [
    {id: 'tracks', label: 'Ground tracks', detail: 'Use antimeridian-cut propagated trajectories'},
    {id: 'density', label: 'Line density', detail: 'Sum kilometres by global cell'},
    {id: 'area', label: 'Normalise', detail: 'Divide by spherical cell area and satellite-hours'},
    {
      id: 'passes',
      label: 'Band entries',
      detail: 'Count latitude-zone entries separately from length'
    }
  ],
  basemap: ground('space', {
    labels: 'none',
    graticule: true,
    referenceLines: ['equator', 'tropics']
  }),
  furniture: {
    title: {
      title: 'Where do ground tracks accumulate?',
      subtitle: 'Track length / area · modelled three-hour window · 7 Oct 2026'
    },
    credit: joinCredits('CelesTrak GP elements; propagation with SGP4', CREDITS.naturalEarth),
    caveat:
      'Web Mercator exaggerates high latitudes; density is not sensor coverage or a point pass.'
  },

  create: async ctx =>
    (await import('./satellite-ground-tracks.compute')).createSatelliteGroundTracks(ctx)
});
