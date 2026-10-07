// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
import {storyFromMarkdown} from '../story-markdown';
import narrative from './wildfire-terrain.md?raw';
import type {WildfireTerrainOptions} from './wildfire-terrain.compute';

export default defineScene<WildfireTerrainOptions>({
  id: 'wildfire-terrain',
  title: 'Do wildfires burn steeper ground?',
  chapter: 'raster',
  order: 5,
  summary:
    'Fire perimeters rasterized onto a California DEM, a distance-field ring of unburned land around each, Horn slope and aspect, and zonal statistics comparing inside against ring: slope, elevation, steepness and a facing-direction chart for 24 fires.',
  contributors: [
    'GPUPolygonRasterization',
    'GPUDistanceField',
    'GPUTerrainDerivatives',
    'GPURasterZonalStatistics'
  ],
  datasets: [
    {id: 'poopdeck-wildfires', role: 'final fire perimeters'},
    {id: 'wildfire-california-dem', role: 'Terrarium elevation grid, 239 m'}
  ],
  initialView: {longitude: -121.7, latitude: 38.7, zoom: 6},

  options: [
    {
      kind: 'select',
      id: 'display',
      label: 'Show',
      group: 'Display',
      apply: 'param',
      default: 'zones',
      help: 'Which GPU raster is painted over the hillshade. All of them are already computed; switching is only a different buffer.',
      options: [
        {
          value: 'zones',
          label: 'Inside and ring',
          help: 'Red: cells inside a perimeter. Blue: the buffer ring around each fire.'
        },
        {
          value: 'boundary',
          label: 'Perimeter cells',
          help: 'The rasterizer boundary flags: cells touched by a polygon edge.'
        },
        {value: 'slope', label: 'Slope', help: 'Horn slope in degrees, 0 to 40.'},
        {
          value: 'southness',
          label: 'Southness',
          help: 'Minus the cosine of the aspect: +1 faces south, -1 faces north. Cells flatter than Flat below are blank.'
        },
        {value: 'hillshade', label: 'Hillshade only'}
      ]
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Slope ramp',
      group: 'Display',
      apply: 'param',
      default: 'magma',
      disabledWhen: state => state.display !== 'slope',
      help: 'Color ramp of the slope map. The legend uses the same table.',
      options: [
        {value: 'magma', label: 'Magma'},
        {value: 'viridis', label: 'Viridis'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'cividis', label: 'Cividis (color-blind optimised)'}
      ]
    },
    {
      kind: 'slider',
      id: 'overlayOpacity',
      label: 'Overlay opacity',
      group: 'Display',
      apply: 'param',
      min: 0.2,
      max: 1,
      step: 0.05,
      default: 0.7,
      help: 'Opacity of the painted raster over the hillshade.'
    },
    {
      kind: 'toggle',
      id: 'showHillshade',
      label: 'Hillshade underlay',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'The GPUTerrainDerivatives hillshade under the overlay.'
    },
    {
      kind: 'toggle',
      id: 'showOutlines',
      label: 'Perimeter outlines',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'The vector rings of the fires on the grid.'
    },
    {
      kind: 'slider',
      id: 'ringOuter',
      label: 'Ring outer distance',
      group: 'Comparison ring (GPUDistanceField)',
      apply: 'param',
      min: 1,
      max: 30,
      step: 0.5,
      default: 5,
      unit: 'km',
      help: 'Unburned cells up to this ground distance from a perimeter belong to that fire ring. The distance field is computed once; this only rewrites a parameter and re-runs the zonal statistics.'
    },
    {
      kind: 'slider',
      id: 'ringInner',
      label: 'Ring inner distance',
      group: 'Comparison ring (GPUDistanceField)',
      apply: 'param',
      min: 0,
      max: 5,
      step: 0.25,
      default: 0,
      unit: 'km',
      help: 'Leaves a gap between the perimeter and the ring, so the ring starts further out. Useful because perimeters are generalised and the edge is rarely a sharp change.'
    },
    {
      kind: 'select',
      id: 'distanceMode',
      label: 'Distance algorithm',
      group: 'Comparison ring (GPUDistanceField)',
      apply: 'compile',
      default: 'exact',
      help: 'exact is the separable lower-envelope transform (exact nearest burned cell); jump-flood is a cheaper approximation with a few correction passes. Rebuilds the raster graph.',
      options: [
        {value: 'exact', label: 'Exact (Felzenszwalb-Huttenlocher)'},
        {value: 'jump-flood', label: 'Jump flooding (approximate)'}
      ]
    },
    {
      kind: 'select',
      id: 'refinement',
      label: 'Jump-flood refinement',
      group: 'Comparison ring (GPUDistanceField)',
      apply: 'compile',
      default: '1',
      disabledWhen: state => state.distanceMode !== 'jump-flood',
      help: 'Extra passes after the step-1 pass: 0 is plain JFA, 1 is JFA+1, 2 is JFA+2. More passes fix more of the approximation errors. Rebuilds the raster graph.',
      options: [
        {value: '0', label: '0 (plain)'},
        {value: '1', label: '1 (JFA+1)'},
        {value: '2', label: '2 (JFA+2)'}
      ]
    },
    {
      kind: 'slider',
      id: 'flatSlope',
      label: 'Flat below',
      group: 'Terrain (GPUTerrainDerivatives)',
      apply: 'param',
      min: 0,
      max: 10,
      step: 0.5,
      default: 2,
      unit: 'deg',
      help: 'Cells flatter than this have no meaningful aspect: they are left out of the facing-direction counts and of southness.'
    },
    {
      kind: 'slider',
      id: 'steepSlope',
      label: 'Steep slope from',
      group: 'Terrain (GPUTerrainDerivatives)',
      apply: 'param',
      min: 5,
      max: 40,
      step: 1,
      default: 15,
      unit: 'deg',
      help: 'A cell counts as steep at or above this slope. At 239 m cells the steepest slopes are smoothed, so 15 degrees is already steep.'
    },
    {
      kind: 'slider',
      id: 'zFactor',
      label: 'Vertical exaggeration',
      group: 'Terrain (GPUTerrainDerivatives)',
      apply: 'param',
      min: 0.5,
      max: 5,
      step: 0.25,
      default: 1,
      help: 'Multiplies elevation differences in the derivatives. Leave at 1 for analysis; larger values steepen every slope and rerun the statistics.'
    },
    {
      kind: 'select',
      id: 'borderMode',
      label: 'Border mode',
      group: 'Terrain (GPUTerrainDerivatives)',
      apply: 'compile',
      default: 'clamp',
      help: 'What the 3x3 window sees beyond the grid. clamp repeats the edge, reflect mirrors it, constant uses zero, nodata marks those cells invalid. Rebuilds the derivatives graph.',
      options: [
        {value: 'clamp', label: 'Clamp'},
        {value: 'reflect', label: 'Reflect'},
        {value: 'constant', label: 'Constant'},
        {value: 'nodata', label: 'No data'}
      ]
    },
    {
      kind: 'slider',
      id: 'sunAzimuth',
      label: 'Sun azimuth',
      group: 'Terrain (GPUTerrainDerivatives)',
      apply: 'param',
      min: 0,
      max: 360,
      step: 5,
      default: 315,
      unit: 'deg',
      help: 'Direction the light comes from, clockwise from north. Only the hillshade changes.'
    },
    {
      kind: 'slider',
      id: 'sunAltitude',
      label: 'Sun altitude',
      group: 'Terrain (GPUTerrainDerivatives)',
      apply: 'param',
      min: 5,
      max: 80,
      step: 5,
      default: 40,
      unit: 'deg',
      help: 'Height of the light above the horizon. Only the hillshade changes.'
    },
    {
      kind: 'select',
      id: 'year',
      label: 'Perimeter year',
      group: 'Fires compared',
      apply: 'param',
      default: 'all',
      help: 'Compares only the fires of one year. Disabled fires leave the zones, charts and readouts.',
      options: [
        {value: 'all', label: 'All years'},
        {value: '2020', label: '2020'},
        {value: '2021', label: '2021'},
        {value: '2022', label: '2022'},
        {value: '2023', label: '2023'}
      ]
    },
    {
      kind: 'slider',
      id: 'minAcres',
      label: 'Smallest fire',
      group: 'Fires compared',
      apply: 'param',
      min: 1000,
      max: 100000,
      step: 1000,
      default: 1000,
      unit: 'acres',
      help: 'Compares only fires at least this large (NIFC acres).'
    },
    {
      kind: 'select',
      id: 'sumOrder',
      label: 'Zonal sum order',
      group: 'Zonal statistics (GPURasterZonalStatistics)',
      apply: 'compile',
      default: 'sorted',
      help: 'sorted sorts the cells by zone and reduces each zone with a fixed tree: reproducible, and fast with few contiguous zones. atomic accumulates with float atomics: up to 60 times slower here, because every lane hits the same few zones. Rebuilds the statistics graph.',
      options: [
        {value: 'sorted', label: 'Sorted (reproducible)'},
        {value: 'atomic', label: 'Float atomics'}
      ]
    }
  ],

  story: storyFromMarkdown<WildfireTerrainOptions>(narrative, {
    'burn-scars': {
      camera: {longitude: -121.7, latitude: 38.7, zoom: 6, transitionMs: 1400},
      options: {display: 'zones'},
      controls: ['display', 'year', 'minAcres'],
      readouts: ['comparable', 'cellsInside', 'cellsRing']
    },
    rasterize: {
      camera: {longitude: -120.85, latitude: 40.3, zoom: 8.2, transitionMs: 1800},
      options: {display: 'boundary', overlayOpacity: 0.9},
      controls: ['display', 'overlayOpacity'],
      readouts: ['rasterizer', 'cellsInside'],
      callout: {coordinate: [-121.09, 40.32], text: 'Dixie Fire'}
    },
    terrain: {
      camera: {longitude: -122.1, latitude: 38.2, zoom: 7.4, transitionMs: 1800},
      options: {display: 'slope'},
      controls: ['display', 'sunAzimuth', 'zFactor', 'borderMode'],
      readouts: []
    },
    ring: {
      camera: {longitude: -120.85, latitude: 40.3, zoom: 8.2, transitionMs: 1800},
      options: {display: 'zones', ringOuter: 5},
      controls: ['ringOuter', 'ringInner', 'distanceMode', 'refinement'],
      readouts: ['cellsInside', 'cellsRing']
    },
    'slope-result': {
      camera: {longitude: -121.7, latitude: 38.7, zoom: 6, transitionMs: 1800},
      options: {display: 'slope'},
      controls: ['steepSlope', 'ringOuter', 'sumOrder'],
      readouts: [
        'scope',
        'slopeMean',
        'steepShare',
        'steeperFires',
        'elevation',
        'slopeChart',
        'differenceChart'
      ]
    },
    'aspect-result': {
      camera: {longitude: -121.7, latitude: 38.7, zoom: 6, transitionMs: 1800},
      options: {display: 'southness'},
      controls: ['flatSlope', 'display', 'year', 'minAcres'],
      readouts: ['southShare', 'southness', 'aspectChart', 'aspectDifferenceChart']
    }
  }),

  legends: state => {
    if (state.display === 'zones') {
      return [
        {
          kind: 'categories',
          title: 'Zones',
          entries: [
            {color: [232, 90, 60, 190], label: 'Inside a fire perimeter'},
            {
              color: [78, 168, 222, 160],
              label: `Ring: ${state.ringInner} to ${state.ringOuter} km from a perimeter`
            }
          ],
          note: 'Overlapping fires: the smaller fire id owns the cell. Ring cells go to the nearest fire.'
        }
      ];
    }
    if (state.display === 'boundary') {
      return [
        {
          kind: 'categories',
          title: 'Rasterizer boundary',
          entries: [{color: [255, 244, 214, 255], label: 'Cell touched by a polygon edge'}]
        }
      ];
    }
    if (state.display === 'slope') {
      return [{kind: 'ramp', title: 'Slope', ramp: state.ramp, extent: [0, 40], unit: 'deg'}];
    }
    if (state.display === 'southness') {
      return [
        {
          kind: 'ramp',
          title: 'Southness',
          ramp: 'diverging',
          extent: [-1, 1],
          labels: ['north-facing', 'south-facing']
        }
      ];
    }
    return [];
  },

  readouts: [
    {
      id: 'scope',
      label: 'Compared',
      help: 'The fires the numbers below pool: every comparable fire, or the one you clicked.'
    },
    {
      id: 'comparable',
      label: 'Comparable fires',
      help: 'Fires on the DEM with at least 20 sloped cells inside and in the ring, that pass the filters.'
    },
    {
      id: 'cellsInside',
      label: 'Cells inside perimeters',
      format: 'integer',
      help: 'Cells of the zone raster inside the compared perimeters (cell is about 239 m).'
    },
    {
      id: 'cellsRing',
      label: 'Cells in the ring',
      format: 'integer',
      help: 'Cells of the ring around the compared fires.'
    },
    {
      id: 'rasterizer',
      label: 'Rasterizer',
      help: 'Crossing records used against the capacity sized from the geometry. Overflow would leave the zones empty.'
    },
    {
      id: 'slopeMean',
      label: 'Mean slope, inside vs ring',
      help: 'Cell-weighted mean slope in degrees, pooled over the compared fires, with the difference.'
    },
    {
      id: 'steepShare',
      label: 'Steep cells, inside vs ring',
      help: 'Share of cells at or above the steep slope threshold.'
    },
    {
      id: 'steeperFires',
      label: 'Fires steeper inside',
      help: 'How many fires have a higher mean slope inside the perimeter than in the ring.'
    },
    {
      id: 'medianDifference',
      label: 'Median slope difference',
      help: 'Median over fires of the inside minus ring mean slope.'
    },
    {
      id: 'elevation',
      label: 'Mean elevation, inside vs ring',
      help: 'Burned ground is usually higher than the surrounding valley land: a confounder for slope.'
    },
    {
      id: 'southShare',
      label: 'South-leaning cells (SE, S, SW)',
      help: 'Share of sloping cells facing SE, S or SW: 37.5% if directions were uniform.'
    },
    {
      id: 'southness',
      label: 'Mean southness, inside vs ring',
      help: 'Mean of minus cosine of aspect over sloping cells: +1 all south-facing, -1 all north-facing.'
    },
    {id: 'slopeChart', label: 'Mean slope', kind: 'chart'},
    {id: 'steepChart', label: 'Share of steep cells', kind: 'chart'},
    {id: 'differenceChart', label: 'Slope difference per fire', kind: 'chart'},
    {id: 'aspectChart', label: 'Facing direction, inside vs ring', kind: 'chart'},
    {id: 'aspectDifferenceChart', label: 'Facing direction excess inside', kind: 'chart'},
    {
      id: 'selected',
      label: 'Selected fire',
      layout: 'block',
      help: 'Click a fire to compare it alone. Click it again to clear.'
    }
  ],

  snippet: state => `import {
  getGPUDistanceFieldParameterValues,
  getGPUPolygonRasterizationExtentValues,
  GPUDistanceField,
  GPUPolygonRasterization,
  GPURasterZonalStatistics
} from '@luma.gl/experimental/gpu-raster';
import {GPUTerrainDerivatives} from '@luma.gl/experimental/gpu-terrain';

// Polygons in raster space: x east from the grid edge, y south from its north edge (row order).
graph.add(new GPUPolygonRasterization({
  width, height, extent,                 // [0, 0, cell, cell]
  polygonPositions, featureOffsets, polygonOffsets, ringOffsets,
  crossingCapacity,                      // sized from the (edge, row) crossings
  zones, boundary, overflow
}));
graph.add(new GPUDistanceField({
  width, height, settings, seedMask,     // seedMask = zone + 1, 0 elsewhere
  mode: '${state.distanceMode}',${state.distanceMode === 'jump-flood' ? `\n  jumpFloodRefinementPasses: ${state.refinement},` : ''}
  output: {distances, allocation}        // nearest fire id per cell
}));
graph.add(new GPUTerrainDerivatives({
  width, height, elevation, settings, slope, aspect, hillshade,
  cellSizeMode: 'web-mercator', rowDirection: 'south', borderMode: '${state.borderMode}'
}));
// ring zone = nearest fire where ${state.ringInner} km <= distance <= ${state.ringOuter} km (a kernel)
graph.add(new GPURasterZonalStatistics({
  width, height, zones: zoneId,          // 2 * fire (+1 for the ring)
  values: slopeBand, zoneCapacity: 2 * fireCount,
  sumOrder: '${state.sumOrder}',
  output: {valueCounts, sums, maximums}
}));`,

  about: {
    what: '`GPUPolygonRasterization` turns polygons into a dense zone raster, `GPUDistanceField` gives every cell its distance to and the id of the nearest burned cell, `GPUTerrainDerivatives` computes slope, aspect and hillshade, and `GPURasterZonalStatistics` summarises any band per zone. Together they answer a vector-raster question without leaving the GPU.',
    why: 'Does a phenomenon sit on different terrain than its surroundings? That is the question behind habitat models, landslide susceptibility and fire risk maps: rasterize the polygons, make a comparison ring, and compare the distributions.',
    howToRead:
      'Red is inside a perimeter, blue the ring. In the charts, inside is the first bar or line; the histogram counts fires. Numbers pool cells over fires (larger fires weigh more) unless you click one. The elevation grid is 239 m, perimeters are final NIFC outlines (not progression), and the archive is not complete. Terrain association is not cause: fuels, weather, ignition sources and suppression decide where fires burn.'
  },

  create: async ctx => (await import('./wildfire-terrain.compute')).createWildfireTerrain(ctx)
});
