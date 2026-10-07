// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {getClassTableLegend, makeClassTable} from '../../cartography/class-table';
import {defineScene, type LegendSpec, type OptionSpec} from '../scene';
import type {IsochronesOptions} from './isochrones.compute';

const ISOCHRONE_TIME_TABLE = makeClassTable({
  breaks: [120, 240, 360],
  scheme: 'YlGnBu',
  reverse: true,
  labels: ['0–2', '2–4', '4–6', '6–8 min'],
  unit: 'min',
  extent: [0, 480],
  noData: {color: [115, 80, 115, 175], label: 'Beyond 8 min'}
});

const FACILITY_LABELS: Record<IsochronesOptions['facilityType'], string> = {
  fire_station: 'fire stations',
  hospital: 'hospitals',
  library: 'public libraries',
  cps_school: 'CPS schools'
};

const options: readonly OptionSpec<IsochronesOptions>[] = [
  {
    kind: 'select',
    id: 'facilityType',
    label: 'Facilities',
    group: 'Facilities',
    apply: 'compile',
    default: 'fire_station',
    options: [
      {value: 'fire_station', label: 'Fire stations'},
      {value: 'hospital', label: 'Hospitals'},
      {value: 'library', label: 'Public libraries'},
      {value: 'cps_school', label: 'CPS schools'}
    ],
    help: 'Which public facilities are the seeds. The facility count is a compile-time size of the graphs, so changing it rebuilds them. Click the map to move the nearest facility and see the catchments change.'
  },
  {
    kind: 'button',
    id: 'reset',
    label: 'Put moved facilities back',
    group: 'Facilities',
    help: 'Restores the surveyed facility positions after you moved some by clicking.'
  },
  {
    kind: 'slider',
    id: 'trafficSlowdown',
    label: 'Traffic',
    group: 'Drive time',
    apply: 'param',
    min: 1,
    max: 3,
    step: 0.1,
    default: 1,
    unit: 'x',
    format: value => `${value.toFixed(1)}x slower`,
    help: 'Multiplies the free-flow travel time of every street. Rewrites the CSR weights; nothing recompiles.'
  },
  {
    kind: 'slider',
    id: 'intersectionDelay',
    label: 'Delay per intersection',
    group: 'Drive time',
    apply: 'param',
    min: 0,
    max: 30,
    step: 1,
    default: 6,
    unit: 's',
    help: 'Seconds added to every edge for the intersection it ends at.'
  },
  {
    kind: 'slider',
    id: 'walkBuffer',
    label: 'Reach off the street',
    group: 'Isochrone surface',
    apply: 'param',
    min: 0,
    max: 120,
    step: 5,
    default: 40,
    unit: 'm',
    help: 'bufferRadius of the isochrone raster: pixels within this distance of a street sample get its cost plus walking time (1.34 m/s). It is capped at 16 pixels, so zoom in to see the full radius.'
  },
  {
    kind: 'select',
    id: 'rasterMode',
    label: 'Surface rule',
    group: 'Isochrone surface',
    apply: 'compile',
    default: 'min',
    options: [
      {value: 'min', label: 'Closest street sample (min)'},
      {value: 'max', label: 'Every street sample (max)'}
    ],
    help: 'min is the usual isochrone: a pixel is as close as the closest street touching it. max is conservative: a pixel is inside a band only when every street sample touching it is. Compile-time.'
  },
  {
    kind: 'select',
    id: 'cellChoice',
    label: 'Coverage rings from',
    group: 'Coverage rings',
    apply: 'compile',
    default: 'h3-9',
    options: [
      {value: 'h3-8', label: 'H3 resolution 8 (about 460 m hexagons)'},
      {value: 'h3-9', label: 'H3 resolution 9 (about 175 m hexagons)'},
      {value: 'quadbin-16', label: 'Quadbin 16 (about 600 m tiles)'},
      {value: 'quadbin-17', label: 'Quadbin 17 (about 300 m tiles)'}
    ],
    help: 'The cell family and resolution used to outline the reached area into closed rings. A compile-time choice of the cell producer. Coarser cells give a smoother, more generous polygon.'
  },
  {
    kind: 'slider',
    id: 'straightRadius',
    label: 'Straight-line radius',
    group: 'Straight-line zones',
    apply: 'param',
    min: 500,
    max: 8000,
    step: 100,
    default: 2000,
    unit: 'm',
    help: 'maxDistance of the distance field: raster cells farther than this from every facility belong to no zone. 2 km is roughly four minutes at 30 km/h, the middle band of the default bands.'
  },
  {
    kind: 'toggle',
    id: 'showReferenceCircle',
    label: 'True-radius comparison circle',
    group: 'Straight-line zones',
    apply: 'param',
    default: false,
    help: 'Draws one geodesic circle at the current straight-line radius around the first current facility. It is a geometric reference, not a drive-time result.'
  },
  {
    kind: 'toggle',
    id: 'comparisonMode',
    label: 'One-facility comparison',
    group: 'Straight-line zones',
    apply: 'compile',
    default: false,
    help: 'Uses one selected facility for a like-for-like network footprint and true-radius ring.'
  },
  {
    kind: 'select',
    id: 'distanceMode',
    label: 'Distance algorithm',
    group: 'Straight-line zones',
    apply: 'compile',
    default: 'exact',
    options: [
      {value: 'exact', label: 'Exact (Felzenszwalb-Huttenlocher)'},
      {value: 'jump-flood', label: 'Jump flood (approximate, fewer passes)'}
    ],
    help: 'exact is a separable transform with exact Euclidean zones; jump-flood is faster on huge rasters but can misassign a few cells at zone borders. Compile-time.'
  },
  {
    kind: 'select',
    id: 'sumOrder',
    label: 'Zone sum order',
    group: 'Straight-line zones',
    apply: 'compile',
    default: 'atomic',
    options: [
      {value: 'atomic', label: 'Atomic (fast, order may vary)'},
      {value: 'sorted', label: 'Sorted (bit-reproducible)'}
    ],
    help: 'How GPURasterZonalStatistics accumulates the population per zone: atomic adds are fastest; sorted sums give identical results on every run. Compile-time.'
  },
  {
    kind: 'toggle',
    id: 'showBands',
    label: 'Drive-time isochrone bands',
    group: 'Layers',
    apply: 'param',
    default: true,
    help: 'The raster isoband polygons of GPUNetworkIsochrones.'
  },
  {
    kind: 'toggle',
    id: 'showServiceAreas',
    label: 'Service areas (nearest facility)',
    group: 'Layers',
    apply: 'param',
    default: false,
    help: 'Streets coloured by the facility that reaches them first (GPUNetworkServiceAreas). The same hue marks the facility.'
  },
  {
    kind: 'toggle',
    id: 'showRings',
    label: 'Coverage rings',
    group: 'Layers',
    apply: 'param',
    default: false,
    help: 'The outline of the cells reached within the time budget, assembled into closed shell and hole rings on the GPU.'
  },
  {
    kind: 'toggle',
    id: 'showStraightLine',
    label: 'Straight-line zones',
    group: 'Layers',
    apply: 'param',
    default: false,
    help: 'Nearest-facility zones as the crow flies (GPUDistanceField allocation), coloured like the facilities.'
  },
  {
    kind: 'toggle',
    id: 'showDemand',
    label: 'Residents (census tract centroids)',
    group: 'Layers',
    apply: 'param',
    default: true,
    help: 'One point per census tract, coloured by the isochrone band its nearest street edge falls in; grey points are beyond the budget.'
  },
  {
    kind: 'toggle',
    id: 'showStreets',
    label: 'Street grid',
    group: 'Layers',
    apply: 'param',
    default: true,
    help: 'Draws the drive network under the analysis layers.'
  }
];

export default defineScene<IsochronesOptions>({
  id: 'isochrones',
  title: 'Who falls inside each drive-time band?',
  chapter: 'networks',
  order: 2,
  summary:
    'Drive-time bands, service areas and coverage rings for Chicago’s fire stations, hospitals, libraries and schools, with the residents each band holds, compared against straight-line zones.',
  contributors: [
    'GPUNetworkIsochrones',
    'GPUNetworkServiceAreas',
    'GPUNetworkSnapping',
    'GPUPointInPolygonJoin',
    'GPUCellSetOutline',
    'GPUSegmentRingAssembly',
    'addDriveTimeCatchmentRecipe',
    'addStraightLineCatchmentsRecipe',
    'GPUDistanceField'
  ],
  datasets: [
    {id: 'chicago-roads', role: 'directed street graph'},
    {id: 'chicago-facilities', role: 'seeds'},
    {id: 'chicago-tracts', role: 'residents'}
  ],
  initialView: {longitude: -87.68, latitude: 41.835, zoom: 9.9},
  basemap: ground('paperCity'),
  furniture: {
    title: {title: 'Nearest facility by drive time'},
    credit: 'City of Chicago · Overture · Census · OpenStreetMap contributors (ODbL)',
    caveat: 'Free-flow network model; tract centroids represent tract residents.',
    scaleBar: {units: 'metric'}
  },
  options,

  readouts: [
    {id: 'facilities', label: 'Facilities', help: 'Seeds in the current set.'},
    {id: 'moved', label: 'Moved by you', help: 'Facilities relocated by clicking the map.'},
    {id: 'nodes', label: 'Intersections / edges', help: 'Size of the street CSR.'},
    {
      id: 'demandPoints',
      label: 'Residents modelled',
      help: 'Census tract centroids carrying the tract population (CDC/ATSDR SVI 2022).'
    },
    {
      id: 'servedBands',
      label: 'Residents within each band',
      help: 'Cumulative share of residents whose nearest street edge is within the band time of a facility, from the network-space band table.'
    },
    {id: 'bandPopulation', label: 'Residents by drive-time band', kind: 'chart'},
    {
      id: 'servedWithin',
      label: 'Residents within the budget',
      help: 'People in tracts whose centroid is reached within the full time budget.'
    },
    {
      id: 'unreachedTracts',
      label: 'Tracts beyond the budget',
      help: 'Tract centroids with no band.'
    },
    {
      id: 'straightServed',
      label: 'Within the straight-line radius',
      help: 'Residents inside the straight-line zones (zonal sum of the population raster).'
    },
    {
      id: 'busiestZone',
      label: 'Busiest straight-line zone',
      help: 'The facility whose Voronoi zone holds the most people.'
    },
    {
      id: 'ringJoin',
      label: 'Residents inside the rings',
      help: 'Population of tract centroids that fall inside the GPU-assembled coverage polygon.'
    },
    {
      id: 'ringJoinGap',
      label: 'Rings minus exact bands',
      help: 'The cell polygon is coarser than the network-space classification; this is the difference in residents counted.'
    },
    {id: 'ringComparison', label: 'Network count versus polygon join', kind: 'chart'},
    {id: 'ringShapes', label: 'Rings', help: 'Shells and holes of the coverage polygon.'},
    {
      id: 'ringHealth',
      label: 'Open / touching segments',
      help: 'Ring assembly diagnostics: both should be zero for clean rings.'
    },
    {
      id: 'cells',
      label: 'Cell outline',
      help: 'Reached cells and outline segments of the cell producer.'
    },
    {
      id: 'rings',
      label: 'Ring count',
      help: 'Number of rings written, against a capacity of 2,048.'
    },
    {
      id: 'triangles',
      label: 'Isoband triangles',
      help: 'Triangles the band raster produced, against a capacity of 800,000.'
    },
    {
      id: 'raster',
      label: 'Isochrone raster',
      help: 'The raster follows the viewport and is re-snapped as you pan or zoom.'
    },
    {
      id: 'straightCells',
      label: 'Cells allocated',
      help: 'Straight-line raster cells with a facility within the radius.'
    }
  ],

  legends: state => {
    const legends: LegendSpec[] = [];
    if (state.showBands) {
      legends.push(
        getClassTableLegend(ISOCHRONE_TIME_TABLE, {title: 'Drive time to nearest facility'})
      );
    }
    if (state.showDemand) {
      const bands = ISOCHRONE_TIME_TABLE.colors.map((color, band) => ({
        color,
        label: ISOCHRONE_TIME_TABLE.labels?.[band] ?? ''
      }));
      legends.push({
        kind: 'categories',
        title: 'Census tract centroids',
        entries: [...bands, {color: [110, 60, 110, 235], label: 'Beyond 8 min (hollow ring)'}],
        note: 'Each tract is classified by its nearest street edge; its centroid represents the whole tract.'
      });
    }
    legends.push({
      kind: 'categories',
      title: 'Map symbols',
      entries: [
        {color: [35, 38, 48, 255], label: `${FACILITY_LABELS[state.facilityType]} (ink ring-dots)`},
        {color: [35, 38, 48, 255], label: '4 min boundary'},
        {color: [70, 76, 90, 175], label: 'Road-data edge (dashed)'},
        ...(state.showRings ? [{color: [255, 80, 60, 255] as const, label: 'Coverage ring'}] : []),
        ...(state.showReferenceCircle
          ? [{color: [40, 190, 255, 255] as const, label: 'True-radius reference circle'}]
          : [])
      ],
      note: 'Click the map to move the nearest facility. Hover a facility for its name.'
    });
    return legends;
  },

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  addDriveTimeCatchmentRecipe,
  addStraightLineCatchmentsRecipe,
  getGPUDistanceFieldParameterValues
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {getGPUNetworkIsochroneParameterValues} from '@luma.gl/experimental/gpu-network';

const graph = new GPUCommandGraph(device, {id: 'catchments'});
const catchment = addDriveTimeCatchmentRecipe(graph, {
  network: {offsets, neighbors, weights, nodePositions},   // CSR in seconds, planar meters
  facilities, demand, demandValues: population,            // float32x2 points
  costLimit, maxIterations: 64,
  bandBreaks: breaks,                                       // ascending seconds, per frame
  isochrones: {
    breaks, parameters,
    raster: {width: 1024, height: 640, mode: '${state.rasterMode}', output: {triangles, triangleBands, count, overflow, vertexCount}}
  },
  assignments, nodeCosts, demandBands, bands: {keys, counts, count, overflow, sumValues}
});
// Closed rings of the reached ${state.cellChoice.startsWith('h3') ? 'H3' : 'Quadbin'} cells (resolution ${state.cellChoice.split('-')[1]}):
graph.add(new GPUNetworkIsochrones({
  offsets, neighbors, weights, nodePositions: lngLat, costs: catchment.nodeCosts, breaks, parameters,
  cellOutline: {family: '${state.cellChoice.startsWith('h3') ? 'h3' : 'quadbin'}', resolution: ${state.cellChoice.split('-')[1]}, output, rings: {output: ringOutput}}
}));
parameters.write(getGPUNetworkIsochroneParameterValues({
  breakCount: 4, extent, bufferRadius: ${state.walkBuffer}, walkCostPerUnit: 1 / 1.34, cellCostLimit: 8 * 60
}));
breaks.write(Float32Array.of(120, 240, 360, 480, 0, 0));

// As the crow flies, for comparison:
const straight = new GPUCommandGraph(device, {id: 'straight'});
addStraightLineCatchmentsRecipe(straight, {
  width, height, settings, seedPositions: facilities, seedCount, values: populationRaster,
  mode: '${state.distanceMode}', sumOrder: '${state.sumOrder}', allocation, distances, statistics
});
settings.write(getGPUDistanceFieldParameterValues({bounds, gridSize: [width, height], maxDistance: ${state.straightRadius}}));
graph.compile(); straight.compile();   // once; moving a facility is one buffer write`,

  about: {
    what: '`addDriveTimeCatchmentRecipe` chains `GPUNetworkSnapping` (each facility and each demand point snaps to its nearest street edge), `GPUNetworkServiceAreas` (a multi-source search that gives every intersection its cost and its nearest facility) and `GPUNetworkIsochrones` (edge-interpolated costs splatted onto a raster and contoured into bands by `GPUIsobands`). A second `GPUNetworkIsochrones` producer outlines the reached H3 or Quadbin cells into rings. `addStraightLineCatchmentsRecipe` gives the as-the-crow-flies answer.',
    why: 'Coverage questions are often expressed in minutes rather than metres. Isochrones show a modelled network-time surface, service areas show who covers whom, and straight-line zones show how much a simple buffer can mislead.',
    howToRead:
      'Stronger bands are closer to a facility. Points are census tract centroids coloured by fixed band; hollow plum rings are beyond eight minutes. Rings outline the cell approximation and the dashed frame marks the road-data edge.'
  },

  create: async ctx => (await import('./isochrones.compute')).createIsochrones(ctx),

  story: [
    {
      id: 'nearest-time',
      headline: 'Who falls inside each band?',
      textAlternative:
        'Chicago fire-station drive-time bands use fixed 2, 4, 6 and 8 minute colours, with centroid demand dots and a band-population chart.',
      optionsMode: 'fresh',
      controls: ['facilityType', 'trafficSlowdown'],
      readouts: ['servedBands', 'bandPopulation', 'servedWithin'],
      title: 'Who falls inside each band?',
      body: 'These are analytical breaks at 2, 4, 6 and 8 minutes, not a compliance finding. Demand dots use the same fixed classes; hollow plum rings are tract centroids beyond eight minutes.',
      camera: {longitude: -87.68, latitude: 41.84, zoom: 10.0, transitionMs: 1200},
      options: {
        facilityType: 'fire_station',
        showBands: true,
        showDemand: true
      },
      highlight: {readout: 'servedBands'}
    },
    {
      id: 'service-areas',
      headline: 'Every street is assigned to its nearest seed',
      textAlternative:
        'Streets are partitioned into five muted facility-allocation colours, with ink ring-dot facilities.',
      optionsMode: 'fresh',
      controls: ['showServiceAreas', 'reset'],
      readouts: ['moved'],
      title: 'Every street belongs to one station',
      body: '`GPUNetworkServiceAreas` runs one search from all stations at once. Each intersection keeps its nearest seed; ties use a stable row order. The muted allocation colours repeat after five hues, while facilities stay ink ring-dots.',
      camera: {longitude: -87.66, latitude: 41.87, zoom: 11.2, transitionMs: 1400},
      options: {
        showServiceAreas: true,
        showBands: false,
        showDemand: false,
        showReferenceCircle: false
      }
    },
    {
      id: 'network-not-circle',
      headline: 'A network band is not a circle',
      textAlternative:
        'One selected facility has a dashed geodesic radius ring and one comparable single-seed network footprint.',
      optionsMode: 'fresh',
      controls: ['straightRadius', 'walkBuffer', 'trafficSlowdown'],
      readouts: ['servedWithin'],
      title: 'A network band is not a circle',
      body: 'The dashed true-radius ring and the isochrone now use one selected facility, so this is a like-for-like comparison. The network footprint follows directed streets; the circle is only a geometric reference.',
      camera: {longitude: -87.64, latitude: 41.82, zoom: 11.6, transitionMs: 1400},
      options: {
        facilityType: 'fire_station',
        comparisonMode: true,
        showReferenceCircle: true,
        showBands: true,
        showDemand: false,
        walkBuffer: 40
      }
    },
    {
      id: 'people',
      headline: 'A tract centroid is a useful but imperfect proxy',
      textAlternative:
        'Census tract centroid dots are coloured by fixed drive-time class; unserved centroids are hollow plum rings.',
      optionsMode: 'fresh',
      controls: ['facilityType', 'showDemand'],
      readouts: ['servedBands', 'bandPopulation', 'servedWithin'],
      title: 'Counting the people in each band',
      body: 'Each tract is represented by its centroid, snapped to its nearest street edge and classed in network space. Population totals are live, but a centroid cannot describe variation inside a tract.',
      camera: {longitude: -87.68, latitude: 41.84, zoom: 10.0, transitionMs: 1400},
      options: {showDemand: true, showBands: true},
      highlight: {readout: 'servedWithin'}
    },
    {
      id: 'cells-to-ring',
      headline: 'Cells trade geometric detail for a coverage outline',
      textAlternative:
        'Faint H3 or Quadbin boundary edges sit below an assembled ink coverage ring, with a chart comparing network and polygon joins.',
      optionsMode: 'fresh',
      controls: ['showRings', 'cellChoice'],
      readouts: ['ringJoin', 'ringJoinGap', 'ringHealth', 'ringComparison'],
      title: 'A coverage polygon from cells',
      body: 'Faint cell boundaries sit below the assembled ink ring. The chart compares the direct network-count with the polygon join, making the cell representation and MAUP difference explicit.',
      camera: {longitude: -87.62, latitude: 41.87, zoom: 10.9, transitionMs: 1400},
      options: {showRings: true, showBands: false, showDemand: true, cellChoice: 'h3-9'},
      highlight: {readout: 'ringJoin'}
    },
    {
      id: 'edge-and-model',
      headline: 'The graph boundary and free-flow model constrain the answer',
      textAlternative:
        'The road-data frame is shown as a dashed edge around the analysis; outside it no network cost is claimed.',
      optionsMode: 'fresh',
      controls: ['rasterMode', 'distanceMode', 'sumOrder'],
      readouts: ['servedWithin', 'raster'],
      title: 'The data ends at the edge',
      body: 'The dashed data frame marks the road-graph extent: outside it, this model makes no service claim. Times are free-flow with fixed intersection delay; raster, cell, distance and sum variants are engineering choices, not observed travel.',
      camera: {longitude: -87.68, latitude: 41.84, zoom: 10.0, transitionMs: 1200},
      options: {showStraightLine: false, showRings: false, showBands: true, showDemand: true}
    }
  ]
});
