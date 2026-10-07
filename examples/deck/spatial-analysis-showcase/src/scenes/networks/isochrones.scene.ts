// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene, type LegendSpec, type OptionSpec} from '../scene';
import {getBandColor} from './b9-shared';
import type {IsochronesOptions} from './isochrones.compute';

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
      {value: 'fire_station', label: 'Fire stations (92)'},
      {value: 'hospital', label: 'Hospitals (53, approximate)'},
      {value: 'library', label: 'Public libraries (82)'},
      {value: 'cps_school', label: 'CPS schools (649)'}
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
    id: 'budgetMinutes',
    label: 'Time budget',
    group: 'Drive time',
    apply: 'param',
    min: 2,
    max: 30,
    step: 1,
    default: 6,
    unit: 'min',
    help: 'The last isochrone threshold and the cost limit of the multi-source search. Bands are equal slices of it.'
  },
  {
    kind: 'slider',
    id: 'bandCount',
    label: 'Isochrone bands',
    group: 'Drive time',
    apply: 'param',
    min: 1,
    max: 6,
    step: 1,
    default: 3,
    help: 'How many of the compile-time break slots are active (breakCount). Three bands of a 6-minute budget give 2, 4 and 6 minutes.'
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
  },
  {
    kind: 'select',
    id: 'ramp',
    label: 'Band colour ramp',
    group: 'Layers',
    apply: 'param',
    default: 'viridis',
    options: [
      {value: 'viridis', label: 'Viridis'},
      {value: 'magma', label: 'Magma'},
      {value: 'inferno', label: 'Inferno'},
      {value: 'cividis', label: 'Cividis (colour-blind optimised)'}
    ],
    help: 'Near bands take the dark end of the ramp, the farthest band the bright end.'
  }
];

export default defineScene<IsochronesOptions>({
  id: 'isochrones',
  title: 'Who is within four minutes of a fire station?',
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
      legends.push({
        kind: 'ramp',
        title: 'Drive time to the nearest facility',
        ramp: state.ramp,
        extent: [0, state.budgetMinutes],
        unit: 'min',
        labels: ['at a facility', `${state.budgetMinutes} min`],
        format: value => value.toFixed(0)
      });
    }
    if (state.showDemand) {
      const bands = Array.from({length: state.bandCount}, (_, band) => {
        const minutes = (state.budgetMinutes * (band + 1)) / state.bandCount;
        return {
          color: getBandColor(state.ramp, band, state.bandCount),
          label: `Within ${minutes % 1 === 0 ? minutes : minutes.toFixed(1)} min`
        };
      });
      legends.push({
        kind: 'categories',
        title: 'Census tract centroids',
        entries: [...bands, {color: [120, 125, 140, 160], label: 'Beyond the budget'}],
        note: 'Each tract is classified by its nearest street edge, in network space.'
      });
    }
    legends.push({
      kind: 'categories',
      title: 'Map symbols',
      entries: [
        {color: [78, 201, 255, 255], label: `${FACILITY_LABELS[state.facilityType]} (hue = zone)`},
        ...(state.showRings ? [{color: [255, 80, 60, 255] as const, label: 'Coverage ring'}] : [])
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
  breakCount: ${state.bandCount}, extent, bufferRadius: ${state.walkBuffer}, walkCostPerUnit: 1 / 1.34, cellCostLimit: ${state.budgetMinutes * 60}
}));
breaks.write(Float32Array.from({length: 6}, (_, k) => ${state.budgetMinutes * 60} * Math.min(k + 1, ${state.bandCount}) / ${state.bandCount}));

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
    why: 'Coverage standards are written in minutes, not metres: a fire engine should reach an address within four minutes, a clinic within fifteen. Isochrones show where the standard is met, service areas show who covers whom, and the straight-line zones show how much a simple buffer would mislead.',
    howToRead:
      'Darker bands are closer to a facility. Points are census tract centroids coloured by band; grey ones are beyond the budget. Rings outline the area within the full budget; the straight-line zones are the faint coloured regions. Compare the people inside each in the readouts.'
  },

  create: async ctx => (await import('./isochrones.compute')).createIsochrones(ctx),

  story: [
    {
      id: 'the-question',
      controls: ['facilityType', 'budgetMinutes', 'bandCount'],
      readouts: ['servedBands', 'servedWithin'],
      title: 'Who can a fire engine reach in four minutes?',
      body: 'US fire-service guidance (NFPA 1710) asks for a first engine at the scene within **four minutes of travel**. Chicago has 92 fire stations. A circle drawn around each station would be easy to draw and wrong: engines drive on streets, one-way streets and expressways included.\n\nThe map shows drive-time bands of 2, 4 and 6 minutes from every station at once, in free-flow traffic with a few seconds lost at each intersection. The dots are census tract centroids; the readouts count the people in each band.',
      camera: {longitude: -87.68, latitude: 41.84, zoom: 10.0, transitionMs: 1200},
      options: {
        facilityType: 'fire_station',
        budgetMinutes: 6,
        bandCount: 3,
        showBands: true,
        showDemand: true
      },
      highlight: {readout: 'servedBands'}
    },
    {
      id: 'service-areas',
      controls: ['showServiceAreas', 'reset'],
      readouts: ['moved'],
      title: 'Every street belongs to one station',
      body: '`GPUNetworkServiceAreas` runs one search from **all** stations at once. Each intersection keeps the cost to its nearest station and *which* station that is (ties go to the lowest row). Facilities are snapped to the middle of their street edge first by `GPUNetworkSnapping`, so a station on a long block starts mid-block.\n\nTurn on **Service areas (nearest facility)** below: streets take the hue of their nearest station. The borders are where two stations are equally far, and they follow streets, not straight lines. Click anywhere on the map to move a station and the whole partition updates in a frame; **Put moved facilities back** undoes it.',
      camera: {longitude: -87.66, latitude: 41.87, zoom: 11.2, transitionMs: 1400},
      options: {showServiceAreas: true, showBands: false, showDemand: false}
    },
    {
      id: 'isochrone-bands',
      controls: ['walkBuffer', 'budgetMinutes', 'bandCount', 'trafficSlowdown'],
      title: 'From node costs to polygons',
      body: 'Costs live on intersections, but a polygon needs a surface. `GPUNetworkIsochrones` samples every street edge, interpolates the cost along it and writes the minimum to a raster, then `GPUIsobands` contours the raster into filled bands. The **Reach off the street** slider is the buffer: pixels near a street get its cost plus walking time.\n\nSlide the **Time budget** and **Isochrone bands** below: they are two parameter buffers, so nothing recompiles. Raise **Traffic** to 2x and the four-minute band shrinks to a patch around each station.',
      camera: {longitude: -87.64, latitude: 41.82, zoom: 11.6, transitionMs: 1400},
      options: {showServiceAreas: false, showBands: true, showDemand: false, walkBuffer: 40}
    },
    {
      id: 'residents',
      controls: ['facilityType', 'budgetMinutes'],
      readouts: ['servedBands', 'servedWithin'],
      title: 'Counting the people in each band',
      body: 'The recipe also snaps every demand point to its street edge, computes its drive time as the cheaper of the edge’s two ends plus the snap offset, and classifies it into a band; `GPUGroupStatistics` then sums population per band. This is **exact in network space**: no polygon is involved.\n\nRead **Residents within each band**: that is the answer to the opening question. Switch **Facilities** to hospitals and watch the same machinery answer a different question; with only 53 sites you will want a 12-minute **Time budget**, and large areas still stay dark. Remember each tract counts as one point, so a tract split by a band edge is classified by its centroid.',
      camera: {longitude: -87.68, latitude: 41.84, zoom: 10.0, transitionMs: 1400},
      options: {showDemand: true, showBands: true},
      highlight: {readout: 'servedWithin'}
    },
    {
      id: 'coverage-rings',
      controls: ['showRings', 'cellChoice'],
      readouts: ['ringJoin', 'ringJoinGap', 'ringHealth'],
      title: 'A coverage polygon from cells',
      body: 'For GIS exports you want a polygon, not triangles. The second `GPUNetworkIsochrones` producer labels every H3 (or Quadbin) cell that holds a node reached within the budget, `GPUCellSetOutline` finds the boundary edges, and `GPUSegmentRingAssembly` chains them into closed shell and hole rings, all on the GPU. `GPUPointInPolygonJoin` then tests the tract centroids against those rings.\n\nCompare **Residents inside the rings** with the exact band count: the cell polygon is coarser, so the two differ. Try H3 resolution 8 against 9 and Quadbin in **Coverage rings from**, and check that **Open / touching segments** stays at zero.',
      camera: {longitude: -87.62, latitude: 41.87, zoom: 10.9, transitionMs: 1400},
      options: {showRings: true, showBands: false, showDemand: true, cellChoice: 'h3-9'},
      highlight: {readout: 'ringJoin'}
    },
    {
      id: 'straight-line',
      controls: ['showStraightLine', 'straightRadius', 'distanceMode'],
      readouts: ['straightServed', 'servedBands'],
      title: 'As the crow flies',
      body: '`addStraightLineCatchmentsRecipe` answers the same question with distance: `GPUDistanceField` allocates every raster cell to its nearest station (Voronoi zones) and `GPURasterZonalStatistics` sums the population raster per zone. The **Straight-line radius** slider is the equivalent of the time budget; at 2 km it is roughly four minutes at 30 km/h.\n\nTurn on **Straight-line zones** over the rings: they claim areas across the river, the lakefront and the expressways that the drive-time answer does not. Compare **Within the straight-line radius** with the four-minute share in **Residents within each band**. Switch **Distance algorithm** to jump flood to see the approximate variant.',
      camera: {longitude: -87.66, latitude: 41.86, zoom: 10.6, transitionMs: 1400},
      options: {
        showStraightLine: true,
        showRings: true,
        showBands: true,
        showDemand: false,
        straightRadius: 3000
      }
    },
    {
      id: 'limits',
      controls: ['facilityType', 'trafficSlowdown', 'rasterMode', 'distanceMode', 'sumOrder'],
      readouts: ['servedWithin'],
      title: 'Limits, and things to try',
      body: 'Travel times are free-flow with a fixed intersection delay: no live traffic, no emergency-vehicle preemption, no turn restrictions, and the raster treats curved streets as straight chords between intersections. Facility locations come from the City and Overture (the hospital list is approximate); tract populations are the 2018-2022 ACS via CDC SVI. A tract is one point, so people are not spread over its area (the straight-line side does spread them).\n\nTry: relocate a station by clicking and watch **Residents within the budget** change; set **Traffic** to 2x; switch **Facilities** to libraries with a 15-minute **Time budget** (the slider reaches 30); set **Surface rule** to max for a conservative surface; use jump flood in **Distance algorithm** and sorted **Zone sum order** and compare the straight-line readouts.',
      camera: {longitude: -87.68, latitude: 41.84, zoom: 10.0, transitionMs: 1200},
      options: {showStraightLine: false, showRings: false, showBands: true, showDemand: true}
    }
  ]
});
