// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene, type LegendSpec} from '../scene';
import type {GreatCirclesOptions} from './b3-line-options';

const HUB_OPTIONS = [
  {value: 'ORD', label: "Chicago O'Hare (ORD)"},
  {value: 'AMS', label: 'Amsterdam Schiphol (AMS)'},
  {value: 'DXB', label: 'Dubai (DXB)'},
  {value: 'SIN', label: 'Singapore Changi (SIN)'},
  {value: 'SYD', label: 'Sydney (SYD)'},
  {value: 'GRU', label: 'Sao Paulo Guarulhos (GRU)'},
  {value: 'JNB', label: 'Johannesburg (JNB)'},
  {value: 'HND', label: 'Tokyo Haneda (HND)'}
] as const;

/** Great-circle arcs and geodesics on the OpenFlights network. GPU work in `great-circles.compute.ts`. */
export default defineScene<GreatCirclesOptions>({
  id: 'great-circles',
  title: 'Great circles from every hub',
  chapter: 'geometry',
  order: 5,
  summary:
    'Draw all 18,930 airline routes of the OpenFlights network as great-circle arcs, measure the geodesic distance and bearing from a hub to all 3,257 airports, and trace a geodesic range ring around it.',
  contributors: ['GPUGreatCircleArcs', 'GPUGeodesicPairs', 'GPUGeodesicDestination'],
  datasets: [{id: 'openflights', role: 'airports and routes'}],
  initialView: {longitude: 10, latitude: 28, zoom: 1.35},

  options: [
    {
      kind: 'toggle',
      id: 'showArcs',
      label: 'Route arcs',
      group: 'Arcs',
      apply: 'param',
      default: true,
      help: 'The great circle of every route.'
    },
    {
      kind: 'select',
      id: 'arcColor',
      label: 'Arcs coloured by',
      group: 'Arcs',
      apply: 'param',
      default: 'distance',
      options: [
        {value: 'distance', label: 'Route length (km)'},
        {value: 'airlines', label: 'Number of airlines'},
        {value: 'routes', label: 'Number of route records'}
      ],
      help: 'A per-route column bound straight into the arc layer.'
    },
    {
      kind: 'slider',
      id: 'arcMinimumSegments',
      label: 'Minimum segments per arc',
      group: 'Arcs',
      apply: 'param',
      min: 1,
      max: 64,
      step: 1,
      default: 24,
      help: 'Every arc gets at least this many segments (up to the compile-time maximum of 64). 1 draws a straight chord in longitude and latitude. A parameter write.'
    },
    {
      kind: 'slider',
      id: 'arcMaximumLength',
      label: 'Maximum segment length',
      group: 'Arcs',
      apply: 'param',
      min: 0,
      max: 2000,
      step: 50,
      default: 0,
      unit: ' km',
      help: 'Adds segments to long arcs so no piece is longer than this. 0 disables the limit.'
    },
    {
      kind: 'select',
      id: 'worldHub',
      label: 'Hub',
      group: 'Hub',
      apply: 'param',
      default: 'ORD',
      options: HUB_OPTIONS,
      help: 'Rewrites the origin column of the distance pairs and the centre of the range ring. A buffer write, no recompile.'
    },
    {
      kind: 'select',
      id: 'geodesicModel',
      label: 'Earth model',
      group: 'Hub',
      apply: 'compile',
      default: 'wgs84',
      help: 'Sphere: haversine on a sphere. WGS84: Vincenty on the ellipsoid (a few tenths of a percent different). Compile-time: a graph per model.',
      options: [
        {value: 'sphere', label: 'Sphere (haversine)'},
        {value: 'wgs84', label: 'WGS84 ellipsoid (Vincenty)'}
      ]
    },
    {
      kind: 'slider',
      id: 'ringDistance',
      label: 'Range ring distance',
      group: 'Hub',
      apply: 'param',
      min: 500,
      max: 15000,
      step: 250,
      default: 5000,
      unit: 'km',
      help: 'Rewrites the distances column of GPUGeodesicDestination: 361 destinations, one per degree of bearing.'
    },
    {
      kind: 'toggle',
      id: 'showRing',
      label: 'Range ring',
      group: 'Hub',
      apply: 'param',
      default: true,
      help: 'The set of points exactly this far from the hub, on the chosen Earth model.'
    },
    {
      kind: 'select',
      id: 'airportColor',
      label: 'Airports coloured by',
      group: 'Hub',
      apply: 'param',
      default: 'distance',
      options: [
        {value: 'distance', label: 'Distance from the hub', help: 'GPUGeodesicPairs distances.'},
        {
          value: 'bearing',
          label: 'Initial bearing from the hub',
          help: 'Degrees clockwise from north at the hub.'
        }
      ],
      help: 'Which GPUGeodesicPairs output colours the airports.'
    }
  ],

  story: [
    {
      id: 'network',
      title: "Where do the world's airline routes go?",
      body: `The OpenFlights network records 18,930 airline route pairs between 3,257 airports. To draw each as the shortest path on the globe, \`GPUGreatCircleArcs\` generates **one great-circle polyline per pair** on the GPU: 64 segments at most, longitudes unwrapped past 180 degrees so arcs across the Pacific stay in one piece (the map draws three copies of the world).

Colour is route length: short regional hops are purple, ultra-long-haul arcs yellow. The routes are a classic hub-and-spoke network: a few airports (Amsterdam, Frankfurt, Paris, Istanbul, Atlanta, Chicago) carry most connections. The data is frozen around 2014, so treat it as structure, not current traffic.`,
      camera: {longitude: 10, latitude: 28, zoom: 1.35, transitionMs: 1000},
      options: {arcColor: 'distance'},
      controls: ['arcColor'],
      readouts: ['worldInputs']
    },
    {
      id: 'curvature',
      title: 'How many segments does an arc need?',
      body: `A great circle is only a straight line in 3D. Projected to longitude and latitude it bends, and the bend needs vertices. **Minimum segments per arc** sets how many each arc gets; with 1 every arc is a straight chord on this map, which is plainly wrong for flights from Chicago to Asia over the pole.

Set it to 1, then 6, then 24 and watch the arcs curve. **Maximum segment length** adds more segments to long arcs only, so short routes stay cheap. Both are parameter writes: the vertex count in the readout changes without a recompile.`,
      options: {arcMinimumSegments: 4},
      controls: ['arcMinimumSegments', 'arcMaximumLength'],
      readouts: ['worldArcVertices'],
      highlight: {readout: 'worldArcVertices'}
    },
    {
      id: 'distances',
      title: 'How far is everywhere from Chicago?',
      body: `\`GPUGeodesicPairs\` computes the **geodesic distance and initial bearing** from one origin to many targets: here from a hub to every airport. The airports are coloured by distance, from the hub outward: nearby airports are dark and the antipodal side of the globe glows.

Choose a different **Hub** below: the origin column is rewritten and the distances recomputed in one pass. The readouts report the farthest airport and the mean distance. Set **Airports coloured by** to *Initial bearing* to see which compass direction each airport lies in from the hub: the great-circle bearing, not the straight line on the map.`,
      options: {showArcs: false, airportColor: 'distance', worldHub: 'ORD'},
      highlight: {readout: 'worldFarthest'},
      controls: ['worldHub', 'airportColor'],
      readouts: ['worldFarthest', 'worldMean']
    },
    {
      id: 'earth-model',
      title: 'Sphere or ellipsoid?',
      body: `The Earth is flattened: its radius is 21 km smaller at the poles than at the equator. \`GPUGeodesicPairs\` can measure on a **sphere** (haversine) or on the **WGS84 ellipsoid** (Vincenty's iteration, matched to PostGIS \`ST_Distance\` on a geography). Between airports the two differ by up to about half a percent, tens of kilometres on a long route.

Switch **Earth model** below: the farthest distance changes (compile-time, one graph per model). The ellipsoid is the reference; the sphere is faster and good enough for many maps. Near-antipodal pairs, where Vincenty may not converge, fall back to the sphere: the \`converged\` output flags those.`,
      options: {geodesicModel: 'sphere'},
      highlight: {readout: 'worldFarthest'},
      controls: ['geodesicModel'],
      readouts: ['worldFarthest']
    },
    {
      id: 'range-ring',
      title: 'Everything within 5,000 km',
      body: `\`GPUGeodesicDestination\` answers the reverse question: starting from a point, a bearing and a distance, where do you end up? One destination per degree of bearing traces a **geodesic range ring**: the set of points exactly 5,000 km from the hub. On this map it is far from a circle, because distance on the globe is not distance on the page.

Slide **Range ring distance** below: the ring grows with no recompile, because the distances column is just rewritten. The readout counts the airports inside the ring: compare how many airports fall inside it from Amsterdam and from Sydney with **Hub**.`,
      options: {
        showRing: true,
        worldHub: 'AMS',
        geodesicModel: 'wgs84',
        showArcs: true,
        arcMinimumSegments: 24,
        ringDistance: 5000
      },
      highlight: {readout: 'worldInside'},
      controls: ['ringDistance', 'worldHub'],
      readouts: ['worldInside']
    },
    {
      id: 'limits',
      title: 'Limits and things to try',
      body: `**Limits.** The arcs are great circles on a sphere (the ellipsoid is only used for distances and destinations); longitudes are unwrapped, so draw three world copies or use a globe. Near-antipodal Vincenty pairs fall back to the sphere. The OpenFlights data is community-maintained and frozen around 2014.

**Try it.** From Sydney, set **Range ring distance** to 12,000 km and find which airports lie outside it. Set **Arcs coloured by** to *Number of airlines* to find the routes shared by many carriers. Switch **Earth model** back and forth and watch the farthest-airport readout.`,
      options: {worldHub: 'SYD', ringDistance: 12000, arcColor: 'airlines'},
      highlight: {readout: 'worldInside'},
      controls: ['worldHub', 'ringDistance', 'arcColor', 'geodesicModel'],
      readouts: ['worldInside', 'worldFarthest']
    }
  ],

  about: {
    what: 'Great-circle arc generation for many origin and destination pairs (GPUGreatCircleArcs), geodesic distance and bearing for many pairs on a sphere or the WGS84 ellipsoid (GPUGeodesicPairs), and the destination point from an origin, bearing and distance (GPUGeodesicDestination).',
    why: 'Distances on the globe are not distances on the page. Flight and shipping maps, range rings and coverage fans all need geodesics, and they need them for thousands of pairs at once.',
    howToRead:
      'Arcs are the shortest paths between airports; their colour is the legend. Airports are coloured by distance or bearing from the chosen hub; the orange ring is the set of points at the ring distance from it.'
  },

  legends: state => {
    const legends: LegendSpec[] = [];
    if (state.showArcs) {
      legends.push(
        state.arcColor === 'distance'
          ? {
              kind: 'ramp',
              title: 'Route length',
              ramp: 'viridis',
              extent: [0, 12000],
              format: value => `${value.toLocaleString('en-US')} km`
            }
          : state.arcColor === 'airlines'
            ? {
                kind: 'ramp',
                title: 'Airlines serving the route',
                ramp: 'magma',
                extent: [1, 8],
                format: value => value.toFixed(0)
              }
            : {
                kind: 'ramp',
                title: 'Route records',
                ramp: 'magma',
                extent: [1, 12],
                format: value => value.toFixed(0)
              }
      );
    }
    legends.push(
      state.airportColor === 'distance'
        ? {
            kind: 'ramp',
            title: 'Airport distance from the hub',
            ramp: 'inferno',
            extent: [0, 15000],
            format: value => `${value.toLocaleString('en-US')} km`
          }
        : {
            kind: 'ramp',
            title: 'Initial bearing from the hub',
            ramp: 'viridis',
            extent: [0, 360],
            format: value => `${value.toFixed(0)}°`
          }
    );
    if (state.showRing) {
      legends.push({
        kind: 'categories',
        title: 'Range ring',
        entries: [
          {
            color: [200, 90, 10, 255],
            label: `${state.ringDistance.toLocaleString('en-US')} km from the hub`
          }
        ]
      });
    }
    return legends;
  },

  readouts: [
    {id: 'worldInputs', label: 'Network'},
    {id: 'worldArcVertices', label: 'Arc vertices'},
    {id: 'worldFarthest', label: 'Farthest airport from the hub'},
    {id: 'worldMean', label: 'Mean distance to an airport'},
    {id: 'worldInside', label: 'Inside the range ring'}
  ],

  snippet: state => `import {
  GPUGreatCircleArcs, GPUGeodesicPairs, GPUGeodesicDestination,
  getGPUGreatCircleArcsParameterValues
} from '@luma.gl/experimental/gpu-spatial-analysis';

// One great-circle polyline per route pair (lng/lat float32x2 columns)
graph.add(new GPUGreatCircleArcs({
  sources, targets, maximumSegments: 64,
  parameters: arcParameters.importToGraph(graph),
  output: {positions, pathOffsets, count, overflow}
}));
arcParameters.write(getGPUGreatCircleArcsParameterValues({
  minimumSegments: ${state.arcMinimumSegments}, maximumSegmentLength: ${state.arcMaximumLength * 1000}
}));

// Distance and bearing from the hub to every airport
graph.add(new GPUGeodesicPairs({
  origins: hubColumn, targets: airports, model: '${state.geodesicModel}',
  output: {distances, initialBearings, converged}
}));

// The range ring: 361 bearings at one distance
graph.add(new GPUGeodesicDestination({
  origins: hubColumn361, bearings: degrees, distances: ringMetres,   // ${state.ringDistance * 1000} m
  model: '${state.geodesicModel}', output: {destinations}
}));`,

  create: async ctx => (await import('./great-circles.compute')).createGreatCircles(ctx)
});
