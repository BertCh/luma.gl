// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CREDITS, joinCredits} from '../../cartography/credits';
import {ground} from '../../cartography/grounds';
import {defineScene, type LegendSpec} from '../scene';
import type {GreatCirclesOptions} from './b3-line-options';
import {B3_PALETTE} from './b3-palette';

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
  title: 'What is straight on a sphere?',
  chapter: 'geometry',
  order: 5,
  summary:
    'Draw all 18,930 airline routes of the OpenFlights network as great-circle arcs, measure the geodesic distance and bearing from a hub to all 3,257 airports, and trace a geodesic range ring around it.',
  contributors: ['GPUGreatCircleArcs', 'GPUGeodesicPairs', 'GPUGeodesicDestination'],
  datasets: [{id: 'openflights', role: 'airports and routes'}],
  initialView: {longitude: 10, latitude: 28, zoom: 1.35},
  basemap: ground('space', {labels: 'none', graticule: true}),
  furniture: {
    title: {
      title: 'Great-circle routes',
      subtitle: 'OpenFlights network · frozen community snapshot'
    },
    credit: joinCredits(CREDITS.naturalEarth, 'OpenFlights community data')
  },

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
      max: 20038,
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
      kind: 'toggle',
      id: 'showRhumbComparison',
      label: 'Rhumb comparison',
      group: 'Reference geometry',
      apply: 'param',
      default: false,
      expert: true,
      help: 'Shows the selected loaded long pair as a great circle and a constant-heading rhumb.'
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
      headline: 'A few hubs organise many routes',
      body: `\`GPUGreatCircleArcs\` generates one shortest-path polyline per loaded OpenFlights pair. The five haul classes and low-alpha accumulation reveal the network without treating the frozen community snapshot as a live schedule. Airport symbols use the square root of their loaded degree so hubs remain legible.`,
      camera: {longitude: 10, latitude: 28, zoom: 1.35, transitionMs: 1000},
      options: {arcColor: 'distance', showRhumbComparison: false},
      controls: ['arcColor'],
      readouts: ['worldInputs']
    },
    {
      id: 'rhumb',
      title: 'Straight on Mercator is not shortest',
      headline: 'Mercator straightness is not shortest',
      body: `The selected long archive pair is drawn twice: teal follows the great circle and dashed orange holds a constant rhumb heading. Their lengths, excess and headings are read from the same endpoints. The comparison is projected in Mercator; the graticule explains its bend, not a change of route.`,
      camera: {longitude: 10, latitude: 28, zoom: 1.35, transitionMs: 1000},
      options: {showArcs: false, showRing: false, showRhumbComparison: true},
      controls: [],
      readouts: ['rhumbPair', 'rhumbLengths', 'rhumbExcess', 'rhumbHeading']
    },
    {
      id: 'segments',
      title: 'Slerp vertices reveal curved paths',
      headline: 'More vertices reveal the projected curve',
      body: `A great circle is only a straight line in 3D. Projected to longitude and latitude it bends, and the bend needs vertices. **Minimum segments per arc** sets how many each arc gets; with 1 every arc is a straight chord on this map, which is plainly wrong for flights from Chicago to Asia over the pole.

Set it to 1, then 6, then 24 and watch the arcs curve. **Maximum segment length** adds more segments to long arcs only, so short routes stay cheap. Both are parameter writes: the vertex count in the readout changes without a recompile.`,
      options: {arcMinimumSegments: 4, showRhumbComparison: false},
      controls: ['arcMinimumSegments', 'arcMaximumLength'],
      readouts: ['worldArcVertices'],
      highlight: {readout: 'worldArcVertices'}
    },
    {
      id: 'distance-bearing',
      title: 'Distance and bearing share one hub',
      headline: 'Distance and direction share an Earth model',
      body: `\`GPUGeodesicPairs\` computes the **geodesic distance and initial bearing** from one origin to many targets: here from a hub to every airport. The airports are coloured by distance, from the hub outward: nearby airports are dark and the antipodal side of the globe glows.

Choose a different **Hub** below: the origin column is rewritten and the distances recomputed in one pass. The readouts report the farthest airport and the mean distance. Set **Airports coloured by** to *Initial bearing* to see which compass direction each airport lies in from the hub: the great-circle bearing, not the straight line on the map.`,
      options: {
        showArcs: false,
        airportColor: 'distance',
        worldHub: 'ORD',
        showRhumbComparison: false
      },
      highlight: {readout: 'worldFarthest'},
      controls: ['worldHub', 'airportColor'],
      readouts: ['worldFarthest', 'worldMean']
    },
    {
      id: 'earth-model',
      title: 'Sphere or ellipsoid?',
      headline: 'Sphere and ellipsoid differ by direction',
      body: `The Earth is flattened: its radius is 21 km smaller at the poles than at the equator. \`GPUGeodesicPairs\` can measure on a **sphere** (haversine) or on the **WGS84 ellipsoid** (Vincenty's iteration, matched to PostGIS \`ST_Distance\` on a geography). Between airports the two differ by up to about half a percent, tens of kilometres on a long route.

Switch **Earth model** below: the farthest distance changes (compile-time, one graph per model). The ellipsoid is the reference; the sphere is faster and good enough for many maps. Near-antipodal pairs, where Vincenty may not converge, fall back to the sphere: the \`converged\` output flags those.`,
      options: {geodesicModel: 'sphere', showRhumbComparison: false},
      highlight: {readout: 'worldFarthest'},
      controls: ['geodesicModel'],
      readouts: ['worldFarthest']
    },
    {
      id: 'range-ring',
      title: 'Everything within 5,000 km',
      headline: 'A geodesic ring can enclose a pole',
      body: `\`GPUGeodesicDestination\` answers the reverse question: starting from a point, a bearing and a distance, where do you end up? One destination per degree of bearing traces a **geodesic range ring**: the set of points exactly 5,000 km from the hub. On this map it is far from a circle, because distance on the globe is not distance on the page.

Slide **Range ring distance** below: the ring grows with no recompile, because the distances column is just rewritten. The readout counts the airports inside the ring: compare how many airports fall inside it from Amsterdam and from Sydney with **Hub**.`,
      options: {
        showRing: true,
        worldHub: 'AMS',
        geodesicModel: 'wgs84',
        showArcs: true,
        arcMinimumSegments: 24,
        ringDistance: 5000,
        showRhumbComparison: false
      },
      highlight: {readout: 'worldInside'},
      controls: ['ringDistance', 'worldHub'],
      readouts: ['worldInside']
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
      legends.push({
        kind: 'categories',
        title: 'Aviation haul class',
        entries: ['regional', 'short', 'medium', 'long', 'ultra-long'].map((label, index) => ({
          color: [...(B3_PALETTE[index] ?? B3_PALETTE[0]), 180] as [number, number, number, number],
          label
        }))
      });
    }
    legends.push(
      state.airportColor === 'distance'
        ? {
            kind: 'ramp',
            title: 'Airport distance from the hub',
            ramp: 'ylorbr',
            extent: [0, 15000],
            unit: 'km',
            format: value => `${value.toLocaleString('en-US')} km`
          }
        : {
            kind: 'ramp',
            title: 'Initial bearing from the hub',
            ramp: 'twilight',
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
    if (state.showRhumbComparison) {
      legends.push({
        kind: 'categories',
        title: 'One loaded route',
        entries: [
          {color: [0, 137, 123, 255], label: 'Great circle (shortest)'},
          {color: [230, 159, 0, 255], label: 'Rhumb (constant heading)'}
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
    {id: 'worldInside', label: 'Inside the range ring'},
    {id: 'worldPoleThreshold', label: 'Pole enclosure threshold'},
    {id: 'rhumbPair', label: 'Selected archive pair'},
    {id: 'rhumbLengths', label: 'Great-circle / rhumb length'},
    {id: 'rhumbExcess', label: 'Rhumb excess'},
    {id: 'rhumbHeading', label: 'Heading'}
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
