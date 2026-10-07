// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getClassTableLegend} from '../../cartography/class-table';
import {CREDITS, joinCredits} from '../../cartography/credits';
import {labelsFor, US, WORLD} from '../../cartography/gazetteer';
import {ground} from '../../cartography/grounds';
import {
  GLOBAL_FURNITURE,
  NATIONAL_FURNITURE,
  mercatorCaveat
} from '../../cartography/projection-notes';
import {defineScene, type LegendSpec} from '../scene';
import type {FlightBundlingOptions, FlightClassStats} from './flight-bundling.compute';
import {
  getDelayTable,
  getDistanceTable,
  getProbeColor,
  GHOST_ALPHA,
  PLAIN_COLOR,
  PROBE_ROUTES,
  UNDER_FLOOR_COLOR
} from './flight-bundling-style';
import {FLOW_CREDITS} from './flows-style';

const WORLD_VIEW = {longitude: 10, latitude: 26, zoom: 1.75};
/** Frames of the two regional steps, as bounds `[west, south, east, north]`. */
const EUROPE_BOUNDS = [-25, 33, 45, 66] as const;
const LOWER_48_BOUNDS = [-125, 24, -66, 50] as const;

/** Airports and waters named on the dark ground (it carries no basemap labels). */
const HUB_NAMES = labelsFor(WORLD, ['atl', 'lhr', 'dxb', 'hnd'], {
  atl: {minZoom: 0},
  lhr: {minZoom: 0},
  dxb: {minZoom: 0},
  hnd: {minZoom: 0}
});
const OCEAN_NAMES = labelsFor(WORLD, ['atlantic-ocean', 'pacific-ocean'], {
  'atlantic-ocean': {tone: 'muted'},
  'pacific-ocean': {tone: 'muted'}
});
const CORRIDOR_ENDS = labelsFor(WORLD, ['lhr', 'dxb', 'hnd'], {
  lhr: {minZoom: 0},
  dxb: {minZoom: 0},
  hnd: {minZoom: 0}
});
const PROBE_ENDS = labelsFor(WORLD, ['jfk', 'lhr', 'sin', 'hnd'], {
  jfk: {minZoom: 0},
  lhr: {minZoom: 0},
  sin: {minZoom: 0},
  hnd: {minZoom: 0}
});
const EUROPE_HUBS = labelsFor(WORLD, ['lhr', 'cdg', 'ams', 'fra', 'ist'], {
  lhr: {minZoom: 0},
  cdg: {minZoom: 0},
  ams: {minZoom: 0},
  fra: {minZoom: 0},
  ist: {minZoom: 0}
});
const US_HUBS = labelsFor(US, ['atl', 'ord', 'dfw', 'den', 'lax', 'jfk'], {
  atl: {minZoom: 0},
  ord: {minZoom: 0},
  dfw: {minZoom: 0},
  den: {minZoom: 0},
  lax: {minZoom: 0},
  jfk: {minZoom: 0}
});

/** The cartouche of one step: the claim, the variable and method, and the vintage chip. */
const cartouche = (title: string, subtitle: string, frozen = true) => ({
  title,
  subtitle,
  ...(frozen ? {chips: ['Frozen 2014'] as const} : {})
});

const WORLD_CREDIT = joinCredits(FLOW_CREDITS.openFlights, CREDITS.naturalEarth);
const US_CREDIT = joinCredits(FLOW_CREDITS.bts, CREDITS.colorBrewer);

const GHOST_COLOR = [
  UNDER_FLOOR_COLOR[0],
  UNDER_FLOOR_COLOR[1],
  UNDER_FLOOR_COLOR[2],
  Math.round(GHOST_ALPHA * 255)
] as const;

export default defineScene<FlightBundlingOptions>({
  id: 'flight-bundling',
  title: 'Untangling the airline network',
  chapter: 'flows',
  order: 7,
  summary:
    'Kernel-density edge bundling on the GPU turns the airline hairball into corridors, and shows what it costs: path stretch, a radius that is relative to the data, and routes that leave the bundle.',
  contributors: ['GPUEdgeBundling'],
  datasets: [
    {id: 'openflights', role: 'world route pairs'},
    {id: 'us-airline-flows', role: 'US flights, July 2023'}
  ],
  initialView: WORLD_VIEW,

  options: [
    {
      kind: 'select',
      id: 'network',
      label: 'Network',
      group: 'Edges',
      apply: 'compile',
      display: 'segmented',
      default: 'world',
      help: 'OpenFlights world routes (frozen in 2014) or scheduled US flights in July 2023 (BTS, merged to pairs). A different edge list rebuilds the graph.',
      options: [
        {value: 'world', label: 'World'},
        {value: 'us', label: 'US'}
      ]
    },
    {
      kind: 'select',
      id: 'region',
      label: 'Region',
      group: 'Edges',
      apply: 'param',
      default: 'all',
      help: 'Keeps pairs with both ends in the region. The edge mask is a per-frame buffer: masked pairs leave the density and the work box without a recompile, so the kernel radius, a fraction of that box, changes in kilometres. “Lower 48” is for the US network.',
      options: [
        {value: 'all', label: 'Everywhere'},
        {value: 'Europe', label: 'Europe'},
        {value: 'Asia', label: 'Asia and Middle East'},
        {value: 'North America', label: 'North America'},
        {value: 'South America', label: 'South America'},
        {value: 'Africa', label: 'Africa'},
        {value: 'Oceania', label: 'Oceania'},
        {value: 'lower48', label: 'Lower 48 states (US network)'}
      ]
    },
    {
      kind: 'range',
      id: 'distanceRange',
      label: 'Route distance',
      group: 'Edges',
      apply: 'param',
      min: 0,
      max: 14000,
      step: 100,
      default: [0, 14000],
      help: 'Keeps pairs whose great-circle length is within the range. Low values leave regional hops; high values leave the intercontinental backbone. Another mask write, so the work box follows.',
      format: value => `${Math.round(value).toLocaleString('en-US')} km`
    },
    {
      kind: 'slider',
      id: 'iterations',
      label: 'Iterations',
      group: 'Bundling',
      apply: 'param',
      min: 0,
      max: 32,
      step: 1,
      default: 15,
      format: value => (value === 0 ? '0 (straight lines)' : String(value)),
      autoSweep: {from: 0, to: 15, durationMs: 7000, ease: 'in-out'},
      help: 'How many advect, resample and smooth rounds run. The GPU gates every round past this count, so cost follows the slider while the compiled maximum stays 32. Play sweeps from straight lines to the default run.'
    },
    {
      kind: 'preset',
      id: 'radiusPreset',
      label: 'Radius presets',
      group: 'Bundling',
      help: 'Three kernel radii as a fraction of the work box side: too small to join routes, a working bundle, and so large that everything merges.',
      presets: [
        {
          label: '0.01',
          values: {kernelRadius: 0.01},
          help: 'Only routes that already run side by side join.'
        },
        {
          label: '0.04',
          values: {kernelRadius: 0.04},
          help: 'Corridors form and individual routes can still be told apart.'
        },
        {
          label: '0.08',
          values: {kernelRadius: 0.08},
          help: 'Everything merges into a few thick trunks: the failing setting.'
        }
      ]
    },
    {
      kind: 'slider',
      id: 'kernelRadius',
      label: 'Kernel radius',
      group: 'Bundling',
      apply: 'param',
      min: 0.005,
      max: 0.1,
      step: 0.005,
      default: 0.03,
      format: value => value.toFixed(3),
      danger: [0.07, 0.1],
      marks: [{value: 0.03, label: 'default'}],
      help: 'Initial attraction radius as a fraction of the work box side, which is computed on the GPU from the live edges. The readouts give it in kilometres. Larger radii merge routes that are farther apart into fewer, thicker bundles.'
    },
    {
      kind: 'slider',
      id: 'decay',
      label: 'Radius decay',
      group: 'Bundling',
      apply: 'param',
      min: 0.5,
      max: 0.9,
      step: 0.01,
      default: 0.85,
      format: value => value.toFixed(2),
      help: 'The radius is multiplied by this every iteration, so after k rounds it is the start radius times decay to the power k. Lower values anneal quickly (coarse bundles, then tight); higher values keep pulling at long range.'
    },
    {
      kind: 'slider',
      id: 'stiffness',
      label: 'Stiffness (smoothing)',
      group: 'Bundling',
      apply: 'param',
      min: 0,
      max: 1,
      step: 0.05,
      default: 0.5,
      format: value => value.toFixed(2),
      help: 'Strength of the Laplacian smoothing after each step. High values give smooth arcs; zero leaves kinks.'
    },
    {
      kind: 'slider',
      id: 'stepScale',
      label: 'Step scale',
      group: 'Bundling',
      apply: 'param',
      min: 0.25,
      max: 2,
      step: 0.05,
      default: 1,
      format: value => value.toFixed(2),
      expert: true,
      help: 'Multiplier on the advection step, which is one kernel radius at 1. Larger steps converge faster but can overshoot.'
    },
    {
      kind: 'select',
      id: 'pointsPerEdge',
      label: 'Control points per edge',
      group: 'Compile-time',
      apply: 'compile',
      default: '16',
      help: 'Polyline resolution of each edge (2 to 64 allowed). More points follow tighter bends but cost memory and splat work. Rebuilds the graph.',
      options: [
        {value: '8', label: '8 (coarse)'},
        {value: '16', label: '16'},
        {value: '24', label: '24'},
        {value: '32', label: '32 (fine)'}
      ]
    },
    {
      kind: 'select',
      id: 'densityResolution',
      label: 'Density grid',
      group: 'Compile-time',
      apply: 'compile',
      default: '256',
      help: 'Cells per axis of the density grid the control points splat onto. Finer grids resolve smaller gaps between bundles. Rebuilds the graph.',
      options: [
        {value: '128', label: '128 × 128'},
        {value: '256', label: '256 × 256'},
        {value: '512', label: '512 × 512'}
      ]
    },
    {
      kind: 'select',
      id: 'colorBy',
      label: 'Colour routes by',
      group: 'Display',
      apply: 'param',
      display: 'segmented',
      default: 'distance',
      help: 'Route length in five fixed classes (cool: corridors, not volume), the mean departure delay of the pair (US network only; falls back to distance on the world network), or one pale colour that shows only where lines pile up.',
      options: [
        {value: 'distance', label: 'Distance'},
        {value: 'delay', label: 'Delay (US)'},
        {value: 'plain', label: 'One colour'}
      ]
    },
    {
      kind: 'slider',
      id: 'minTraffic',
      label: 'Minimum flights',
      group: 'Display',
      apply: 'param',
      min: 1,
      max: 300,
      step: 1,
      default: 1,
      danger: [1, 20],
      marks: [{value: 100, label: '100'}],
      disabledWhen: state => state.colorBy !== 'delay',
      help: 'Delay colours only: pairs with fewer scheduled flights in July are drawn grey. A pair with a handful of flights can show an extreme mean delay from a single bad day, so the worst corridors at a low floor are the smallest pairs.'
    },
    {
      kind: 'slider',
      id: 'brightness',
      label: 'Edge brightness',
      group: 'Display',
      apply: 'param',
      min: 0.25,
      max: 3,
      step: 0.05,
      default: 1,
      help: 'Scales the light of every route. Routes add up on the night ground, so crowded corridors read brighter; lower it to see through the hairball.'
    },
    {
      kind: 'select',
      id: 'straight',
      label: 'Straight routes',
      group: 'Display',
      apply: 'param',
      display: 'segmented',
      default: 'off',
      help: 'Ghost draws the original straight routes faintly under the bundles, to see what bundling changed. Full draws them in the same class colours as the bundles (the straight side of the swipe).',
      options: [
        {value: 'off', label: 'Off'},
        {value: 'ghost', label: 'Ghost'},
        {value: 'full', label: 'Full'}
      ]
    },
    {
      kind: 'toggle',
      id: 'probes',
      label: 'Probe routes',
      group: 'Display',
      apply: 'param',
      default: false,
      disabledWhen: state => state.network !== 'world',
      help: 'Draws three routes in flat colours, bundled and as a dashed straight chord, and dims every other route: you see them leave the bundle and rejoin it.'
    },
    {
      kind: 'toggle',
      id: 'corridorLabels',
      label: 'Name the corridors',
      group: 'Display',
      apply: 'param',
      default: false,
      disabledWhen: state => state.network !== 'world',
      help: 'Names three corridors on the middle of the bundled path of a representative route (read back from the GPU).'
    },
    {
      kind: 'toggle',
      id: 'showAirports',
      label: 'Airports and hubs',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'White rings mark the 24 best-connected airports; other airports appear from zoom 3. Hover an airport for its name and degree.'
    }
  ],

  readouts: [
    {
      id: 'routes',
      label: 'Route pairs',
      format: 'integer',
      help: 'Airport pairs that pass the region and distance filters.'
    },
    {id: 'airports', label: 'Airports', format: 'integer'},
    {
      id: 'stretch',
      label: 'Mean path stretch',
      emphasis: 'tile',
      help: 'Total length of the bundled polylines divided by the total length of the straight edges. Always at least 1: bundling trades length for grouping. It is the honest cost of the display.'
    },
    {
      id: 'stretchChart',
      label: 'How much longer is a bundled route?',
      kind: 'chart',
      help: 'Bundled path length divided by the straight length for every live route longer than 50 km. Updates a moment after the controls settle.'
    },
    {
      id: 'radiusKm',
      label: 'Kernel radius on the ground',
      help: 'The starting kernel radius (a fraction of the work box side) times the side of the work box around the live edges, in kilometres. A CPU mirror of the box the GPU computes.'
    },
    {
      id: 'radiusKmWorld',
      label: 'Same setting on the whole network',
      help: 'The same fraction times the work box of every route that passes the distance filter, ignoring the region.'
    },
    {
      id: 'decayChart',
      label: 'Radius by iteration',
      kind: 'chart',
      help: 'The annealing schedule: radius times decay to the power k. The marker is the active iteration; click or drag the chart to set it.'
    },
    {
      id: 'pairsShown',
      label: 'Pairs at or above the floor',
      help: 'US pairs with at least the minimum number of flights, of the pairs that pass the filters.'
    },
    {
      id: 'worstCorridor',
      label: 'Worst corridor above the floor',
      help: 'The pair with the highest mean departure delay among pairs at or above the floor. A mean, not a median, so the pair name changes as the floor rises.'
    },
    {
      id: 'topClassShare',
      label: 'Flights in the slowest class',
      format: 'percent',
      help: 'Share of the flights on pairs at or above the floor that sit on pairs in the top delay class.'
    },
    {
      id: 'edges',
      label: 'Edges live',
      hood: true,
      help: 'Pairs that pass the filters, of all pairs.'
    },
    {
      id: 'controlPoints',
      label: 'Control points',
      format: 'integer',
      hood: true,
      help: 'Live edges (split at the antimeridian) times control points per edge.'
    },
    {id: 'pointsPerEdge', label: 'Control points per edge', format: 'integer', hood: true},
    {
      id: 'boxKm',
      label: 'Work box side',
      hood: true,
      help: 'Side of the square box the density grid covers, from the live edges, padded.'
    },
    {
      id: 'radiusNowKm',
      label: 'Radius at the active iteration',
      hood: true,
      help: 'Start radius times decay to the power of the iterations run.'
    }
  ],

  pipeline: [
    {
      id: 'box',
      label: 'Work box',
      detail: 'A square box around the live edges, computed on the GPU'
    },
    {
      id: 'splat',
      label: 'Density splat',
      detail: 'Epanechnikov weights of every control point into an atomic grid'
    },
    {
      id: 'advect',
      label: 'Advect',
      detail: 'Each interior point moves one radius up the normalised density gradient'
    },
    {
      id: 'resample',
      label: 'Resample',
      detail: 'Equal arc length, then one Laplacian smoothing pass; endpoints stay pinned'
    },
    {
      id: 'anneal',
      label: 'Anneal',
      detail: 'The radius is multiplied by the decay, so corridors tighten'
    }
  ],

  legends: (state, data) => getLegends(state, data.classStats as FlightClassStats | undefined),

  // Night ground: bundles are light. The straight routes are drawn in lon/lat, not great circles.
  basemap: ground('night'),
  furniture: {
    ...GLOBAL_FURNITURE,
    title: cartouche(
      'Which corridors do airlines share?',
      'Route pairs, straight in longitude and latitude'
    ),
    credit: WORLD_CREDIT,
    caveat: mercatorCaveat({latitudes: [0, 60], kind: 'area'})
  },
  annotations: [],

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUEdgeBundling,
  GPU_EDGE_BUNDLING_WORK_BOX_PADDING,
  createGPUEdgeBundlingParameterValues
} from '@luma.gl/experimental/gpu-network';

const graph = new GPUCommandGraph(device, {id: 'bundling'});
graph.add(
  new GPUEdgeBundling({
    positions,                         // float32x2 [lon, lat] per airport (plus antimeridian copies)
    sourceVertices, targetVertices,    // uint32 airport per edge end
    edgeMask,                          // uint32 per edge: the region and distance filters
    geographic: true,                  // lon/lat degrees: longitude scaled by cos(mid-latitude)
    pointsPerEdge: ${state.pointsPerEdge},
    iterations: 32,                    // compile-time maximum
    densityResolution: ${state.densityResolution},
    parameters: parameters.importToGraph(graph),
    paths                              // float32x2 [edge * pointsPerEdge + point]
  })
);
const compiled = graph.compile();      // once
// when a slider moves: write the parameter buffer, then encode again
parameters.write(createGPUEdgeBundlingParameterValues({
  activeIterations: ${state.iterations},
  kernelRadius: ${state.kernelRadius},   // fraction of the work box side
  lambda: ${state.decay},
  smoothing: ${state.stiffness},
  stepScale: ${state.stepScale}
}, 'uint32'));
compiled.encode(commandEncoder, {parameters: undefined});
// radius in km: kernelRadius * side of the box around the live edges, where the box is
// max((maxLon - minLon) * cos(midLat), maxLat - minLat) * (1 + 2 * GPU_EDGE_BUNDLING_WORK_BOX_PADDING)
// draw: quads read paths[instance / (points - 1) * points + instance % (points - 1)], coloured by class`,

  about: {
    what: 'Previously: who holds the airline network together (airline network). Next: the same pairs as a matrix (flight matrix).\n\n`GPUEdgeBundling` is kernel-density edge bundling (KDEEB, Hurter, Ersoy and Telea) as compute: every edge becomes a polyline whose control points are repeatedly pulled up the gradient of a control-point density field, resampled to even spacing and smoothed, with the endpoints pinned. The density grid covers a square work box computed on the GPU from the live edges, so the kernel radius is a fraction of that box.',
    why: 'Origin-destination data has far more edges than a map can show. Bundling turns “everything crosses everything” into corridors, which is how you see where flows concentrate. It is a display transformation: it generalises the way a cartographer does, and it invents corridor positions. Know when not to bundle: to compare pairs, use a matrix.',
    howToRead:
      'Cool, dim lines are short routes; bright ones are long. Where many lines run together the light adds up. A bundle tells you that routes share a corridor, not which way any route flies: its curvature carries no meaning. White rings are the best-connected airports. Caveats: OpenFlights is a community snapshot frozen in 2014 (routes, not seats or schedules); the US data is scheduled flights of the largest carriers in July 2023 from the BTS On-Time file, merged to pairs; the radius is relative to the data you feed in; float32 chaos means bundles agree statistically, not bit for bit, between GPUs; and edges are straight in longitude and latitude and bundled in that space, not on the sphere (great-circle initial subdivision would be a library addition).'
  },

  create: async ctx => (await import('./flight-bundling.compute')).createFlightBundling(ctx),

  story: [
    {
      id: 'hairball',
      title: 'Straight lines hide the corridors',
      headline: 'Straight lines hide the corridors',
      textAlternative:
        'Dark world map covered by thousands of thin pale straight route lines that glow where they overlap, with four large airports and two oceans named.',
      body: 'OpenFlights lists **{{routes}}** airport pairs between **{{airports}}** airports. Each is one straight line in longitude and latitude, in pale light that adds up where lines pile up. Pull **Edge brightness** below up or down: the overlap hides where traffic actually goes.\n\nWhich corridors do the world’s airlines share?',
      optionsMode: 'fresh',
      options: {network: 'world', iterations: 0, colorBy: 'plain', brightness: 1},
      controls: ['brightness'],
      readouts: ['routes', 'airports'],
      stage: 'box',
      camera: {...WORLD_VIEW, transitionMs: 1400},
      furniture: {
        title: cartouche(
          'Straight lines hide the corridors',
          'Route pairs, straight in longitude and latitude'
        )
      },
      annotations: [
        ...HUB_NAMES,
        ...OCEAN_NAMES,
        {
          kind: 'note',
          coordinate: WORLD.places['tasman-sea'].lngLat,
          title: 'Straight in longitude and latitude',
          text: 'Not great circles: the map is Web Mercator.'
        }
      ],
      highlight: {readout: 'routes'}
    },
    {
      id: 'bundle',
      title: 'Density pulls routes into corridors',
      headline: 'Density pulls routes into corridors',
      textAlternative:
        'The same world map after bundling: routes gather into bright cool corridors across the North Atlantic, from Europe to the Gulf and across the Pacific, named on the map.',
      body: '`GPUEdgeBundling` splats the density of every control point onto a grid, moves each point one kernel radius up the density gradient, then resamples and smooths the line. The radius shrinks by **Radius decay** every pass, so coarse corridors form first. Press play on **Iterations** below and watch the marker ride the curve; the mean path stretch ends at **{{stretch}}**. Colour is route length: cool and dim for short hops.',
      optionsMode: 'fresh',
      options: {network: 'world', iterations: 15, colorBy: 'distance', corridorLabels: true},
      controls: ['iterations', 'decay'],
      readouts: ['radiusKm', 'decayChart', 'stretch'],
      stage: 'advect',
      camera: {...WORLD_VIEW, transitionMs: 1400},
      furniture: {
        title: cartouche(
          'Density pulls routes into corridors',
          'Kernel-density bundling, colour is route length'
        )
      },
      annotations: CORRIDOR_ENDS,
      highlight: {readout: 'stretch'}
    },
    {
      id: 'radius',
      title: 'Bundles show grouping, not routes',
      headline: 'Bundles show grouping, not routes',
      textAlternative:
        'World map with three flat-coloured routes drawn bundled and as dashed straight chords over a faint straight-route ghost; the bundled routes bow away from their chords.',
      body: 'Bundling groups routes; it does not draw them. The three coloured routes are drawn bundled and, dashed, straight, over a faint ghost of the straight routes. Pick a **Kernel radius** preset below: the smallest barely bundles, the largest merges everything into a few trunks, and the mean path stretch is the price, now **{{stretch}}** at **{{radiusKm}}**.\n\n*A bundle is a summary, not a flight path.*',
      optionsMode: 'fresh',
      options: {
        network: 'world',
        iterations: 15,
        colorBy: 'distance',
        kernelRadius: 0.04,
        straight: 'ghost',
        probes: true
      },
      controls: ['radiusPreset', 'kernelRadius'],
      readouts: ['stretch', 'stretchChart', 'radiusKm'],
      stage: 'anneal',
      camera: {...WORLD_VIEW, transitionMs: 1400},
      furniture: {
        title: cartouche('Bundles show grouping, not routes', 'Mean path stretch by kernel radius')
      },
      annotations: PROBE_ENDS,
      highlight: {readout: 'stretch'}
    },
    {
      id: 'europe',
      title: 'A region is bundled at its own scale',
      headline: 'A region is bundled at its own scale',
      textAlternative:
        'Map of Europe with bundled routes between European airports, five hubs named, and a scale bar with a tick at the kernel radius.',
      body: 'Filtering to Europe runs the bundling again on **{{routes}}** routes. The radius is a fraction of the work box around the live edges, so the same setting is **{{radiusKmWorld}}** across the whole network but **{{radiusKm}}** here, the tick on the scale bar. Change **Region** below to see the box, and the corridors, rescale.\n\n*Scale is relative to the data you feed in.*',
      optionsMode: 'fresh',
      options: {network: 'world', region: 'Europe', iterations: 15, colorBy: 'distance'},
      controls: ['region'],
      readouts: ['radiusKmWorld', 'radiusKm', 'routes'],
      stage: 'box',
      camera: {bounds: EUROPE_BOUNDS, transitionMs: 1600},
      furniture: {
        scaleBar: {units: 'metric', minZoom: 3},
        title: cartouche(
          'A region is bundled at its own scale',
          'Radius as a fraction of the work box'
        )
      },
      annotations: EUROPE_HUBS,
      highlight: {readout: 'radiusKm'}
    },
    {
      id: 'us-delay',
      title: 'Small pairs make the worst corridors look random',
      headline: 'Small pairs make delay look random',
      textAlternative:
        'Map of the contiguous United States with bundled routes between airports coloured in five classes of mean departure delay from pale to dark red, pairs with too few flights in faint grey, and six hubs named.',
      body: 'Colour is each pair’s mean departure delay for the month: a mean, not a median, so one bad day moves it. The worst pair now is **{{worstCorridor}}**, among **{{pairsShown}}** pairs shown. Drag **Minimum flights** up, or take [a hundred flights](action:set?minTraffic=100), and watch it change; pairs under the floor fade to grey.\n\n*Small numbers make the extremes.*',
      optionsMode: 'fresh',
      options: {
        network: 'us',
        region: 'lower48',
        iterations: 15,
        kernelRadius: 0.02,
        colorBy: 'delay',
        minTraffic: 1
      },
      controls: ['minTraffic'],
      readouts: ['pairsShown', 'worstCorridor', 'topClassShare'],
      stage: 'splat',
      camera: {bounds: LOWER_48_BOUNDS, transitionMs: 1600},
      furniture: {
        ...NATIONAL_FURNITURE,
        title: cartouche(
          'Small pairs make delay look random',
          'Mean departure delay per airport pair',
          false
        ),
        credit: US_CREDIT
      },
      annotations: US_HUBS,
      highlight: {readout: 'worstCorridor'}
    },
    {
      id: 'when-not-to',
      title: 'Straight and bundled tell different truths',
      headline: 'Straight and bundled tell different truths',
      textAlternative:
        'World map split by a vertical divider: straight route lines on the left, bundled corridors on the right, in the same distance colours.',
      body: 'Drag the divider. Straight lines keep every route and its length but hide the corridors; bundles show the corridors and hide the routes, at a mean stretch of **{{stretch}}**. To compare origin and destination pairs without drawing lines at all, use the [flight matrix](#/story/flight-matrix). Explore **Kernel radius**, **Iterations** and **Route distance** below.\n\n*Bundling is a display transformation, not an analysis.*',
      optionsMode: 'fresh',
      options: {network: 'world', iterations: 15, colorBy: 'distance', straight: 'full'},
      controls: ['kernelRadius', 'iterations', 'distanceRange'],
      readouts: ['stretch', 'routes'],
      stage: 'resample',
      compare: {mode: 'swipe', labels: ['Straight', 'Bundled'], position: 0.5},
      camera: {...WORLD_VIEW, transitionMs: 1400},
      furniture: {
        title: cartouche('Straight and bundled tell different truths', 'Same routes, two displays')
      },
      annotations: OCEAN_NAMES,
      highlight: {readout: 'stretch'}
    }
  ]
});

function getLegends(
  state: FlightBundlingOptions,
  stats: FlightClassStats | undefined
): LegendSpec[] {
  const legends: LegendSpec[] = [];
  const isUs = state.network === 'us';
  if (state.colorBy === 'plain') {
    legends.push({
      kind: 'categories',
      title: 'Routes',
      entries: [{color: PLAIN_COLOR, label: 'Airline route pair', shape: 'line'}],
      note: 'Pale additive light: it brightens where routes overlap.'
    });
  } else if (state.colorBy === 'delay' && isUs) {
    legends.push(
      getClassTableLegend(getDelayTable(state.minTraffic, stats?.underFloor), {
        title: 'Mean departure delay',
        id: 'flight-delay',
        basis: 'per airport pair',
        counts: stats?.counts,
        layout: 'list'
      })
    );
  } else {
    legends.push(
      getClassTableLegend(getDistanceTable(), {
        title: 'Route distance',
        id: 'flight-distance',
        counts: stats?.counts,
        layout: 'list',
        note: 'Cool means corridors of grouped routes; the chapter’s gold means volume.'
      })
    );
  }
  const overlays: Extract<LegendSpec, {kind: 'categories'}>['entries'][number][] = [
    {color: [255, 255, 255, 255], label: 'Top-24 hub airport', shape: 'ring'}
  ];
  if (state.straight === 'ghost') {
    overlays.push({color: GHOST_COLOR, label: 'Straight route (ghost)', shape: 'line'});
  }
  if (state.probes && !isUs) {
    const names = ['New York to London', 'New York to Tokyo', 'London to Singapore'];
    PROBE_ROUTES.forEach((probe, index) => {
      overlays.push({
        color: getProbeColor(probe),
        label: `${names[index]}: bundled and dashed straight`,
        shape: 'line'
      });
    });
  }
  legends.push({kind: 'categories', title: 'Overlays', entries: overlays});
  return legends;
}
