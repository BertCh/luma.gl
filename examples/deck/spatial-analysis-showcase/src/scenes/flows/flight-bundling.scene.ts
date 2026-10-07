// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {playbackOptions} from '../../engine/playback';
import {defineScene} from '../scene';
import {storyFromMarkdown} from '../story-markdown';
import narrative from './flight-bundling.md?raw';
import type {FlightBundlingOptions} from './flight-bundling.compute';

const VALUE_TITLES: Record<string, {title: string; unit: string; sqrt: boolean}> = {
  distance: {title: 'Route distance', unit: 'km', sqrt: false},
  traffic: {title: 'Route records on the pair', unit: 'records', sqrt: true},
  airlines: {title: 'Airlines serving the pair', unit: 'airlines', sqrt: false},
  delay: {title: 'Mean departure delay', unit: 'min', sqrt: false},
  cancel: {title: 'Cancelled flights', unit: 'share', sqrt: false}
};

export default defineScene<FlightBundlingOptions>({
  id: 'flight-bundling',
  title: 'Untangling the airline network',
  chapter: 'flows',
  order: 20,
  summary:
    'Kernel-density edge bundling on the GPU: 18,930 world airline routes (and July 2023 US flights) pulled into readable corridors, with iterations, kernel radius and stiffness as live parameters.',
  contributors: ['GPUEdgeBundling'],
  datasets: [
    {id: 'openflights', role: 'world route pairs'},
    {id: 'us-airline-flows', role: 'US flights, July 2023'}
  ],
  initialView: {longitude: 10, latitude: 26, zoom: 1.75},

  options: [
    {
      kind: 'select',
      id: 'network',
      label: 'Network',
      group: 'Edges',
      apply: 'compile',
      default: 'world',
      help: 'OpenFlights world routes (about 2014) or scheduled US flights in July 2023. A different edge list rebuilds the graph.',
      options: [
        {value: 'world', label: 'World routes (OpenFlights, 18,930 pairs)'},
        {value: 'us', label: 'US flights, July 2023 (BTS, merged pairs)'}
      ]
    },
    {
      kind: 'select',
      id: 'region',
      label: 'Region',
      group: 'Edges',
      apply: 'param',
      default: 'all',
      help: 'Keeps edges with both endpoints in the region. The edge mask is a per-frame buffer: masked edges leave the density and the work box without a recompile. “Lower 48” is for the US network.',
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
      help: 'Keeps edges whose great-circle length is within the range. Low values leave regional hops; high values leave the intercontinental backbone.',
      format: value => `${Math.round(value).toLocaleString('en-US')} km`
    },
    {
      kind: 'slider',
      id: 'minTraffic',
      label: 'Minimum traffic',
      group: 'Edges',
      apply: 'param',
      min: 1,
      max: 60,
      step: 1,
      default: 1,
      help: 'Keeps pairs with at least this many airline-route records (world) or scheduled flights per month (US).'
    },
    ...playbackOptions<FlightBundlingOptions>({
      ids: {play: 'play', time: 'iterations', speed: 'playSpeed', loop: 'loop'},
      group: 'Bundling',
      playing: false,
      time: {
        min: 0,
        max: 32,
        step: 1,
        default: 15,
        label: 'Iterations',
        format: value => (value === 0 ? '0 (straight lines)' : String(value)),
        help: 'How many advect, resample and smooth rounds run. The GPU gates every round past this count, so cost follows the slider while the compiled maximum stays 32. Play sweeps it from straight lines to the full run: watch the corridors form.'
      },
      speed: {
        min: 1,
        max: 8,
        step: 1,
        default: 3,
        unit: 'it/s',
        label: 'Play speed',
        help: 'Iterations per second. 3 it/s takes about ten seconds from straight lines to 32 iterations.'
      },
      loop: true
    }),
    {
      kind: 'slider',
      id: 'kernelRadius',
      label: 'Kernel radius',
      group: 'Bundling',
      apply: 'param',
      min: 0.005,
      max: 0.1,
      step: 0.005,
      default: 0.015,
      format: value => value.toFixed(3),
      help: 'Initial attraction radius as a fraction of the work box side. Larger radii merge edges that are farther apart into fewer, thicker bundles.'
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
      help: 'The radius is multiplied by this every iteration. Lower values anneal quickly (coarse bundles, then tight); higher values keep pulling at long range.'
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
      label: 'Colour edges by',
      group: 'Display',
      apply: 'param',
      default: 'distance',
      help: 'Edge attribute mapped through the ramp. Delay and cancellation exist for the US network; airlines for the world network; other combinations fall back to distance.',
      options: [
        {value: 'distance', label: 'Route distance'},
        {value: 'traffic', label: 'Traffic (records or flights)'},
        {value: 'airlines', label: 'Airlines on the pair (world)'},
        {value: 'delay', label: 'Mean departure delay (US)'},
        {value: 'cancel', label: 'Cancellation rate (US)'},
        {value: 'plain', label: 'Single colour'}
      ]
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Colour ramp',
      group: 'Display',
      apply: 'param',
      default: 'viridis',
      disabledWhen: state => state.colorBy === 'plain',
      help: 'Viridis and cividis stay legible on both basemaps; magma and inferno fade to black at the low end.',
      options: [
        {value: 'viridis', label: 'Viridis'},
        {value: 'cividis', label: 'Cividis'},
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'}
      ]
    },
    {
      kind: 'slider',
      id: 'opacity',
      label: 'Edge opacity',
      group: 'Display',
      apply: 'param',
      min: 0.05,
      max: 1,
      step: 0.05,
      default: 0.4,
      help: 'Lines are 1 px and translucent, so they add up, so crowded bundles read brighter. Lower it for the straight hairball.'
    },
    {
      kind: 'toggle',
      id: 'showStraight',
      label: 'Straight edges (ghost)',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Draws the original straight edges faintly under the bundles, to see what bundling changed.'
    },
    {
      kind: 'toggle',
      id: 'showAirports',
      label: 'Airports and hubs',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Dots for every airport; orange dots mark the 24 airports with the most connections. Hover an airport for its name and degree.'
    }
  ],

  readouts: [
    {
      id: 'distanceChart',
      label: 'How long are the routes?',
      kind: 'chart',
      help: 'Length of the routes that pass the region, distance and traffic filters.'
    },
    {
      id: 'stretchChart',
      label: 'How much longer is a bundled route?',
      kind: 'chart',
      help: 'Bundled path length divided by the straight length for every live route longer than 50 km. Updates a moment after the controls settle.'
    },
    {id: 'airports', label: 'Airports', format: 'integer'},
    {
      id: 'edges',
      label: 'Edges live',
      help: 'Airport pairs that pass the region, distance and traffic filters.'
    },
    {
      id: 'controlPoints',
      label: 'Control points',
      format: 'integer',
      help: 'Live edges (split at the antimeridian) times control points per edge.'
    },
    {id: 'pointsPerEdge', label: 'Control points per edge', format: 'integer'},
    {
      id: 'stretch',
      label: 'Mean path stretch',
      format: 'decimal',
      help: 'Total length of the bundled polylines divided by the total length of the straight edges. Always at least 1: bundling trades length for grouping.'
    },
    {
      id: 'coverageStraight',
      label: 'Map cells: straight',
      format: 'integer',
      help: 'Half-degree cells crossed by the straight edges.'
    },
    {
      id: 'coverageBundled',
      label: 'Map cells: bundled',
      format: 'integer',
      help: 'Half-degree cells crossed by the bundled polylines. Fewer cells means less of the map is covered by lines.'
    },
    {
      id: 'inkSaved',
      label: 'Map coverage removed',
      format: 'percent',
      help: 'One minus bundled cells over straight cells.'
    }
  ],

  legends: state => {
    const entries: ReturnType<typeof legendsFor> = legendsFor(state);
    return entries;
  },

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUEdgeBundling,
  createGPUEdgeBundlingParameterValues
} from '@luma.gl/experimental/gpu-network';

const graph = new GPUCommandGraph(device, {id: 'bundling'});
graph.add(
  new GPUEdgeBundling({
    positions,                         // float32x2 [lon, lat] per airport
    sourceVertices, targetVertices,    // uint32 airport per edge end
    edgeMask,                          // uint32 per edge: filters rewrite this buffer
    geographic: true,                  // lon/lat degrees: longitude scaled by cos(latitude)
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
  kernelRadius: ${state.kernelRadius},
  lambda: ${state.decay},
  smoothing: ${state.stiffness},
  stepScale: ${state.stepScale}
}, 'uint32'));
compiled.encode(commandEncoder, {parameters: undefined});
// draw: a line strip per edge reading paths[instance * pointsPerEdge + vertex]`,

  about: {
    what: '`GPUEdgeBundling` is kernel-density edge bundling (KDEEB) as compute: every edge becomes a polyline whose control points are repeatedly pulled along the gradient of a control-point density field, resampled to even spacing and smoothed, with the endpoints pinned.',
    why: 'Origin-destination data has far more edges than a map can show. Bundling turns “everything crosses everything” into corridors, which is how you see where flows concentrate and which places act as junctions.',
    howToRead:
      'Where many lines run together the colour is brighter and the line is thicker. A bundle tells you that those edges share a corridor, not that they follow a physical route. Orange dots are the best-connected airports.'
  },

  create: async ctx => (await import('./flight-bundling.compute')).createFlightBundling(ctx),

  story: storyFromMarkdown<FlightBundlingOptions>(narrative, {
    hairball: {
      controls: ['iterations', 'colorBy', 'showAirports'],
      readouts: ['edges', 'distanceChart'],
      camera: {longitude: 10, latitude: 26, zoom: 1.75, transitionMs: 1200},
      options: {network: 'world', iterations: 0, opacity: 0.25},
      callout: {coordinate: [-84.43, 33.64], text: 'Atlanta (ATL)'},
      highlight: {readout: 'edges'}
    },
    bundle: {
      controls: ['play', 'iterations', 'playSpeed'],
      readouts: ['stretch', 'stretchChart'],
      options: {iterations: 15, opacity: 0.4},
      highlight: {readout: 'stretch'}
    },
    kernel: {
      controls: ['kernelRadius', 'decay'],
      readouts: ['stretch', 'coverageBundled', 'inkSaved', 'stretchChart'],
      options: {kernelRadius: 0.04},
      highlight: {readout: 'inkSaved'}
    },
    europe: {
      controls: ['region'],
      readouts: ['edges'],
      camera: {longitude: 14, latitude: 49, zoom: 3.6},
      options: {region: 'Europe'},
      callout: {coordinate: [8.57, 50.03], text: 'Frankfurt (FRA): 244 airports'}
    },
    'long-haul': {
      controls: ['distanceRange', 'minTraffic'],
      readouts: ['edges', 'distanceChart'],
      camera: {longitude: 20, latitude: 28, zoom: 1.75},
      options: {region: 'all', distanceRange: [3000, 14000]}
    },
    'us-delay': {
      controls: ['network', 'colorBy', 'region'],
      readouts: ['edges'],
      camera: {longitude: -96, latitude: 38.5, zoom: 3.3},
      options: {
        network: 'us',
        region: 'lower48',
        distanceRange: [0, 14000],
        colorBy: 'delay',
        opacity: 0.55,
        kernelRadius: 0.02
      },
      callout: {coordinate: [-84.43, 33.64], text: 'Atlanta: 59,834 flights'}
    },
    limits: {
      controls: ['iterations', 'stiffness', 'densityResolution', 'colorBy', 'showStraight'],
      camera: {longitude: 10, latitude: 26, zoom: 1.75},
      options: {
        network: 'world',
        region: 'all',
        distanceRange: [0, 14000],
        colorBy: 'distance',
        opacity: 0.4,
        kernelRadius: 0.04
      }
    }
  })
});

function legendsFor(state: FlightBundlingOptions) {
  const info = VALUE_TITLES[state.colorBy];
  if (state.colorBy === 'plain' || !info) {
    return [
      {
        kind: 'categories' as const,
        title: 'Edges',
        entries: [
          {color: [96, 214, 255, 255] as const, label: 'Airline route pair'},
          {color: [255, 184, 64, 255] as const, label: 'Top-24 hub airport'}
        ]
      }
    ];
  }
  return [
    {
      kind: 'ramp' as const,
      id: 'edge-value',
      title: info.title,
      ramp: state.ramp,
      extent: 'gpu' as const,
      sqrtScale: info.sqrt,
      unit: info.unit,
      format: (value: number) =>
        state.colorBy === 'cancel'
          ? `${(value * 100).toFixed(0)}%`
          : value >= 100
            ? Math.round(value).toLocaleString('en-US')
            : value.toFixed(value < 10 ? 1 : 0)
    },
    {
      kind: 'categories' as const,
      title: 'Airports',
      entries: [{color: [255, 184, 64, 255] as const, label: 'Top-24 hub airport'}]
    }
  ];
}
