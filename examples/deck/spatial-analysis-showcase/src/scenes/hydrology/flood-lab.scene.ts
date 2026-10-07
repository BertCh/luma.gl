// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {FloodLabOptions} from './flood-lab.compute';

/** A live shallow-water sandbox over the real Grand Canyon elevation grid. */
export default defineScene<FloodLabOptions>({
  id: 'flood-lab',
  title: 'Flood Lab: can water find the Canyon?',
  chapter: 'hydrology',
  order: 20,
  summary:
    'Rain, infiltration and a movable release drive a persistent shallow-water simulation over the Grand Canyon, with water depth rendered directly from its GPU state.',
  contributors: ['FrontierFloodSimulation'],
  datasets: [
    {id: 'grand-canyon-dem', role: 'real 15 m terrain, block-averaged to the solver grid'}
  ],
  initialView: {longitude: -112.1, latitude: 36.1, zoom: 11.2},
  basemap: ground('relief'),
  furniture: {
    title: {title: 'Flood Lab', subtitle: 'Persistent shallow water over the Grand Canyon DEM'},
    scaleBar: {units: 'metric'},
    credit: 'Terrain Tiles on AWS / USGS 3DEP',
    caveat: 'Interactive screening scenario, not a calibrated flood forecast.'
  },
  options: [
    {
      kind: 'toggle',
      id: 'play',
      label: 'Run simulation',
      group: 'Simulation',
      apply: 'param',
      default: true,
      help: 'Encodes three conservative solver steps into each rendered frame. Pausing preserves all GPU state.'
    },
    {
      kind: 'slider',
      id: 'rainfall',
      label: 'Rainfall',
      group: 'Forcing',
      apply: 'param',
      min: 0,
      max: 120,
      step: 2,
      default: 18,
      unit: 'mm/h',
      help: 'Uniform water added across the terrain. This is a scenario control, not a forecast.'
    },
    {
      kind: 'slider',
      id: 'infiltration',
      label: 'Infiltration',
      group: 'Forcing',
      apply: 'param',
      min: 0,
      max: 40,
      step: 1,
      default: 6,
      unit: 'mm/h',
      help: 'Uniform loss into the ground. Values above rainfall make shallow cells dry out.'
    },
    {
      kind: 'slider',
      id: 'sourceRate',
      label: 'Release rate',
      group: 'Forcing',
      apply: 'param',
      min: 0,
      max: 8,
      step: 0.25,
      default: 2,
      unit: 'mm/s',
      help: 'Adds a localized pulse near Bright Angel Creek to make channel routing visible quickly.'
    },
    {
      kind: 'slider',
      id: 'sourceRadius',
      label: 'Release radius',
      group: 'Forcing',
      apply: 'param',
      min: 1,
      max: 18,
      step: 1,
      default: 5,
      unit: 'cells',
      help: 'Radius of the localized release on the solver lattice.'
    },
    {
      kind: 'slider',
      id: 'roughness',
      label: 'Manning roughness',
      group: 'Physics',
      apply: 'param',
      min: 0.015,
      max: 0.1,
      step: 0.005,
      default: 0.04,
      help: 'Hydraulic friction. Higher values slow and broaden the advancing water.'
    },
    {
      kind: 'slider',
      id: 'timeStep',
      label: 'Solver time step',
      group: 'Physics',
      apply: 'param',
      min: 0.25,
      max: 3,
      step: 0.25,
      default: 1,
      unit: 's',
      help: 'Simulated seconds per solver step. Face discharge is Courant-clamped for stability.'
    },
    {
      kind: 'slider',
      id: 'opacity',
      label: 'Water opacity',
      group: 'Display',
      apply: 'param',
      min: 0.25,
      max: 1,
      step: 0.05,
      default: 0.82,
      help: 'Opacity of the depth surface over the shaded relief.'
    },
    {
      kind: 'button',
      id: 'reset',
      label: 'Drain and restart',
      group: 'Simulation',
      help: 'Clears depth and face flux without rebuilding the graph.'
    }
  ],
  story: [
    {
      id: 'rain',
      title: 'Where does an impossible cloudburst go?',
      headline: 'Water follows the surface, not a cached drainage map',
      body: 'This is the real Grand Canyon elevation window under a deliberately adjustable storm. `FrontierFloodSimulation` stores depth in cells and discharge on their faces, so every drop leaving one cell enters the next. Blue-gold color is **water depth**; dry cells remain transparent.\n\nChange **Rainfall** and **Infiltration** below. Both are four-byte parameter changes, while the water already on the GPU keeps moving.',
      controls: ['rainfall', 'infiltration'],
      readouts: ['simulatedTime']
    },
    {
      id: 'release',
      title: 'Give the solver something dramatic to route',
      headline: 'A localized pulse exposes the drainage structure',
      body: 'A localized release near Bright Angel Creek makes the drainage structure legible in seconds. Increase **Release rate** or spread it with **Release radius** below. The advancing front bends around ridges because its face flux follows the water-surface slope, not a precomputed flow-direction map.',
      controls: ['sourceRate', 'sourceRadius'],
      readouts: ['grid']
    },
    {
      id: 'friction',
      title: 'Friction changes timing, not topography',
      headline: 'Rough ground slows and broadens the front',
      body: '**Manning roughness** below damps fast shallow flow. Low values produce a sharp, quick front; high values leave a broader, slower sheet. **Solver time step** changes how much simulated time each frame advances; the Courant clamp limits any face from moving too much water at once.',
      controls: ['roughness', 'timeStep'],
      readouts: ['simulatedTime']
    },
    {
      id: 'persistent',
      title: 'The map is the state, not a CPU snapshot',
      headline: 'Pause and the exact GPU state stays on screen',
      body: 'Pause **Run simulation** below: the exact GPU buffers remain on screen. Resume and the next command buffer continues from them. There is no depth download in the frame loop, and deck.gl shades the same storage buffer that the solver just wrote.\n\nUse **Drain and restart** to clear the experiment, then try a short violent pulse followed by infiltration.',
      controls: ['play', 'reset', 'opacity'],
      readouts: ['simulatedTime']
    }
  ],
  legends: () => [
    {
      kind: 'ramp',
      title: 'Water depth',
      ramp: 'cividis',
      extent: [0, 2.5],
      unit: 'm',
      labels: ['film', 'deep']
    }
  ],
  readouts: [
    {id: 'simulatedTime', label: 'Simulated time', format: 'text'},
    {id: 'grid', label: 'Solver lattice', format: 'text'}
  ],
  snippet: state => `const flood = new FrontierFloodSimulation({
  width, height, terrain, parameters, state: {depth, xFlux, yFlux}, display,
  stepsPerEncoding: 3
});
parameters.write(getFrontierFloodParameterValues({
  cellSize, timeStep: ${state.timeStep}, rainfallRate: ${state.rainfall} / 3.6e6
}));`,
  about: {
    what: 'A local-inertial shallow-water approximation with persistent depth and face-discharge buffers. It is a screening sandbox, not a calibrated hydraulic forecast.',
    why: 'Flooding makes WebGPU’s advantage tangible: many small dependent updates, persistent state and immediate visual feedback.',
    howToRead:
      'Colored cells are simulated water depth over the real DEM. The fixed local release and uniform rainfall are scenarios, not observations.'
  },
  create: async ctx => (await import('./flood-lab.compute')).createFloodLab(ctx)
});
