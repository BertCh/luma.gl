// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {FloodLabOptions} from './flood-lab.compute';

/** A live shallow-water sandbox over the real Grand Canyon elevation grid. */
export default defineScene<FloodLabOptions>({
  id: 'flood-lab',
  title: 'Flood routing over the Grand Canyon DEM',
  chapter: 'hydrology',
  order: 20,
  summary:
    'Uniform rain, infiltration and a fixed release drive an approximately 124 m local-inertial shallow-water screening simulation over the Grand Canyon DEM. Water depth, maximum depth and stored volume come directly from GPU state.',
  contributors: ['FrontierFloodSimulation'],
  datasets: [
    {id: 'grand-canyon-dem', role: 'real 15 m terrain, block-averaged to the solver grid'}
  ],
  initialView: {longitude: -112.1, latitude: 36.1, zoom: 11.2},
  basemap: ground('relief'),
  furniture: {
    title: {title: 'Flood routing', subtitle: 'Approximately 124 m local-inertial grid'},
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
      title: 'Route uniform rainfall across the elevation surface',
      headline: 'Surface gradient controls simulated discharge',
      textAlternative:
        'Simulated water depth follows the block-averaged Grand Canyon elevation gradient on an approximately 124 metre grid under uniform rainfall and infiltration inputs.',
      body: 'The source DEM is block-averaged eight cells at a time; **Solver lattice** reports the resulting ground resolution. `FrontierFloodSimulation` stores depth in cells and discharge on their faces with closed outer boundaries. Blue-gold color is **water depth**; dry cells remain transparent.\n\nChange **Rainfall** and **Infiltration** below. Both update parameter values while the existing GPU state persists. These uniform rates define a screening scenario, not an observed storm or calibrated forecast.',
      controls: ['rainfall', 'infiltration'],
      readouts: ['simulatedTime', 'grid', 'storedWater', 'maximumDepth']
    },
    {
      id: 'release',
      title: 'Add a fixed release near Bright Angel Creek',
      headline: 'The fixed release enters mapped channels',
      textAlternative:
        'A fixed synthetic release near Bright Angel Creek adds water to the depth field; the map shows flow entering terrain-defined channels as stored volume and maximum depth increase.',
      body: 'A fixed release near Bright Angel Creek makes the routing pattern visible before uniform rainfall accumulates. Increase **Release rate** or spread it with **Release radius** below. **Scenario forcing** reports imposed input and the upper bound on infiltration; effective infiltration is limited by available water.',
      controls: ['sourceRate', 'sourceRadius'],
      readouts: ['forcing', 'storedWater', 'maximumDepth']
    },
    {
      id: 'friction',
      title: 'Friction changes timing, not topography',
      headline: 'Roughness reduces simulated discharge',
      textAlternative:
        'Increasing Manning roughness slows and broadens the simulated water front while the elevation surface and closed outer boundary remain fixed.',
      body: '**Manning roughness** below damps fast shallow flow. Low values produce a sharp, quick front; high values leave a broader, slower sheet. **Solver time step** changes how much simulated time each frame advances; the Courant clamp limits any face from moving too much water at once.',
      controls: ['roughness', 'timeStep'],
      readouts: ['simulatedTime']
    },
    {
      id: 'persistent',
      title: 'Pause or reset the persistent simulation state',
      headline: 'Pause preserves the current depth field',
      textAlternative:
        'Pausing preserves GPU depth and face-discharge buffers; resuming continues from that state, while reset clears depth, discharge, elapsed time, and summary diagnostics.',
      body: 'Pause **Run simulation** below: the GPU depth and face-discharge buffers remain unchanged. Resume and the next command buffer continues from that state. A periodic deterministic reduction reads maximum depth and stored volume; deck.gl draws the full depth field without downloading it.\n\nUse **Drain and restart** to clear the experiment, then compare a short high-rate release with a lower sustained release.',
      controls: ['play', 'reset', 'opacity'],
      readouts: ['simulatedTime', 'storedWater', 'maximumDepth']
    }
  ],
  legends: () => [
    {
      kind: 'ramp',
      id: 'depth',
      title: 'Water depth',
      ramp: 'cividis',
      extent: 'gpu',
      unit: 'm'
    }
  ],
  readouts: [
    {id: 'simulatedTime', label: 'Simulated time', format: 'text'},
    {id: 'grid', label: 'Solver lattice', format: 'text'},
    {id: 'forcing', label: 'Scenario forcing', format: 'text'},
    {id: 'maximumDepth', label: 'Maximum depth', format: 'decimal', unit: 'm'},
    {id: 'storedWater', label: 'Stored water', format: 'text'}
  ],
  snippet: state => `const flood = new FrontierFloodSimulation({
  width, height, terrain, parameters, state: {depth, xFlux, yFlux}, display,
  stepsPerEncoding: 3
});
parameters.write(getFrontierFloodParameterValues({
  cellSize, timeStep: ${state.timeStep}, rainfallRate: ${state.rainfall} / 3.6e6
}));`,
  about: {
    what: 'A local-inertial shallow-water approximation on an approximately 124 m grid with persistent depth and face-discharge buffers, uniform forcing and closed outer boundaries. It is a screening simulation, not a calibrated hydraulic forecast.',
    why: 'The workload consists of repeated dependent flux and depth updates over a persistent raster. GPU reductions provide depth and storage diagnostics without transferring the full field.',
    howToRead:
      'Colored cells are simulated water depth over the real DEM. The fixed local release and uniform rainfall are scenarios, not observations.'
  },
  create: async ctx => (await import('./flood-lab.compute')).createFloodLab(ctx)
});
