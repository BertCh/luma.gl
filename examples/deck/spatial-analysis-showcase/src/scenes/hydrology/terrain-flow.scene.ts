// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene, type LegendSpec} from '../scene';
import type {TerrainFlowOptions} from './terrain-flow.compute';

const BEARING_NAMES = [
  'north',
  'north-east',
  'east',
  'south-east',
  'south',
  'south-west',
  'west',
  'north-west'
];
const describeBearing = (bearing: number): string =>
  `${bearing}° (${BEARING_NAMES[Math.round(bearing / 45) % 8]})`;

/**
 * Wind over the Grand Canyon: metadata, options and narrative (light); the GPU work lives in
 * `terrain-flow.compute.ts`.
 */
export default defineScene<TerrainFlowOptions>({
  id: 'terrain-flow',
  title: 'Grand Canyon terrain-adjusted wind speed',
  chapter: 'hydrology',
  order: 2,
  summary:
    'GPU terrain projection and particle advection apply a specified wind to the Grand Canyon elevation grid. The outputs are relative speed, shelter fractions and trails; the kinematic model omits pressure, buoyancy and mass conservation.',
  contributors: ['GPUTerrainFlowField', 'GPUParticleAdvection', 'GPUTerrainDerivatives'],
  datasets: [{id: 'grand-canyon-dem', role: 'elevation under the wind'}],
  initialView: {longitude: -112.1, latitude: 36.1, zoom: 11.2},
  basemap: ground('relief'),
  furniture: {
    title: {
      title: 'Grand Canyon terrain-adjusted wind',
      subtitle: 'Relative field speed, shelter and particle trajectories'
    },
    scaleBar: {units: 'metric'},
    credit: 'US Geological Survey (public domain)',
    caveat: 'Kinematic terrain steering is not an atmospheric forecast.'
  },

  options: [
    {
      kind: 'slider',
      id: 'windSpeed',
      label: 'Free-stream wind speed',
      group: 'Wind',
      apply: 'param',
      min: 1,
      max: 30,
      step: 1,
      default: 14,
      unit: 'm/s',
      help: 'Speed of the uniform wind before terrain deflects it. 14 m/s is a strong breeze (50 km/h). Per-frame parameter of the field kernel.'
    },
    {
      kind: 'slider',
      id: 'windBearing',
      label: 'Wind blows toward',
      group: 'Wind',
      apply: 'param',
      min: 0,
      max: 355,
      step: 5,
      default: 75,
      format: describeBearing,
      help: 'Direction the air moves, clockwise from north. 75° is a west-south-westerly wind, typical of a winter front crossing the Colorado Plateau.'
    },
    {
      kind: 'slider',
      id: 'exaggeration',
      label: 'Terrain deflection',
      group: 'Wind',
      apply: 'param',
      min: 1,
      max: 20,
      step: 1,
      default: 8,
      format: value => `${value}×`,
      help: 'Multiplies the surface gradient before the wind is projected onto the tangent plane. 1 is the real terrain; larger values make the ground a stiffer obstacle, as air close to the ground behaves in stable conditions.'
    },
    {
      kind: 'select',
      id: 'background',
      label: 'Backdrop',
      group: 'Display',
      apply: 'param',
      default: 'relief',
      help: 'Shaded relief only, or the wind speed of the deflected field as a share of the free-stream speed.',
      options: [
        {value: 'relief', label: 'Shaded relief'},
        {value: 'speed', label: 'Field speed (share of free stream)'}
      ]
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Trail color ramp',
      group: 'Display',
      apply: 'param',
      default: 'isolum',
      help: 'Color of a trail by the speed of its particle.',
      options: [
        {value: 'isolum', label: 'Isoluminant terrain overlay'},
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'cividis', label: 'Cividis'}
      ]
    },
    {
      kind: 'slider',
      id: 'trailWidth',
      label: 'Trail width',
      group: 'Display',
      apply: 'param',
      min: 0.8,
      max: 3.5,
      step: 0.1,
      default: 1.5,
      unit: 'px',
      help: 'Screen width of every particle trail.'
    },
    {
      kind: 'slider',
      id: 'trailOpacity',
      label: 'Trail opacity',
      group: 'Display',
      apply: 'param',
      min: 0.2,
      max: 1,
      step: 0.05,
      default: 0.9,
      help: 'Opacity at the head of a trail; the tail fades with age.'
    },
    {
      kind: 'slider',
      id: 'timeStep',
      label: 'Time step',
      group: 'Particles',
      apply: 'param',
      min: 1,
      max: 20,
      step: 1,
      default: 8,
      unit: 's / frame',
      help: 'Simulated seconds per rendered frame. Higher values move particles further each frame and lengthen the trails in space.'
    },
    {
      kind: 'slider',
      id: 'speedScale',
      label: 'Playback speed',
      group: 'Particles',
      apply: 'param',
      min: 0.25,
      max: 4,
      step: 0.25,
      default: 1,
      format: value => `${value}×`,
      help: 'Multiplier on the time step, like a video playback rate.'
    },
    {
      kind: 'slider',
      id: 'dropRate',
      label: 'Random respawn',
      group: 'Particles',
      apply: 'param',
      min: 0,
      max: 0.05,
      step: 0.002,
      default: 0.008,
      format: value => `${(value * 100).toFixed(1)} % per frame`,
      help: 'Chance per frame that a particle jumps to a new random position, which keeps the sky filled evenly.'
    },
    {
      kind: 'slider',
      id: 'maximumAge',
      label: 'Maximum age',
      group: 'Particles',
      apply: 'param',
      min: 30,
      max: 400,
      step: 10,
      default: 160,
      unit: 'frames',
      help: 'Particles respawn after this many frames, so no trail drifts forever.'
    },
    {
      kind: 'slider',
      id: 'minimumSpeed',
      label: 'Respawn below speed',
      group: 'Particles',
      apply: 'param',
      min: 0,
      max: 4,
      step: 0.1,
      default: 0.3,
      unit: 'm/s',
      help: 'Particles slower than this respawn, so they do not pile up in dead air behind cliffs.'
    },
    {
      kind: 'select',
      id: 'spawnRegion',
      label: 'Respawn region',
      group: 'Particles',
      apply: 'param',
      default: 'window',
      help: 'Where new particles appear: anywhere in the 32 km window, or only inside the visible map, which concentrates all 40,000 particles when you zoom in.',
      options: [
        {value: 'window', label: 'Whole DEM window'},
        {value: 'view', label: 'Visible map only'}
      ]
    },
    {
      kind: 'slider',
      id: 'seed',
      label: 'Random seed',
      group: 'Particles',
      apply: 'param',
      min: 1,
      max: 99,
      step: 1,
      default: 7,
      help: 'The generator (Philox 4x32-10) is a pure function of the seed, so the same seed replays the same particle motion bit for bit. Changing it restarts the particles.',
      format: value => `${value}`
    },
    {
      kind: 'toggle',
      id: 'paused',
      label: 'Pause particles',
      group: 'Particles',
      apply: 'param',
      default: false,
      help: 'Stops encoding the particle graph; the field stays live.'
    },
    {
      kind: 'button',
      id: 'reset',
      label: 'Restart particles',
      group: 'Particles',
      help: 'Respawns every particle with staggered ages and clears the trails.'
    }
  ],

  story: [
    {
      id: 'question',
      title: 'Where does a storm front thread the Canyon?',
      headline: 'Terrain reduces modeled wind speed near steep walls',
      textAlternative:
        'Colored particle trails cross Grand Canyon relief, moving faster along open or aligned terrain and slowing near steep opposing walls.',
      body: 'A winter front crosses the Colorado Plateau from the west-south-west at 50 km/h. The rims are 1.9 km above the river, so the air cannot simply go straight on: it climbs, stalls against cliffs and squeezes along side canyons. `GPUTerrainFlowField` and `GPUParticleAdvection` turn the DEM into an animated picture of that.\n\nEach streak is one of 40,000 particles. Its **color is its speed** (legend), its tail fades with age. Hover the map for elevations.',
      camera: {longitude: -112.1, latitude: 36.1, zoom: 11.2, transitionMs: 1400},
      options: {background: 'relief'},
      controls: ['windSpeed', 'windBearing']
    },
    {
      id: 'field',
      title: 'A field made of one rule',
      headline: 'Terrain projection creates sheltered low-speed cells',
      textAlternative:
        'A continuous speed surface shows the terrain-projected wind as a share of free-stream speed, with sheltered fractions reported.',
      body: '`GPUTerrainFlowField` is a single kernel with one rule: air cannot enter the ground, so the 3D wind is projected onto the terrain’s tangent plane. With the surface gradient `g`, the horizontal wind becomes\n\n`v = w − (w · g) g / (1 + |g|²)`\n\nWind **across** a slope (`w · g = 0`) is unchanged; wind **straight up** a slope loses a factor `1 / (1 + |g|²)`; a vertical wall stops it. Set by **Backdrop** below, the map now shows the result as a share of the free-stream speed (dark = sheltered). The readout counts how much of the window sits below half and a tenth of it.',
      options: {background: 'speed'},
      controls: ['background', 'windSpeed'],
      readouts: ['sheltered'],
      highlight: {readout: 'sheltered'}
    },
    {
      id: 'funnel',
      title: 'Turn the wind and watch it funnel',
      headline: 'Wind alignment preserves speed along Bright Angel Canyon',
      textAlternative:
        'After the bearing changes, higher relative speeds follow Bright Angel Canyon while opposing walls contain darker low-speed cells.',
      body: 'The wind now blows toward the north-north-east (bearing 20°), almost straight up Bright Angel Canyon from the Colorado to the North Rim. Because only the component *across* the walls survives a cliff, air that meets the side walls head-on stalls and the part along the fault line keeps its speed: the canyon floor and its trails glow while the walls go dark.\n\nDrag **Wind blows toward** below around the compass. The field kernel re-runs on the next frame as a parameter write; nothing recompiles.',
      camera: {longitude: -112.09, latitude: 36.12, zoom: 12.3, transitionMs: 1800},
      options: {windBearing: 20, background: 'speed'},
      controls: ['windBearing'],
      callout: {coordinate: [-112.0953, 36.107], text: 'Phantom Ranch, Bright Angel Canyon'}
    },
    {
      id: 'deflection',
      title: 'How stiff is the ground?',
      headline: 'Greater deflection expands modeled shelter',
      textAlternative:
        'Increasing terrain deflection darkens a larger share of the speed field and raises sheltered and nearly blocked readouts.',
      body: '**Terrain deflection** multiplies the gradient before the projection. At 1× only the steepest walls matter; at 8× (default) every slope above a few degrees acts like an obstacle, which models stable evening air that goes around hills rather than over them; at 20× the Canyon becomes a maze of near-closed channels. Slide it and read the *Sheltered* and *Nearly blocked* readouts change: it comes from a `GPUHistogram` of the field speed, refreshed whenever the field is rewritten.\n\nThis is a kinematic model, not a weather simulation: it has no pressure, no buoyancy and no mass conservation, so it shows where terrain can steer a wind, not what a real storm will do.',
      options: {exaggeration: 14, windBearing: 20, background: 'speed'},
      controls: ['exaggeration'],
      readouts: ['blocked', 'sheltered'],
      highlight: {readout: 'blocked'}
    },
    {
      id: 'particles',
      title: 'Particles in data space',
      headline: 'Particle trails reproduce the computed velocity field',
      textAlternative:
        'Persistent colored trails advect through the field, with length and travel readouts responding to time step, playback speed and particle age.',
      body: '`GPUParticleAdvection` moves every particle with a midpoint (RK2) step through the field, using bilinear sampling by hand so no float-filterable texture is needed. Positions live in ground meters, so trails stay attached to the terrain as you pan and zoom. Particles respawn when they age out, drop out randomly, or slow below **Respawn below speed**, which is why there is no dead pile-up behind cliffs.\n\nTry **Time step**, **Playback speed** and **Maximum age** below. The random numbers are Philox 4x32-10: choose a **Random seed**, press **Restart particles**, and the same motion replays bit for bit.',
      options: {windBearing: 75, exaggeration: 8, background: 'relief', speedScale: 2},
      controls: ['timeStep', 'speedScale', 'maximumAge', 'seed', 'reset'],
      camera: {longitude: -112.1, latitude: 36.1, zoom: 11.2, transitionMs: 1600}
    },
    {
      id: 'zoom',
      title: 'Spend every particle where you are looking',
      headline: 'View-based respawning increases local trail density',
      textAlternative:
        'At a closer canyon view, all particles respawn inside the current map bounds, increasing trail density without changing the velocity field.',
      body: 'Set **Respawn region** below to *Visible map only* and zoom into a side canyon: all 40,000 particles now respawn inside the view, so the density rises many-fold and individual eddies become visible. `spawnBounds` is a per-frame rectangle in the particle parameter buffer, so following the camera costs nothing.\n\nThings to try with the controls below: a wind blowing along the canyon axis (about 250° or 70°), a calm 3 m/s breeze with low deflection, or the speed backdrop to find the stagnation zones directly upwind of the Redwall cliffs.',
      camera: {longitude: -112.07, latitude: 36.105, zoom: 13.2, transitionMs: 2000},
      options: {spawnRegion: 'view', windBearing: 75},
      controls: ['spawnRegion', 'windBearing', 'windSpeed', 'exaggeration', 'background']
    }
  ],

  about: {
    what: '`GPUTerrainFlowField` deflects a uniform horizontal wind around the DEM with a tangent-plane projection; `GPUParticleAdvection` integrates 40,000 particles through the field with RK2 and writes a trail ring buffer that a layer draws directly.',
    why: 'Terrain-steered flow is the first-order picture behind wind shelter, smoke and fire-spread corridors, cold-air drainage and wind-energy siting, and it is cheap enough to animate for a whole 32 km window.',
    howToRead:
      'A streak is the recent path of one particle; color is its speed in m/s. In the speed backdrop, dark means a cell receives a small share of the free-stream wind (sheltered or blocked), light means it keeps most of it.'
  },

  legends: state => {
    const legends: LegendSpec[] = [
      {
        kind: 'ramp',
        title: 'Particle speed',
        ramp: state.ramp,
        extent: [0, state.windSpeed * 1.1],
        unit: 'm/s',
        format: value => value.toFixed(1)
      }
    ];
    if (state.background === 'speed') {
      legends.push({
        kind: 'ramp',
        title: 'Field speed',
        ramp: 'magma',
        extent: [0, 1],
        unit: 'of the free-stream wind',
        format: value => `${Math.round(value * 100)} %`
      });
    }
    return legends;
  },

  readouts: [
    {
      id: 'sheltered',
      label: 'Sheltered',
      help: 'Share of cells where the deflected wind is below half of the free-stream speed (GPUHistogram of the field).'
    },
    {
      id: 'blocked',
      label: 'Nearly blocked',
      help: 'Share of cells below a tenth of the free-stream speed.'
    },
    {id: 'particles', label: 'Particles', format: 'integer'},
    {
      id: 'travel',
      label: 'Step length',
      help: 'Distance a particle moves in one frame at the free-stream speed.'
    },
    {
      id: 'trail',
      label: 'Trail length',
      help: 'Ground length of a full trail at the free-stream speed.'
    }
  ],

  snippet: state => `import {
  GPUTerrainFlowField, getGPUTerrainFlowFieldParameterValues
} from '@luma.gl/experimental/gpu-terrain';
import {
  GPUParticleAdvection, getGPUParticleAdvectionParameterValues,
  getGPUParticleAdvectionWordParameterValues
} from '@luma.gl/experimental/gpu-raster';

// Wind field: one kernel; raster is south-first so +y is north.
windGraph.add(new GPUTerrainFlowField({
  width, height, elevation, settings: windSettings.importToGraph(windGraph),
  cellSizeMode: 'web-mercator', velocities        // float32x2 (u, v) in m/s
}));
particleGraph.add(new GPUParticleAdvection({
  velocities, fieldWidth: width, fieldHeight: height,
  parameters, wordParameters,
  state: {positions, ages, generations},
  speeds, trails: {positions: trailPositions, length: 24}   // ring: frame f writes slot f % 24
}));

// Per frame: parameter writes only.
const bearing = ${state.windBearing} * Math.PI / 180;
windSettings.write(getGPUTerrainFlowFieldParameterValues({
  cellSize, northEdge, southEdge,
  wind: [${state.windSpeed} * Math.sin(bearing), ${state.windSpeed} * Math.cos(bearing)],
  verticalExaggeration: ${state.exaggeration}
}));
particleParameters.write(getGPUParticleAdvectionParameterValues({
  fieldExtent: [minX, minY, cellWidth, cellHeight], timeStep: ${state.timeStep},
  speedScale: ${state.speedScale}, dropRate: ${state.dropRate}, minimumSpeed: ${state.minimumSpeed},
  spawnBounds: ${state.spawnRegion === 'view' ? 'visibleMapBounds' : 'undefined /* whole field */'}
}, [width, height]));
particleWords.write(getGPUParticleAdvectionWordParameterValues({
  seed: ${state.seed}, frame, maximumAge: ${state.maximumAge}, reset: frame === 0
}));
compiledWind.encode(commandEncoder, {parameters: undefined});      // when the wind changed
compiledParticles.encode(commandEncoder, {parameters: undefined}); // every frame`,

  create: async ctx => (await import('./terrain-flow.compute')).createTerrainFlow(ctx)
});
