// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
import type {WindFlowOptions} from './wind-flow.compute';
import {WIND_RAMP} from './b16-colors';

/** Where Helene's wind peaked in the 10 m GFS frames (frame 14, `properties.wind10m`). */
const HELENE_PEAK: readonly [number, number] = [-83.75, 28.75];

const formatHour = (value: number) => {
  const time = new Date(Date.UTC(2024, 8, 26, 12 + Math.round(value)));
  return `${time.toISOString().slice(5, 13).replace('T', ' ')}Z`;
};

export default defineScene<WindFlowOptions>({
  id: 'wind-flow',
  title: 'How did Hurricane Helene organise the wind?',
  chapter: 'raster',
  order: 4,
  summary:
    'GFS winds around Hurricane Helene drawn three ways from one GPU velocity field: 30,000 advected particles with trails, a line-integral-convolution texture and evenly spaced streamlines that follow the camera.',
  contributors: ['GPUParticleAdvection', 'GPULineIntegralConvolution', 'GPUStreamlines'],
  datasets: [{id: 'gfs-wind', role: '10 m and 250 hPa u/v wind, 2024-09-26 12Z run'}],
  initialView: {longitude: -83, latitude: 28.5, zoom: 4.6},

  options: [
    {
      kind: 'select',
      id: 'level',
      label: 'Level',
      group: 'Field',
      apply: 'param',
      default: '10m',
      help: 'Which GFS field is written into the velocity buffer. Switching is one buffer write; nothing recompiles.',
      options: [
        {
          value: '10m',
          label: '10 m above ground (hourly)',
          help: 'The wind people feel; Helene peaks near 45 m/s.'
        },
        {
          value: '250hPa',
          label: '250 hPa jet stream (3-hourly)',
          help: 'About 10 km up; the trough that steered Helene north, up to 78 m/s.'
        }
      ]
    },
    {
      kind: 'slider',
      id: 'hour',
      label: 'Forecast hour',
      group: 'Field',
      apply: 'param',
      min: 0,
      max: 23,
      step: 0.5,
      default: 14,
      format: formatHour,
      help: 'Hours after the 2024-09-26 12Z GFS run. Fractional hours blend the two bracketing frames. The 250 hPa field is available every 3 h up to hour 21. Helene makes landfall near hour 15.'
    },
    {
      kind: 'toggle',
      id: 'play',
      label: 'Play the forecast',
      group: 'Field',
      apply: 'param',
      default: false,
      help: 'Advances the hour on every frame (the time readout shows it) and rewrites the velocity buffer ten times a second.'
    },
    {
      kind: 'slider',
      id: 'playSpeed',
      label: 'Playback speed',
      group: 'Field',
      apply: 'param',
      min: 0.25,
      max: 4,
      step: 0.25,
      default: 1.5,
      unit: 'h / s',
      disabledWhen: state => !state.play,
      help: 'Forecast hours per second of wall time.'
    },
    {
      kind: 'slider',
      id: 'speedMax',
      label: 'Speed at the end of the ramp',
      group: 'Field',
      apply: 'param',
      min: 10,
      max: 80,
      step: 1,
      default: 35,
      unit: 'm/s',
      help: 'Colors run blue to red up to this speed in meters per second. Use about 80 for the jet.'
    },
    {
      kind: 'toggle',
      id: 'showParticles',
      label: 'Particles and trails',
      group: 'Particles (GPUParticleAdvection)',
      apply: 'param',
      default: true,
      help: 'Particles are advected every frame with RK2 on the GPU; their last positions form the trails.'
    },
    {
      kind: 'select',
      id: 'particleCount',
      label: 'Particle count',
      group: 'Particles (GPUParticleAdvection)',
      apply: 'compile',
      default: '30000',
      help: 'Compile-time: the state and trail buffers are sized by it, so a new count rebuilds the advection graph.',
      options: [
        {value: '10000', label: '10,000'},
        {value: '30000', label: '30,000'},
        {value: '60000', label: '60,000'}
      ]
    },
    {
      kind: 'select',
      id: 'trailLength',
      label: 'Trail length',
      group: 'Particles (GPUParticleAdvection)',
      apply: 'compile',
      default: '16',
      help: 'Compile-time: slots of the trail ring buffer per particle (frame f writes slot f mod length).',
      options: [
        {value: '8', label: '8 frames'},
        {value: '16', label: '16 frames'},
        {value: '32', label: '32 frames'}
      ]
    },
    {
      kind: 'slider',
      id: 'speedScale',
      label: 'Speed scale',
      group: 'Particles (GPUParticleAdvection)',
      apply: 'param',
      min: 0.25,
      max: 4,
      step: 0.25,
      default: 1,
      unit: 'x',
      help: 'Multiplies the time step: particles move faster without changing the field.'
    },
    {
      kind: 'slider',
      id: 'dropRate',
      label: 'Drop rate',
      group: 'Particles (GPUParticleAdvection)',
      apply: 'param',
      min: 0,
      max: 0.05,
      step: 0.002,
      default: 0.01,
      format: value => value.toFixed(3),
      help: 'Probability per frame that a particle respawns at random. Higher values keep the pattern fresh.'
    },
    {
      kind: 'slider',
      id: 'maximumAge',
      label: 'Maximum age',
      group: 'Particles (GPUParticleAdvection)',
      apply: 'param',
      min: 40,
      max: 400,
      step: 20,
      default: 120,
      unit: 'frames',
      help: 'A particle respawns when its age reaches this many frames.'
    },
    {
      kind: 'slider',
      id: 'minimumSpeed',
      label: 'Respawn slower than',
      group: 'Particles (GPUParticleAdvection)',
      apply: 'param',
      min: 0,
      max: 10,
      step: 0.5,
      default: 0.5,
      unit: 'm/s',
      help: 'Particles in nearly calm air (the eye, ridges) respawn instead of piling up. Converted to degrees per second.'
    },
    {
      kind: 'toggle',
      id: 'spawnInView',
      label: 'Spawn only in the view',
      group: 'Particles (GPUParticleAdvection)',
      apply: 'param',
      default: true,
      help: 'The respawn rectangle is a per-frame parameter. Restricted to the viewport, all particles work where you look and density grows as you zoom in.'
    },
    {
      kind: 'toggle',
      id: 'zoomAdaptive',
      label: 'Keep screen speed constant when zooming',
      group: 'Particles (GPUParticleAdvection)',
      apply: 'param',
      default: true,
      help: 'Scales the time step with the zoom so particles cross about the same number of pixels per frame at every scale.'
    },
    {
      kind: 'slider',
      id: 'seed',
      label: 'Random seed',
      group: 'Particles (GPUParticleAdvection)',
      apply: 'param',
      min: 0,
      max: 99,
      step: 1,
      default: 7,
      help: 'Philox counter-based random numbers: the same seed replays the same particles bit for bit. Changing it restarts the particles.'
    },
    {
      kind: 'slider',
      id: 'trailWidth',
      label: 'Trail width',
      group: 'Particles (GPUParticleAdvection)',
      apply: 'param',
      min: 0.8,
      max: 3.5,
      step: 0.1,
      default: 1.4,
      unit: 'px',
      help: 'Line width of the trails on screen.'
    },
    {
      kind: 'slider',
      id: 'particleOpacity',
      label: 'Trail opacity',
      group: 'Particles (GPUParticleAdvection)',
      apply: 'param',
      min: 0.2,
      max: 1,
      step: 0.05,
      default: 0.9,
      help: 'Opacity at the head of a trail; it fades with age.'
    },
    {
      kind: 'button',
      id: 'reset',
      label: 'Restart the particles',
      group: 'Particles (GPUParticleAdvection)',
      help: 'Re-initialises every particle with a staggered age (the `reset` flag of the word parameters).'
    },
    {
      kind: 'toggle',
      id: 'showLic',
      label: 'Line integral convolution texture',
      group: 'Texture (GPULineIntegralConvolution)',
      apply: 'param',
      default: true,
      help: 'White noise smeared along the streamlines of the field: streaks show direction everywhere. The raster follows the camera at screen resolution.'
    },
    {
      kind: 'select',
      id: 'licSteps',
      label: 'Taps in each direction',
      group: 'Texture (GPULineIntegralConvolution)',
      apply: 'compile',
      default: '32',
      help: 'Compile-time `stepCount`: more taps give longer streaks and cost more per pixel.',
      options: [
        {value: '16', label: '16 (short streaks)'},
        {value: '32', label: '32'},
        {value: '64', label: '64 (long streaks)'}
      ]
    },
    {
      kind: 'slider',
      id: 'licStep',
      label: 'Step length',
      group: 'Texture (GPULineIntegralConvolution)',
      apply: 'param',
      min: 0.3,
      max: 1.6,
      step: 0.1,
      default: 0.8,
      unit: 'px',
      help: 'Integration step in LIC pixels. Longer steps stretch the streaks.'
    },
    {
      kind: 'slider',
      id: 'licMinimumSpeed',
      label: 'Texture stops where slower than',
      group: 'Texture (GPULineIntegralConvolution)',
      apply: 'param',
      min: 0,
      max: 5,
      step: 0.1,
      default: 0.1,
      unit: 'm/s',
      help: 'Streaks end in nearly calm air, so the eye and ridges stay noisy rather than smeared.'
    },
    {
      kind: 'toggle',
      id: 'animateLic',
      label: 'Animate the ripple phase',
      group: 'Texture (GPULineIntegralConvolution)',
      apply: 'param',
      default: true,
      help: 'Advances the phase every frame so ripples travel downstream. Off, the texture is encoded only when something changes.'
    },
    {
      kind: 'slider',
      id: 'licPeriod',
      label: 'Ripple period',
      group: 'Texture (GPULineIntegralConvolution)',
      apply: 'param',
      min: 0,
      max: 24,
      step: 1,
      default: 10,
      unit: 'steps',
      help: 'Wavelength of the animated ripple along a streamline. 0 gives a static LIC.'
    },
    {
      kind: 'slider',
      id: 'licSeed',
      label: 'Noise seed',
      group: 'Texture (GPULineIntegralConvolution)',
      apply: 'param',
      min: 0,
      max: 99,
      step: 1,
      default: 3,
      help: 'Seed of the white noise (Philox keyed by seed, column and row).'
    },
    {
      kind: 'slider',
      id: 'licContrast',
      label: 'Contrast',
      group: 'Texture (GPULineIntegralConvolution)',
      apply: 'param',
      min: 1,
      max: 8,
      step: 0.5,
      default: 3,
      help: 'Stretches the LIC values around their mean in the layer shader.'
    },
    {
      kind: 'slider',
      id: 'licOpacity',
      label: 'Texture opacity',
      group: 'Texture (GPULineIntegralConvolution)',
      apply: 'param',
      min: 0.1,
      max: 1,
      step: 0.05,
      default: 0.5,
      help: 'Maximum alpha of the texture.'
    },
    {
      kind: 'toggle',
      id: 'showStreamlines',
      label: 'Streamlines',
      group: 'Streamlines (GPUStreamlines)',
      apply: 'param',
      default: false,
      help: 'Evenly spaced polylines along the flow, extracted on the GPU (Jobard and Lefer spacing with deterministic priorities).'
    },
    {
      kind: 'slider',
      id: 'spacing',
      label: 'Line spacing',
      group: 'Streamlines (GPUStreamlines)',
      apply: 'param',
      min: 10,
      max: 48,
      step: 1,
      default: 22,
      unit: 'px',
      help: 'The occupancy-grid cell size is the minimum distance between lines. It is written from the zoom, in screen pixels, around the view center; the grid and seed lattice sizes are compile-time.'
    },
    {
      kind: 'slider',
      id: 'streamMinimumPoints',
      label: 'Shortest line',
      group: 'Streamlines (GPUStreamlines)',
      apply: 'param',
      min: 2,
      max: 40,
      step: 1,
      default: 6,
      unit: 'points',
      help: 'Lines with fewer points after trimming are rejected.'
    },
    {
      kind: 'slider',
      id: 'streamMinimumSpeed',
      label: 'Streamlines stop where slower than',
      group: 'Streamlines (GPUStreamlines)',
      apply: 'param',
      min: 0,
      max: 5,
      step: 0.1,
      default: 0.3,
      unit: 'm/s',
      help: 'Lines end in nearly calm air.'
    },
    {
      kind: 'slider',
      id: 'streamSeed',
      label: 'Seed',
      group: 'Streamlines (GPUStreamlines)',
      apply: 'param',
      min: 0,
      max: 99,
      step: 1,
      default: 5,
      help: 'Seeds the jittered seed lattice and the random priority of each line: a new seed gives a different but equally even layout.'
    },
    {
      kind: 'slider',
      id: 'streamWidth',
      label: 'Line width',
      group: 'Streamlines (GPUStreamlines)',
      apply: 'param',
      min: 0.6,
      max: 3,
      step: 0.1,
      default: 1.2,
      unit: 'px',
      help: 'Line width on screen.'
    },
    {
      kind: 'button',
      id: 'measure',
      label: 'Measure the three graphs',
      group: 'Cost',
      help: 'Times the advection, LIC and streamline graphs outside the frame.'
    }
  ],

  readouts: [
    {
      id: 'time',
      label: 'Valid time',
      help: 'The forecast time of the field in the velocity buffer.'
    },
    {
      id: 'peak',
      label: 'Peak speed',
      help: 'Fastest cell of the current field, with the peak that the GFS file records for hour 14.'
    },
    {
      id: 'field',
      label: 'Field',
      help: 'u and v are converted to degrees per second per cell (u / (R cos latitude)), so the contributors integrate in longitude and latitude.'
    },
    {id: 'particles', label: 'Particles'},
    {id: 'licRaster', label: 'LIC raster'},
    {
      id: 'streamlines',
      label: 'Streamlines',
      help: 'Published line and point counts, the capacity overflow flag and the unconverged flag, read back after each extraction.'
    },
    {id: 'timing', label: 'GPU time'}
  ],

  legends: state => [
    {
      kind: 'categories',
      title: 'Wind speed (m/s)',
      entries: [0, 1 / 3, 2 / 3, 1].map((fraction, index) => ({
        color: [...WIND_RAMP[index], 255] as const,
        label: `${Math.round(state.speedMax * fraction)}${fraction === 1 ? ' or more' : ''}`
      })),
      note: 'Particles, texture and streamlines read the speed in m/s from one raster.'
    }
  ],

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUParticleAdvection, GPULineIntegralConvolution, GPUStreamlines,
  getGPUParticleAdvectionParameterValues, getGPUParticleAdvectionWordParameterValues,
  getGPULineIntegralConvolutionParameterValues, getGPUStreamlinesParameterValues
} from '@luma.gl/experimental/gpu-raster';

// velocities: float32x2 (u, v) per cell in degrees per second, row 0 = smallest latitude
const particleGraph = new GPUCommandGraph(device, {id: 'particles'});
particleGraph.add(new GPUParticleAdvection({
  velocities, fieldWidth: 181, fieldHeight: 153,
  parameters, wordParameters,
  state: {positions, ages, generations},
  speeds,
  trails: {positions: trailRing, length: ${state.trailLength}}      // ${state.particleCount} particles (compile-time)
}));
const lic = new GPUCommandGraph(device, {id: 'lic'});
lic.add(new GPULineIntegralConvolution({velocities, fieldWidth: 181, fieldHeight: 153,
  width: 896, height: 704, stepCount: ${state.licSteps}, parameters: licParameters, wordParameters: licWords,
  output: {values: licValues}}));
const streamlines = new GPUCommandGraph(device, {id: 'streamlines'});
streamlines.add(new GPUStreamlines({velocities, fieldWidth: 181, fieldHeight: 153,
  gridWidth: 128, gridHeight: 112, seedColumns: 192, seedRows: 160,
  stepsPerDirection: 48, roundCount: 16, parameters: streamParameters, wordParameters: streamWords,
  output: {lines: {ids, count, overflow}, pathOffsets, points, pointCount, unconverged}}));

// per frame: parameters only
particleParameters.write(getGPUParticleAdvectionParameterValues({
  fieldExtent: [-105.125, 11.875, 0.25, 0.25], timeStep: 120, speedScale: ${state.speedScale},
  dropRate: ${state.dropRate}, spawnBounds: viewportBounds}, [181, 153]));
particleWords.write(getGPUParticleAdvectionWordParameterValues({
  seed: ${state.seed}, frame, maximumAge: ${state.maximumAge}, reset: false}));
licParameters.write(getGPULineIntegralConvolutionParameterValues({
  fieldExtent, outputExtent: viewportExtent, stepLength: ${state.licStep}, phase, period: ${state.licPeriod}}));
streamParameters.write(getGPUStreamlinesParameterValues({
  fieldExtent, gridExtent: [x0, y0, cell, cell], stepLength: 0.4 * cell}));
particleGraph.compile().encode(commandEncoder, {parameters: undefined});`,

  about: {
    what: 'One velocity buffer feeds three contributors. `GPUParticleAdvection` moves tens of thousands of particles with a midpoint (RK2) step and keeps a ring buffer of their last positions; `GPULineIntegralConvolution` smears a seeded white-noise texture along the flow; `GPUStreamlines` extracts evenly spaced streamlines as polylines. The layers draw straight from the output buffers.',
    why: 'A wind map has to answer two questions at once: which way is the air going (direction) and how strong is it (speed). Particles show motion, LIC shows the whole pattern at once, and streamlines give clean, printable lines. Because the field is just a buffer, the hour slider is a buffer write, and none of the three graphs recompiles.',
    howToRead:
      'Color is the wind speed in m/s (blue calm to red strong). Trails point the way the air moves. Around Helene the flow spirals counter-clockwise, as it does in the northern hemisphere, with the fastest air on the right of the track near the eye wall. The 250 hPa field is the jet that steers the storm.'
  },

  create: async ctx => (await import('./wind-flow.compute')).createWindFlow(ctx),

  story: [
    {
      id: 'the-question',
      title: 'How did Hurricane Helene organise the wind as it reached Florida?',
      body: 'Hurricane Helene made landfall near Perry, Florida, as a category 4 storm at about **03 UTC on 27 September 2024**. This map is the **NOAA GFS 10 m wind** for that hour, from the run of 26 September 12 UTC, on a 0.25 degree grid.\n\nThe moving streaks are **particles**: every one is advected through the wind field on the GPU, every frame. Trails fade with age and are colored by speed. The faint texture underneath is **line integral convolution**. Hover for the wind at any point, or change **Level**, **Forecast hour**, **Particles and trails** and **Line integral convolution texture** below; the peak (45 m/s) sits at the eye wall just off the Big Bend coast.',
      camera: {longitude: -84, latitude: 28.5, zoom: 5.2, transitionMs: 1500},
      options: {level: '10m', hour: 14, showParticles: true, showLic: true, showStreamlines: false},
      controls: ['level', 'hour', 'showParticles', 'showLic'],
      readouts: ['peak', 'time'],
      callout: {coordinate: HELENE_PEAK, text: 'Peak wind, hour 14'},
      highlight: {readout: 'peak'}
    },
    {
      id: 'particles',
      title: 'Particles are advected through the field',
      body: '`GPUParticleAdvection` moves each particle with a **midpoint (RK2) step**: it samples the velocity at the particle, steps half way, samples again there and takes the full step with that second velocity. The field is interpolated bilinearly inside the shader, so no float texture filtering is needed. A particle respawns when it leaves the field, gets too old, is too slow or loses the **drop rate** lottery.\n\nThe velocities are in **degrees per second**: u becomes `u / (R cos latitude)` and v becomes `v / R`, so the integration is exact on this longitude and latitude grid. Try **Drop rate**, **Speed scale** and **Maximum age** below; every one is a parameter write.',
      options: {showParticles: true, showLic: false, showStreamlines: false, speedScale: 1},
      controls: ['dropRate', 'speedScale', 'maximumAge'],
      readouts: ['particles']
    },
    {
      id: 'time',
      title: 'Let the forecast run: time is a buffer write',
      body: 'Switch on **Play the forecast** below (it is on now). The hour advances, the two bracketing GFS frames are blended on the CPU and written into the velocity buffer ten times a second. The particles never restart: their positions live on the GPU, and the next step simply uses the new wind. Watch the storm come ashore and decay as the **valid time** readout counts.\n\nNothing is recompiled; the **Under the hood** panel keeps its rebuild count at zero. Change **Playback speed**, or turn playback off and drag **Forecast hour** to scrub.',
      options: {play: true, playSpeed: 1.5, hour: 12},
      controls: ['play', 'playSpeed', 'hour'],
      readouts: ['time']
    },
    {
      id: 'lic',
      title: 'Line integral convolution shows the whole pattern',
      body: '`GPULineIntegralConvolution` takes a white-noise texture and, for every output pixel, averages the noise along the streamline through that pixel (a Hann-weighted kernel, 32 taps each way). Pixels on one streamline share noise, so streaks appear along the flow, showing direction and shape everywhere at once: the closed spiral of the eye is obvious.\n\nThe output raster covers the **visible map** (an output extent independent of the field), so it stays sharp when you zoom. **Ripple period** and **Animate the ripple phase** make the texture flow; **Taps in each direction** is compile-time and rebuilds just this graph.',
      options: {play: false, hour: 14, showParticles: false, showLic: true, licOpacity: 0.9},
      controls: ['licPeriod', 'animateLic', 'licSteps'],
      readouts: ['licRaster'],
      camera: {longitude: -85, latitude: 28.5, zoom: 5.4, transitionMs: 1200}
    },
    {
      id: 'streamlines',
      title: 'Streamlines give clean, evenly spaced lines',
      body: "`GPUStreamlines` traces candidate lines from a jittered seed lattice in both directions and keeps them in order of a random priority unless they come closer than the **line spacing** to a line already kept (Jobard and Lefer spacing, with the pruning made deterministic). The result is a set of polylines in CSR form, drawn from the output buffer; the point count becomes the draw call's instance count without a readback.\n\nThe occupancy grid is placed from the camera and the **spacing in pixels**, so zooming re-extracts lines at the same screen density. Try **Line spacing** 12 for a dense field and 40 for a sparse one; change the **Seed** below for another equally even layout, or raise **Shortest line** to drop stubs.",
      options: {showParticles: false, showLic: false, showStreamlines: true, spacing: 22},
      controls: ['spacing', 'streamSeed', 'streamMinimumPoints'],
      readouts: ['streamlines']
    },
    {
      id: 'jet',
      title: 'The jet stream at 250 hPa steered the storm',
      body: 'The field is now the **250 hPa** level (**Level** below), about 10 km up. Wind there reaches 78 m/s in this window, and a **trough over the central United States** with its jet is what pulled Helene north. Only the buffer changes: the same graphs, the same compiled kernels.\n\nThe 250 hPa frames are 3-hourly, so fractional hours blend frames three hours apart. **Speed at the end of the ramp** is raised to about 80 m/s so the colors span the jet; lower it to see the ramp saturate. Zoom out to see the whole domain.',
      options: {
        level: '250hPa',
        hour: 12,
        speedMax: 80,
        showParticles: true,
        showLic: true,
        showStreamlines: false,
        licOpacity: 0.5
      },
      controls: ['level', 'hour', 'speedMax'],
      readouts: ['peak', 'time'],
      camera: {longitude: -82, latitude: 31, zoom: 3.9, transitionMs: 1500}
    },
    {
      id: 'caveats',
      title: 'Limits, and things to try',
      body: '**Limits:** the field is a 0.25 degree (about 28 km) model analysis and forecast, so it smooths the hurricane: GFS winds are weaker and wider than the real eye wall. The PNGs quantise u and v to 8 bits (steps of about 0.3 to 0.5 m/s). Hourly frames are blended linearly in time, which is not a model state between them. Particle paths are not trajectories of real air, because the wind changes while they move.\n\n**Try**, below: **Particle count** 60,000, **Trail length** 32 frames, **Drop rate** 0, **Speed scale** 4, and **Spawn only in the view** off to see particles spread over the whole domain.',
      options: {
        level: '10m',
        hour: 15,
        speedMax: 35,
        showParticles: true,
        showLic: true,
        showStreamlines: true,
        licOpacity: 0.35
      },
      controls: ['particleCount', 'trailLength', 'dropRate', 'speedScale', 'spawnInView'],
      camera: {longitude: -84, latitude: 28.5, zoom: 5.0, transitionMs: 1500}
    }
  ]
});
