// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
import {ALPINE_ELEVATION_STOPS} from './relief-visualization.style';
import {formatDay, formatHour, type SunOptions} from './sun-and-shadow.style';

const SCALAR_RANGES: Partial<
  Record<SunOptions['product'], {low: number; high: number; unit: string; title: string}>
> = {
  illumination: {low: 0, high: 1, unit: '', title: 'Illumination: ambient + sun x cos(incidence)'},
  shadow: {low: 0, high: 1, unit: '', title: 'Sun visibility (horizon map)'},
  'cast-shadow': {low: 0, high: 1, unit: '', title: 'Sun visibility (cast shadow)'},
  'shadow-difference': {
    low: 0,
    high: 0.5,
    unit: '',
    title: 'Difference between the two shadow methods'
  },
  'horizon-angle': {low: -10, high: 60, unit: '°', title: 'Terrain horizon angle toward the sun'},
  'sun-hours': {low: 0, high: 16, unit: 'h', title: 'Hours of direct sun'},
  insolation: {
    low: 0,
    high: 10,
    unit: 'kWh/m²',
    title: 'Clear-sky solar energy per day (direct + diffuse)'
  },
  'sky-view': {low: 0.35, high: 1, unit: '', title: 'Sky-view factor'}
};

const FIXED_RAMP: Partial<Record<SunOptions['product'], 'grayscale'>> = {
  shadow: 'grayscale',
  'cast-shadow': 'grayscale'
};

const irradianceProducts: SunOptions['product'][] = ['sun-hours', 'insolation'];

/**
 * Sun and shadow in the Zermatt valley: sun position, soft horizon-map shadows, exact cast
 * shadows, the time of day animated, and a whole day of sun hours and insolation.
 */
export default defineScene<SunOptions>({
  id: 'sun-and-shadow',
  title: 'When does the sun reach Zermatt?',
  chapter: 'terrain',
  order: 4,
  summary:
    'Sun position, soft shadows, exact cast shadows, sun hours and clear-sky insolation over the Matterhorn and the Zermatt valley, on any day and hour, animated on the GPU.',
  contributors: [
    'GPUSolarPosition',
    'GPUTerrainHorizon',
    'GPUSolarShadowMask',
    'GPUTerrainCastShadow',
    'GPUSolarIrradiance',
    'GPUReliefShading',
    'GPUTerrainDerivatives'
  ],
  datasets: [{id: 'alps-dem', role: 'elevation'}],
  initialView: {longitude: 7.742, latitude: 45.985, zoom: 11.9},

  options: [
    {
      kind: 'slider',
      id: 'dayOfYear',
      label: 'Date',
      group: 'Sun (GPUSolarPosition)',
      apply: 'param',
      min: 1,
      max: 365,
      step: 1,
      default: 355,
      format: value => formatDay(value),
      help: 'Day of 2026. Dec 21 is the winter solstice, when the valley floor sees little sun; Jun 21 the summer solstice.'
    },
    {
      kind: 'slider',
      id: 'hour',
      label: 'Time of day',
      group: 'Sun (GPUSolarPosition)',
      apply: 'param',
      min: 0,
      max: 24,
      step: 0.05,
      default: 11,
      format: value => formatHour(value),
      disabledWhen: state => irradianceProducts.includes(state.product),
      help: 'Local time in Switzerland (CET or CEST). A parameter write: the sun moves, the horizon map is reused. The sun-hours and insolation views integrate the whole day.'
    },
    {
      kind: 'toggle',
      id: 'animate',
      label: 'Animate time of day',
      group: 'Sun (GPUSolarPosition)',
      apply: 'param',
      default: false,
      disabledWhen: state => irradianceProducts.includes(state.product),
      help: 'Advances the clock every frame. The solar position, the shadow mask and the composite re-run each frame (about 3 ms of GPU time on this tile). The slider shows where you started; the clock readout shows the animated time.'
    },
    {
      kind: 'slider',
      id: 'animationSpeed',
      label: 'Animation speed',
      group: 'Sun (GPUSolarPosition)',
      apply: 'param',
      min: 0.25,
      max: 6,
      step: 0.25,
      default: 1.5,
      unit: 'h/s',
      disabledWhen: state => !state.animate,
      help: 'Hours of the day that pass per second.'
    },
    {
      kind: 'select',
      id: 'twilight',
      label: 'Daylight threshold',
      group: 'Sun (GPUSolarPosition)',
      apply: 'param',
      default: '-0.833',
      help: 'Sun altitude below which the GPU daylight flag reads "night": sunrise (-0.833, the sun\'s upper limb with refraction) or civil, nautical or astronomical twilight.',
      options: [
        {value: '-0.833', label: 'Sunrise / sunset (-0.83°)'},
        {value: '-6', label: 'Civil twilight (-6°)'},
        {value: '-12', label: 'Nautical twilight (-12°)'},
        {value: '-18', label: 'Astronomical twilight (-18°)'}
      ]
    },
    {
      kind: 'toggle',
      id: 'refraction',
      label: 'Atmospheric refraction',
      group: 'Sun (GPUSolarPosition)',
      apply: 'compile',
      default: true,
      help: 'Lifts the apparent sun a fraction of a degree near the horizon (NOAA). Part of the contributor topology, so toggling it rebuilds the sun graph.'
    },
    {
      kind: 'select',
      id: 'horizonDirections',
      label: 'Azimuth sectors',
      group: 'Horizon map (GPUTerrainHorizon)',
      apply: 'compile',
      default: '16',
      help: 'Compass directions of the horizon map. The shadow mask interpolates between neighbouring sectors, so more sectors give crisper shadow edges at higher memory.',
      options: [
        {value: '8', label: '8 sectors'},
        {value: '16', label: '16 sectors'},
        {value: '32', label: '32 sectors'}
      ]
    },
    {
      kind: 'select',
      id: 'horizonRadius',
      label: 'Search radius',
      group: 'Horizon map (GPUTerrainHorizon)',
      apply: 'compile',
      default: '256',
      help: 'How far, in pixels (6.6 m each), a ray looks for a blocking horizon. At low sun the Matterhorn casts shadows for kilometres; a short radius cuts them off.',
      options: [
        {value: '128', label: '128 px (0.85 km)'},
        {value: '256', label: '256 px (1.7 km)'},
        {value: '384', label: '384 px (2.5 km)'},
        {value: '512', label: '512 px (3.4 km)'}
      ]
    },
    {
      kind: 'slider',
      id: 'horizonGrowth',
      label: 'Step growth',
      group: 'Horizon map (GPUTerrainHorizon)',
      apply: 'compile',
      min: 1,
      max: 1.3,
      step: 0.02,
      default: 1.1,
      disabledWhen: state => state.horizonAlgorithm === 'sweep',
      help: 'Each step along a ray is this much longer than the last. 1 samples every pixel; 1.1 reaches 1.7 km with a few dozen samples.'
    },
    {
      kind: 'select',
      id: 'horizonFormat',
      label: 'Horizon storage',
      group: 'Horizon map (GPUTerrainHorizon)',
      apply: 'compile',
      default: 'unorm16',
      help: 'float32 stores one 4-byte angle per pixel and sector; unorm16 packs two 16-bit codes per word (0.00275° steps), half the memory (see the readout).',
      options: [
        {value: 'unorm16', label: 'unorm16 (half the memory)'},
        {value: 'float32', label: 'float32'}
      ]
    },
    {
      kind: 'select',
      id: 'horizonAlgorithm',
      label: 'Algorithm',
      group: 'Horizon map (GPUTerrainHorizon)',
      apply: 'compile',
      default: 'march',
      help: 'March samples each ray (cheap at short radii); sweep is an exact upper-hull sweep (Stewart 1998) that wins at radii near the tile size.',
      options: [
        {value: 'march', label: 'March'},
        {value: 'sweep', label: 'Sweep (step growth fixed at 1)'}
      ]
    },
    {
      kind: 'slider',
      id: 'softness',
      label: 'Shadow softness (solar disk)',
      group: 'Shadows',
      apply: 'param',
      min: 0,
      max: 8,
      step: 0.5,
      default: 1,
      unit: 'x disk',
      format: value => (value === 0 ? 'hard' : `${(value * 0.2665).toFixed(2)}° radius`),
      help: 'Angular radius of the sun as a multiple of the real disk (0.27°). 0 gives hard shadows; larger values widen the penumbra. Applies to the mask, the cast shadow and the irradiance.'
    },
    {
      kind: 'slider',
      id: 'sunIntensity',
      label: 'Direct sun intensity',
      group: 'Shadows',
      apply: 'param',
      min: 0,
      max: 2,
      step: 0.05,
      default: 1,
      disabledWhen: state => !['composite', 'illumination'].includes(state.product),
      help: 'Multiplier of the direct term of the illumination: sun x visibility x cos(incidence).'
    },
    {
      kind: 'slider',
      id: 'ambient',
      label: 'Ambient light (x sky-view)',
      group: 'Shadows',
      apply: 'param',
      min: 0,
      max: 0.8,
      step: 0.05,
      default: 0.2,
      disabledWhen: state => !['composite', 'illumination'].includes(state.product),
      help: 'Sky light that reaches shaded ground, scaled by the sky-view factor so a deep gorge stays darker than an open ridge.'
    },
    {
      kind: 'select',
      id: 'castRadius',
      label: 'Cast shadow reach',
      group: 'Cast shadow (GPUTerrainCastShadow)',
      apply: 'compile',
      default: '1024',
      disabledWhen: state =>
        !['cast-shadow', 'shadow-difference', 'horizon-angle'].includes(state.product),
      help: 'Longest shadow the exact sweep can cast, in pixels (also the halo a tile needs). The whole tile diagonal is exact but slowest.',
      options: [
        {value: '512', label: '512 px'},
        {value: '1024', label: '1,024 px (6.8 km)'},
        {value: '2048', label: '2,048 px'},
        {value: 'tile', label: 'Tile diagonal (2,896 px)'}
      ]
    },
    {
      kind: 'select',
      id: 'directIrradiance',
      label: 'Direct sun model',
      group: 'Irradiance (GPUSolarIrradiance)',
      apply: 'param',
      default: 'meinel',
      disabledWhen: state => !irradianceProducts.includes(state.product),
      help: 'Direct normal irradiance while the sun is up: the Meinel clear-sky model with Kasten-Young air mass (falls toward the horizon), or a constant.',
      options: [
        {value: 'meinel', label: 'Meinel clear sky (air mass)'},
        {value: '1000', label: 'Constant 1000 W/m²'},
        {value: '600', label: 'Constant 600 W/m² (hazy)'}
      ]
    },
    {
      kind: 'slider',
      id: 'diffuseIrradiance',
      label: 'Diffuse sky irradiance',
      group: 'Irradiance (GPUSolarIrradiance)',
      apply: 'param',
      min: 0,
      max: 300,
      step: 10,
      default: 100,
      unit: 'W/m²',
      disabledWhen: state => !irradianceProducts.includes(state.product),
      help: "Sky light added while the sun is up, scaled by each pixel's sky-view factor, so enclosed gorges get less."
    },
    {
      kind: 'select',
      id: 'tableStep',
      label: 'Sun table step',
      group: 'Irradiance (GPUSolarIrradiance)',
      apply: 'param',
      default: '5',
      disabledWhen: state => !irradianceProducts.includes(state.product),
      help: 'Minutes between samples of the sun path that the GPU integrates (rewritable per frame; the table holds at most 288 rows).',
      options: [
        {value: '5', label: '5 minutes (288 samples)'},
        {value: '10', label: '10 minutes'},
        {value: '15', label: '15 minutes'},
        {value: '30', label: '30 minutes'}
      ]
    },
    {
      kind: 'slider',
      id: 'shadowStrength',
      label: 'Shadow strength in the composite',
      group: 'Composite',
      apply: 'param',
      min: 0,
      max: 1,
      step: 0.05,
      default: 1,
      disabledWhen: state => state.product !== 'composite',
      help: 'How much of the sun illumination darkens the relief color: 0 shows the relief alone, 1 is fully lit.'
    },
    {
      kind: 'slider',
      id: 'lightFloor',
      label: 'Brightness of shadowed ground',
      group: 'Composite',
      apply: 'param',
      min: 0,
      max: 0.6,
      step: 0.02,
      default: 0.16,
      disabledWhen: state => state.product !== 'composite',
      help: 'Lowest brightness, relative to full sun, that shaded ground is dimmed to.'
    },
    {
      kind: 'slider',
      id: 'exposure',
      label: 'Exposure',
      group: 'Composite',
      apply: 'param',
      min: 0.8,
      max: 3,
      step: 0.1,
      default: 1.7,
      disabledWhen: state => state.product !== 'composite',
      help: 'Brightens the illumination before it is clamped. The winter sun is low, so cosines are small.'
    },
    {
      kind: 'slider',
      id: 'reliefShade',
      label: 'Relief shading in the color',
      group: 'Composite',
      apply: 'param',
      min: 0,
      max: 1,
      step: 0.05,
      default: 0.55,
      disabledWhen: state => state.product !== 'composite',
      help: 'Strength of the multidirectional hillshade baked into the relief color (GPUReliefShading); the sun adds its own light on top.'
    },
    {
      kind: 'toggle',
      id: 'elevationTint',
      label: 'Elevation tint',
      group: 'Composite',
      apply: 'param',
      default: true,
      disabledWhen: state => state.product !== 'composite',
      help: 'Colors the relief by height, from meadow to snow.'
    },
    {
      kind: 'select',
      id: 'product',
      label: 'Show',
      group: 'Display',
      apply: 'param',
      default: 'composite',
      help: 'Each view is a set of compiled stages built the first time you pick it. The horizon map and slope are shared by all of them.',
      options: [
        {value: 'composite', label: 'Relief lit by the sun (composite)'},
        {value: 'illumination', label: 'Illumination (sun + ambient)'},
        {value: 'shadow', label: 'Sun visibility: horizon map (soft shadow)'},
        {value: 'cast-shadow', label: 'Sun visibility: exact cast shadow'},
        {value: 'shadow-difference', label: 'Where the two shadow methods differ'},
        {value: 'horizon-angle', label: 'Terrain horizon angle toward the sun'},
        {value: 'sun-hours', label: 'Hours of direct sun on the chosen day'},
        {value: 'insolation', label: 'Clear-sky insolation on the chosen day'},
        {value: 'sky-view', label: 'Sky-view factor'}
      ]
    },
    {
      kind: 'slider',
      id: 'opacity',
      label: 'Layer opacity',
      group: 'Display',
      apply: 'param',
      min: 0.2,
      max: 1,
      step: 0.05,
      default: 0.95,
      help: 'Lower it to see the basemap under the raster.'
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp',
      group: 'Display',
      apply: 'param',
      default: 'inferno',
      disabledWhen: state => ['composite', 'shadow', 'cast-shadow'].includes(state.product),
      help: 'Ramp of the scalar views (the shadow views are grayscale).',
      options: [
        {value: 'inferno', label: 'Inferno'},
        {value: 'magma', label: 'Magma'},
        {value: 'viridis', label: 'Viridis'},
        {value: 'cividis', label: 'Cividis (color-blind optimised)'}
      ]
    },
    {
      kind: 'button',
      id: 'measure',
      label: 'Time this view',
      group: 'Under the hood',
      help: 'Runs every graph the displayed view depends on outside the frame and reports the total GPU time.'
    }
  ],

  readouts: [
    {id: 'grid', label: 'Raster'},
    {id: 'localTime', label: 'Local time (Switzerland)'},
    {id: 'sunCpu', label: 'Sun at the tile center (float64 CPU)'},
    {
      id: 'sunGpu',
      label: 'Sun at Zermatt (GPUSolarPosition)',
      help: 'Azimuth and altitude computed on the GPU for four lon/lat rows at once; the first row drives the shadow mask.'
    },
    {id: 'dayLength', label: 'Day length (from the sun table)'},
    {id: 'meanValue', label: 'Tile average of the displayed raster'},
    {id: 'horizonMemory', label: 'Horizon map memory'},
    {id: 'timing', label: 'GPU time'}
  ],

  legends: state => {
    if (state.product === 'composite') {
      return [
        {
          kind: 'ramp',
          title: 'Sun illumination of the relief',
          ramp: 'grayscale',
          extent: [0, 1],
          labels: ['shaded', 'full sun']
        },
        ...(state.elevationTint
          ? [
              {
                kind: 'categories' as const,
                title: 'Elevation tint',
                entries: ALPINE_ELEVATION_STOPS.map(stop => ({
                  color: [
                    Math.round(stop.color[0] * 255),
                    Math.round(stop.color[1] * 255),
                    Math.round(stop.color[2] * 255)
                  ] as [number, number, number],
                  label: `${stop.elevation.toLocaleString('en-US')} m`
                }))
              }
            ]
          : [])
      ];
    }
    const spec = SCALAR_RANGES[state.product];
    if (!spec) return [];
    return [
      {
        kind: 'ramp',
        title: spec.title,
        ramp: FIXED_RAMP[state.product] ?? state.ramp,
        extent: [spec.low, spec.high],
        unit: spec.unit,
        format: value => (Number.isInteger(value) ? String(value) : value.toFixed(2))
      }
    ];
  },

  story: [
    {
      id: 'question',
      controls: ['dayOfYear', 'hour'],
      readouts: ['localTime'],
      title: 'When does the winter sun reach the Zermatt valley?',
      body: 'Zermatt sits at 1,600 m at the foot of the Matterhorn, in a valley walled in by 4,000 m peaks. On the winter solstice the sun never gets higher than 20 degrees at this latitude, so for much of the day the valley floor is in shadow cast by mountains kilometres away. A hotel owner, a solar-panel installer and a skier all want the same thing: *where* and *when* does the sun actually arrive?\n\nThe map is the relief colored by the real sun at 11:00 local time on 21 December. Dark ground is in shadow and lit slopes are bright, with an Alpine elevation tint. Everything is computed on the GPU from one elevation raster. Hover any cell for the exact elevation, and move the sun with **Date** and **Time of day** below.',
      options: {product: 'composite', dayOfYear: 355, hour: 11},
      camera: {longitude: 7.742, latitude: 45.985, zoom: 11.9, transitionMs: 1400},
      callout: {coordinate: [7.7491, 46.0207], text: 'Zermatt 1,600 m'}
    },
    {
      id: 'sun-position',
      controls: ['hour', 'dayOfYear', 'twilight'],
      readouts: ['sunGpu', 'sunCpu'],
      title: 'First, where is the sun?',
      body: "**`GPUSolarPosition`** evaluates the NOAA solar-position algorithm for every longitude/latitude row of a buffer, at one instant: azimuth (compass bearing), altitude (angle above the horizon, with atmospheric refraction) and a daylight flag. The readouts compare it with the float64 CPU version: they agree to a few thousandths of a degree.\n\nThis is also what drives the shadows: a one-thread kernel copies the GPU's azimuth and altitude into the shadow-mask settings, so the sun really is computed on the GPU. Drag **Time of day** and **Date** below, and try the **Daylight threshold** (civil twilight is the sun 6 degrees below the horizon).",
      options: {product: 'composite', dayOfYear: 355, hour: 9.5},
      highlight: {readout: 'sunGpu'}
    },
    {
      id: 'horizon-shadow',
      controls: ['softness', 'horizonRadius'],
      readouts: ['meanValue'],
      title: 'Soft shadows from a horizon map',
      body: 'Shadows need to know, for every pixel, how high the terrain rises in the direction of the sun. **`GPUTerrainHorizon`** computes that once: for each pixel and each of 16 compass sectors it marches a ray across the DEM and keeps the highest angle. **`GPUSolarShadowMask`** then needs only two reads per pixel to interpolate the horizon at the sun azimuth and compare it with the sun altitude. Moving the sun costs about 3 ms; the horizon map is reused.\n\nThe map shows *sun visibility*: white is full sun, black full shadow, grays are penumbra where the sun is partly hidden. Slide **Shadow softness (solar disk)** below from hard (0) to wide, or shorten the **Search radius** and watch distant shadows vanish.',
      options: {product: 'shadow', dayOfYear: 355, hour: 14.5, softness: 1},
      camera: {longitude: 7.742, latitude: 45.99, zoom: 12.2, transitionMs: 1500},
      highlight: {readout: 'meanValue'}
    },
    {
      id: 'cast-shadow',
      controls: ['product', 'castRadius'],
      readouts: ['horizonMemory'],
      title: 'An exact alternative: cast the shadow directly',
      body: "A horizon map costs memory: 16 sectors at 2048 x 2048 is 134 MB even packed to 16 bits (the **Horizon map memory** readout). If only one sun matters, **`GPUTerrainCastShadow`** needs no map: it sweeps along the sun azimuth and finds each pixel's horizon *in that direction exactly*, then applies the same solar-disk penumbra. Compare this view with the previous one, or switch **Show** below to *Where the two shadow methods differ* (**Cast shadow reach** sets the longest shadow the sweep can cast): the horizon map interpolates between 16 sectors, so its edges are slightly softer, while the cast shadow is exact along the sun line.\n\nUse the horizon map when you will move the sun a lot (animation, sun hours); use the cast shadow for a single exact sun or a tile too large for the map.",
      options: {product: 'cast-shadow', dayOfYear: 355, hour: 14.5},
      camera: {longitude: 7.742, latitude: 45.99, zoom: 12.2, transitionMs: 1200}
    },
    {
      id: 'animate',
      controls: ['animate', 'animationSpeed', 'hour', 'dayOfYear'],
      readouts: ['localTime'],
      title: 'Watch a summer day go by',
      body: 'The clock is running on 21 June, the longest day: watch the shadows of the big peaks swing across the valleys as the sun climbs, crosses the south and sinks. Each frame re-runs the solar position, the shadow mask and the composite on the GPU; the rebuild counter in *Under the hood* stays at zero because every change is a parameter write.\n\nToggle **Animate time of day** or change **Animation speed** below, drag **Time of day** to jump (the clock restarts from the slider), or change the **Date** to see how winter shortens the lit part of the valley.',
      options: {product: 'composite', dayOfYear: 172, hour: 5, animate: true, animationSpeed: 1.5},
      camera: {longitude: 7.742, latitude: 45.985, zoom: 11.9, transitionMs: 1200}
    },
    {
      id: 'sun-hours',
      controls: ['product', 'dayOfYear'],
      readouts: ['dayLength', 'meanValue'],
      title: 'How many hours of sun? Integrate the day',
      body: '**`GPUSolarIrradiance`** integrates the sun over a whole day at once. A table of sun positions (every 5 minutes, from the same solar algorithm) is written to a buffer, and each pixel sums the time the sun disk is above its horizon: **sun hours**, and, weighting by the Meinel clear-sky irradiance and the cosine of the incidence angle on that slope, **insolation** in kWh/m². A diffuse sky term scaled by the sky-view factor is added.\n\nThis is 21 December. Sunlit ridges and south-facing slopes collect the most hours; the black areas, mostly the valley floors and north-facing ground, get none. Use *Hours of direct sun* to find where a south-facing terrace is worth building, and change the **Date** below (the integral re-runs on the GPU when you do).',
      options: {product: 'sun-hours', dayOfYear: 355, animate: false},
      camera: {longitude: 7.742, latitude: 45.985, zoom: 11.9, transitionMs: 1200},
      highlight: {readout: 'dayLength'}
    },
    {
      id: 'limits',
      controls: [
        'dayOfYear',
        'horizonRadius',
        'horizonFormat',
        'horizonDirections',
        'directIrradiance'
      ],
      readouts: ['meanValue'],
      title: 'Insolation, and what this does not model',
      body: 'This is *Clear-sky insolation* on 21 June: the same slopes collect several times more energy than in December (move **Date** below to compare), and north faces that had no direct sun in December receive some. That is how a panel installer would compare roofs.\n\n**Caveats.** The sun and shadow are exact for *terrain*; trees, buildings and clouds are ignored. The irradiance is a clear-sky model (Meinel), not weather: use it for ranking sites, not for forecasting yield. The horizon map is only as long as the **Search radius** and the tile edge: a shadow cast by a peak outside this 13.6 km tile is missing near the border. **Try:** compare the two **Horizon storage** modes, change **Azimuth sectors**, or set **Direct sun model** to a constant.',
      options: {product: 'insolation', dayOfYear: 172},
      highlight: {readout: 'meanValue'}
    }
  ],

  about: {
    what: '`GPUSolarPosition` computes NOAA sun azimuth, altitude and a daylight flag for each lon/lat row on the GPU. `GPUTerrainHorizon` stores the terrain horizon angle per pixel and compass sector. `GPUSolarShadowMask` turns a horizon map and a sun into a soft shadow mask and an illumination; `GPUTerrainCastShadow` casts one exact sun shadow without a map; `GPUSolarIrradiance` integrates a day of sun into sun hours and insolation; `GPUReliefShading` colors the relief the sun lights.',
    why: 'Sun access decides where to build, farm, ski, place solar panels and when a street is in shade. Computing it per pixel for any day and hour lets you ask many "when" and "where" questions interactively.',
    howToRead:
      'Composite: lit slopes are bright, shadowed ground is dim but not black (ambient sky light, scaled by the sky-view factor). Sun visibility: white is full sun, black is shadow, grays are the penumbra of the sun disk. Sun hours: bright is more direct sun. Insolation: kWh/m² of clear-sky energy on the slope itself. The NOAA algorithm is accurate to a small fraction of a degree. The irradiance model is Meinel clear sky with Kasten-Young air mass.'
  },

  create: async ctx => (await import('./sun-and-shadow.compute')).createSunAndShadow(ctx),

  snippet: state => `import {
  GPUSolarPosition, GPUTerrainHorizon, GPUSolarShadowMask, GPUSolarIrradiance, GPUTerrainCastShadow,
  getGPUSolarPositionParameterValues, getGPUSolarShadowMaskParameterValues,
  getGPUSolarIrradianceSunTable, getGPUSolarIrradianceParameterValues, getSolarPosition
} from '@luma.gl/experimental/gpu-terrain';

// once: the horizon map does not depend on the sun
graph.add(new GPUTerrainHorizon({
  width, height, elevation, settings: horizonSettings.importToGraph(graph),
  directionCount: ${state.horizonDirections}, maximumRadius: ${state.horizonRadius}, stepGrowth: ${state.horizonGrowth},
  horizonFormat: '${state.horizonFormat}', cellSizeMode: 'web-mercator', horizon, skyViewFactor
}));
// every frame: the sun, then the mask
graph.add(new GPUSolarPosition({positions, settings: solarSettings.importToGraph(graph), azimuth, altitude, daylight}));
graph.add(new GPUSolarShadowMask({
  width, height, directionCount: ${state.horizonDirections}, horizonFormat: '${state.horizonFormat}', horizon,
  settings: maskSettings, slope, aspect, skyViewFactor, sunVisibility, illumination
}));
graph.add(new GPUSolarIrradiance({
  width, height, directionCount: ${state.horizonDirections}, horizonFormat: '${state.horizonFormat}', horizon,
  sunTable, sampleCapacity: 288, settings: irradianceSettings.importToGraph(graph), slope, aspect, skyViewFactor,
  sunHours, insolation
}));
const compiled = graph.compile(); // once

const timestamp = Date.UTC(2026, 11, 21, 10); // 11:00 CET
solarSettings.write(getGPUSolarPositionParameterValues({timestamp}));
const sun = getSolarPosition(timestamp, 7.742, 45.985); // float64 reference
maskSettings.write(getGPUSolarShadowMaskParameterValues({
  azimuthDegrees: sun.azimuthDegrees, altitudeDegrees: sun.altitudeDegrees,
  angularRadiusDegrees: ${(0.2665 * state.softness).toFixed(3)}, ambientIntensity: ${state.ambient}
}));
const {values, sampleCount} = getGPUSolarIrradianceSunTable({
  longitude: 7.742, latitude: 45.985, start: Date.UTC(2026, 11, 20, 23), end: Date.UTC(2026, 11, 21, 23),
  stepMinutes: ${state.tableStep}, directNormalIrradiance: ${state.directIrradiance === 'meinel' ? "'meinel'" : state.directIrradiance}
});
sunTable.write(values);
irradianceSettings.write(getGPUSolarIrradianceParameterValues({sampleCount, diffuseIrradiance: ${state.diffuseIrradiance}}));
compiled.encode(commandEncoder, {parameters: undefined});`
});
