// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import {
  CLEAR_SKY_CHIP,
  getTerrainFurniture,
  terrainCartouche,
  TERRAIN_FRAMES
} from './terrain-furniture';
import {getStepPlaces, STEP_PLACES} from './sun-and-shadow-places';
import {
  CLOCK_ORIGIN_ISO,
  DATE_PRESETS,
  formatDay,
  formatHour,
  getSunLegends,
  type SunOptions
} from './sun-and-shadow.style';

const CLOCK_RANGE: readonly [number, number] = [0, 8760];

/** Shared terrain furniture, with a step-specific cartouche and civil-time map clock. */
function furnitureFor(subtitle: string, clock = false) {
  return {
    ...getTerrainFurniture({
      cartouche: terrainCartouche('When does the sun reach Zermatt?', subtitle, undefined, [
        CLEAR_SKY_CHIP
      ])
    }),
    ...(clock
      ? {
          clock: {
            option: 'instant',
            time: {origin: CLOCK_ORIGIN_ISO, unit: 'hours' as const},
            zones: ['Europe/Zurich', 'UTC'],
            show: 'time' as const,
            progress: CLOCK_RANGE
          }
        }
      : {})
  };
}

/** Sun visibility and daily exposure over the wide Zermatt terrain window. */
export default defineScene<SunOptions>({
  id: 'sun-and-shadow',
  title: 'When does the sun reach Zermatt?',
  chapter: 'terrain',
  order: 7,
  summary:
    'A clear-sky sun map for Zermatt: horizon-map and exact cast shadows, the daily sun window of the village, and classed sun hours or insolation over the Matterhorn and Monte Rosa cirque.',
  contributors: [
    'GPUSolarPosition',
    'GPUTerrainHorizon',
    'GPUSolarShadowMask',
    'GPUTerrainCastShadow',
    'GPUSolarIrradiance',
    'GPUTerrainDerivatives'
  ],
  datasets: [
    {id: 'alps-dem-wide', role: 'elevation (Terrarium, Web Mercator)'},
    {id: 'alps-context', role: 'glaciers and place names (OpenStreetMap)'}
  ],
  initialView: {...TERRAIN_FRAMES.valley},
  basemap: ground('relief'),
  furniture: furnitureFor('Sun visibility · clear sky · wide terrain window', true),

  options: [
    {
      kind: 'slider',
      id: 'dayOfYear',
      label: 'Date',
      group: 'Sun position',
      apply: 'param',
      min: 1,
      max: 365,
      step: 1,
      default: DATE_PRESETS.december,
      format: value => formatDay(value),
      marks: [
        {value: DATE_PRESETS.december, label: '21 Dec'},
        {value: DATE_PRESETS.march, label: '21 Mar'},
        {value: DATE_PRESETS.june, label: '21 Jun'}
      ],
      help: 'The local day used for the sun position and day integral. The map keeps the same class scale unless you ask it to fit this day.'
    },
    {
      kind: 'slider',
      id: 'hour',
      label: 'Local time',
      group: 'Sun position',
      apply: 'param',
      min: 0,
      max: 24,
      step: 0.05,
      default: 11,
      format: value => formatHour(value),
      disabledWhen: state => state.product === 'sun-hours' || state.product === 'insolation',
      help: 'Civil time in Europe/Zurich. Moving it rewrites the sun and shadow settings; the horizon map remains reusable.'
    },
    {
      kind: 'toggle',
      id: 'animate',
      label: 'Animate the daylight window',
      group: 'Sun position',
      apply: 'param',
      default: false,
      disabledWhen: state => state.product === 'sun-hours' || state.product === 'insolation',
      help: 'Plays from shortly before sunrise to shortly after sunset. It changes shadow parameters, not horizon-map topology.'
    },
    {
      kind: 'slider',
      id: 'animationSpeed',
      label: 'Animation speed',
      group: 'Sun position',
      apply: 'param',
      min: 0.25,
      max: 6,
      step: 0.25,
      default: 1.5,
      unit: 'h/s',
      disabledWhen: state => !state.animate,
      help: 'Local daylight hours advanced each second.'
    },
    {
      kind: 'select',
      id: 'light',
      label: 'Light',
      group: 'Light',
      apply: 'param',
      display: 'segmented',
      default: 'map',
      disabledWhen: state => state.product !== 'light',
      options: [
        {value: 'map', label: 'Map light (NW)'},
        {value: 'sun', label: 'Real sun'}
      ],
      help: 'Map light is fixed north-west illumination for readable relief; real sun follows the date and local time.'
    },
    {
      kind: 'select',
      id: 'product',
      label: 'Map product',
      group: 'Display',
      apply: 'param',
      display: 'chips',
      default: 'shadow',
      options: [
        {value: 'shadow', label: 'Shadow'},
        {value: 'light', label: 'Light'},
        {value: 'sun-hours', label: 'Sun hours'},
        {value: 'insolation', label: 'Insolation'}
      ],
      help: 'Shadow makes visibility visible; light multiplies it into relief; day products integrate every sun-table row.'
    },
    {
      kind: 'select',
      id: 'method',
      label: 'Shadow method',
      group: 'Shadow',
      apply: 'param',
      display: 'segmented',
      default: 'horizon',
      disabledWhen: state => state.product !== 'shadow',
      options: [
        {value: 'horizon', label: 'Horizon map'},
        {value: 'cast', label: 'Exact cast'},
        {value: 'difference', label: 'Difference'}
      ],
      help: 'The horizon map stores many directions for a moving sun. An exact cast sweeps one direction without storing a map.'
    },
    {
      kind: 'toggle',
      id: 'showSearch',
      label: 'Show horizon search',
      group: 'Horizon map',
      apply: 'param',
      default: false,
      disabledWhen: state => state.product !== 'shadow' || state.method !== 'horizon',
      help: 'Shows the search radius around Zermatt, or a cell you click, and marks the stored sectors in the sky diagram.'
    },
    {
      kind: 'select',
      id: 'horizonRadius',
      label: 'Horizon search radius',
      group: 'Horizon map',
      apply: 'compile',
      display: 'segmented',
      default: '1024',
      options: [
        {value: '128', label: 'Short'},
        {value: '256', label: 'Near'},
        {value: '512', label: 'Far'},
        {value: '1024', label: 'Wide'},
        {value: '1536', label: 'Farthest'}
      ],
      help: 'Maximum ray length in DEM cells. A short radius can stop a distant peak’s shadow before it reaches the valley.'
    },
    {
      kind: 'select',
      id: 'horizonDirections',
      label: 'Horizon directions',
      group: 'Horizon map',
      apply: 'compile',
      default: '16',
      expert: true,
      options: [
        {value: '8', label: '8 sectors'},
        {value: '16', label: '16 sectors'},
        {value: '32', label: '32 sectors'}
      ],
      help: 'Directions stored per cell. More sectors reduce interpolation error but increase memory.'
    },
    {
      kind: 'slider',
      id: 'softness',
      label: 'Sun-disc softness',
      group: 'Shadow',
      apply: 'param',
      min: 0,
      max: 8,
      step: 0.5,
      default: 1,
      unit: '× disk',
      format: value => (value === 0 ? 'Hard edge' : `${value.toFixed(1)} × solar disc`),
      help: 'Widens the solar disc used for penumbra. One is the apparent disc; zero makes a hard edge.'
    },
    {
      kind: 'select',
      id: 'castRadius',
      label: 'Exact cast reach',
      group: 'Exact cast',
      apply: 'compile',
      default: '2048',
      expert: true,
      disabledWhen: state => state.product !== 'shadow' || state.method === 'horizon',
      options: [
        {value: '512', label: '512 cells'},
        {value: '1024', label: '1,024 cells'},
        {value: '2048', label: '2,048 cells'},
        {value: 'tile', label: 'Tile diagonal'}
      ],
      help: 'Farthest exact shadow a single sweep can cast; changing it rebuilds the sweep.'
    },
    {
      kind: 'select',
      id: 'scale',
      label: 'Day-product scale',
      group: 'Day products',
      apply: 'param',
      display: 'segmented',
      default: 'june',
      disabledWhen: state => state.product !== 'sun-hours' && state.product !== 'insolation',
      options: [
        {value: 'june', label: 'Same scale as June'},
        {value: 'day', label: 'Fit this day'}
      ],
      help: 'The common scale makes dates comparable. Fitting uses the day’s available sun.'
    },
    {
      kind: 'select',
      id: 'display',
      label: 'Day-product colours',
      group: 'Day products',
      apply: 'param',
      display: 'segmented',
      default: 'classes',
      disabledWhen: state => state.product !== 'sun-hours' && state.product !== 'insolation',
      options: [
        {value: 'classes', label: 'Classes'},
        {value: 'continuous', label: 'Continuous'}
      ],
      help: 'Classes retain the decision thresholds in the legend; continuous colour smooths them.'
    },
    {
      kind: 'slider',
      id: 'sunIntensity',
      label: 'Direct-sun intensity',
      group: 'Light',
      apply: 'param',
      min: 0,
      max: 2,
      step: 0.05,
      default: 1,
      expert: true,
      disabledWhen: state => state.product !== 'light' || state.light !== 'sun',
      help: 'Multiplier of direct light in the real-sun relief composite.'
    },
    {
      kind: 'slider',
      id: 'ambient',
      label: 'Diffuse sky light',
      group: 'Light',
      apply: 'param',
      min: 0,
      max: 0.8,
      step: 0.05,
      default: 0.2,
      expert: true,
      disabledWhen: state => state.product !== 'light' || state.light !== 'sun',
      help: 'Sky light in shade, weighted by how much sky the cell sees.'
    },
    {
      kind: 'slider',
      id: 'exposure',
      label: 'Real-sun exposure',
      group: 'Light',
      apply: 'param',
      min: 0.8,
      max: 3,
      step: 0.1,
      default: 1.7,
      expert: true,
      disabledWhen: state => state.product !== 'light' || state.light !== 'sun',
      help: 'Exposure of the multiply composite, clamped before snow can turn pure white.'
    },
    {
      kind: 'select',
      id: 'directIrradiance',
      label: 'Direct-sun model',
      group: 'Day products',
      apply: 'param',
      default: 'meinel',
      expert: true,
      disabledWhen: state => state.product !== 'sun-hours' && state.product !== 'insolation',
      options: [
        {value: 'meinel', label: 'Meinel clear sky'},
        {value: '1000', label: 'Constant 1000 W/m²'},
        {value: '600', label: 'Constant 600 W/m²'}
      ],
      help: 'Direct normal irradiance used only by the insolation integral.'
    },
    {
      kind: 'slider',
      id: 'diffuseIrradiance',
      label: 'Diffuse irradiance',
      group: 'Day products',
      apply: 'param',
      min: 0,
      max: 300,
      step: 10,
      default: 100,
      unit: 'W/m²',
      expert: true,
      disabledWhen: state => state.product !== 'sun-hours' && state.product !== 'insolation',
      help: 'Diffuse clear-sky energy scaled by the sky-view factor; it is intentionally the weakest term.'
    },
    {
      kind: 'select',
      id: 'tableStep',
      label: 'Sun-table interval',
      group: 'Day products',
      apply: 'param',
      default: '5',
      expert: true,
      disabledWhen: state => state.product !== 'sun-hours' && state.product !== 'insolation',
      options: [
        {value: '5', label: '5 minutes'},
        {value: '10', label: '10 minutes'},
        {value: '15', label: '15 minutes'},
        {value: '30', label: '30 minutes'}
      ],
      help: 'Spacing of the precomputed sun positions that the irradiance stage integrates.'
    },
    {
      kind: 'toggle',
      id: 'refraction',
      label: 'Atmospheric refraction',
      group: 'Engine',
      apply: 'compile',
      default: true,
      expert: true,
      help: 'Uses the NOAA apparent-sun correction near the horizon; it rebuilds the sun graph.'
    },
    {
      kind: 'slider',
      id: 'horizonGrowth',
      label: 'Horizon ray growth',
      group: 'Engine',
      apply: 'compile',
      min: 1,
      max: 1.3,
      step: 0.02,
      default: 1.1,
      expert: true,
      disabledWhen: state => state.horizonAlgorithm === 'sweep',
      help: 'Growth between samples on a horizon ray. A sweep instead uses an exact digital-line upper hull.'
    },
    {
      kind: 'select',
      id: 'horizonFormat',
      label: 'Horizon storage',
      group: 'Engine',
      apply: 'compile',
      default: 'unorm16',
      expert: true,
      options: [
        {value: 'unorm16', label: 'Packed 16-bit'},
        {value: 'float32', label: 'Float32'}
      ],
      help: 'Packed angles halve the horizon-map footprint; float32 keeps raw angles.'
    },
    {
      kind: 'select',
      id: 'horizonAlgorithm',
      label: 'Horizon algorithm',
      group: 'Engine',
      apply: 'compile',
      default: 'march',
      expert: true,
      options: [
        {value: 'march', label: 'Ray march'},
        {value: 'sweep', label: 'Exact sweep'}
      ],
      help: 'March samples a ray; sweep constructs the upper horizon exactly along digital lines.'
    },
    {
      kind: 'slider',
      id: 'twilight',
      label: 'Daylight threshold',
      group: 'Engine',
      apply: 'param',
      min: -18,
      max: 0,
      step: 0.1,
      default: -0.833,
      unit: '°',
      expert: true,
      help: 'Altitude below which the solar-position daylight flag reports night.'
    },
    {
      kind: 'slider',
      id: 'instant',
      label: 'Map-clock instant',
      group: 'Engine',
      apply: 'param',
      min: CLOCK_RANGE[0],
      max: CLOCK_RANGE[1],
      step: 0.05,
      default: 8506,
      expert: true,
      help: 'Derived civil-time position used only by the map clock.'
    },
    {
      kind: 'button',
      id: 'measure',
      label: 'Time this product',
      group: 'Engine',
      expert: true,
      help: 'Runs every GPU stage needed by the displayed product outside the animation frame.'
    }
  ],

  readouts: [
    {id: 'zermattSunHours', label: 'Zermatt direct sun', emphasis: 'tile'},
    {id: 'firstSun', label: 'First direct sun', emphasis: 'tile'},
    {
      id: 'pixelTimeline',
      label: 'Zermatt through the day',
      kind: 'chart',
      placement: 'map',
      mapCorner: 'bottom-right',
      help: 'A local-day timeline of night, terrain shadow and direct sun at the village.'
    },
    {
      id: 'skyDiagram',
      label: 'Sun path and skyline',
      kind: 'chart',
      placement: 'map',
      mapCorner: 'top-right',
      help: 'Polar sky diagram: azimuth is around the ring and altitude moves toward its centre.'
    },
    {id: 'sunAltitude', label: 'Sun altitude', emphasis: 'tile'},
    {id: 'sunAzimuth', label: 'Sun azimuth', emphasis: 'tile'},
    {id: 'zermattNow', label: 'Zermatt now', emphasis: 'tile'},
    {id: 'horizonMemory', label: 'Horizon-map memory', emphasis: 'tile'},
    {id: 'horizonSize', label: 'Horizon-map layout', hood: true},
    {id: 'castMemory', label: 'Exact-cast memory', hood: true},
    {id: 'differenceShare', label: 'Cells that disagree', emphasis: 'tile'},
    {id: 'tileMean', label: 'Window average', emphasis: 'tile'},
    {
      id: 'sunHistogram',
      label: 'Map-value distribution',
      kind: 'chart',
      help: 'Cells by the same sun-hour or insolation classes the map uses. The marker is Zermatt.'
    },
    {id: 'sunHoursCheck', label: 'Zermatt map / CPU check', hood: true},
    {id: 'sunCpuGpu', label: 'GPU / float64 sun check', hood: true},
    {id: 'localTime', label: 'Local time', hood: true},
    {id: 'grid', label: 'Raster', hood: true},
    {id: 'timing', label: 'GPU time', hood: true}
  ],

  pipeline: [
    {id: 'sun', label: 'Sun position', detail: 'NOAA position for the local date and time'},
    {
      id: 'horizon',
      label: 'Horizon map',
      detail: 'Highest terrain angle in every stored direction'
    },
    {id: 'shadow', label: 'Shadow mask', detail: 'Sun altitude against the interpolated skyline'},
    {id: 'irradiance', label: 'Irradiance', detail: 'A day of sun positions summed on every slope'}
  ],

  legends: getSunLegends,

  story: [
    {
      id: 'question',
      title: 'Zermatt waits for the winter sun',
      headline: 'Zermatt waits for the winter sun',
      textAlternative:
        'A paper-toned relief map of the Zermatt valley under a transparent indigo shadow veil. Zermatt is marked by a gold eye, with Matterhorn, Gornergrat, Riffelberg, Dufourspitze and Gornergletscher labelled.',
      body: 'At Zermatt, the village gets **{{zermattSunHours}}** of direct sun on this day; first light arrives at **{{firstSun}}**. The indigo veil is terrain shadow, and clear ground is in sun. The timeline follows the gold village eye through night, shade and direct light.\n\n*A shadow is a viewshed from the sun.*',
      optionsMode: 'fresh',
      options: {
        product: 'shadow',
        method: 'horizon',
        dayOfYear: DATE_PRESETS.december,
        hour: 11,
        animate: false,
        showSearch: false
      },
      controls: ['dayOfYear', 'hour'],
      readouts: ['zermattSunHours', 'firstSun', 'pixelTimeline'],
      camera: {...TERRAIN_FRAMES.valley, transitionMs: 1400},
      furniture: furnitureFor('Sun visibility · clear sky · Zermatt valley', true),
      annotations: getStepPlaces(STEP_PLACES.question),
      stage: 'shadow',
      highlight: {readout: 'zermattSunHours'}
    },
    {
      id: 'two-lights',
      title: 'Map light is not real sunlight',
      headline: 'Map light comes from north-west; the sun does not',
      textAlternative:
        'The shaded alpine relief is lit from the north-west, with an arrow labelled Map light, north-west. The adjacent polar inset shows the winter sun path and Zermatt skyline.',
      body: 'Relief uses a fixed north-west light so landforms read consistently. Real sunlight has altitude **{{sunAltitude}}** and azimuth **{{sunAzimuth}}** now; choose **Light** below to multiply it into the ground instead. The polar inset makes the comparison tangible: a gold dot within grey terrain is still behind the mountains.\n\n*Map light describes relief; real light answers a solar question.*',
      optionsMode: 'fresh',
      options: {
        product: 'light',
        light: 'map',
        dayOfYear: DATE_PRESETS.december,
        hour: 11,
        animate: false
      },
      controls: ['light', 'hour'],
      readouts: ['sunAltitude', 'sunAzimuth', 'skyDiagram'],
      camera: {...TERRAIN_FRAMES.valley, transitionMs: 1200},
      furniture: furnitureFor('Map light compared with the real sun', true),
      annotations: getStepPlaces(STEP_PLACES.light),
      stage: 'sun'
    },
    {
      id: 'horizon-map',
      title: 'Each cell stores its skyline',
      headline: 'Each cell stores its skyline',
      textAlternative:
        'An indigo shadow veil over the relief, a dashed circle around the gold Zermatt eye marking the horizon search distance, and sector points around the skyline in the polar inset.',
      body: '`GPUTerrainHorizon` keeps the highest terrain angle in each direction for every cell. This map stores **{{horizonMemory}}** as **{{horizonSize}}**; click elsewhere to inspect that cell’s skyline. Shorten **Horizon search radius** and distant peaks stop casting their shadows.\n\n*A shadow is a viewshed from the sun.*',
      optionsMode: 'fresh',
      options: {
        product: 'shadow',
        method: 'horizon',
        dayOfYear: DATE_PRESETS.december,
        hour: 11,
        horizonRadius: '1024',
        showSearch: true
      },
      controls: ['horizonRadius', 'showSearch'],
      readouts: ['horizonMemory', 'skyDiagram', 'timing'],
      camera: {...TERRAIN_FRAMES.valley, transitionMs: 1200},
      furniture: furnitureFor('Horizon angle · stored directions · search distance'),
      annotations: getStepPlaces(STEP_PLACES.horizon),
      stage: 'horizon',
      highlight: {readout: 'horizonMemory'}
    },
    {
      id: 'cast-exactly',
      title: 'One sun, one sweep, no memory',
      headline: 'One sun, one sweep, no memory',
      textAlternative:
        'The terrain map is coloured in orange difference classes where the interpolated horizon-map shadow differs from the exact cast shadow, with unremarkable cells transparent over relief.',
      body: '`GPUTerrainCastShadow` sweeps the current sun direction, so it stores **{{castMemory}}** instead of a horizon map. Choose **Shadow method** to compare it with stored directions; **{{differenceShare}}** of cells differ.\n\n*Precompute many directions, or sweep one direction exactly.*',
      optionsMode: 'fresh',
      options: {
        product: 'shadow',
        method: 'difference',
        dayOfYear: DATE_PRESETS.december,
        hour: 11,
        showSearch: false
      },
      controls: ['method'],
      readouts: ['differenceShare', 'castMemory'],
      camera: {...TERRAIN_FRAMES.valley, transitionMs: 1200},
      furniture: furnitureFor('Exact cast against the stored horizon'),
      annotations: getStepPlaces(STEP_PLACES.cast),
      stage: 'shadow',
      highlight: {readout: 'differenceShare'}
    },
    {
      id: 'one-day',
      title: 'The east wall wakes first',
      headline: 'The east wall wakes first',
      textAlternative:
        'An animated indigo terrain-shadow veil moves across the Zermatt valley. A local Europe/Zurich clock appears over the map, alongside a pixel timeline for the gold Zermatt eye.',
      body: 'Play the winter daylight window. The clock and **Zermatt now** update while the same horizon map answers each new sun position; no horizon topology is rebuilt. Switch **Date** to compare the short winter path with higher spring and summer paths.\n\n*The skyline stays; the sun moves.*',
      optionsMode: 'fresh',
      options: {
        product: 'shadow',
        method: 'horizon',
        dayOfYear: DATE_PRESETS.december,
        hour: 9,
        animate: true,
        showSearch: false
      },
      controls: ['animate', 'dayOfYear'],
      readouts: ['zermattNow', 'pixelTimeline', 'localTime'],
      camera: {...TERRAIN_FRAMES.valley, transitionMs: 1200},
      furniture: furnitureFor('Animated terrain shadow · Europe/Zurich', true),
      annotations: getStepPlaces(STEP_PLACES.day),
      stage: 'shadow',
      highlight: {readout: 'pixelTimeline'}
    },
    {
      id: 'sun-hours',
      title: 'Sun hours are the village map',
      headline: 'Sun hours: the map the village wants',
      textAlternative:
        'A classed cold-to-warm map of hours of direct sun across the Zermatt valley, with pale relief beneath. A histogram has the same class boundaries and a Zermatt marker.',
      body: '`GPUSolarIrradiance` sums a day of visible sun for every slope. Zermatt receives **{{zermattSunHours}}**; the window average is **{{tileMean}}**. Keep **Day-product scale** common when dates must compare, or fit it to inspect one day. Choose Sun hours or Insolation with **Map product**.\n\nClear sky only: clouds, trees and buildings are absent; diffuse sky is the weakest term and snow albedo is not modelled.\n\n*Class a continuous field at the decision threshold.*',
      optionsMode: 'fresh',
      options: {
        product: 'sun-hours',
        dayOfYear: DATE_PRESETS.december,
        scale: 'june',
        display: 'classes',
        animate: false
      },
      controls: ['product', 'scale', 'display'],
      readouts: ['zermattSunHours', 'tileMean', 'sunHistogram'],
      camera: {...TERRAIN_FRAMES.valley, transitionMs: 1200},
      furniture: furnitureFor('Hours of direct sun · common day scale'),
      annotations: getStepPlaces(STEP_PLACES.hours),
      stage: 'irradiance',
      highlight: {readout: 'sunHistogram'}
    }
  ],

  about: {
    what: '`GPUSolarPosition` evaluates the NOAA sun position. `GPUTerrainHorizon` records the highest terrain angle by direction; `GPUSolarShadowMask` compares that skyline with the current sun, while `GPUTerrainCastShadow` makes one direction exact without storing a map. `GPUSolarIrradiance` sums a daily sun table into direct-sun hours and clear-sky slope insolation.',
    why: 'Winter sun access is a question people can read in hours, but it starts as a visibility calculation. Keeping the horizon map separate makes an animated sun cheap; the exact cast reveals the cost and error of approximating direction.',
    howToRead:
      'For shadows, transparent ground is directly sunlit and the indigo veil is hidden by terrain; penumbra is blended between them. For day products, warmer classes mean more direct sun or energy. The relief remains context, with glaciers and named landforms above it.'
  },

  create: async ctx => (await import('./sun-and-shadow.compute')).createSunAndShadow(ctx),

  snippet: state => `import {
  GPUSolarPosition, GPUTerrainHorizon, GPUSolarShadowMask, GPUTerrainCastShadow, GPUSolarIrradiance
} from '@luma.gl/experimental/gpu-terrain';

graph.add(new GPUTerrainHorizon({
  width, height, elevation, directionCount: ${state.horizonDirections},
  maximumRadius: ${state.horizonRadius}, horizon, skyViewFactor
}));
graph.add(new GPUSolarPosition({positions, settings: sunSettings, azimuth, altitude, daylight}));
graph.add(new GPUSolarShadowMask({
  width, height, directionCount: ${state.horizonDirections}, horizon,
  settings: maskSettings, slope, aspect, skyViewFactor, sunVisibility, illumination
}));
graph.add(new GPUSolarIrradiance({
  width, height, directionCount: ${state.horizonDirections}, horizon, sunTable,
  settings: irradianceSettings, slope, aspect, skyViewFactor, sunHours, insolation
}));
const compiled = graph.compile();
sunSettings.write({timestamp});
maskSettings.write({azimuthDegrees, altitudeDegrees, angularRadiusDegrees});
sunTable.write(daySamples);
compiled.encode(commandEncoder);`
});
