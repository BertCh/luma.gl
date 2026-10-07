// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene, type LegendSpec} from '../scene';
import {
  BAND_COUNT,
  BAND_PALETTE,
  LAND_COVER_CLASSES,
  LAND_COVER_PALETTE,
  NO_COST_LIMIT_MINUTES,
  NO_DISTANCE_CAP_KILOMETERS,
  START_PALETTE,
  type LeastCostOptions
} from './least-cost-paths.compute';

const formatMinutes = (minutes: number): string =>
  minutes >= 60
    ? `${Math.floor(minutes / 60)} h${minutes % 60 ? ` ${minutes % 60} min` : ''}`
    : `${minutes} min`;

const CANYON_VIEW = {longitude: -112.1, latitude: 36.095, zoom: 11.4} as const;
const DIXIE_VIEW = {longitude: -121.0, latitude: 40.17, zoom: 11.1} as const;

/**
 * Least-cost travel over the Grand Canyon and the Dixie fire area: metadata, options and narrative
 * (light); the GPU work lives in `least-cost-paths.compute.ts`.
 */
export default defineScene<LeastCostOptions>({
  id: 'least-cost-paths',
  title: 'The easiest way down, and the way around a fire',
  chapter: 'hydrology',
  order: 3,
  summary:
    'Price every cell by slope, cliffs and land cover, spread travel time from a start on the GPU, snap the cheapest route to a draggable destination and compare it with the straight line.',
  contributors: [
    'GPUCostDistance',
    'GPUCostDistancePath',
    'GPUDistanceField',
    'GPUTerrainDerivatives'
  ],
  datasets: [
    {id: 'grand-canyon-dem', role: 'slope friction: South Rim to Phantom Ranch'},
    {id: 'dixie-fire', role: 'slope and ESA WorldCover friction: Greenville to Lake Almanor'}
  ],
  initialView: CANYON_VIEW,

  options: [
    {
      kind: 'slider',
      id: 'speed',
      label: 'Walking pace on flat ground',
      group: 'Travel',
      apply: 'param',
      min: 2,
      max: 8,
      step: 0.5,
      default: 5,
      unit: 'km/h',
      help: 'Re-prices the whole map by writing the GPUCostDistance frictionParameters (scale = 6 / pace). No recompile.'
    },
    {
      kind: 'slider',
      id: 'slopePenalty',
      label: 'Slope penalty',
      group: 'Travel',
      apply: 'param',
      min: 0,
      max: 2,
      step: 0.1,
      default: 1,
      format: value => (value === 0 ? '0 (ignore slope)' : `${value.toFixed(1)}× Tobler`),
      help: 'Scales the exponent of Tobler’s hiking function, time per metre × exp(3.5 · tan slope). 0 gives straight-line walking everywhere; 1 is Tobler; 2 punishes steep ground twice as hard.'
    },
    {
      kind: 'slider',
      id: 'maxSlope',
      label: 'Cliff limit',
      group: 'Travel',
      apply: 'param',
      min: 15,
      max: 80,
      step: 1,
      default: 38,
      unit: '°',
      help: 'Cells steeper than this are impassable barriers (NaN friction). Lower it and routes must find the side canyons and breaks in the cliff bands.'
    },
    {
      kind: 'slider',
      id: 'costLimit',
      label: 'Time budget',
      group: 'Travel',
      apply: 'param',
      min: 30,
      max: NO_COST_LIMIT_MINUTES,
      step: 30,
      default: NO_COST_LIMIT_MINUTES,
      format: value => (value >= NO_COST_LIMIT_MINUTES ? 'no limit' : formatMinutes(value)),
      help: 'Cells that cost more than this stay unreached (GPUCostDistance costLimit). Per-frame parameter: use it for a one-hour reach or a day-hike budget.'
    },
    {
      kind: 'slider',
      id: 'bandMinutes',
      label: 'Band width',
      group: 'Travel',
      apply: 'param',
      min: 15,
      max: 180,
      step: 15,
      default: 60,
      format: formatMinutes,
      help: 'Width of each of the eight isochrone bands, written to the bandThresholds buffer.'
    },
    {
      kind: 'slider',
      id: 'rangeHours',
      label: 'Color range of travel time',
      group: 'Travel',
      apply: 'param',
      min: 1,
      max: 12,
      step: 1,
      default: 6,
      unit: 'h',
      help: 'Travel time mapped to the end of the ramp in the accumulated-time display.'
    },
    {
      kind: 'select',
      id: 'placement',
      label: 'A click places',
      group: 'Endpoints',
      apply: 'param',
      default: 'destination',
      help: 'Click either landscape to place its destination, replace its start, or add another start (up to eight). Drag the red destination to watch the route re-snap.',
      options: [
        {value: 'destination', label: 'Destination (red)'},
        {value: 'start', label: 'Start (replaces the starts)'},
        {value: 'add-start', label: 'Another start'}
      ]
    },
    {
      kind: 'slider',
      id: 'extraDelay',
      label: 'Head start delay of extra starts',
      group: 'Endpoints',
      apply: 'param',
      min: 0,
      max: 240,
      step: 15,
      default: 0,
      format: formatMinutes,
      help: 'Extra starts begin with this initial cost (GPUCostDistance sourceCosts): a crew that sets off later. The first start always begins at 0.'
    },
    {
      kind: 'toggle',
      id: 'secondStart',
      label: 'Add a second start',
      group: 'Endpoints',
      apply: 'param',
      default: false,
      help: 'Adds the North Rim (Bright Angel Point) in the canyon and Round Valley in the Dixie area as an extra start. Costs spread from the nearest start; distance and allocation use both.'
    },
    {
      kind: 'button',
      id: 'resetEndpoints',
      label: 'Restore the story start and destination',
      group: 'Endpoints'
    },
    {
      kind: 'select',
      id: 'distanceAlgorithm',
      label: 'Distance algorithm',
      group: 'Distance field',
      apply: 'compile',
      default: 'exact',
      help: 'Compile-time choice of GPUDistanceField mode; each variant is compiled the first time you pick it.',
      options: [
        {
          value: 'exact',
          label: 'Exact (Felzenszwalb-Huttenlocher)',
          help: 'Separable lower envelopes: exact nearest seed per cell.'
        },
        {
          value: 'jump-flood-0',
          label: 'Jump flood, no refinement',
          help: 'Cheapest preview; a few cells can pick the wrong seed.'
        },
        {
          value: 'jump-flood-1',
          label: 'Jump flood + 1 refinement pass',
          help: 'JFA+1: misallocated at most 0.009 % of cells in the contributor tests.'
        },
        {
          value: 'jump-flood-2',
          label: 'Jump flood + 2 refinement passes',
          help: 'JFA+2: exact in the measured scenes, not guaranteed.'
        }
      ]
    },
    {
      kind: 'slider',
      id: 'distanceCap',
      label: 'Distance cap',
      group: 'Distance field',
      apply: 'param',
      min: 1,
      max: NO_DISTANCE_CAP_KILOMETERS,
      step: 1,
      default: NO_DISTANCE_CAP_KILOMETERS,
      format: value => (value >= NO_DISTANCE_CAP_KILOMETERS ? 'no cap' : `${value} km`),
      help: 'Cells farther than this from every seed get infinite distance and no allocation (the maxDistance setting).'
    },
    {
      kind: 'select',
      id: 'distanceSeeds',
      label: 'Seeds (Dixie)',
      group: 'Distance field',
      apply: 'param',
      default: 'starts',
      help: 'What the Dixie distance field measures from: the start points, any WorldCover water (a helicopter dip site is never far), or built-up cells. Raster seeds use the seedMask input.',
      options: [
        {value: 'starts', label: 'The start points'},
        {value: 'water', label: 'Nearest water (WorldCover class 80)'},
        {value: 'built-up', label: 'Nearest built-up cell (class 50)'}
      ]
    },
    {
      kind: 'button',
      id: 'timeDistance',
      label: 'Time exact vs jump-flood on the canyon grid',
      group: 'Distance field'
    },
    {
      kind: 'slider',
      id: 'forest',
      label: 'Tree cover pace factor',
      group: 'Land cover (Dixie)',
      apply: 'param',
      min: 0.5,
      max: 5,
      step: 0.1,
      default: 1.6,
      format: value => `${value.toFixed(1)}×`,
      help: 'Time multiplier for WorldCover tree cover. Dense conifer slows a hiker or a crew cutting line. Multiplies the slope friction.'
    },
    {
      kind: 'slider',
      id: 'openGround',
      label: 'Open ground pace factor',
      group: 'Land cover (Dixie)',
      apply: 'param',
      min: 0.5,
      max: 5,
      step: 0.1,
      default: 1,
      format: value => `${value.toFixed(1)}×`,
      help: 'Grassland, shrub, cropland and bare ground.'
    },
    {
      kind: 'slider',
      id: 'builtUp',
      label: 'Built-up (roads, town) pace factor',
      group: 'Land cover (Dixie)',
      apply: 'param',
      min: 0.2,
      max: 3,
      step: 0.1,
      default: 0.7,
      format: value => `${value.toFixed(1)}×`,
      help: 'Below 1 makes towns faster than forest: WorldCover has no road layer, so built-up cells stand in for them.'
    },
    {
      kind: 'slider',
      id: 'wetland',
      label: 'Wetland pace factor',
      group: 'Land cover (Dixie)',
      apply: 'param',
      min: 0.5,
      max: 8,
      step: 0.5,
      default: 3,
      format: value => `${value.toFixed(1)}×`,
      help: 'Herbaceous wetland and mangrove.'
    },
    {
      kind: 'toggle',
      id: 'waterBarrier',
      label: 'Water is impassable',
      group: 'Land cover (Dixie)',
      apply: 'param',
      default: true,
      help: 'Permanent water is a barrier (NaN friction). Off: water is passable at four times the pace.'
    },
    {
      kind: 'select',
      id: 'display',
      label: 'Map shows',
      group: 'Display',
      apply: 'param',
      default: 'bands',
      help: 'Which output buffer is drawn. The route, the start points and the destination are always on top.',
      options: [
        {value: 'bands', label: 'Travel-time bands (isochrones)'},
        {value: 'cost', label: 'Accumulated travel time'},
        {value: 'friction', label: 'Friction: minutes per kilometre'},
        {value: 'distance', label: 'Straight-line distance (distance field)'},
        {value: 'allocation', label: 'Nearest start by straight line'},
        {value: 'detour', label: 'Detour: travel time vs straight line'},
        {value: 'landcover', label: 'Land cover (Dixie only)'}
      ]
    },
    {
      kind: 'slider',
      id: 'overlayOpacity',
      label: 'Overlay opacity',
      group: 'Display',
      apply: 'param',
      min: 0.3,
      max: 1,
      step: 0.05,
      default: 0.8,
      format: value => `${Math.round(value * 100)} %`,
      help: 'Blend of the chosen raster over the shaded relief.'
    },
    {
      kind: 'toggle',
      id: 'showPath',
      label: 'Show least-cost route',
      group: 'Display',
      apply: 'param',
      default: true
    },
    {
      kind: 'toggle',
      id: 'showStraightLine',
      label: 'Show straight line',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Draws the straight line from the nearest start to the destination, to see how much the terrain forces you off it.'
    },
    {
      kind: 'toggle',
      id: 'showBarriers',
      label: 'Mark impassable cells',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Shades the cells the cost surface treats as barriers: slopes above the cliff limit, and water in the Dixie area.'
    }
  ],

  story: [
    {
      id: 'question',
      title: 'How long is the walk to Phantom Ranch?',
      body: 'Grand Canyon Village is on the South Rim at 2,100 m; Phantom Ranch sits beside Bright Angel Creek at 830 m, 8 km away as the crow flies. A hiker cannot cross the Redwall and Supai cliffs, so the real walk follows the breaks in them. **Which way is cheapest, and how long does it take?**\n\nThe colored bands are **travel time from the start** (white dot): each band is an hour of walking, yellow nearest. The red line is the cheapest route to the red dot. Both are computed on the GPU on a 31 m grid from the 15 m DEM. Change **Map shows** or the walking pace below to see what else the model can draw.',
      camera: {...CANYON_VIEW, transitionMs: 1400},
      options: {display: 'bands', placement: 'destination'},
      controls: ['display', 'speed'],
      readouts: ['canyonTime', 'canyonPath'],
      highlight: {readout: 'canyonTime'},
      callout: {coordinate: [-112.0953, 36.107], text: 'Phantom Ranch'}
    },
    {
      id: 'friction',
      title: 'Price the ground: friction',
      body: 'A cost surface starts from **friction**: the cost of crossing one metre of each cell. Here it is walking minutes per metre from **Tobler’s hiking function**, `time = 0.01 min/m × exp(3.5 · tan(slope))`, from the slope that `GPUTerrainDerivatives` computes once. Flat ground at 6 km/h costs 10 minutes per kilometre; a 20 degree slope costs about 3.5 times that.\n\nSlopes above the **Cliff limit** below (38 degrees) are barriers: they get NaN friction, which `GPUCostDistance` treats as impassable, and are shaded red. The legend is minutes per kilometre at the chosen **Walking pace on flat ground**. Move **Slope penalty** to 0 to see the cliffs matter even when slope costs nothing.',
      options: {display: 'friction'},
      controls: ['slopePenalty', 'maxSlope', 'speed'],
      camera: {longitude: -112.115, latitude: 36.08, zoom: 12.2, transitionMs: 1600}
    },
    {
      id: 'isochrones',
      title: 'Spread travel time: GPUCostDistance',
      body: '`GPUCostDistance` computes the **least accumulated cost to every cell** from the start. Moves are 8-connected; the cost of a move is its ground length times the mean friction of the two cells. It is a tiled min-relaxation, so a front advances 16 cells per iteration and switches to directional sweeps for long corridors. The *Converged* readout reports iterations used out of the compile-time limit; nothing is read back to decide when to stop.\n\n**Time budget** below (a parameter, no recompile) cuts the spread off: drag it to 4 hours and the unreached canyon floor disappears. **Walking pace on flat ground** re-prices the map through `frictionParameters`. Bands are cost thresholds in a buffer (`bandThresholds`), so **Band width** is also a write.',
      options: {display: 'bands', costLimit: 480},
      camera: {...CANYON_VIEW, transitionMs: 1400},
      controls: ['costLimit', 'speed', 'bandMinutes'],
      readouts: ['canyonConverged', 'canyonBands'],
      highlight: {readout: 'canyonConverged'}
    },
    {
      id: 'route',
      title: 'Drag the destination: GPUCostDistancePath',
      body: '`GPUCostDistance` also writes **back-links**: for each cell, the direction of the next step on the cheapest way home. `GPUCostDistancePath` follows them from the destination to the start in one GPU thread and returns the cell list, which a small kernel turns into line segments drawn straight from the buffer.\n\n**Drag the red dot** and the route re-snaps in the same frame: no relaxation is repeated, only the path graph runs. The readouts give the time, length, climb and descent along the route, and the **detour**: travel time compared to walking the straight line (switched on with **Show straight line**) over flat ground. Real trails use these breaks: Bright Angel and the Kaibabs ride fault lines.',
      options: {display: 'cost', showStraightLine: true},
      camera: {longitude: -112.1, latitude: 36.085, zoom: 11.9, transitionMs: 1400},
      controls: ['showStraightLine', 'display'],
      readouts: ['canyonTime', 'canyonPath', 'canyonDetour'],
      highlight: {readout: 'canyonDetour'}
    },
    {
      id: 'distance-field',
      title: 'Straight-line distance: GPUDistanceField',
      body: '`GPUDistanceField` is the cost-free counterpart: the **exact Euclidean distance** from every cell to the nearest seed, plus which seed it is (allocation, Voronoi zones). The map now shows the straight-line distance in kilometres from the South Rim *and* the North Rim (Bright Angel Point), the two starts. Compare it with the travel-time bands: the Colorado gorge, two kilometres across but a day to cross, is where the two pictures disagree most.\n\nThe exact mode is the separable Felzenszwalb-Huttenlocher transform. **Distance algorithm** below switches to the jump-flood preview (compiled on first use); press the "Time exact vs jump-flood on the canyon grid" button for GPU timings. **Distance cap** sets `maxDistance`. Pick *Nearest start by straight line* in **Map shows** for the allocation.',
      options: {display: 'distance', secondStart: true, placement: 'add-start'},
      camera: {...CANYON_VIEW, transitionMs: 1400},
      controls: ['display', 'distanceAlgorithm', 'distanceCap', 'timeDistance'],
      readouts: ['distanceTiming'],
      highlight: {readout: 'distanceTiming'}
    },
    {
      id: 'land-cover',
      title: 'Another landscape: land cover as friction (Dixie fire area)',
      body: 'North-east of here the same graphs run on 20 m cells around **Greenville, California**, burned by the 2021 Dixie Fire. Slope is only half the story there: 91 % of the window is conifer forest, which slows a crew; towns are faster; **Lake Almanor** (west edge) is a barrier. The friction kernel multiplies the Tobler slope cost by a **per-class factor** from ESA WorldCover, a table of eleven floats in a parameter buffer.\n\nThe map shows the land cover. The route from Greenville to the Almanor east shore threads valleys and meadows rather than ridges. Try **Tree cover pace factor** at 3 below and watch the route and the bands bend toward open ground.',
      options: {display: 'landcover'},
      camera: {...DIXIE_VIEW, transitionMs: 3000},
      controls: ['display', 'forest', 'builtUp', 'waterBarrier'],
      readouts: ['dixieTime', 'dixiePath'],
      callout: {coordinate: [-120.9507, 40.1394], text: 'Greenville'}
    },
    {
      id: 'limits',
      title: 'Limits, and things to try',
      body: 'Read the model as a **screening tool**. Friction is a symmetric Tobler curve (real uphill and downhill differ), cliffs are a slope threshold at 31 m or 20 m resolution, WorldCover has no roads or trails, the Colorado is a flat, crossable surface in the canyon, and the Dixie raster is drawn square although the UTM grid is rotated 1.3 degrees from north (up to 170 m at the corners). `GPUCostDistance` has no direction-dependent costs and no per-source allocation yet; `GPUDistanceField` allocates by straight line only.\n\nTry, with the controls below: set **Seeds (Dixie)** to *Nearest water* and **Map shows** to *Straight-line distance* (helicopter dip distances); raise **Time budget** to no limit and show the *Detour* map to see where terrain costs most; add a second start with **A click places**; turn off **Water is impassable** to let the route cross the lake.',
      options: {display: 'bands', bandMinutes: 60, showStraightLine: true},
      camera: {...DIXIE_VIEW, transitionMs: 1200},
      controls: ['distanceSeeds', 'display', 'costLimit', 'placement', 'waterBarrier'],
      readouts: ['dixieDetour'],
      highlight: {readout: 'dixieDetour'}
    }
  ],

  about: {
    what: 'A friction kernel (Tobler hiking time x land cover) feeds `GPUCostDistance`, which accumulates travel time from the starts with back-links and isochrone bands; `GPUCostDistancePath` extracts the cheapest route to a destination; `GPUDistanceField` gives the straight-line distance and nearest seed.',
    why: 'Where can a crew be in an hour? Which way around a cliff band is cheapest? How far is water? Least-cost surfaces answer access, evacuation, corridor and logistics questions, and making them parameter-driven makes them explorable.',
    howToRead:
      'Bands and the viridis ramp show travel time from the nearest start (yellow near, purple far). The red line is the cheapest route; the thin line is the straight line. Red-shaded cells are impassable. In the detour display, 1x means the terrain costs nothing extra over flat ground, 4x means four times slower than the straight line.'
  },

  legends: state => {
    const legends: LegendSpec[] = [];
    switch (state.display) {
      case 'cost':
        legends.push({
          kind: 'ramp',
          title: 'Accumulated travel time',
          ramp: 'viridis',
          extent: [0, state.rangeHours * 60],
          unit: 'h',
          format: value => `${(value / 60).toFixed(0)}`
        });
        break;
      case 'bands':
        legends.push({
          kind: 'categories',
          title: 'Travel time from the nearest start',
          entries: BAND_PALETTE.map((color, band) => ({
            color,
            label:
              band === 0
                ? `up to ${formatMinutes(state.bandMinutes)}`
                : `${formatMinutes(band * state.bandMinutes)} to ${formatMinutes((band + 1) * state.bandMinutes)}`
          })),
          note: `Eight bands of ${formatMinutes(state.bandMinutes)}; beyond ${formatMinutes(BAND_COUNT * state.bandMinutes)} is unshaded.`
        });
        break;
      case 'friction':
        legends.push({
          kind: 'ramp',
          title: 'Friction at the chosen pace',
          ramp: 'inferno',
          extent: [5, 60],
          unit: 'min per km'
        });
        break;
      case 'distance':
        legends.push({
          kind: 'ramp',
          title: 'Straight-line distance',
          ramp: 'cividis',
          extent: [0, 20],
          unit: 'km'
        });
        break;
      case 'allocation':
        legends.push({
          kind: 'categories',
          title: 'Nearest start by straight line',
          entries: START_PALETTE.map((color, index) => ({color, label: `start ${index + 1}`}))
        });
        break;
      case 'detour':
        legends.push({
          kind: 'ramp',
          title: 'Detour factor',
          ramp: 'magma',
          extent: [1, 4],
          unit: '× the flat straight-line time',
          format: value => `${value.toFixed(0)}×`
        });
        break;
      case 'landcover':
        legends.push({
          kind: 'categories',
          title: 'Land cover (ESA WorldCover)',
          entries: LAND_COVER_PALETTE.map((color, index) => ({
            color,
            label: LAND_COVER_CLASSES[index]
          }))
        });
        break;
    }
    if (state.showBarriers) {
      legends.push({
        kind: 'categories',
        title: 'Barriers',
        entries: [
          {
            color: [200, 30, 30, 200],
            label: `Impassable: steeper than ${state.maxSlope}°, or water`
          }
        ]
      });
    }
    legends.push({
      kind: 'categories',
      title: 'Markers',
      entries: [
        {color: [255, 190, 60, 255], label: 'Start (one color per start)'},
        {color: [235, 40, 60, 255], label: 'Destination: drag me'},
        {color: [255, 80, 70, 255], label: 'Least-cost route'}
      ]
    });
    return legends;
  },

  readouts: [
    {
      id: 'canyonTime',
      label: 'Canyon: travel time',
      help: 'Accumulated cost at the destination, in hours and minutes of walking at the chosen pace.'
    },
    {
      id: 'canyonPath',
      label: 'Canyon: route',
      help: 'Length of the least-cost route on the 31 m grid and the number of cells GPUCostDistancePath returned.'
    },
    {
      id: 'canyonClimb',
      label: 'Canyon: climb',
      help: 'Sum of uphill and downhill steps from start to destination along the route, from the 31 m DEM.'
    },
    {
      id: 'canyonDetour',
      label: 'Canyon: detour',
      help: 'Route travel time over the time to walk the straight line on flat ground at the same pace.'
    },
    {
      id: 'canyonConverged',
      label: 'Canyon: relaxation',
      help: 'Whether the cost relaxation reached a fixed point, and the iterations used out of the compile-time limit.'
    },
    {
      id: 'canyonBands',
      label: 'Canyon: cells per band',
      help: 'Cells in each travel-time band, nearest first (GPUCostDistance bandCounts).'
    },
    {
      id: 'dixieTime',
      label: 'Dixie: travel time',
      help: 'Accumulated cost at the Almanor shore destination.'
    },
    {id: 'dixiePath', label: 'Dixie: route'},
    {id: 'dixieClimb', label: 'Dixie: climb'},
    {id: 'dixieDetour', label: 'Dixie: detour'},
    {id: 'dixieConverged', label: 'Dixie: relaxation'},
    {id: 'dixieBands', label: 'Dixie: cells per band'},
    {
      id: 'distanceTiming',
      label: 'Distance field timing',
      help: 'GPU time of one GPUDistanceField encoding per algorithm on the canyon grid. Press the "Time exact vs jump-flood on the canyon grid" button to fill this in.'
    }
  ],

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUCostDistance, GPUCostDistancePath, GPUDistanceField,
  getGPUCostDistanceParameterValues, getGPUDistanceFieldParameterValues
} from '@luma.gl/experimental/gpu-raster';

// Friction kernel: minutes per metre, NaN for barriers (cliffs, water).
//   friction = 0.01 * exp(3.5 * ${state.slopePenalty} * tan(slope)) * landCoverFactor[class]
costGraph.add(new GPUCostDistance({
  width, height, cellSizeMode: 'web-mercator',
  friction: {id: 'friction', format: 'float32', storage: {kind: 'buffer', values: friction}},
  frictionParameters,                      // [scale, offset]: scale = 6 / ${state.speed} km/h
  settings: costSettings.importToGraph(costGraph),
  sources, sourceCosts, sourceCount,       // up to 8 starts, per-start head start
  maxIterations: 384,
  costs, backLinks, bandThresholds, bands, bandCounts, converged, iterationCount
}));
pathGraph.add(new GPUCostDistancePath({
  width, height, backLinks, target,        // target cell: a parameter write while you drag
  output: {ids, count, overflow, totalCount}
}));
distanceGraph.add(new GPUDistanceField({
  width, height, mode: '${state.distanceAlgorithm.startsWith('jump') ? 'jump-flood' : 'exact'}',${state.distanceAlgorithm.startsWith('jump') ? `\n  jumpFloodRefinementPasses: ${state.distanceAlgorithm.slice(-1)},` : ''}
  settings: distanceSettings.importToGraph(distanceGraph),
  seedPositions, seedCount, seedMask,      // points and/or a raster of seeds
  output: {distances, allocation}
}));

// Per change (parameter writes, no recompile):
costSettings.write(getGPUCostDistanceParameterValues({
  cellSize, northEdge, southEdge, costLimit: ${state.costLimit >= NO_COST_LIMIT_MINUTES ? 'Infinity' : state.costLimit}
}));
bandThresholds.write(Float32Array.from({length: 8}, (_, i) => (i + 1) * ${state.bandMinutes}));
target.write(Uint32Array.of(destinationCell));
costGraph.compiled.encode(commandEncoder, {parameters: undefined});   // when sources or costs changed
pathGraph.compiled.encode(commandEncoder, {parameters: undefined});   // also when only the target moved`,

  create: async ctx => (await import('./least-cost-paths.compute')).createLeastCostPaths(ctx)
});
