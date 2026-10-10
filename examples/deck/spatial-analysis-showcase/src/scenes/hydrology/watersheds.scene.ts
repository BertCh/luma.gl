// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene, type LegendSpec} from '../scene';
import type {WatershedsOptions} from './watersheds.compute';

const BASIN_COLORS = [
  [230, 90, 90, 255],
  [240, 170, 60, 255],
  [225, 215, 80, 255],
  [110, 205, 100, 255],
  [70, 200, 190, 255],
  [80, 140, 240, 255],
  [150, 110, 235, 255],
  [225, 100, 200, 255]
] as const;
const ORDER_COLORS = [
  [120, 190, 255, 255],
  [60, 225, 225, 255],
  [110, 235, 110, 255],
  [250, 230, 70, 255],
  [255, 150, 50, 255],
  [255, 80, 80, 255],
  [240, 90, 235, 255]
] as const;

const formatSquareKilometers = (logSquareMeters: number): string => {
  const squareKilometers = 10 ** logSquareMeters / 1e6;
  return squareKilometers >= 10
    ? `${squareKilometers.toFixed(0)} km²`
    : squareKilometers >= 1
      ? `${squareKilometers.toFixed(1)} km²`
      : `${squareKilometers.toFixed(2)} km²`;
};

/**
 * Drainage of the central Grand Canyon: the metadata, options and narrative live here (light, read
 * by the gallery); the GPU work lives in `watersheds.compute.ts`, imported when the story opens.
 */
export default defineScene<WatershedsOptions>({
  id: 'watersheds',
  title: 'Grand Canyon drainage area and watershed structure',
  chapter: 'hydrology',
  order: 1,
  summary:
    'GPU terrain routing derives drainage area, stream order, watersheds and hydrologic indices from a 15 m Grand Canyon elevation grid. The outputs are terrain proxies; they do not model rainfall, discharge or flood timing.',
  contributors: [
    'GPUTerrainFlow',
    'GPUTerrainWatersheds',
    'GPUTerrainStreamOrder',
    'GPUTerrainHydrologicIndices',
    'GPUTerrainHeightAboveDrainage',
    'GPUTerrainDerivatives'
  ],
  datasets: [{id: 'grand-canyon-dem', role: 'elevation, 15 m Terrarium tiles (Web Mercator)'}],
  initialView: {longitude: -112.1, latitude: 36.1, zoom: 11.1},
  basemap: ground('relief'),
  furniture: {
    title: {
      title: 'Grand Canyon drainage structure',
      subtitle: 'Contributing area, stream order, watersheds and terrain indices'
    },
    scaleBar: {units: 'metric'},
    credit: 'US Geological Survey (public domain)',
    caveat: 'Terrain-derived products are not rainfall, discharge or flood forecasts.'
  },

  options: [
    {
      kind: 'select',
      id: 'routing',
      label: 'Flow routing',
      group: 'Flow routing',
      apply: 'compile',
      default: 'd8',
      help: 'How a cell shares its accumulated flow with lower neighbours. Compile-time: the graph is rebuilt, the buffers are reused.',
      options: [
        {
          value: 'd8',
          label: 'D8 (steepest neighbour)',
          help: 'All flow to the single steepest receiver (O’Callaghan and Mark 1984). Crisp, thin channels; parallel lines on planar slopes.'
        },
        {
          value: 'd-infinity',
          label: 'D-infinity (Tarboton)',
          help: 'Flow follows the steepest triangular facet and splits between its two neighbours by angle (Tarboton 1997).'
        },
        {
          value: 'mfd-freeman',
          label: 'Multiple flow direction, Freeman',
          help: 'Flow split among all lower neighbours in proportion to tan(slope)^p, p = 1.1 (Freeman 1991).'
        },
        {
          value: 'mfd-quinn',
          label: 'Multiple flow direction, Quinn',
          help: 'Like Freeman but weighted by tan(slope) times contour length, p = 1 (Quinn et al. 1991).'
        }
      ]
    },
    {
      kind: 'toggle',
      id: 'resolveFlats',
      label: 'Resolve flats',
      group: 'Flow routing',
      apply: 'compile',
      default: true,
      help: 'Routes flow across the flat areas left by depression filling and the 0.5 m DEM steps (Barnes, Lehman and Mulla 2014). Compile-time.'
    },
    {
      kind: 'slider',
      id: 'flowExponent',
      label: 'Flow exponent p',
      group: 'Flow routing',
      apply: 'param',
      min: 0,
      max: 4,
      step: 0.1,
      default: 0,
      disabledWhen: state => state.routing === 'd8' || state.routing === 'd-infinity',
      format: value => (value === 0 ? 'published (1.1 Freeman, 1 Quinn)' : value.toFixed(1)),
      help: 'Exponent of the multiple-flow-direction weights. Higher values concentrate flow on the steepest neighbour; 0 uses the published exponent.'
    },
    {
      kind: 'slider',
      id: 'fillEpsilon',
      label: 'Fill gradient',
      group: 'Flow routing',
      apply: 'param',
      min: 0,
      max: 4,
      step: 1,
      default: 0,
      format: value =>
        value === 0 ? '0 (leave flats)' : `${[0, 0.0001, 0.001, 0.01, 0.1][value]} m per cell`,
      help: 'Minimum rise per step on the filled surface. 0 leaves filled pits as exact flats; a positive value tilts them so every cell drains.'
    },
    {
      kind: 'slider',
      id: 'streamArea',
      label: 'Stream threshold',
      group: 'Streams',
      apply: 'param',
      min: -2.5,
      max: 1.5,
      step: 0.1,
      default: -0.5,
      format: value => {
        const squareKilometers = 10 ** value;
        return `${squareKilometers < 0.1 ? squareKilometers.toFixed(3) : squareKilometers.toFixed(2)} km² contributing area`;
      },
      help: 'A cell is a stream when the area draining through it passes this value. Small thresholds give a dense network, large ones only the trunk streams. Per-frame parameter.'
    },
    {
      kind: 'select',
      id: 'basinMode',
      label: 'Basins from',
      group: 'Watersheds',
      apply: 'param',
      default: 'pour-points',
      disabledWhen: state => state.product !== 'watersheds',
      help: 'With pour points every cell is labelled by the nearest downstream pour point (nested watersheds). With outlets, each grid-edge or pit outlet gets its own basin.',
      options: [
        {value: 'pour-points', label: 'Pour points (click the map)'},
        {value: 'outlets', label: 'Every outlet (automatic basins)'}
      ]
    },
    {
      kind: 'button',
      id: 'resetPourPoints',
      label: 'Restore the three story pour points',
      group: 'Watersheds'
    },
    {kind: 'button', id: 'clearPourPoints', label: 'Clear pour points', group: 'Watersheds'},
    {
      kind: 'slider',
      id: 'handRange',
      label: 'HAND color range',
      group: 'Flooding',
      apply: 'param',
      min: 10,
      max: 300,
      step: 10,
      default: 100,
      unit: 'm',
      help: 'Height above nearest drainage that maps to the end of the ramp.'
    },
    {
      kind: 'slider',
      id: 'floodStage',
      label: 'Flood stage',
      group: 'Flooding',
      apply: 'param',
      min: 1,
      max: 80,
      step: 1,
      default: 12,
      unit: 'm',
      help: 'Cells with HAND at or below this value are flooded; the color is depth of water above the ground. Per-frame parameter.'
    },
    {
      kind: 'select',
      id: 'indexKind',
      label: 'Hydrologic index',
      group: 'Indices',
      apply: 'param',
      default: 'wetness',
      disabledWhen: state => state.product !== 'indices',
      help: 'Which of the three outputs of GPUTerrainHydrologicIndices to draw.',
      options: [
        {value: 'wetness', label: 'Topographic wetness ln(a / tan β)'},
        {value: 'catchment', label: 'Specific catchment area a (log10 m)'},
        {value: 'power', label: 'Stream power a · tan β (log10)'}
      ]
    },
    {
      kind: 'slider',
      id: 'minimumSlope',
      label: 'Minimum slope',
      group: 'Indices',
      apply: 'param',
      min: 0.1,
      max: 5,
      step: 0.1,
      default: 0.1,
      unit: '%',
      disabledWhen: state => state.product !== 'indices',
      help: 'Lower bound on tan β so flat cells do not make the wetness index infinite. 0.1 % is the contributor default.'
    },
    {
      kind: 'select',
      id: 'product',
      label: 'Product',
      group: 'Display',
      apply: 'param',
      default: 'accumulation',
      help: 'Every product is computed in the same graph; this only chooses which output buffer the map draws.',
      options: [
        {value: 'accumulation', label: 'Flow accumulation (contributing area)'},
        {value: 'stream-order', label: 'Strahler stream order'},
        {value: 'watersheds', label: 'Watersheds (click pour points)'},
        {value: 'hand', label: 'Height above nearest drainage (HAND)'},
        {value: 'flood', label: 'Flood stage map (HAND below a stage)'},
        {value: 'indices', label: 'Wetness, catchment and stream power'},
        {value: 'fill', label: 'Depression fill depth'}
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
      default: 0.85,
      format: value => `${Math.round(value * 100)} %`,
      help: 'Blend of the product over the shaded relief. Lower it to read the cliffs and benches under the flow.'
    },
    {
      kind: 'toggle',
      id: 'showStreams',
      label: 'Stream overlay',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Draws the cells whose contributing area passes the stream threshold on top of the product.'
    }
  ],

  story: [
    {
      id: 'question',
      title: 'Where does South Rim rain end up?',
      headline: 'Drainage area concentrates along canyon channels',
      textAlternative:
        'A terrain map colors each Grand Canyon cell by contributing area, with narrow high-value corridors following the main drainage network.',
      body: 'Rain that falls on the South Rim at Grand Canyon Village does not reach the Colorado River by one route: it is gathered by dozens of side canyons first. `GPUTerrainFlow` answers the first question every hydrologist asks of a DEM: **how much land drains through each cell?**\n\nThe map shows that **contributing area** on a log scale over shaded relief (bamako: dark is a hillside rill, light is the Colorado draining the whole window). The Colorado is the light trunk; Bright Angel Creek is the light branch from the North Rim. The window is 32 km wide with 1.9 km of relief from rim to river.',
      camera: {longitude: -112.1, latitude: 36.1, zoom: 11.1, transitionMs: 1400},
      options: {product: 'accumulation'},
      controls: ['product', 'overlayOpacity'],
      readouts: ['relief'],
      highlight: {readout: 'relief'}
    },
    {
      id: 'fill-route-accumulate',
      title: 'Fill, route, accumulate',
      headline: 'Depression filling produces continuous downslope drainage',
      textAlternative:
        'Filled terrain and flow accumulation reveal connected downslope routes from uplands into tributaries and the Colorado River corridor.',
      body: '`GPUTerrainFlow` does three things in one node. It **fills depressions** (Planchon-Darboux relaxation, so water has a way out of every pit), **routes** each cell to a downhill neighbour (D8: the steepest of eight, ESRI codes 1 to 128) and **accumulates** the area of every upstream cell in square metres. Cell sizes come from the Web Mercator tiles, so a cell is 15.4 m on the ground, not the 19.1 m a pixel has on screen.\n\nWatch the *Converged* readout: the relaxation loops stop as soon as nothing changes, and it reports iterations used out of the compile-time limit. Zoom to Bright Angel Canyon to see the creek carve its fault line to the river at Phantom Ranch. Cross-check: this is the same D8 model as `richdem.FlowAccumulation`, `pysheds` and GRASS `r.watershed`. Switch **Product** below to *Depression fill depth* to see what the fill changed, and try **Resolve flats** and **Fill gradient**.',
      camera: {longitude: -112.085, latitude: 36.135, zoom: 12.3, transitionMs: 1800},
      controls: ['product', 'resolveFlats', 'fillEpsilon'],
      readouts: ['converged'],
      highlight: {readout: 'converged'},
      callout: {coordinate: [-112.0953, 36.107], text: 'Phantom Ranch'}
    },
    {
      id: 'routing',
      title: 'Split the flow: D8 versus multiple directions',
      headline: 'Routing choice changes flow concentration on slopes',
      textAlternative:
        'The accumulation surface changes between single-direction D8 and multiple-flow routing, especially across broad slopes and convergent channels.',
      body: 'D8 sends everything down one neighbour, so channels are one cell thin and flow on a smooth slope runs in parallel lines. **Multiple flow direction** (`mfd-freeman`, set here) splits a cell’s flow among *all* lower neighbours in proportion to `tan(slope)^p`, so water spreads over the Tonto Platform and the Redwall benches before it gathers into a channel. `d-infinity` is the middle way (Tarboton 1997).\n\nThe routing choice is compile-time, so the control is marked *rebuild*; the buffers are shared and the graph is rebuilt in a few milliseconds. Slide **Flow exponent p** below (per-frame) to make the split sharper. Ridges stay dark, benches turn smooth.',
      camera: {longitude: -112.115, latitude: 36.085, zoom: 12.6, transitionMs: 1600},
      options: {routing: 'mfd-freeman'},
      controls: ['routing', 'flowExponent']
    },
    {
      id: 'stream-order',
      title: 'Rank the network with Strahler order',
      headline: 'Stream order increases at tributary confluences',
      textAlternative:
        'Colored stream lines become wider and advance through ordered classes where tributaries of equal order join.',
      body: '`GPUTerrainStreamOrder` ranks every stream cell. Headwater reaches are **order 1**; where two streams of equal order meet the order rises by one, otherwise the larger order continues (Strahler 1957). A high order marks a trunk stream that carries many tributaries, so order is a quick proxy for channel size and flood hazard.\n\nThe thicker, warmer lines are the higher orders (legend). The **Stream threshold** slider below decides which cells count as streams at all: lower it to see side-canyon rills, raise it to keep only the main stems. Routing returns to D8 so the accumulation and the D8 tracing that ranks the streams agree.',
      camera: {longitude: -112.1, latitude: 36.1, zoom: 11.4, transitionMs: 1600},
      options: {product: 'stream-order', routing: 'd8', streamArea: -1},
      controls: ['streamArea'],
      readouts: ['orders', 'maximumOrder'],
      highlight: {readout: 'orders'}
    },
    {
      id: 'watersheds',
      title: 'Watersheds from pour points',
      headline: 'Pour points delineate distinct upstream catchments',
      textAlternative:
        'Three colored catchment polygons extend upstream from selected pour points, with mapped area totals reported for each watershed.',
      body: '`GPUTerrainWatersheds` labels every cell with the **nearest downstream pour point**. Three are placed for you: Phantom Ranch on Bright Angel Creek (1, red), Indian Garden on Garden Creek (2, orange) and the Colorado at the creek’s mouth (3, yellow). Because labelling follows the flow downhill, watershed 1 is *nested inside* watershed 3, and the areas appear in the readout.\n\n**Click the map** to add a pour point: it snaps to the highest-accumulation cell within five cells so a click near a creek lands on the creek. Switch **Basins from** below to *every outlet* to see the automatic basins the contributor builds without pour points.',
      camera: {longitude: -112.09, latitude: 36.095, zoom: 11.7, transitionMs: 1600},
      options: {product: 'watersheds', showStreams: true},
      controls: ['basinMode', 'resetPourPoints', 'clearPourPoints'],
      readouts: ['watershedAreas', 'pourPoints'],
      highlight: {readout: 'watershedAreas'}
    },
    {
      id: 'flood',
      title: 'How high above the river is that campground?',
      headline: 'Low HAND values follow the drainage network',
      textAlternative:
        'Height-above-drainage values identify low cells adjacent to channels, and the flood-stage overlay selects terrain below the chosen relative height.',
      body: '`GPUTerrainHeightAboveDrainage` (HAND, Rennó et al. 2008) follows each cell’s flow path down to the first stream cell and reports the **height above it**. A flat-ish bench 10 m above the creek and 400 m from it is HAND 10; a cliff top is HAND 300. Thresholding HAND at a stage gives an inundation map without a hydraulic model: the colour here is the depth of water at a **Flood stage** of 12 m.\n\nDrag **Flood stage** up to 40 m and watch the inner gorge, the Tonto benches and the lower side canyons fill. HAND uses the *streams* you chose, so changing **Stream threshold** changes what counts as the nearby drainage. This is a screening tool for relative exposure, not a flood forecast.',
      camera: {longitude: -112.093, latitude: 36.106, zoom: 13.2, transitionMs: 1800},
      options: {product: 'flood', floodStage: 12},
      controls: ['floodStage', 'streamArea'],
      readouts: ['floodArea'],
      highlight: {readout: 'floodArea'},
      callout: {coordinate: [-112.0953, 36.107], text: 'Phantom Ranch'}
    },
    {
      id: 'indices',
      title: 'Where does the ground stay wet?',
      headline: 'Wetness index peaks in convergent low-slope terrain',
      textAlternative:
        'A sequential terrain overlay highlights cells with high topographic wetness, concentrated where large contributing area combines with low slope.',
      body: '`GPUTerrainHydrologicIndices` combines the contributing area with the local slope. With **specific catchment area** `a = A / width`, the **topographic wetness index** is `ln(a / tan β)`: high where a large area drains to a gentle slope (benches, creek flats, springs such as Indian Garden), low on steep ridges. **Stream power** `a · tan β` marks where flowing water has the most erosive energy, the cliffs under big side canyons.\n\nLimits: the DEM is a 15 m model of a canyon whose steepest walls are vertical, flow accumulation depends on the routing you chose, and slopes are D8 descent slopes (TauDEM and SAGA use 3 x 3 or D-infinity slopes, so values differ). Below, switch **Hydrologic index** between the three outputs, try wetness with **Flow routing** on *Multiple flow direction*, then raise **Minimum slope** to see how the flat bench floors change.',
      camera: {longitude: -112.12, latitude: 36.08, zoom: 12.2, transitionMs: 1600},
      options: {product: 'indices', indexKind: 'wetness'},
      controls: ['indexKind', 'routing', 'minimumSlope']
    }
  ],

  about: {
    what: 'A chain of six GPU contributors on one depression-filled DEM of the central Grand Canyon: fill and routing (`GPUTerrainFlow`), height above drainage, watersheds, Strahler order and hydrologic indices.',
    why: 'Catchment area, stream networks, flood exposure and wetness are the standard terrain products behind hydrologic models, erosion studies, trail and campsite planning and habitat mapping. Doing it in a GPU graph makes the stream threshold, pour points and flood stage interactive.',
    howToRead:
      'Dark to bright means little to much contributing area (log scale). Stream order grows from blue (1) to magenta (7). Watershed colours follow the pour-point number. Flood colours show the depth of water at the chosen stage. Everything draws over a hillshade from `GPUTerrainDerivatives`.'
  },

  legends: state => {
    const legends: LegendSpec[] = [];
    switch (state.product) {
      case 'accumulation':
        legends.push({
          kind: 'ramp',
          title: 'Contributing area',
          ramp: 'bamako',
          extent: [4, 8.8],
          format: formatSquareKilometers,
          unit: 'drained through the cell'
        });
        break;
      case 'fill':
        legends.push({
          kind: 'ramp',
          title: 'Depression fill depth',
          ramp: 'inferno',
          extent: [0, 10],
          unit: 'm raised'
        });
        break;
      case 'hand':
        legends.push({
          kind: 'ramp',
          title: 'Height above nearest drainage',
          ramp: 'inferno',
          extent: [0, state.handRange],
          unit: 'm'
        });
        break;
      case 'flood':
        legends.push({
          kind: 'ramp',
          title: `Flood depth at ${state.floodStage} m stage`,
          ramp: 'cividis',
          extent: [0, state.floodStage],
          unit: 'm of water'
        });
        break;
      case 'stream-order':
        legends.push({
          kind: 'categories',
          title: 'Strahler order',
          entries: ORDER_COLORS.map((color, index) => ({color, label: `${index + 1}`})),
          note: 'Line width grows with order.'
        });
        break;
      case 'watersheds':
        legends.push({
          kind: 'categories',
          title: state.basinMode === 'pour-points' ? 'Pour point' : 'Outlet basin',
          entries:
            state.basinMode === 'pour-points'
              ? BASIN_COLORS.map((color, index) => ({color, label: `${index + 1}`}))
              : BASIN_COLORS.map((color, index) => ({color, label: `basin class ${index + 1}`})),
          note:
            state.basinMode === 'pour-points'
              ? 'Story points: 1 Phantom Ranch, 2 Indian Garden, 3 Colorado at Bright Angel Creek. Click to add (oldest dropped after 8).'
              : 'Colors repeat; neighbouring basins differ.'
        });
        break;
      case 'indices':
        legends.push(
          state.indexKind === 'wetness'
            ? {
                kind: 'ramp',
                title: 'Topographic wetness index',
                ramp: 'bamako',
                extent: [2, 18],
                unit: 'ln(a / tan β)'
              }
            : state.indexKind === 'catchment'
              ? {
                  kind: 'ramp',
                  title: 'Specific catchment area',
                  ramp: 'bamako',
                  extent: [0.5, 3.5],
                  unit: 'log10 m',
                  format: value => `10^${value.toFixed(1)}`
                }
              : {
                  kind: 'ramp',
                  title: 'Stream power index',
                  ramp: 'inferno',
                  extent: [-1, 3],
                  unit: 'log10 of a · tan β'
                }
        );
        break;
    }
    if (state.showStreams && state.product !== 'stream-order' && state.product !== 'accumulation') {
      legends.push({
        kind: 'categories',
        title: 'Streams',
        entries: [
          {
            color: [40, 140, 240, 255],
            label: `Cells draining at least ${(10 ** state.streamArea).toFixed(3)} km²`
          }
        ]
      });
    }
    return legends;
  },

  readouts: [
    {id: 'grid', label: 'Grid', help: 'Raster size. Cells are 15.4 m on the ground.'},
    {
      id: 'cellSize',
      label: 'Cell size',
      help: 'Ground size of a cell at the window centre; the contributors correct it per row for Web Mercator.'
    },
    {
      id: 'relief',
      label: 'Elevation range',
      help: 'Lowest and highest DEM cell: the Colorado River gorge to the North Rim.'
    },
    {
      id: 'converged',
      label: 'Converged',
      help: 'Whether each relaxation loop converged, with iterations used out of the compile-time limit.'
    },
    {
      id: 'streams',
      label: 'Stream cells',
      help: 'Cells whose contributing area is at or above the stream threshold.'
    },
    {id: 'maximumOrder', label: 'Highest Strahler order', format: 'integer'},
    {
      id: 'orders',
      label: 'Stream cells by order',
      help: 'Cell counts per Strahler order, from GPUHistogram.'
    },
    {id: 'pourPoints', label: 'Pour points', format: 'integer'},
    {
      id: 'watershedAreas',
      label: 'Watershed areas',
      help: 'Area draining to each pour point (nested watersheds include their upstream pour points).'
    },
    {
      id: 'floodArea',
      label: 'Flooded area',
      help: 'Area with height above nearest drainage at or below the flood stage.'
    }
  ],

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUTerrainFlowParameterValues, GPUTerrainFlow, GPUTerrainHeightAboveDrainage,
  GPUTerrainWatersheds, GPUTerrainStreamOrder, GPUTerrainHydrologicIndices
} from '@luma.gl/experimental/gpu-terrain';

const graph = new GPUCommandGraph(device, {id: 'watersheds'});
graph.add(new GPUTerrainFlow({
  width, height, elevation, settings: flowSettings.importToGraph(graph),
  cellSizeMode: 'web-mercator',             // cellSize = 19.1 m Mercator pixels, edges from the tile bounds
  fillDepressions: true, resolveFlats: ${state.resolveFlats},
  flowRouting: '${state.routing}',          // compile-time
  accumulationUnits: 'area',                // m², so the stream threshold is a drainage area
  filledElevation, flowDirections, accumulation, streams
}));
graph.add(new GPUTerrainStreamOrder({width, height, flowDirections, streams, streamOrder}));
graph.add(new GPUTerrainWatersheds({width, height, flowDirections, pourPoints, labels}));
graph.add(new GPUTerrainHeightAboveDrainage({
  width, height, elevation: filledBand, flowDirections, streams, heightAboveDrainage
}));
graph.add(new GPUTerrainHydrologicIndices({
  width, height, elevation: filledBand, accumulation, settings: indicesSettings.importToGraph(graph),
  cellSizeMode: 'web-mercator', wetnessIndex, streamPowerIndex, specificCatchmentArea
}));
const compiled = graph.compile();            // once

// Per change (no recompile): thresholds are parameter-buffer writes.
flowSettings.write(getGPUTerrainFlowParameterValues({
  cellSize: [19.109, 19.109], northEdge, southEdge,
  streamThreshold: ${(10 ** state.streamArea * 1e6).toFixed(0)}, // m², ${(10 ** state.streamArea).toFixed(3)} km²
  fillEpsilon: 0, flowExponent: ${state.flowExponent}
}));
compiled.encode(commandEncoder, {parameters: undefined});`,

  create: async ctx => (await import('./watersheds.compute')).createWatersheds(ctx)
});
