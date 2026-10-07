// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getClassTableLegend} from '../../cartography/class-table';
import {CREDITS, joinCredits} from '../../cartography/credits';
import {CITY_FRAMES, labelsFor, NYC} from '../../cartography/gazetteer';
import {ground} from '../../cartography/grounds';
import type {ClassTable} from '../../cartography/types';
import {defineScene, type LegendSpec} from '../scene';
import {FLOW_CREDITS} from './flows-style';
import {getPlaceCenter, getPlacesBounds} from './nyc-taxi-crossfilter-style';
import {NYC_TAXI_HOURS} from './nyc-taxi-data';
import type {NycTaxiDensityOptions} from './nyc-taxi-density.compute';
import {
  DENSITY_RAMP,
  DENSITY_RAMP_RANGE,
  DROPOFF_DENSITY_RAMP,
  formatDensity,
  formatWindowTime
} from './nyc-taxi-density-style';

/** The cartouche of one step: the claim, the variable and method, and the sample chip. */
const cartouche = (title: string, subtitle: string) => ({
  title,
  subtitle,
  chips: ['Sample'] as const
});

/** Place names from the NYC gazetteer, drawn above the data (lowered zooms for the city frame). */
const places = (ids: readonly string[]) =>
  labelsFor(NYC, ids, {
    midtown: {minZoom: 9},
    'lower-manhattan': {minZoom: 9},
    'penn-station': {minZoom: 11.2},
    'grand-central': {minZoom: 11.2}
  });

const MIDTOWN_CENTER = getPlaceCenter('midtown');
const CITY_AND_AIRPORTS = getPlacesBounds(['midtown', 'jfk', 'lga'], 0.05);

export default defineScene<NycTaxiDensityOptions>({
  id: 'nyc-taxi-density',
  title: 'Where do taxi rides begin?',
  chapter: 'flows',
  order: 3,
  summary:
    'GPUPointDensity bins taxi pickups and drop-offs into a grid that follows the camera. Learn the stretch of a colour scale, why a count per cell becomes a density per km2, how pickups and drop-offs differ, and why a mean fare needs a minimum number of rides.',
  contributors: ['GPUPointDensity'],
  datasets: [{id: 'poopdeck-nyc-taxi', role: 'taxi trips as pickup and drop-off points'}],
  initialView: {...CITY_FRAMES.nyc},

  options: [
    {
      kind: 'select',
      id: 'measure',
      label: 'Map shows',
      group: 'Field',
      apply: 'param',
      default: 'rides',
      display: 'segmented',
      help: 'Rides per square kilometre (a count per cell, divided by the cell area), or the mean fare of the rides that start in a cell. Each reads its own graph; the fare map uses an unsmoothed mean and a ride tally.',
      options: [
        {value: 'rides', label: 'Rides per km²'},
        {value: 'fare', label: 'Mean fare'}
      ]
    },
    {
      kind: 'select',
      id: 'show',
      label: 'Show',
      group: 'Field',
      apply: 'param',
      default: 'pickups',
      display: 'segmented',
      disabledWhen: state => state.measure === 'fare' || state.swipe === 'ends',
      help: 'Pickups or drop-offs. Both are rows of one 880,000-point buffer; a mask picks the half, so switching rewrites a buffer and nothing recompiles.',
      options: [
        {value: 'pickups', label: 'Pickups'},
        {value: 'dropoffs', label: 'Drop-offs'}
      ]
    },
    {
      kind: 'select',
      id: 'swipe',
      label: 'Compare',
      group: 'Field',
      apply: 'param',
      default: 'off',
      disabledWhen: state => state.measure === 'fare',
      help: 'A swipe divider over two versions of the same field: a linear stretch against the one you chose, or pickups against drop-offs on one scale.',
      options: [
        {value: 'off', label: 'Off'},
        {value: 'stretch', label: 'Linear against the stretch'},
        {value: 'ends', label: 'Pickups against drop-offs'}
      ]
    },
    {
      kind: 'select',
      id: 'scale',
      label: 'Stretch',
      group: 'Field',
      apply: 'param',
      default: 'sqrt',
      display: 'segmented',
      disabledWhen: state => state.measure === 'fare',
      help: 'How trips per km² become colour. Linear gives every ride the same step; the square root lifts quiet places; quantile classes give seven classes the same number of cells (breaks fixed at load).',
      options: [
        {value: 'linear', label: 'Linear'},
        {value: 'sqrt', label: 'Square root'},
        {value: 'quantile', label: 'Quantiles'}
      ]
    },
    {
      kind: 'select',
      id: 'binning',
      label: 'Cell shape',
      group: 'Grid',
      apply: 'compile',
      default: 'grid',
      display: 'segmented',
      help: 'Squares can be smoothed; hexagons have six equidistant neighbours and are not smoothed. Compile-time binning.',
      options: [
        {value: 'grid', label: 'Squares'},
        {value: 'hexagon', label: 'Hexagons'}
      ]
    },
    {
      kind: 'select',
      id: 'resolution',
      label: 'Resolution',
      group: 'Grid',
      apply: 'compile',
      default: 'medium',
      display: 'segmented',
      help: 'Number of cells across the screen (`gridSize`, compile-time): coarse is about 70, medium 110, fine 170. The grid is a fixed number of cells on screen, so zooming changes the cell in metres, not in pixels.',
      options: [
        {value: 'coarse', label: 'Coarse'},
        {value: 'medium', label: 'Medium'},
        {value: 'fine', label: 'Fine'}
      ]
    },
    {
      kind: 'select',
      id: 'smoothing',
      label: 'Smoothing',
      group: 'Grid',
      apply: 'param',
      default: 'gaussian',
      display: 'segmented',
      disabledWhen: state => state.binning === 'hexagon' || state.measure === 'fare',
      help: 'A Gaussian blur of the ride counts (square grid only). The kernel is a parameter buffer, so switching it or changing sigma never recompiles. The mean-fare map is never blurred: a mean of blurred means is not a rate.',
      options: [
        {value: 'off', label: 'Raw cells'},
        {value: 'gaussian', label: 'Gaussian'}
      ]
    },
    {
      kind: 'slider',
      id: 'sigma',
      label: 'Smoothing radius (sigma)',
      group: 'Grid',
      apply: 'param',
      min: 0.5,
      max: 2.5,
      step: 0.25,
      default: 1,
      unit: 'cells',
      disabledWhen: state =>
        state.binning === 'hexagon' || state.smoothing === 'off' || state.measure === 'fare',
      describe: value =>
        `${value} cells; a ride reaches ${Math.min(8, Math.ceil(3 * value))} cells`,
      help: 'Standard deviation of the Gaussian in cells. Larger values give a regional pattern; smaller ones keep single blocks. Bandwidth is the subject of the nature story.',
      expert: true
    },
    {
      kind: 'preset',
      id: 'hourPresets',
      label: 'Time window',
      group: 'Time',
      help: 'Three windows of equal length, so they compare: New Year night, a Friday morning and a Thursday evening. They share one colour scale.',
      presets: [
        {label: 'New Year 00-03', values: {hours: [0, 3]}},
        {label: 'Friday 07-10', values: {hours: [31, 34]}},
        {label: 'Thursday 18-21', values: {hours: [18, 21]}},
        {label: 'All hours', values: {hours: [0, 39]}}
      ]
    },
    {
      kind: 'range',
      id: 'hours',
      label: 'Hours',
      group: 'Time',
      apply: 'param',
      min: 0,
      max: 39,
      step: 0.5,
      default: [0, 39],
      format: formatWindowTime,
      help: 'Keeps points whose time is in [from, to), measured from midnight on Thursday 1 January. Pickups use the pickup time, drop-offs the drop-off time. The mask buffers are rewritten; nothing is recompiled. A window narrower than the whole period uses one locked rate scale.'
    },
    {
      kind: 'slider',
      id: 'minCount',
      label: 'Minimum rides per cell',
      group: 'Fare',
      apply: 'param',
      min: 1,
      max: 100,
      step: 1,
      default: 20,
      unit: 'rides',
      disabledWhen: state => state.measure !== 'fare',
      marks: [{value: 20, label: 'default'}],
      // One or two rides: a "mean" that is one trip.
      danger: [1, 5],
      help: 'A cell shows its mean fare only when at least this many rides start in it. Fewer: the cell is hatched. At one to five rides the map speckles with single trips.'
    },
    {
      kind: 'toggle',
      id: 'showGrid',
      label: 'Show the grid',
      group: 'Display',
      apply: 'param',
      default: false,
      disabledWhen: state => state.binning === 'hexagon',
      help: 'Draws the cell edges while the cells are large enough on screen, with the scale bar ticked at the cell width: the grid follows the camera.'
    },
    {
      kind: 'slider',
      id: 'opacity',
      label: 'Field opacity',
      group: 'Display',
      apply: 'param',
      min: 0.2,
      max: 1,
      step: 0.05,
      default: 0.9,
      disabledWhen: state => state.measure === 'fare',
      help: 'Lower it to read the streets under the field.'
    }
  ],

  readouts: [
    {
      id: 'kept',
      label: 'Rides kept',
      format: 'integer',
      emphasis: 'tile',
      help: 'Points passing the time window, on the side the map shows.'
    },
    {
      id: 'peakDensity',
      label: 'Busiest cell, density',
      help: 'The brightest cell of the field divided by its ground area, read back once the camera settles.'
    },
    {
      id: 'peakCount',
      label: 'Busiest cell, rides',
      help: 'The largest cell value of the field (smoothed when the blur is on).'
    },
    {
      id: 'peakPlace',
      label: 'Busiest cell is',
      help: 'Nearest named place in the New York gazetteer.'
    },
    {id: 'peakPlaceB', label: 'Busiest drop-off cell is'},
    {
      id: 'cellSize',
      label: 'Cell size',
      help: 'Follows the zoom: the bounds buffer is rewritten every frame.'
    },
    {
      id: 'corePickups',
      label: 'Pickups in the Midtown core',
      format: 'percent',
      help: 'Share of pickups within four kilometres of Midtown.'
    },
    {
      id: 'coreDropoffs',
      label: 'Drop-offs in the Midtown core',
      format: 'percent',
      help: 'Share of drop-offs within four kilometres of Midtown.'
    },
    {
      id: 'jfkFare',
      label: 'Mean fare at JFK',
      help: 'The mean fare of the cell nearest the airport that holds enough rides.'
    },
    {
      id: 'lgaFare',
      label: 'Mean fare at LaGuardia',
      help: 'The mean fare of the cell nearest the airport that holds enough rides.'
    },
    {
      id: 'hiddenCells',
      label: 'Cells hidden by the rule',
      help: 'Cells with rides, but fewer than the minimum: hatched on the map.'
    },
    {id: 'window', label: 'Time window'},
    {id: 'valueChart', label: 'Cells by density', kind: 'chart'},
    {id: 'pulseChart', label: 'Rides per hour', kind: 'chart'},
    {id: 'points', label: 'Points on the GPU', format: 'integer', hood: true},
    {id: 'grid', label: 'Grid', hood: true},
    {
      id: 'cellArea',
      label: 'Cell area',
      hood: true,
      help: 'Ground area of one cell: the divisor that turns a count into a density.'
    }
  ],

  pipeline: [
    {
      id: 'bounds',
      label: 'Camera bounds',
      detail: 'Four floats written from the camera every frame; no recompile'
    },
    {
      id: 'bin',
      label: 'Bin',
      detail: 'Every point adds itself to its cell; hotspot sums are pre-aggregated per workgroup'
    },
    {id: 'blur', label: 'Gaussian', detail: 'An optional convolution of the cell counts'},
    {id: 'reduce', label: 'Extent', detail: 'A reduction finds the busiest cell'},
    {id: 'draw', label: 'Draw', detail: 'The raster layer reads the same GPU buffer'}
  ],

  legends: (state, data) => {
    if (state.measure === 'fare') {
      const table = data['fareTable'] as ClassTable | undefined;
      return table
        ? [
            getClassTableLegend(table, {
              title: 'Mean fare of rides that start in a cell'
            })
          ]
        : [];
    }
    const clip = (data['clip'] as number | undefined) ?? 1;
    const histogram = data['histogram'] as number[] | undefined;
    const marker = data['marker'] as number | null | undefined;
    const quantile = data['quantileTable'] as ClassTable | undefined;
    const dropoffQuantile = data['dropoffQuantileTable'] as ClassTable | undefined;
    const wholePeriod = state.hours[0] <= 0 && state.hours[1] >= NYC_TAXI_HOURS;
    const bars = histogram?.map(count => Math.sqrt(count));
    const entry = (
      side: 'pickups' | 'dropoffs',
      title: string,
      withStrip: boolean,
      scale: NycTaxiDensityOptions['scale']
    ): LegendSpec => {
      const table = side === 'pickups' ? quantile : dropoffQuantile;
      if (scale === 'quantile' && table) {
        return getClassTableLegend(table, {
          title,
          basis: 'per km²',
          histogram: withStrip ? bars : undefined,
          note: 'Seven classes with the same number of cells; breaks fixed at load.'
        });
      }
      return {
        kind: 'ramp',
        id: `density-${side}`,
        title,
        ramp: side === 'pickups' ? DENSITY_RAMP : DROPOFF_DENSITY_RAMP,
        range: DENSITY_RAMP_RANGE,
        extent: [0, clip],
        sqrtScale: scale === 'sqrt',
        unit: 'trips',
        basis: 'per km²',
        marker: withStrip && marker != null ? marker : undefined,
        histogram: withStrip ? bars : undefined,
        note: `${scale === 'sqrt' ? 'Square-root stretch. ' : 'Linear stretch. '}Top 2 % of cells clipped; empty cells not drawn.${
          wholePeriod ? '' : ' One scale for every window.'
        }`,
        format: formatDensity
      };
    };
    if (state.swipe === 'ends') {
      return [
        entry('pickups', 'Pickups (left)', true, state.scale),
        entry('dropoffs', 'Drop-offs (right)', false, state.scale)
      ];
    }
    const side = state.show;
    const title =
      state.swipe === 'stretch'
        ? `${side === 'pickups' ? 'Pickups' : 'Drop-offs'} (right of the divider)`
        : side === 'pickups'
          ? 'Pickups'
          : 'Drop-offs';
    return [entry(side, title, true, state.scale)];
  },

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUPointDensity} from '@luma.gl/experimental/gpu-spatial-analysis';

// pickups are rows 0..N-1 and drop-offs rows N..2N-1 of one float32x2 buffer
const graph = new GPUCommandGraph(device, {id: 'taxi-density'});
graph.add(
  new GPUPointDensity({
    positions,                         // 880,000 points, uploaded once
    mask,                              // uint32: which end, which hours (rewritten on change)${
      state.measure === 'fare'
        ? `
    weights: fare,                     // float32 fare per point`
        : ''
    }
    bounds: bounds.importToGraph(graph), // rewritten from the camera every frame
    gridSize: ${state.binning === 'grid' ? '[110, 70]' : '[52, 38]'},
    binning: '${state.binning}',${
      state.binning === 'grid' && state.measure === 'rides'
        ? `
    smoothing: {kernel, kernelWidth: 17, kernelHeight: 17, strategy: 'direct'},`
        : state.binning === 'hexagon'
          ? `
    hexagonRadius: hexagonRadius.importToGraph(graph),`
          : ''
    }
    statistic: '${state.measure === 'fare' ? 'mean' : 'count'}',
    output: {values, counts, extent}
  })
);
const compiled = graph.compile();      // once per lattice, resolution, side and statistic
// every frame: write the small parameter buffers, then
compiled.encode(commandEncoder, {parameters: undefined});

// colours are per km2 at every zoom: the layer divides each cell by its ground area
new SpatialAnalysisRasterLayer({gridSize, bounds: bounds.buffer, values,
  valueScale: 1 / cellAreaKm2, valueRange: [0, clipPerKm2],   // fixed at load
  colormap: '${DENSITY_RAMP}', rampRange: [0.15, 1], sqrtScale: ${state.scale === 'sqrt'}});`,

  about: {
    what: 'Previously: brushing 440,000 trips with a crossfilter. Next: tides, the signed balance of arrivals and departures. `GPUPointDensity` bins points into a square grid or a hexagon lattice that covers the visible map, optionally convolves the field with a Gaussian kernel, and reduces it to an extent, all in compute shaders. The statistic per cell is a count, a sum of a weight, or its mean.',
    why: 'A density answers "where is activity concentrated?" without choosing a boundary first, but three choices shape the picture: the stretch of the colour scale, the size of the cells, and the minimum number of points behind a rate. The grid follows the camera, so only a density per square kilometre is comparable between zooms.',
    howToRead:
      'Brighter cells hold more rides per square kilometre on a fixed scale (the brightest colour is the 98th percentile of the cells, read from the data at load). Cells with no rides are not drawn. **Sample bias:** one dataset of 38 hours that starts at New Year, yellow cabs only, so quiet cells in the outer boroughs mean few yellow taxis, not few people. Cell areas are in ground metres at this latitude (Web Mercator stretches the screen, the cell is measured on the ground). Drop-offs are the end of a routed path and snap to the road network.'
  },

  // Night ground in both page themes for the glows; the fare step switches to paper.
  basemap: ground('night'),
  furniture: {
    title: cartouche('Where do taxi rides begin?', 'Pickups per km², yellow cabs, 1-2 Jan 2015'),
    scaleBar: {units: 'metric'},
    credit: joinCredits(FLOW_CREDITS.nycTaxi, FLOW_CREDITS.osrmRoutes, CREDITS.carto),
    caveat: 'Yellow cabs only, in 38 hours that start at New Year.'
  },

  create: async ctx => (await import('./nyc-taxi-density.compute')).createNycTaxiDensity(ctx),

  story: [
    {
      id: 'glow',
      title: 'A city that glows',
      headline: 'Midtown outshines the rest of the city',
      textAlternative:
        'Dark map of New York with a glowing density of taxi pickups: brightest in Midtown, fading through Lower Manhattan and Brooklyn, with small glows at both airports.',
      body: 'Every ride begins with a pickup; the GPU bins **{{kept}}** of them into cells that follow the camera and shades them by rides per square kilometre. The busiest cell, **{{cellSize}}**, holds **{{peakDensity}}**, {{peakPlace}}. The scale is a square root with the brightest 2 % clipped, so quiet boroughs still show. Hover a cell. Switch **Smoothing** off to see raw cells.',
      optionsMode: 'fresh',
      options: {measure: 'rides', show: 'pickups', swipe: 'off', scale: 'sqrt'},
      controls: ['smoothing'],
      readouts: ['kept', 'peakDensity', 'peakPlace', 'cellSize'],
      stage: 'bin',
      camera: {...CITY_FRAMES.nyc, transitionMs: 1400},
      furniture: {
        title: cartouche('Where do taxi rides begin?', 'Pickups per km², yellow cabs, 1-2 Jan 2015')
      },
      annotations: places(['midtown', 'lower-manhattan', 'brooklyn', 'queens', 'jfk', 'lga'])
    },
    {
      id: 'stretch',
      title: 'The stretch',
      headline: 'The stretch decides what you can see',
      textAlternative:
        'The same pickup density split by a swipe divider: on the left a linear colour scale where only Midtown glows, on the right a square-root scale where the whole city shows.',
      body: 'Drag the divider: on the left the same field on a linear scale, on the right the **Stretch** you choose. Linearly, the busiest cell, **{{peakDensity}}**, takes the whole ramp and the rest of the city goes dark. A square root lifts the quiet streets; quantile classes give each of seven classes the same number of cells.\n\n*The stretch is an argument about which differences matter.*',
      optionsMode: 'fresh',
      options: {measure: 'rides', swipe: 'stretch', scale: 'sqrt'},
      controls: ['scale'],
      readouts: ['peakDensity', 'valueChart'],
      stage: 'draw',
      camera: {...CITY_FRAMES.nyc, transitionMs: 1400},
      compare: {mode: 'swipe', labels: ['Linear', 'Your stretch']},
      furniture: {
        title: cartouche('Which stretch shows the city?', 'Pickups per km², two stretches')
      },
      annotations: places(['midtown', 'lower-manhattan', 'jfk', 'brooklyn'])
    },
    {
      id: 'per-km2',
      title: 'Zoom changes the cell',
      headline: 'Zoom changes the cell, not the density',
      textAlternative:
        'Close view of Midtown with white cell outlines over a glowing density; the scale bar carries a tick the width of one cell.',
      body: 'Zoom in: the grid has a fixed number of cells on screen, so a cell is now **{{cellSize}}** and its busiest holds **{{peakCount}}**, **{{peakDensity}}**. Zoom out and the count in a cell grows, but the legend is always trips per km², so the colours stay comparable. Try **Show the grid** and **Resolution**.\n\n*When the cell follows the camera, only a density is comparable.*',
      optionsMode: 'fresh',
      options: {
        measure: 'rides',
        swipe: 'off',
        showGrid: true,
        resolution: 'coarse',
        scale: 'sqrt'
      },
      controls: ['showGrid', 'resolution'],
      readouts: ['cellSize', 'peakCount', 'peakDensity'],
      stage: 'bounds',
      camera: {
        longitude: MIDTOWN_CENTER[0],
        latitude: MIDTOWN_CENTER[1],
        zoom: 13.2,
        transitionMs: 1800
      },
      furniture: {
        title: cartouche('Why a density, not a count?', 'Pickups per km², cell follows the camera')
      },
      annotations: places(['penn-station', 'grand-central', 'midtown'])
    },
    {
      id: 'pickups-dropoffs',
      title: 'Pickups and drop-offs',
      headline: 'Rides start in the core and end farther out',
      textAlternative:
        'Pickups on the left in warm colours and drop-offs on the right in cool colours, split by a swipe divider on one shared scale: the drop-off field is more spread out.',
      body: 'Swipe between pickups (left) and drop-offs (right) on one colour scale. **{{corePickups}}** of pickups start within the Midtown core, but only **{{coreDropoffs}}** of drop-offs end there: the field spreads outward. The busiest pickup cell is {{peakPlace}}, the busiest drop-off cell {{peakPlaceB}}. Net flow, drop-offs minus pickups, is the tides story. Change the **Stretch** to move both.',
      optionsMode: 'fresh',
      options: {measure: 'rides', swipe: 'ends', scale: 'sqrt'},
      controls: ['scale'],
      readouts: ['corePickups', 'coreDropoffs', 'peakPlace', 'peakPlaceB'],
      stage: 'bin',
      camera: {...CITY_FRAMES.nyc, transitionMs: 1400},
      compare: {mode: 'swipe', labels: ['Pickups', 'Drop-offs']},
      furniture: {
        title: cartouche(
          'Where do rides start and end?',
          'Pickups and drop-offs per km², one scale'
        )
      },
      annotations: places(['midtown', 'lower-manhattan', 'jfk', 'lga', 'brooklyn', 'queens'])
    },
    {
      id: 'fare',
      title: 'A rate needs rides',
      headline: 'Airport rides cost the most',
      textAlternative:
        'Paper map of New York coloured in cool classes by the mean fare of rides starting in each cell: the two airports in the darkest classes, Manhattan in the lightest, with sparse cells hatched.',
      body: 'A price per ride is a rate, not a glow, so the ground turns to paper and the colours to classes. The mean fare in the JFK cell is **{{jfkFare}}** and at LaGuardia **{{lgaFare}}**. Lower **Minimum rides per cell**: single-ride cells speckle the map; **{{hiddenCells}}** are hatched now.\n\n*A rate needs enough rides behind it.*',
      optionsMode: 'fresh',
      options: {measure: 'fare', minCount: 20, smoothing: 'off'},
      controls: ['minCount'],
      readouts: ['jfkFare', 'lgaFare', 'hiddenCells'],
      stage: 'bin',
      basemap: ground('paperCity'),
      camera: {bounds: CITY_AND_AIRPORTS, transitionMs: 1800},
      furniture: {
        title: cartouche(
          'Where do rides cost the most?',
          'Mean fare per ride, cells with enough rides'
        )
      },
      annotations: places(['midtown', 'jfk', 'lga', 'queens'])
    },
    {
      id: 'when',
      title: 'Midnight and morning',
      headline: 'Midnight and morning on one scale',
      textAlternative:
        'Dark map of New York with a pickup glow for one three-hour window, drawn on a colour scale shared by all windows, with a bar chart of rides per hour below.',
      body: 'Same map, windows of equal length, one colour scale locked to the busiest window, so brightness compares. This window is **{{window}}** and keeps **{{kept}}** rides. Click the chart or pick a **Time window**; then try **Cell shape**, or **Show** drop-offs. The data start at New Year, so a window is not a typical day.',
      optionsMode: 'fresh',
      options: {measure: 'rides', swipe: 'off', scale: 'sqrt', hours: [0, 3]},
      controls: ['hourPresets', 'binning', 'show'],
      readouts: ['kept', 'window', 'pulseChart'],
      stage: 'reduce',
      basemap: ground('night'),
      camera: {...CITY_FRAMES.nyc, transitionMs: 1800},
      furniture: {
        title: cartouche('When does the city glow?', 'Pickups per km² in a window, locked scale')
      },
      annotations: places(['midtown', 'lower-manhattan', 'jfk'])
    }
  ]
});
