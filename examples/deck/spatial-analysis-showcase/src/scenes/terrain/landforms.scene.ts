// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
import {
  CURVATURE_KINDS,
  GEOMORPHON_CLASSES,
  GROUND_PIXEL_METERS,
  SCALE_PRESETS,
  WEISS_CLASSES,
  type LandformOptions
} from './landforms.style';

const notProduct =
  (...products: LandformOptions['product'][]) =>
  (state: LandformOptions) =>
    !products.includes(state.product);

const TPI_PRODUCTS: LandformOptions['product'][] = ['tpi', 'dev', 'devmax', 'scale'];

/**
 * Landforms of the Matterhorn: geomorphons, curvature, multi-scale topographic position and
 * Weiss landform classes computed on the GPU from one 2048 x 2048 DEM.
 */
export default defineScene<LandformOptions>({
  id: 'landforms',
  title: 'Ridges, valleys, hollows: landform classification',
  chapter: 'terrain',
  order: 3,
  summary:
    'Classify the Matterhorn and the Gorner glacier into ridges, spurs, hollows and valleys with geomorphons, curvature, multi-scale topographic position and Weiss landforms, all on the GPU.',
  contributors: [
    'GPUGeomorphons',
    'GPUTerrainCurvature',
    'GPUTerrainTopographicPosition',
    'GPUTerrainWeissLandforms',
    'GPUReliefShading'
  ],
  datasets: [{id: 'alps-dem', role: 'elevation'}],
  initialView: {longitude: 7.742, latitude: 45.985, zoom: 11.9},

  options: [
    {
      kind: 'select',
      id: 'geomorphonView',
      label: 'Output',
      group: 'Geomorphons (GPUGeomorphons)',
      apply: 'compile',
      default: 'forms',
      disabledWhen: notProduct('geomorphons'),
      help: 'The ten forms, or the raw rotation-invariant ternary pattern (GRASS ternary, 498 classes) from which they are derived.',
      options: [
        {value: 'forms', label: 'Ten landform forms'},
        {value: 'ternary', label: 'Ternary pattern code (498 classes)'}
      ]
    },
    {
      kind: 'slider',
      id: 'geomorphonRadius',
      label: 'Search radius',
      group: 'Geomorphons (GPUGeomorphons)',
      apply: 'compile',
      min: 4,
      max: 80,
      step: 2,
      default: 20,
      unit: 'px',
      disabledWhen: notProduct('geomorphons'),
      help: 'How far (in 6.6 m pixels) each of the 8 lines of sight looks. Small radii find small landforms (boulders, gullies); large radii find the shape of whole mountains. Changing it rebuilds the graph.'
    },
    {
      kind: 'slider',
      id: 'geomorphonSkip',
      label: 'Skip radius',
      group: 'Geomorphons (GPUGeomorphons)',
      apply: 'compile',
      min: 0,
      max: 10,
      step: 1,
      default: 0,
      unit: 'px',
      disabledWhen: notProduct('geomorphons'),
      help: 'Ignore this many pixels next to the cell. Skipping removes the finest noise; cells within skip + 1 of the border are invalid, as in GRASS.'
    },
    {
      kind: 'select',
      id: 'geomorphonComparison',
      label: 'Line-of-sight comparison',
      group: 'Geomorphons (GPUGeomorphons)',
      apply: 'compile',
      default: 'anglev1',
      disabledWhen: notProduct('geomorphons'),
      help: 'How each ray decides "higher", "lower" or "flat": by the largest elevation angle (v1), by the angle between the zenith and nadir lines (v2), or v2 with distance weighting. GRASS r.geomorphon offers the same three.',
      options: [
        {value: 'anglev1', label: 'anglev1 (GRASS default)'},
        {value: 'anglev2', label: 'anglev2'},
        {value: 'anglev2-distance', label: 'anglev2 with distance'}
      ]
    },
    {
      kind: 'slider',
      id: 'geomorphonFlatAngle',
      label: 'Flatness threshold',
      group: 'Geomorphons (GPUGeomorphons)',
      apply: 'param',
      min: 0.1,
      max: 15,
      step: 0.1,
      default: 1,
      unit: '°',
      disabledWhen: notProduct('geomorphons'),
      help: 'An elevation angle below this counts as flat. Raising it merges gentle slopes into flat and shoulder classes; it is a parameter write.'
    },
    {
      kind: 'slider',
      id: 'geomorphonFlatDistance',
      label: 'Flat distance',
      group: 'Geomorphons (GPUGeomorphons)',
      apply: 'param',
      min: 0,
      max: 2000,
      step: 50,
      default: 0,
      unit: 'm',
      format: value => (value === 0 ? 'off' : `${value} m`),
      disabledWhen: notProduct('geomorphons'),
      help: 'Beyond this ground distance the flatness threshold shrinks to the same height, so far cells need less elevation difference to count. 0 turns the rule off.'
    },
    {
      kind: 'select',
      id: 'curvatureKind',
      label: 'Curvature kind',
      group: 'Curvature (GPUTerrainCurvature)',
      apply: 'compile',
      default: 'profile',
      disabledWhen: notProduct('curvature'),
      help: 'Which of the Florinsky curvatures to compute. Convex hills are positive and valleys negative (WhiteboxTools signs).',
      options: CURVATURE_KINDS.map(entry => ({
        value: entry.value,
        label: entry.label,
        help: entry.help
      }))
    },
    {
      kind: 'select',
      id: 'curvatureMethod',
      label: 'Derivative estimator',
      group: 'Curvature (GPUTerrainCurvature)',
      apply: 'compile',
      default: 'evans-young',
      disabledWhen: notProduct('curvature'),
      help: 'How the partial derivatives are estimated: Evans-Young and Zevenbergen-Thorne use 3 x 3 windows; Florinsky fits a 5 x 5 polynomial (the WhiteboxTools estimator) that smooths noise. The 5 x 5 is less noisy and slightly blurrier.',
      options: [
        {value: 'evans-young', label: 'Evans-Young 3 x 3'},
        {value: 'zevenbergen-thorne', label: 'Zevenbergen-Thorne 3 x 3'},
        {value: 'florinsky', label: 'Florinsky 5 x 5'}
      ]
    },
    {
      kind: 'select',
      id: 'curvatureBorder',
      label: 'Border',
      group: 'Curvature (GPUTerrainCurvature)',
      apply: 'compile',
      default: 'clamp',
      disabledWhen: notProduct('curvature'),
      help: 'Edge handling: repeat the edge value or make the border cells nodata.',
      options: [
        {value: 'clamp', label: 'Clamp'},
        {value: 'nodata', label: 'Nodata'}
      ]
    },
    {
      kind: 'slider',
      id: 'flatGradient',
      label: 'Flat gradient',
      group: 'Curvature (GPUTerrainCurvature)',
      apply: 'param',
      min: 0,
      max: 0.3,
      step: 0.005,
      default: 0,
      disabledWhen: notProduct('curvature'),
      format: value => (value === 0 ? '1e-6 (default)' : value.toFixed(3)),
      help: 'Slope (rise over run) below which a cell is flat and the direction-dependent curvatures (profile, plan, ...) are set to 0, so noise on level ground is not read as shape.'
    },
    {
      kind: 'slider',
      id: 'zFactor',
      label: 'Vertical exaggeration',
      group: 'Curvature (GPUTerrainCurvature)',
      apply: 'param',
      min: 0.5,
      max: 3,
      step: 0.1,
      default: 1,
      unit: 'x',
      disabledWhen: notProduct('curvature'),
      help: 'Multiplies heights before curvature is taken.'
    },
    {
      kind: 'slider',
      id: 'ringRadiusInner',
      label: 'Ring radius, inner',
      group: 'Ring curvature (multi-radius)',
      apply: 'compile',
      min: 1,
      max: 8,
      step: 1,
      default: 2,
      unit: 'px',
      disabledWhen: state =>
        state.product !== 'curvature' || state.curvatureKind !== 'ring-multi-radius',
      help: 'First ring of 8 samples, in pixels. Ring curvature is the sum over rings of the mean height difference to the ring, divided by the radius.'
    },
    {
      kind: 'slider',
      id: 'ringRadiusOuter',
      label: 'Ring radius, outer',
      group: 'Ring curvature (multi-radius)',
      apply: 'compile',
      min: 2,
      max: 24,
      step: 1,
      default: 8,
      unit: 'px',
      disabledWhen: state =>
        state.product !== 'curvature' || state.curvatureKind !== 'ring-multi-radius',
      help: 'Second ring (kept above the inner radius). Together they pick the ridge widths that stand out.'
    },
    {
      kind: 'slider',
      id: 'ringGainInner',
      label: 'Inner ring gain',
      group: 'Ring curvature (multi-radius)',
      apply: 'param',
      min: 0,
      max: 2,
      step: 0.05,
      default: 0.55,
      disabledWhen: state =>
        state.product !== 'curvature' || state.curvatureKind !== 'ring-multi-radius',
      help: 'Weight of the inner ring (mt-image default 0.55).'
    },
    {
      kind: 'slider',
      id: 'ringGainOuter',
      label: 'Outer ring gain',
      group: 'Ring curvature (multi-radius)',
      apply: 'param',
      min: 0,
      max: 2,
      step: 0.05,
      default: 0.45,
      disabledWhen: state =>
        state.product !== 'curvature' || state.curvatureKind !== 'ring-multi-radius',
      help: 'Weight of the outer ring (mt-image default 0.45).'
    },
    {
      kind: 'toggle',
      id: 'ringSquash',
      label: 'Soft clip (Pade tanh)',
      group: 'Ring curvature (multi-radius)',
      apply: 'compile',
      default: false,
      disabledWhen: state =>
        state.product !== 'curvature' || state.curvatureKind !== 'ring-multi-radius',
      help: 'Squashes extreme values with a rational tanh so a few steep cliffs do not dominate the range.'
    },
    {
      kind: 'select',
      id: 'scalePreset',
      label: 'Scale set',
      group: 'Topographic position (GPUTerrainTopographicPosition)',
      apply: 'compile',
      default: 'landscape',
      disabledWhen: notProduct(...TPI_PRODUCTS),
      help: 'Eight window radii. Every scale costs almost nothing extra: one summed-area table serves them all in O(1) per pixel per scale.',
      options: [
        {
          value: 'fine',
          label: `Fine: 1 to 16 px (${Math.round(1 * GROUND_PIXEL_METERS)} to ${Math.round(16 * GROUND_PIXEL_METERS)} m)`
        },
        {
          value: 'landscape',
          label: `Landscape: 2 to 256 px (${Math.round(2 * GROUND_PIXEL_METERS)} m to ${((256 * GROUND_PIXEL_METERS) / 1000).toFixed(1)} km)`
        },
        {
          value: 'broad',
          label: `Broad: 8 to 512 px (${Math.round(8 * GROUND_PIXEL_METERS)} m to ${((512 * GROUND_PIXEL_METERS) / 1000).toFixed(1)} km)`
        }
      ]
    },
    {
      kind: 'slider',
      id: 'scaleIndex',
      label: 'Scale shown',
      group: 'Topographic position (GPUTerrainTopographicPosition)',
      apply: 'param',
      min: 0,
      max: 7,
      step: 1,
      default: 3,
      disabledWhen: notProduct('tpi', 'dev'),
      format: value => `scale ${value + 1} of 8`,
      help: 'Which of the eight scales to display (the radius appears in the readouts and tooltip). A parameter write: all eight scales are computed together.'
    },
    {
      kind: 'slider',
      id: 'innerFraction',
      label: 'Annulus inner fraction',
      group: 'Topographic position (GPUTerrainTopographicPosition)',
      apply: 'compile',
      min: 0,
      max: 0.8,
      step: 0.1,
      default: 0,
      disabledWhen: notProduct('tpi'),
      format: value => (value === 0 ? 'full disc' : `${(value * 100).toFixed(0)} % of the radius`),
      help: 'TPI compares a cell with the mean over an annulus. 0 uses the whole window; larger values exclude the middle so a feature does not hide itself.'
    },
    {
      kind: 'select',
      id: 'quantum',
      label: 'Elevation quantum',
      group: 'Topographic position (GPUTerrainTopographicPosition)',
      apply: 'compile',
      default: '256',
      disabledWhen: notProduct(...TPI_PRODUCTS, 'weiss'),
      help: 'The summed-area table is exact integer math on heights rounded to multiples of this step. 1/256 m (4 mm) is below the data quantisation; coarser steps trade accuracy for nothing here, but show the idea.',
      options: [
        {value: '256', label: '1/256 m (default, 4 mm)'},
        {value: '64', label: '1/64 m (16 mm)'},
        {value: '4', label: '1/4 m (25 cm)'}
      ]
    },
    {
      kind: 'slider',
      id: 'weissSmall',
      label: 'Small scale radius',
      group: 'Weiss landforms (GPUTerrainWeissLandforms)',
      apply: 'compile',
      min: 1,
      max: 20,
      step: 1,
      default: 4,
      unit: 'px',
      disabledWhen: notProduct('weiss'),
      help: 'Window radius of the small TPI, which finds ridges and valleys at the scale of a gully or a rib.'
    },
    {
      kind: 'slider',
      id: 'weissLarge',
      label: 'Large scale radius',
      group: 'Weiss landforms (GPUTerrainWeissLandforms)',
      apply: 'compile',
      min: 5,
      max: 120,
      step: 5,
      default: 40,
      unit: 'px',
      disabledWhen: notProduct('weiss'),
      help: 'Window radius of the large TPI, which decides whether the surroundings are a valley or a mountain.'
    },
    {
      kind: 'select',
      id: 'weissStandardization',
      label: 'Standardization',
      group: 'Weiss landforms (GPUTerrainWeissLandforms)',
      apply: 'compile',
      default: 'global',
      disabledWhen: notProduct('weiss'),
      help: "Global z-scores each TPI against the whole tile (Weiss 2001, Jenness): classes depend on the tile. Local divides by the window's own variation (DEV): tile-independent and cheaper.",
      options: [
        {value: 'global', label: 'Global (mean and standard deviation of the tile)'},
        {value: 'local', label: 'Local (DEV of each window)'}
      ]
    },
    {
      kind: 'slider',
      id: 'weissThreshold',
      label: 'Position threshold',
      group: 'Weiss landforms (GPUTerrainWeissLandforms)',
      apply: 'param',
      min: 0.2,
      max: 2,
      step: 0.05,
      default: 1,
      unit: 'sd',
      disabledWhen: notProduct('weiss'),
      help: 'Standardized TPI below minus this is "low", above plus this is "high", between is "mid". Weiss used 1 standard deviation.'
    },
    {
      kind: 'slider',
      id: 'weissSlope',
      label: 'Plain slope limit',
      group: 'Weiss landforms (GPUTerrainWeissLandforms)',
      apply: 'param',
      min: 0,
      max: 25,
      step: 1,
      default: 5,
      unit: '°',
      disabledWhen: notProduct('weiss'),
      help: 'A cell that is mid on both scales is a plain if its slope is at or below this, an open slope otherwise.'
    },
    {
      kind: 'select',
      id: 'product',
      label: 'Classification',
      group: 'Display',
      apply: 'param',
      default: 'geomorphons',
      help: 'Each is a compiled graph built the first time you pick it (and rebuilt when one of its compile-time options changes).',
      options: [
        {value: 'geomorphons', label: 'Geomorphons (10 landform forms)'},
        {value: 'curvature', label: 'Curvature (15 kinds + ring)'},
        {value: 'tpi', label: 'TPI at one scale'},
        {value: 'dev', label: 'DEV (standardized) at one scale'},
        {value: 'devmax', label: 'DEVmax over all scales'},
        {value: 'scale', label: 'Characteristic scale (radius of DEVmax)'},
        {value: 'weiss', label: 'Weiss landforms (10 classes)'}
      ]
    },
    {
      kind: 'toggle',
      id: 'underlay',
      label: 'Hillshade underlay',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Gray multidirectional hillshade beneath the classes so you can read the terrain.'
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
      default: 0.75,
      help: 'Lower it to see more of the underlay.'
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp',
      group: 'Display',
      apply: 'param',
      default: 'viridis',
      disabledWhen: state =>
        state.product !== 'scale' &&
        !(state.product === 'geomorphons' && state.geomorphonView === 'ternary'),
      help: 'Ramp for the sequential products (characteristic scale and ternary patterns). Signed products use a diverging blue-white-red ramp.',
      options: [
        {value: 'viridis', label: 'Viridis'},
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'cividis', label: 'Cividis (color-blind optimised)'}
      ]
    },
    {
      kind: 'toggle',
      id: 'autoStretch',
      label: 'Percentile range',
      group: 'Display',
      apply: 'param',
      default: true,
      disabledWhen: notProduct('curvature', 'tpi'),
      help: 'Signed curvature and TPI have no natural scale: set the symmetric color range from a percentile of the values (read back from a GPU histogram).'
    },
    {
      kind: 'slider',
      id: 'clipPercent',
      label: 'Percent clipped at each end',
      group: 'Display',
      apply: 'param',
      min: 0.5,
      max: 10,
      step: 0.5,
      default: 2,
      unit: '%',
      disabledWhen: state => !['curvature', 'tpi'].includes(state.product) || !state.autoStretch,
      help: 'Cells beyond these percentiles saturate. Larger values give more contrast in the middle.'
    },
    {
      kind: 'slider',
      id: 'rangeScale',
      label: 'Color range scale',
      group: 'Display',
      apply: 'param',
      min: 0.25,
      max: 3,
      step: 0.05,
      default: 1,
      unit: 'x',
      disabledWhen: state =>
        state.product === 'weiss' ||
        (state.product === 'geomorphons' && state.geomorphonView === 'forms'),
      help: 'Multiplies the value range of the ramp (for DEV, +/- 2.5 standard deviations at 1x).'
    },
    {
      kind: 'button',
      id: 'measure',
      label: 'Time this classification',
      group: 'Under the hood',
      help: 'Runs the displayed graph outside the frame and reports GPU time.'
    }
  ],

  readouts: [
    {id: 'grid', label: 'Raster'},
    {
      id: 'classes',
      label: 'Largest classes',
      help: 'Share of the valid cells in each of the four most common classes, from a GPU readback after the map settles.'
    },
    {id: 'peakShare', label: 'Peak / mountain-top share'},
    {id: 'scaleRadius', label: 'Scale shown'},
    {id: 'stretch', label: 'Color range'},
    {id: 'median', label: 'Median value'},
    {id: 'extent', label: 'Value range'},
    {id: 'timing', label: 'GPU time'}
  ],

  legends: state => {
    if (state.product === 'geomorphons' && state.geomorphonView === 'forms') {
      return [
        {
          kind: 'categories',
          title: 'Geomorphon forms (GRASS colors)',
          entries: GEOMORPHON_CLASSES.map(entry => ({color: entry.color, label: entry.label})),
          note: 'Each cell is classified from the pattern of higher, equal and lower cells along 8 lines of sight.'
        }
      ];
    }
    if (state.product === 'weiss') {
      return [
        {
          kind: 'categories',
          title: 'Weiss landforms',
          entries: WEISS_CLASSES.map(entry => ({color: entry.color, label: entry.label}))
        }
      ];
    }
    if (state.product === 'geomorphons') {
      return [
        {
          kind: 'ramp',
          title: 'Ternary pattern code',
          ramp: state.ramp,
          extent: [0, 6561 * state.rangeScale],
          labels: ['code 0', 'code 6561']
        }
      ];
    }
    if (state.product === 'curvature' || state.product === 'tpi') {
      return [
        {
          kind: 'ramp',
          id: 'stretch',
          title:
            state.product === 'tpi'
              ? 'TPI: below (blue) / above (red) the surroundings'
              : 'Curvature: concave (blue) / convex (red)',
          ramp: 'diverging',
          extent: state.autoStretch
            ? 'gpu'
            : state.product === 'tpi'
              ? [-10 * state.rangeScale, 10 * state.rangeScale]
              : [-0.1 * state.rangeScale, 0.1 * state.rangeScale],
          unit: state.product === 'tpi' ? 'm' : '1/m',
          format: value => (state.product === 'tpi' ? value.toFixed(1) : value.toPrecision(2))
        }
      ];
    }
    if (state.product === 'scale') {
      const radii = SCALE_PRESETS[state.scalePreset];
      return [
        {
          kind: 'ramp',
          title: 'Window radius of the strongest landform',
          ramp: state.ramp,
          extent: [0, radii[radii.length - 1] * GROUND_PIXEL_METERS * state.rangeScale],
          unit: 'm',
          format: value => value.toFixed(0)
        }
      ];
    }
    return [
      {
        kind: 'ramp',
        title: `${state.product === 'dev' ? 'DEV' : 'DEVmax'}: below (blue) / above (red), standard deviations`,
        ramp: 'diverging',
        extent: [-2.5 * state.rangeScale, 2.5 * state.rangeScale],
        unit: 'sd',
        format: value => value.toFixed(1)
      }
    ];
  },

  story: [
    {
      id: 'question',
      controls: ['product'],
      readouts: ['classes', 'peakShare'],
      title: 'Is this a ridge, a hollow or a valley?',
      body: 'Ask a mountaineer and they will name landforms by eye: the Hornli ridge, the hollow of a cirque, the valley the glacier carved. A computer needs a rule. **`GPUGeomorphons`** (after GRASS `r.geomorphon`, Jasiewicz and Stepinski 2013) gives it one: from every cell it looks along 8 lines of sight and records whether the terrain ahead is higher, level or lower. The pattern of eight such signs *is* the landform.\n\nThere are 498 distinct patterns, which collapse to **ten forms**: flat, peak, ridge, shoulder, spur, slope, hollow, footslope, valley and pit. The legend uses the standard GRASS colors. Peaks and ridges are dark red, valleys blue; the readouts count how much of the tile each form covers. Pick another map with **Classification** below to compare it with this one.',
      options: {product: 'geomorphons', underlay: true, opacity: 0.75},
      camera: {longitude: 7.742, latitude: 45.985, zoom: 11.9, transitionMs: 1400}
    },
    {
      id: 'geomorphon-scale',
      controls: ['geomorphonRadius', 'geomorphonFlatAngle', 'geomorphonSkip'],
      readouts: ['classes'],
      title: 'The answer depends on how far you look',
      body: 'Landform is scale-dependent: a rock rib is a ridge at 100 m and part of a slope at 2 km. Slide the **Search radius** below (this is a compile-time option, so the graph rebuilds once) and the Matterhorn changes from a mosaic of small ribs and gullies to a handful of large ridges.\n\nThe **Flatness threshold** below works the other way: it is a parameter write, so you can drag it and watch gentle slopes turn into *flat* or *shoulder* instead. Raise it on the glaciers to see them go flat. **Skip radius** drops the cells right next to the center, which removes pixel-scale noise.',
      options: {geomorphonRadius: 40, geomorphonFlatAngle: 3},
      camera: {longitude: 7.7, latitude: 45.98, zoom: 12.9, transitionMs: 1500}
    },
    {
      id: 'curvature',
      controls: ['curvatureKind', 'curvatureMethod'],
      readouts: ['stretch'],
      title: 'Curvature: where the surface bends',
      body: "**`GPUTerrainCurvature`** measures how the surface bends. **Profile curvature** is the bending along the line of steepest descent: positive (red) where a slope steepens downhill, negative (blue) where it flattens, so convex shoulders are red and concave foot-slopes blue. **Plan curvature** bends across the slope and separates spurs (convex) from gullies (concave); try it with **Curvature kind** below.\n\nCurvature is noisy by nature: it is a second derivative. The **Derivative estimator** below swaps the 3 x 3 windows for Florinsky's 5 x 5 polynomial, which smooths. The color range is set from the 2nd to 98th percentile; the legend shows the value.",
      options: {product: 'curvature', curvatureKind: 'profile', curvatureMethod: 'evans-young'},
      camera: {longitude: 7.742, latitude: 45.985, zoom: 12.4, transitionMs: 1400}
    },
    {
      id: 'ring',
      controls: [
        'ringRadiusInner',
        'ringRadiusOuter',
        'ringGainInner',
        'ringGainOuter',
        'ringSquash'
      ],
      readouts: [],
      title: 'Ring curvature: ridge lines at chosen widths',
      body: 'The classic measures give a result at pixel scale, which on a 6.6 m DEM is mostly rock texture. The **ring curvature** (ported from mt-image) averages the height difference between a cell and rings of 8 samples at chosen radii, `sum(gain_k * sum(z_c - z_i) / (8 r_k * cell))`, so only ridges and gullies at the ring scale stand out. Set **Ring radius, inner** and **Ring radius, outer** and their gains (**Inner ring gain**, **Outer ring gain**) below: a small inner ring picks out ribs, a large outer ring the main ridge system.\n\nThe optional **Soft clip (Pade tanh)** squashes the extremes so a few cliffs do not set the range.',
      options: {
        product: 'curvature',
        curvatureKind: 'ring-multi-radius',
        ringRadiusInner: 2,
        ringRadiusOuter: 8
      },
      camera: {longitude: 7.7, latitude: 45.98, zoom: 12.9, transitionMs: 1400}
    },
    {
      id: 'tpi',
      controls: ['scaleIndex', 'innerFraction'],
      readouts: ['scaleRadius'],
      title: 'Topographic position: high or low relative to the surroundings',
      body: '**`GPUTerrainTopographicPosition`** answers "is this cell above or below its neighbourhood?". **TPI** is the cell height minus the mean over a window; **DEV** divides that by the window\'s own variation so scales are comparable. It evaluates eight window sizes at once, because one **summed-area table** (exact 64-bit integer arithmetic on heights rounded to 4 mm) gives any window mean in O(1).\n\nDrag **Scale shown** below from fine to broad: a small window finds individual ribs, a large one separates the whole valley from the ridge system. Red cells sit above their surroundings, blue below. **Annulus inner fraction** hollows the window into a ring, so a feature does not hide itself in its own mean.',
      options: {product: 'tpi', scalePreset: 'landscape', scaleIndex: 3},
      camera: {longitude: 7.742, latitude: 45.985, zoom: 11.9, transitionMs: 1400}
    },
    {
      id: 'scale',
      controls: ['product', 'scalePreset'],
      readouts: ['extent'],
      title: 'Which scale does each place belong to?',
      body: 'Instead of picking one window, **DEVmax** (Lindsay et al. 2015) takes, for every cell, the largest absolute DEV over all scales and also records *which* scale produced it. The map shown now is that characteristic scale in meters: small values mark features that are strongest at the rib scale, large ones belong to the big landforms (the Matterhorn itself, the Gorner basin).\n\nSwitch **Classification** below to *DEVmax* to see the strength. Change **Scale set** to *Fine* or *Broad* to look at other ranges.',
      options: {product: 'scale', scalePreset: 'landscape'}
    },
    {
      id: 'weiss',
      controls: ['weissSmall', 'weissLarge', 'weissStandardization', 'weissThreshold', 'product'],
      readouts: ['classes'],
      title: 'Ten classes from two scales, and the limits',
      body: '**`GPUTerrainWeissLandforms`** (Weiss 2001) combines a small and a large standardized TPI with the slope into ten classes: canyons and drainage lines, plains, open slopes, local and midslope ridges and mountain tops. The small scale (default 4 px, 26 m) says what the cell is *locally*; the large scale (40 px, 260 m) says what *setting* it is in.\n\n**Caveats.** Every classification here depends on its scales: change them and the map changes. *Global* standardization uses statistics of this tile, so the same terrain can be classed differently in a different tile; *local* standardization fixes that. All results are tile-local and none of them replaces field knowledge. **Try:** change the **Position threshold**, set **Small scale radius** to 10 px, switch **Standardization** between *Global* and *Local*, or compare *Weiss* with *Geomorphons* in **Classification**.',
      options: {product: 'weiss', weissSmall: 4, weissLarge: 40, weissThreshold: 1},
      camera: {longitude: 7.742, latitude: 45.985, zoom: 11.9, transitionMs: 1400}
    }
  ],

  about: {
    what: '`GPUGeomorphons` classifies each cell into one of ten forms from eight lines of sight (GRASS `r.geomorphon`). `GPUTerrainCurvature` computes the 15 Florinsky curvature kinds and an mt-image multi-radius ring curvature. `GPUTerrainTopographicPosition` evaluates TPI, DEV and DEVmax at up to 64 scales in O(1) per pixel per scale from an exact summed-area table. `GPUTerrainWeissLandforms` classifies ten landforms from two TPI scales and slope.',
    why: 'Landform classes drive habitat and soil models, hazard mapping (rockfall source areas, avalanche starting zones) and the generalisation of terrain for maps. Scale is the main modelling choice, so being able to change it interactively is the point.',
    howToRead:
      'Geomorphons and Weiss: categorical, see the legend; hover for the form name and its meaning. Curvature, TPI and DEV: blue is concave or below, red is convex or above, centered on zero with the middle faded so the hillshade shows through. Characteristic scale: brighter means a larger landform. The data is a 6.6 m DEM, so the smallest forms are rock texture; the references are GRASS r.geomorphon (geomorphons), WhiteboxTools (curvature signs and the 5 x 5 estimator) and Weiss 2001 / Lindsay et al. 2015 (TPI, DEVmax).'
  },

  create: async ctx => (await import('./landforms.compute')).createLandforms(ctx),

  snippet: state => `import {
  GPUGeomorphons, GPUTerrainCurvature, GPUTerrainTopographicPosition, GPUTerrainWeissLandforms,
  getGPUGeomorphonsParameterValues, getGPUTerrainWeissLandformsParameterValues
} from '@luma.gl/experimental/gpu-terrain';

// the tile is Web Mercator: cellSize is equatorial Mercator meters, edges are normalized Mercator y
const cell = {cellSize: [9.5546, 9.5546], northEdge: 0.3556, southEdge: 0.3112};
graph.add(new GPUGeomorphons({
  width, height, elevation, settings: geomorphonSettings.importToGraph(graph),
  searchRadius: ${state.geomorphonRadius}, skipRadius: ${state.geomorphonSkip}, comparison: '${state.geomorphonComparison}',
  forms, cellSizeMode: 'web-mercator'
}));
graph.add(new GPUTerrainCurvature({
  width, height, elevation, settings: curvatureSettings.importToGraph(graph),
  method: '${state.curvatureMethod}', curvatures: {${state.curvatureKind === 'ring-multi-radius' ? "'profile'" : JSON.stringify(state.curvatureKind)}: curvature}, cellSizeMode: 'web-mercator'
}));
graph.add(new GPUTerrainTopographicPosition({
  width, height, elevation, scales: [${SCALE_PRESETS[state.scalePreset].map(radius => `{radius: ${radius}}`).join(', ')}],
  deviationFromMean, maximumDeviation, maximumDeviationRadius
}));
graph.add(new GPUTerrainWeissLandforms({
  width, height, elevation, settings: weissSettings.importToGraph(graph),
  smallScale: {radius: ${state.weissSmall}}, largeScale: {radius: ${state.weissLarge}},
  standardization: '${state.weissStandardization}', landforms, cellSizeMode: 'web-mercator'
}));
const compiled = graph.compile(); // once

// per frame: parameter writes, then one encode
geomorphonSettings.write(getGPUGeomorphonsParameterValues({
  ...cell, flatThresholdDegrees: ${state.geomorphonFlatAngle}, flatDistance: ${state.geomorphonFlatDistance}
}));
weissSettings.write(getGPUTerrainWeissLandformsParameterValues({
  ...cell, standardThreshold: ${state.weissThreshold}, slopeThresholdDegrees: ${state.weissSlope}
}));
compiled.encode(commandEncoder, {parameters: undefined});`
});
