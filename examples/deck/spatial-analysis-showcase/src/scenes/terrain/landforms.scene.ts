// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import type {ClassTable} from '../../cartography/types';
import {defineScene, type LegendSpec, type OptionSpec} from '../scene';
import {
  getAgreementLegend,
  getCurvatureKind,
  getCurvatureName,
  getLandformLegend,
  getReadsPerCell,
  getScaleLegend,
  getSignedKey,
  getSignedProductLegend,
  getWeissKey,
  GROUND_PIXEL_METERS,
  makeDeviationTable,
  makeLandformTables,
  OTHER_CURVATURE_KINDS,
  SCALE_PRESETS,
  type LandformOptions,
  type LandformTables
} from './landforms.style';
import {getTerrainFurniture, TERRAIN_FRAMES, terrainCartouche} from './terrain-furniture';
import type {TerrainGroundTone} from './terrain-palettes';

const onlyProduct =
  (...products: LandformOptions['product'][]) =>
  (state: LandformOptions) =>
    !products.includes(state.product);

/** Geomorphons are shown: the product itself, or the Weiss product in its geomorphon view. */
const showsGeomorphons = (state: LandformOptions) =>
  state.product === 'geomorphons' || (state.product === 'weiss' && state.view !== 'weiss');

const notGeomorphons = (state: LandformOptions) => !showsGeomorphons(state);

const GEOMORPHON_GROUP = 'Geomorphons (GPUGeomorphons)';
const CURVATURE_GROUP = 'Curvature (GPUTerrainCurvature)';
const POSITION_GROUP = 'Topographic position (GPUTerrainTopographicPosition)';
const WEISS_GROUP = 'Weiss landforms (GPUTerrainWeissLandforms)';

const metersOf = (pixels: number) => Math.round(pixels * GROUND_PIXEL_METERS);

const options: readonly OptionSpec<LandformOptions>[] = [
  {
    kind: 'select',
    id: 'product',
    label: 'Classification',
    group: 'Display',
    apply: 'compile',
    display: 'chips',
    default: 'geomorphons',
    help: 'Which classification to show. Each is compiled the first time you pick it.',
    options: [
      {value: 'geomorphons', label: 'Geomorphons'},
      {value: 'curvature', label: 'Curvature'},
      {value: 'position', label: 'Topographic position'},
      {value: 'weiss', label: 'Weiss and agreement'}
    ]
  },
  {
    kind: 'select',
    id: 'palette',
    label: 'Palette',
    group: GEOMORPHON_GROUP,
    apply: 'param',
    display: 'segmented',
    default: 'quiet',
    disabledWhen: notGeomorphons,
    help: 'Quiet ground keeps the common slope clear and colours the rare forms; the GRASS standard colours every class at full strength.',
    options: [
      {value: 'quiet', label: 'Quiet ground'},
      {value: 'grass', label: 'GRASS standard'}
    ]
  },
  {
    kind: 'slider',
    id: 'geomorphonRadius',
    label: 'Look-out radius',
    group: GEOMORPHON_GROUP,
    apply: 'compile',
    min: 6,
    max: 96,
    step: 2,
    default: 20,
    unit: 'px',
    disabledWhen: notGeomorphons,
    describe: value =>
      `${metersOf(value)} m on the ground, ${getReadsPerCell(value)} reads per cell`,
    help: 'How far each of the eight rays looks, in 6.6 m pixels. Small radii find ribs and gullies; large radii find the shape of whole mountains. Changing it compiles the graph once.'
  },
  {
    kind: 'preset',
    id: 'radiusPreset',
    label: 'Look-out',
    group: GEOMORPHON_GROUP,
    help: 'Three look-out radii to compare: a rib, a ridge system, a whole mountain. Each writes the look-out radius.',
    presets: [
      {label: '8 px', values: {geomorphonRadius: 8}},
      {label: '30 px', values: {geomorphonRadius: 30}},
      {label: '90 px', values: {geomorphonRadius: 90}}
    ]
  },
  {
    kind: 'button',
    id: 'sweep',
    label: 'Sweep the look-out radius',
    group: GEOMORPHON_GROUP,
    disabledWhen: onlyProduct('geomorphons') as (state: never) => boolean,
    help: 'Runs five radii once, compiling each graph once, and plots the share of each class group against the radius.'
  },
  {
    kind: 'slider',
    id: 'geomorphonFlatAngle',
    label: 'Flatness threshold',
    group: GEOMORPHON_GROUP,
    apply: 'param',
    min: 0.1,
    max: 15,
    step: 0.1,
    default: 1,
    unit: '°',
    disabledWhen: notGeomorphons,
    help: 'An elevation angle below this counts as flat. Raising it merges gentle slopes into flat and shoulder classes; it is a parameter write.'
  },
  {
    kind: 'toggle',
    id: 'showRays',
    label: 'Show the eight rays',
    group: GEOMORPHON_GROUP,
    apply: 'param',
    default: false,
    disabledWhen: onlyProduct('geomorphons'),
    help: 'Draws the eight lines of sight of one cell. Click the map to pin another cell.'
  },
  {
    kind: 'toggle',
    id: 'compareRadius',
    label: 'Compare with a short look-out',
    group: GEOMORPHON_GROUP,
    apply: 'compile',
    default: false,
    disabledWhen: onlyProduct('geomorphons'),
    help: 'Computes a second class raster with a short look-out; the divider compares it with the chosen radius.'
  },
  {
    kind: 'slider',
    id: 'geomorphonSkip',
    label: 'Skip radius',
    group: GEOMORPHON_GROUP,
    apply: 'compile',
    min: 0,
    max: 10,
    step: 1,
    default: 0,
    unit: 'px',
    expert: true,
    disabledWhen: notGeomorphons,
    help: 'Ignore this many pixels next to the cell. Skipping removes the finest noise; cells within skip + 1 of the border are invalid, as in GRASS.'
  },
  {
    kind: 'select',
    id: 'geomorphonComparison',
    label: 'Line-of-sight comparison',
    group: GEOMORPHON_GROUP,
    apply: 'compile',
    default: 'anglev1',
    expert: true,
    disabledWhen: notGeomorphons,
    help: 'How each ray decides higher, lower or flat: by the largest elevation angle (v1), by the angle between the zenith and nadir lines (v2), or v2 with distance weighting. GRASS r.geomorphon offers the same three.',
    options: [
      {value: 'anglev1', label: 'anglev1 (GRASS default)'},
      {value: 'anglev2', label: 'anglev2'},
      {value: 'anglev2-distance', label: 'anglev2 with distance'}
    ]
  },
  {
    kind: 'slider',
    id: 'geomorphonFlatDistance',
    label: 'Flat distance',
    group: GEOMORPHON_GROUP,
    apply: 'param',
    min: 0,
    max: 2000,
    step: 50,
    default: 0,
    unit: 'm',
    expert: true,
    format: value => (value === 0 ? 'off' : `${value} m`),
    disabledWhen: notGeomorphons,
    help: 'Beyond this ground distance the flatness threshold shrinks to the same height, so far cells need less elevation difference to count. 0 turns the rule off.'
  },
  {
    kind: 'select',
    id: 'curvatureKind',
    label: 'Bend',
    group: CURVATURE_GROUP,
    apply: 'param',
    display: 'segmented',
    default: 'profile',
    disabledWhen: onlyProduct('curvature'),
    help: 'Profile curvature bends along the slope (it steepens or flattens downhill); plan curvature bends across it (contours bow outward on spurs, inward in gullies). Both come from one graph.',
    options: [
      {value: 'profile', label: 'Profile'},
      {value: 'plan', label: 'Plan'}
    ]
  },
  {
    kind: 'select',
    id: 'curvatureMethod',
    label: 'Fit',
    group: CURVATURE_GROUP,
    apply: 'compile',
    display: 'chips',
    default: 'evans-young',
    disabledWhen: onlyProduct('curvature'),
    help: 'How the derivatives are estimated: Evans-Young and Zevenbergen-Thorne fit 3 x 3 windows; Florinsky fits a 5 x 5 polynomial, which smooths the quantised heights and blurs slightly.',
    options: [
      {value: 'evans-young', label: '3 x 3 (Evans-Young)'},
      {value: 'zevenbergen-thorne', label: '3 x 3 (Zevenbergen-Thorne)'},
      {value: 'florinsky', label: '5 x 5 (Florinsky)'}
    ]
  },
  {
    kind: 'select',
    id: 'curvatureMoreKinds',
    label: 'Other curvature kinds',
    group: CURVATURE_GROUP,
    apply: 'compile',
    default: 'none',
    expert: true,
    disabledWhen: onlyProduct('curvature'),
    help: 'The other thirteen Florinsky kinds and the multi-radius ring curvature; they replace the profile and plan chips while one is chosen.',
    options: [
      {value: 'none', label: 'Use profile or plan'},
      ...OTHER_CURVATURE_KINDS.map(entry => ({
        value: entry.value,
        label: entry.label,
        help: entry.help
      }))
    ]
  },
  {
    kind: 'select',
    id: 'curvatureBorder',
    label: 'Border',
    group: CURVATURE_GROUP,
    apply: 'compile',
    default: 'clamp',
    expert: true,
    disabledWhen: onlyProduct('curvature'),
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
    group: CURVATURE_GROUP,
    apply: 'param',
    min: 0,
    max: 0.3,
    step: 0.005,
    default: 0,
    expert: true,
    disabledWhen: onlyProduct('curvature'),
    format: value => (value === 0 ? '1e-6 (default)' : value.toFixed(3)),
    help: 'Slope (rise over run) below which a cell is flat and the direction-dependent curvatures are set to 0, so noise on level ground is not read as shape.'
  },
  {
    kind: 'slider',
    id: 'zFactor',
    label: 'Vertical exaggeration',
    group: CURVATURE_GROUP,
    apply: 'param',
    min: 0.5,
    max: 3,
    step: 0.1,
    default: 1,
    unit: 'x',
    expert: true,
    disabledWhen: onlyProduct('curvature'),
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
    expert: true,
    disabledWhen: state =>
      state.product !== 'curvature' || getCurvatureKind(state) !== 'ring-multi-radius',
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
    expert: true,
    disabledWhen: state =>
      state.product !== 'curvature' || getCurvatureKind(state) !== 'ring-multi-radius',
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
    expert: true,
    disabledWhen: state =>
      state.product !== 'curvature' || getCurvatureKind(state) !== 'ring-multi-radius',
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
    expert: true,
    disabledWhen: state =>
      state.product !== 'curvature' || getCurvatureKind(state) !== 'ring-multi-radius',
    help: 'Weight of the outer ring (mt-image default 0.45).'
  },
  {
    kind: 'toggle',
    id: 'ringSquash',
    label: 'Soft clip (Pade tanh)',
    group: 'Ring curvature (multi-radius)',
    apply: 'compile',
    default: false,
    expert: true,
    disabledWhen: state =>
      state.product !== 'curvature' || getCurvatureKind(state) !== 'ring-multi-radius',
    help: 'Squashes extreme values with a rational tanh so a few steep cliffs do not dominate the range.'
  },
  {
    kind: 'select',
    id: 'positionProduct',
    label: 'Measure',
    group: POSITION_GROUP,
    apply: 'compile',
    display: 'chips',
    default: 'tpi',
    disabledWhen: onlyProduct('position'),
    help: "TPI is the height above the window mean in metres; DEV divides it by the window's own variation; the dominant scale is the window at which a cell stands out most (DEVmax).",
    options: [
      {value: 'tpi', label: 'TPI'},
      {value: 'dev', label: 'DEV'},
      {value: 'scale', label: 'Dominant scale'}
    ]
  },
  {
    kind: 'slider',
    id: 'scaleIndex',
    label: 'Scale',
    group: POSITION_GROUP,
    apply: 'param',
    display: 'stepper',
    min: 0,
    max: 7,
    step: 1,
    default: 3,
    disabledWhen: state => state.product !== 'position' || state.positionProduct === 'scale',
    format: value => `scale ${value + 1} of 8`,
    describe: (value, state) => {
      const radius = SCALE_PRESETS[state.scalePreset][value];
      return `radius ${radius} px = ${metersOf(radius)} m, window ${metersOf(2 * radius + 1)} m wide`;
    },
    help: 'Which of the eight window sizes to show. All eight are computed together from one summed-area table, so this is a parameter write.'
  },
  {
    kind: 'select',
    id: 'scalePreset',
    label: 'Scale set',
    group: POSITION_GROUP,
    apply: 'compile',
    default: 'landscape',
    disabledWhen: onlyProduct('position'),
    help: 'Eight window radii. Every scale costs almost nothing extra: one summed-area table serves them all.',
    options: [
      {
        value: 'fine',
        label: `Fine: 1 to 16 px (${metersOf(1)} to ${metersOf(16)} m)`
      },
      {
        value: 'landscape',
        label: `Landscape: 2 to 256 px (${metersOf(2)} m to ${(metersOf(256) / 1000).toFixed(1)} km)`
      },
      {
        value: 'broad',
        label: `Broad: 8 to 512 px (${metersOf(8)} m to ${(metersOf(512) / 1000).toFixed(1)} km)`
      }
    ]
  },
  {
    kind: 'slider',
    id: 'innerFraction',
    label: 'Annulus inner fraction',
    group: POSITION_GROUP,
    apply: 'compile',
    min: 0,
    max: 0.8,
    step: 0.1,
    default: 0,
    expert: true,
    disabledWhen: onlyProduct('position'),
    format: value => (value === 0 ? 'full window' : `${(value * 100).toFixed(0)} % of the radius`),
    help: 'TPI compares a cell with the mean over an annulus. 0 uses the whole window; larger values exclude the middle so a feature does not hide itself.'
  },
  {
    kind: 'select',
    id: 'quantum',
    label: 'Elevation quantum',
    group: POSITION_GROUP,
    apply: 'compile',
    default: '256',
    expert: true,
    disabledWhen: onlyProduct('position', 'weiss'),
    help: 'The summed-area table is exact integer maths on heights rounded to multiples of this step. 1/256 m (4 mm) is below the data quantisation; coarser steps show the idea.',
    options: [
      {value: '256', label: '1/256 m (default, 4 mm)'},
      {value: '64', label: '1/64 m (16 mm)'},
      {value: '4', label: '1/4 m (25 cm)'}
    ]
  },
  {
    kind: 'select',
    id: 'view',
    label: 'Show',
    group: WEISS_GROUP,
    apply: 'param',
    display: 'chips',
    default: 'weiss',
    disabledWhen: onlyProduct('weiss'),
    help: 'Weiss landforms, geomorphons, or the map of where the two disagree. All three are computed; this only changes what is drawn.',
    options: [
      {value: 'weiss', label: 'Weiss'},
      {value: 'geomorphons', label: 'Geomorphons'},
      {value: 'agreement', label: 'Agreement'}
    ]
  },
  {
    kind: 'slider',
    id: 'weissSmall',
    label: 'Small scale radius',
    group: WEISS_GROUP,
    apply: 'compile',
    min: 1,
    max: 20,
    step: 1,
    default: 4,
    unit: 'px',
    disabledWhen: onlyProduct('weiss'),
    describe: value => `${metersOf(value)} m radius`,
    help: 'Window radius of the small TPI, which finds ridges and valleys at the scale of a gully or a rib.'
  },
  {
    kind: 'slider',
    id: 'weissLarge',
    label: 'Large scale radius',
    group: WEISS_GROUP,
    apply: 'compile',
    min: 5,
    max: 120,
    step: 5,
    default: 40,
    unit: 'px',
    disabledWhen: onlyProduct('weiss'),
    describe: value => `${metersOf(value)} m radius`,
    help: 'Window radius of the large TPI, which decides whether the surroundings are a valley or a mountain.'
  },
  {
    kind: 'select',
    id: 'weissStandardization',
    label: 'Standardisation',
    group: WEISS_GROUP,
    apply: 'compile',
    display: 'segmented',
    default: 'global',
    disabledWhen: onlyProduct('weiss'),
    help: "Global z-scores each TPI against the whole tile (Weiss 2001, Jenness): classes depend on the tile. Local divides by the window's own variation (DEV): tile-independent.",
    options: [
      {value: 'global', label: 'Global'},
      {value: 'local', label: 'Local'}
    ]
  },
  {
    kind: 'slider',
    id: 'weissThreshold',
    label: 'Position threshold',
    group: WEISS_GROUP,
    apply: 'param',
    min: 0.2,
    max: 2,
    step: 0.05,
    default: 1,
    unit: 'sd',
    disabledWhen: onlyProduct('weiss'),
    help: 'Standardised TPI below minus this is low, above plus this is high, between is mid. Weiss used 1 standard deviation.'
  },
  {
    kind: 'slider',
    id: 'weissSlope',
    label: 'Plain slope limit',
    group: WEISS_GROUP,
    apply: 'param',
    min: 0,
    max: 25,
    step: 1,
    default: 5,
    unit: '°',
    expert: true,
    disabledWhen: onlyProduct('weiss'),
    help: 'A cell that is mid on both scales is a plain if its slope is at or below this, an open slope otherwise.'
  },
  {
    kind: 'toggle',
    id: 'underlay',
    label: 'Relief ground',
    group: 'Display',
    apply: 'param',
    default: true,
    expert: true,
    help: 'The warm-lit, cool-shaded relief under the classes. Switch it off to see the classes alone on paper.'
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
    default: 1,
    expert: true,
    help: 'Scales the opacity of the classes; lower it to see more of the relief.'
  },
  {
    kind: 'button',
    id: 'measure',
    label: 'Time this classification',
    group: 'Under the hood',
    expert: true,
    help: 'Runs the displayed graphs outside the frame and reports GPU time.'
  }
];

/** What the compute module stores with `ctx.setLegendData`. */
type LegendData = {
  tables?: LandformTables;
  ground?: TerrainGroundTone;
  /** Shares by geomorphon code, 0-1 (index 0 unused). */
  formShares?: readonly number[];
  /** Shares by scale code, 0-1. */
  scaleShares?: readonly number[];
  /** Shares of agree, disagree, opposite, 0-1. */
  agreementShares?: readonly number[];
  signedTable?: ClassTable;
  signedFor?: string;
};

function getLegends(state: LandformOptions, data: Readonly<Record<string, unknown>>): LegendSpec[] {
  const stored = data as LegendData;
  const groundTone = stored.ground ?? 'light';
  const tables = stored.tables ?? makeLandformTables(groundTone);
  const geomorphonTable = state.palette === 'grass' ? tables.grass : tables.quiet;
  if (showsGeomorphons(state)) {
    return [getLandformLegend(geomorphonTable, groundTone, {shares: stored.formShares})];
  }
  if (state.product === 'weiss') {
    return state.view === 'agreement'
      ? [getAgreementLegend(tables.agreement, stored.agreementShares)]
      : [getWeissKey(groundTone)];
  }
  if (state.product === 'position') {
    if (state.positionProduct === 'scale') {
      return [getScaleLegend(tables.scale, groundTone, stored.scaleShares)];
    }
    if (state.positionProduct === 'dev') {
      return [
        getSignedProductLegend(makeDeviationTable(groundTone), 'Height against the window (SD)', {
          high: 'above the window mean',
          low: 'below it'
        })
      ];
    }
  }
  const table = stored.signedFor === getSignedKey(state) ? stored.signedTable : undefined;
  if (!table) return [];
  if (state.product === 'curvature') {
    const kind = getCurvatureKind(state);
    const words =
      kind === 'profile'
        ? {high: 'convex: the slope steepens downhill', low: 'concave: the slope flattens'}
        : kind === 'plan'
          ? {high: 'convex: contours bow outward', low: 'concave: contours bow inward'}
          : {high: 'positive (convex)', low: 'negative (concave)'};
    return [getSignedProductLegend(table, `${getCurvatureName(kind)} (1/m)`, words)];
  }
  return [
    getSignedProductLegend(table, 'Height against the window mean (m)', {
      high: 'above the mean of its window',
      low: 'below it'
    })
  ];
}

const FRAME_HOME = {...TERRAIN_FRAMES.home, transitionMs: 1400};

/**
 * Landforms of the Matterhorn: ten forms from eight lines of sight, curvature, topographic position
 * from a summed-area table, and Weiss landforms, with the two classifiers compared. One question
 * runs through it: a class is a statement about a neighbourhood.
 */
export default defineScene<LandformOptions>({
  id: 'landforms',
  title: 'Ridges, valleys, hollows: landform classification',
  chapter: 'terrain',
  order: 3,
  summary:
    'Classify the Matterhorn and the Gorner glacier into ridges, spurs, hollows and valleys with geomorphons, curvature, topographic position and Weiss landforms, and see how far the answer depends on how far you look.',
  contributors: [
    'GPUGeomorphons',
    'GPUTerrainCurvature',
    'GPUTerrainTopographicPosition',
    'GPUTerrainWeissLandforms'
  ],
  datasets: [
    {id: 'alps-dem', role: 'elevation'},
    {id: 'alps-context', role: 'glaciers, peaks, places'}
  ],
  initialView: TERRAIN_FRAMES.home,
  basemap: ground('relief'),
  furniture: getTerrainFurniture({
    cartouche: terrainCartouche(
      'Ridge, hollow or valley?',
      'Landform classes · look-out radius and flatness threshold'
    )
  }),
  options,

  readouts: [
    {
      id: 'classShares',
      label: 'Share of cells by class',
      kind: 'chart',
      help: 'Share of the classified cells in each of the ten forms, ridge-like on the left, valley-like on the right, from a GPU readback after the map settles.'
    },
    {
      id: 'lookout',
      label: 'Look-out',
      help: 'How far each ray looks, in pixels and on the ground.'
    },
    {
      id: 'cells',
      label: 'Cells classified',
      hood: true,
      format: 'integer',
      help: 'Cells with a full window: those closer to the border than the look-out are not classified.'
    },
    {
      id: 'peakShare',
      label: 'Peak cells',
      emphasis: 'tile',
      format: 'percent',
      help: 'Share of the classified cells that are higher than everything their rays see.'
    },
    {
      id: 'sweepChart',
      label: 'Class groups against look-out radius',
      kind: 'chart',
      help: 'Share of cells in each group of forms at five radii. Press Sweep the look-out radius to fill it.'
    },
    {
      id: 'pattern',
      label: 'Rays of the pinned cell',
      emphasis: 'tile',
      help: 'The eight signs in GRASS order (NE, N, NW, W, SW, S, SE, E): + the ground rises, 0 level, - it falls.'
    },
    {
      id: 'readsPerCell',
      label: 'Cells read per cell',
      emphasis: 'tile',
      format: 'integer',
      help: 'Heights the eight rays read for one cell: up to the look-out radius along the straight rays, and the same ground distance along the diagonals.'
    },
    {
      id: 'breaks',
      label: 'Class breaks',
      help: 'The breaks of the frozen scale, as fixed shares of the 98th percentile of the absolute curvature.'
    },
    {
      id: 'curvatureHistogram',
      label: 'Curvature of all cells',
      kind: 'chart',
      help: 'Histogram of the displayed curvature with the class breaks; the tails beyond the outer breaks are clipped to the end bars.'
    },
    {
      id: 'p98',
      label: '98th percentile of |value|',
      hood: true,
      help: 'The magnitude the breaks scale with, measured once when the product is first shown and then frozen.'
    },
    {
      id: 'naiveReads',
      label: 'Window read directly',
      emphasis: 'tile',
      format: 'integer',
      unit: 'reads',
      help: 'Heights a direct mean over the window reads: the width of the window squared.'
    },
    {
      id: 'tableReads',
      label: 'From the table',
      emphasis: 'tile',
      format: 'integer',
      unit: 'reads',
      help: 'A summed-area table gives the sum of any window from its four corners.'
    },
    {
      id: 'timing',
      label: 'GPU time',
      hood: true,
      help: 'Time of the displayed graphs, measured once outside the frame.'
    },
    {
      id: 'agreeShare',
      label: 'Agree',
      emphasis: 'tile',
      format: 'percent',
      help: 'Share of the cells where Weiss and geomorphons give the same kind of landform (convex, neutral or concave).'
    },
    {
      id: 'oppositeShare',
      label: 'Opposite',
      emphasis: 'tile',
      format: 'percent',
      help: 'Share of the cells where one classifier says ridge-like and the other valley-like.'
    }
  ],

  pipeline: [
    {id: 'elevation', label: 'Elevation', detail: 'The 6.6 m DEM, one height per cell'},
    {
      id: 'rays',
      label: 'Rays',
      detail: 'Geomorphons: eight lines of sight per cell, zenith and nadir'
    },
    {id: 'fit', label: 'Fit', detail: 'Curvature: a polynomial fitted to a 3 x 3 or 5 x 5 window'},
    {
      id: 'table',
      label: 'Summed-area table',
      detail: 'Topographic position: any window mean from four table reads'
    },
    {id: 'classes', label: 'Classes', detail: 'Ten forms, signed classes or Weiss landforms'},
    {id: 'colorize', label: 'Colorize', detail: 'One class table colours the cells and the legend'}
  ],

  legends: getLegends,

  story: [
    {
      id: 'forms',
      title: 'Ten landforms from eight lines of sight',
      headline: 'Ridges and valleys are read from lines of sight',
      textAlternative:
        'Relief map of the Matterhorn and Zermatt valley with landform classes: ridges and peaks in red, hollows and valleys in blue, slopes left clear.',
      body: 'A cell is a ridge, a hollow or a valley only relative to what it sees. **`GPUGeomorphons`** sends eight rays out to **{{lookout}}** and records whether the ground ahead rises, stays level or falls; the pattern of signs is the landform. The legend gives each form its share. Switch **Palette** for the GRASS standard.\n\n*Colour the rare forms loudly, the common slope quietly.*',
      options: {
        product: 'geomorphons',
        palette: 'quiet',
        geomorphonRadius: 20,
        geomorphonFlatAngle: 1
      },
      optionsMode: 'fresh',
      controls: ['palette'],
      readouts: ['classShares', 'lookout', 'cells'],
      camera: FRAME_HOME,
      furniture: getTerrainFurniture({
        cartouche: terrainCartouche(
          'Ridge, hollow or valley?',
          'Geomorphons · look-out radius · flatness threshold'
        )
      }),
      stage: 'classes'
    },
    {
      id: 'lines-of-sight',
      title: 'Each cell looks out along eight rays',
      headline: "A cell's landform is the pattern of its rays",
      textAlternative:
        'Close view of the Matterhorn north-east ridge with eight lines of sight drawn from one pinned cell: red where the ground rises, grey where level, blue where it falls.',
      body: 'This cell sits on the crest near the Hörnli Hut. Its eight rays are red where the ground rises, grey where level and blue where it falls; dots mark the highest sight line, rings the lowest. The pattern **{{pattern}}** names one form, found with **{{readsPerCell}}** height reads, for every cell. Click to pin another, or change **Look-out radius**.\n\n*Eight signs, one name.*',
      options: {
        product: 'geomorphons',
        geomorphonRadius: 20,
        geomorphonFlatAngle: 1,
        showRays: true
      },
      optionsMode: 'fresh',
      controls: ['geomorphonRadius', 'geomorphonFlatAngle'],
      readouts: ['pattern', 'readsPerCell'],
      camera: {...TERRAIN_FRAMES.matterhorn, transitionMs: 1600},
      furniture: getTerrainFurniture({
        cartouche: terrainCartouche(
          'What does one cell see?',
          'Eight rays to the look-out radius · zenith and nadir'
        )
      }),
      stage: 'rays'
    },
    {
      id: 'scale',
      title: 'Look farther and the ridges merge',
      headline: 'Look farther and the ridges merge',
      textAlternative:
        'Matterhorn landform map split by a divider: a short look-out on the left shows many small peaks and ridges, the chosen look-out on the right fewer, larger ones.',
      body: 'Drag the divider: the left looks a short way, the right as far as **Look-out** says, and the same cell can change class. Peaks hold **{{peakShare}}** of cells now. **Sweep the look-out radius** plots every class group against the radius, each graph compiled once, then cached.\n\n*A landform is a statement about a neighbourhood.*',
      options: {
        product: 'geomorphons',
        geomorphonRadius: 30,
        geomorphonFlatAngle: 1,
        compareRadius: true
      },
      optionsMode: 'fresh',
      controls: ['radiusPreset', 'sweep'],
      readouts: ['sweepChart', 'peakShare'],
      camera: {longitude: 7.7, latitude: 45.98, zoom: 12.9, transitionMs: 1500},
      compare: {labels: ['Short look-out', 'Chosen look-out'], position: 0.5},
      furniture: getTerrainFurniture({
        cartouche: terrainCartouche(
          'How far should a cell look?',
          'Geomorphons at two look-out radii'
        )
      }),
      stage: 'rays'
    },
    {
      id: 'curvature',
      title: 'Curvature: where the slope bends',
      headline: 'Curvature marks every bend but amplifies noise',
      textAlternative:
        'Relief map coloured by profile curvature: orange where the slope steepens downhill, purple where it flattens, with the near-flat middle class left clear.',
      body: "**`GPUTerrainCurvature`** measures how the surface bends. **Bend** switches between profile curvature (along the slope: orange where it steepens downhill, purple where it flattens) and plan curvature (across it). Breaks are fixed shares of the 98th percentile, **{{p98}}**. A second derivative also amplifies the DEM's height steps: compare the **Fit** windows.\n\n*A signed measure gets a diverging scale, centred on zero.*",
      options: {product: 'curvature', curvatureKind: 'profile', curvatureMethod: 'evans-young'},
      optionsMode: 'fresh',
      controls: ['curvatureKind', 'curvatureMethod'],
      readouts: ['breaks', 'curvatureHistogram', 'p98'],
      camera: FRAME_HOME,
      furniture: getTerrainFurniture({
        cartouche: terrainCartouche(
          'Where does the slope bend?',
          'Curvature, 1/m · polynomial fit · 6.6 m cells'
        )
      }),
      stage: 'fit'
    },
    {
      id: 'position',
      title: 'One window mean, four table reads',
      headline: 'A summed-area table makes any window mean cheap',
      textAlternative:
        'Relief map coloured by topographic position at one scale: orange cells stand above the mean of their window, purple cells below it.',
      body: '**`GPUTerrainTopographicPosition`** asks whether a cell stands above or below the mean of its window. A summed-area table gives any window mean from **{{tableReads}}** instead of **{{naiveReads}}**, so every scale is cheap, and exact integer sums avoid float32 drift. Step **Scale** and hover for the window; **Measure** swaps TPI for DEV.\n\n*Size is a parameter, not a property of the ground.*',
      options: {
        product: 'position',
        positionProduct: 'tpi',
        scalePreset: 'landscape',
        scaleIndex: 3
      },
      optionsMode: 'fresh',
      controls: ['scaleIndex', 'positionProduct'],
      readouts: ['naiveReads', 'tableReads', 'timing'],
      camera: FRAME_HOME,
      furniture: getTerrainFurniture({
        cartouche: terrainCartouche(
          'Is a cell above its surroundings?',
          'Topographic position · window mean from a summed-area table'
        )
      }),
      stage: 'table'
    },
    {
      id: 'agreement',
      title: 'Where two classifiers disagree',
      headline: 'Two defensible classifiers do not always agree',
      textAlternative:
        'Relief map with the cells where Weiss landforms and geomorphons disagree drawn purple, and the cells where one says ridge and the other valley drawn orange.',
      body: "Weiss compares each cell with its surroundings at two sizes; geomorphons look along rays. Both are defensible, yet **{{agreeShare}}** of cells get the same kind of landform and **{{oppositeShare}}** get opposite ones (orange). Switch **Show**, or change **Classification** to explore. Global standardisation uses this tile's statistics, so another tile moves the classes.\n\n*A class belongs to the classifier, not the ground.*",
      options: {
        product: 'weiss',
        view: 'agreement',
        weissSmall: 4,
        weissLarge: 40,
        weissStandardization: 'global',
        geomorphonRadius: 20
      },
      optionsMode: 'fresh',
      controls: ['view', 'product', 'geomorphonRadius'],
      readouts: ['agreeShare', 'oppositeShare'],
      camera: FRAME_HOME,
      furniture: getTerrainFurniture({
        cartouche: terrainCartouche(
          'Do two classifiers agree?',
          'Weiss landforms against geomorphons · convex, neutral, concave'
        )
      }),
      stage: 'classes'
    }
  ],

  about: {
    what: '`GPUGeomorphons` classifies each cell into one of ten forms from eight lines of sight (GRASS `r.geomorphon`). `GPUTerrainCurvature` computes the Florinsky curvature kinds from a 3 x 3 or 5 x 5 polynomial fit. `GPUTerrainTopographicPosition` evaluates TPI, DEV and DEVmax at several scales from one exact summed-area table. `GPUTerrainWeissLandforms` classifies ten landforms from two TPI scales and slope.',
    why: 'Landform classes feed habitat and soil models, hazard mapping (rockfall source areas, avalanche starting zones) and the generalisation of terrain for maps. The neighbourhood is the main modelling choice, so seeing it change the answer is the point.',
    howToRead:
      'Geomorphons and Weiss are nominal classes: ridge-like forms are warm, valley-like forms cool, and the common slope is left nearly clear. Curvature, TPI and DEV are signed: orange is convex or above the surroundings, purple concave or below, and the middle class is not drawn. The data is a 6.6 m DEM of bare earth, so the smallest forms are rock texture. References: GRASS r.geomorphon (Jasiewicz and Stepinski 2013), WhiteboxTools (curvature signs), Weiss 2001 and Lindsay et al. 2015 (topographic position).'
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
  method: '${state.curvatureMethod}', curvatures: {profile, plan}, cellSizeMode: 'web-mercator'
}));
graph.add(new GPUTerrainTopographicPosition({
  width, height, elevation, scales: [${SCALE_PRESETS[state.scalePreset].map(radius => `{radius: ${radius}}`).join(', ')}],
  topographicPositionIndex, deviationFromMean, maximumDeviation, maximumDeviationRadius
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
