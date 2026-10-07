// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Pure (luma-free) pieces of the relief-visualization scene: option state, product table, the
 * Imhof recipe ladder, camera frames, legends and the two diagrams of the card. Shared by the
 * scene file and its compute module.
 */

import {getClassTableLegend, hexToRgba} from '../../cartography/class-table';
import {TERRAIN_EYE_GOLD} from '../../cartography/hue-registry';
import type {ClassTable} from '../../cartography/types';
import type {DiagramChartData} from '../chart-types';
import type {LegendSpec} from '../scene';
import {
  getCurvatureBreaks,
  getElevationLegend,
  makeSignedClassTable,
  type TerrainGroundTone
} from './terrain-palettes';

/** Every product the scene can display. */
export type ReliefProduct =
  | 'hillshade'
  | 'mdow'
  | 'swing'
  | 'texture'
  | 'sky-view'
  | 'anisotropic'
  | 'positive-openness'
  | 'negative-openness'
  | 'slrm'
  | 'msrm'
  | 'local-dominance'
  | 'imhof'
  | 'imhof-full'
  | 'vat'
  | 'ground';

/** Blend mode choice of a VAT layer; `preset` keeps the preset's mode. */
export type VatBlendChoice =
  | 'preset'
  | 'normal'
  | 'multiply'
  | 'screen'
  | 'overlay'
  | 'soft-light'
  | 'luminosity';

/** Option state of the relief-visualization scene. */
export type ReliefOptions = {
  product: ReliefProduct;
  /** Fixed left side of a swipe: `none` draws no reference. */
  reference: 'none' | 'single' | 'mdow';
  // Light
  lightAzimuth: number;
  lightAltitude: number;
  zFactor: number;
  // Imhof recipe
  recipe: number;
  imhofSwing: number;
  curvatureStrength: number;
  contrastStrength: number;
  contrastHighElevation: number;
  hillshadeStrength: number;
  skyViewStrength: number;
  textureStrength: number;
  exposure: number;
  tintStrength: number;
  // Texture shading
  textureLevels: number;
  textureBaseSigma: number;
  textureHasNodata: boolean;
  textureDownsample: boolean;
  textureDetail: number;
  textureGain: number;
  // Horizon, sky-view, openness
  horizonDirections: '8' | '16' | '32';
  horizonRadius: '32' | '64' | '96' | '192' | '384';
  horizonAlgorithm: 'march' | 'sweep';
  horizonGrowth: number;
  anisotropyAzimuth: number;
  anisotropyLevel: number;
  anisotropyMinimumWeight: number;
  // Local relief
  stretch: 'fixed' | 'fit';
  rvtExaggeration: number;
  slrmRadius: '6' | '13' | '20' | '40';
  msrmMinimumFeature: number;
  msrmMaximumFeature: number;
  msrmScaling: number;
  dominanceMinimumRadius: number;
  dominanceMaximumRadius: number;
  dominanceIncrement: number;
  dominanceAngle: '10' | '15' | '20' | '30';
  dominanceObserverHeight: number;
  // VAT blend
  vatPreset: 'archaeological' | 'flat' | 'hillshade-only';
  vatSlopeOpacity: number;
  vatOpennessOpacity: number;
  vatSkyViewOpacity: number;
  vatSlopeBlend: VatBlendChoice;
  vatOpennessBlend: VatBlendChoice;
  vatSkyViewBlend: VatBlendChoice;
  // Display
  clipPercent: number;
  opacity: number;
};

/**
 * Hypsometric tint of the first version of the Swiss relief, elevation meters to RGB (0-1).
 *
 * @deprecated Superseded by `ALPINE_TINTS` (`engine/relief.ts`), the chapter tint. Kept so
 * `sun-and-shadow` compiles until it moves over.
 */
export const ALPINE_ELEVATION_STOPS: readonly {
  elevation: number;
  color: [number, number, number];
}[] = [
  {elevation: 1500, color: [0.42, 0.56, 0.34]},
  {elevation: 2100, color: [0.66, 0.7, 0.42]},
  {elevation: 2700, color: [0.78, 0.7, 0.54]},
  {elevation: 3300, color: [0.84, 0.8, 0.76]},
  {elevation: 3800, color: [0.95, 0.95, 0.97]}
];

/** Ground metres per cell of `alps-dem` (`cellSizeGroundM` of its manifest). */
export const GROUND_CELL_METERS = 6.639;

/** Width of the square SLRM window of radius `radius` cells, in ground metres: `(2r + 1)` cells. */
export function getWindowMeters(radius: number, cellMeters = GROUND_CELL_METERS): number {
  return (2 * radius + 1) * cellMeters;
}

/** The camera frame the drawings of steps 1, 3 and 5 share, so they are compared on one landform. */
export const RELIEF_FRAME = {
  longitude: 7.735,
  latitude: 45.985,
  zoom: 12.3,
  pitch: 0,
  bearing: 0
} as const;

/** The Matterhorn north face (step 2), nudged east so the frame stays on the tile. */
export const NORTH_FACE_FRAME = {
  longitude: 7.672,
  latitude: 45.98,
  zoom: 13.2,
  pitch: 0,
  bearing: 0
} as const;

/** The Gorner moraines (step 4). */
export const MORAINE_FRAME = {
  longitude: 7.78,
  latitude: 45.968,
  zoom: 13.2,
  pitch: 0,
  bearing: 0
} as const;

/** The fixed light of the single-light reference: the cartographic convention. */
export const REFERENCE_LIGHT = {azimuthDegrees: 315, altitudeDegrees: 40} as const;

/** Share of the paper tone replaced by the hypsometric tint (the chapter ground uses the same). */
export const RECIPE_PALE_TINT = 0.3;

/** Lit and shaded colours of the Imhof recipe (the chapter relief ground). */
export const RECIPE_LIT_COLOR = '#FFF4DE';
export const RECIPE_SHADE_COLOR = '#5C6B8A';

/** The stages of the Imhof recipe the card steps through. */
export const RECIPE_STAGES: readonly {
  label: string;
  tint: boolean;
  warmCool: boolean;
  contrast: boolean;
  swing: boolean;
}[] = [
  {label: 'Grey hillshade', tint: false, warmCool: false, contrast: false, swing: false},
  {label: '+ Elevation tint', tint: true, warmCool: false, contrast: false, swing: false},
  {label: '+ Warm light, cool shade', tint: true, warmCool: true, contrast: false, swing: false},
  {label: '+ Contrast by height', tint: true, warmCool: true, contrast: true, swing: false},
  {label: '+ Imhof swing', tint: true, warmCool: true, contrast: true, swing: true}
];

/** Strength of the warm / cool aspect tint and of the contrast by height in the recipe. */
export const RECIPE_WARM_COOL_STRENGTH = 0.55;
export const RECIPE_CONTRAST = {from: 1800, to: 4200, strength: 0.6} as const;

/** Paint of the grey hillshade products, the same on both sides of a swipe. */
export const HILLSHADE_PAINT = {low: 0.05, high: 0.95} as const;

/** What the colorize pass does for a product. */
export type ProductScale =
  /** A physical range that never moves: honest between views. */
  | 'fixed'
  /** Percentile stretch of the displayed view. */
  | 'fit'
  /** Fixed class breaks (or fitted ones when `stretch` is `fit`). */
  | 'classes'
  /** Packed colours drawn as they are. */
  | 'picture';

/** A product's label, unit, scale kind and the nominal range of the fixed or fallback scale. */
export type ProductSpec = {
  label: string;
  scale: ProductScale;
  low: number;
  high: number;
  unit: string;
  /** The two ends of the grey legend, low then high. */
  ends: readonly [string, string];
};

/** Every product, in the order of the explore list. */
export const PRODUCTS: Record<ReliefProduct, ProductSpec> = {
  hillshade: {
    label: 'Hillshade: one light',
    scale: 'fixed',
    low: 0.05,
    high: 0.95,
    unit: '',
    ends: ['shaded', 'lit']
  },
  mdow: {
    label: 'Multidirectional hillshade',
    scale: 'fixed',
    low: 0.05,
    high: 0.95,
    unit: '',
    ends: ['shaded', 'lit']
  },
  swing: {
    label: 'Hillshade: Imhof swing',
    scale: 'fixed',
    low: 0.05,
    high: 0.95,
    unit: '',
    ends: ['shaded', 'lit']
  },
  texture: {
    label: 'Texture shading',
    scale: 'fit',
    low: -0.6,
    high: 0.6,
    unit: '',
    ends: ['groove', 'bump']
  },
  'sky-view': {
    label: 'Sky-view factor',
    scale: 'fixed',
    low: 0.35,
    high: 1,
    unit: 'share of sky',
    ends: ['enclosed', 'open sky']
  },
  anisotropic: {
    label: 'Anisotropic sky-view factor',
    scale: 'fixed',
    low: 0.35,
    high: 1,
    unit: 'share of sky',
    ends: ['enclosed', 'open sky']
  },
  'positive-openness': {
    label: 'Positive openness',
    scale: 'fixed',
    low: 60,
    high: 98,
    unit: '°',
    ends: ['enclosed', 'exposed']
  },
  'negative-openness': {
    label: 'Negative openness',
    scale: 'fixed',
    low: 60,
    high: 98,
    unit: '°',
    ends: ['convex', 'hollow']
  },
  slrm: {
    label: 'Simple local relief',
    scale: 'classes',
    low: -15,
    high: 15,
    unit: 'm',
    ends: ['below', 'above']
  },
  msrm: {
    label: 'Multi-scale relief',
    scale: 'classes',
    low: -20,
    high: 20,
    unit: 'm',
    ends: ['below', 'above']
  },
  'local-dominance': {
    label: 'Local dominance',
    scale: 'fit',
    low: 0.3,
    high: 1.8,
    unit: '°',
    ends: ['low', 'dominant']
  },
  imhof: {
    label: 'Imhof recipe',
    scale: 'picture',
    low: 0,
    high: 1,
    unit: '',
    ends: ['shaded', 'lit']
  },
  'imhof-full': {
    label: 'Swiss relief with sky-view and texture',
    scale: 'picture',
    low: 0,
    high: 1,
    unit: '',
    ends: ['shaded', 'lit']
  },
  vat: {
    label: 'VAT blend',
    scale: 'picture',
    low: 0,
    high: 1,
    unit: '',
    ends: ['shaded', 'lit']
  },
  ground: {
    label: 'Chapter relief ground',
    scale: 'picture',
    low: 0,
    high: 1,
    unit: '',
    ends: ['shaded', 'lit']
  }
};

/** The products drawn on class breaks (signed height against a window mean). */
export const CLASSED_PRODUCTS: ReadonlySet<ReliefProduct> = new Set(['slrm', 'msrm']);

/** The hillshade-like products: a grey brightness, not a measured quantity. */
export const HILLSHADE_PRODUCTS: ReadonlySet<ReliefProduct> = new Set([
  'hillshade',
  'mdow',
  'swing'
]);

/** The 16-point compass name of an azimuth in degrees clockwise from north. */
export function getCompassName(azimuthDegrees: number): string {
  const names = [
    'north',
    'north-north-east',
    'north-east',
    'east-north-east',
    'east',
    'east-south-east',
    'south-east',
    'south-south-east',
    'south',
    'south-south-west',
    'south-west',
    'west-south-west',
    'west',
    'west-north-west',
    'north-west',
    'north-north-west'
  ];
  const index = Math.round((((azimuthDegrees % 360) + 360) % 360) / 22.5) % 16;
  return names[index];
}

/** `North-west` from `north-west` for a title. */
function capitalise(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/** Formats a metre value with a thousands separator and a unit: `1,275 m`. */
export function formatMeters(value: number): string {
  return `${Math.round(value).toLocaleString('en-US')} m`;
}

/** Rounds a class limit to a tenth of a metre, never below 0.1. */
export function roundBreakLimit(value: number): number {
  return Math.max(0.1, Math.round(value * 10) / 10);
}

/**
 * The seven classes of the local relief maps: PuOr, purple below the surroundings and orange
 * above, the middle class transparent, limits at 0.25, 0.6 and 1 of `limitMeters`.
 */
export function makeLocalReliefTable(limitMeters: number, ground: TerrainGroundTone): ClassTable {
  const breaks = getCurvatureBreaks(roundBreakLimit(limitMeters));
  return makeSignedClassTable(breaks, ground, 'm', {
    alpha: 204,
    method: 'Purple below the window mean, orange above; the middle class is not drawn',
    format: value => (Math.abs(value) < 10 ? value.toFixed(1) : value.toFixed(0))
  });
}

/** Cells per class of a table from a histogram over `[minimum, maximum]` (bin accuracy). */
export function getClassCounts(
  table: ClassTable,
  histogram: ArrayLike<number>,
  minimum: number,
  maximum: number
): number[] {
  const counts = new Array<number>(table.breaks.length + 1).fill(0);
  const binCount = histogram.length;
  for (let bin = 0; bin < binCount; bin++) {
    const center = minimum + ((bin + 0.5) / binCount) * (maximum - minimum);
    let index = 0;
    while (index < table.breaks.length && center >= table.breaks[index]) index++;
    counts[index] += histogram[bin];
  }
  return counts;
}

// ---------------------------------------------------------------------------------------------
// Legends
// ---------------------------------------------------------------------------------------------

const LIT_SWATCH = [236, 236, 236] as const;
const SHADED_SWATCH = [52, 52, 52] as const;

/** Legend entries `lit slope` and `shaded slope` (a hillshade is a brightness, not a quantity). */
function getLightLegend(title: string, note: string): LegendSpec {
  return {
    kind: 'categories',
    title,
    entries: [
      {color: LIT_SWATCH, label: 'Lit slope'},
      {color: SHADED_SWATCH, label: 'Shaded slope'}
    ],
    note
  };
}

/** The note under a hillshade legend: the light as a number and a compass name. */
export function getLightNote(state: ReliefOptions, product: ReliefProduct): string {
  if (product === 'mdow') {
    return 'Four lights from the west-south-west to the north, each weighted by how squarely it meets the slope';
  }
  const {lightAzimuth: azimuth, lightAltitude: altitude} = state;
  const direction = `${azimuth}° (${getCompassName(azimuth)}) at ${altitude}°`;
  return product === 'swing'
    ? `One light from ${direction}, swung toward each slope by up to ${state.imhofSwing}°`
    : `Light from ${direction}`;
}

/** The left (reference) side of a swipe, as a legend. */
function getReferenceLegend(state: ReliefOptions): LegendSpec {
  return state.reference === 'single'
    ? getLightLegend(
        'Left side: one light',
        `Light from ${REFERENCE_LIGHT.azimuthDegrees}° (${getCompassName(REFERENCE_LIGHT.azimuthDegrees)}) at ${REFERENCE_LIGHT.altitudeDegrees}°`
      )
    : getLightLegend('Left side: multidirectional hillshade', getLightNote(state, 'mdow'));
}

/** The lit / shade chips of the recipe. */
function getWarmCoolLegend(): LegendSpec {
  return {
    kind: 'categories',
    title: 'Light and shade',
    entries: [
      {color: hexToRgba(RECIPE_LIT_COLOR), label: 'Lit slope, warm'},
      {color: hexToRgba(RECIPE_SHADE_COLOR), label: 'Shaded slope, cool'}
    ]
  };
}

/** The legend of the Imhof recipe at a stage: only the cues that are switched on. */
function getRecipeLegends(stage: number, ground: TerrainGroundTone): LegendSpec[] {
  const spec = RECIPE_STAGES[Math.min(Math.max(stage, 0), RECIPE_STAGES.length - 1)];
  const legends: LegendSpec[] = [];
  if (spec.tint) {
    legends.push(
      getElevationLegend(ground, {
        note: 'Pale on purpose: green does not mean forest. Glaciers are painted on the chapter ground only'
      })
    );
  } else {
    legends.push(
      getLightLegend('Right side: grey hillshade', 'One light, no tint: form only, no height')
    );
  }
  if (spec.warmCool) legends.push(getWarmCoolLegend());
  return legends;
}

/** Everything the legend of the current state may need besides the option state. */
export type ReliefLegendData = {
  ground?: TerrainGroundTone;
  /** Local relief table in force (frozen or fitted). */
  table?: ClassTable;
  /** Cells per class of that table. */
  counts?: readonly number[];
  /** Why the table is what it is. */
  tableNote?: string;
};

/** The legends of the current state: the fixed left side of a swipe first, then the right side. */
export function getReliefLegends(
  state: ReliefOptions,
  data: Readonly<Record<string, unknown>>
): LegendSpec[] {
  const info = data as ReliefLegendData;
  const ground = info.ground ?? 'light';
  const legends: LegendSpec[] = [];
  if (state.reference !== 'none') legends.push(getReferenceLegend(state));
  const {product} = state;
  const spec = PRODUCTS[product];
  if (HILLSHADE_PRODUCTS.has(product)) {
    legends.push(getLightLegend(spec.label, getLightNote(state, product)));
  } else if (product === 'texture' || product === 'local-dominance') {
    legends.push({
      kind: 'ramp',
      id: 'stretch',
      title: spec.label,
      ramp: 'grayscale',
      extent: 'gpu',
      labels: spec.ends,
      note: `Stretched between the ${state.clipPercent}th and ${100 - state.clipPercent}th percentile of this view`,
      format: value => (Math.abs(value) >= 10 ? value.toFixed(0) : value.toPrecision(2))
    });
  } else if (spec.scale === 'fixed') {
    legends.push({
      kind: 'ramp',
      title: spec.label,
      ramp: 'grayscale',
      extent: [spec.low, spec.high],
      unit: spec.unit,
      labels: spec.ends,
      note: 'Fixed range: the same grey means the same value in every view'
    });
  } else if (spec.scale === 'classes') {
    const table = info.table ?? makeLocalReliefTable(4, ground);
    legends.push(
      getClassTableLegend(table, {
        title: `${spec.label}, height against the window mean (m)`,
        counts: info.counts,
        layout: info.counts ? 'list' : 'bar',
        note: info.tableNote ?? table.method
      })
    );
  } else if (product === 'imhof') {
    legends.push(...getRecipeLegends(state.recipe, ground));
  } else if (product === 'imhof-full') {
    legends.push(...getRecipeLegends(RECIPE_STAGES.length - 1, ground));
  } else if (product === 'ground') {
    legends.push(getElevationLegend(ground), getWarmCoolLegend());
  } else if (product === 'vat') {
    legends.push(
      getLightLegend(
        'VAT blend',
        'Four stacked drawings (see the diagram): a picture to read form from, not a measurement'
      )
    );
  }
  return legends;
}

// ---------------------------------------------------------------------------------------------
// Diagrams
// ---------------------------------------------------------------------------------------------

const DIAL_CENTER = {x: 66, y: 68};
const DIAL_RADIUS = 48;

/** A point on the compass dial `fraction` of the radius out, at an azimuth clockwise from north. */
function getDialPoint(azimuthDegrees: number, fraction: number): [number, number] {
  const radians = (azimuthDegrees * Math.PI) / 180;
  return [
    DIAL_CENTER.x + Math.sin(radians) * DIAL_RADIUS * fraction,
    DIAL_CENTER.y - Math.cos(radians) * DIAL_RADIUS * fraction
  ];
}

/**
 * The light dial of step 1: a compass with the sun glyph at the azimuth and an arrow for the
 * direction the light travels, and a side view with the sun at its altitude above the horizon.
 */
export function makeLightDialDiagram(
  azimuthDegrees: number,
  altitudeDegrees: number
): DiagramChartData {
  const [sunX, sunY] = getDialPoint(azimuthDegrees, 1);
  const [tailX, tailY] = getDialPoint(azimuthDegrees, 0.72);
  const [headX, headY] = getDialPoint(azimuthDegrees, 0.2);
  const rays = Array.from({length: 8}, (_, index) => {
    const angle = (index * Math.PI) / 4;
    const x1 = sunX + Math.cos(angle) * 8;
    const y1 = sunY + Math.sin(angle) * 8;
    const x2 = sunX + Math.cos(angle) * 12;
    const y2 = sunY + Math.sin(angle) * 12;
    return `<line class="diagram-signal" x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke-width="1.5"/>`;
  }).join('');
  const ticks = [0, 90, 180, 270]
    .map(angle => {
      const [x1, y1] = getDialPoint(angle, 1);
      const [x2, y2] = getDialPoint(angle, 0.86);
      return `<line class="diagram-muted" x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}"/>`;
    })
    .join('');
  const labels = [
    ['N', 0],
    ['E', 90],
    ['S', 180],
    ['W', 270]
  ]
    .map(([name, angle]) => {
      const [x, y] = getDialPoint(angle as number, 1.3);
      return `<text class="diagram-muted" x="${x.toFixed(1)}" y="${(y + 4).toFixed(1)}" text-anchor="middle">${name}</text>`;
    })
    .join('');
  // Side view: the horizon line, and the sun on a quarter circle at its altitude.
  const originX = 168;
  const originY = 104;
  const sideRadius = 70;
  const altitude = (Math.min(Math.max(altitudeDegrees, 0), 90) * Math.PI) / 180;
  const sideSunX = originX + Math.cos(altitude) * sideRadius;
  const sideSunY = originY - Math.sin(altitude) * sideRadius;
  const description = `Light dial: the sun is ${azimuthDegrees} degrees clockwise from north (${getCompassName(azimuthDegrees)}) and ${altitudeDegrees} degrees above the horizon.`;
  const svg = `
<circle class="diagram-muted" cx="${DIAL_CENTER.x}" cy="${DIAL_CENTER.y}" r="${DIAL_RADIUS}"/>
${ticks}${labels}
<line class="diagram-ink" x1="${tailX.toFixed(1)}" y1="${tailY.toFixed(1)}" x2="${headX.toFixed(1)}" y2="${headY.toFixed(1)}"/>
<circle class="diagram-signal" cx="${sunX.toFixed(1)}" cy="${sunY.toFixed(1)}" r="5.5" fill="${TERRAIN_EYE_GOLD}"/>
${rays}
<text class="diagram-ink" x="${DIAL_CENTER.x}" y="${DIAL_CENTER.y + DIAL_RADIUS + 22}" text-anchor="middle">${azimuthDegrees}° ${capitalise(getCompassName(azimuthDegrees))}</text>
<line class="diagram-muted" x1="${originX - 6}" y1="${originY}" x2="${originX + sideRadius + 14}" y2="${originY}"/>
<path class="diagram-muted" d="M ${originX + sideRadius} ${originY} A ${sideRadius} ${sideRadius} 0 0 0 ${(originX + Math.cos(Math.PI / 2) * sideRadius).toFixed(1)} ${(originY - sideRadius).toFixed(1)}" stroke-dasharray="3 3"/>
<line class="diagram-ink" x1="${originX}" y1="${originY}" x2="${sideSunX.toFixed(1)}" y2="${sideSunY.toFixed(1)}" stroke-dasharray="2 3"/>
<circle class="diagram-signal" cx="${sideSunX.toFixed(1)}" cy="${sideSunY.toFixed(1)}" r="5.5" fill="${TERRAIN_EYE_GOLD}"/>
<text class="diagram-ink" x="${originX}" y="${originY + 22}" text-anchor="start">${altitudeDegrees}° high</text>`;
  return {kind: 'diagram', svg, width: 270, height: 150, description};
}

/**
 * The exploded layer stack of step 6: four sheets bottom to top with their blend modes, as
 * `GPUReliefBlend` stacks them in the VAT recipe.
 */
export function makeStackDiagram(): DiagramChartData {
  const layers = [
    {name: 'Hillshade', mode: 'normal'},
    {name: 'Slope, inverted', mode: 'luminosity'},
    {name: 'Positive openness', mode: 'overlay'},
    {name: 'Sky-view factor', mode: 'multiply'}
  ];
  const sheets = layers
    .map((layer, index) => {
      const y = 112 - index * 28;
      const x = 20 + index * 6;
      return `
<path class="diagram-ink" d="M ${x} ${y} l 60 -14 l 80 0 l -60 14 z"/>
<path class="diagram-fill" d="M ${x} ${y} l 60 -14 l 80 0 l -60 14 z"/>
<text class="diagram-ink" x="${x + 156}" y="${y - 4}" text-anchor="start">${layer.name}</text>
<text class="diagram-muted" x="${x + 156}" y="${y + 9}" text-anchor="start">${layer.mode}</text>`;
    })
    .join('');
  return {
    kind: 'diagram',
    svg: sheets,
    width: 320,
    height: 136,
    description:
      'Exploded stack of four sheets, bottom to top: hillshade drawn normally, inverted slope blended by luminosity, positive openness by overlay and sky-view factor by multiply.'
  };
}
