// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ALPS, placeToAnnotation} from '../../cartography/gazetteer';
import {ground} from '../../cartography/grounds';
import type {MapAnnotation} from '../../cartography/types';
import {defineScene, type OptionSpec, type ReadoutSpec} from '../scene';
import {
  getCompassName,
  getReliefLegends,
  getWindowMeters,
  GROUND_CELL_METERS,
  formatMeters,
  makeStackDiagram,
  MORAINE_FRAME,
  NORTH_FACE_FRAME,
  RECIPE_STAGES,
  RELIEF_FRAME,
  type ReliefOptions,
  type VatBlendChoice
} from './relief-visualization.style';
import {terrainCartouche, TERRAIN_CREDIT} from './terrain-furniture';

const BLEND_CHOICES: {value: VatBlendChoice; label: string}[] = [
  {value: 'preset', label: 'Use the preset'},
  {value: 'normal', label: 'Normal'},
  {value: 'multiply', label: 'Multiply'},
  {value: 'screen', label: 'Screen'},
  {value: 'overlay', label: 'Overlay'},
  {value: 'soft-light', label: 'Soft light'},
  {value: 'luminosity', label: 'Luminosity'}
];

const HORIZON_PRODUCTS = ['sky-view', 'anisotropic', 'positive-openness', 'negative-openness'];

const isProduct =
  (...products: ReliefOptions['product'][]) =>
  (state: ReliefOptions) =>
    !products.includes(state.product);

/** A search radius option label: pixels and the ground distance they cover. */
const getRadiusLabel = (pixels: number) =>
  `${pixels} px, ${formatMeters(pixels * GROUND_CELL_METERS)}`;

/** A window radius option label: pixels and the width of the whole square window. */
const getWindowLabel = (radius: number) => `${radius} px, ${formatMeters(getWindowMeters(radius))}`;

/** The gazetteer label of a place, with its elevation as detail for towns and huts. */
function placeLabel(id: string): MapAnnotation {
  const place = ALPS.places[id];
  const annotation = placeToAnnotation(place);
  return place.elevationM !== undefined && annotation.kind === 'point'
    ? {...annotation, detail: formatMeters(place.elevationM)}
    : annotation;
}

/** The title cartouche of a step: the question, and the variable line (refined live by the scene). */
const cartouche = (question: string, subtitle: string) =>
  terrainCartouche(question, subtitle, undefined, undefined);

const OPTIONS: readonly OptionSpec<ReliefOptions>[] = [
  // --- The reader's controls -----------------------------------------------------------------------
  {
    kind: 'slider',
    id: 'lightAzimuth',
    label: 'Light from',
    group: 'Light',
    apply: 'param',
    min: 0,
    max: 360,
    step: 5,
    default: 315,
    unit: '°',
    marks: [
      {value: 315, label: 'convention'},
      {value: 135, label: 'inverts'}
    ],
    describe: value => `light from the ${getCompassName(value)}`,
    help: 'Direction the light comes from, clockwise from north. Used by the single-light hillshade, the Imhof swing and the VAT hillshade layer.'
  },
  {
    kind: 'preset',
    id: 'lightPreset',
    label: 'Light preset',
    group: 'Light',
    presets: [
      {label: 'North-west (convention)', values: {lightAzimuth: 315}},
      {label: 'South-east (inverts)', values: {lightAzimuth: 135}}
    ],
    help: 'Writes the light direction. North-west light is the cartographic convention; from the south-east many eyes read hills as hollows.'
  },
  {
    kind: 'slider',
    id: 'lightAltitude',
    label: 'Light altitude',
    group: 'Light',
    apply: 'param',
    min: 5,
    max: 85,
    step: 1,
    default: 40,
    unit: '°',
    help: 'Height of the light above the horizon. Low light exaggerates texture; high light flattens it.'
  },
  {
    kind: 'preset',
    id: 'lightModelPreset',
    label: 'Drawing',
    group: 'Light',
    presets: [
      {label: 'Single', values: {product: 'hillshade'}},
      {label: 'Multidirectional', values: {product: 'mdow'}},
      {label: 'Imhof swing', values: {product: 'swing'}}
    ],
    help: 'The drawing on the right of the divider: one fixed light, the USGS multidirectional weighting of several lights, or one light swung toward each slope.'
  },
  {
    kind: 'preset',
    id: 'lightFreePreset',
    label: 'Light-free drawing',
    group: 'Horizon',
    presets: [
      {label: 'Texture shading', values: {product: 'texture'}},
      {label: 'Sky-view', values: {product: 'sky-view'}},
      {label: 'Openness', values: {product: 'positive-openness'}}
    ],
    help: 'The drawing on the right of the divider. None of them uses a light.'
  },
  {
    kind: 'select',
    id: 'horizonRadius',
    label: 'Search radius',
    group: 'Horizon',
    apply: 'compile',
    default: '96',
    disabledWhen: isProduct(...(HORIZON_PRODUCTS as ReliefOptions['product'][]), 'vat'),
    help: 'How far each ray looks for a blocking horizon, in cells and ground metres. A bigger radius sees the far valley walls and costs more.',
    options: [
      {value: '32', label: getRadiusLabel(32)},
      {value: '64', label: getRadiusLabel(64)},
      {value: '96', label: getRadiusLabel(96)},
      {value: '192', label: getRadiusLabel(192)},
      {value: '384', label: getRadiusLabel(384)}
    ]
  },
  {
    kind: 'select',
    id: 'slrmRadius',
    label: 'Window',
    group: 'Local relief',
    apply: 'compile',
    default: '13',
    display: 'segmented',
    help: 'Half width of the square window whose mean height is subtracted from each cell. The label gives the whole window on the ground.',
    options: [
      {value: '6', label: getWindowLabel(6)},
      {value: '13', label: getWindowLabel(13)},
      {value: '20', label: getWindowLabel(20)},
      {value: '40', label: getWindowLabel(40)}
    ]
  },
  {
    kind: 'select',
    id: 'stretch',
    label: 'Class limits',
    group: 'Local relief',
    apply: 'param',
    default: 'fixed',
    display: 'segmented',
    disabledWhen: isProduct('slrm'),
    help: 'Fixed breaks use one set of limits for every window, so windows can be compared. Fit to view re-stretches each window to its own range, which hides how much stronger a large window is.',
    options: [
      {value: 'fixed', label: 'Fixed breaks'},
      {value: 'fit', label: 'Fit to view'}
    ]
  },
  {
    kind: 'slider',
    id: 'recipe',
    label: 'Recipe',
    group: 'Swiss relief',
    apply: 'param',
    min: 0,
    max: RECIPE_STAGES.length - 1,
    step: 1,
    default: RECIPE_STAGES.length - 1,
    display: 'stepper',
    format: value => RECIPE_STAGES[Math.round(value)]?.label ?? String(value),
    disabledWhen: isProduct('imhof'),
    help: 'Switches one cue of the Swiss relief on at a time: a pale elevation tint, warm light with cool shade, contrast that grows with height, then the Imhof swing. Each stage is a parameter write.'
  },
  {
    kind: 'select',
    id: 'product',
    label: 'Drawing',
    group: 'Explore',
    apply: 'param',
    default: 'hillshade',
    help: 'Each drawing is a stage of compiled graphs, built the first time you pick it and kept. The Swiss relief with sky-view and texture reuses the horizon, texture and curvature stages.',
    options: [
      {value: 'hillshade', label: 'Hillshade: one light'},
      {value: 'mdow', label: 'Multidirectional hillshade (USGS MDOW)'},
      {value: 'swing', label: 'Hillshade: Imhof swing'},
      {value: 'texture', label: 'Texture shading (Leland Brown)'},
      {value: 'sky-view', label: 'Sky-view factor'},
      {value: 'anisotropic', label: 'Anisotropic sky-view factor'},
      {value: 'positive-openness', label: 'Positive openness'},
      {value: 'negative-openness', label: 'Negative openness'},
      {value: 'slrm', label: 'Simple local relief (SLRM)'},
      {value: 'msrm', label: 'Multi-scale relief (MSRM)'},
      {value: 'local-dominance', label: 'Local dominance'},
      {value: 'imhof', label: 'Imhof recipe (tint, warm light, haze)'},
      {value: 'imhof-full', label: 'Swiss relief with sky-view, texture, curvature'},
      {value: 'vat', label: 'VAT blend'},
      {value: 'ground', label: 'Chapter relief ground (CPU build)'}
    ]
  },
  {
    kind: 'slider',
    id: 'zFactor',
    label: 'Vertical exaggeration',
    group: 'Explore',
    apply: 'param',
    min: 0.5,
    max: 4,
    step: 0.1,
    default: 1,
    unit: 'x',
    help: 'Multiplies heights for the shading products (hillshade, sky-view, openness, VAT slope). It changes what is drawn and what is computed.'
  },
  {
    kind: 'select',
    id: 'vatPreset',
    label: 'VAT preset',
    group: 'Explore',
    apply: 'param',
    default: 'archaeological',
    disabledWhen: isProduct('vat'),
    help: 'Layer stretches of the RVT "Archaeological" and "Flat terrain" combinations. Hillshade only zeroes the other layers.',
    options: [
      {value: 'archaeological', label: 'Archaeological'},
      {value: 'flat', label: 'Flat terrain'},
      {value: 'hillshade-only', label: 'Hillshade only'}
    ]
  },

  // --- Reference side of a swipe (set by the steps) ------------------------------------------------
  {
    kind: 'select',
    id: 'reference',
    label: 'Fixed left side',
    group: 'Compare',
    apply: 'param',
    default: 'none',
    expert: true,
    help: 'The drawing held on the left of a swipe: one light at the convention, or the multidirectional hillshade. The steps set it; none draws no left side.',
    options: [
      {value: 'none', label: 'None'},
      {value: 'single', label: 'One light, convention'},
      {value: 'mdow', label: 'Multidirectional'}
    ]
  },

  // --- Imhof recipe and Swiss relief ---------------------------------------------------------------
  {
    kind: 'slider',
    id: 'imhofSwing',
    label: 'Imhof swing',
    group: 'Swiss relief',
    apply: 'param',
    min: 0,
    max: 90,
    step: 5,
    default: 65,
    unit: '°',
    expert: true,
    help: 'Largest angle the light is swung toward the side a slope faces. 0 is a fixed light.'
  },
  {
    kind: 'slider',
    id: 'curvatureStrength',
    label: 'Curvature strength',
    group: 'Swiss relief',
    apply: 'param',
    min: 0,
    max: 1.5,
    step: 0.05,
    default: 0.5,
    expert: true,
    disabledWhen: isProduct('imhof-full'),
    help: 'Adds the multi-radius ring curvature (convex bright, concave dark) to the full Swiss relief.'
  },
  {
    kind: 'slider',
    id: 'hillshadeStrength',
    label: 'Hillshade strength',
    group: 'Swiss relief',
    apply: 'param',
    min: 0,
    max: 1.5,
    step: 0.05,
    default: 1,
    expert: true,
    disabledWhen: isProduct('imhof-full'),
    help: 'Weight of the hillshade term in the full Swiss relief.'
  },
  {
    kind: 'slider',
    id: 'skyViewStrength',
    label: 'Sky-view strength',
    group: 'Swiss relief',
    apply: 'param',
    min: 0,
    max: 1.5,
    step: 0.05,
    default: 1,
    expert: true,
    disabledWhen: isProduct('imhof-full'),
    help: 'Weight of the sky-view factor in the full Swiss relief: valleys and cirques darken as less sky is visible.'
  },
  {
    kind: 'slider',
    id: 'textureStrength',
    label: 'Texture-shade strength',
    group: 'Swiss relief',
    apply: 'param',
    min: 0,
    max: 1.5,
    step: 0.05,
    default: 0.4,
    expert: true,
    disabledWhen: isProduct('imhof-full'),
    help: 'Weight of the texture shading term in the full Swiss relief.'
  },
  {
    kind: 'slider',
    id: 'contrastStrength',
    label: 'Contrast with elevation',
    group: 'Swiss relief',
    apply: 'param',
    min: 0,
    max: 1,
    step: 0.05,
    default: 0.5,
    expert: true,
    disabledWhen: isProduct('imhof-full'),
    help: 'Swiss cartographers raise relief contrast with height so high rock stands out. 0 turns it off. The recipe stages use their own fixed strength.'
  },
  {
    kind: 'slider',
    id: 'contrastHighElevation',
    label: 'Full contrast at',
    group: 'Swiss relief',
    apply: 'param',
    min: 2500,
    max: 4500,
    step: 50,
    default: 3800,
    unit: 'm',
    expert: true,
    disabledWhen: isProduct('imhof-full'),
    help: 'Elevation where the contrast boost of the full Swiss relief reaches its full strength.'
  },
  {
    kind: 'slider',
    id: 'exposure',
    label: 'Exposure',
    group: 'Swiss relief',
    apply: 'param',
    min: 0.6,
    max: 1.8,
    step: 0.05,
    default: 1.1,
    expert: true,
    disabledWhen: isProduct('imhof', 'imhof-full'),
    help: 'Overall brightness of the Swiss relief.'
  },
  {
    kind: 'slider',
    id: 'tintStrength',
    label: 'Warm / cool aspect tint',
    group: 'Swiss relief',
    apply: 'param',
    min: 0,
    max: 1,
    step: 0.05,
    default: 0.3,
    expert: true,
    disabledWhen: isProduct('imhof-full'),
    help: 'Tints lit slopes warm and shaded slopes cool in the full Swiss relief.'
  },

  // --- Texture shading ----------------------------------------------------------------------------
  {
    kind: 'slider',
    id: 'textureLevels',
    label: 'Cascade levels',
    group: 'Texture shading',
    apply: 'compile',
    min: 1,
    max: 8,
    step: 1,
    default: 6,
    expert: true,
    help: 'Number of Gaussian band-pass levels, each twice as wide as the last. More levels reach larger landforms. Rebuilds the texture graph.'
  },
  {
    kind: 'slider',
    id: 'textureBaseSigma',
    label: 'Finest level sigma',
    group: 'Texture shading',
    apply: 'compile',
    min: 0.5,
    max: 4,
    step: 0.5,
    default: 1,
    unit: 'px',
    expert: true,
    help: 'Blur width of the finest level in cells; the other levels are 2x, 4x, 8x that.'
  },
  {
    kind: 'slider',
    id: 'textureDetail',
    label: 'Detail (alpha)',
    group: 'Texture shading',
    apply: 'param',
    min: 0,
    max: 1,
    step: 0.05,
    default: 0.5,
    expert: true,
    help: 'The exponent of the fractional Laplacian: 0 weights all scales equally, 1 favors fine texture. A parameter write.'
  },
  {
    kind: 'slider',
    id: 'textureGain',
    label: 'Gain',
    group: 'Texture shading',
    apply: 'param',
    min: 0.001,
    max: 0.05,
    step: 0.001,
    default: 0.01,
    expert: true,
    help: 'Multiplier from elevation metres to the texture-shade value. Only matters in the full Swiss relief.'
  },
  {
    kind: 'toggle',
    id: 'textureHasNodata',
    label: 'Handle nodata',
    group: 'Texture shading',
    apply: 'compile',
    default: true,
    expert: true,
    help: 'Turn off when the DEM has no nodata (this one does not): the blur then needs one channel instead of two, about half the cost, with identical output.'
  },
  {
    kind: 'toggle',
    id: 'textureDownsample',
    label: 'Downsample wide levels',
    group: 'Texture shading',
    apply: 'compile',
    default: true,
    expert: true,
    help: 'Computes wide levels on a half-resolution grid and upsamples them. Output stays within 1 % of the exact result and runs much faster.'
  },

  // --- Horizon ------------------------------------------------------------------------------------
  {
    kind: 'select',
    id: 'horizonDirections',
    label: 'Azimuth sectors',
    group: 'Horizon',
    apply: 'compile',
    default: '16',
    expert: true,
    help: 'How many compass directions the horizon is sampled in. More sectors give smoother sky-view and openness at proportional cost.',
    options: [
      {value: '8', label: '8 sectors'},
      {value: '16', label: '16 sectors'},
      {value: '32', label: '32 sectors'}
    ]
  },
  {
    kind: 'select',
    id: 'horizonAlgorithm',
    label: 'Algorithm',
    group: 'Horizon',
    apply: 'compile',
    default: 'march',
    expert: true,
    help: 'March samples each ray at growing steps (cheap at short radii). Sweep is an exact upper-hull sweep (Stewart 1998): fixed cost per cell, so it wins at radii near the tile size and loses at short ones.',
    options: [
      {value: 'march', label: 'March: bilinear ray marching'},
      {value: 'sweep', label: 'Sweep: exact hull sweep'}
    ]
  },
  {
    kind: 'slider',
    id: 'horizonGrowth',
    label: 'Step growth',
    group: 'Horizon',
    apply: 'compile',
    min: 1,
    max: 1.3,
    step: 0.02,
    default: 1.1,
    expert: true,
    disabledWhen: state => state.horizonAlgorithm === 'sweep',
    help: 'Each ray step is this much longer than the last. 1 samples every cell; larger values reach farther with fewer samples (the sweep requires 1).'
  },
  {
    kind: 'slider',
    id: 'anisotropyAzimuth',
    label: 'Preferred direction',
    group: 'Horizon',
    apply: 'param',
    min: 0,
    max: 360,
    step: 5,
    default: 315,
    unit: '°',
    expert: true,
    disabledWhen: isProduct('anisotropic'),
    help: 'Compass bearing the anisotropic sky-view favors (RVT uses the sun direction).'
  },
  {
    kind: 'slider',
    id: 'anisotropyLevel',
    label: 'Anisotropy level',
    group: 'Horizon',
    apply: 'param',
    min: 1,
    max: 8,
    step: 1,
    default: 4,
    expert: true,
    disabledWhen: isProduct('anisotropic'),
    help: 'How sharply directions are weighted: 1 is broad, 8 is a narrow beam.'
  },
  {
    kind: 'slider',
    id: 'anisotropyMinimumWeight',
    label: 'Minimum direction weight',
    group: 'Horizon',
    apply: 'param',
    min: 0,
    max: 1,
    step: 0.05,
    default: 0.4,
    expert: true,
    disabledWhen: isProduct('anisotropic'),
    help: 'Weight given to the directions opposite the preferred one. 1 makes the result isotropic.'
  },

  // --- Local relief -------------------------------------------------------------------------------
  {
    kind: 'slider',
    id: 'rvtExaggeration',
    label: 'Relief exaggeration',
    group: 'Local relief',
    apply: 'param',
    min: 0.5,
    max: 5,
    step: 0.1,
    default: 1,
    unit: 'x',
    expert: true,
    disabledWhen: isProduct('slrm', 'msrm', 'local-dominance'),
    help: 'Multiplies the elevation differences of the three local-relief models before display.'
  },
  {
    kind: 'slider',
    id: 'msrmMinimumFeature',
    label: 'MSRM smallest feature',
    group: 'Local relief',
    apply: 'compile',
    min: 8,
    max: 80,
    step: 4,
    default: 20,
    unit: 'm',
    expert: true,
    disabledWhen: isProduct('msrm'),
    help: 'Smallest feature size in metres the multi-scale model resolves.'
  },
  {
    kind: 'slider',
    id: 'msrmMaximumFeature',
    label: 'MSRM largest feature',
    group: 'Local relief',
    apply: 'compile',
    min: 100,
    max: 1200,
    step: 50,
    default: 500,
    unit: 'm',
    expert: true,
    disabledWhen: isProduct('msrm'),
    help: 'Largest feature size in metres. The filter radii span smallest to largest.'
  },
  {
    kind: 'slider',
    id: 'msrmScaling',
    label: 'MSRM radius growth',
    group: 'Local relief',
    apply: 'compile',
    min: 1,
    max: 3,
    step: 1,
    default: 2,
    expert: true,
    disabledWhen: isProduct('msrm'),
    help: 'Filter radii are k to the power of this integer: 1 is linear, 2 quadratic (denser at small scales).'
  },
  {
    kind: 'slider',
    id: 'dominanceMinimumRadius',
    label: 'Dominance inner radius',
    group: 'Local relief',
    apply: 'compile',
    min: 3,
    max: 20,
    step: 1,
    default: 10,
    unit: 'px',
    expert: true,
    disabledWhen: isProduct('local-dominance'),
    help: 'Closest ring of samples the observer looks at.'
  },
  {
    kind: 'slider',
    id: 'dominanceMaximumRadius',
    label: 'Dominance outer radius',
    group: 'Local relief',
    apply: 'compile',
    min: 12,
    max: 40,
    step: 1,
    default: 20,
    unit: 'px',
    expert: true,
    disabledWhen: isProduct('local-dominance'),
    help: 'Farthest ring of samples. Together with the inner radius it sets the annulus the observer scans.'
  },
  {
    kind: 'slider',
    id: 'dominanceIncrement',
    label: 'Dominance ring spacing',
    group: 'Local relief',
    apply: 'compile',
    min: 1,
    max: 4,
    step: 1,
    default: 1,
    unit: 'px',
    expert: true,
    disabledWhen: isProduct('local-dominance'),
    help: 'Distance between rings. Rings times directions is the number of samples per cell (see the readout).'
  },
  {
    kind: 'select',
    id: 'dominanceAngle',
    label: 'Dominance angular step',
    group: 'Local relief',
    apply: 'compile',
    default: '15',
    expert: true,
    disabledWhen: isProduct('local-dominance'),
    help: 'Angle between sample directions on a ring. 15 degrees (24 directions) is the RVT default.',
    options: [
      {value: '10', label: '10°'},
      {value: '15', label: '15°'},
      {value: '20', label: '20°'},
      {value: '30', label: '30°'}
    ]
  },
  {
    kind: 'slider',
    id: 'dominanceObserverHeight',
    label: 'Observer height',
    group: 'Local relief',
    apply: 'param',
    min: 0.5,
    max: 30,
    step: 0.5,
    default: 5,
    unit: 'm',
    expert: true,
    disabledWhen: isProduct('local-dominance'),
    help: 'Height of the observer above each cell. A taller observer sees over small bumps, so only larger landforms dominate. A parameter write (the heavy graph re-runs at most every half second).'
  },

  // --- VAT blend ----------------------------------------------------------------------------------
  {
    kind: 'slider',
    id: 'vatSlopeOpacity',
    label: 'Slope layer opacity (x preset)',
    group: 'VAT blend',
    apply: 'param',
    min: 0,
    max: 2,
    step: 0.1,
    default: 1,
    expert: true,
    disabledWhen: isProduct('vat'),
    help: 'Scales the opacity of the inverted slope layer relative to the preset.'
  },
  {
    kind: 'slider',
    id: 'vatOpennessOpacity',
    label: 'Openness layer opacity (x preset)',
    group: 'VAT blend',
    apply: 'param',
    min: 0,
    max: 2,
    step: 0.1,
    default: 1,
    expert: true,
    disabledWhen: isProduct('vat'),
    help: 'Scales the opacity of the positive-openness layer relative to the preset.'
  },
  {
    kind: 'slider',
    id: 'vatSkyViewOpacity',
    label: 'Sky-view layer opacity (x preset)',
    group: 'VAT blend',
    apply: 'param',
    min: 0,
    max: 4,
    step: 0.2,
    default: 1,
    expert: true,
    disabledWhen: isProduct('vat'),
    help: 'Scales the opacity of the sky-view layer relative to the preset.'
  },
  {
    kind: 'select',
    id: 'vatSlopeBlend',
    label: 'Slope blend mode',
    group: 'VAT blend',
    apply: 'param',
    default: 'preset',
    expert: true,
    disabledWhen: isProduct('vat'),
    help: 'How the slope layer combines with the layers below it.',
    options: BLEND_CHOICES
  },
  {
    kind: 'select',
    id: 'vatOpennessBlend',
    label: 'Openness blend mode',
    group: 'VAT blend',
    apply: 'param',
    default: 'preset',
    expert: true,
    disabledWhen: isProduct('vat'),
    help: 'How the positive-openness layer combines with the layers below it.',
    options: BLEND_CHOICES
  },
  {
    kind: 'select',
    id: 'vatSkyViewBlend',
    label: 'Sky-view blend mode',
    group: 'VAT blend',
    apply: 'param',
    default: 'preset',
    expert: true,
    disabledWhen: isProduct('vat'),
    help: 'How the sky-view layer combines with the layers below it.',
    options: BLEND_CHOICES
  },

  // --- Display ------------------------------------------------------------------------------------
  {
    kind: 'slider',
    id: 'clipPercent',
    label: 'Percent clipped at each end',
    group: 'Display',
    apply: 'param',
    min: 0,
    max: 10,
    step: 0.5,
    default: 2,
    unit: '%',
    expert: true,
    help: 'Share of cells allowed to saturate at each end of a percentile stretch (texture shading, local dominance) and in the fitted local relief limits.'
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
    help: 'Lower it to see the paper ground through the drawing.'
  },
  {
    kind: 'button',
    id: 'measure',
    label: 'Time this drawing',
    group: 'Under the hood',
    expert: true,
    help: 'Runs the graphs of the displayed drawing (and the stages it reads) outside the frame and sums their GPU time. It also runs once, a moment after each drawing settles.'
  }
];

const READOUTS: readonly ReadoutSpec[] = [
  {
    id: 'lightDial',
    label: 'Light',
    kind: 'chart',
    help: 'Where the light comes from (compass) and how high it stands (side view).'
  },
  {
    id: 'lights',
    label: 'Lights summed',
    emphasis: 'tile',
    help: 'Lights the drawing adds up: one for a single or swung light, the USGS set for the multidirectional hillshade.'
  },
  {
    id: 'svfTaps',
    label: 'Horizon samples',
    hood: true,
    help: 'Azimuth sectors times march steps per cell: the cost of the sky-view and openness passes.'
  },
  {
    id: 'median',
    label: 'Median cell',
    emphasis: 'tile',
    help: 'The middle value of the drawing over the whole tile (measured from a GPU histogram).'
  },
  {
    id: 'horizonProfile',
    label: 'Horizon at the pinned cell',
    kind: 'chart',
    help: 'Click the map to pin a cell: the horizon angle in each direction, computed on the CPU with the same rays.'
  },
  {
    id: 'windowMeters',
    label: 'Window width',
    format: 'meters',
    emphasis: 'tile',
    help: 'The square window of the local relief model on the ground: (2r + 1) cells.'
  },
  {
    id: 'clip',
    label: 'Class limits',
    help: 'The class limits of the local relief map in metres, mirrored below and above zero.'
  },
  {id: 'recipeStage', label: 'Recipe stage', help: 'The cues of the Imhof recipe switched on now.'},
  {
    id: 'gpuTime',
    label: 'GPU time',
    format: 'milliseconds',
    emphasis: 'tile',
    help: 'Time of every graph behind the drawing, measured outside the frame.'
  },
  {
    id: 'timing',
    label: 'GPU time (detail)',
    hood: true,
    help: 'The graphs of the displayed drawing and the stages it reads.'
  },
  {id: 'grid', label: 'Raster', hood: true},
  {
    id: 'stretch',
    label: 'Colour stretch',
    hood: true,
    help: 'The value range the grey ramp spans: percentile-clipped, or a fixed physical range.'
  },
  {id: 'extent', label: 'Value range', hood: true},
  {id: 'msrmRadii', label: 'MSRM mean-filter radii', hood: true},
  {id: 'dominanceTaps', label: 'Local dominance samples per cell', hood: true}
];

/**
 * Relief visualization of the Matterhorn and the Gorner glacier: one light, many lights,
 * light-free measures, local relief, Imhof's recipe and the VAT blend, computed on the GPU from one
 * DEM and compared on the same landform.
 */
export default defineScene<ReliefOptions>({
  id: 'relief-visualization',
  title: 'Seeing terrain: relief visualization',
  chapter: 'terrain',
  order: 2,
  summary:
    'Which drawing shows the mountain best? One Alpine DEM drawn by one light, many lights, sky-view and texture, local relief and the Swiss recipe, each compared on the same landform.',
  contributors: [
    'GPUReliefShading',
    'GPUTextureShading',
    'GPUTerrainHorizon',
    'GPUSimpleLocalRelief',
    'GPUMultiScaleRelief',
    'GPULocalDominance',
    'GPUReliefBlend',
    'GPUTerrainCurvature',
    'GPUTerrainDerivatives'
  ],
  datasets: [
    {id: 'alps-dem', role: 'elevation'},
    {id: 'alps-context', role: 'glaciers, peaks, places'}
  ],
  initialView: RELIEF_FRAME,

  basemap: ground('relief'),
  furniture: {
    title: cartouche(
      'Which drawing shows the mountain best?',
      'Relief drawings of the Matterhorn and the Gornergrat'
    ),
    scaleBar: {units: 'metric'},
    credit: TERRAIN_CREDIT
  },

  options: OPTIONS,
  readouts: READOUTS,

  pipeline: [
    {id: 'elevation', label: 'Elevation', detail: 'One decoded DEM, read by every drawing'},
    {
      id: 'light',
      label: 'Light',
      detail: 'Hillshade: slope and aspect against one or several lights',
      show: {option: 'product', value: 'hillshade'}
    },
    {
      id: 'horizon',
      label: 'Horizon',
      detail: 'Sky-view and openness from horizon angles in every direction',
      show: {option: 'product', value: 'sky-view'}
    },
    {
      id: 'local',
      label: 'Local relief',
      detail: 'Height minus the mean of a window',
      show: {option: 'product', value: 'slrm'}
    },
    {
      id: 'composite',
      label: 'Composite',
      detail: 'Tint, warm light, cool shade and haze with height',
      show: {option: 'product', value: 'imhof'}
    }
  ],

  legends: getReliefLegends,

  story: [
    {
      id: 'light-from',
      title: 'Light the mountain from the north-west',
      headline: 'North-west light makes the peaks rise',
      textAlternative:
        'Grey hillshade of the Matterhorn, Zermatt and Gornergrat area lit from the north-west: slopes facing the light are bright, slopes facing away are dark.',
      body: 'Each cell is as bright as its slope faces the light: `cos(zenith)·cos(slope) + sin(zenith)·sin(slope)·cos(light − aspect)`. **`GPUReliefShading`** evaluates it for every cell in one pass, and moving **Light from** only rewrites a parameter, so nothing recompiles.\n\nNow choose **Light preset** south-east: many eyes read the same ground as hollows instead of peaks.\n\n*North-west light is a convention, not physics: the eye assumes light from above-left.*',
      optionsMode: 'fresh',
      options: {product: 'hillshade', lightAzimuth: 315, lightAltitude: 40},
      controls: ['lightAzimuth', 'lightPreset'],
      readouts: ['lightDial', 'timing'],
      camera: {...RELIEF_FRAME, transitionMs: 1400},
      furniture: {
        title: cartouche(
          'Which way does the light come from?',
          'Hillshade, single light · 6.6 m cells'
        )
      },
      annotations: [placeLabel('zermatt')],
      stage: 'light'
    },
    {
      id: 'many-lights',
      title: 'Light it from several sides',
      headline: 'One light hides the faces it grazes',
      textAlternative:
        'The Matterhorn north face split by a divider: on the left one light from the north-west, on the right the drawing chosen below.',
      body: 'A face parallel to a single ray is lit the same on both flanks, so a ridge running along the light disappears. The multidirectional hillshade sums **{{lights}}**, each weighted by `sin²(aspect − azimuth)`; the Imhof swing turns one light toward each slope. All three are the same graph with different parameters.\n\nDrag the divider and pick the **Drawing** on its right.\n\n*Any one light hides something.*',
      optionsMode: 'fresh',
      options: {
        product: 'mdow',
        reference: 'single',
        lightAzimuth: 315,
        lightAltitude: 40
      },
      controls: ['lightModelPreset'],
      readouts: ['lights', 'timing'],
      compare: {mode: 'swipe', labels: ['One light', 'Chosen drawing'], position: 0.5},
      camera: {...NORTH_FACE_FRAME, transitionMs: 1600},
      furniture: {
        title: cartouche(
          'What does one light hide?',
          'Hillshade, one light against the chosen drawing'
        )
      },
      annotations: [placeLabel('hornli-hut')],
      stage: 'light'
    },
    {
      id: 'no-light',
      title: 'Draw form without a sun',
      headline: 'Sky-view shows form without a sun',
      textAlternative:
        'The Matterhorn and Gornergrat area split by a divider: on the left the multidirectional hillshade, on the right a light-free drawing of enclosure, such as sky-view.',
      body: 'Light-free drawings measure the terrain around each cell. The **sky-view factor** is the share of sky a cell sees, `1 − mean(sin horizon)`; **`GPUTerrainHorizon`** takes **{{svfTaps}}**, and the median cell sees **{{median}}**.\n\nClick the map to pin a cell and see its horizon rays; change the **Search radius** to see how far they look.\n\n*Light-free drawings trade direction for honesty: enclosure, not slope facing.*',
      optionsMode: 'fresh',
      options: {product: 'sky-view', reference: 'mdow', horizonRadius: '96'},
      controls: ['lightFreePreset', 'horizonRadius'],
      readouts: ['svfTaps', 'median', 'horizonProfile', 'timing'],
      compare: {
        mode: 'swipe',
        labels: ['Multidirectional hillshade', 'Light-free drawing'],
        position: 0.5
      },
      camera: {...RELIEF_FRAME, transitionMs: 1600},
      furniture: {
        title: cartouche('How does form look without a sun?', 'Sky-view factor, fixed range')
      },
      annotations: [placeLabel('zermatt')],
      stage: 'horizon'
    },
    {
      id: 'scale-of-bumps',
      title: 'Local relief depends on the window',
      headline: 'What counts as a bump depends on the window',
      textAlternative:
        'The Gorner moraines drawn as local relief over a grey hillshade: orange classes are higher than the window mean, purple lower, and the middle class is not drawn.',
      body: '**`GPUSimpleLocalRelief`** subtracts the mean of a square window from each cell. The window is **{{windowMeters}}** wide: smaller features stand out, larger landforms average away. Purple is below the surroundings, orange above; the middle class is not drawn, so the hillshade shows through.\n\nPick the **Window**. With **Fixed breaks** one set of limits (**{{clip}}**) serves every window, so a larger window honestly looks stronger; **Fit to view** hides that.\n\n*A stretch chosen per view cannot be compared between views.*',
      optionsMode: 'fresh',
      options: {product: 'slrm', slrmRadius: '13', stretch: 'fixed'},
      controls: ['slrmRadius', 'stretch'],
      readouts: ['windowMeters', 'clip', 'timing'],
      camera: {...MORAINE_FRAME, transitionMs: 1600},
      furniture: {
        title: cartouche('What counts as a bump?', 'Local relief, metres · fixed breaks')
      },
      annotations: [placeLabel('riffelsee')],
      stage: 'local'
    },
    {
      id: 'swiss-relief',
      title: 'Imhof’s recipe: tint, light, shade, haze',
      headline: 'Imhof’s recipe: tint, warm light, cool shade, haze',
      textAlternative:
        'The Matterhorn and Gornergrat area split by a divider: on the left a grey hillshade, on the right the Swiss relief with the cues of the chosen recipe stage switched on.',
      body: 'Swiss relief adds one cue at a time. Step **Recipe** up: a pale elevation tint, warm light and cool shade, contrast that grows with height (**{{recipeStage}}** now), then the Imhof swing. Every stage is a parameter write to **`GPUReliefShading`**.\n\nThe tint is pale on purpose: green does not mean forest, and a strong tint reads as land cover. This is the ground under every other story of the chapter, built there once on the CPU with glaciers from OpenStreetMap.\n\n*Aerial perspective: contrast rises with height.*',
      optionsMode: 'fresh',
      options: {product: 'imhof', reference: 'mdow', recipe: 1},
      controls: ['recipe'],
      readouts: ['recipeStage', 'gpuTime'],
      compare: {mode: 'swipe', labels: ['Grey hillshade', 'Recipe stage'], position: 0.5},
      camera: {...RELIEF_FRAME, transitionMs: 1600},
      furniture: {
        title: cartouche('What does Imhof add to a hillshade?', 'Swiss relief, built up by cue')
      },
      annotations: [placeLabel('zermatt')],
      stage: 'composite'
    },
    {
      id: 'stack-and-explore',
      title: 'Stack the drawings, then explore',
      headline: 'Stacking drawings helps only with care',
      textAlternative:
        'The VAT blend of the Matterhorn and Gornergrat area, with an exploded diagram of the four stacked sheets and their blend modes.',
      body: '**`GPUReliefBlend`** stacks hillshade, inverted slope, openness and sky-view with blend modes (the VAT recipe). Order and mode matter, and every stretch fitted to a view is a private scale: fix the range, as in [the local relief step](action:step?id=scale-of-bumps).\n\nThe ladder so far: light, many lights, light-free measures, local relief, composite. Pick any **Drawing**, tilt **Vertical exaggeration**, or change the **VAT preset**. This one takes **{{gpuTime}}**.',
      optionsMode: 'fresh',
      options: {product: 'vat'},
      controls: ['product', 'zFactor', 'vatPreset'],
      readouts: ['gpuTime', 'grid'],
      camera: {...RELIEF_FRAME, transitionMs: 1600},
      furniture: {
        title: cartouche('Which drawing for which question?', 'VAT blend of four drawings')
      },
      annotations: [placeLabel('zermatt')],
      diagram: makeStackDiagram(),
      stage: 'composite'
    }
  ],

  about: {
    what: '`GPUReliefShading` computes single, multidirectional (USGS MDOW) and Imhof-swing hillshades and blends a Swiss relief from tint, warm light and cool shade. `GPUTextureShading` approximates Leland Brown texture shading with a Gaussian cascade. `GPUTerrainHorizon` gives horizon angles, sky-view factor and positive and negative openness. `GPUSimpleLocalRelief`, `GPUMultiScaleRelief` and `GPULocalDominance` follow the Relief Visualization Toolbox; `GPUReliefBlend` is the VAT layer blend.',
    why: 'Elevation is only legible after it has been turned into a drawing, and each drawing answers a different question: faces, enclosure, small bumps or a finished map. Computing them on the GPU makes the light, the scale and the blend interactive, and a swipe lets you compare them on the same landform instead of from memory.',
    howToRead:
      'Hillshades are a brightness, not a measurement: bright faces turn toward the light. Texture shade: bright is a bump, dark a groove. Sky-view and openness: dark is enclosed, bright is exposed. Local relief: orange is higher than the window mean, purple lower. The Swiss recipe and the VAT blend read as shaded maps. Reference implementations: RVT-py (SLRM, MSRM, local dominance, sky-view, openness, VAT), GDAL (multidirectional hillshade) and mt-image (Imhof swing).'
  },

  create: async ctx =>
    (await import('./relief-visualization.compute')).createReliefVisualization(ctx),

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUTerrainHorizon, GPUReliefShading, GPUSimpleLocalRelief, GPUReliefBlend,
  getGPUReliefShadingParameterValues, GPU_RELIEF_BLEND_VAT_ARCHAEOLOGICAL
} from '@luma.gl/experimental/gpu-terrain';

const cell = {cellSize: [9.5546, 9.5546], northEdge, southEdge}; // Web Mercator tile
const graph = new GPUCommandGraph(device, {id: 'relief'});
graph.add(new GPUTerrainHorizon({
  width, height, elevation, settings: horizonSettings.importToGraph(graph),
  directionCount: ${state.horizonDirections}, maximumRadius: ${state.horizonRadius}, algorithm: '${state.horizonAlgorithm}',
  cellSizeMode: 'web-mercator', skyViewFactor
}));
graph.add(new GPUReliefShading({
  width, height, elevation, settings: reliefSettings.importToGraph(graph),
  imhofSwing: true, cellSizeMode: 'web-mercator', hillshade, color
}));
graph.add(new GPUSimpleLocalRelief({
  width, height, elevation, settings: slrmSettings.importToGraph(graph), radius: ${state.slrmRadius}, relief
}));
const compiled = graph.compile(); // once

// per frame or on input: parameter writes only
reliefSettings.write(getGPUReliefShadingParameterValues({
  ...cell,
  lights: [{azimuthDegrees: ${state.lightAzimuth}, altitudeDegrees: ${state.lightAltitude}}],
  lightWeighting: '${state.recipe >= 4 ? 'imhof-swing' : 'fixed'}', imhofSwingDegrees: ${state.imhofSwing},
  tintStrength: ${state.recipe >= 2 ? 0.55 : 0}, contrastStrength: ${state.recipe >= 3 ? 0.6 : 0},
  contrastLowElevation: 1800, contrastHighElevation: 4200,
  elevationStops: ${state.recipe >= 1 ? 'paleTintStops' : '[]'}
}));
compiled.encode(commandEncoder, {parameters: undefined});`
});
