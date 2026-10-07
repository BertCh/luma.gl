// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
import {
  ALPINE_ELEVATION_STOPS,
  SCALAR_PRODUCTS,
  type ReliefOptions,
  type VatBlendChoice
} from './relief-visualization.style';

const BLEND_CHOICES: {value: VatBlendChoice; label: string}[] = [
  {value: 'preset', label: 'Use the preset'},
  {value: 'normal', label: 'Normal'},
  {value: 'multiply', label: 'Multiply'},
  {value: 'screen', label: 'Screen'},
  {value: 'overlay', label: 'Overlay'},
  {value: 'soft-light', label: 'Soft light'},
  {value: 'luminosity', label: 'Luminosity'}
];

const isProduct =
  (...products: ReliefOptions['product'][]) =>
  (state: ReliefOptions) =>
    !products.includes(state.product);

/**
 * Relief visualization of the Matterhorn and the Gorner glacier: hillshade families, texture
 * shading, sky-view and openness, the Swiss/Imhof relief, three RVT micro-relief techniques and
 * the Visualization for Archaeological Topography blend.
 */
export default defineScene<ReliefOptions>({
  id: 'relief-visualization',
  title: 'Seeing terrain: relief visualization',
  chapter: 'terrain',
  order: 2,
  summary:
    'Twelve ways to draw the same Alpine DEM on the GPU: hillshade, multidirectional, texture shading, sky-view factor and openness, the Swiss/Imhof relief, local relief models, local dominance and the VAT blend.',
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
  datasets: [{id: 'alps-dem', role: 'elevation'}],
  initialView: {longitude: 7.76, latitude: 45.98, zoom: 12.4},

  options: [
    {
      kind: 'slider',
      id: 'lightAzimuth',
      label: 'Light azimuth',
      group: 'Hillshade light',
      apply: 'param',
      min: 0,
      max: 360,
      step: 5,
      default: 315,
      unit: '°',
      disabledWhen: isProduct('hillshade', 'imhof', 'vat'),
      help: 'Direction the light comes from, clockwise from north. Used by the single-light hillshade, the Swiss relief (Imhof swing and single-light models) and the VAT hillshade layer.'
    },
    {
      kind: 'slider',
      id: 'lightAltitude',
      label: 'Light altitude',
      group: 'Hillshade light',
      apply: 'param',
      min: 5,
      max: 85,
      step: 1,
      default: 35,
      unit: '°',
      disabledWhen: isProduct('hillshade', 'imhof', 'vat'),
      help: 'Height of the light above the horizon. Low light exaggerates texture; high light flattens it.'
    },
    {
      kind: 'slider',
      id: 'zFactor',
      label: 'Vertical exaggeration',
      group: 'Hillshade light',
      apply: 'param',
      min: 0.5,
      max: 4,
      step: 0.1,
      default: 1,
      unit: 'x',
      help: 'Multiplies heights for the shading products (hillshade, Swiss relief, sky-view, openness, VAT slope). RVT calls it vertical exaggeration; it makes subtle terrain show up.'
    },
    {
      kind: 'slider',
      id: 'textureLevels',
      label: 'Cascade levels',
      group: 'Texture shading (GPUTextureShading)',
      apply: 'compile',
      min: 1,
      max: 8,
      step: 1,
      default: 6,
      disabledWhen: isProduct('texture', 'imhof'),
      help: 'Number of Gaussian band-pass levels, each twice as wide as the last. More levels reach larger landforms. Rebuilds the texture graph.'
    },
    {
      kind: 'slider',
      id: 'textureBaseSigma',
      label: 'Finest level sigma',
      group: 'Texture shading (GPUTextureShading)',
      apply: 'compile',
      min: 0.5,
      max: 4,
      step: 0.5,
      default: 1,
      unit: 'px',
      disabledWhen: isProduct('texture', 'imhof'),
      help: 'Blur width of the finest level in pixels; the other levels are 2x, 4x, 8x that.'
    },
    {
      kind: 'slider',
      id: 'textureDetail',
      label: 'Detail (alpha)',
      group: 'Texture shading (GPUTextureShading)',
      apply: 'param',
      min: 0,
      max: 1,
      step: 0.05,
      default: 0.5,
      disabledWhen: isProduct('texture', 'imhof'),
      help: 'The exponent of the fractional Laplacian: 0 weights all scales equally, 1 favors fine texture. Leland Brown uses 0.5 to 0.7. A parameter write.'
    },
    {
      kind: 'slider',
      id: 'textureGain',
      label: 'Gain',
      group: 'Texture shading (GPUTextureShading)',
      apply: 'param',
      min: 0.001,
      max: 0.05,
      step: 0.001,
      default: 0.01,
      disabledWhen: isProduct('texture', 'imhof'),
      help: 'Multiplier from elevation meters to the texture-shade value. Only matters for the Swiss blend, where the shade is added to the hillshade, or with a fixed range.'
    },
    {
      kind: 'toggle',
      id: 'textureHasNodata',
      label: 'Handle nodata',
      group: 'Texture shading (GPUTextureShading)',
      apply: 'compile',
      default: true,
      disabledWhen: isProduct('texture', 'imhof'),
      help: 'Turn off when the DEM has no nodata (this one does not): the blur then needs one channel instead of two, about half the cost, with identical output.'
    },
    {
      kind: 'toggle',
      id: 'textureDownsample',
      label: 'Downsample wide levels',
      group: 'Texture shading (GPUTextureShading)',
      apply: 'compile',
      default: true,
      disabledWhen: isProduct('texture', 'imhof'),
      help: 'Computes wide levels on a half-resolution grid and upsamples them. Output stays within 1 % of the exact result and runs much faster.'
    },
    {
      kind: 'select',
      id: 'horizonDirections',
      label: 'Azimuth sectors',
      group: 'Sky-view and openness (GPUTerrainHorizon)',
      apply: 'compile',
      default: '16',
      disabledWhen: isProduct(
        'sky-view',
        'anisotropic',
        'positive-openness',
        'negative-openness',
        'imhof',
        'vat'
      ),
      help: 'How many compass directions the horizon is sampled in. More sectors give smoother sky-view and openness at proportional cost.',
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
      group: 'Sky-view and openness (GPUTerrainHorizon)',
      apply: 'compile',
      default: '96',
      disabledWhen: isProduct(
        'sky-view',
        'anisotropic',
        'positive-openness',
        'negative-openness',
        'imhof',
        'vat'
      ),
      help: 'How far (in pixels, 6.6 m each) each ray looks for a blocking horizon. A bigger radius sees the far valley walls, and costs more.',
      options: [
        {value: '32', label: '32 px (210 m)'},
        {value: '64', label: '64 px (420 m)'},
        {value: '96', label: '96 px (630 m)'},
        {value: '192', label: '192 px (1.3 km)'},
        {value: '384', label: '384 px (2.5 km)'}
      ]
    },
    {
      kind: 'select',
      id: 'horizonAlgorithm',
      label: 'Algorithm',
      group: 'Sky-view and openness (GPUTerrainHorizon)',
      apply: 'compile',
      default: 'march',
      disabledWhen: isProduct(
        'sky-view',
        'anisotropic',
        'positive-openness',
        'negative-openness',
        'imhof',
        'vat'
      ),
      help: 'March samples each ray at growing steps (cheap at short radii). Sweep is an exact upper-hull sweep along digital lines (Stewart 1998): fixed cost per pixel, so it wins at radii near the tile size and loses at short ones.',
      options: [
        {value: 'march', label: 'March: bilinear ray marching'},
        {value: 'sweep', label: 'Sweep: exact hull sweep'}
      ]
    },
    {
      kind: 'slider',
      id: 'horizonGrowth',
      label: 'Step growth',
      group: 'Sky-view and openness (GPUTerrainHorizon)',
      apply: 'compile',
      min: 1,
      max: 1.3,
      step: 0.02,
      default: 1.1,
      disabledWhen: state =>
        ![
          'sky-view',
          'anisotropic',
          'positive-openness',
          'negative-openness',
          'imhof',
          'vat'
        ].includes(state.product) || state.horizonAlgorithm === 'sweep',
      help: 'Each ray step is this much longer than the last. 1 samples every pixel; larger values reach farther with fewer samples (the sweep requires 1).'
    },
    {
      kind: 'slider',
      id: 'anisotropyAzimuth',
      label: 'Preferred direction',
      group: 'Sky-view and openness (GPUTerrainHorizon)',
      apply: 'param',
      min: 0,
      max: 360,
      step: 5,
      default: 315,
      unit: '°',
      disabledWhen: isProduct('anisotropic'),
      help: 'Compass bearing the anisotropic sky-view favors (RVT uses the sun direction).'
    },
    {
      kind: 'slider',
      id: 'anisotropyLevel',
      label: 'Anisotropy level',
      group: 'Sky-view and openness (GPUTerrainHorizon)',
      apply: 'param',
      min: 1,
      max: 8,
      step: 1,
      default: 4,
      disabledWhen: isProduct('anisotropic'),
      help: 'How sharply directions are weighted: 1 is broad, 8 is a narrow beam.'
    },
    {
      kind: 'slider',
      id: 'anisotropyMinimumWeight',
      label: 'Minimum direction weight',
      group: 'Sky-view and openness (GPUTerrainHorizon)',
      apply: 'param',
      min: 0,
      max: 1,
      step: 0.05,
      default: 0.4,
      disabledWhen: isProduct('anisotropic'),
      help: 'Weight given to the directions opposite the preferred one. 1 makes the result isotropic.'
    },
    {
      kind: 'select',
      id: 'swissLight',
      label: 'Swiss relief light model',
      group: 'Swiss / Imhof relief',
      apply: 'param',
      default: 'imhof-swing',
      disabledWhen: isProduct('imhof'),
      help: 'How the lights are weighted: Imhof swing turns the light toward the slope side by up to the swing angle, so no face is left flat; MDOW weights 4 to 8 lights by aspect (GDAL -multidirectional); fixed is one light.',
      options: [
        {value: 'imhof-swing', label: 'Imhof swing (turn the light toward the slope)'},
        {value: 'mdow', label: 'USGS multidirectional (aspect weighted)'},
        {value: 'fixed', label: 'One fixed light'}
      ]
    },
    {
      kind: 'slider',
      id: 'imhofSwing',
      label: 'Imhof swing',
      group: 'Swiss / Imhof relief',
      apply: 'param',
      min: 0,
      max: 90,
      step: 5,
      default: 65,
      unit: '°',
      disabledWhen: state => state.product !== 'imhof' || state.swissLight !== 'imhof-swing',
      help: 'Largest angle the light is swung toward the side a slope faces. 0 is a fixed light.'
    },
    {
      kind: 'slider',
      id: 'curvatureStrength',
      label: 'Curvature strength',
      group: 'Swiss / Imhof relief',
      apply: 'param',
      min: 0,
      max: 1.5,
      step: 0.05,
      default: 0.5,
      disabledWhen: isProduct('imhof'),
      help: 'Adds the multi-radius ring curvature (convex bright, concave dark) to the shading, which sharpens ridges and gullies.'
    },
    {
      kind: 'slider',
      id: 'hillshadeStrength',
      label: 'Hillshade strength',
      group: 'Swiss / Imhof relief',
      apply: 'param',
      min: 0,
      max: 1.5,
      step: 0.05,
      default: 1,
      disabledWhen: isProduct('imhof'),
      help: 'Weight of the hillshade term in the blend.'
    },
    {
      kind: 'slider',
      id: 'skyViewStrength',
      label: 'Sky-view strength',
      group: 'Swiss / Imhof relief',
      apply: 'param',
      min: 0,
      max: 1.5,
      step: 0.05,
      default: 1,
      disabledWhen: isProduct('imhof'),
      help: 'Weight of the sky-view factor: valleys and cirques darken as less sky is visible.'
    },
    {
      kind: 'slider',
      id: 'textureStrength',
      label: 'Texture-shade strength',
      group: 'Swiss / Imhof relief',
      apply: 'param',
      min: 0,
      max: 1.5,
      step: 0.05,
      default: 0.4,
      disabledWhen: isProduct('imhof'),
      help: 'Weight of the texture shading term (the texture stage computes it for this blend).'
    },
    {
      kind: 'slider',
      id: 'contrastStrength',
      label: 'Contrast with elevation',
      group: 'Swiss / Imhof relief',
      apply: 'param',
      min: 0,
      max: 1,
      step: 0.05,
      default: 0.5,
      disabledWhen: isProduct('imhof'),
      help: 'Swiss cartographers raise relief contrast with height so high rock stands out against pale glaciers. 0 turns it off.'
    },
    {
      kind: 'slider',
      id: 'contrastHighElevation',
      label: 'Full contrast at',
      group: 'Swiss / Imhof relief',
      apply: 'param',
      min: 2500,
      max: 4500,
      step: 50,
      default: 3800,
      unit: 'm',
      disabledWhen: state => state.product !== 'imhof' || state.contrastStrength === 0,
      help: 'Elevation where the contrast boost reaches its full strength (it starts at 1,500 m).'
    },
    {
      kind: 'slider',
      id: 'exposure',
      label: 'Exposure',
      group: 'Swiss / Imhof relief',
      apply: 'param',
      min: 0.6,
      max: 1.8,
      step: 0.05,
      default: 1.1,
      disabledWhen: isProduct('imhof'),
      help: 'Overall brightness of the blend.'
    },
    {
      kind: 'toggle',
      id: 'elevationTint',
      label: 'Elevation tint',
      group: 'Swiss / Imhof relief',
      apply: 'param',
      default: true,
      disabledWhen: isProduct('imhof'),
      help: 'Colors the relief by height: meadow green in the valley, rock tones, then snow white (an elevation-stop table of up to 8 colors).'
    },
    {
      kind: 'slider',
      id: 'tintStrength',
      label: 'Warm / cool aspect tint',
      group: 'Swiss / Imhof relief',
      apply: 'param',
      min: 0,
      max: 1,
      step: 0.05,
      default: 0.3,
      disabledWhen: isProduct('imhof'),
      help: 'Tints lit slopes warm and shaded slopes cool, the classic Swiss hand-shading trick.'
    },
    {
      kind: 'slider',
      id: 'rvtExaggeration',
      label: 'Vertical exaggeration',
      group: 'Local relief (RVT)',
      apply: 'param',
      min: 0.5,
      max: 5,
      step: 0.1,
      default: 1,
      unit: 'x',
      disabledWhen: isProduct('slrm', 'msrm', 'local-dominance'),
      help: 'Multiplies the elevation differences of the three local-relief models before display.'
    },
    {
      kind: 'slider',
      id: 'slrmRadius',
      label: 'SLRM window radius',
      group: 'Local relief (RVT)',
      apply: 'compile',
      min: 3,
      max: 50,
      step: 1,
      default: 20,
      unit: 'px',
      disabledWhen: isProduct('slrm'),
      help: 'The local mean is taken over a (2r+1) x (2r+1) window; features smaller than the window show up. 20 px is 130 m here. Cost grows with the radius.'
    },
    {
      kind: 'slider',
      id: 'msrmMinimumFeature',
      label: 'MSRM smallest feature',
      group: 'Local relief (RVT)',
      apply: 'compile',
      min: 8,
      max: 80,
      step: 4,
      default: 20,
      unit: 'm',
      disabledWhen: isProduct('msrm'),
      help: 'Smallest feature size in meters the multi-scale model resolves.'
    },
    {
      kind: 'slider',
      id: 'msrmMaximumFeature',
      label: 'MSRM largest feature',
      group: 'Local relief (RVT)',
      apply: 'compile',
      min: 100,
      max: 1200,
      step: 50,
      default: 500,
      unit: 'm',
      disabledWhen: isProduct('msrm'),
      help: 'Largest feature size in meters. The filter radii span smallest to largest.'
    },
    {
      kind: 'slider',
      id: 'msrmScaling',
      label: 'MSRM radius growth',
      group: 'Local relief (RVT)',
      apply: 'compile',
      min: 1,
      max: 3,
      step: 1,
      default: 2,
      disabledWhen: isProduct('msrm'),
      help: 'Filter radii are k to the power of this integer: 1 is linear, 2 quadratic (denser at small scales).'
    },
    {
      kind: 'slider',
      id: 'dominanceMinimumRadius',
      label: 'Dominance inner radius',
      group: 'Local relief (RVT)',
      apply: 'compile',
      min: 3,
      max: 20,
      step: 1,
      default: 10,
      unit: 'px',
      disabledWhen: isProduct('local-dominance'),
      help: 'Closest ring of samples the observer looks at.'
    },
    {
      kind: 'slider',
      id: 'dominanceMaximumRadius',
      label: 'Dominance outer radius',
      group: 'Local relief (RVT)',
      apply: 'compile',
      min: 12,
      max: 40,
      step: 1,
      default: 20,
      unit: 'px',
      disabledWhen: isProduct('local-dominance'),
      help: 'Farthest ring of samples. Together with the inner radius it sets the annulus the observer scans.'
    },
    {
      kind: 'slider',
      id: 'dominanceIncrement',
      label: 'Dominance ring spacing',
      group: 'Local relief (RVT)',
      apply: 'compile',
      min: 1,
      max: 4,
      step: 1,
      default: 1,
      unit: 'px',
      disabledWhen: isProduct('local-dominance'),
      help: 'Distance between rings. The product of rings and directions is the number of samples per pixel (see the readout).'
    },
    {
      kind: 'select',
      id: 'dominanceAngle',
      label: 'Dominance angular step',
      group: 'Local relief (RVT)',
      apply: 'compile',
      default: '15',
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
      group: 'Local relief (RVT)',
      apply: 'param',
      min: 0.5,
      max: 30,
      step: 0.5,
      default: 5,
      unit: 'm',
      disabledWhen: isProduct('local-dominance'),
      help: 'Height of the observer above each cell. A taller observer sees over small bumps, so only larger landforms dominate. A parameter write (the heavy graph re-runs at most every half second).'
    },
    {
      kind: 'select',
      id: 'vatPreset',
      label: 'VAT preset',
      group: 'VAT blend (GPUReliefBlend)',
      apply: 'param',
      default: 'archaeological',
      disabledWhen: isProduct('vat'),
      help: 'Layer stretches of the RVT "Archaeological" and "Flat terrain" combinations. Hillshade only zeroes the other layers.',
      options: [
        {
          value: 'archaeological',
          label: 'Archaeological (slope 0-50°, openness 68-93°, SVF 0.7-1)'
        },
        {value: 'flat', label: 'Flat terrain (slope 0-15°, openness 85-93°, SVF 0.9-1)'},
        {value: 'hillshade-only', label: 'Hillshade only'}
      ]
    },
    {
      kind: 'slider',
      id: 'vatSlopeOpacity',
      label: 'Slope layer opacity (x preset)',
      group: 'VAT blend (GPUReliefBlend)',
      apply: 'param',
      min: 0,
      max: 2,
      step: 0.1,
      default: 1,
      disabledWhen: isProduct('vat'),
      help: 'Scales the opacity of the inverted slope layer relative to the preset (50 % luminosity).'
    },
    {
      kind: 'slider',
      id: 'vatOpennessOpacity',
      label: 'Openness layer opacity (x preset)',
      group: 'VAT blend (GPUReliefBlend)',
      apply: 'param',
      min: 0,
      max: 2,
      step: 0.1,
      default: 1,
      disabledWhen: isProduct('vat'),
      help: 'Scales the opacity of the positive-openness layer relative to the preset (50 % overlay).'
    },
    {
      kind: 'slider',
      id: 'vatSkyViewOpacity',
      label: 'Sky-view layer opacity (x preset)',
      group: 'VAT blend (GPUReliefBlend)',
      apply: 'param',
      min: 0,
      max: 4,
      step: 0.2,
      default: 1,
      disabledWhen: isProduct('vat'),
      help: 'Scales the opacity of the sky-view layer relative to the preset (25 % multiply).'
    },
    {
      kind: 'select',
      id: 'vatSlopeBlend',
      label: 'Slope blend mode',
      group: 'VAT blend (GPUReliefBlend)',
      apply: 'param',
      default: 'preset',
      disabledWhen: isProduct('vat'),
      help: 'How the slope layer combines with the layers below it.',
      options: BLEND_CHOICES
    },
    {
      kind: 'select',
      id: 'vatOpennessBlend',
      label: 'Openness blend mode',
      group: 'VAT blend (GPUReliefBlend)',
      apply: 'param',
      default: 'preset',
      disabledWhen: isProduct('vat'),
      help: 'How the positive-openness layer combines with the layers below it.',
      options: BLEND_CHOICES
    },
    {
      kind: 'select',
      id: 'vatSkyViewBlend',
      label: 'Sky-view blend mode',
      group: 'VAT blend (GPUReliefBlend)',
      apply: 'param',
      default: 'preset',
      disabledWhen: isProduct('vat'),
      help: 'How the sky-view layer combines with the layers below it.',
      options: BLEND_CHOICES
    },
    {
      kind: 'select',
      id: 'product',
      label: 'Visualization',
      group: 'Display',
      apply: 'param',
      default: 'hillshade',
      help: 'Each technique is a stage of compiled graphs built the first time you pick it. Stages are shared: the Swiss relief reuses the horizon, texture and curvature stages.',
      options: [
        {value: 'hillshade', label: 'Hillshade: one light'},
        {value: 'mdow', label: 'Multidirectional hillshade (USGS MDOW)'},
        {value: 'texture', label: 'Texture shading (Leland Brown)'},
        {value: 'sky-view', label: 'Sky-view factor'},
        {value: 'anisotropic', label: 'Anisotropic sky-view factor'},
        {value: 'positive-openness', label: 'Positive openness'},
        {value: 'negative-openness', label: 'Negative openness'},
        {value: 'imhof', label: 'Swiss / Imhof relief (blended, tinted)'},
        {value: 'slrm', label: 'Simple local relief model (SLRM)'},
        {value: 'msrm', label: 'Multi-scale relief model (MSRM)'},
        {value: 'local-dominance', label: 'Local dominance'},
        {value: 'vat', label: 'VAT blend (hillshade + slope + openness + SVF)'}
      ]
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp',
      group: 'Display',
      apply: 'param',
      default: 'grayscale',
      disabledWhen: state => !(state.product in SCALAR_PRODUCTS),
      help: 'Ramp of the scalar products. Archaeologists use grayscale; a perceptual ramp separates close values better.',
      options: [
        {value: 'grayscale', label: 'Grayscale (RVT convention)'},
        {value: 'viridis', label: 'Viridis'},
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'cividis', label: 'Cividis'}
      ]
    },
    {
      kind: 'toggle',
      id: 'autoStretch',
      label: 'Percent-clip stretch',
      group: 'Display',
      apply: 'param',
      default: true,
      disabledWhen: state => !(state.product in SCALAR_PRODUCTS),
      help: 'Stretch the color ramp between two percentiles of the displayed raster, measured from a GPU histogram readback after it settles. This is what RVT does by default. Turn it off for a fixed physical range.'
    },
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
      disabledWhen: state => !(state.product in SCALAR_PRODUCTS) || !state.autoStretch,
      help: 'Share of cells allowed to saturate at each end of the ramp. More clipping gives more contrast in the middle.'
    },
    {
      kind: 'slider',
      id: 'rangeScale',
      label: 'Fixed range scale',
      group: 'Display',
      apply: 'param',
      min: 0.25,
      max: 3,
      step: 0.05,
      default: 1,
      unit: 'x',
      disabledWhen: state => !(state.product in SCALAR_PRODUCTS) || state.autoStretch,
      help: 'Scales the fixed value range of the ramp when the percent-clip stretch is off.'
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
      help: 'Lower it to compare with the basemap.'
    },
    {
      kind: 'button',
      id: 'measure',
      label: 'Time this visualization',
      group: 'Under the hood',
      help: 'Runs the graphs of the displayed visualization (and the stages it reads) outside the frame and sums their GPU time.'
    }
  ],

  readouts: [
    {id: 'grid', label: 'Raster'},
    {
      id: 'stretch',
      label: 'Color stretch',
      help: 'The value range the ramp spans: percentile-clipped when the stretch is on.'
    },
    {id: 'median', label: 'Median value'},
    {id: 'extent', label: 'Value range'},
    {id: 'msrmRadii', label: 'MSRM mean-filter radii'},
    {id: 'dominanceTaps', label: 'Local dominance samples per pixel'},
    {id: 'timing', label: 'GPU time'}
  ],

  legends: state => {
    const {product} = state;
    if (product === 'hillshade' || product === 'mdow') {
      return [
        {
          kind: 'ramp',
          title: 'Hillshade',
          ramp: 'grayscale',
          extent: [0, 1],
          labels: ['shaded', 'lit']
        }
      ];
    }
    if (product === 'imhof') {
      return state.elevationTint
        ? [
            {
              kind: 'categories',
              title: 'Elevation tint (shading multiplies it)',
              entries: ALPINE_ELEVATION_STOPS.map(stop => ({
                color: [
                  Math.round(stop.color[0] * 255),
                  Math.round(stop.color[1] * 255),
                  Math.round(stop.color[2] * 255)
                ],
                label: `${stop.elevation.toLocaleString('en-US')} m`
              })),
              note: 'Brightness is the blended hillshade, sky-view, texture and curvature terms; warm and cool tints mark lit and shaded slopes.'
            }
          ]
        : [];
    }
    if (product === 'vat') {
      return [
        {
          kind: 'categories',
          title: 'VAT layers, bottom to top',
          entries: [
            {color: [210, 210, 210], label: 'Hillshade (normal, 100 %)'},
            {color: [150, 150, 150], label: 'Slope, inverted (luminosity)'},
            {color: [105, 105, 105], label: 'Positive openness (overlay)'},
            {color: [60, 60, 60], label: 'Sky-view factor (multiply)'}
          ],
          note: 'The swatches are only a key to the layer order; the blend itself is grayscale.'
        }
      ];
    }
    const spec = SCALAR_PRODUCTS[product];
    if (!spec) return [];
    return [
      {
        kind: 'ramp',
        id: 'stretch',
        title: spec.label,
        ramp: state.ramp,
        extent: state.autoStretch
          ? 'gpu'
          : [spec.low * state.rangeScale, spec.high * state.rangeScale],
        unit: spec.unit,
        labels: product === 'positive-openness' ? ['enclosed', 'exposed'] : undefined,
        format: value => (Math.abs(value) >= 100 ? value.toFixed(0) : value.toPrecision(3))
      }
    ];
  },

  story: [
    {
      id: 'question',
      controls: ['lightAzimuth', 'lightAltitude'],
      readouts: [],
      title: 'Which drawing shows what is really there?',
      body: 'The same 13.6 km of Swiss Alps can be drawn a dozen ways, and each one answers a different question. A glacier geologist wants moraines and crevasse fields, a mountaineer wants the faces, a cartographer wants a map you can read at a glance. This scene computes them all on the GPU from one elevation raster.\n\nStart with the baseline: a **hillshade** lit from the north-west by **`GPUReliefShading`**. It is the most familiar relief image, and it has a blind spot: any face whose normal is perpendicular to the light looks flat, and slopes facing the light vanish into a uniform bright tone. Turn **Light azimuth** and **Light altitude** below and watch which faces disappear.',
      options: {product: 'hillshade', lightAzimuth: 315, lightAltitude: 35},
      camera: {longitude: 7.765, latitude: 45.975, zoom: 12.6, transitionMs: 1400},
      callout: {coordinate: [7.8, 45.975], text: 'Gorner glacier'}
    },
    {
      id: 'multidirectional',
      controls: ['product'],
      readouts: [],
      title: 'Light it from everywhere at once',
      body: 'With one light, turning the azimuth made some ridges disappear while others popped out. The multidirectional hillshade fixes that by combining several lights, each weighted by how well it faces the slope: the weight is `sin^2(aspect - azimuth)` (the USGS "MDOW" scheme, the same as GDAL `-multidirectional`).\n\nThere is no single azimuth to turn here, so compare the two by switching **Visualization** below between *Hillshade: one light* and *Multidirectional hillshade (USGS MDOW)*. Notice the faces that were washed out now carry texture, at the cost of a flatter, lower-contrast look. Everything here is a parameter-buffer write: the graph is the same one.',
      options: {product: 'mdow'}
    },
    {
      id: 'texture',
      controls: ['textureDetail', 'textureLevels'],
      readouts: ['stretch'],
      title: 'Texture shading: detail without a light',
      body: '**`GPUTextureShading`** (after Leland Brown) takes a different route: it emphasises terrain texture at every scale by weighting the elevation spectrum with `|f|^alpha`. It is computed as a cascade of Gaussian blurs of doubling width whose differences are summed, so ridges are bright and drainage lines dark whatever direction they run, and there is no light to choose.\n\nSlide **Detail (alpha)** below: low values favour the large landforms, high values the fine rock texture. Add **Cascade levels** to reach farther. Colors are stretched between the 2nd and 98th percentile of the result, read back from a GPU histogram (see the **Color stretch** readout).',
      options: {product: 'texture', textureDetail: 0.5},
      camera: {longitude: 7.7, latitude: 45.98, zoom: 12.9, transitionMs: 1400}
    },
    {
      id: 'sky-view',
      controls: ['product', 'horizonRadius'],
      readouts: [],
      title: 'How much sky can you see? Sky-view and openness',
      body: '**`GPUTerrainHorizon`** marches rays in 16 compass directions from every pixel and records the horizon angle. The **sky-view factor** is the fraction of the sky dome that is open, `1 - mean(sin(horizon))`: 1 on a summit, much lower in a gorge or the bottom of a cirque. **Openness** averages the horizon angles themselves; the negative form works on the inverted terrain.\n\nIt needs no light and no direction, which is why archaeologists use it. Switch **Visualization** between *Sky-view factor*, *Positive openness* and *Negative openness*, then raise the **Search radius** to let the ray reach the far valley walls. Try *Anisotropic sky-view factor*: it favours one direction, so it recovers some of the directional cue that hillshade gave.',
      options: {product: 'sky-view', horizonRadius: '96'},
      camera: {longitude: 7.74, latitude: 45.99, zoom: 12.9, transitionMs: 1400}
    },
    {
      id: 'imhof',
      controls: ['imhofSwing', 'contrastStrength', 'curvatureStrength'],
      readouts: [],
      title: 'Put it together: the Swiss relief',
      body: 'Swiss mapmakers (Eduard Imhof) blend several cues: a hillshade whose light swings toward each slope, a darkening in sheltered ground, a rock-texture term, and colors that change with height. **`GPUReliefShading`** composes exactly that from the hillshade, the sky-view factor, the texture shade and a curvature term, with a warm/cool tint and an elevation ramp.\n\nTune the blend below: **Imhof swing** turns the light toward the slope side, **Contrast with elevation** lifts rock against glacier, **Curvature strength** sharpens the ridge lines. The blend is one parameter write; the sky-view, texture and curvature stages only re-run when the elevation or their own options change.',
      options: {product: 'imhof'},
      camera: {longitude: 7.742, latitude: 45.985, zoom: 12.1, transitionMs: 1400}
    },
    {
      id: 'rvt',
      controls: ['product', 'slrmRadius'],
      readouts: [],
      title: 'Micro-relief: what is bumpy at a given scale?',
      body: 'The Relief Visualization Toolbox (RVT) techniques were built for lidar and archaeology, where the question is "what is a small bump on a gentle surface?". **`GPUSimpleLocalRelief`** subtracts the local mean: `z - mean_r(z)` over a window, so features smaller than the window stand out (moraine ridges, rock glacier lobes, avalanche tracks). **`GPUMultiScaleRelief`** repeats that over a range of window sizes; **`GPULocalDominance`** asks how far an observer standing above each cell would see it dominate its surroundings.\n\nSwitch between the three with the **Visualization** selector. On a 6.6 m alpine DEM they bring out the ridges and flow structure on the glacier surfaces and the lateral moraines beside them; on 0.5 m lidar they would reveal ruined walls. Change **SLRM window radius** below to move the scale.',
      options: {product: 'slrm', slrmRadius: 20},
      camera: {longitude: 7.785, latitude: 45.97, zoom: 13.0, transitionMs: 1500}
    },
    {
      id: 'vat',
      controls: ['vatPreset', 'vatOpennessBlend', 'zFactor'],
      readouts: [],
      title: 'Blend them: VAT, and what to watch for',
      body: '**`GPUReliefBlend`** implements the Visualization for Archaeological Topography: up to five rasters, each stretched between a minimum and maximum (and optionally inverted), stacked with Photoshop-style blend modes. The preset uses hillshade at the bottom, then inverted slope (luminosity), positive openness (overlay) and sky-view (multiply), so steep ground darkens and open ground brightens without losing the shading.\n\n**Caveats.** Every technique here shows a property of the surface, not the truth: sky-view and openness depend on the search radius; local relief depends on the window; none of them is a substitute for an orthophoto. The percent-clip stretch is recomputed per view of the data, so two images are comparable only with a fixed range. **Try:** switch **VAT preset** to *Flat terrain*, change **Openness blend mode** to *Soft light*, or raise **Vertical exaggeration**.',
      options: {product: 'vat', vatPreset: 'archaeological'},
      camera: {longitude: 7.742, latitude: 45.985, zoom: 12.1, transitionMs: 1400}
    }
  ],

  about: {
    what: '`GPUReliefShading` blends hillshade, sky-view factor, texture shade, curvature and elevation tints into a Swiss-style relief and offers single, multidirectional (USGS MDOW) and Imhof-swing lights. `GPUTextureShading` approximates Leland Brown texture shading with a Gaussian cascade. `GPUTerrainHorizon` gives horizon angles, sky-view factor and positive and negative openness (march or exact sweep). `GPUSimpleLocalRelief`, `GPUMultiScaleRelief` and `GPULocalDominance` follow the Relief Visualization Toolbox; `GPUReliefBlend` is the VAT layer blend.',
    why: 'Elevation is only legible after it has been turned into light. The right visualization depends on the question: faces, drainage, micro-relief or a finished map. Computing them on the GPU makes the light, the scale and the blend interactive.',
    howToRead:
      'Hillshade and MDOW: bright faces turn toward the light. Texture shade: bright is a convex bump, dark is a groove. Sky-view and openness: dark is enclosed, bright is exposed. SLRM and MSRM: bright is higher than its surroundings at that scale. Local dominance: bright is a feature that rises above the ring around it. VAT and Swiss: read as a shaded map; the legend shows the tint stops. The reference implementations are RVT-py (SLRM, MSRM, local dominance, sky-view, openness, VAT), GDAL (multidirectional hillshade) and mt-image (Imhof swing).'
  },

  create: async ctx =>
    (await import('./relief-visualization.compute')).createReliefVisualization(ctx),

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUTerrainHorizon, GPUTextureShading, GPUReliefShading, GPUReliefBlend, GPUSimpleLocalRelief,
  getGPUReliefShadingParameterValues, getGPUTerrainHorizonParameterValues,
  GPU_RELIEF_BLEND_VAT_ARCHAEOLOGICAL
} from '@luma.gl/experimental/gpu-terrain';

const cell = {cellSize: [9.5546, 9.5546], northEdge: 0.3556, southEdge: 0.3112}; // Web Mercator tile
const graph = new GPUCommandGraph(device, {id: 'relief'});
graph.add(new GPUTerrainHorizon({
  width, height, elevation, settings: horizonSettings.importToGraph(graph),
  directionCount: ${state.horizonDirections}, maximumRadius: ${state.horizonRadius}, algorithm: '${state.horizonAlgorithm}',
  cellSizeMode: 'web-mercator', skyViewFactor, positiveOpenness
}));
graph.add(new GPUTextureShading({
  width, height, elevation, settings: textureSettings.importToGraph(graph),
  levelCount: ${state.textureLevels}, baseSigma: ${state.textureBaseSigma}, textureShade
}));
graph.add(new GPUReliefShading({
  width, height, elevation, settings: reliefSettings.importToGraph(graph),
  skyViewFactor, textureShade, curvature, imhofSwing: true, cellSizeMode: 'web-mercator', color
}));
graph.add(new GPUSimpleLocalRelief({
  width, height, elevation, settings: slrmSettings.importToGraph(graph), radius: ${state.slrmRadius}, relief
}));
graph.add(new GPUReliefBlend({
  width, height, layers: [hillshade, slope, positiveOpenness, skyViewFactor],
  settings: blendSettings.importToGraph(graph), color: blendColor
}));
const compiled = graph.compile(); // once

// per frame / on input: parameter writes only
reliefSettings.write(getGPUReliefShadingParameterValues({
  ...cell, lights: [{azimuthDegrees: ${state.lightAzimuth}, altitudeDegrees: ${state.lightAltitude}}],
  lightWeighting: '${state.swissLight}', imhofSwingDegrees: ${state.imhofSwing},
  skyViewStrength: ${state.skyViewStrength}, textureShadeStrength: ${state.textureStrength}, exposure: ${state.exposure}
}));
compiled.encode(commandEncoder, {parameters: undefined});`
});
