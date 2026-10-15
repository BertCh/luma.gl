// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RasterOperatorTable} from '../../../gpu-raster/computation-pack/types';

const CELL = {kind: 'cell', label: 'grid cell', indexSpace: '2d'} as const;
const ILLUMINATION_DIRECTORY = 'modules/experimental/src/gpu-terrain/terrain-illumination';
const CELL_SIZE_CHOICE = {
  stage: 'support',
  name: 'cellSizeMode',
  alternatives: ['uniform', 'web-mercator', 'geographic'],
  sweepable: false
} as const;

/** Terrain pack entries of the `illumination` family, keyed by exported class name. */
export const TERRAIN_ILLUMINATION_OPERATORS: RasterOperatorTable = {
  GPUTerrainHorizon: {
    family: 'illumination',
    descriptor: {scope: 'focal', reduction: 'select'},
    variables: {
      dem: 'elevation',
      horizonSkyViewFactor: 'fraction',
      horizonPositiveOpenness: 'rate'
    },
    stages: ['support', 'neighbourhood', 'operator'],
    datumKinds: [CELL],
    neighbourhood: 'ray',
    guards: ['metric-cell-size'],
    choices: [
      CELL_SIZE_CHOICE,
      {stage: 'neighbourhood', name: 'directionCount', sweepable: true},
      {stage: 'neighbourhood', name: 'maximumRadius', unit: 'cell', sweepable: true},
      {stage: 'operator', name: 'zFactor', label: 'z factor', sweepable: true},
      {stage: 'operator', name: 'curvatureCoefficient', unit: '1/m', sweepable: true}
    ],
    // Ray march as implemented: per sector the largest elevation tangent over the baked step
    // schedule (bilinear samples, curvature drop c d^2, 0 when no sample is valid), then the sky
    // view factor is one minus the mean sine of the non-negative horizon angles.
    formula: {
      id: 'gpu-terrain.terrain-horizon.sky-view',
      tex: String.raw`\term{skyView}{S} = 1 - \frac{1}{\term{directionCount}{N}}\sum_{k=0}^{N-1}\sin\max(\term{horizonAngle}{h_k},\,0),\quad h_k = \arctan\max_{s}\frac{\term{zFactor}{\zeta}\,(z_s - z_0) - \term{curvature}{c}\,\term{groundDistance}{d_s}^2}{d_s}`,
      ascii:
        'skyView = 1 - mean over N sectors of sin(max(h, 0)); h = atan(max over steps s of (zFactor * (z_s - z_0) - c * d_s^2) / d_s) in degrees, d_s = ground distance of step s; 0 when no sample is valid, NaN when the centre is nodata; the ray stops at the grid edge or maximumDistance',
      derivation: 'curated',
      symbols: [
        {id: 'skyView', role: 'output', label: 'S', binding: 'wgsl:skyView'},
        {id: 'directionCount', role: 'parameter', label: 'N', binding: 'wgsl:DIRECTION_COUNT'},
        {id: 'horizonAngle', role: 'intermediate', label: 'h_k', binding: 'wgsl:horizonAngle'},
        {id: 'zFactor', role: 'parameter', label: 'ζ', binding: 'wgsl:zFactor'},
        {id: 'curvature', role: 'parameter', label: 'c', binding: 'wgsl:curvature'},
        {id: 'groundDistance', role: 'intermediate', label: 'd_s', binding: 'wgsl:groundDistance'}
      ]
    },
    sources: [`${ILLUMINATION_DIRECTORY}/gpu-terrain-horizon.ts`]
  },
  GPUSolarShadowMask: {
    family: 'illumination',
    descriptor: {scope: 'local', reduction: 'none'},
    variables: {dem: 'elevation', sunShadowVisibility: 'fraction'},
    stages: ['support', 'operator', 'presentation'],
    datumKinds: [CELL],
    neighbourhood: 'none',
    guards: [],
    choices: [
      {stage: 'operator', name: 'directionCount', sweepable: true},
      {stage: 'operator', name: 'azimuthDegrees', unit: 'deg', sweepable: true},
      {stage: 'operator', name: 'altitudeDegrees', unit: 'deg', sweepable: true},
      {stage: 'operator', name: 'angularRadiusDegrees', unit: 'deg', sweepable: true}
    ],
    // Reads the horizon map two sectors at the sun azimuth (linear blend, wraparound), then the
    // area fraction of the solar disk above that horizon.
    formula: {
      id: 'gpu-terrain.solar-shadow-mask.disk-visibility',
      tex: String.raw`\term{visibility}{V} = 1 - \frac{\arccos \term{diskOffset}{u} - u\sqrt{1 - u^2}}{\pi},\quad u = \operatorname{clip}\!\left(\frac{\term{altitude}{\alpha} - \term{horizonAngle}{h}}{\term{radius}{\rho}},\,-1,\,1\right)`,
      ascii:
        'V = 1 - (acos(u) - u * sqrt(1 - u^2)) / pi; u = clamp((altitude - h) / radius, -1, 1); h = horizon blended linearly between the two sectors around the sun azimuth; radius 0 gives the step V = altitude > h; V = 0 when altitude + radius <= 0; NaN where the horizon is NaN',
      derivation: 'curated',
      symbols: [
        {id: 'visibility', role: 'output', label: 'V', binding: 'wgsl:visibilityValues'},
        {id: 'diskOffset', role: 'intermediate', label: 'u', binding: 'wgsl:u'},
        {id: 'altitude', role: 'parameter', label: 'α', binding: 'wgsl:altitudeDegrees'},
        {id: 'horizonAngle', role: 'input', label: 'h', binding: 'wgsl:horizonAngle'},
        {id: 'radius', role: 'parameter', label: 'ρ', binding: 'wgsl:radiusDegrees'}
      ]
    },
    sources: [
      `${ILLUMINATION_DIRECTORY}/gpu-solar-shadow-mask.ts`,
      `${ILLUMINATION_DIRECTORY}/terrain-illumination-utils.ts`
    ]
  },
  GPUTerrainCastShadow: {
    family: 'illumination',
    descriptor: {scope: 'focal', reduction: 'select'},
    variables: {
      dem: 'elevation',
      castShadowVisibility: 'fraction',
      castShadowHorizon: 'rate'
    },
    stages: ['support', 'neighbourhood', 'operator'],
    datumKinds: [CELL],
    neighbourhood: 'ray',
    guards: ['metric-cell-size'],
    choices: [
      CELL_SIZE_CHOICE,
      {stage: 'neighbourhood', name: 'maximumRadius', unit: 'cell', sweepable: true},
      {stage: 'operator', name: 'azimuthDegrees', unit: 'deg', sweepable: true},
      {stage: 'operator', name: 'altitudeDegrees', unit: 'deg', sweepable: true},
      {stage: 'operator', name: 'angularRadiusDegrees', unit: 'deg', sweepable: true}
    ],
    // One hull sweep along the digital line family of the sun azimuth gives the horizon toward the
    // sun; the visibility is the same solar-disk fraction as GPUSolarShadowMask.
    formula: {
      id: 'gpu-terrain.terrain-cast-shadow.sun-horizon',
      tex: String.raw`\term{horizonAngle}{h} = \arctan\max_{k}\frac{\term{rise}{\zeta (z_k - z_0) - c\,d_k^2}}{\term{groundDistance}{d_k}},\quad \term{visibility}{V} = 1 - \frac{\arccos u - u\sqrt{1 - u^2}}{\pi},\ u = \operatorname{clip}\!\left(\frac{\term{altitude}{\alpha} - h}{\term{radius}{\rho}},\,-1,\,1\right)`,
      ascii:
        'h = atan(max over pixel centres k on the digital line toward the sun of rise_k / d_k), rise_k = zFactor * (z_k - z_0) - c * d_k^2, in degrees; V = 1 - (acos(u) - u * sqrt(1 - u^2)) / pi with u = clamp((altitude - h) / radius, -1, 1); radius 0 gives a step; V = 0 when altitude + radius <= 0; h = 0 with no valid sample; NaN at a nodata centre',
      derivation: 'curated',
      symbols: [
        {id: 'horizonAngle', role: 'intermediate', label: 'h', binding: 'wgsl:horizonAngle'},
        {id: 'rise', role: 'intermediate', label: 'rise', binding: 'wgsl:rise'},
        {id: 'groundDistance', role: 'intermediate', label: 'd_k', binding: 'wgsl:groundDistance'},
        {id: 'visibility', role: 'output', label: 'V', binding: 'wgsl:visibilityValues'},
        {id: 'altitude', role: 'parameter', label: 'α', binding: 'wgsl:altitudeDegrees'},
        {id: 'radius', role: 'parameter', label: 'ρ', binding: 'wgsl:radiusDegrees'}
      ]
    },
    sources: [
      `${ILLUMINATION_DIRECTORY}/gpu-terrain-cast-shadow.ts`,
      `${ILLUMINATION_DIRECTORY}/terrain-horizon-sweep.ts`
    ]
  },
  GPUSolarIrradiance: {
    family: 'illumination',
    descriptor: {scope: 'local', reduction: 'sum'},
    variables: {
      dem: 'elevation',
      irradianceSunHours: 'measure',
      irradianceInsolation: 'measure'
    },
    stages: ['support', 'operator', 'presentation'],
    datumKinds: [CELL],
    neighbourhood: 'none',
    guards: [],
    choices: [
      {stage: 'operator', name: 'directionCount', sweepable: true},
      {stage: 'operator', name: 'stepMinutes', unit: 'min', sweepable: true},
      {stage: 'operator', name: 'angularRadiusDegrees', unit: 'deg', sweepable: true},
      {stage: 'operator', name: 'diffuseIrradiance', unit: 'W/m²', sweepable: true}
    ],
    // One thread per pixel loops over the sun table: disk visibility at the row's azimuth, then
    // the direct term DNI * V * cos(incidence) * duration and the optional diffuse term.
    formula: {
      id: 'gpu-terrain.solar-irradiance.day-sum',
      tex: String.raw`\term{sunHours}{T_{sun}} = \sum_{s} V_s\,\term{duration}{\Delta t_s},\quad \term{insolation}{E} = \sum_{s} \left[\term{directNormal}{I_s}\,V_s\max(\term{cosIncidence}{\cos\theta_s},\,0) + \term{diffuse}{D}\,\term{skyViewFactor}{S}\,[\alpha_s > 0]\right]\Delta t_s`,
      ascii:
        'sunHours = sum over table rows of V * duration; insolation = sum of DNI * V * max(cosIncidence, 0) * duration (+ diffuse * svf * duration while altitude > 0), in Wh/m^2; rows with altitude + radius <= 0 are skipped; V = solar-disk visibility over the horizon at the row azimuth; cosIncidence = sin(altitude) without slope and aspect or on flat ground, else cos(slope) sin(altitude) + sin(slope) cos(altitude) cos(azimuth - aspect); NaN where the horizon is NaN',
      derivation: 'curated',
      symbols: [
        {id: 'sunHours', role: 'output', label: 'T_sun', binding: 'wgsl:hoursValues'},
        {id: 'insolation', role: 'output', label: 'E', binding: 'wgsl:insolationValues'},
        {id: 'duration', role: 'input', label: 'Δt_s', binding: 'wgsl:duration'},
        {id: 'directNormal', role: 'input', label: 'I_s', binding: 'wgsl:sunTable'},
        {id: 'cosIncidence', role: 'intermediate', label: 'cos θ_s', binding: 'wgsl:cosIncidence'},
        {id: 'diffuse', role: 'parameter', label: 'D', binding: 'wgsl:diffuseIrradiance'},
        {id: 'skyViewFactor', role: 'input', label: 'S', binding: 'wgsl:skyViewFactor'}
      ]
    },
    sources: [
      `${ILLUMINATION_DIRECTORY}/gpu-solar-irradiance.ts`,
      `${ILLUMINATION_DIRECTORY}/terrain-illumination-utils.ts`
    ]
  },
  GPUReliefShading: {
    family: 'illumination',
    descriptor: {scope: 'focal', reduction: 'weighted-mean'},
    variables: {dem: 'elevation', reliefHillshade: 'fraction', reliefLuminance: 'fraction'},
    stages: ['support', 'neighbourhood', 'operator', 'presentation'],
    datumKinds: [CELL],
    neighbourhood: 'stencil-3x3',
    guards: ['metric-cell-size'],
    choices: [
      CELL_SIZE_CHOICE,
      {stage: 'operator', name: 'lights', sweepable: false},
      {
        stage: 'operator',
        name: 'lightWeighting',
        alternatives: ['fixed', 'aspect', 'imhof-swing'],
        sweepable: false
      },
      {stage: 'presentation', name: 'hillshadeStrength', sweepable: true},
      {stage: 'presentation', name: 'exposure', sweepable: true}
    ],
    // Horn normals (Sobel / 8 cell sizes), each light max(n . l, 0), weights sin^2(aspect - azimuth)
    // for the default MDOW set; the relief blend then mixes the hillshade and scales by exposure.
    formula: {
      id: 'gpu-terrain.relief-shading.multidirectional',
      tex: String.raw`\term{hillshade}{H} = \frac{\sum_i \term{weight}{w_i}\,\max(\mathbf n\cdot\mathbf l_i,\,0)}{\sum_i w_i},\ \ w_i = \sin^2(\term{aspect}{A} - \alpha_i),\ \ \mathbf n \propto (-\term{eastGradient}{g_e},\,-\term{northGradient}{g_n},\,1),\quad \term{relief}{R} = \operatorname{clip}\!\big(\term{exposure}{\varepsilon}\,(1 - \term{hillshadeStrength}{\eta} + \eta H),\,0,\,1\big)`,
      ascii:
        'g_e = zFactor * Gx / (8 dx); g_n = northSign * zFactor * Gy / (8 dy); n = normalize(-g_e, -g_n, 1); aspect = atan2(-g_e, -g_n); hillshade = sum(w_i * max(n . l_i, 0)) / sum(w_i) with w_i = sin^2(aspect - azimuth_i) (1 on flat cells) for aspect weighting, the lights own weights for fixed; relief = clamp(exposure * (1 - eta + eta * hillshade), 0, 1) without sky-view, texture or curvature inputs; NaN where the 3x3 window is invalid',
      derivation: 'curated',
      symbols: [
        {id: 'hillshade', role: 'intermediate', label: 'H', binding: 'wgsl:hillshadeValues'},
        {id: 'weight', role: 'intermediate', label: 'w_i', binding: 'wgsl:weight'},
        {id: 'aspect', role: 'intermediate', label: 'A', binding: 'wgsl:aspect'},
        {id: 'eastGradient', role: 'intermediate', label: 'g_e', binding: 'wgsl:eastGradient'},
        {id: 'northGradient', role: 'intermediate', label: 'g_n', binding: 'wgsl:northGradient'},
        {id: 'relief', role: 'output', label: 'R', binding: 'wgsl:reliefValues'},
        {id: 'exposure', role: 'parameter', label: 'ε', binding: 'wgsl:exposure'},
        {id: 'hillshadeStrength', role: 'parameter', label: 'η', binding: 'wgsl:hillshadeStrength'}
      ]
    },
    sources: [`${ILLUMINATION_DIRECTORY}/gpu-relief-shading.ts`]
  },
  GPUTextureShading: {
    family: 'illumination',
    descriptor: {scope: 'focal', reduction: 'weighted-mean'},
    variables: {dem: 'elevation', textureShade: 'score'},
    stages: ['support', 'neighbourhood', 'operator', 'presentation'],
    datumKinds: [CELL],
    neighbourhood: 'window',
    guards: [],
    choices: [
      {stage: 'neighbourhood', name: 'levelCount', sweepable: true},
      {stage: 'neighbourhood', name: 'baseSigma', unit: 'cell', sweepable: true},
      {stage: 'operator', name: 'detail', sweepable: true},
      {stage: 'presentation', name: 'gain', sweepable: true}
    ],
    // Brown's fractional Laplacian |f|^alpha as a Gaussian band-pass pyramid: bands G_(k-1) - G_k
    // of the normalized blur, weighted 2^(-k alpha); the sum is scaled by the gain.
    formula: {
      id: 'gpu-terrain.texture-shading.band-pyramid',
      tex: String.raw`\term{textureShade}{T} = \term{gain}{g}\sum_{k=0}^{K-1}\term{levelWeights}{w_k}\,\big(\term{previousValue}{G_{k-1}} - \term{currentValue}{G_k}\big),\quad w_k = 2^{-k\,\term{detail}{\alpha}},\ \ G_{-1} = z,\ \ G_k = z * \mathcal{N}\!\left(0,\,\sigma_k^2\right),\ \ \sigma_k = \term{baseSigma}{\sigma_0}\,2^k`,
      ascii:
        'T = gain * sum over levels k of w_k * (G_(k-1) - G_k); G_-1 = elevation, G_k = Gaussian blur of the elevation with sigma_k = baseSigma * 2^k (separable cascade, edge pixels replicated, nodata by normalized convolution), w_k = 2^(-k * detail); output in metres times gain, NaN at an invalid centre',
      derivation: 'curated',
      symbols: [
        {id: 'textureShade', role: 'output', label: 'T', binding: 'wgsl:textureShade'},
        {id: 'gain', role: 'parameter', label: 'g', binding: 'wgsl:gain'},
        {id: 'levelWeights', role: 'parameter', label: 'w_k', binding: 'wgsl:levelWeights'},
        {
          id: 'previousValue',
          role: 'intermediate',
          label: 'G_(k-1)',
          binding: 'wgsl:previousValue'
        },
        {id: 'currentValue', role: 'intermediate', label: 'G_k', binding: 'wgsl:currentValue'},
        {id: 'detail', role: 'parameter', label: 'α', binding: 'wgsl:detail'},
        {id: 'baseSigma', role: 'parameter', label: 'σ_0', binding: 'wgsl:baseSigma'}
      ]
    },
    sources: [`${ILLUMINATION_DIRECTORY}/gpu-texture-shading.ts`]
  },
  GPUSolarPosition: {
    family: 'illumination',
    descriptor: {scope: 'local', reduction: 'none'},
    variables: {
      solarSites: 'measure',
      solarAzimuth: 'direction',
      solarAltitude: 'rate',
      solarDaylight: 'category'
    },
    stages: ['support', 'operator'],
    datumKinds: [{kind: 'site', label: 'site', indexSpace: '1d'}],
    neighbourhood: 'none',
    guards: [],
    choices: [
      {stage: 'operator', name: 'timestamp', sweepable: true},
      {stage: 'operator', name: 'refraction', alternatives: [true, false], sweepable: false},
      {stage: 'operator', name: 'daylightAltitudeDegrees', unit: 'deg', sweepable: true}
    ],
    // NOAA solar calculator (Meinel / Meeus low precision) in float32: the declination and the
    // equation of time come from the Julian-century series; the rest is spherical trigonometry.
    formula: {
      id: 'gpu-terrain.solar-position.noaa',
      tex: String.raw`\term{hourAngle}{H} = \frac{(\term{minutesUTC}{m} + \term{equationOfTime}{E} + 4\,\term{longitude}{\lambda}) \bmod 1440}{4} - 180^\circ,\quad \sin \term{geometricAltitude}{\alpha_g} = \sin\term{latitude}{\varphi}\,\sin\term{declination}{\delta} + \cos\varphi\cos\delta\cos H`,
      ascii:
        'H = ((minutesUTC + E + 4 * longitude) mod 1440) / 4 - 180; sin(alt_geometric) = sin(lat) sin(dec) + cos(lat) cos(dec) cos(H); azimuth = (atan2(sin H, cos H sin(lat) - tan(dec) cos(lat)) + 180) mod 360, clockwise from north; altitude = alt_geometric + NOAA refraction (default on); daylight = alt_geometric > threshold (-0.833); NaN azimuth and altitude and daylight 0 for non-finite coordinates or |lat| > 90; dec and E (equation of time) from the NOAA Julian-century series',
      derivation: 'curated',
      symbols: [
        {id: 'hourAngle', role: 'intermediate', label: 'H', binding: 'wgsl:hourAngleDegrees'},
        {id: 'minutesUTC', role: 'input', label: 'm', binding: 'wgsl:minutesUTC'},
        {id: 'equationOfTime', role: 'intermediate', label: 'E', binding: 'wgsl:equationOfTime'},
        {id: 'longitude', role: 'input', label: 'λ', binding: 'wgsl:longitude'},
        {id: 'geometricAltitude', role: 'output', label: 'α_g', binding: 'wgsl:geometricAltitude'},
        {id: 'latitude', role: 'input', label: 'φ', binding: 'wgsl:latitude'},
        {id: 'declination', role: 'intermediate', label: 'δ', binding: 'wgsl:declination'}
      ]
    },
    sources: [`${ILLUMINATION_DIRECTORY}/gpu-solar-position.ts`]
  }
};
