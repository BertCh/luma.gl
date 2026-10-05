// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Relief tools: the round-8 terrain illumination and relief visualization contributors on the San
 * Francisco elevation raster, one product on screen at a time.
 *
 * The mode is a small dependency graph of lazily compiled stages, each a `GPUCommandGraph` over
 * caller-owned buffers. A stage is encoded only when the selected product needs it and one of its
 * inputs changed:
 * - `setup`: `GPUTerrainDerivatives` (slope and aspect).
 * - `horizon`: `GPUTerrainHorizon` with the sky-view factor, anisotropic sky-view factor and
 *   positive and negative openness. The algorithm (`'march'` or `'sweep'`) and the horizon storage
 *   (`'float32'` or `'unorm16'`) are topology, so each combination is a separate compiled graph that
 *   is built the first time it is selected; the anisotropy direction is a parameter write.
 * - `shadow`: `GPUTerrainCastShadow`, a single-sun hull sweep. Azimuth, altitude and softness are
 *   parameter writes, so dragging the sun never recompiles.
 * - `irradiance`: `GPUSolarIrradiance` over a sun table sampled for the chosen date.
 * - `look`: `GPUTerrainCurvature` (multi-radius ring curvature) feeding `GPUReliefShading` with the
 *   Imhof azimuth swing and elevation-dependent contrast.
 * - `rvt`: `GPUSimpleLocalRelief`, `GPUMultiScaleRelief` and `GPULocalDominance`.
 * - `blend`: `GPUReliefBlend` with the VAT presets over hillshade, slope, openness and sky-view.
 *
 * "Measure GPU cost" times the stages and the march and sweep horizon at the chosen radius
 * (32 px to the full tile, compile-time like the algorithm); the sweep wins at large radii.
 */

import {
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPULocalDominanceParameterValues,
  getGPUMultiScaleReliefParameterValues,
  getGPUReliefBlendParameterValues,
  getGPUReliefShadingParameterValues,
  getGPUSimpleLocalReliefParameterValues,
  getGPUSolarIrradianceParameterValues,
  getGPUSolarIrradianceSunTable,
  getGPUTerrainCastShadowParameterValues,
  getGPUTerrainCurvatureParameterValues,
  getGPUTerrainDerivativesParameterValues,
  getGPUTerrainHorizonParameterValues,
  getSolarPosition,
  GPU_LOCAL_DOMINANCE_PARAMETER_LENGTH,
  GPU_MULTI_SCALE_RELIEF_PARAMETER_LENGTH,
  GPU_RELIEF_BLEND_PARAMETER_LENGTH,
  GPU_RELIEF_BLEND_VAT_ARCHAEOLOGICAL,
  GPU_RELIEF_BLEND_VAT_FLAT,
  GPU_RELIEF_SHADING_PARAMETER_LENGTH,
  GPU_SIMPLE_LOCAL_RELIEF_PARAMETER_LENGTH,
  GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES,
  GPU_SOLAR_IRRADIANCE_PARAMETER_LENGTH,
  GPU_SOLAR_IRRADIANCE_SUN_TABLE_STRIDE,
  GPU_TERRAIN_CAST_SHADOW_PARAMETER_LENGTH,
  GPU_TERRAIN_CURVATURE_PARAMETER_LENGTH,
  GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH,
  GPU_TERRAIN_HORIZON_ANISOTROPIC_PARAMETER_LENGTH,
  GPULocalDominance,
  GPUMultiScaleRelief,
  GPUReliefBlend,
  GPUReliefShading,
  GPUSimpleLocalRelief,
  GPUSolarIrradiance,
  GPUTerrainCastShadow,
  GPUTerrainCurvature,
  GPUTerrainDerivatives,
  GPUTerrainHorizon,
  type GPUReliefBlendLayerSettings,
  type GPUReliefShadingStop,
  type GPUTerrainHorizonAlgorithm,
  type GPUTerrainHorizonFormat
} from '@luma.gl/experimental/gpu-terrain';
import {importGraphBuffer} from '../graph-buffers';
import {LocalMetricProjection} from '../spatial-analysis-data';
import {
  SpatialAnalysisRasterLayer,
  type SpatialAnalysisRasterLayerProps
} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {ReliefRasterLayer} from './relief-layers';
import {
  formatCompiledGraphTiming,
  measureCompiledGraph,
  type CompiledGraphTiming
} from './vector-timing';

/** Compile-time horizon sectors. */
const DIRECTION_COUNT = 16;
/** Horizon search radius stops in pixels; `0` means the full tile. The sweep needs step growth 1. */
const HORIZON_RADIUS_STOPS = [32, 64, 128, 256, 0] as const;
const DEFAULT_HORIZON_RADIUS_STOP = 2;
/** Sun table step in minutes; one day is `24 * 60 / 10` rows. */
const SUN_TABLE_STEP_MINUTES = 10;
/** Sun table capacity (topology): one day of 10 minute rows. */
const SUN_TABLE_CAPACITY = (24 * 60) / SUN_TABLE_STEP_MINUTES;
/** San Francisco is UTC-8 in standard time. */
const LOCAL_UTC_OFFSET_HOURS = -8;
/** Year of the date slider. */
const YEAR = 2026;
/** Compile-time radius in pixels of the simple local relief mean filter. */
const LOCAL_RELIEF_RADIUS = 10;
/** Compile-time local dominance sampling ring, in pixels. */
const DOMINANCE_MINIMUM_RADIUS = 5;
const DOMINANCE_MAXIMUM_RADIUS = 15;
const DOMINANCE_RADIUS_INCREMENT = 2;
const DOMINANCE_ANGULAR_RESOLUTION = 15;
/** Compile-time multi-scale feature sizes, in cells, and the radius growth exponent. */
const MULTI_SCALE_FEATURE_MINIMUM_CELLS = 3;
const MULTI_SCALE_FEATURE_MAXIMUM_CELLS = 51;
const MULTI_SCALE_SCALING_FACTOR = 2;
/** Compile-time ring radii of the curvature that feeds the Imhof relief. */
const CURVATURE_RING_RADII = [2, 6] as const;

type Product =
  | 'sky-view'
  | 'anisotropic-sky-view'
  | 'positive-openness'
  | 'negative-openness'
  | 'cast-shadow'
  | 'sun-horizon-angle'
  | 'shadow-on-relief'
  | 'sun-hours'
  | 'insolation'
  | 'imhof'
  | 'ring-curvature'
  | 'simple-local-relief'
  | 'multi-scale-relief'
  | 'local-dominance'
  | 'vat';
type Stage = 'setup' | 'horizon' | 'shadow' | 'irradiance' | 'look' | 'rvt' | 'blend';
type LightModel = 'fixed' | 'mdow' | 'imhof-swing';
type VatPreset = 'archaeological' | 'flat' | 'hillshade-only';

/** Stages in encode order. */
const STAGE_ORDER: readonly Stage[] = [
  'setup',
  'horizon',
  'shadow',
  'irradiance',
  'look',
  'rvt',
  'blend'
];
/** Stages whose results change when a stage is re-encoded. */
const DOWNSTREAM_STAGES: Record<Stage, readonly Stage[]> = {
  setup: ['irradiance', 'blend'],
  horizon: ['irradiance', 'look', 'blend'],
  shadow: [],
  irradiance: [],
  look: ['blend'],
  rvt: [],
  blend: []
};
/** Stages each product reads, upstream included. */
const PRODUCT_STAGES: Record<Product, readonly Stage[]> = {
  'sky-view': ['horizon'],
  'anisotropic-sky-view': ['horizon'],
  'positive-openness': ['horizon'],
  'negative-openness': ['horizon'],
  'cast-shadow': ['shadow'],
  'sun-horizon-angle': ['shadow'],
  'shadow-on-relief': ['horizon', 'look', 'shadow'],
  'sun-hours': ['setup', 'horizon', 'irradiance'],
  insolation: ['setup', 'horizon', 'irradiance'],
  imhof: ['horizon', 'look'],
  'ring-curvature': ['look'],
  'simple-local-relief': ['rvt'],
  'multi-scale-relief': ['rvt'],
  'local-dominance': ['rvt'],
  vat: ['setup', 'horizon', 'look', 'blend']
};

const ELEVATION_STOPS: readonly GPUReliefShadingStop[] = [
  {elevation: 0, color: [0.4, 0.62, 0.36]},
  {elevation: 40, color: [0.62, 0.74, 0.42]},
  {elevation: 100, color: [0.86, 0.8, 0.5]},
  {elevation: 180, color: [0.85, 0.65, 0.45]},
  {elevation: 260, color: [0.95, 0.88, 0.8]}
];

/** Display style of every scalar product: raster style, legend note and one-line meaning. */
const SCALAR_STYLES: Partial<
  Record<
    Product,
    {
      colormap: 'viridis' | 'inferno' | 'grayscale';
      valueRange: readonly [number, number];
      description: string;
    }
  >
> = {
  'sky-view': {
    colormap: 'viridis',
    valueRange: [0.6, 1],
    description: 'Sky-view factor: share of the sky dome that is open (1 on flat open ground).'
  },
  'anisotropic-sky-view': {
    colormap: 'viridis',
    valueRange: [0.6, 1],
    description: 'Anisotropic sky-view factor: sky openness weighted toward one azimuth.'
  },
  'positive-openness': {
    colormap: 'grayscale',
    valueRange: [70, 93],
    description: 'Positive openness in degrees: bright on ridges and peaks, dark in hollows.'
  },
  'negative-openness': {
    colormap: 'grayscale',
    valueRange: [70, 93],
    description: 'Negative openness in degrees: bright in pits and valleys, dark on ridges.'
  },
  'cast-shadow': {
    colormap: 'grayscale',
    valueRange: [0, 1],
    description: 'Fraction of the solar disk above the terrain horizon toward the sun.'
  },
  'sun-horizon-angle': {
    colormap: 'inferno',
    valueRange: [-5, 35],
    description: 'Terrain horizon angle toward the sun in degrees.'
  },
  'sun-hours': {
    colormap: 'inferno',
    valueRange: [0, 16],
    description: 'Hours the solar disk is visible on the chosen date.'
  },
  insolation: {
    colormap: 'inferno',
    valueRange: [0, 10000],
    description: 'Clear-sky direct plus diffuse energy on the sloped surface, Wh/m^2 (0 to 10 kWh).'
  },
  'ring-curvature': {
    colormap: 'grayscale',
    valueRange: [-0.4, 0.4],
    description: 'Multi-radius ring curvature (convex bright, concave dark).'
  },
  'simple-local-relief': {
    colormap: 'grayscale',
    valueRange: [-8, 8],
    description: `Simple local relief: elevation minus its ${LOCAL_RELIEF_RADIUS} pixel local mean, meters.`
  },
  'multi-scale-relief': {
    colormap: 'grayscale',
    valueRange: [-20, 20],
    description: 'Multi-scale relief: mean difference of consecutive low-pass surfaces, meters.'
  },
  'local-dominance': {
    colormap: 'grayscale',
    valueRange: [0.3, 1.8],
    description: 'Local dominance: how far an observer stands above the ring of surrounding ground.'
  }
};

export const reliefVisualizationMode: SpatialAnalysisModeDefinition = {
  id: 'relief-visualization',
  title: 'Relief tools',
  contributors: [
    'GPUTerrainHorizon',
    'GPUTerrainCastShadow',
    'GPUSolarIrradiance',
    'GPUReliefShading',
    'GPUTerrainCurvature',
    'GPUTerrainDerivatives',
    'GPUSimpleLocalRelief',
    'GPUMultiScaleRelief',
    'GPULocalDominance',
    'GPUReliefBlend'
  ],
  description:
    'Round-8 relief tools over San Francisco: a sweep horizon (compare it with the ray march), ' +
    'sky-view, anisotropic sky-view and openness, single-sun cast shadows, sun hours and ' +
    'insolation for a date, Imhof relief with curvature and elevation contrast, and the RVT ' +
    'products with VAT blends.',
  initialViewState: {longitude: -122.44, latitude: 37.76, zoom: 11.6},

  async create(context) {
    const terrain = await context.data.getSanFranciscoTerrain();
    context.signal.throwIfAborted();
    const {device} = context;
    const {width, height, bounds, cellSize} = terrain;
    const pixelCount = width * height;
    const projection = new LocalMetricProjection(terrain.origin);
    const origin: [number, number, number] = [terrain.origin[0], terrain.origin[1], 0];
    const resources = new SpatialAnalysisResources(device, 'relief-visualization');
    const centerLongitudeLatitude = projection.unproject(
      (bounds[0] + bounds[2]) / 2,
      (bounds[1] + bounds[3]) / 2
    );

    // Sea (elevation 0) is invalid for every contributor.
    const validityValues = new Uint32Array(pixelCount);
    let landCount = 0;
    for (let index = 0; index < pixelCount; index++) {
      const isLand = terrain.elevation[index] > 0.5;
      validityValues[index] = isLand ? 1 : 0;
      if (isLand) landCount++;
    }

    // --- Buffers (shared by every stage graph) -------------------------------------------------
    const elevationBuffer = resources.createBuffer('elevation', terrain.elevation);
    const validityBuffer = resources.createBuffer('validity', validityValues);
    const slopeBuffer = resources.createBuffer('slope', pixelCount * 4);
    const aspectBuffer = resources.createBuffer('aspect', pixelCount * 4);
    const skyViewBuffer = resources.createBuffer('sky-view', pixelCount * 4);
    const anisotropicBuffer = resources.createBuffer('anisotropic-sky-view', pixelCount * 4);
    const positiveOpennessBuffer = resources.createBuffer('positive-openness', pixelCount * 4);
    const negativeOpennessBuffer = resources.createBuffer('negative-openness', pixelCount * 4);
    const sunVisibilityBuffer = resources.createBuffer('sun-visibility', pixelCount * 4);
    const sunHorizonBuffer = resources.createBuffer('sun-horizon', pixelCount * 4);
    const sunHoursBuffer = resources.createBuffer('sun-hours', pixelCount * 4);
    const insolationBuffer = resources.createBuffer('insolation', pixelCount * 4);
    const ringCurvatureBuffer = resources.createBuffer('ring-curvature', pixelCount * 4);
    const hillshadeBuffer = resources.createBuffer('hillshade', pixelCount * 4);
    const reliefColorBuffer = resources.createBuffer('relief-color', pixelCount * 4);
    const simpleReliefBuffer = resources.createBuffer('simple-local-relief', pixelCount * 4);
    const multiScaleBuffer = resources.createBuffer('multi-scale-relief', pixelCount * 4);
    const dominanceBuffer = resources.createBuffer('local-dominance', pixelCount * 4);
    const blendColorBuffer = resources.createBuffer('blend-color', pixelCount * 4);
    const sunTableBuffer = resources.createBuffer(
      'sun-table',
      SUN_TABLE_CAPACITY * GPU_SOLAR_IRRADIANCE_SUN_TABLE_STRIDE * 4
    );

    const derivativesSettings = resources.createParameterBuffer(
      'derivatives-settings',
      'float32',
      GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH
    );
    const horizonSettings = resources.createParameterBuffer(
      'horizon-settings',
      'float32',
      GPU_TERRAIN_HORIZON_ANISOTROPIC_PARAMETER_LENGTH
    );
    const shadowSettings = resources.createParameterBuffer(
      'shadow-settings',
      'float32',
      GPU_TERRAIN_CAST_SHADOW_PARAMETER_LENGTH
    );
    const irradianceSettings = resources.createParameterBuffer(
      'irradiance-settings',
      'float32',
      GPU_SOLAR_IRRADIANCE_PARAMETER_LENGTH
    );
    const curvatureSettings = resources.createParameterBuffer(
      'curvature-settings',
      'float32',
      GPU_TERRAIN_CURVATURE_PARAMETER_LENGTH
    );
    const reliefSettings = resources.createParameterBuffer(
      'relief-settings',
      'float32',
      GPU_RELIEF_SHADING_PARAMETER_LENGTH
    );
    const simpleReliefSettings = resources.createParameterBuffer(
      'simple-local-relief-settings',
      'float32',
      GPU_SIMPLE_LOCAL_RELIEF_PARAMETER_LENGTH
    );
    const multiScaleSettings = resources.createParameterBuffer(
      'multi-scale-relief-settings',
      'float32',
      GPU_MULTI_SCALE_RELIEF_PARAMETER_LENGTH
    );
    const dominanceSettings = resources.createParameterBuffer(
      'local-dominance-settings',
      'float32',
      GPU_LOCAL_DOMINANCE_PARAMETER_LENGTH
    );
    const blendSettings = resources.createParameterBuffer(
      'blend-settings',
      'float32',
      GPU_RELIEF_BLEND_PARAMETER_LENGTH
    );

    // Horizon maps are allocated the first time their format is selected.
    const horizonBuffers: Partial<Record<GPUTerrainHorizonFormat, ReturnType<typeof makeHorizon>>> =
      {};
    function makeHorizon(format: GPUTerrainHorizonFormat) {
      const elementCount = pixelCount * DIRECTION_COUNT;
      const byteLength = format === 'unorm16' ? Math.ceil(elementCount / 2) * 4 : elementCount * 4;
      return {
        buffer: resources.createBuffer(`horizon-${format}`, byteLength),
        byteLength,
        elementCount
      };
    }
    function getHorizonMap(format: GPUTerrainHorizonFormat) {
      let map = horizonBuffers[format];
      if (!map) {
        map = makeHorizon(format);
        horizonBuffers[format] = map;
      }
      return map;
    }

    // --- Graph construction --------------------------------------------------------------------
    const importElevation = (graph: GPUCommandGraph<void>) => ({
      id: 'elevation',
      format: 'float32' as const,
      storage: {
        kind: 'buffer' as const,
        values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', pixelCount)
      },
      validity: importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', pixelCount)
    });
    const importFloats = (
      graph: GPUCommandGraph<void>,
      id: string,
      buffer: typeof skyViewBuffer
    ): GraphDataView<'float32'> => importGraphBuffer(graph, id, buffer, 'float32', pixelCount);
    const importHorizon = (graph: GPUCommandGraph<void>, format: GPUTerrainHorizonFormat) => {
      const map = getHorizonMap(format);
      return format === 'unorm16'
        ? importGraphBuffer(graph, 'horizon', map.buffer, 'uint32', Math.ceil(map.elementCount / 2))
        : importGraphBuffer(graph, 'horizon', map.buffer, 'float32', map.elementCount);
    };

    const compiledGraphs = new Map<string, CompiledGPUCommandGraph<void>>();
    /** Builds the graph for `key` the first time it is asked for. */
    function getGraph(
      key: string,
      build: (graph: GPUCommandGraph<void>) => void
    ): CompiledGPUCommandGraph<void> {
      let compiled = compiledGraphs.get(key);
      if (!compiled) {
        const graph = new GPUCommandGraph<void>(device, {id: `relief-visualization-${key}`});
        build(graph);
        compiled = resources.track(graph.compile());
        compiledGraphs.set(key, compiled);
      }
      return compiled;
    }

    function getHorizonGraph(
      algorithm: GPUTerrainHorizonAlgorithm,
      format: GPUTerrainHorizonFormat,
      radius: number = horizonRadius
    ) {
      return getGraph(`horizon-${algorithm}-${format}-r${radius}`, graph => {
        graph.add(
          new GPUTerrainHorizon({
            id: 'horizon',
            width,
            height,
            elevation: importElevation(graph),
            settings: horizonSettings.importToGraph(graph),
            directionCount: DIRECTION_COUNT,
            maximumRadius: radius,
            algorithm,
            horizonFormat: format,
            cellSizeMode: 'uniform',
            rowDirection: 'south',
            horizon: importHorizon(graph, format),
            skyViewFactor: importFloats(graph, 'sky-view', skyViewBuffer),
            anisotropicSkyViewFactor: importFloats(graph, 'anisotropic', anisotropicBuffer),
            positiveOpenness: importFloats(graph, 'positive-openness', positiveOpennessBuffer),
            negativeOpenness: importFloats(graph, 'negative-openness', negativeOpennessBuffer)
          })
        );
      });
    }

    function getStageGraph(stage: Stage): CompiledGPUCommandGraph<void> {
      switch (stage) {
        case 'setup':
          return getGraph('setup', graph => {
            graph.add(
              new GPUTerrainDerivatives({
                id: 'derivatives',
                width,
                height,
                elevation: importElevation(graph),
                settings: derivativesSettings.importToGraph(graph),
                slope: importFloats(graph, 'slope', slopeBuffer),
                aspect: importFloats(graph, 'aspect', aspectBuffer),
                cellSizeMode: 'uniform',
                rowDirection: 'south'
              })
            );
          });
        case 'horizon':
          return getHorizonGraph(horizonAlgorithm, horizonFormat, horizonRadius);
        case 'shadow':
          return getGraph('shadow', graph => {
            graph.add(
              new GPUTerrainCastShadow({
                id: 'cast-shadow',
                width,
                height,
                elevation: importElevation(graph),
                settings: shadowSettings.importToGraph(graph),
                cellSizeMode: 'uniform',
                rowDirection: 'south',
                sunVisibility: importFloats(graph, 'sun-visibility', sunVisibilityBuffer),
                horizonAngle: importFloats(graph, 'sun-horizon', sunHorizonBuffer)
              })
            );
          });
        case 'irradiance':
          return getGraph(`irradiance-${horizonFormat}`, graph => {
            graph.add(
              new GPUSolarIrradiance({
                id: 'solar-irradiance',
                width,
                height,
                directionCount: DIRECTION_COUNT,
                horizonFormat,
                horizon: importHorizon(graph, horizonFormat),
                sunTable: importGraphBuffer(
                  graph,
                  'sun-table',
                  sunTableBuffer,
                  'float32',
                  SUN_TABLE_CAPACITY * GPU_SOLAR_IRRADIANCE_SUN_TABLE_STRIDE
                ),
                sampleCapacity: SUN_TABLE_CAPACITY,
                settings: irradianceSettings.importToGraph(graph),
                slope: importFloats(graph, 'slope', slopeBuffer),
                aspect: importFloats(graph, 'aspect', aspectBuffer),
                skyViewFactor: importFloats(graph, 'sky-view', skyViewBuffer),
                sunHours: importFloats(graph, 'sun-hours', sunHoursBuffer),
                insolation: importFloats(graph, 'insolation', insolationBuffer)
              })
            );
          });
        case 'look':
          return getGraph('look', graph => {
            const elevation = importElevation(graph);
            const curvature = importFloats(graph, 'ring-curvature', ringCurvatureBuffer);
            graph.add(
              new GPUTerrainCurvature({
                id: 'ring-curvature',
                width,
                height,
                elevation,
                settings: curvatureSettings.importToGraph(graph),
                ringCurvature: curvature,
                ringRadii: CURVATURE_RING_RADII,
                ringSquash: 'pade-tanh',
                cellSizeMode: 'uniform',
                rowDirection: 'south'
              })
            );
            graph.add(
              new GPUReliefShading({
                id: 'relief-shading',
                width,
                height,
                elevation,
                settings: reliefSettings.importToGraph(graph),
                skyViewFactor: importFloats(graph, 'sky-view', skyViewBuffer),
                curvature,
                imhofSwing: true,
                cellSizeMode: 'uniform',
                rowDirection: 'south',
                hillshade: importFloats(graph, 'hillshade', hillshadeBuffer),
                color: importGraphBuffer(
                  graph,
                  'relief-color',
                  reliefColorBuffer,
                  'uint32',
                  pixelCount
                )
              })
            );
          });
        case 'rvt':
          return getGraph('rvt', graph => {
            const elevation = importElevation(graph);
            graph.add(
              new GPUSimpleLocalRelief({
                id: 'simple-local-relief',
                width,
                height,
                elevation,
                settings: simpleReliefSettings.importToGraph(graph),
                radius: LOCAL_RELIEF_RADIUS,
                relief: importFloats(graph, 'simple-local-relief', simpleReliefBuffer)
              })
            );
            graph.add(
              new GPUMultiScaleRelief({
                id: 'multi-scale-relief',
                width,
                height,
                elevation,
                settings: multiScaleSettings.importToGraph(graph),
                resolution: cellSize[0],
                featureMinimum: MULTI_SCALE_FEATURE_MINIMUM_CELLS * cellSize[0],
                featureMaximum: MULTI_SCALE_FEATURE_MAXIMUM_CELLS * cellSize[0],
                scalingFactor: MULTI_SCALE_SCALING_FACTOR,
                relief: importFloats(graph, 'multi-scale-relief', multiScaleBuffer)
              })
            );
            graph.add(
              new GPULocalDominance({
                id: 'local-dominance',
                width,
                height,
                elevation,
                settings: dominanceSettings.importToGraph(graph),
                minimumRadius: DOMINANCE_MINIMUM_RADIUS,
                maximumRadius: DOMINANCE_MAXIMUM_RADIUS,
                radiusIncrement: DOMINANCE_RADIUS_INCREMENT,
                angularResolution: DOMINANCE_ANGULAR_RESOLUTION,
                dominance: importFloats(graph, 'local-dominance', dominanceBuffer)
              })
            );
          });
        case 'blend':
          return getGraph('blend', graph => {
            graph.add(
              new GPUReliefBlend({
                id: 'relief-blend',
                width,
                height,
                // Bottom to top, the VAT input order: hillshade, slope, positive openness, SVF.
                layers: [
                  importFloats(graph, 'hillshade', hillshadeBuffer),
                  importFloats(graph, 'slope', slopeBuffer),
                  importFloats(graph, 'positive-openness', positiveOpennessBuffer),
                  importFloats(graph, 'sky-view', skyViewBuffer)
                ],
                settings: blendSettings.importToGraph(graph),
                color: importGraphBuffer(
                  graph,
                  'blend-color',
                  blendColorBuffer,
                  'uint32',
                  pixelCount
                )
              })
            );
          });
      }
    }

    /** Resolves a stop (`0` is the full tile) to a pixel radius the contributor accepts. */
    function resolveHorizonRadius(stop: number): number {
      return stop === 0
        ? Math.max(1, Math.max(width, height) - 1)
        : Math.min(stop, Math.max(width, height) - 1);
    }

    // --- State ---------------------------------------------------------------------------------
    let product: Product = 'sky-view';
    let horizonAlgorithm: GPUTerrainHorizonAlgorithm = 'sweep';
    let horizonFormat: GPUTerrainHorizonFormat = 'float32';
    let horizonRadius = resolveHorizonRadius(HORIZON_RADIUS_STOPS[DEFAULT_HORIZON_RADIUS_STOP]);
    let anisotropyAzimuth = 315;
    let anisotropyLevel = 4;
    let anisotropyMinimumWeight = 0.4;
    let sunAzimuth = 250;
    let sunAltitude = 10;
    let sunSoftness = 1;
    let animateSun = false;
    let dayOfYear = 172;
    let diffuseIrradiance = 80;
    let lightModel: LightModel = 'imhof-swing';
    let swingDegrees = 65;
    let curvatureStrength = 0.6;
    let contrastStrength = 0.5;
    let contrastHighElevation = 160;
    let exposure = 1.1;
    let elevationTint = true;
    let shadowOnReliefStrength = 0.7;
    let verticalExaggeration = 1;
    let observerHeight = 3;
    let vatPreset: VatPreset = 'archaeological';
    let opacity = 0.92;
    let destroyed = false;
    const dirtyStages = new Set<Stage>(STAGE_ORDER);
    const encodedOnce = new Set<string>();

    function invalidate(stage: Stage): void {
      dirtyStages.add(stage);
      for (const downstream of DOWNSTREAM_STAGES[stage]) dirtyStages.add(downstream);
    }

    function writeDerivatives(): void {
      derivativesSettings.write(
        getGPUTerrainDerivativesParameterValues({
          cellSize,
          azimuthDegrees: 315,
          altitudeDegrees: 45
        })
      );
      curvatureSettings.write(
        getGPUTerrainCurvatureParameterValues({cellSize, ringGains: [0.55, 0.45]})
      );
    }
    function writeHorizon(): void {
      horizonSettings.write(
        getGPUTerrainHorizonParameterValues({
          cellSize,
          anisotropyAzimuthDegrees: anisotropyAzimuth,
          anisotropyLevel,
          anisotropyMinimumWeight
        })
      );
      invalidate('horizon');
    }
    function writeShadow(): void {
      shadowSettings.write(
        getGPUTerrainCastShadowParameterValues(
          {
            cellSize,
            azimuthDegrees: sunAzimuth,
            altitudeDegrees: sunAltitude,
            angularRadiusDegrees: GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES * sunSoftness
          },
          'south'
        )
      );
      invalidate('shadow');
    }

    let dayLengthHours = 0;
    function writeIrradiance(): void {
      // Local standard-time midnight to midnight of the chosen day.
      const start = Date.UTC(YEAR, 0, dayOfYear, -LOCAL_UTC_OFFSET_HOURS, 0, 0);
      const table = getGPUSolarIrradianceSunTable({
        longitude: centerLongitudeLatitude[0],
        latitude: centerLongitudeLatitude[1],
        start,
        end: start + 24 * 3600 * 1000,
        stepMinutes: SUN_TABLE_STEP_MINUTES
      });
      sunTableBuffer.write(table.values);
      irradianceSettings.write(
        getGPUSolarIrradianceParameterValues({
          sampleCount: table.sampleCount,
          diffuseIrradiance
        })
      );
      dayLengthHours = 0;
      for (let row = 0; row < table.sampleCount; row++) {
        const base = row * GPU_SOLAR_IRRADIANCE_SUN_TABLE_STRIDE;
        if (table.values[base + 1] > 0) dayLengthHours += table.values[base + 2];
      }
      invalidate('irradiance');
    }

    function writeLook(): void {
      const lightSettings =
        lightModel === 'mdow'
          ? {lights: 'mdow' as const, lightWeighting: 'aspect' as const}
          : {
              lights: [{azimuthDegrees: 315, altitudeDegrees: 40}] as const,
              lightWeighting: lightModel
            };
      reliefSettings.write(
        getGPUReliefShadingParameterValues({
          cellSize,
          ...lightSettings,
          imhofSwingDegrees: lightModel === 'imhof-swing' ? swingDegrees : 0,
          curvatureStrength,
          contrastLowElevation: 0,
          contrastHighElevation,
          contrastStrength,
          hillshadeStrength: 1,
          skyViewStrength: 1,
          textureShadeStrength: 0,
          exposure,
          tintStrength: 0.35,
          elevationStops: elevationTint ? ELEVATION_STOPS : []
        })
      );
      invalidate('look');
    }
    function writeRvt(): void {
      simpleReliefSettings.write(getGPUSimpleLocalReliefParameterValues({verticalExaggeration}));
      multiScaleSettings.write(getGPUMultiScaleReliefParameterValues({verticalExaggeration}));
      dominanceSettings.write(
        getGPULocalDominanceParameterValues({observerHeight, verticalExaggeration})
      );
      invalidate('rvt');
    }
    function writeBlend(): void {
      const presets: Record<VatPreset, readonly GPUReliefBlendLayerSettings[]> = {
        archaeological: GPU_RELIEF_BLEND_VAT_ARCHAEOLOGICAL,
        flat: GPU_RELIEF_BLEND_VAT_FLAT,
        // Only the bottom layer: the other layers have zero opacity.
        'hillshade-only': GPU_RELIEF_BLEND_VAT_ARCHAEOLOGICAL.map(layer => ({...layer, opacity: 0}))
      };
      blendSettings.write(getGPUReliefBlendParameterValues(presets[vatPreset]));
      invalidate('blend');
    }

    writeDerivatives();
    writeHorizon();
    writeShadow();
    writeIrradiance();
    writeLook();
    writeRvt();
    writeBlend();

    // --- Controls ------------------------------------------------------------------------------
    const productDescription = context.controls.addReadout('Showing', '');
    const updateProduct = () => {
      productDescription.setValue(describeProduct(product));
      context.updateLayers();
    };

    context.controls.addSelect<Product>({
      label: 'Product',
      options: [
        {value: 'sky-view', label: 'Sky-view factor'},
        {value: 'anisotropic-sky-view', label: 'Anisotropic sky-view factor'},
        {value: 'positive-openness', label: 'Positive openness'},
        {value: 'negative-openness', label: 'Negative openness'},
        {value: 'cast-shadow', label: 'Cast shadow (single sun)'},
        {value: 'sun-horizon-angle', label: 'Horizon angle toward the sun'},
        {value: 'shadow-on-relief', label: 'Cast shadow on Imhof relief'},
        {value: 'sun-hours', label: 'Sun hours (date)'},
        {value: 'insolation', label: 'Insolation (date)'},
        {value: 'imhof', label: 'Imhof relief shading'},
        {value: 'ring-curvature', label: 'Ring curvature'},
        {value: 'simple-local-relief', label: 'Simple local relief (RVT)'},
        {value: 'multi-scale-relief', label: 'Multi-scale relief (RVT)'},
        {value: 'local-dominance', label: 'Local dominance (RVT)'},
        {value: 'vat', label: 'VAT blend (RVT)'}
      ],
      value: product,
      onChange: value => {
        product = value;
        updateProduct();
      }
    });

    context.controls.addSelect<GPUTerrainHorizonAlgorithm>({
      label: 'Horizon algorithm (compiles once per choice)',
      options: [
        {value: 'sweep', label: 'sweep (exact hull sweep, O(1) per pixel)'},
        {value: 'march', label: 'march (bounded ray march, O(steps))'}
      ],
      value: horizonAlgorithm,
      onChange: value => {
        horizonAlgorithm = value;
        invalidate('horizon');
      }
    });
    context.controls.addSelect<string>({
      label: 'Horizon radius (compiles once per choice)',
      options: HORIZON_RADIUS_STOPS.map(stop => ({
        value: String(stop),
        label: stop === 0 ? 'full tile' : `${stop} px`
      })),
      value: String(HORIZON_RADIUS_STOPS[DEFAULT_HORIZON_RADIUS_STOP]),
      onChange: value => {
        horizonRadius = resolveHorizonRadius(Number(value));
        horizonMemoryReadout.setValue(describeHorizonMemory());
        speedupReadout.setValue('measure to compare');
        invalidate('horizon');
      }
    });
    context.controls.addSelect<GPUTerrainHorizonFormat>({
      label: 'Horizon storage',
      options: [
        {value: 'float32', label: 'float32 degrees'},
        {value: 'unorm16', label: 'unorm16 (half the memory, 0.0027° steps)'}
      ],
      value: horizonFormat,
      onChange: value => {
        horizonFormat = value;
        horizonMemoryReadout.setValue(describeHorizonMemory());
        invalidate('horizon');
      }
    });
    context.controls.addSlider({
      label: 'Anisotropic SVF direction',
      min: 0,
      max: 355,
      step: 5,
      value: anisotropyAzimuth,
      format: value => `${value}°`,
      onChange: value => {
        anisotropyAzimuth = value;
        writeHorizon();
      }
    });
    context.controls.addSlider({
      label: 'Anisotropy level (exponent)',
      min: 0,
      max: 12,
      step: 1,
      value: anisotropyLevel,
      format: value => String(value),
      onChange: value => {
        anisotropyLevel = value;
        writeHorizon();
      }
    });
    context.controls.addSlider({
      label: 'Anisotropy minimum weight',
      min: 0,
      max: 1,
      step: 0.05,
      value: anisotropyMinimumWeight,
      format: value => value.toFixed(2),
      onChange: value => {
        anisotropyMinimumWeight = value;
        writeHorizon();
      }
    });

    const sunAzimuthSlider = context.controls.addSlider({
      label: 'Sun azimuth (per-frame)',
      min: 0,
      max: 360,
      step: 1,
      value: sunAzimuth,
      format: value => `${value.toFixed(0)}°`,
      onChange: value => {
        sunAzimuth = value;
        writeShadow();
      }
    });
    context.controls.addSlider({
      label: 'Sun altitude (per-frame)',
      min: 0,
      max: 80,
      step: 1,
      value: sunAltitude,
      format: value => `${value}°`,
      onChange: value => {
        sunAltitude = value;
        writeShadow();
      }
    });
    context.controls.addSlider({
      label: 'Shadow softness (solar disk radius)',
      min: 0,
      max: 8,
      step: 0.5,
      value: sunSoftness,
      format: value =>
        value === 0 ? 'hard' : `${(value * GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES).toFixed(2)}°`,
      onChange: value => {
        sunSoftness = value;
        writeShadow();
      }
    });
    context.controls.addToggle({
      label: 'Animate sun azimuth',
      value: animateSun,
      onChange: value => {
        animateSun = value;
      }
    });
    context.controls.addSlider({
      label: 'Shadow strength on relief',
      min: 0,
      max: 1,
      step: 0.05,
      value: shadowOnReliefStrength,
      format: value => value.toFixed(2),
      onChange: value => {
        shadowOnReliefStrength = value;
        context.updateLayers();
      }
    });

    const dateReadout = context.controls.addReadout('Date', '');
    const updateDateReadout = () => {
      dateReadout.setValue(`${formatDay(dayOfYear)} · day length ${dayLengthHours.toFixed(1)} h`);
    };
    context.controls.addSlider({
      label: 'Date (sun hours, insolation)',
      min: 1,
      max: 365,
      step: 1,
      value: dayOfYear,
      format: formatDay,
      onChange: value => {
        dayOfYear = value;
        writeIrradiance();
        updateDateReadout();
      }
    });
    context.controls.addSlider({
      label: 'Diffuse sky irradiance',
      min: 0,
      max: 200,
      step: 10,
      value: diffuseIrradiance,
      format: value => `${value} W/m²`,
      onChange: value => {
        diffuseIrradiance = value;
        writeIrradiance();
      }
    });

    context.controls.addSelect<LightModel>({
      label: 'Imhof light model',
      options: [
        {value: 'imhof-swing', label: 'Imhof azimuth swing'},
        {value: 'mdow', label: 'USGS MDOW (aspect-weighted)'},
        {value: 'fixed', label: 'Single light, 315°'}
      ],
      value: lightModel,
      onChange: value => {
        lightModel = value;
        writeLook();
      }
    });
    context.controls.addSlider({
      label: 'Imhof swing',
      min: 0,
      max: 90,
      step: 5,
      value: swingDegrees,
      format: value => `${value}°`,
      onChange: value => {
        swingDegrees = value;
        writeLook();
      }
    });
    context.controls.addSlider({
      label: 'Curvature strength',
      min: 0,
      max: 2,
      step: 0.1,
      value: curvatureStrength,
      format: value => value.toFixed(1),
      onChange: value => {
        curvatureStrength = value;
        writeLook();
      }
    });
    context.controls.addSlider({
      label: 'Elevation contrast',
      min: 0,
      max: 1,
      step: 0.05,
      value: contrastStrength,
      format: value => value.toFixed(2),
      onChange: value => {
        contrastStrength = value;
        writeLook();
      }
    });
    context.controls.addSlider({
      label: 'Full-contrast elevation',
      min: 20,
      max: 280,
      step: 10,
      value: contrastHighElevation,
      format: value => `${value} m`,
      onChange: value => {
        contrastHighElevation = value;
        writeLook();
      }
    });
    context.controls.addSlider({
      label: 'Relief exposure',
      min: 0.6,
      max: 1.6,
      step: 0.05,
      value: exposure,
      format: value => value.toFixed(2),
      onChange: value => {
        exposure = value;
        writeLook();
      }
    });
    context.controls.addToggle({
      label: 'Elevation tint',
      value: elevationTint,
      onChange: value => {
        elevationTint = value;
        writeLook();
      }
    });

    context.controls.addSlider({
      label: 'RVT vertical exaggeration',
      min: 1,
      max: 5,
      step: 0.5,
      value: verticalExaggeration,
      format: value => `${value}x`,
      onChange: value => {
        verticalExaggeration = value;
        writeRvt();
      }
    });
    context.controls.addSlider({
      label: 'Local dominance observer height',
      min: 1,
      max: 20,
      step: 1,
      value: observerHeight,
      format: value => `${value} m`,
      onChange: value => {
        observerHeight = value;
        writeRvt();
      }
    });
    context.controls.addSelect<VatPreset>({
      label: 'VAT preset (GPUReliefBlend)',
      options: [
        {value: 'archaeological', label: 'Archaeological (SVF, openness, slope, hillshade)'},
        {value: 'flat', label: 'Flat terrain stretches'},
        {value: 'hillshade-only', label: 'Hillshade layer only'}
      ],
      value: vatPreset,
      onChange: value => {
        vatPreset = value;
        writeBlend();
      }
    });
    context.controls.addSlider({
      label: 'Layer opacity',
      min: 0.3,
      max: 1,
      step: 0.05,
      value: opacity,
      format: value => value.toFixed(2),
      onChange: value => {
        opacity = value;
        context.updateLayers();
      }
    });

    context.controls.addLegend({
      title: 'Sky-view factor (viridis)',
      gradient: {
        colors: [
          [68, 1, 84],
          [33, 144, 141],
          [253, 231, 37]
        ],
        minimumLabel: '0.6 (enclosed)',
        maximumLabel: '1.0 (open sky)'
      }
    });
    context.controls.addLegend({
      title: 'Sun hours (inferno)',
      gradient: {
        colors: [
          [0, 0, 4],
          [187, 55, 84],
          [252, 255, 164]
        ],
        minimumLabel: '0 h (all shadow)',
        maximumLabel: '16 h'
      }
    });
    context.controls.addLegend({
      title: 'Insolation (inferno)',
      gradient: {
        colors: [
          [0, 0, 4],
          [187, 55, 84],
          [252, 255, 164]
        ],
        minimumLabel: '0 kWh/m²',
        maximumLabel: '10 kWh/m²'
      }
    });
    context.controls.addLegend({
      title: 'Openness, curvature, RVT, VAT, shadow (grey)',
      gradient: {
        colors: [
          [0, 0, 0],
          [128, 128, 128],
          [255, 255, 255]
        ],
        minimumLabel: 'low / concave / shadow',
        maximumLabel: 'high / convex / lit'
      }
    });
    context.controls.addNote(
      'Sun azimuth, altitude, softness, anisotropy direction, date, light model, curvature, ' +
        'contrast, exaggeration and the VAT preset are parameter-buffer writes. The horizon ' +
        'algorithm and storage are topology: the first selection of each compiles one graph, ' +
        'which the panel counts as a rebuild.'
    );
    context.controls.addReadout(
      'Raster',
      `${width} × ${height} cells, ${formatCount(landCount)} land, ${cellSize[0].toFixed(1)} m`
    );
    const horizonMemoryReadout = context.controls.addReadout('Horizon map', '');
    function describeHorizonMemory(): string {
      const byteLength = getHorizonMap(horizonFormat).byteLength;
      return `${DIRECTION_COUNT} sectors × ${horizonRadius} px, ${horizonFormat}, ${(byteLength / 1048576).toFixed(1)} MB`;
    }
    horizonMemoryReadout.setValue(describeHorizonMemory());
    const marchReadout = context.controls.addReadout('Horizon march', 'not measured');
    const sweepReadout = context.controls.addReadout('Horizon sweep', 'not measured');
    const speedupReadout = context.controls.addReadout('Sweep vs march time', '...');
    context.controls.addNote(
      'The sweep has a large fixed cost per pixel and sector, so it wins at large radii ' +
        '(near the full tile) and the march wins at small radii; on this 512 × 512 tile the ' +
        'crossover is between 256 px and the full tile.'
    );
    const stageReadouts: Partial<Record<Stage, ReturnType<typeof context.controls.addReadout>>> = {
      shadow: context.controls.addReadout('Cast shadow (per sun move)', 'not measured'),
      irradiance: context.controls.addReadout('Solar irradiance (per date)', 'not measured'),
      look: context.controls.addReadout('Curvature + Imhof relief', 'not measured'),
      rvt: context.controls.addReadout('RVT (SLRM, MSRM, dominance)', 'not measured'),
      blend: context.controls.addReadout('VAT blend', 'not measured')
    };
    context.controls.addReadout('Data', terrain.attribution);
    updateDateReadout();
    productDescription.setValue(describeProduct(product));

    const measureOptions = {
      parameters: undefined,
      completionBuffer: skyViewBuffer,
      signal: context.signal
    };
    const timeGraph = (compiled: CompiledGPUCommandGraph<void>) =>
      measureCompiledGraph(device, compiled, {
        ...measureOptions,
        runs: 3,
        warmUpRuns: 1,
        repetitions: 2
      });

    const measureHorizon = async () => {
      const radius = horizonRadius;
      const march = await timeGraph(getHorizonGraph('march', horizonFormat, radius));
      marchReadout.setValue(formatCompiledGraphTiming(march));
      const sweep = await timeGraph(getHorizonGraph('sweep', horizonFormat, radius));
      sweepReadout.setValue(formatCompiledGraphTiming(sweep));
      const ratio = (sweep.milliseconds / march.milliseconds).toFixed(2);
      speedupReadout.setValue(`r${radius}: sweep takes ${ratio}× the march time`);
      // Measuring overwrote the horizon outputs with the last variant: refresh the active one.
      invalidate('horizon');
    };
    const measureAll = async () => {
      try {
        await measureHorizon();
        for (const stage of ['shadow', 'irradiance', 'look', 'rvt', 'blend'] as const) {
          const timing: CompiledGraphTiming = await timeGraph(getStageGraph(stage));
          stageReadouts[stage]?.setValue(formatCompiledGraphTiming(timing));
        }
      } catch (error) {
        if (!destroyed) speedupReadout.setValue(`failed: ${(error as Error).message}`);
      }
    };
    context.controls.addButton({label: 'Measure GPU cost', onClick: () => void measureAll()});

    // --- Encode and layers ---------------------------------------------------------------------
    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [...compiledGraphs.values()],
      encode(commandEncoder, frame) {
        if (animateSun) {
          sunAzimuth = (sunAzimuth + frame.deltaSeconds * 20) % 360;
          sunAzimuthSlider.setValue(sunAzimuth);
          writeShadow();
        }
        const needed = PRODUCT_STAGES[product];
        for (const stage of STAGE_ORDER) {
          if (!needed.includes(stage) || !dirtyStages.has(stage)) continue;
          getStageGraph(stage).encode(commandEncoder, {parameters: undefined});
          dirtyStages.delete(stage);
          encodedOnce.add(stage);
        }
      },
      getLayers() {
        if (product === 'imhof' || product === 'shadow-on-relief' || product === 'vat') {
          const showsShadow = product === 'shadow-on-relief';
          return [
            new ReliefRasterLayer({
              id: `relief-visualization-${product}`,
              coordinateOrigin: origin,
              colors: product === 'vat' ? blendColorBuffer : reliefColorBuffer,
              light: showsShadow ? sunVisibilityBuffer : null,
              lightStrength: showsShadow ? shadowOnReliefStrength : 0,
              gridSize: [width, height],
              bounds,
              lightGain: 1.4,
              opacity
            })
          ];
        }
        const buffers: Partial<Record<Product, typeof skyViewBuffer>> = {
          'sky-view': skyViewBuffer,
          'anisotropic-sky-view': anisotropicBuffer,
          'positive-openness': positiveOpennessBuffer,
          'negative-openness': negativeOpennessBuffer,
          'cast-shadow': sunVisibilityBuffer,
          'sun-horizon-angle': sunHorizonBuffer,
          'sun-hours': sunHoursBuffer,
          insolation: insolationBuffer,
          'ring-curvature': ringCurvatureBuffer,
          'simple-local-relief': simpleReliefBuffer,
          'multi-scale-relief': multiScaleBuffer,
          'local-dominance': dominanceBuffer
        };
        const style = SCALAR_STYLES[product];
        const values = buffers[product];
        if (!style || !values) return [];
        return [
          new SpatialAnalysisRasterLayer({
            id: `relief-visualization-${product}`,
            coordinateOrigin: origin,
            gridSize: [width, height],
            bounds,
            rowOrigin: 'north',
            valueFormat: 'float32',
            values,
            colormap: style.colormap,
            valueRange: style.valueRange,
            opacity
          } as SpatialAnalysisRasterLayerProps)
        ];
      },
      destroy() {
        destroyed = true;
        resources.destroy();
      }
    };

    // Build the product's graphs before the first frame, then time the horizon pair once.
    for (const stage of PRODUCT_STAGES[product]) getStageGraph(stage);
    setTimeout(() => {
      if (!destroyed) void measureHorizon().catch(() => {});
    }, 1500);
    // Keep the sun and sample the solar position once for the status line.
    const solstice = getSolarPosition(
      Date.UTC(YEAR, 0, dayOfYear, 12 - LOCAL_UTC_OFFSET_HOURS),
      centerLongitudeLatitude[0],
      centerLongitudeLatitude[1]
    );
    context.setStatus(
      `Solar noon altitude on the chosen date: ${solstice.altitudeDegrees.toFixed(0)}°`
    );
    return instance;
  }
};

/** One-line meaning of a product for the "Showing" readout. */
function describeProduct(product: Product): string {
  const style = SCALAR_STYLES[product];
  if (style) return style.description;
  switch (product) {
    case 'imhof':
      return 'Imhof relief: azimuth-swung light, curvature added to the shade, elevation contrast.';
    case 'shadow-on-relief':
      return 'Imhof relief darkened by the single-sun cast shadow.';
    default:
      return 'VAT blend of hillshade, slope, positive openness and sky-view factor.';
  }
}

/** Formats a 1-based day of year as `"Jun 21"`. */
function formatDay(dayOfYear: number): string {
  const date = new Date(Date.UTC(YEAR, 0, dayOfYear));
  return `${date.toLocaleString('en-US', {month: 'short', timeZone: 'UTC'})} ${date.getUTCDate()}`;
}
