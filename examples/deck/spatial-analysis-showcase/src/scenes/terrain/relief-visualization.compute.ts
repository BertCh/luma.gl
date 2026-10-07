// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {
  getGPUReliefBlendParameterValues,
  getGPUReliefShadingParameterValues,
  getGPUSimpleLocalReliefParameterValues,
  getGPUTerrainCurvatureParameterValues,
  getGPUTerrainDerivativesParameterValues,
  getGPUTerrainHorizonParameterValues,
  getGPUTextureShadingParameterValues,
  getGPULocalDominanceParameterValues,
  getGPULocalDominanceShifts,
  getGPUMultiScaleReliefParameterValues,
  getGPUMultiScaleReliefRadii,
  GPU_LOCAL_DOMINANCE_PARAMETER_LENGTH,
  GPU_MULTI_SCALE_RELIEF_PARAMETER_LENGTH,
  GPU_RELIEF_BLEND_LAYER_PARAMETER_LENGTH,
  GPU_RELIEF_BLEND_VAT_ARCHAEOLOGICAL,
  GPU_RELIEF_BLEND_VAT_FLAT,
  GPU_RELIEF_SHADING_PARAMETER_LENGTH,
  GPU_SIMPLE_LOCAL_RELIEF_PARAMETER_LENGTH,
  GPU_TERRAIN_CURVATURE_PARAMETER_LENGTH,
  GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH,
  GPU_TERRAIN_HORIZON_ANISOTROPIC_PARAMETER_LENGTH,
  GPU_TEXTURE_SHADING_PARAMETER_LENGTH,
  GPULocalDominance,
  GPUMultiScaleRelief,
  GPUReliefBlend,
  GPUReliefShading,
  GPUSimpleLocalRelief,
  GPUTerrainCurvature,
  GPUTerrainDerivatives,
  GPUTerrainHorizon,
  GPUTextureShading,
  type GPUReliefBlendLayerSettings
} from '@luma.gl/experimental/gpu-terrain';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import type {SceneContext, SceneInstance} from '../scene';
import type {PaintSpec} from './b14a-colorize';
import {loadAlpsGrid} from './b14a-grid';
import {TerrainSession, type ProductBuild, type Stage, type ValueStats} from './b14a-session';
import {StageRegistry, type StageEntry} from './b14a-stage-registry';
import {
  ALPINE_ELEVATION_STOPS,
  SCALAR_PRODUCTS,
  type ReliefOptions,
  type ReliefProduct
} from './relief-visualization.style';

export type {ReliefOptions, ReliefProduct};

type StageId =
  | 'horizon'
  | 'texture'
  | 'curvature'
  | 'look-plain'
  | 'look-full'
  | 'slope'
  | 'slrm'
  | 'msrm'
  | 'dominance'
  | 'vat';

const DEPENDENCIES: Record<StageId, readonly StageId[]> = {
  horizon: [],
  texture: [],
  curvature: [],
  'look-plain': [],
  'look-full': ['horizon', 'texture', 'curvature'],
  slope: [],
  slrm: [],
  msrm: [],
  dominance: [],
  vat: ['look-plain', 'slope', 'horizon']
};

/** Where each product's raster comes from. */
const SOURCES: Record<
  ReliefProduct,
  {stage: StageId; buffer: string; format: 'float32' | 'rgba8'}
> = {
  hillshade: {stage: 'look-plain', buffer: 'hillshade', format: 'float32'},
  mdow: {stage: 'look-plain', buffer: 'hillshade', format: 'float32'},
  imhof: {stage: 'look-full', buffer: 'color', format: 'rgba8'},
  texture: {stage: 'texture', buffer: 'textureShade', format: 'float32'},
  'sky-view': {stage: 'horizon', buffer: 'svf', format: 'float32'},
  anisotropic: {stage: 'horizon', buffer: 'aniso', format: 'float32'},
  'positive-openness': {stage: 'horizon', buffer: 'pos', format: 'float32'},
  'negative-openness': {stage: 'horizon', buffer: 'neg', format: 'float32'},
  slrm: {stage: 'slrm', buffer: 'relief', format: 'float32'},
  msrm: {stage: 'msrm', buffer: 'relief', format: 'float32'},
  'local-dominance': {stage: 'dominance', buffer: 'dominance', format: 'float32'},
  vat: {stage: 'vat', buffer: 'color', format: 'rgba8'}
};

/** Compile-time options of each stage; changing one rebuilds that stage and its dependents. */
function getStageKey(id: StageId, state: ReliefOptions): string {
  switch (id) {
    case 'horizon':
      return `${state.horizonDirections}|${state.horizonRadius}|${state.horizonAlgorithm}|${state.horizonAlgorithm === 'sweep' ? 1 : state.horizonGrowth}`;
    case 'texture':
      return `${state.textureLevels}|${state.textureBaseSigma}|${state.textureHasNodata}|${state.textureDownsample}`;
    case 'slrm':
      return `${state.slrmRadius}`;
    case 'msrm':
      return `${state.msrmMinimumFeature}|${state.msrmMaximumFeature}|${state.msrmScaling}`;
    case 'dominance':
      return `${state.dominanceMinimumRadius}|${state.dominanceMaximumRadius}|${state.dominanceIncrement}|${state.dominanceAngle}`;
    default:
      return id;
  }
}

/** Option id to the stages whose parameter buffers it feeds. */
const PARAMETER_STAGES: Partial<Record<keyof ReliefOptions, readonly StageId[]>> = {
  lightAzimuth: ['look-plain', 'look-full'],
  lightAltitude: ['look-plain', 'look-full'],
  zFactor: ['look-plain', 'look-full', 'horizon', 'slope', 'curvature'],
  swissLight: ['look-full'],
  imhofSwing: ['look-full'],
  curvatureStrength: ['look-full'],
  contrastStrength: ['look-full'],
  contrastHighElevation: ['look-full'],
  hillshadeStrength: ['look-full'],
  skyViewStrength: ['look-full'],
  textureStrength: ['look-full'],
  exposure: ['look-full'],
  elevationTint: ['look-full'],
  tintStrength: ['look-full'],
  textureDetail: ['texture'],
  textureGain: ['texture'],
  anisotropyAzimuth: ['horizon'],
  anisotropyLevel: ['horizon'],
  anisotropyMinimumWeight: ['horizon'],
  rvtExaggeration: ['slrm', 'msrm', 'dominance'],
  dominanceObserverHeight: ['dominance'],
  vatPreset: ['vat'],
  vatSlopeOpacity: ['vat'],
  vatOpennessOpacity: ['vat'],
  vatSkyViewOpacity: ['vat'],
  vatSlopeBlend: ['vat'],
  vatOpennessBlend: ['vat'],
  vatSkyViewBlend: ['vat']
};

const COMPILE_OPTIONS = new Set<keyof ReliefOptions>([
  'textureLevels',
  'textureBaseSigma',
  'textureHasNodata',
  'textureDownsample',
  'horizonDirections',
  'horizonRadius',
  'horizonAlgorithm',
  'horizonGrowth',
  'slrmRadius',
  'msrmMinimumFeature',
  'msrmMaximumFeature',
  'msrmScaling',
  'dominanceMinimumRadius',
  'dominanceMaximumRadius',
  'dominanceIncrement',
  'dominanceAngle'
]);

const REBUILD_DELAY_MILLISECONDS = 250;

/** Paint of a product given the state and the percentile range of its histogram, if known. */
export function getReliefPaint(
  state: ReliefOptions,
  range?: {low: number; high: number}
): Partial<PaintSpec> {
  const {product} = state;
  if (product === 'hillshade' || product === 'mdow') {
    return {mode: 'ramp', ramp: 'grayscale', low: 0.05, high: 0.95, alpha: 1, fadeMiddle: false};
  }
  const nominal = SCALAR_PRODUCTS[product];
  if (!nominal) return {};
  const low = range?.low ?? nominal.low * state.rangeScale;
  const high = range?.high ?? nominal.high * state.rangeScale;
  return {mode: 'ramp', ramp: state.ramp, low, high, alpha: 1, fadeMiddle: false};
}

function describeCell(product: ReliefProduct, value: number, elevation: number): string | null {
  const elevationText = Number.isFinite(elevation) ? `\nElevation ${elevation.toFixed(1)} m` : '';
  if (!Number.isFinite(value)) return `No ${product} value (edge or nodata)${elevationText}`;
  switch (product) {
    case 'hillshade':
    case 'mdow':
      return `Hillshade ${value.toFixed(2)}${elevationText}`;
    case 'texture':
      return `Texture shade ${value.toFixed(3)}${elevationText}`;
    case 'sky-view':
    case 'anisotropic':
      return `Sky-view factor ${value.toFixed(3)} (${(value * 100).toFixed(0)} % of the sky visible)${elevationText}`;
    case 'positive-openness':
    case 'negative-openness':
      return `${product === 'positive-openness' ? 'Positive' : 'Negative'} openness ${value.toFixed(1)}°${elevationText}`;
    case 'slrm':
      return `${value >= 0 ? 'Above' : 'Below'} the local mean by ${Math.abs(value).toFixed(1)} m${elevationText}`;
    case 'msrm':
      return `Multi-scale relief ${value.toFixed(1)} m${elevationText}`;
    case 'local-dominance':
      return `Local dominance ${value.toFixed(2)}°${elevationText}`;
    default:
      return null;
  }
}

/**
 * Relief visualization of the Matterhorn tile. Every analysis is a `Stage` (compiled graphs plus
 * their parameter writes) that re-encodes only when its own parameters, the elevation, or a stage
 * it reads changed; heavy stages are rate limited. Compile-time options (radii, level counts,
 * direction counts, algorithms) rebuild only the stage they belong to and its dependents.
 */
export async function createReliefVisualization(
  ctx: SceneContext<ReliefOptions>
): Promise<SceneInstance<ReliefOptions>> {
  const {device} = ctx;
  const grid = await loadAlpsGrid(ctx.datasets.get('alps-dem'), ctx.signal);
  ctx.signal.throwIfAborted();
  const {width, height, pixelCount} = grid;
  const resources = new SpatialAnalysisResources(device, 'relief-visualization');
  const session = new TerrainSession(ctx, resources, grid);
  // The decoded elevation is loaded once; the contributors read it from this buffer.
  session.elevationBuffer.write(grid.cpuElevation);
  session.validityBuffer.write(new Uint32Array(pixelCount).fill(1));
  session.elevationChanged();

  const registry: StageRegistry<StageId, ReliefOptions> = new StageRegistry(
    session,
    DEPENDENCIES,
    getStageKey,
    (id, state, stages) => buildStage(id, state, stages)
  );
  const cell = grid.cellSettings;
  let destroyed = false;
  let timers: ReturnType<typeof setTimeout>[] = [];
  let latestStats: ValueStats | null = null;

  // --- Stage builders ----------------------------------------------------------------------------
  function buildStage(
    id: StageId,
    state: ReliefOptions,
    stages: StageRegistry<StageId, ReliefOptions>
  ): StageEntry {
    const builder = session.builder(id);
    const elevation = builder.elevation();
    const dependency = (stage: StageId) => stages.require(stage);
    const pixelFloats = (stage: StageId, name: string) =>
      importGraphBuffer(
        builder.graph,
        name,
        dependency(stage).builder.getBuffer(name),
        'float32',
        pixelCount
      );
    let stage: Stage;
    switch (id) {
      case 'horizon': {
        const settings = builder.settings(GPU_TERRAIN_HORIZON_ANISOTROPIC_PARAMETER_LENGTH);
        builder.graph.add(
          new GPUTerrainHorizon({
            id: 'horizon',
            width,
            height,
            elevation,
            settings: settings.view,
            directionCount: Number(state.horizonDirections),
            maximumRadius: Math.min(Number(state.horizonRadius), Math.max(width, height) - 1),
            algorithm: state.horizonAlgorithm,
            ...(state.horizonAlgorithm === 'march' ? {stepGrowth: state.horizonGrowth} : {}),
            cellSizeMode: 'web-mercator',
            rowDirection: 'south',
            skyViewFactor: builder.floats('svf'),
            anisotropicSkyViewFactor: builder.floats('aniso'),
            positiveOpenness: builder.floats('pos'),
            negativeOpenness: builder.floats('neg')
          })
        );
        stage = builder.finishStage({
          write: () =>
            settings.parameters.write(
              getGPUTerrainHorizonParameterValues({
                ...cell,
                zFactor: ctx.options.zFactor,
                anisotropyAzimuthDegrees: ctx.options.anisotropyAzimuth,
                anisotropyLevel: ctx.options.anisotropyLevel,
                anisotropyMinimumWeight: ctx.options.anisotropyMinimumWeight
              })
            )
        });
        break;
      }
      case 'texture': {
        const settings = builder.settings(GPU_TEXTURE_SHADING_PARAMETER_LENGTH);
        builder.graph.add(
          new GPUTextureShading({
            id: 'texture-shading',
            width,
            height,
            elevation,
            settings: settings.view,
            levelCount: state.textureLevels,
            baseSigma: state.textureBaseSigma,
            hasNodata: state.textureHasNodata,
            downsampleLevels: state.textureDownsample,
            textureShade: builder.floats('textureShade')
          })
        );
        stage = builder.finishStage({
          write: () =>
            settings.parameters.write(
              getGPUTextureShadingParameterValues({
                detail: ctx.options.textureDetail,
                gain: ctx.options.textureGain
              })
            )
        });
        break;
      }
      case 'curvature': {
        const settings = builder.settings(GPU_TERRAIN_CURVATURE_PARAMETER_LENGTH);
        builder.graph.add(
          new GPUTerrainCurvature({
            id: 'ring-curvature',
            width,
            height,
            elevation,
            settings: settings.view,
            ringCurvature: builder.floats('curvature'),
            ringRadii: [2, 6],
            ringSquash: 'pade-tanh',
            cellSizeMode: 'web-mercator',
            rowDirection: 'south'
          })
        );
        stage = builder.finishStage({
          write: () =>
            settings.parameters.write(
              getGPUTerrainCurvatureParameterValues({
                ...cell,
                zFactor: ctx.options.zFactor,
                ringGains: [0.55, 0.45]
              })
            )
        });
        break;
      }
      case 'look-plain':
      case 'look-full': {
        const full = id === 'look-full';
        const settings = builder.settings(GPU_RELIEF_SHADING_PARAMETER_LENGTH);
        builder.graph.add(
          new GPUReliefShading({
            id: 'relief-shading',
            width,
            height,
            elevation,
            settings: settings.view,
            cellSizeMode: 'web-mercator',
            rowDirection: 'south',
            hillshade: builder.floats('hillshade'),
            ...(full
              ? {
                  skyViewFactor: pixelFloats('horizon', 'svf'),
                  textureShade: pixelFloats('texture', 'textureShade'),
                  curvature: pixelFloats('curvature', 'curvature'),
                  imhofSwing: true,
                  color: builder.words('color')
                }
              : {})
          })
        );
        stage = builder.finishStage({
          dependencies: full
            ? [
                dependency('horizon').stage,
                dependency('texture').stage,
                dependency('curvature').stage
              ]
            : [],
          write: () => {
            const o = ctx.options;
            const single = [
              {azimuthDegrees: o.lightAzimuth, altitudeDegrees: o.lightAltitude}
            ] as const;
            const lights = full
              ? o.swissLight === 'mdow'
                ? {lights: 'mdow' as const, lightWeighting: 'aspect' as const}
                : {lights: single, lightWeighting: o.swissLight}
              : o.product === 'mdow'
                ? {lights: 'mdow' as const, lightWeighting: 'aspect' as const}
                : {lights: single, lightWeighting: 'fixed' as const};
            settings.parameters.write(
              getGPUReliefShadingParameterValues({
                ...cell,
                zFactor: o.zFactor,
                ...lights,
                imhofSwingDegrees: full && o.swissLight === 'imhof-swing' ? o.imhofSwing : 0,
                curvatureStrength: full ? o.curvatureStrength : 0,
                contrastLowElevation: 1500,
                contrastHighElevation: o.contrastHighElevation,
                contrastStrength: full ? o.contrastStrength : 0,
                hillshadeStrength: full ? o.hillshadeStrength : 1,
                skyViewStrength: full ? o.skyViewStrength : 0,
                textureShadeStrength: full ? o.textureStrength : 0,
                exposure: full ? o.exposure : 1,
                tintStrength: full ? o.tintStrength : 0,
                elevationStops: full && o.elevationTint ? ALPINE_ELEVATION_STOPS : []
              })
            );
          }
        });
        break;
      }
      case 'slope': {
        const settings = builder.settings(GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH);
        builder.graph.add(
          new GPUTerrainDerivatives({
            id: 'derivatives',
            width,
            height,
            elevation,
            settings: settings.view,
            slope: builder.floats('slope'),
            cellSizeMode: 'web-mercator',
            rowDirection: 'south'
          })
        );
        stage = builder.finishStage({
          write: () =>
            settings.parameters.write(
              getGPUTerrainDerivativesParameterValues({...cell, zFactor: ctx.options.zFactor})
            )
        });
        break;
      }
      case 'slrm': {
        const settings = builder.settings(GPU_SIMPLE_LOCAL_RELIEF_PARAMETER_LENGTH);
        builder.graph.add(
          new GPUSimpleLocalRelief({
            id: 'simple-local-relief',
            width,
            height,
            elevation,
            settings: settings.view,
            radius: state.slrmRadius,
            relief: builder.floats('relief')
          })
        );
        stage = builder.finishStage({
          minIntervalMs: 80,
          write: () =>
            settings.parameters.write(
              getGPUSimpleLocalReliefParameterValues({
                verticalExaggeration: ctx.options.rvtExaggeration
              })
            )
        });
        break;
      }
      case 'msrm': {
        const settings = builder.settings(GPU_MULTI_SCALE_RELIEF_PARAMETER_LENGTH);
        builder.graph.add(
          new GPUMultiScaleRelief({
            id: 'multi-scale-relief',
            width,
            height,
            elevation,
            settings: settings.view,
            resolution: grid.groundCellSize,
            featureMinimum: state.msrmMinimumFeature,
            featureMaximum: state.msrmMaximumFeature,
            scalingFactor: state.msrmScaling,
            relief: builder.floats('relief')
          })
        );
        stage = builder.finishStage({
          minIntervalMs: 80,
          write: () =>
            settings.parameters.write(
              getGPUMultiScaleReliefParameterValues({
                verticalExaggeration: ctx.options.rvtExaggeration
              })
            )
        });
        break;
      }
      case 'dominance': {
        const settings = builder.settings(GPU_LOCAL_DOMINANCE_PARAMETER_LENGTH);
        builder.graph.add(
          new GPULocalDominance({
            id: 'local-dominance',
            width,
            height,
            elevation,
            settings: settings.view,
            minimumRadius: state.dominanceMinimumRadius,
            maximumRadius: Math.max(
              state.dominanceMaximumRadius,
              state.dominanceMinimumRadius + state.dominanceIncrement
            ),
            radiusIncrement: state.dominanceIncrement,
            angularResolution: Number(state.dominanceAngle),
            dominance: builder.floats('dominance')
          })
        );
        stage = builder.finishStage({
          // 264 taps per pixel: let the GPU finish one run before the next is queued.
          minIntervalMs: 450,
          write: () =>
            settings.parameters.write(
              getGPULocalDominanceParameterValues({
                observerHeight: ctx.options.dominanceObserverHeight,
                verticalExaggeration: ctx.options.rvtExaggeration
              })
            )
        });
        break;
      }
      case 'vat': {
        const settings = builder.settings(GPU_RELIEF_BLEND_LAYER_PARAMETER_LENGTH * 4);
        builder.graph.add(
          new GPUReliefBlend({
            id: 'relief-blend',
            width,
            height,
            // Bottom to top, the VAT input order: hillshade, slope, positive openness, sky-view.
            layers: [
              pixelFloats('look-plain', 'hillshade'),
              pixelFloats('slope', 'slope'),
              pixelFloats('horizon', 'pos'),
              pixelFloats('horizon', 'svf')
            ],
            settings: settings.view,
            color: builder.words('color')
          })
        );
        stage = builder.finishStage({
          dependencies: [
            dependency('look-plain').stage,
            dependency('slope').stage,
            dependency('horizon').stage
          ],
          write: () =>
            settings.parameters.write(getGPUReliefBlendParameterValues(getVatLayers(ctx.options)))
        });
        break;
      }
    }
    return {key: '', stage, builder};
  }

  function getVatLayers(state: ReliefOptions): GPUReliefBlendLayerSettings[] {
    const preset =
      state.vatPreset === 'flat' ? GPU_RELIEF_BLEND_VAT_FLAT : GPU_RELIEF_BLEND_VAT_ARCHAEOLOGICAL;
    const opacityScales = [
      1,
      state.vatSlopeOpacity,
      state.vatOpennessOpacity,
      state.vatSkyViewOpacity
    ];
    const blends = ['preset', state.vatSlopeBlend, state.vatOpennessBlend, state.vatSkyViewBlend];
    return preset.map((layer, index) => ({
      ...layer,
      opacity:
        state.vatPreset === 'hillshade-only' && index > 0
          ? 0
          : Math.min(1, (layer.opacity ?? 1) * opacityScales[index]),
      blendMode:
        blends[index] === 'preset'
          ? layer.blendMode
          : (blends[index] as GPUReliefBlendLayerSettings['blendMode'])
    }));
  }

  // --- Display -----------------------------------------------------------------------------------
  function getStretchRange(stats: ValueStats | null): {low: number; high: number} | undefined {
    const state = ctx.options;
    if (!state.autoStretch || !stats || stats.kind !== 'float' || stats.count === 0)
      return undefined;
    const clip = state.clipPercent / 100;
    let low = stats.quantile(clip);
    let high = stats.quantile(1 - clip);
    if (state.product === 'slrm' || state.product === 'msrm') {
      const extent = Math.max(Math.abs(low), Math.abs(high));
      low = -extent;
      high = extent;
    }
    if (!(high > low)) return undefined;
    return {low, high};
  }

  function applyPaint(): void {
    const state = ctx.options;
    const range = getStretchRange(latestStats);
    session.setPaint(getReliefPaint(state, range));
    if (range) {
      ctx.setLegendExtent('stretch', [range.low, range.high]);
      ctx.setReadout('stretch', `${range.low.toPrecision(3)} to ${range.high.toPrecision(3)}`);
    } else {
      ctx.setReadout('stretch', state.autoStretch ? 'measuring...' : 'fixed range');
    }
  }

  function updateInfoReadouts(): void {
    const state = ctx.options;
    const shifts = getGPULocalDominanceShifts({
      minimumRadius: state.dominanceMinimumRadius,
      maximumRadius: Math.max(
        state.dominanceMaximumRadius,
        state.dominanceMinimumRadius + state.dominanceIncrement
      ),
      radiusIncrement: state.dominanceIncrement,
      angularResolution: Number(state.dominanceAngle)
    });
    ctx.setReadout(
      'dominanceTaps',
      `${shifts.count} taps (${shifts.distanceCount} rings x ${shifts.angleCount} directions)`
    );
    try {
      const radii = getGPUMultiScaleReliefRadii({
        resolution: grid.groundCellSize,
        featureMinimum: state.msrmMinimumFeature,
        featureMaximum: state.msrmMaximumFeature,
        scalingFactor: state.msrmScaling
      });
      ctx.setReadout(
        'msrmRadii',
        `${radii.radii.length} filters, ${radii.firstRadius} to ${radii.lastRadius} px`
      );
    } catch {
      ctx.setReadout('msrmRadii', 'feature range too narrow');
    }
  }

  function showProduct(): void {
    const state = ctx.options;
    const source = SOURCES[state.product];
    let build: ProductBuild | undefined;
    try {
      const key = registry.getFullKey(source.stage, state);
      build = session.getBuild(state.product, key);
      if (!build) {
        const entry = registry.ensure(source.stage, state);
        build = session.addBuild(
          session.createBuild(
            state.product,
            entry.stage,
            entry.builder.getBuffer(source.buffer),
            source.format
          ),
          key
        );
      }
      ctx.setStatus('');
    } catch (error) {
      ctx.setStatus(`Cannot build ${state.product}: ${(error as Error).message}`);
      return;
    }
    if (state.product === 'hillshade' || state.product === 'mdow' || state.product === 'vat') {
      // The shared plain look follows the product (single light or multidirectional).
      registry.markDirty(['look-plain']);
    }
    latestStats = null;
    session.activate(build, getReliefPaint(state));
    applyPaint();
    ctx.requestLayers();
  }

  session.describeHover = ({value, elevation}) =>
    describeCell(ctx.options.product, value, elevation);
  session.onStats = (_id, stats) => {
    latestStats = stats;
    applyPaint();
    if (stats.kind === 'float') {
      ctx.setReadout('median', stats.quantile(0.5).toPrecision(3));
      ctx.setReadout('extent', `${stats.min.toPrecision(3)} to ${stats.max.toPrecision(3)}`);
    }
  };

  ctx.setReadout(
    'grid',
    `${width} x ${height} px, ${grid.groundCellSize.toFixed(2)} m ground per pixel`
  );
  updateInfoReadouts();
  showProduct();

  return {
    getCompiledGraphs: () => session.getCompiledGraphs(),

    setOption(id) {
      const key = id as keyof ReliefOptions;
      if (key === 'product') {
        showProduct();
        return;
      }
      if (COMPILE_OPTIONS.has(key)) {
        updateInfoReadouts();
        timers.push(setTimeout(() => !destroyed && showProduct(), REBUILD_DELAY_MILLISECONDS));
        return;
      }
      const stages = PARAMETER_STAGES[key];
      if (stages) {
        registry.markDirty(stages);
        if (key.startsWith('dominance')) updateInfoReadouts();
        return;
      }
      if (
        key === 'ramp' ||
        key === 'autoStretch' ||
        key === 'clipPercent' ||
        key === 'rangeScale'
      ) {
        applyPaint();
      } else if (key === 'opacity') {
        ctx.requestLayers();
      }
    },

    onAction(id) {
      if (id !== 'measure') return;
      ctx.setReadout('timing', 'measuring...');
      void session.measure().then(results => {
        if (destroyed) return;
        const total = results.reduce((sum, result) => sum + result.milliseconds, 0);
        ctx.setReadout(
          'timing',
          results.length === 0
            ? 'n/a'
            : `${total.toFixed(1)} ms for ${results.length} graph${results.length > 1 ? 's' : ''}, ${formatCount(pixelCount)} px (${results[0].method === 'gpu-timestamps' ? 'GPU timestamps' : 'wall clock'})`
        );
      });
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip: event => session.getTooltip(event),

    encode(commandEncoder) {
      session.encode(commandEncoder);
    },

    getLayers(): Layer[] {
      return session.getLayers({underlay: false, underlayAlpha: 1, alpha: ctx.options.opacity});
    },

    destroy() {
      destroyed = true;
      for (const timer of timers) clearTimeout(timer);
      timers = [];
      session.destroy();
      resources.destroy();
    }
  };
}
