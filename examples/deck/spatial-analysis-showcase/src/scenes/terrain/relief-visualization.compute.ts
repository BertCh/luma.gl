// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPULocalDominanceParameterValues,
  getGPULocalDominanceShifts,
  getGPUMultiScaleReliefParameterValues,
  getGPUMultiScaleReliefRadii,
  getGPUReliefBlendParameterValues,
  getGPUReliefShadingParameterValues,
  getGPUSimpleLocalReliefParameterValues,
  getGPUTerrainCurvatureParameterValues,
  getGPUTerrainDerivativesParameterValues,
  getGPUTerrainHorizonParameterValues,
  getGPUTerrainHorizonStepDistances,
  getGPUTextureShadingParameterValues,
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
  type GPUReliefBlendLayerSettings,
  type GPUReliefShadingStop
} from '@luma.gl/experimental/gpu-terrain';
import {getClassIndexOf} from '../../cartography/class-table';
import {formatCount} from '../../cartography/live-text';
import type {ClassTable, MapAnnotation} from '../../cartography/types';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {sampleRamp} from '../../engine/ramps';
import {getReliefTintColor} from '../../engine/relief';
import {SpatialAnalysisResources} from '../../engine/resources';
import type {
  SceneContext,
  SceneInstance,
  ScenePointerEvent,
  TooltipContent,
  TooltipRow
} from '../scene';
import {
  createColorizeGraph,
  PAINT_PARAMETER_LENGTH,
  writePaint,
  type PaintSpec
} from './b14a-colorize';
import {loadAlpsGrid} from './b14a-grid';
import {ColorRasterLayer} from './b14a-layers';
import {TerrainSession, type ProductBuild, type Stage, type ValueStats} from './b14a-session';
import {StageRegistry, type StageEntry} from './b14a-stage-registry';
import {createDemProbe, createTerrainDemFromGrid} from './cpu-dem';
import {
  createValueClassPaintGraph,
  packValueClassTable,
  VALUE_CLASS_TABLE_LENGTH
} from './relief-visualization-paint';
import {getLocalReliefPercentile} from './relief-visualization-slrm';
import {
  CLASSED_PRODUCTS,
  getClassCounts,
  getCompassName,
  getWindowMeters,
  GROUND_CELL_METERS,
  HILLSHADE_PAINT,
  HILLSHADE_PRODUCTS,
  makeLightDialDiagram,
  makeLocalReliefTable,
  NORTH_FACE_FRAME,
  PRODUCTS,
  RECIPE_CONTRAST,
  RECIPE_LIT_COLOR,
  RECIPE_PALE_TINT,
  RECIPE_SHADE_COLOR,
  RECIPE_STAGES,
  RECIPE_WARM_COOL_STRENGTH,
  REFERENCE_LIGHT,
  roundBreakLimit,
  type ReliefOptions,
  type ReliefProduct
} from './relief-visualization.style';
import {demSampleLine} from './terrain-furniture';
import {createTerrainGround, rasterizeGlacierMask, type TerrainGround} from './terrain-ground';
import {
  glacierLabels,
  loadAlpsContext,
  peakLabels,
  snapPeaksToDem,
  snapLngLatToHighestCell
} from './terrain-places';

export type {ReliefOptions, ReliefProduct};

type StageId =
  | 'horizon'
  | 'texture'
  | 'curvature'
  | 'look-plain'
  | 'look-full'
  | 'recipe'
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
  recipe: [],
  slope: [],
  slrm: [],
  msrm: [],
  dominance: [],
  vat: ['look-plain', 'slope', 'horizon']
};

/** Where each drawn product's raster comes from (the chapter ground is drawn by its own layer). */
const SOURCES: Record<
  Exclude<ReliefProduct, 'ground'>,
  {stage: StageId; buffer: string; format: 'float32' | 'rgba8'}
> = {
  hillshade: {stage: 'look-plain', buffer: 'hillshade', format: 'float32'},
  mdow: {stage: 'look-plain', buffer: 'hillshade', format: 'float32'},
  swing: {stage: 'look-plain', buffer: 'hillshade', format: 'float32'},
  imhof: {stage: 'recipe', buffer: 'color', format: 'rgba8'},
  'imhof-full': {stage: 'look-full', buffer: 'color', format: 'rgba8'},
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

/** The drawings that need the horizon and so can show the 16-spoke fan. */
const HORIZON_PRODUCTS: ReadonlySet<ReliefProduct> = new Set([
  'sky-view',
  'anisotropic',
  'positive-openness',
  'negative-openness'
]);

/** Window radius, in cells, the frozen class limits of the local relief maps come from. */
const REFERENCE_WINDOW_RADIUS = 13;

/** Most horizon sectors the fan buffers hold. */
const MAXIMUM_FAN_SPOKES = 32;

/** The centre of the north-face frame and the half extent of the search for a grazed face. */
const NORTH_FACE_CENTER: [number, number] = [NORTH_FACE_FRAME.longitude, NORTH_FACE_FRAME.latitude];
const FACE_SEARCH_HALF_WIDTH_METERS = 1900;
const FACE_SEARCH_HALF_HEIGHT_METERS = 1100;
const FACE_MINIMUM_ELEVATION_METERS = 2800;
const FACE_MINIMUM_SLOPE_DEGREES = 30;
const FACE_ASPECT_TOLERANCE_DEGREES = 15;

/** Delay before the GPU time of a freshly shown drawing is measured, in milliseconds. */
const MEASURE_DELAY_MILLISECONDS = 1100;

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
  lightAzimuth: ['look-plain', 'look-full', 'recipe'],
  lightAltitude: ['look-plain', 'look-full', 'recipe'],
  zFactor: ['look-plain', 'look-full', 'recipe', 'horizon', 'slope', 'curvature'],
  recipe: ['recipe'],
  imhofSwing: ['look-plain', 'look-full', 'recipe'],
  curvatureStrength: ['look-full'],
  contrastStrength: ['look-full'],
  contrastHighElevation: ['look-full'],
  hillshadeStrength: ['look-full'],
  skyViewStrength: ['look-full'],
  textureStrength: ['look-full'],
  exposure: ['look-full', 'recipe'],
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

/** The tone the pale tint is multiplied into, per ground (light is the chapter paper). */
const TINT_BASE_TONE = {
  light: [0xf3, 0xef, 0xe6],
  // A mid tone: the picture is multiplied by shade, so a dark paper would vanish.
  dark: [0x7d, 0x85, 0x95]
} as const;

/** The pale hypsometric stops of the Imhof recipe, 0-1 channels: paper multiplied by the tint. */
function getPaleTintStops(ground: 'light' | 'dark'): GPUReliefShadingStop[] {
  const base = TINT_BASE_TONE[ground];
  return [1500, 1900, 2300, 2800, 3300, 3800, 4476].map(elevation => {
    const tint = getReliefTintColor('alpine', elevation);
    const color = [0, 1, 2].map(
      channel =>
        (base[channel] * (1 - RECIPE_PALE_TINT + (RECIPE_PALE_TINT * tint[channel]) / 255)) / 255
    );
    return {elevation, color: [color[0], color[1], color[2]] as const};
  });
}

/** `#RRGGBB` to 0-1 channels. */
function getUnitColor(hex: string): [number, number, number] {
  const digits = hex.replace('#', '');
  return [0, 2, 4].map(start => Number.parseInt(digits.slice(start, start + 2), 16) / 255) as [
    number,
    number,
    number
  ];
}

/** Paint of the products drawn on a grey ramp; the percentile range overrides the nominal one. */
export function getReliefPaint(
  state: ReliefOptions,
  range?: {low: number; high: number}
): Partial<PaintSpec> {
  const {product} = state;
  if (HILLSHADE_PRODUCTS.has(product)) {
    return {
      mode: 'ramp',
      ramp: 'grayscale',
      low: HILLSHADE_PAINT.low,
      high: HILLSHADE_PAINT.high,
      alpha: 1,
      fadeMiddle: false
    };
  }
  const nominal = PRODUCTS[product];
  if (nominal.scale === 'classes') return {mode: 'classes', alpha: 1};
  if (nominal.scale === 'picture') return {};
  const low = nominal.scale === 'fit' ? (range?.low ?? nominal.low) : nominal.low;
  const high = nominal.scale === 'fit' ? (range?.high ?? nominal.high) : nominal.high;
  return {mode: 'ramp', ramp: 'grayscale', low, high, alpha: 1, fadeMiddle: false};
}

/** One probed cell as the hover reader reports it (serialised through the session's text). */
type HoverCell = {value: number | null; elevation: number | null; column: number; row: number};

/** The one-line subtitle of the cartouche: the drawing and its key parameters. */
function getSubtitle(state: ReliefOptions): string {
  const radius = Number(state.horizonRadius);
  const horizon = `${state.horizonDirections} rays to ${formatCount(radius * GROUND_CELL_METERS)} m`;
  switch (state.product) {
    case 'hillshade':
      return `Hillshade · light ${state.lightAzimuth}° / ${state.lightAltitude}° · 6.6 m cells`;
    case 'mdow':
      return 'Multidirectional hillshade · USGS weights · 6.6 m cells';
    case 'swing':
      return `Hillshade · Imhof swing up to ${state.imhofSwing}° · 6.6 m cells`;
    case 'texture':
      return `Texture shading · ${state.textureLevels} scales · percentile stretch`;
    case 'sky-view':
    case 'anisotropic':
      return `Sky-view, share of sky · ${horizon} · fixed range`;
    case 'positive-openness':
    case 'negative-openness':
      return `${PRODUCTS[state.product].label}, degrees · ${horizon} · fixed range`;
    case 'slrm':
      return `Local relief, metres · ${formatCount(getWindowMeters(Number(state.slrmRadius)))} m window · ${state.stretch === 'fixed' ? 'fixed' : 'fitted'} breaks`;
    case 'msrm':
      return 'Multi-scale local relief, metres · fitted breaks';
    case 'local-dominance':
      return 'Local dominance, degrees · percentile stretch';
    case 'imhof':
      return `Imhof recipe · ${RECIPE_STAGES[state.recipe]?.label ?? ''}`;
    case 'imhof-full':
      return 'Swiss relief · hillshade, sky-view, texture, curvature, tint';
    case 'vat':
      return 'VAT blend · hillshade, slope, openness, sky-view';
    default:
      return 'Chapter relief ground · built on the CPU, glaciers from OpenStreetMap';
  }
}

/**
 * Relief visualization of the Matterhorn tile. Every drawing is a `Stage` (compiled graphs plus
 * their parameter writes) that re-encodes only when its own parameters, the elevation, or a stage
 * it reads changed; heavy stages are rate limited. Compile-time options (radii, level counts,
 * direction counts, algorithms) rebuild only the stage they belong to and its dependents.
 *
 * A swipe compare needs two colourised rasters at once: the session's product is side B and a
 * scene-local reference stage with its own colorize pass is side A.
 */
export async function createReliefVisualization(
  ctx: SceneContext<ReliefOptions>
): Promise<SceneInstance<ReliefOptions>> {
  const {device} = ctx;
  const [grid, context] = await Promise.all([
    loadAlpsGrid(ctx.datasets.get('alps-dem'), ctx.signal),
    loadAlpsContext(ctx.datasets.get('alps-context'), ctx.signal)
  ]);
  ctx.signal.throwIfAborted();
  const {width, height, pixelCount} = grid;
  const resources = new SpatialAnalysisResources(device, 'relief-visualization');
  const session = new TerrainSession(ctx, resources, grid);
  // The decoded elevation is loaded once; the contributors read it from this buffer.
  session.elevationBuffer.write(grid.cpuElevation);
  session.validityBuffer.write(new Uint32Array(pixelCount).fill(1));
  session.elevationChanged();

  const dem = createTerrainDemFromGrid(grid);
  const probe = createDemProbe(dem);
  const origin: [number, number, number] = [grid.origin[0], grid.origin[1], 0];
  const cell = grid.cellSettings;
  let groundTone: 'light' | 'dark' = ctx.ground();
  const terrainGround: TerrainGround = createTerrainGround({
    dem,
    device,
    ground: groundTone,
    glacierMask: rasterizeGlacierMask(context.glacierGeoJson, dem)
  });
  let groundReady = false;

  const registry: StageRegistry<StageId, ReliefOptions> = new StageRegistry(
    session,
    DEPENDENCIES,
    getStageKey,
    (id, state, stages) => buildStage(id, state, stages)
  );
  let destroyed = false;
  let timers: ReturnType<typeof setTimeout>[] = [];
  let latestStats: ValueStats | null = null;
  let measuring = false;
  let measureTimer: ReturnType<typeof setTimeout> | undefined;

  // --- The class table of the local relief maps --------------------------------------------------
  const classTableParameters = resources.createParameterBuffer(
    'local-relief-classes',
    'float32',
    VALUE_CLASS_TABLE_LENGTH
  );
  let classTable: ClassTable | null = null;
  let classTableKey = '';

  /** The p98 of |SLRM| at the reference window in metres, computed once (without exaggeration). */
  let referencePercentileMeters: number | null = null;

  /** The class limit of the frozen scale: that percentile, times the relief exaggeration. */
  function getReferenceLimit(): number {
    referencePercentileMeters ??= getLocalReliefPercentile(
      grid.cpuElevation,
      width,
      height,
      REFERENCE_WINDOW_RADIUS
    );
    return roundBreakLimit(referencePercentileMeters * ctx.options.rvtExaggeration);
  }

  // --- The reference (left) side of a swipe -----------------------------------------------------
  const reference = buildReferenceStage();
  let referencePaintDirty = true;

  function buildReferenceStage() {
    const builder = session.builder('reference');
    const settings = builder.settings(GPU_RELIEF_SHADING_PARAMETER_LENGTH);
    builder.graph.add(
      new GPUReliefShading({
        id: 'reference-shading',
        width,
        height,
        elevation: builder.elevation(),
        settings: settings.view,
        cellSizeMode: 'web-mercator',
        rowDirection: 'south',
        hillshade: builder.floats('hillshade')
      })
    );
    const stage = builder.finishStage({
      write: () => {
        const lights =
          ctx.options.reference === 'mdow'
            ? {lights: 'mdow' as const, lightWeighting: 'aspect' as const}
            : {
                lights: [REFERENCE_LIGHT] as const,
                lightWeighting: 'fixed' as const
              };
        settings.parameters.write(
          getGPUReliefShadingParameterValues({
            ...cell,
            zFactor: ctx.options.zFactor,
            ...lights,
            hillshadeStrength: 1,
            skyViewStrength: 0,
            textureShadeStrength: 0,
            tintStrength: 0,
            exposure: 1,
            elevationStops: []
          })
        );
      }
    });
    const colors = resources.createBuffer('reference-colors', pixelCount * 4);
    const paintParameters = resources.createParameterBuffer(
      'reference-paint',
      'float32',
      PAINT_PARAMETER_LENGTH
    );
    writePaint(paintParameters, {
      mode: 'ramp',
      ramp: 'grayscale',
      low: HILLSHADE_PAINT.low,
      high: HILLSHADE_PAINT.high,
      alpha: 1
    });
    const paintGraph = createColorizeGraph(
      resources,
      device,
      'reference',
      {buffer: builder.getBuffer('hillshade'), format: 'float32', length: pixelCount},
      paintParameters,
      colors
    );
    return {stage, colors, paintGraph};
  }

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
      case 'look-plain': {
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
            imhofSwing: true,
            hillshade: builder.floats('hillshade')
          })
        );
        stage = builder.finishStage({
          write: () => {
            const o = ctx.options;
            const single = [
              {azimuthDegrees: o.lightAzimuth, altitudeDegrees: o.lightAltitude}
            ] as const;
            const lights =
              o.product === 'mdow'
                ? {lights: 'mdow' as const, lightWeighting: 'aspect' as const}
                : o.product === 'swing'
                  ? {lights: single, lightWeighting: 'imhof-swing' as const}
                  : {lights: single, lightWeighting: 'fixed' as const};
            settings.parameters.write(
              getGPUReliefShadingParameterValues({
                ...cell,
                zFactor: o.zFactor,
                ...lights,
                imhofSwingDegrees: o.product === 'swing' ? o.imhofSwing : 0,
                hillshadeStrength: 1,
                skyViewStrength: 0,
                textureShadeStrength: 0,
                tintStrength: 0,
                exposure: 1,
                elevationStops: []
              })
            );
          }
        });
        break;
      }
      case 'look-full': {
        const settings = builder.settings(GPU_RELIEF_SHADING_PARAMETER_LENGTH);
        builder.graph.add(
          new GPUReliefShading({
            id: 'relief-shading-full',
            width,
            height,
            elevation,
            settings: settings.view,
            cellSizeMode: 'web-mercator',
            rowDirection: 'south',
            hillshade: builder.floats('hillshade'),
            skyViewFactor: pixelFloats('horizon', 'svf'),
            textureShade: pixelFloats('texture', 'textureShade'),
            curvature: pixelFloats('curvature', 'curvature'),
            imhofSwing: true,
            color: builder.words('color')
          })
        );
        stage = builder.finishStage({
          dependencies: [
            dependency('horizon').stage,
            dependency('texture').stage,
            dependency('curvature').stage
          ],
          write: () => {
            const o = ctx.options;
            settings.parameters.write(
              getGPUReliefShadingParameterValues({
                ...cell,
                zFactor: o.zFactor,
                lights: [{azimuthDegrees: o.lightAzimuth, altitudeDegrees: o.lightAltitude}],
                lightWeighting: 'imhof-swing',
                imhofSwingDegrees: o.imhofSwing,
                curvatureStrength: o.curvatureStrength,
                contrastLowElevation: RECIPE_CONTRAST.from,
                contrastHighElevation: o.contrastHighElevation,
                contrastStrength: o.contrastStrength,
                hillshadeStrength: o.hillshadeStrength,
                skyViewStrength: o.skyViewStrength,
                textureShadeStrength: o.textureStrength,
                exposure: o.exposure,
                tintStrength: o.tintStrength,
                warmColor: getUnitColor(RECIPE_LIT_COLOR),
                coolColor: getUnitColor(RECIPE_SHADE_COLOR),
                elevationStops: getPaleTintStops(groundTone)
              })
            );
          }
        });
        break;
      }
      case 'recipe': {
        const settings = builder.settings(GPU_RELIEF_SHADING_PARAMETER_LENGTH);
        builder.graph.add(
          new GPUReliefShading({
            id: 'relief-recipe',
            width,
            height,
            elevation,
            settings: settings.view,
            cellSizeMode: 'web-mercator',
            rowDirection: 'south',
            imhofSwing: true,
            hillshade: builder.floats('hillshade'),
            color: builder.words('color')
          })
        );
        stage = builder.finishStage({
          write: () => {
            const o = ctx.options;
            const spec = RECIPE_STAGES[Math.min(Math.max(o.recipe, 0), RECIPE_STAGES.length - 1)];
            settings.parameters.write(
              getGPUReliefShadingParameterValues({
                ...cell,
                zFactor: o.zFactor,
                lights: [{azimuthDegrees: o.lightAzimuth, altitudeDegrees: o.lightAltitude}],
                lightWeighting: spec.swing ? 'imhof-swing' : 'fixed',
                imhofSwingDegrees: spec.swing ? o.imhofSwing : 0,
                hillshadeStrength: 1,
                skyViewStrength: 0,
                textureShadeStrength: 0,
                exposure: o.exposure,
                tintStrength: spec.warmCool ? RECIPE_WARM_COOL_STRENGTH : 0,
                warmColor: getUnitColor(RECIPE_LIT_COLOR),
                coolColor: getUnitColor(RECIPE_SHADE_COLOR),
                contrastLowElevation: RECIPE_CONTRAST.from,
                contrastHighElevation: RECIPE_CONTRAST.to,
                contrastStrength: spec.contrast ? RECIPE_CONTRAST.strength : 0,
                elevationStops: spec.tint ? getPaleTintStops(groundTone) : []
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
            radius: Number(state.slrmRadius),
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
  /** Percentile range of the displayed raster, symmetric for the signed local relief. */
  function getStretchRange(stats: ValueStats | null): {low: number; high: number} | undefined {
    if (!stats || stats.kind !== 'float' || stats.count === 0) return undefined;
    const clip = ctx.options.clipPercent / 100;
    const low = stats.quantile(clip);
    const high = stats.quantile(1 - clip);
    return high > low ? {low, high} : undefined;
  }

  /** Rebuilds the class table of the local relief maps when its limit or ground changed. */
  function updateClassTable(): void {
    const state = ctx.options;
    if (!CLASSED_PRODUCTS.has(state.product)) return;
    const fitted = state.product === 'msrm' || state.stretch === 'fit';
    let limit = getReferenceLimit();
    if (fitted) {
      const range = getStretchRange(latestStats);
      if (range) limit = Math.max(Math.abs(range.low), Math.abs(range.high));
    }
    const table = makeLocalReliefTable(limit, groundTone);
    const key = `${groundTone}|${table.breaks.join(',')}`;
    if (key !== classTableKey) {
      classTableKey = key;
      classTable = table;
      classTableParameters.write(packValueClassTable(table));
      ctx.setLegendData('table', table);
      ctx.setLegendData(
        'tableNote',
        fitted
          ? 'Fit to view: the limits follow the window drawn, so windows are not comparable'
          : `Fixed: limits from the ${formatCount(getWindowMeters(REFERENCE_WINDOW_RADIUS))} m window, the same for every window`
      );
      const limits = table.breaks.filter(value => value > 0);
      ctx.setReadout(
        'clip',
        `${limits.map(value => `±${value < 10 ? value.toFixed(1) : value.toFixed(0)}`).join(', ')} m${fitted ? ' (fitted)' : ' (fixed)'}`
      );
    }
    if (latestStats?.kind === 'float' && classTable) {
      ctx.setLegendData(
        'counts',
        getClassCounts(classTable, latestStats.histogram, latestStats.min, latestStats.max)
      );
    }
  }

  function applyPaint(): void {
    const state = ctx.options;
    const nominal = PRODUCTS[state.product];
    const range = nominal.scale === 'fit' ? getStretchRange(latestStats) : undefined;
    if (nominal.scale === 'classes') updateClassTable();
    session.setPaint(getReliefPaint(state, range));
    if (range) {
      ctx.setLegendExtent('stretch', [range.low, range.high]);
      ctx.setReadout('stretch', `${range.low.toPrecision(3)} to ${range.high.toPrecision(3)}`);
    } else if (nominal.scale === 'fit') {
      ctx.setReadout('stretch', 'measuring...');
    } else {
      ctx.setReadout('stretch', nominal.scale === 'fixed' ? 'fixed range' : 'not a ramp');
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

  /** The readouts that follow the option state, not the GPU. */
  function updateStateReadouts(): void {
    const state = ctx.options;
    const lightCount = state.product === 'mdow' ? 4 : 1;
    ctx.setReadout('lights', `${lightCount} ${lightCount === 1 ? 'light' : 'lights'}`);
    const directions = Number(state.horizonDirections);
    const radius = Number(state.horizonRadius);
    if (state.horizonAlgorithm === 'sweep') {
      ctx.setReadout('svfTaps', `${directions} directions, exact hull sweep`);
    } else {
      const steps = getGPUTerrainHorizonStepDistances(radius, state.horizonGrowth).length;
      ctx.setReadout(
        'svfTaps',
        `${directions} directions x ${steps} steps = ${formatCount(directions * steps)} taps per cell`
      );
    }
    ctx.setReadout(
      'windowMeters',
      state.product === 'slrm' ? getWindowMeters(Number(state.slrmRadius)) : null
    );
    ctx.setReadout('recipeStage', RECIPE_STAGES[state.recipe]?.label ?? null);
  }

  /** The cartouche line and the scale bar tick that follow the option state. */
  function updateFurniture(): void {
    const state = ctx.options;
    const ticks: number[] = [];
    if (state.product === 'slrm') ticks.push(getWindowMeters(Number(state.slrmRadius)));
    if (HORIZON_PRODUCTS.has(state.product)) {
      ticks.push(Number(state.horizonRadius) * GROUND_CELL_METERS);
    }
    ctx.setFurniture({
      title: {subtitle: getSubtitle(state), sample: demSampleLine(grid)},
      scaleBar: {units: 'metric', ticks}
    });
  }

  function updateLightDial(): void {
    const state = ctx.options;
    ctx.setChart('lightDial', makeLightDialDiagram(state.lightAzimuth, state.lightAltitude));
  }

  function showProduct(): void {
    const state = ctx.options;
    updateStateReadouts();
    if (state.product === 'ground') {
      latestStats = null;
      if (!groundReady) {
        ctx.setStatus('Building the chapter relief on the CPU...');
        void terrainGround.prepare().then(() => {
          if (destroyed) return;
          groundReady = true;
          ctx.setStatus('');
          ctx.requestLayers();
        });
      }
      ctx.requestLayers();
      return;
    }
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
      if (CLASSED_PRODUCTS.has(state.product)) {
        if (!build.paintGraph) {
          build.paintGraph = createValueClassPaintGraph(
            resources,
            device,
            state.product,
            {buffer: build.value.buffer, length: pixelCount},
            classTableParameters,
            session.colorsBuffer
          );
        }
        // The grey multidirectional hillshade under the classes, computed once.
        session.enableUnderlay();
      }
      ctx.setStatus('');
    } catch (error) {
      ctx.setStatus(`Cannot build ${state.product}: ${(error as Error).message}`);
      return;
    }
    if (state.product in {hillshade: 1, mdow: 1, swing: 1, vat: 1}) {
      // The shared plain look follows the product (single light, multidirectional or swung).
      registry.markDirty(['look-plain']);
    }
    latestStats = null;
    session.activate(build, getReliefPaint(state));
    applyPaint();
    ctx.setCost({records: pixelCount, passes: session.getCompiledGraphs().length});
    ctx.requestLayers();
  }

  // --- GPU time ----------------------------------------------------------------------------------
  function measureNow(): void {
    if (measuring || destroyed || ctx.options.product === 'ground') return;
    measuring = true;
    ctx.setReadout('timing', 'measuring...');
    void session
      .measure()
      .then(results => {
        if (destroyed) return;
        const total = results.reduce((sum, result) => sum + result.milliseconds, 0);
        ctx.setReadout(
          'timing',
          results.length === 0
            ? 'n/a'
            : `${total.toFixed(1)} ms for ${results.length} graph${results.length > 1 ? 's' : ''}, ${formatCount(pixelCount)} cells (${results[0].method === 'gpu-timestamps' ? 'GPU timestamps' : 'wall clock'})`
        );
        ctx.setReadout('gpuTime', results.length === 0 ? null : total);
      })
      .catch(() => {
        // The device or the scene was destroyed while timing.
      })
      .finally(() => {
        measuring = false;
      });
  }

  /** Measures the drawing a moment after it settled, so a step opens with its cost filled in. */
  function scheduleMeasure(): void {
    clearTimeout(measureTimer);
    measureTimer = setTimeout(measureNow, MEASURE_DELAY_MILLISECONDS);
  }

  // --- The pinned cell and its horizon fan ------------------------------------------------------
  const fanSegments = resources.createBuffer('fan-segments', MAXIMUM_FAN_SPOKES * 16);
  const fanValues = resources.createBuffer('fan-values', MAXIMUM_FAN_SPOKES * 4);
  let fanSpokes = 0;
  let pinned: [number, number] | null = null;

  /** Draws the horizon rays of a cell: the algorithm in motion, one spoke per azimuth sector. */
  function placeFan(lngLat: readonly [number, number]): void {
    const state = ctx.options;
    const pixel = grid.getPixel(lngLat[0], lngLat[1]);
    if (!pixel) return;
    const center = grid.getLongitudeLatitude(pixel[0], pixel[1]);
    pinned = center;
    const directions = Math.min(MAXIMUM_FAN_SPOKES, Number(state.horizonDirections));
    const reachMeters = Number(state.horizonRadius) * grid.getGroundCellSize(pixel[1]);
    const angles = probe.horizonAngles(center, directions, reachMeters, {eyeHeight: 0});
    const [x, y] = grid.projection.project(center[0], center[1]);
    const segments = new Float32Array(directions * 4);
    const values = new Float32Array(directions);
    let sineSum = 0;
    for (let spoke = 0; spoke < directions; spoke++) {
      const azimuth = (spoke * 2 * Math.PI) / directions;
      segments.set(
        [x, y, x + Math.sin(azimuth) * reachMeters, y + Math.cos(azimuth) * reachMeters],
        spoke * 4
      );
      values[spoke] = angles[spoke];
      sineSum += Math.sin((Math.max(angles[spoke], 0) * Math.PI) / 180);
    }
    fanSegments.write(segments);
    fanValues.write(values);
    fanSpokes = directions;
    const skyView = 1 - sineSum / directions;
    ctx.setChart('horizonProfile', {
      kind: 'bars',
      values: Array.from(angles, angle => Math.round(angle * 10) / 10),
      labels: Array.from({length: directions}, (_, spoke) =>
        directions <= 16 || spoke % 2 === 0
          ? getCompassName((spoke * 360) / directions)
              .split('-')
              .map(part => part.charAt(0).toUpperCase())
              .join('')
          : ''
      ),
      title: `Horizon angle by direction · sky-view ${skyView.toFixed(2)} (CPU)`,
      yLabel: 'degrees above level',
      description: `The horizon angle in each of ${directions} directions around the pinned cell; the sky-view factor of these rays is ${skyView.toFixed(2)}.`
    });
    ctx.requestLayers();
    updateAnnotations(true);
  }

  // --- Annotations -------------------------------------------------------------------------------
  const snappedCache = new Map<string, MapAnnotation[]>();
  let annotationKey = '';

  /** Which set of place labels the current drawing frames: the north face, the moraines or home. */
  function getLabelSet(state: ReliefOptions): 'home' | 'north-face' | 'moraines' {
    if (state.reference === 'single') return 'north-face';
    if (CLASSED_PRODUCTS.has(state.product)) return 'moraines';
    return 'home';
  }

  function getPlaceLabels(set: 'home' | 'north-face' | 'moraines'): MapAnnotation[] {
    const cached = snappedCache.get(set);
    if (cached) return cached;
    const names =
      set === 'north-face'
        ? ['matterhorn']
        : set === 'moraines'
          ? ['gornergrat']
          : ['matterhorn', 'gornergrat', 'breithorn'];
    const peaks = snapPeaksToDem(peakLabels(context, {names}), dem);
    const glaciers =
      set === 'north-face'
        ? []
        : glacierLabels(context, {
            window: grid.lngLatBounds,
            names: ['Gornergletscher'],
            max: 1,
            minWindowShare: 0
          });
    const labels = [...peaks, ...glaciers];
    snappedCache.set(set, labels);
    return labels;
  }

  /** The face the single light only grazes, found once on the CPU (see `findGrazedFace`). */
  let grazedFace: {lngLat: [number, number]; slope: number} | null | undefined;

  /**
   * The steepest high cell, in the north-face frame, whose aspect is within 15 degrees of a
   * direction perpendicular to the light azimuth (north-east or south-west for a north-west
   * light): a face parallel to the ray, so both flanks of a ridge along it get the same shade.
   */
  function findGrazedFace(): {lngLat: [number, number]; slope: number} | null {
    const centerPixel = grid.getPixel(NORTH_FACE_CENTER[0], NORTH_FACE_CENTER[1]);
    if (!centerPixel) return null;
    const halfColumns = Math.round(FACE_SEARCH_HALF_WIDTH_METERS / grid.groundCellSize);
    const halfRows = Math.round(FACE_SEARCH_HALF_HEIGHT_METERS / grid.groundCellSize);
    const perpendicular = [
      (REFERENCE_LIGHT.azimuthDegrees + 90) % 360,
      (REFERENCE_LIGHT.azimuthDegrees + 270) % 360
    ];
    let best: {column: number; row: number; slope: number} | null = null;
    for (
      let row = Math.max(2, centerPixel[1] - halfRows);
      row < Math.min(height - 2, centerPixel[1] + halfRows);
      row++
    ) {
      for (
        let column = Math.max(2, centerPixel[0] - halfColumns);
        column < Math.min(width - 2, centerPixel[0] + halfColumns);
        column++
      ) {
        if (grid.cpuElevation[row * width + column] < FACE_MINIMUM_ELEVATION_METERS) continue;
        const horn = probe.hornAt(column, row);
        if (!(horn.slopeDeg > (best?.slope ?? FACE_MINIMUM_SLOPE_DEGREES))) continue;
        if (!Number.isFinite(horn.aspectDeg)) continue;
        const grazed = perpendicular.some(
          direction =>
            Math.abs(((horn.aspectDeg - direction + 540) % 360) - 180) <=
            FACE_ASPECT_TOLERANCE_DEGREES
        );
        if (grazed) best = {column, row, slope: horn.slopeDeg};
      }
    }
    return best
      ? {lngLat: grid.getLongitudeLatitude(best.column, best.row), slope: best.slope}
      : null;
  }

  /** Sets the data-driven annotation groups: place labels, the light arrow, the face note, the pin. */
  function updateAnnotations(force = false): void {
    const state = ctx.options;
    const set = getLabelSet(state);
    const lightArrow =
      state.reference === 'none' && (state.product === 'hillshade' || state.product === 'swing');
    const pinVisible = HORIZON_PRODUCTS.has(state.product) && pinned !== null;
    const key = `${set}|${lightArrow ? state.lightAzimuth : '-'}|${state.reference}|${pinVisible ? `${state.horizonRadius}|${pinned}` : '-'}`;
    if (!force && key === annotationKey) return;
    annotationKey = key;
    ctx.setAnnotations('places', getPlaceLabels(set));
    if (lightArrow) {
      const tail = grid.projection.unproject(
        Math.sin((state.lightAzimuth * Math.PI) / 180) * 3600,
        Math.cos((state.lightAzimuth * Math.PI) / 180) * 3600
      );
      const head = grid.projection.unproject(
        Math.sin((state.lightAzimuth * Math.PI) / 180) * 2500,
        Math.cos((state.lightAzimuth * Math.PI) / 180) * 2500
      );
      ctx.setAnnotations('light', [
        {
          kind: 'arrow',
          id: 'light-arrow',
          from: tail,
          to: head,
          text: 'light',
          tone: 'signal'
        }
      ]);
    } else {
      ctx.setAnnotations('light', null);
    }
    if (state.reference === 'single') {
      grazedFace ??= findGrazedFace();
      const face = grazedFace;
      const matterhorn = snapLngLatToHighestCell(dem, NORTH_FACE_CENTER, 400);
      const coordinate = face?.lngLat ?? matterhorn?.lngLat ?? NORTH_FACE_CENTER;
      ctx.setAnnotations('face', [
        {
          kind: 'note',
          id: 'grazed-face',
          coordinate,
          title: 'Flat under one light',
          text: face
            ? `a ${Math.round(face.slope)}° face the ray only grazes`
            : 'the north face is nearly parallel to the ray',
          tone: 'accent'
        }
      ]);
    } else {
      ctx.setAnnotations('face', null);
    }
    if (pinVisible && pinned) {
      ctx.setAnnotations('pin', [
        {
          kind: 'ring',
          id: 'horizon-pin',
          coordinate: pinned,
          radiusMeters: Number(state.horizonRadius) * GROUND_CELL_METERS,
          text: `${state.horizonRadius} px`,
          dashed: true,
          tone: 'signal'
        }
      ]);
    } else {
      ctx.setAnnotations('pin', null);
    }
  }

  // --- Tooltip -----------------------------------------------------------------------------------
  session.describeHover = ({value, elevation, column, row}) => {
    // The probe resolves a frame after the pointer: show the card again when it has.
    queueMicrotask(() => !destroyed && ctx.refreshTooltip());
    return JSON.stringify({
      value: Number.isFinite(value) ? value : null,
      elevation: Number.isFinite(elevation) ? elevation : null,
      column,
      row
    } satisfies HoverCell);
  };
  session.onStats = (_id, stats) => {
    latestStats = stats;
    applyPaint();
    if (stats.kind === 'float') {
      const state = ctx.options;
      const median = stats.quantile(0.5);
      ctx.setReadout(
        'median',
        state.product === 'sky-view' || state.product === 'anisotropic'
          ? `${(median * 100).toFixed(0)} % of the sky`
          : state.product === 'positive-openness' || state.product === 'negative-openness'
            ? `${median.toFixed(0)}°`
            : median.toPrecision(2)
      );
      ctx.setReadout('extent', `${stats.min.toPrecision(3)} to ${stats.max.toPrecision(3)}`);
    }
  };

  /** The grey of a ramp position as a tooltip swatch. */
  function getGreySwatch(t: number): readonly [number, number, number, number] {
    const [r, g, b] = sampleRamp('grayscale', t);
    return [r, g, b, 255];
  }

  /** The mapped-value row of the tooltip for the displayed drawing, with its class swatch. */
  function getValueRow(state: ReliefOptions, value: number): TooltipRow | null {
    const spec = PRODUCTS[state.product];
    if (HILLSHADE_PRODUCTS.has(state.product)) {
      const share = Math.min(
        Math.max((value - HILLSHADE_PAINT.low) / (HILLSHADE_PAINT.high - HILLSHADE_PAINT.low), 0),
        1
      );
      return {
        label: 'Brightness',
        value: Math.round(value * 100),
        unit: '% of full light',
        swatch: getGreySwatch(share),
        emphasis: true
      };
    }
    if (spec.scale === 'classes' && classTable) {
      const index = Math.max(getClassIndexOf(classTable, value), 0);
      const color = classTable.colors[index];
      return {
        label: 'Height vs window mean',
        value: `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(1)}`,
        unit: 'm',
        swatch: [color[0], color[1], color[2], 255],
        emphasis: true
      };
    }
    const t = Math.min(Math.max((value - spec.low) / (spec.high - spec.low), 0), 1);
    const sky = state.product === 'sky-view' || state.product === 'anisotropic';
    return {
      label: spec.label,
      value: sky ? Math.round(value * 100) : value.toPrecision(3),
      unit: sky ? '% of the sky' : spec.unit,
      swatch: getGreySwatch(t),
      emphasis: true
    };
  }

  function getTooltip(event: ScenePointerEvent): TooltipContent | null {
    const state = ctx.options;
    const text = state.product === 'ground' ? null : session.getTooltip(event);
    const pixel = event.coordinate ? grid.getPixel(event.coordinate[0], event.coordinate[1]) : null;
    if (!pixel) return null;
    const horn = probe.hornAt(pixel[0], pixel[1]);
    const elevation = grid.cpuElevation[pixel[1] * width + pixel[0]];
    const rows: TooltipRow[] = [];
    if (text) {
      const cellValue = JSON.parse(text) as HoverCell;
      if (cellValue.value !== null) {
        const row = getValueRow(state, cellValue.value);
        if (row) rows.push(row);
      }
    }
    rows.push({
      label: 'Elevation',
      value: Math.round(elevation).toLocaleString('en-US'),
      unit: 'm'
    });
    rows.push({label: 'Slope', value: Math.round(horn.slopeDeg), unit: '°'});
    if (Number.isFinite(horn.aspectDeg) && horn.slopeDeg >= 5) {
      rows.push({label: 'Faces', value: getCompassName(horn.aspectDeg)});
    }
    return {title: PRODUCTS[state.product].label, rows};
  }

  // --- Start -------------------------------------------------------------------------------------
  ctx.setReadout(
    'grid',
    `${width} x ${height} cells, ${grid.groundCellSize.toFixed(2)} m ground per cell`
  );
  ctx.setLegendData('ground', groundTone);
  updateInfoReadouts();
  updateStateReadouts();
  updateLightDial();
  showProduct();
  updateFurniture();
  updateAnnotations(true);
  // The reference scale of the local relief step is ready before the step opens.
  timers.push(
    setTimeout(() => {
      if (!destroyed) getReferenceLimit();
    }, 400)
  );
  scheduleMeasure();

  return {
    getCompiledGraphs: () =>
      [
        ...session.getCompiledGraphs(),
        reference.paintGraph
      ] as unknown as CompiledGPUCommandGraph<never>[],

    setOption(id) {
      const key = id as keyof ReliefOptions;
      if (key === 'product') {
        showProduct();
        updateFurniture();
        updateAnnotations();
        scheduleMeasure();
        return;
      }
      if (key === 'reference') {
        reference.stage.markDirty();
        referencePaintDirty = true;
        updateAnnotations();
        ctx.requestLayers();
        return;
      }
      if (COMPILE_OPTIONS.has(key)) {
        updateInfoReadouts();
        updateStateReadouts();
        timers.push(
          setTimeout(() => {
            if (destroyed) return;
            showProduct();
            updateFurniture();
            if (pinned && key.startsWith('horizon')) placeFan(pinned);
            scheduleMeasure();
          }, REBUILD_DELAY_MILLISECONDS)
        );
        return;
      }
      const stages = PARAMETER_STAGES[key];
      if (stages) {
        registry.markDirty(stages);
        if (key === 'zFactor') reference.stage.markDirty();
        if (key.startsWith('dominance')) updateInfoReadouts();
        if (key === 'recipe') updateStateReadouts();
      }
      if (key === 'lightAzimuth' || key === 'lightAltitude') updateLightDial();
      if (key === 'stretch' || key === 'clipPercent') applyPaint();
      if (key === 'opacity') ctx.requestLayers();
      updateFurniture();
      updateAnnotations();
    },

    onAction(id) {
      if (id === 'measure') measureNow();
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    onGroundChange(next) {
      groundTone = next;
      terrainGround.setGround(next);
      ctx.setLegendData('ground', next);
      registry.markDirty(['recipe', 'look-full']);
      classTableKey = '';
      applyPaint();
      ctx.requestLayers();
    },

    onCompareChange() {
      ctx.requestLayers();
    },

    onClick(event) {
      if (!event.coordinate || !HORIZON_PRODUCTS.has(ctx.options.product)) return false;
      placeFan(event.coordinate);
      return true;
    },

    getTooltip,

    encode(commandEncoder) {
      session.encode(commandEncoder);
      if (ctx.options.reference !== 'none') {
        if (reference.stage.run(commandEncoder, performance.now(), 1)) referencePaintDirty = true;
        if (referencePaintDirty) {
          reference.paintGraph.encode(commandEncoder, {parameters: undefined});
          referencePaintDirty = false;
        }
      }
    },

    getLayers(): Layer[] {
      const state = ctx.options;
      const layers: Layer[] = [];
      if (state.product === 'ground') {
        if (groundReady) layers.push(terrainGround.getLayer({opacity: 1}));
      } else {
        const productLayers = session.getLayers({
          underlay: CLASSED_PRODUCTS.has(state.product),
          underlayAlpha: 1,
          alpha: state.opacity
        });
        if (state.reference !== 'none' && ctx.getCompare() !== null) {
          layers.push(
            new ColorRasterLayer({
              id: 'reference-product',
              coordinateOrigin: origin,
              gridSize: [width, height],
              bounds: grid.bounds,
              colors: reference.colors,
              alpha: state.opacity,
              compareSide: 'a'
            }),
            ...productLayers.map(layer =>
              layer.id === 'terrain-product'
                ? (layer as ColorRasterLayer).clone({compareSide: 'b'})
                : layer
            )
          );
        } else {
          layers.push(...productLayers);
        }
      }
      if (fanSpokes > 0 && HORIZON_PRODUCTS.has(state.product)) {
        const paper = groundTone === 'dark' ? [20, 23, 28, 255] : [243, 239, 230, 255];
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'horizon-fan',
            coordinateOrigin: origin,
            segments: fanSegments,
            instanceCount: fanSpokes,
            values: fanValues,
            valueFormat: 'float32',
            colormap: 'grayscale',
            valueRange: [0, 50],
            reverseRamp: groundTone === 'dark',
            widthPixels: 3,
            outlineColor: paper as [number, number, number, number],
            outlineWidthPixels: 1.4,
            cap: 'round'
          })
        );
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      clearTimeout(measureTimer);
      for (const timer of timers) clearTimeout(timer);
      timers = [];
      terrainGround.destroy();
      session.destroy();
      resources.destroy();
    }
  };
}
