// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {
  getGPUGeomorphonsParameterValues,
  getGPUTerrainCurvatureParameterValues,
  getGPUTerrainWeissLandformsParameterValues,
  GPU_GEOMORPHONS_PARAMETER_LENGTH,
  GPU_TERRAIN_CURVATURE_PARAMETER_LENGTH,
  GPU_TERRAIN_WEISS_LANDFORMS_PARAMETER_LENGTH,
  GPUGeomorphons,
  GPUTerrainCurvature,
  GPUTerrainTopographicPosition,
  GPUTerrainWeissLandforms,
  type GPUTerrainCurvatureKind
} from '@luma.gl/experimental/gpu-terrain';
import {addKernelPass} from '../../engine/mode-kernels';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import type {SceneContext, SceneInstance} from '../scene';
import type {PaintSpec} from './b14a-colorize';
import {loadAlpsGrid} from './b14a-grid';
import {TerrainSession, type ProductBuild, type ValueStats} from './b14a-session';
import {
  CURVATURE_KINDS,
  GEOMORPHON_CLASSES,
  SCALE_PRESETS,
  WEISS_CLASSES,
  type LandformOptions,
  type LandformProduct
} from './landforms.style';

export type {LandformOptions, LandformProduct};

const REBUILD_DELAY_MILLISECONDS = 250;
const QUANTUMS = {'4': 1 / 4, '64': 1 / 64, '256': 1 / 256} as const;

/** Compile-time options of each product, as a key. */
function getConfigKey(state: LandformOptions): string {
  switch (state.product) {
    case 'curvature':
      return `${state.curvatureKind}|${state.curvatureMethod}|${state.curvatureBorder}|${state.ringRadiusInner}|${state.ringRadiusOuter}|${state.ringSquash}`;
    case 'geomorphons':
      return `${state.geomorphonView}|${state.geomorphonRadius}|${state.geomorphonSkip}|${state.geomorphonComparison}`;
    case 'tpi':
    case 'dev':
    case 'devmax':
    case 'scale':
      return `${state.scalePreset}|${state.innerFraction}|${state.quantum}`;
    case 'weiss':
      return `${state.weissSmall}|${state.weissLarge}|${state.weissStandardization}|${state.quantum}`;
  }
}

const SYMMETRIC_PRODUCTS = new Set<LandformProduct>(['curvature', 'tpi']);

/** Paint of the current product given the percentile range of its histogram, if known. */
export function getLandformPaint(
  state: LandformOptions,
  groundMeters: number,
  range?: {low: number; high: number}
): Partial<PaintSpec> {
  switch (state.product) {
    case 'curvature':
    case 'tpi': {
      const extent =
        (range
          ? Math.max(Math.abs(range.low), Math.abs(range.high))
          : state.product === 'tpi'
            ? 10
            : 0.1) * state.rangeScale;
      return {
        mode: 'ramp',
        ramp: 'diverging',
        low: -extent,
        high: extent,
        alpha: 1,
        fadeMiddle: true
      };
    }
    case 'dev':
    case 'devmax': {
      const extent = 2.5 * state.rangeScale;
      return {
        mode: 'ramp',
        ramp: 'diverging',
        low: -extent,
        high: extent,
        alpha: 1,
        fadeMiddle: true
      };
    }
    case 'scale': {
      const radii = SCALE_PRESETS[state.scalePreset];
      return {
        mode: 'ramp',
        ramp: state.ramp,
        low: 0,
        high: radii[radii.length - 1] * groundMeters * state.rangeScale,
        alpha: 1,
        fadeMiddle: false
      };
    }
    case 'geomorphons':
      return state.geomorphonView === 'forms'
        ? {mode: 'classes', alpha: 1}
        : {
            mode: 'ramp',
            ramp: state.ramp,
            low: 0,
            high: 6561 * state.rangeScale,
            alpha: 1,
            fadeMiddle: false
          };
    case 'weiss':
      return {mode: 'classes', alpha: 1};
  }
}

function describeCell(
  state: LandformOptions,
  value: number,
  elevation: number,
  groundMeters: number
): string | null {
  const elevationText = Number.isFinite(elevation) ? `\nElevation ${elevation.toFixed(1)} m` : '';
  switch (state.product) {
    case 'geomorphons': {
      if (state.geomorphonView === 'ternary') {
        return value > 0
          ? `Ternary pattern code ${value}${elevationText}`
          : `No pattern (edge)${elevationText}`;
      }
      const entry = GEOMORPHON_CLASSES[value - 1];
      return entry
        ? `${entry.label}: ${entry.help}${elevationText}`
        : `No class (edge)${elevationText}`;
    }
    case 'weiss': {
      const entry = WEISS_CLASSES[value - 1];
      return entry ? `${entry.label}${elevationText}` : `No class (nodata)${elevationText}`;
    }
    default:
      break;
  }
  if (!Number.isFinite(value)) return `No value (edge or nodata)${elevationText}`;
  switch (state.product) {
    case 'curvature': {
      const kind =
        CURVATURE_KINDS.find(entry => entry.value === state.curvatureKind)?.label ?? 'Curvature';
      return `${kind}: ${value.toExponential(2)} 1/m${value > 0 ? ' (convex)' : value < 0 ? ' (concave)' : ''}${elevationText}`;
    }
    case 'tpi': {
      const radius = SCALE_PRESETS[state.scalePreset][state.scaleIndex];
      return `TPI over ${(radius * groundMeters).toFixed(0)} m: ${value.toFixed(1)} m ${value >= 0 ? 'above' : 'below'} the surroundings${elevationText}`;
    }
    case 'dev': {
      const radius = SCALE_PRESETS[state.scalePreset][state.scaleIndex];
      return `DEV over ${(radius * groundMeters).toFixed(0)} m: ${value.toFixed(2)} standard deviations${elevationText}`;
    }
    case 'devmax':
      return `DEVmax ${value.toFixed(2)} standard deviations${elevationText}`;
    case 'scale':
      return `Strongest landform at ${value.toFixed(0)} m window radius${elevationText}`;
    default:
      return null;
  }
}

/**
 * Landforms of the Matterhorn tile: each product is a stage compiled the first time it is shown
 * and again only when a compile-time option of that product changes. Per-frame options (the
 * geomorphon flatness angle, the Weiss thresholds, the TPI scale shown, curvature gains) are
 * parameter-buffer writes.
 */
export async function createLandforms(
  ctx: SceneContext<LandformOptions>
): Promise<SceneInstance<LandformOptions>> {
  const {device} = ctx;
  const grid = await loadAlpsGrid(ctx.datasets.get('alps-dem'), ctx.signal);
  ctx.signal.throwIfAborted();
  const {width, height, pixelCount} = grid;
  const resources = new SpatialAnalysisResources(device, 'landforms');
  const session = new TerrainSession(ctx, resources, grid);
  session.enableUnderlay();
  session.elevationBuffer.write(grid.cpuElevation);
  session.validityBuffer.write(new Uint32Array(pixelCount).fill(1));
  session.elevationChanged();
  const cell = grid.cellSettings;
  const groundMeters = grid.groundCellSize;
  let destroyed = false;
  let timers: ReturnType<typeof setTimeout>[] = [];
  let latestStats: ValueStats | null = null;

  function buildProduct(state: LandformOptions): ProductBuild {
    const {product} = state;
    const builder = session.builder(product);
    const elevation = builder.elevation();

    if (product === 'curvature') {
      const settings = builder.settings(GPU_TERRAIN_CURVATURE_PARAMETER_LENGTH);
      const output = builder.floats('curvature');
      const isRing = state.curvatureKind === 'ring-multi-radius';
      const outerRadius = Math.max(state.ringRadiusOuter, state.ringRadiusInner + 1);
      builder.graph.add(
        new GPUTerrainCurvature({
          id: 'curvature',
          width,
          height,
          elevation,
          settings: settings.view,
          method: state.curvatureMethod,
          ...(isRing
            ? {
                ringCurvature: output,
                ringRadii: [state.ringRadiusInner, outerRadius],
                ringSquash: state.ringSquash ? ('pade-tanh' as const) : ('none' as const)
              }
            : {curvatures: {[state.curvatureKind as GPUTerrainCurvatureKind]: output}}),
          cellSizeMode: 'web-mercator',
          rowDirection: 'south',
          borderMode: state.curvatureBorder
        })
      );
      return builder.finish({
        value: 'curvature',
        format: 'float32',
        write: () =>
          settings.parameters.write(
            getGPUTerrainCurvatureParameterValues({
              ...cell,
              zFactor: ctx.options.zFactor,
              flatGradient: Math.max(ctx.options.flatGradient, 1e-6),
              ringGains: [ctx.options.ringGainInner, ctx.options.ringGainOuter]
            })
          )
      });
    }

    if (product === 'geomorphons') {
      const settings = builder.settings(GPU_GEOMORPHONS_PARAMETER_LENGTH);
      const forms = builder.words('forms');
      const ternary = state.geomorphonView === 'ternary' ? builder.words('ternary') : undefined;
      builder.graph.add(
        new GPUGeomorphons({
          id: 'geomorphons',
          width,
          height,
          elevation,
          settings: settings.view,
          searchRadius: state.geomorphonRadius,
          skipRadius: Math.min(state.geomorphonSkip, state.geomorphonRadius - 2),
          comparison: state.geomorphonComparison,
          forms,
          ...(ternary ? {ternary} : {}),
          cellSizeMode: 'web-mercator',
          rowDirection: 'south'
        })
      );
      return builder.finish({
        value: ternary ? 'ternary' : 'forms',
        format: 'uint32',
        palette: GEOMORPHON_CLASSES.map(entry => entry.color),
        write: () =>
          settings.parameters.write(
            getGPUGeomorphonsParameterValues({
              ...cell,
              flatThresholdDegrees: ctx.options.geomorphonFlatAngle,
              flatDistance: ctx.options.geomorphonFlatDistance
            })
          )
      });
    }

    if (product === 'weiss') {
      const settings = builder.settings(GPU_TERRAIN_WEISS_LANDFORMS_PARAMETER_LENGTH);
      builder.graph.add(
        new GPUTerrainWeissLandforms({
          id: 'weiss',
          width,
          height,
          elevation,
          settings: settings.view,
          smallScale: {radius: state.weissSmall},
          largeScale: {radius: Math.max(state.weissLarge, state.weissSmall + 1)},
          standardization: state.weissStandardization,
          quantum: QUANTUMS[state.quantum],
          landforms: builder.words('landforms'),
          cellSizeMode: 'web-mercator'
        })
      );
      return builder.finish({
        value: 'landforms',
        format: 'uint32',
        palette: WEISS_CLASSES.map(entry => entry.color),
        write: () =>
          settings.parameters.write(
            getGPUTerrainWeissLandformsParameterValues({
              ...cell,
              standardThreshold: ctx.options.weissThreshold,
              slopeThresholdDegrees: ctx.options.weissSlope
            })
          )
      });
    }

    // Multi-scale topographic position: tpi, dev, devmax, scale.
    const radii = SCALE_PRESETS[state.scalePreset];
    const scales = radii.map(radius => ({
      radius,
      ...(state.innerFraction > 0 && Math.floor(radius * state.innerFraction) >= 1
        ? {innerRadius: Math.floor(radius * state.innerFraction)}
        : {})
    }));
    const common = {
      id: 'topographic-position',
      width,
      height,
      elevation,
      scales,
      quantum: QUANTUMS[state.quantum]
    };
    if (product === 'tpi' || product === 'dev') {
      const planes = builder.floats('planes', radii.length * pixelCount);
      builder.graph.add(
        new GPUTerrainTopographicPosition({
          ...common,
          ...(product === 'tpi' ? {topographicPositionIndex: planes} : {deviationFromMean: planes})
        })
      );
      const select = builder.settings(4);
      const value = builder.floats('value');
      addKernelPass(builder.graph, {
        id: `${product}-select-scale`,
        invocationCount: pixelCount,
        bindings: [
          {name: 'planes', view: planes, type: 'f32', access: 'read'},
          {name: 'select', view: select.view, type: 'f32', access: 'read'},
          {name: 'value', view: value, type: 'f32', access: 'read_write'}
        ],
        declarations: `const PIXEL_COUNT: u32 = ${pixelCount}u;`,
        body: /* wgsl */ `
  let plane = u32(select[selectOffset]);
  value[valueOffset + index] = planes[planesOffset + plane * PIXEL_COUNT + index];`
      });
      return builder.finish({
        value: 'value',
        format: 'float32',
        write: () =>
          select.parameters.write(
            Float32Array.of(Math.min(ctx.options.scaleIndex, radii.length - 1), 0, 0, 0)
          )
      });
    }
    // devmax and scale
    const maximumDeviation = builder.floats('devmax');
    const maximumRadius = builder.words('devmax-radius');
    builder.graph.add(
      new GPUTerrainTopographicPosition({
        ...common,
        maximumDeviation,
        maximumDeviationRadius: maximumRadius
      })
    );
    if (product === 'devmax') {
      return builder.finish({value: 'devmax', format: 'float32', write: () => {}});
    }
    const meters = builder.floats('scale-meters');
    addKernelPass(builder.graph, {
      id: 'scale-meters',
      invocationCount: pixelCount,
      bindings: [
        {name: 'devmax', view: maximumDeviation, type: 'f32', access: 'read'},
        {name: 'devmaxRadius', view: maximumRadius, type: 'u32', access: 'read'},
        {name: 'meters', view: meters, type: 'f32', access: 'read_write'}
      ],
      declarations: `const GROUND_METERS: f32 = ${groundMeters.toFixed(5)};`,
      body: /* wgsl */ `
  let radius = devmaxRadius[devmaxRadiusOffset + index];
  var notANumber = 0x7fc00000u;
  if (radius == 0u || (bitcast<u32>(devmax[devmaxOffset + index]) & 0x7f800000u) == 0x7f800000u) {
    meters[metersOffset + index] = bitcast<f32>(notANumber);
  } else {
    meters[metersOffset + index] = f32(radius) * GROUND_METERS;
  }`
    });
    return builder.finish({value: 'scale-meters', format: 'float32', write: () => {}});
  }

  // --- Display -----------------------------------------------------------------------------------
  function getRange(stats: ValueStats | null): {low: number; high: number} | undefined {
    const state = ctx.options;
    if (
      !state.autoStretch ||
      !SYMMETRIC_PRODUCTS.has(state.product) ||
      !stats ||
      stats.kind !== 'float' ||
      stats.count === 0
    ) {
      return undefined;
    }
    const clip = state.clipPercent / 100;
    const low = stats.quantile(clip);
    const high = stats.quantile(1 - clip);
    return high > low || Math.abs(high) > 0 ? {low, high} : undefined;
  }

  function applyPaint(): void {
    const state = ctx.options;
    const range = getRange(latestStats);
    const paint = getLandformPaint(state, groundMeters, range);
    session.setPaint(paint);
    if (
      SYMMETRIC_PRODUCTS.has(state.product) &&
      paint.low !== undefined &&
      paint.high !== undefined
    ) {
      ctx.setLegendExtent('stretch', [paint.low, paint.high]);
      ctx.setReadout('stretch', `${paint.low.toPrecision(3)} to ${paint.high.toPrecision(3)}`);
    } else {
      ctx.setReadout('stretch', null);
    }
  }

  function showProduct(): void {
    const state = ctx.options;
    const key = getConfigKey(state);
    let build = session.getBuild(state.product, key);
    try {
      build ??= session.addBuild(buildProduct(state), key);
      ctx.setStatus('');
    } catch (error) {
      ctx.setStatus(`Cannot build ${state.product}: ${(error as Error).message}`);
      return;
    }
    latestStats = null;
    session.activate(build, getLandformPaint(state, groundMeters));
    applyPaint();
    ctx.setReadout('classes', null);
    ctx.requestLayers();
  }

  function describeClasses(stats: ValueStats): void {
    const state = ctx.options;
    if (stats.kind !== 'classes' || stats.total === 0) return;
    if (state.product === 'geomorphons' && state.geomorphonView === 'ternary') return;
    const names = state.product === 'weiss' ? WEISS_CLASSES : GEOMORPHON_CLASSES;
    const shares = names
      .map((entry, index) => ({
        label: entry.label.split(',')[0],
        share: stats.counts[index + 1] / stats.total
      }))
      .sort((a, b) => b.share - a.share)
      .slice(0, 4);
    ctx.setReadout(
      'classes',
      shares.map(entry => `${entry.label} ${(entry.share * 100).toFixed(1)}%`).join(' · ')
    );
    ctx.setReadout(
      'peakShare',
      `${(((stats.counts[state.product === 'weiss' ? 10 : 2] ?? 0) / stats.total) * 100).toFixed(2)}%`
    );
  }

  session.describeHover = ({value, elevation}) =>
    describeCell(ctx.options, value, elevation, groundMeters);
  session.onStats = (_id, stats) => {
    latestStats = stats;
    applyPaint();
    if (stats.kind === 'classes') describeClasses(stats);
    else {
      ctx.setReadout('median', stats.quantile(0.5).toPrecision(3));
      ctx.setReadout('extent', `${stats.min.toPrecision(3)} to ${stats.max.toPrecision(3)}`);
    }
  };

  function describeScale(): string {
    const radius = SCALE_PRESETS[ctx.options.scalePreset][ctx.options.scaleIndex];
    return `${radius} px = ${(radius * groundMeters).toFixed(0)} m radius`;
  }

  ctx.setReadout('grid', `${width} x ${height} px, ${groundMeters.toFixed(2)} m ground per pixel`);
  ctx.setReadout('scaleRadius', describeScale());
  showProduct();

  const COMPILE_OPTIONS = new Set<keyof LandformOptions>([
    'curvatureKind',
    'curvatureMethod',
    'curvatureBorder',
    'ringRadiusInner',
    'ringRadiusOuter',
    'ringSquash',
    'geomorphonView',
    'geomorphonRadius',
    'geomorphonSkip',
    'geomorphonComparison',
    'scalePreset',
    'innerFraction',
    'quantum',
    'weissSmall',
    'weissLarge',
    'weissStandardization'
  ]);
  const PARAMETER_OPTIONS = new Set<keyof LandformOptions>([
    'flatGradient',
    'ringGainInner',
    'ringGainOuter',
    'zFactor',
    'geomorphonFlatAngle',
    'geomorphonFlatDistance',
    'scaleIndex',
    'weissThreshold',
    'weissSlope'
  ]);

  return {
    getCompiledGraphs: () => session.getCompiledGraphs(),

    setOption(id) {
      const key = id as keyof LandformOptions;
      if (key === 'product') {
        showProduct();
      } else if (COMPILE_OPTIONS.has(key)) {
        if (key === 'scalePreset') ctx.setReadout('scaleRadius', describeScale());
        timers.push(setTimeout(() => !destroyed && showProduct(), REBUILD_DELAY_MILLISECONDS));
      } else if (PARAMETER_OPTIONS.has(key)) {
        session.markAllDirty();
        if (key === 'scaleIndex') ctx.setReadout('scaleRadius', describeScale());
      } else if (
        key === 'ramp' ||
        key === 'autoStretch' ||
        key === 'clipPercent' ||
        key === 'rangeScale'
      ) {
        applyPaint();
      } else {
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
            : `${total.toFixed(2)} ms, ${formatCount(pixelCount)} px (${results[0].method === 'gpu-timestamps' ? 'GPU timestamps' : 'wall clock'})`
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
      const state = ctx.options;
      return session.getLayers({
        underlay: state.underlay,
        underlayAlpha: 1,
        alpha: state.opacity
      });
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
