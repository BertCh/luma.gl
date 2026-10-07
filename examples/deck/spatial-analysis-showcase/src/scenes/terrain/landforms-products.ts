// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The GPU products of the landforms story, each built once by the scene (never in a frame) and
 * returned as a {@link LandformBuild}: a stage that encodes the contributor graph, the value
 * rasters it writes, and the paint graphs that colour them. A build is released together with its
 * graphs and buffers.
 *
 * - geomorphons: `GPUGeomorphons` at one search radius (compile time); the flatness threshold is a
 *   parameter write;
 * - curvature: `GPUTerrainCurvature`, profile and plan from one graph so switching is free;
 * - position: `GPUTerrainTopographicPosition`, eight scales from one summed-area table, and the
 *   scale a cell stands out at;
 * - Weiss: `GPUTerrainWeissLandforms`, and the agreement kernel that compares it with geomorphons.
 */

import type {Buffer} from '@luma.gl/core';
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
import type {CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {addKernelPass} from '../../engine/mode-kernels';
import type {SpatialAnalysisResources} from '../../engine/resources';
import type {TerrainSession} from './b14a-session';
import {Stage} from './b14a-session';
import {
  getCurvatureKind,
  SCALE_PRESETS,
  type CurvatureChoice,
  type LandformOptions,
  type PositionProduct
} from './landforms.style';

const QUANTUMS = {'4': 1 / 4, '64': 1 / 64, '256': 1 / 256} as const;

/** One raster a build writes, with the paint graphs that colour it (one per target buffer). */
export type ValueRef = {
  id: string;
  buffer: Buffer;
  format: 'float32' | 'uint32';
  paints: Map<string, CompiledGPUCommandGraph<void>>;
};

/** A compiled product: stage, rasters and release. */
export type LandformBuild = {
  /** Unique id of the build, also the key of its cache entry. */
  id: string;
  /** Encodes the contributor graphs (dependencies first). */
  stage: Stage;
  values: Record<string, ValueRef>;
  /** Frees the paint graphs, then the stage with its graphs and buffers. */
  release: () => void;
};

/** Counts builds so that a rebuilt product never shares raster ids with the one it replaces. */
let buildSerial = 0;

/** Wraps the buffers of a finished stage as a build. */
function createBuild(
  session: TerrainSession,
  id: string,
  stage: Stage,
  rasters: Record<string, {buffer: Buffer; format: 'float32' | 'uint32'}>
): LandformBuild {
  const values: Record<string, ValueRef> = {};
  for (const [key, raster] of Object.entries(rasters)) {
    values[key] = {id: `${id}-${key}#${buildSerial++}`, ...raster, paints: new Map()};
  }
  return {
    id,
    stage,
    values,
    release() {
      for (const value of Object.values(values)) {
        for (const graph of value.paints.values()) session.resources.release(graph);
        value.paints.clear();
      }
      session.unregisterStage(stage);
      stage.release();
    }
  };
}

/** Frees the paint graph of a value for one target, if it has one. */
export function releasePaint(
  resources: SpatialAnalysisResources,
  value: ValueRef,
  target: string
): void {
  const graph = value.paints.get(target);
  if (graph) resources.release(graph);
  value.paints.delete(target);
}

// ---------------------------------------------------------------------------------------------
// Geomorphons
// ---------------------------------------------------------------------------------------------

/** The compile-time key of a geomorphon build. */
export function getGeomorphonKey(
  radius: number,
  options: Pick<LandformOptions, 'geomorphonSkip' | 'geomorphonComparison'>
): string {
  return `${radius}|${options.geomorphonSkip}|${options.geomorphonComparison}`;
}

/**
 * Geomorphons at one search radius. `radius`, the skip radius and the comparison are compile-time;
 * the flatness threshold and flat distance are read from `getOptions` on every encode.
 */
export function buildGeomorphons(
  session: TerrainSession,
  radius: number,
  getOptions: () => LandformOptions
): LandformBuild {
  const {grid} = session;
  const options = getOptions();
  const id = `geo-${getGeomorphonKey(radius, options).replace(/\|/g, '-')}`;
  const builder = session.builder(id);
  const elevation = builder.elevation();
  const settings = builder.settings(GPU_GEOMORPHONS_PARAMETER_LENGTH);
  const forms = builder.words('forms');
  builder.graph.add(
    new GPUGeomorphons({
      id: 'geomorphons',
      width: grid.width,
      height: grid.height,
      elevation,
      settings: settings.view,
      searchRadius: radius,
      skipRadius: Math.min(options.geomorphonSkip, radius - 2),
      comparison: options.geomorphonComparison,
      forms,
      cellSizeMode: 'web-mercator',
      rowDirection: 'south'
    })
  );
  const stage = builder.finishStage({
    minIntervalMs: 40,
    write: () => {
      const state = getOptions();
      settings.parameters.write(
        getGPUGeomorphonsParameterValues({
          ...grid.cellSettings,
          flatThresholdDegrees: state.geomorphonFlatAngle,
          flatDistance: state.geomorphonFlatDistance
        })
      );
    }
  });
  return createBuild(session, id, stage, {
    forms: {buffer: builder.getBuffer('forms'), format: 'uint32'}
  });
}

// ---------------------------------------------------------------------------------------------
// Curvature
// ---------------------------------------------------------------------------------------------

/** The compile-time key of the curvature build. */
export function getCurvatureConfigKey(state: LandformOptions): string {
  const kind = getCurvatureKind(state);
  const family =
    kind === 'profile' || kind === 'plan'
      ? 'standard'
      : kind === 'ring-multi-radius'
        ? 'ring'
        : kind;
  return `${family}|${state.curvatureMethod}|${state.curvatureBorder}|${state.ringRadiusInner}|${state.ringRadiusOuter}|${state.ringSquash}`;
}

/**
 * Curvature. The standard build writes profile and plan from one graph (switching between them
 * recompiles nothing); any other kind, and the ring curvature, is a build of its own. The
 * displayed raster is `values.curvature` for those and `values.profile` / `values.plan`.
 */
export function buildCurvature(
  session: TerrainSession,
  getOptions: () => LandformOptions
): LandformBuild {
  const {grid} = session;
  const state = getOptions();
  const kind = getCurvatureKind(state);
  const standard = kind === 'profile' || kind === 'plan';
  const id = `curvature-${standard ? 'standard' : kind}`;
  const builder = session.builder(id);
  const elevation = builder.elevation();
  const settings = builder.settings(GPU_TERRAIN_CURVATURE_PARAMETER_LENGTH);
  const outputs: Record<string, {buffer: Buffer; format: 'float32'}> = {};
  const common = {
    id: 'curvature',
    width: grid.width,
    height: grid.height,
    elevation,
    settings: settings.view,
    method: state.curvatureMethod,
    cellSizeMode: 'web-mercator' as const,
    rowDirection: 'south' as const,
    borderMode: state.curvatureBorder
  };
  if (standard) {
    const profile = builder.floats('profile');
    const plan = builder.floats('plan');
    builder.graph.add(new GPUTerrainCurvature({...common, curvatures: {profile, plan}}));
    outputs.profile = {buffer: builder.getBuffer('profile'), format: 'float32'};
    outputs.plan = {buffer: builder.getBuffer('plan'), format: 'float32'};
  } else if (kind === 'ring-multi-radius') {
    const output = builder.floats('curvature');
    const outerRadius = Math.max(state.ringRadiusOuter, state.ringRadiusInner + 1);
    builder.graph.add(
      new GPUTerrainCurvature({
        ...common,
        ringCurvature: output,
        ringRadii: [state.ringRadiusInner, outerRadius],
        ringSquash: state.ringSquash ? ('pade-tanh' as const) : ('none' as const)
      })
    );
    outputs.curvature = {buffer: builder.getBuffer('curvature'), format: 'float32'};
  } else {
    const output = builder.floats('curvature');
    builder.graph.add(
      new GPUTerrainCurvature({
        ...common,
        curvatures: {[kind as GPUTerrainCurvatureKind]: output}
      })
    );
    outputs.curvature = {buffer: builder.getBuffer('curvature'), format: 'float32'};
  }
  const stage = builder.finishStage({
    write: () => {
      const current = getOptions();
      settings.parameters.write(
        getGPUTerrainCurvatureParameterValues({
          ...grid.cellSettings,
          zFactor: current.zFactor,
          flatGradient: Math.max(current.flatGradient, 1e-6),
          ringGains: [current.ringGainInner, current.ringGainOuter]
        })
      );
    }
  });
  return createBuild(session, id, stage, outputs);
}

/** The id of the value a curvature kind displays in a build made by {@link buildCurvature}. */
export function getCurvatureValueId(kind: CurvatureChoice): 'profile' | 'plan' | 'curvature' {
  return kind === 'profile' ? 'profile' : kind === 'plan' ? 'plan' : 'curvature';
}

// ---------------------------------------------------------------------------------------------
// Topographic position
// ---------------------------------------------------------------------------------------------

/** The compile-time key of a position build. */
export function getPositionConfigKey(state: LandformOptions): string {
  return `${state.positionProduct}|${state.scalePreset}|${state.innerFraction}|${state.quantum}`;
}

/**
 * The topographic position of one product: `tpi` and `dev` write the eight scales into one
 * float plane stack and a small kernel picks the scale shown (a parameter write, so the stepper
 * recompiles nothing); `scale` keeps DEVmax and the radius it occurs at and writes the scale
 * class: code 1-8 for the eight windows, 0 for none.
 */
export function buildPosition(
  session: TerrainSession,
  product: PositionProduct,
  getOptions: () => LandformOptions
): LandformBuild {
  const {grid} = session;
  const {pixelCount} = grid;
  const state = getOptions();
  const radii = SCALE_PRESETS[state.scalePreset];
  const id = `position-${product}`;
  const builder = session.builder(id);
  const elevation = builder.elevation();
  const scales = radii.map(radius => ({
    radius,
    ...(state.innerFraction > 0 && Math.floor(radius * state.innerFraction) >= 1
      ? {innerRadius: Math.floor(radius * state.innerFraction)}
      : {})
  }));
  const common = {
    id: 'topographic-position',
    width: grid.width,
    height: grid.height,
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
    const stage = builder.finishStage({
      write: () =>
        select.parameters.write(
          Float32Array.of(Math.min(getOptions().scaleIndex, radii.length - 1), 0, 0, 0)
        )
    });
    return createBuild(session, id, stage, {
      value: {buffer: builder.getBuffer('value'), format: 'float32'}
    });
  }
  const maximumDeviation = builder.floats('devmax');
  const maximumRadius = builder.words('devmax-radius');
  builder.graph.add(
    new GPUTerrainTopographicPosition({
      ...common,
      maximumDeviation,
      maximumDeviationRadius: maximumRadius
    })
  );
  const codes = builder.words('codes');
  const radiusList = radii.map(radius => `${radius}u`).join(', ');
  addKernelPass(builder.graph, {
    id: 'scale-class',
    invocationCount: pixelCount,
    bindings: [
      {name: 'devmax', view: maximumDeviation, type: 'f32', access: 'read'},
      {name: 'devmaxRadius', view: maximumRadius, type: 'u32', access: 'read'},
      {name: 'codes', view: codes, type: 'u32', access: 'read_write'}
    ],
    declarations: /* wgsl */ `
var<private> SCALE_RADII: array<u32, ${radii.length}> = array<u32, ${radii.length}>(${radiusList});`,
    body: /* wgsl */ `
  let radius = devmaxRadius[devmaxRadiusOffset + index];
  var code = 0u;
  let notFinite = (bitcast<u32>(devmax[devmaxOffset + index]) & 0x7f800000u) == 0x7f800000u;
  if (radius != 0u && !notFinite) {
    for (var scaleIndex = 0u; scaleIndex < ${radii.length}u; scaleIndex = scaleIndex + 1u) {
      if (SCALE_RADII[scaleIndex] == radius) {
        code = scaleIndex + 1u;
      }
    }
  }
  codes[codesOffset + index] = code;`
  });
  const stage = builder.finishStage({write: () => {}});
  return createBuild(session, id, stage, {
    codes: {buffer: builder.getBuffer('codes'), format: 'uint32'}
  });
}

// ---------------------------------------------------------------------------------------------
// Weiss landforms and agreement
// ---------------------------------------------------------------------------------------------

/** The compile-time key of the Weiss build. */
export function getWeissConfigKey(state: LandformOptions): string {
  return `${state.weissSmall}|${state.weissLarge}|${state.weissStandardization}|${state.quantum}`;
}

/** Weiss (2001) landforms from a small and a large TPI scale and the Horn slope. */
export function buildWeiss(
  session: TerrainSession,
  getOptions: () => LandformOptions
): LandformBuild {
  const {grid} = session;
  const state = getOptions();
  const id = 'weiss';
  const builder = session.builder(id);
  const elevation = builder.elevation();
  const settings = builder.settings(GPU_TERRAIN_WEISS_LANDFORMS_PARAMETER_LENGTH);
  const codes = builder.words('codes');
  builder.graph.add(
    new GPUTerrainWeissLandforms({
      id: 'weiss',
      width: grid.width,
      height: grid.height,
      elevation,
      settings: settings.view,
      smallScale: {radius: state.weissSmall},
      largeScale: {radius: Math.max(state.weissLarge, state.weissSmall + 1)},
      standardization: state.weissStandardization,
      quantum: QUANTUMS[state.quantum],
      landforms: codes,
      cellSizeMode: 'web-mercator'
    })
  );
  const stage = builder.finishStage({
    write: () => {
      const current = getOptions();
      settings.parameters.write(
        getGPUTerrainWeissLandformsParameterValues({
          ...grid.cellSettings,
          standardThreshold: current.weissThreshold,
          slopeThresholdDegrees: current.weissSlope
        })
      );
    }
  });
  return createBuild(session, id, stage, {
    codes: {buffer: builder.getBuffer('codes'), format: 'uint32'}
  });
}

/** Class codes of the agreement raster: classes of the agreement table, then "not compared". */
export const AGREEMENT_CODES = {agree: 0, disagree: 1, opposite: 2, none: 3} as const;

/**
 * Collapses both classifiers to convex, neutral and concave and compares them cell by cell.
 * Convex: Weiss upper slope, local ridge, midslope ridge, mountain top; geomorphon peak, ridge,
 * shoulder, spur. Neutral: plain, open slope; flat, slope. Concave: the drainage and valley classes;
 * hollow, footslope, valley, pit. Output: 0 agree, 1 disagree, 2 opposite (convex against
 * concave), 3 where either has no class. Depends on both stages, so it re-runs when either does.
 */
export function buildAgreement(
  session: TerrainSession,
  weiss: LandformBuild,
  geomorphons: LandformBuild
): LandformBuild {
  const {pixelCount} = session.grid;
  const id = 'agreement';
  const builder = session.builder(id);
  const weissView = importGraphBuffer(
    builder.graph,
    'weiss-codes',
    weiss.values.codes.buffer,
    'uint32',
    pixelCount
  );
  const formsView = importGraphBuffer(
    builder.graph,
    'forms-codes',
    geomorphons.values.forms.buffer,
    'uint32',
    pixelCount
  );
  const codes = builder.words('codes');
  addKernelPass(builder.graph, {
    id: 'agreement-kernel',
    invocationCount: pixelCount,
    bindings: [
      {name: 'weissCodes', view: weissView, type: 'u32', access: 'read'},
      {name: 'formCodes', view: formsView, type: 'u32', access: 'read'},
      {name: 'agreement', view: codes, type: 'u32', access: 'read_write'}
    ],
    // Groups: 0 none, 1 concave, 2 neutral, 3 convex.
    declarations: /* wgsl */ `
fn weissGroup(code: u32) -> u32 {
  if (code >= 1u && code <= 4u) { return 1u; }
  if (code == 5u || code == 6u) { return 2u; }
  if (code >= 7u && code <= 10u) { return 3u; }
  return 0u;
}
fn formGroup(code: u32) -> u32 {
  if (code == 1u || code == 6u) { return 2u; }
  if (code >= 2u && code <= 5u) { return 3u; }
  if (code >= 7u && code <= 10u) { return 1u; }
  return 0u;
}`,
    body: /* wgsl */ `
  let weissKind = weissGroup(weissCodes[weissCodesOffset + index]);
  let formKind = formGroup(formCodes[formCodesOffset + index]);
  var result = 3u;
  if (weissKind != 0u && formKind != 0u) {
    if (weissKind == formKind) {
      result = 0u;
    } else if (weissKind + formKind == 4u && weissKind != 2u) {
      result = 2u;
    } else {
      result = 1u;
    }
  }
  agreement[agreementOffset + index] = result;`
  });
  const stage = builder.finishStage({
    write: () => {},
    dependencies: [weiss.stage, geomorphons.stage]
  });
  return createBuild(session, id, stage, {
    codes: {buffer: builder.getBuffer('codes'), format: 'uint32'}
  });
}
