// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPURasterBand} from '../../gpu-raster/index';
import {
  createMapGraphKernelNode,
  getWGSLFloatLiteral,
  type MapGraphKernelBinding
} from '../map-graph-kernels';
import type {GPUMapGraphRecipe} from '../map-graph-types';
import {validateGraphViewsBelongToGraph} from '../map-graph-utils';
import type {GPUTerrainCellSizeMode} from '../terrain-analysis/gpu-terrain-derivatives';
import {
  getTerrainBandViews,
  getTerrainElevationNodes,
  TERRAIN_WGSL_HELPERS,
  validateTerrainBandBelongsToGraph,
  validateTerrainBuffersDistinct,
  validateTerrainGrid,
  validateTerrainSettings,
  type TerrainElevationSource
} from '../terrain-analysis/terrain-analysis-utils';
import {
  getTerrainGroundCellSizeWGSL,
  validateTerrainCellSizeMode,
  validateTerrainRowDirection,
  writeTerrainGeomorphometryCellSettings,
  TERRAIN_GEOMORPHOMETRY_SHARED_PARAMETER_LENGTH,
  type TerrainGeomorphometryCellSettings
} from './terrain-geomorphometry-utils';

/**
 * Estimator of the local partial derivatives `p, q, r, s, t`.
 *
 * - `'evans-young'`: 3x3 quadratic least squares (Evans 1979, Young 1978).
 * - `'zevenbergen-thorne'`: 3x3 partial quartic (Zevenbergen and Thorne 1987).
 * - `'florinsky'`: 5x5 quadratic least squares (Florinsky 2016).
 */
export type GPUTerrainCurvatureMethod = 'evans-young' | 'zevenbergen-thorne' | 'florinsky';

/**
 * One curvature measure. With `p = dz/dx` (east), `q = dz/dy` (north), `r = d2z/dx2`,
 * `s = d2z/dxdy`, `t = d2z/dy2`, `G = p^2 + q^2`, `W = 1 + G` (Florinsky 2016, WhiteboxTools sign
 * convention: a convex hill top is positive, a concave valley floor is negative):
 *
 * - `'tangential'`: `-(q^2 r - 2pqs + p^2 t) / (G sqrt(W))`
 * - `'plan'`: `-(q^2 r - 2pqs + p^2 t) / G^1.5`
 * - `'profile'`: `-(p^2 r + 2pqs + q^2 t) / (G W^1.5)`
 * - `'mean'`: `-((1+q^2) r - 2pqs + (1+p^2) t) / (2 W^1.5)`
 * - `'gaussian'`: `(rt - s^2) / W^2`
 * - `'unsphericity'`: `sqrt(max(H^2 - K, 0))` for mean `H` and Gaussian `K`
 * - `'minimal'`, `'maximal'`: `H -/+ unsphericity`
 * - `'difference'`: `(profile - tangential) / 2`
 * - `'horizontal-excess'`: `tangential - minimal`; `'vertical-excess'`: `profile - minimal`
 * - `'accumulation'`: `tangential * profile`
 * - `'ring'`: `horizontal-excess * vertical-excess`
 * - `'rotor'`: `((p^2 - q^2) s - pq (r - t)) / G^1.5`
 * - `'laplacian'`: `r + t`
 */
export type GPUTerrainCurvatureKind =
  | 'profile'
  | 'plan'
  | 'tangential'
  | 'mean'
  | 'gaussian'
  | 'minimal'
  | 'maximal'
  | 'unsphericity'
  | 'difference'
  | 'horizontal-excess'
  | 'vertical-excess'
  | 'accumulation'
  | 'ring'
  | 'rotor'
  | 'laplacian';

/** Number of float32 values read from `GPUTerrainCurvatureProps.settings`. */
export const GPU_TERRAIN_CURVATURE_PARAMETER_LENGTH = 12;

/** Maximum number of rings of the multi-radius ring curvature. */
const MAXIMUM_RING_COUNT = 4;
const DEFAULT_FLAT_GRADIENT = 1e-6;
const DEFAULT_RING_GAINS = [0.55, 0.45, 0, 0] as const;
const OUTPUTS_PER_KERNEL = 5;

/** Every curvature kind in the fixed output order used to group kernels. */
const CURVATURE_KINDS: readonly GPUTerrainCurvatureKind[] = [
  'profile',
  'plan',
  'tangential',
  'mean',
  'gaussian',
  'minimal',
  'maximal',
  'unsphericity',
  'difference',
  'horizontal-excess',
  'vertical-excess',
  'accumulation',
  'ring',
  'rotor',
  'laplacian'
];

/**
 * CPU-side description packed by {@link getGPUTerrainCurvatureParameterValues}.
 *
 * Cell-size model: `cellSize` is in projected metres for `cellSizeMode: 'uniform'`, in equatorial
 * Web Mercator metres for `'web-mercator'` (ground size divides by `cosh(PI (1 - 2y))` of the row
 * centre, `northEdge`/`southEdge` are normalized Mercator y), and in degrees for `'geographic'`
 * (x size multiplies by cos(latitude) of the row centre, `northEdge`/`southEdge` are latitude
 * degrees, one degree is 111319.49 m). Curvatures are in 1/ground-unit (Gaussian in 1/unit^2).
 */
export type GPUTerrainCurvatureSettings = TerrainGeomorphometryCellSettings & {
  /**
   * Gradients with `p^2 + q^2 <= flatGradient^2` are flat: the direction-dependent curvatures
   * (profile, plan, tangential, difference, excesses, accumulation, ring, rotor) are 0 there.
   * Defaults to 1e-6.
   */
  flatGradient?: number;
  /**
   * Up to four per-ring gains of the multi-radius ring curvature, one per entry of `ringRadii`.
   * Defaults to `[0.55, 0.45, 0, 0]`.
   *
   * These are mt-image's pre-division weights. mt-image additionally divides ring 0 by 0.35 and
   * ring 1 by 0.3; reproduce it with `[0.55 / 0.35, 0.45 / 0.3]`.
   */
  ringGains?: readonly number[];
};

/**
 * Packs settings into the 12-float layout read by {@link GPUTerrainCurvature}:
 * `[cellSizeX, cellSizeY, zFactor, northEdge, southEdge, flatGradient, gain0, gain1, gain2, gain3, 0, 0]`.
 *
 * Cell-size model: see {@link GPUTerrainCurvatureSettings}.
 */
export function getGPUTerrainCurvatureParameterValues(
  settings: GPUTerrainCurvatureSettings,
  target: Float32Array = new Float32Array(GPU_TERRAIN_CURVATURE_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_TERRAIN_CURVATURE_PARAMETER_LENGTH) {
    throw new Error('Terrain curvature settings target must hold 12 values');
  }
  const gains = settings.ringGains ?? DEFAULT_RING_GAINS;
  if (gains.length > MAXIMUM_RING_COUNT) {
    throw new Error('Terrain curvature ringGains must hold at most 4 values');
  }
  target.fill(0, 0, GPU_TERRAIN_CURVATURE_PARAMETER_LENGTH);
  writeTerrainGeomorphometryCellSettings(settings, target);
  const shared = TERRAIN_GEOMORPHOMETRY_SHARED_PARAMETER_LENGTH;
  target[shared] = settings.flatGradient ?? DEFAULT_FLAT_GRADIENT;
  for (const [index, gain] of gains.entries()) {
    target[shared + 1 + index] = gain;
  }
  return target;
}

/**
 * Properties for {@link GPUTerrainCurvature}.
 *
 * Cell-size model: `settings.cellSize` and `cellSizeMode` follow
 * {@link GPUTerrainCurvatureSettings} (projected metres `'uniform'`, `'web-mercator'`, or
 * `'geographic'` degrees with latitude-dependent row spacing).
 *
 * Topology: grid size, elevation format, `method`, which outputs exist, `ringRadii`,
 * `ringSquash`, `cellSizeMode`, `rowDirection`, `borderMode`. Per-frame: `settings`, elevation.
 */
export type GPUTerrainCurvatureProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-curvature'`. */
  id?: string;
  /** Grid width in pixels. */
  width: number;
  /** Grid height in pixels. */
  height: number;
  /** Elevation band, buffer or texture; scale, offset, nodata and validity are honored. */
  elevation: GPURasterBand;
  /** Per-frame settings with at least 12 float32 values, see {@link getGPUTerrainCurvatureParameterValues}. */
  settings: GraphDataView<'float32'>;
  /** Partial derivative estimator. Defaults to `'evans-young'`. */
  method?: GPUTerrainCurvatureMethod;
  /**
   * Optional float32 outputs, one value per pixel, NaN where invalid. Sign convention: a convex
   * hill gives positive profile, plan, tangential and mean curvature (see
   * {@link GPUTerrainCurvatureKind}).
   */
  curvatures?: Partial<Record<GPUTerrainCurvatureKind, GraphDataView<'float32'>>>;
  /**
   * Optional multi-radius ring curvature (mt-image): per ring of radius `r` cells, the sum of
   * the centre-minus-sample differences of 8 samples (axis samples at `r`, diagonal samples at
   * `max(1, round(r / sqrt(2)))`) divided by `8 r sqrt(cellX cellY)`, weighted by `ringGains` and
   * multiplied by `zFactor`. Positive on convex ground. Any sample outside the raster or invalid
   * makes the value invalid regardless of `borderMode`.
   */
  ringCurvature?: GraphDataView<'float32'>;
  /** One to four strictly ascending integer ring radii in cells. Defaults to `[2, 8]`. */
  ringRadii?: readonly number[];
  /** `'pade-tanh'` clamps to [-3, 3] and applies `x (27 + x^2) / (27 + 9 x^2)`. Defaults to `'none'`. */
  ringSquash?: 'none' | 'pade-tanh';
  /** Optional per-pixel 1 where every requested output is valid, else 0. */
  validity?: GraphDataView<'uint32'>;
  /** Cell size interpretation. Defaults to `'uniform'`. */
  cellSizeMode?: GPUTerrainCellSizeMode;
  /** Direction in which the row index increases. Defaults to `'south'` (north-up rasters). */
  rowDirection?: 'south' | 'north';
  /** `'clamp'` repeats edge samples; `'nodata'` invalidates windows leaving the raster. Defaults to `'clamp'`. */
  borderMode?: 'clamp' | 'nodata';
};

type OutputSlot =
  | {kind: 'curvature'; curvature: GPUTerrainCurvatureKind; view: GraphDataView<'float32'>}
  | {kind: 'ring'; view: GraphDataView<'float32'>}
  | {kind: 'validity'; view: GraphDataView<'uint32'>};

/**
 * Computes second-order terrain curvatures (profile, plan, tangential, mean, Gaussian, principal,
 * excess, and more) and the multi-radius ring curvature from an elevation raster.
 *
 * Cell-size model: projected metres (`'uniform'`), Web Mercator metres (`'web-mercator'`), or
 * geographic degrees with latitude-dependent row spacing (`'geographic'`), per row, with a
 * spherical row-constant approximation like `GPUTerrainDerivatives`.
 *
 * Window samples are differenced against the centre elevation before any sum, so large base
 * heights do not cost f32 precision. Requested outputs are split into kernels of at most five
 * outputs, each recomputing the partials. Invalid or non-finite window samples, and (with
 * `borderMode: 'nodata'`) windows leaving the raster, give NaN and validity 0. Satisfies the
 * `GPURasterHaloStage` contract through `requiredHalo`.
 */
export class GPUTerrainCurvature implements GPUMapGraphRecipe {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'terrain-curvature';
  /** Validated properties. */
  readonly props: GPUTerrainCurvatureProps;
  /** Receptive field in pixels: 1 (3x3), 2 (florinsky), or the largest ring radius if larger. */
  readonly requiredHalo: number;

  constructor(props: GPUTerrainCurvatureProps) {
    this.id = props.id ?? this.recipe;
    this.props = props;
    const {id} = this;
    const pixelCount = validateTerrainGrid(id, props.width, props.height);
    const method = props.method ?? 'evans-young';
    if (!['evans-young', 'zevenbergen-thorne', 'florinsky'].includes(method)) {
      throw new Error(`${id} method must be evans-young, zevenbergen-thorne, or florinsky`);
    }
    const kinds = Object.keys(props.curvatures ?? {});
    for (const kind of kinds) {
      if (!CURVATURE_KINDS.includes(kind as GPUTerrainCurvatureKind)) {
        throw new Error(`${id} unknown curvature kind ${kind}`);
      }
    }
    const curvatureViews = kinds
      .map(kind => props.curvatures?.[kind as GPUTerrainCurvatureKind])
      .filter(view => view !== undefined);
    if (curvatureViews.length === 0 && !props.ringCurvature && !props.validity) {
      throw new Error(`${id} requires at least one output`);
    }
    for (const [name, view] of [
      ...kinds.map(kind => [kind, props.curvatures?.[kind as GPUTerrainCurvatureKind]] as const),
      ['ringCurvature', props.ringCurvature] as const
    ]) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
        if (view.length !== pixelCount) {
          throw new Error(`${id} ${name} must contain one value per pixel`);
        }
      }
    }
    if (props.validity) {
      validatePackedUint32View(props.validity, `${id} validity`);
      if (props.validity.length !== pixelCount) {
        throw new Error(`${id} validity must contain one value per pixel`);
      }
    }
    validateTerrainSettings(id, props.settings, GPU_TERRAIN_CURVATURE_PARAMETER_LENGTH);
    validateTerrainCellSizeMode(id, props.cellSizeMode);
    validateTerrainRowDirection(id, props.rowDirection);
    if (!['clamp', 'nodata'].includes(props.borderMode ?? 'clamp')) {
      throw new Error(`${id} borderMode must be clamp or nodata`);
    }
    if (!['none', 'pade-tanh'].includes(props.ringSquash ?? 'none')) {
      throw new Error(`${id} ringSquash must be none or pade-tanh`);
    }
    const radii = props.ringRadii ?? [2, 8];
    if (
      radii.length < 1 ||
      radii.length > MAXIMUM_RING_COUNT ||
      radii.some(
        (radius, index) =>
          !Number.isInteger(radius) || radius < 1 || (index > 0 && radius <= radii[index - 1])
      )
    ) {
      throw new Error(`${id} ringRadii must be 1 to 4 strictly ascending integers of at least 1`);
    }
    validateTerrainBuffersDistinct(
      id,
      [...curvatureViews, props.ringCurvature, props.validity],
      [...getTerrainBandViews(props.elevation), props.settings]
    );
    const methodHalo = method === 'florinsky' ? 2 : 1;
    this.requiredHalo = props.ringCurvature
      ? Math.max(methodHalo, radii[radii.length - 1])
      : methodHalo;
  }

  /** Returns the elevation canonicalization nodes and one kernel per group of five outputs. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateTerrainBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, [
      props.settings,
      ...CURVATURE_KINDS.map(kind => props.curvatures?.[kind]),
      props.ringCurvature,
      props.validity
    ]);
    const source = getTerrainElevationNodes(
      graph,
      id,
      props.elevation,
      props.width,
      props.height,
      true
    );
    const slots: OutputSlot[] = [];
    for (const curvature of CURVATURE_KINDS) {
      const view = props.curvatures?.[curvature];
      if (view) slots.push({kind: 'curvature', curvature, view});
    }
    if (props.ringCurvature) slots.push({kind: 'ring', view: props.ringCurvature});
    if (props.validity) slots.push({kind: 'validity', view: props.validity});
    const hasWindowOutput = slots.some(slot => slot.kind === 'curvature');
    const hasRingOutput = Boolean(props.ringCurvature);

    const nodes: GPUCommandNode<Parameters>[] = [...source.nodes];
    for (let start = 0, group = 0; start < slots.length; start += OUTPUTS_PER_KERNEL, group++) {
      nodes.push(
        getCurvatureNode(graph, {
          id: `${id}-curvature-${group}`,
          props,
          source,
          slots: slots.slice(start, start + OUTPUTS_PER_KERNEL),
          hasWindowOutput,
          hasRingOutput
        })
      );
    }
    return nodes;
  }
}

/** Declares `name` as the WGSL float expression of one curvature kind. */
const CURVATURE_EXPRESSIONS: Record<GPUTerrainCurvatureKind, string> = {
  profile: 'profile',
  plan: 'plan',
  tangential: 'tangential',
  mean: 'meanCurvature',
  gaussian: 'gaussian',
  minimal: 'minimal',
  maximal: 'maximal',
  unsphericity: 'unsphericity',
  difference: 'differenceCurvature',
  'horizontal-excess': 'horizontalExcess',
  'vertical-excess': 'verticalExcess',
  accumulation: 'accumulation',
  ring: 'ringProduct',
  rotor: 'rotor',
  laplacian: 'laplacian'
};

/** Returns WGSL `let z.. = readWindowDelta(...)` lines and the partial derivative expressions. */
function getWindowSource(
  method: GPUTerrainCurvatureMethod,
  northSign: number
): {samples: string; partials: string} {
  const lines: string[] = [];
  /** `northOffset` > 0 is toward geographic north. */
  const read = (name: string, dx: number, northOffset: number) => {
    lines.push(
      `let ${name} = readWindowDelta(column, row, ${dx}, ${northOffset * northSign}, centre);`
    );
  };
  if (method === 'florinsky') {
    for (let n = 0; n < 25; n++) {
      read(`z${n}`, (n % 5) - 2, 2 - Math.floor(n / 5));
    }
    return {
      samples: lines.join('\n  '),
      partials: `let r = zFactor * (2.0 * (z0 + z4 + z5 + z9 + z10 + z14 + z15 + z19 + z20 + z24) - 2.0 * (z2 + z7 + z12 + z17 + z22) - z1 - z3 - z6 - z8 - z11 - z13 - z16 - z18 - z21 - z23) / (35.0 * wx * wx);
  let t = zFactor * (2.0 * (z0 + z1 + z2 + z3 + z4 + z20 + z21 + z22 + z23 + z24) - 2.0 * (z10 + z11 + z12 + z13 + z14) - z5 - z6 - z7 - z8 - z9 - z15 - z16 - z17 - z18 - z19) / (35.0 * wy * wy);
  let s = zFactor * (z8 + z16 - z6 - z18 + 4.0 * (z4 + z20 - z0 - z24) + 2.0 * (z3 + z9 + z15 + z21 - z1 - z5 - z19 - z23)) / (100.0 * wx * wy);
  let p = zFactor * (44.0 * (z3 + z23 - z1 - z21) + 31.0 * (z0 + z20 - z4 - z24 + 2.0 * (z8 + z18 - z6 - z16)) + 17.0 * (z14 - z10 + 4.0 * (z13 - z11)) + 5.0 * (z9 + z19 - z5 - z15)) / (420.0 * wx);
  let q = zFactor * (44.0 * (z5 + z9 - z15 - z19) + 31.0 * (z20 + z24 - z0 - z4 + 2.0 * (z6 + z8 - z16 - z18)) + 17.0 * (z2 - z22 + 4.0 * (z7 - z17)) + 5.0 * (z1 + z3 - z21 - z23)) / (420.0 * wy);`
    };
  }
  // 3x3 labels (north-up): z1 NW, z2 N, z3 NE, z4 W, z5 centre (0), z6 E, z7 SW, z8 S, z9 SE.
  const labels: [number, number, number][] = [
    [1, -1, 1],
    [2, 0, 1],
    [3, 1, 1],
    [4, -1, 0],
    [6, 1, 0],
    [7, -1, -1],
    [8, 0, -1],
    [9, 1, -1]
  ];
  for (const [label, dx, northOffset] of labels) {
    read(`z${label}`, dx, northOffset);
  }
  const cross = 'zFactor * (z3 + z7 - z1 - z9) / (4.0 * wx * wy)';
  if (method === 'zevenbergen-thorne') {
    return {
      samples: lines.join('\n  '),
      partials: `let p = zFactor * (z6 - z4) / (2.0 * wx);
  let q = zFactor * (z2 - z8) / (2.0 * wy);
  let r = zFactor * (z4 + z6) / (wx * wx);
  let t = zFactor * (z2 + z8) / (wy * wy);
  let s = ${cross};`
    };
  }
  return {
    samples: lines.join('\n  '),
    partials: `let p = zFactor * (z3 + z6 + z9 - z1 - z4 - z7) / (6.0 * wx);
  let q = zFactor * (z1 + z2 + z3 - z7 - z8 - z9) / (6.0 * wy);
  let r = zFactor * (z1 + z3 + z4 + z6 + z7 + z9 - 2.0 * (z2 + z8)) / (3.0 * wx * wx);
  let t = zFactor * (z1 + z2 + z3 + z7 + z8 + z9 - 2.0 * (z4 + z6)) / (3.0 * wy * wy);
  let s = ${cross};`
  };
}

/** Returns WGSL computing `ringValue` (and `ringValid`) from the multi-radius ring samples. */
function getRingSource(radii: readonly number[], squash: 'none' | 'pade-tanh'): string {
  const terms: string[] = [];
  const lines: string[] = [];
  for (const [index, radius] of radii.entries()) {
    const diagonal = Math.max(1, Math.round(radius / Math.SQRT2));
    const offsets: [number, number][] = [
      [radius, 0],
      [-radius, 0],
      [0, radius],
      [0, -radius],
      [diagonal, diagonal],
      [diagonal, -diagonal],
      [-diagonal, diagonal],
      [-diagonal, -diagonal]
    ];
    const sum = offsets
      .map(([dx, dy]) => `readRingDelta(column, row, ${dx}, ${dy}, centre)`)
      .join(' + ');
    lines.push(`let ringSum${index} = ${sum};`);
    // Centre minus sample is the negated sum of the differences.
    terms.push(
      `settings[settingsOffset + ${TERRAIN_GEOMORPHOMETRY_SHARED_PARAMETER_LENGTH + 1 + index}u] * (-ringSum${index}) / (${getWGSLFloatLiteral(8 * radius)} * cellMetres)`
    );
  }
  return `${lines.join('\n  ')}
  var ringValue = zFactor * (${terms.join(' + ')});
  ${
    squash === 'pade-tanh'
      ? `let squashed = clamp(ringValue, -3.0, 3.0);
  ringValue = squashed * (27.0 + squashed * squashed) / (27.0 + 9.0 * squashed * squashed);`
      : ''
  }`;
}

/** Builds one kernel writing up to five outputs. */
function getCurvatureNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  options: {
    id: string;
    props: GPUTerrainCurvatureProps;
    source: TerrainElevationSource<Parameters>;
    slots: readonly OutputSlot[];
    hasWindowOutput: boolean;
    hasRingOutput: boolean;
  }
): GPUCommandNode<Parameters> {
  const {props, source, slots} = options;
  const method = props.method ?? 'evans-young';
  const northSign = (props.rowDirection ?? 'south') === 'south' ? -1 : 1;
  const clamp = (props.borderMode ?? 'clamp') === 'clamp';
  const needsWindow =
    slots.some(slot => slot.kind === 'curvature') ||
    (options.hasWindowOutput && slots.some(slot => slot.kind === 'validity'));
  const needsRing =
    slots.some(slot => slot.kind === 'ring') ||
    (options.hasRingOutput && slots.some(slot => slot.kind === 'validity'));
  const sourceBand = source.band;
  const bindings: MapGraphKernelBinding[] = [
    {name: 'elevationValues', view: sourceBand.storage.values, type: 'f32', access: 'read'},
    {
      name: 'elevationValidity',
      view: sourceBand.validity as GraphDataView<'uint32'>,
      type: 'u32',
      access: 'read'
    },
    {name: 'settings', view: props.settings, type: 'f32', access: 'read'}
  ];
  for (const [index, slot] of slots.entries()) {
    bindings.push({
      name: `output${index}`,
      view: slot.view,
      type: slot.kind === 'validity' ? 'u32' : 'f32',
      access: 'read_write'
    });
  }
  const window = getWindowSource(method, northSign);
  const radii = props.ringRadii ?? [2, 8];
  const writes = slots
    .map((slot, index) => {
      const name = `output${index}`;
      if (slot.kind === 'validity') {
        const conditions = [
          options.hasWindowOutput ? 'windowValid' : 'true',
          options.hasRingOutput ? 'ringValid' : 'true'
        ];
        return `${name}[${name}Offset + index] = select(0u, 1u, centreValid && ${conditions.join(' && ')});`;
      }
      if (slot.kind === 'ring') {
        return `${name}[${name}Offset + index] = select(invalidValue, ringValue, ringValid && centreValid && isFiniteValue(ringValue));`;
      }
      return `${name}[${name}Offset + index] = select(invalidValue, ${CURVATURE_EXPRESSIONS[slot.curvature]}, windowValid && centreValid);`;
    })
    .join('\n  ');
  const flatCurvatureSource = /* wgsl */ `
  let gradientSquared = p * p + q * q;
  let weight = 1.0 + gradientSquared;
  let weightRoot = sqrt(weight);
  let weightPower = weight * weightRoot;
  let flatLimit = settings[settingsOffset + 5u];
  let isFlat = gradientSquared <= flatLimit * flatLimit;
  let safeGradient = select(gradientSquared, 1.0, isFlat);
  let safeGradientPower = safeGradient * sqrt(safeGradient);
  let planNumerator = q * q * r - 2.0 * p * q * s + p * p * t;
  let profileNumerator = p * p * r + 2.0 * p * q * s + q * q * t;
  let tangential = select(-planNumerator / (safeGradient * weightRoot), 0.0, isFlat);
  let plan = select(-planNumerator / safeGradientPower, 0.0, isFlat);
  let profile = select(-profileNumerator / (safeGradient * weightPower), 0.0, isFlat);
  let meanCurvature = -((1.0 + q * q) * r - 2.0 * p * q * s + (1.0 + p * p) * t) / (2.0 * weightPower);
  let gaussian = (r * t - s * s) / (weight * weight);
  let unsphericity = sqrt(max(meanCurvature * meanCurvature - gaussian, 0.0));
  let minimal = meanCurvature - unsphericity;
  let maximal = meanCurvature + unsphericity;
  let differenceCurvature = (profile - tangential) * 0.5;
  let horizontalExcess = select(tangential - minimal, 0.0, isFlat);
  let verticalExcess = select(profile - minimal, 0.0, isFlat);
  let accumulation = tangential * profile;
  let ringProduct = horizontalExcess * verticalExcess;
  let rotor = select(((p * p - q * q) * s - p * q * (r - t)) / safeGradientPower, 0.0, isFlat);
  let laplacian = r + t;
  windowValid = windowValid && isFiniteValue(p) && isFiniteValue(q) && isFiniteValue(r) &&
    isFiniteValue(s) && isFiniteValue(t) && isFiniteValue(gradientSquared);`;
  const body = `let row = index / WIDTH;
  let column = index - row * WIDTH;
  let centre = elevationValues[elevationValuesOffset + index];
  let centreValid = elevationValidity[elevationValidityOffset + index] != 0u && isFiniteValue(centre);
  let groundCell = getGroundCellSize(row);
  let wx = groundCell.x;
  let wy = groundCell.y;
  let zFactor = settings[settingsOffset + 2u];
  let cellsValid = wx > 0.0 && wy > 0.0 && isFiniteValue(wx) && isFiniteValue(wy) && isFiniteValue(zFactor);
  let cellMetres = sqrt(wx * wy);
  let invalidValue = bitcast<f32>(0x7fc00000u | (index & 0u));
  windowValid = cellsValid;
  ringValid = cellsValid;
  ${
    needsWindow
      ? `${window.samples}
  ${window.partials}${flatCurvatureSource}`
      : ''
  }
  ${needsRing ? getRingSource(radii, props.ringSquash ?? 'none') : ''}
  ${writes}`;
  return createMapGraphKernelNode<Parameters>(graph, {
    id: options.id,
    operation: 'GPUTerrainCurvature',
    variant: `${method}-${props.cellSizeMode ?? 'uniform'}-${props.borderMode ?? 'clamp'}`,
    bindings,
    invocationCount: props.width * props.height,
    declarations: `const WIDTH: u32 = ${props.width}u;
const HEIGHT: u32 = ${props.height}u;
const WIDTH_SIGNED: i32 = ${props.width};
const HEIGHT_SIGNED: i32 = ${props.height};
var<private> windowValid: bool;
var<private> ringValid: bool;
${TERRAIN_WGSL_HELPERS}
${getTerrainGroundCellSizeWGSL(props.cellSizeMode ?? 'uniform')}
// Elevation of an offset sample minus the centre elevation; flags the window invalid on failure.
fn readWindowDelta(column: u32, row: u32, dx: i32, dy: i32, centre: f32) -> f32 {
  var sampleColumn = i32(column) + dx;
  var sampleRow = i32(row) + dy;
  ${
    clamp
      ? `sampleColumn = clamp(sampleColumn, 0, WIDTH_SIGNED - 1);
  sampleRow = clamp(sampleRow, 0, HEIGHT_SIGNED - 1);`
      : `if (sampleColumn < 0 || sampleRow < 0 || sampleColumn >= WIDTH_SIGNED || sampleRow >= HEIGHT_SIGNED) {
    windowValid = false;
    return 0.0;
  }`
  }
  let sampleIndex = u32(sampleRow) * WIDTH + u32(sampleColumn);
  let value = elevationValues[elevationValuesOffset + sampleIndex];
  if (elevationValidity[elevationValidityOffset + sampleIndex] == 0u || !isFiniteValue(value)) {
    windowValid = false;
    return 0.0;
  }
  return value - centre;
}
// Ring samples never clamp: any sample outside the raster or invalid invalidates the ring.
fn readRingDelta(column: u32, row: u32, dx: i32, dy: i32, centre: f32) -> f32 {
  let sampleColumn = i32(column) + dx;
  let sampleRow = i32(row) + dy;
  if (sampleColumn < 0 || sampleRow < 0 || sampleColumn >= WIDTH_SIGNED || sampleRow >= HEIGHT_SIGNED) {
    ringValid = false;
    return 0.0;
  }
  let sampleIndex = u32(sampleRow) * WIDTH + u32(sampleColumn);
  let value = elevationValues[elevationValuesOffset + sampleIndex];
  if (elevationValidity[elevationValidityOffset + sampleIndex] == 0u || !isFiniteValue(value)) {
    ringValid = false;
    return 0.0;
  }
  return value - centre;
}`,
    body
  });
}
