// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPURasterBand} from '../../gpu-raster/index';
import {createMapGraphKernelNode, getWGSLFloatLiteral} from '../map-graph-kernels';
import type {GPUMapGraphRecipe} from '../map-graph-types';
import {validateGraphViewsBelongToGraph} from '../map-graph-utils';
import {
  getTerrainBandViews,
  getTerrainElevationNodes,
  validateTerrainBandBelongsToGraph,
  validateTerrainBuffersDistinct,
  validateTerrainGrid,
  validateTerrainSettings
} from '../terrain-analysis/terrain-analysis-utils';
import {
  TERRAIN_ILLUMINATION_WGSL_CONSTANTS,
  validateTerrainIlluminationView
} from '../terrain-illumination/terrain-illumination-utils';
import {roundHalfEven} from './relief-visualization-utils';

/** Number of float32 values read from `GPULocalDominanceProps.settings`. */
export const GPU_LOCAL_DOMINANCE_PARAMETER_LENGTH = 4;

/** Largest supported number of sample offsets (`distances * angles`) of one recipe. */
export const GPU_LOCAL_DOMINANCE_MAX_SHIFT_COUNT = 8192;

/** CPU-side description packed by {@link getGPULocalDominanceParameterValues}. */
export type GPULocalDominanceSettings = {
  /** Height of the observer above the surface, in elevation units. Positive. Defaults to 1.7. */
  observerHeight?: number;
  /** Vertical exaggeration applied to the elevation. Defaults to 1. */
  verticalExaggeration?: number;
};

/**
 * Packs settings into the 4-float layout read by {@link GPULocalDominance}:
 * `[observerHeight, verticalExaggeration, 0, 0]`.
 *
 * @throws If a value is not finite, `observerHeight <= 0`, or `target` holds fewer than 4 values.
 */
export function getGPULocalDominanceParameterValues(
  settings: GPULocalDominanceSettings = {},
  target: Float32Array = new Float32Array(GPU_LOCAL_DOMINANCE_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_LOCAL_DOMINANCE_PARAMETER_LENGTH) {
    throw new Error('Local dominance settings target must hold 4 values');
  }
  const observerHeight = settings.observerHeight ?? 1.7;
  const verticalExaggeration = settings.verticalExaggeration ?? 1;
  if (!Number.isFinite(observerHeight) || !Number.isFinite(verticalExaggeration)) {
    throw new Error('Local dominance settings must be finite');
  }
  if (observerHeight <= 0) {
    throw new Error('Local dominance observerHeight must be positive');
  }
  target.fill(0, 0, GPU_LOCAL_DOMINANCE_PARAMETER_LENGTH);
  target[0] = observerHeight;
  target[1] = verticalExaggeration;
  return target;
}

/** Sampling geometry of local dominance, see {@link getGPULocalDominanceShifts}. */
export type GPULocalDominanceGeometry = {
  /** Minimum radial distance in pixels. */
  minimumRadius?: number;
  /** Maximum radial distance in pixels. */
  maximumRadius?: number;
  /** Radial step in pixels. */
  radiusIncrement?: number;
  /** Angular step in degrees. */
  angularResolution?: number;
};

/** Baked sample offsets of local dominance. */
export type GPULocalDominanceShifts = {
  /** Number of sample offsets, `distanceCount * angleCount`. */
  count: number;
  /** Row shift per offset, applied as `row - rowShift`. */
  rowShifts: Int32Array;
  /** Column shift per offset, applied as `column - columnShift`. */
  columnShifts: Int32Array;
  /** Pixel distance of each offset. */
  distances: Float64Array;
  /** `2 * distance + radiusIncrement`, the ring-area weight of each offset. */
  distanceFactors: Float64Array;
  /** `sum_d (1 / d) * (2 d + increment) * angleCount`, so `norma = observerHeight * normalization`. */
  normalization: number;
  /** Number of distances. */
  distanceCount: number;
  /** Number of angles. */
  angleCount: number;
};

/**
 * Returns the sample offsets exactly as RVT's `local_dominance` builds them.
 *
 * Distances are `min + inc * k` for `k < int((max - min) / inc + 1)`; angles are `res * j` for
 * `j < int(359 / res + 1)`; the row shift is `round(sin(a) * d)` and the column shift
 * `round(cos(a) * d)`, with Python's round-half-to-even on float64. Offsets are ordered angle
 * major, like RVT's `outer(...).reshape`. They are baked on the CPU because the rounding must
 * not depend on GPU trigonometry. `Math.cos` and numpy's `cos` can differ in the last bit; that
 * matters only when `cos(a) * d` lands within 1 ulp of a half integer (for example
 * `cos(60 deg) * 1`, which is `0.5000000000000001` in both) and then flips the rounding of a
 * single offset by one pixel.
 *
 * @throws If a parameter is not a positive integer, `minimumRadius > maximumRadius`, or the
 * offset count exceeds {@link GPU_LOCAL_DOMINANCE_MAX_SHIFT_COUNT}.
 */
export function getGPULocalDominanceShifts(
  geometry: GPULocalDominanceGeometry = {}
): GPULocalDominanceShifts {
  const minimumRadius = geometry.minimumRadius ?? 10;
  const maximumRadius = geometry.maximumRadius ?? 20;
  const radiusIncrement = geometry.radiusIncrement ?? 1;
  const angularResolution = geometry.angularResolution ?? 15;
  for (const [name, value] of [
    ['minimumRadius', minimumRadius],
    ['maximumRadius', maximumRadius],
    ['radiusIncrement', radiusIncrement],
    ['angularResolution', angularResolution]
  ] as const) {
    if (!Number.isSafeInteger(value) || value < 1) {
      throw new Error(`Local dominance ${name} must be an integer of at least 1`);
    }
  }
  if (minimumRadius > maximumRadius) {
    throw new Error('Local dominance minimumRadius must not exceed maximumRadius');
  }
  if (angularResolution > 359) {
    throw new Error('Local dominance angularResolution must be at most 359');
  }
  const distanceCount = Math.trunc((maximumRadius - minimumRadius) / radiusIncrement + 1);
  const angleCount = Math.trunc(359 / angularResolution + 1);
  const count = distanceCount * angleCount;
  if (count > GPU_LOCAL_DOMINANCE_MAX_SHIFT_COUNT) {
    throw new Error(
      `Local dominance needs ${count} sample offsets, above the limit ${GPU_LOCAL_DOMINANCE_MAX_SHIFT_COUNT}`
    );
  }
  const rowShifts = new Int32Array(count);
  const columnShifts = new Int32Array(count);
  const distances = new Float64Array(count);
  const distanceFactors = new Float64Array(count);
  let normalization = 0;
  for (let angleIndex = 0; angleIndex < angleCount; angleIndex++) {
    const radians = (angularResolution * angleIndex * Math.PI) / 180;
    for (let distanceIndex = 0; distanceIndex < distanceCount; distanceIndex++) {
      const distance = minimumRadius + radiusIncrement * distanceIndex;
      const shift = angleIndex * distanceCount + distanceIndex;
      rowShifts[shift] = roundHalfEven(Math.sin(radians) * distance);
      columnShifts[shift] = roundHalfEven(Math.cos(radians) * distance);
      distances[shift] = distance;
      distanceFactors[shift] = 2 * distance + radiusIncrement;
      normalization += (1 / distance) * (2 * distance + radiusIncrement);
    }
  }
  return {
    count,
    rowShifts,
    columnShifts,
    distances,
    distanceFactors,
    normalization,
    distanceCount,
    angleCount
  };
}

/**
 * Properties for {@link GPULocalDominance}.
 *
 * Topology: grid size, elevation format and calibration, the sampling geometry, and which
 * outputs exist. Per-frame: `settings` (observer height, exaggeration) and elevation contents.
 *
 * Cell size model: pixel units. Radii are counted in pixels and the cell size is not used, as in
 * RVT, so the observer height is in elevation units regardless of the ground resolution.
 */
export type GPULocalDominanceProps = GPULocalDominanceGeometry & {
  /** Prefix for node and transient IDs. Defaults to `'local-dominance'`. */
  id?: string;
  /** Grid width in pixels. */
  width: number;
  /** Grid height in pixels. */
  height: number;
  /** Elevation band, buffer or texture; scale, offset, nodata, and validity are honored. */
  elevation: GPURasterBand;
  /** Per-frame settings with at least 4 float32 values, see {@link getGPULocalDominanceParameterValues}. */
  settings: GraphDataView<'float32'>;
  /** Optional local dominance per pixel, NaN where the center is invalid. */
  dominance?: GraphDataView<'float32'>;
  /** Optional per-pixel 1 where the center elevation is valid, else 0. */
  validity?: GraphDataView<'uint32'>;
};

/**
 * Local dominance: how much an observer standing on the surface at a fixed height rises above the
 * terrain in a ring of distances, which brings out mounds, banks and pits.
 *
 * References: Hesse (2016), "Visualising archaeological topography: Local dominance", in
 * Kokalj and Somrak (2019), Remote Sensing 11(7); formulas ported from the Apache-2.0 Relief
 * Visualization Toolbox (`rvt.vis.local_dominance`, adapted from Hesse's LiVT).
 *
 * For every offset `(distance d, angle a)` the sample is the edge-clamped pixel at
 * `(row - round(sin a * d), column - round(cos a * d))`. When `ve * z + h > ve * z_s` the offset
 * contributes `(ve * z + h - ve * z_s) / d * (2 d + increment)`, and the sum is divided by
 * `norma = h * sum_d ((2 d + increment) / d) * angleCount`, so a flat surface scores exactly 1
 * up to float32 rounding. Invalid samples contribute 0 but stay in `norma`; invalid centers
 * receive NaN (RVT leaves them 0). The difference `ve * (z - z_s)` is formed before the observer
 * height is added, which is more accurate than RVT's float32 `ve * z` products and identical in
 * exact arithmetic. Offsets are baked into the WGSL (see {@link getGPULocalDominanceShifts}), so
 * cost is `distances * angles` samples per pixel. `requiredHalo` is `maximumRadius`.
 */
export class GPULocalDominance implements GPUMapGraphRecipe {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'local-dominance';
  /** Validated properties. */
  readonly props: GPULocalDominanceProps;
  /** Baked sample offsets. */
  readonly shifts: GPULocalDominanceShifts;
  /** Receptive field in pixels: the maximum radius. */
  readonly requiredHalo: number;

  constructor(props: GPULocalDominanceProps) {
    this.id = props.id ?? this.recipe;
    this.props = props;
    const {id} = this;
    const pixelCount = validateTerrainGrid(id, props.width, props.height);
    this.shifts = getGPULocalDominanceShifts(props);
    this.requiredHalo = props.maximumRadius ?? 20;
    if (!props.dominance && !props.validity) {
      throw new Error(`${id} requires at least one output`);
    }
    validateTerrainIlluminationView(id, 'dominance', props.dominance, 'float32', pixelCount);
    validateTerrainIlluminationView(id, 'validity', props.validity, 'uint32', pixelCount);
    validateTerrainSettings(id, props.settings, GPU_LOCAL_DOMINANCE_PARAMETER_LENGTH);
    validateTerrainBuffersDistinct(
      id,
      [props.dominance, props.validity],
      [...getTerrainBandViews(props.elevation), props.settings]
    );
  }

  /** Returns canonicalization and the single sampling node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, shifts} = this;
    const {width, height} = props;
    validateTerrainBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, [props.settings, props.dominance, props.validity]);
    const pixelCount = width * height;
    const source = getTerrainElevationNodes(graph, id, props.elevation, width, height, true);
    const values = source.band.storage.values as GraphDataView<'float32'>;
    const validity = source.band.validity as GraphDataView<'uint32'>;
    const target =
      props.dominance ?? createTransientView(graph, `${id}-dominance`, 'float32', pixelCount);
    const bindings = [
      {name: 'values', view: values, type: 'f32', access: 'read'},
      {name: 'validity', view: validity, type: 'u32', access: 'read'},
      {name: 'settings', view: props.settings, type: 'f32', access: 'read'},
      {name: 'dominance', view: target, type: 'f32', access: 'read_write'}
    ] as const;
    const offsets = Array.from(
      shifts.rowShifts,
      (rowShift, shift) => `vec2<i32>(${rowShift}, ${shifts.columnShifts[shift]})`
    ).join(', ');
    const factors = Array.from(shifts.distanceFactors, (factor, shift) =>
      getWGSLFloatLiteral(factor / shifts.distances[shift])
    ).join(', ');
    return [
      ...source.nodes,
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-sample`,
        operation: 'GPULocalDominance',
        variant: 'sample',
        bindings: props.validity
          ? [
              ...bindings,
              {name: 'validityValues', view: props.validity, type: 'u32', access: 'read_write'}
            ]
          : bindings,
        invocationCount: pixelCount,
        declarations: `${TERRAIN_ILLUMINATION_WGSL_CONSTANTS}
const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const SHIFT_COUNT: u32 = ${shifts.count}u;
const NORMALIZATION: f32 = ${getWGSLFloatLiteral(shifts.normalization)};
var<private> SHIFTS: array<vec2<i32>, ${shifts.count}> = array<vec2<i32>, ${shifts.count}>(${offsets});
var<private> FACTORS: array<f32, ${shifts.count}> = array<f32, ${shifts.count}>(${factors});`,
        body: `let isValid = validity[validityOffset + index] != 0u;
  var value = getNaN(index);
  if (isValid) {
    let observerHeight = settings[settingsOffset];
    let verticalExaggeration = settings[settingsOffset + 1u];
    let center = values[valuesOffset + index];
    let column = i32(index % WIDTH);
    let row = i32(index / WIDTH);
    var sum = 0.0;
    for (var shift = 0u; shift < SHIFT_COUNT; shift++) {
      let offset = SHIFTS[shift];
      let sampleRow = clamp(row - offset.x, 0, i32(HEIGHT) - 1);
      let sampleColumn = clamp(column - offset.y, 0, i32(WIDTH) - 1);
      let sampleIndex = u32(sampleRow) * WIDTH + u32(sampleColumn);
      if (validity[validityOffset + sampleIndex] != 0u) {
        let rise = verticalExaggeration * (center - values[valuesOffset + sampleIndex]) + observerHeight;
        if (rise > 0.0) {
          sum += rise * FACTORS[shift];
        }
      }
    }
    value = sum / (observerHeight * NORMALIZATION);
  }
  dominance[dominanceOffset + index] = value;
  ${props.validity ? 'validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);' : ''}`
      })
    ];
  }
}
