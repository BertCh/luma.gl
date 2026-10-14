// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPURasterBand, GPURasterMetadata} from '../../gpu-raster/index';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import {
  getRasterBandViews,
  getRasterBufferBandNodes,
  validateRasterBandBelongsToGraph,
  validateParameterTarget,
  validateRasterBuffersDistinct,
  validateRasterGrid,
  validateRasterParameters
} from '../../gpu-raster/raster-band-utils';
import {
  TERRAIN_ILLUMINATION_WGSL_CONSTANTS,
  validateTerrainIlluminationView
} from '../terrain-illumination/terrain-illumination-utils';
import {
  createReliefMeanFilterSetup,
  GPU_RELIEF_DEFAULT_QUANTUM,
  validateReliefMeanFilter,
  validateReliefMeanFilterRadius
} from './relief-visualization-utils';

/** Number of float32 values read from `GPUMultiScaleReliefProps.parameters`. */
export const GPU_MULTI_SCALE_RELIEF_PARAMETER_LENGTH = 4;

/** Feature-size description shared by {@link getGPUMultiScaleReliefRadii} and the contributor. */
export type GPUMultiScaleReliefScales = {
  /** Pixel size in projected metres. Positive. */
  resolution: number;
  /** Smallest feature to detect in metres; values below `resolution` are raised to it. */
  featureMinimum: number;
  /** Largest feature to detect in metres. */
  featureMaximum: number;
  /** Radius growth exponent. Truncated to an integer, at least 1. 1 gives linear radii. */
  scalingFactor: number;
};

/** Mean filter radii of a multi-scale relief model, see {@link getGPUMultiScaleReliefRadii}. */
export type GPUMultiScaleReliefRadii = {
  /** First index `i` of Orengo and Petrie (2018). */
  firstIndex: number;
  /** Last index `n`, greater than `firstIndex`. */
  lastIndex: number;
  /** Radius `i^s` in pixels of the finest mean filter; 0 means the unfiltered elevation. */
  firstRadius: number;
  /** Radius `n^s` in pixels of the coarsest mean filter. */
  lastRadius: number;
  /** Every RVT filter radius `k^s` for `k = i..n`. */
  radii: number[];
};

/**
 * Computes the mean filter radii of RVT's multi-scale relief model.
 *
 * `featureMinimum` is raised to `resolution`, `i = floor(((fmin - res) / (2 res))^(1/s))`,
 * `n = ceil(((fmax - res) / (2 res))^(1/s))`, and the radii are `k^s` for `k = i..n`
 * (Orengo and Petrie 2018, as implemented by RVT `msrm`). Radii are in pixels.
 *
 * @throws If an input is not finite or `resolution <= 0`, `scalingFactor < 1`, or `n <= i` (the
 * feature range is too narrow to span two filters).
 */
export function getGPUMultiScaleReliefRadii(
  scales: GPUMultiScaleReliefScales
): GPUMultiScaleReliefRadii {
  const {resolution, featureMaximum} = scales;
  if (
    ![resolution, scales.featureMinimum, featureMaximum, scales.scalingFactor].every(
      Number.isFinite
    )
  ) {
    throw new Error('Multi-scale relief scales must be finite');
  }
  if (resolution <= 0) {
    throw new Error('Multi-scale relief resolution must be positive');
  }
  const scalingFactor = Math.trunc(scales.scalingFactor);
  if (scalingFactor < 1) {
    throw new Error('Multi-scale relief scalingFactor must be at least 1');
  }
  const featureMinimum = Math.max(scales.featureMinimum, resolution);
  const firstIndex = Math.floor(
    ((featureMinimum - resolution) / (2 * resolution)) ** (1 / scalingFactor)
  );
  const lastIndex = Math.ceil(
    ((featureMaximum - resolution) / (2 * resolution)) ** (1 / scalingFactor)
  );
  if (!(lastIndex > firstIndex)) {
    throw new Error(
      'Multi-scale relief feature range is too narrow: need n > i, widen featureMaximum or lower featureMinimum'
    );
  }
  const radii = Array.from(
    {length: lastIndex - firstIndex + 1},
    (_, offset) => (firstIndex + offset) ** scalingFactor
  );
  return {
    firstIndex,
    lastIndex,
    firstRadius: radii[0],
    lastRadius: radii[radii.length - 1],
    radii
  };
}

/** CPU-side description packed by {@link getGPUMultiScaleReliefParameterValues}. */
export type GPUMultiScaleReliefSettings = {
  /** Vertical exaggeration applied to the relief. Defaults to 1. */
  verticalExaggeration?: number;
};

/**
 * Packs settings into the 4-float layout read by {@link GPUMultiScaleRelief}:
 * `[verticalExaggeration, 0, 0, 0]`.
 *
 * @throws If a value is not finite or `target` holds fewer than 4 values.
 */
export function getGPUMultiScaleReliefParameterValues(
  settings: GPUMultiScaleReliefSettings = {},
  target: Float32Array = new Float32Array(GPU_MULTI_SCALE_RELIEF_PARAMETER_LENGTH)
): Float32Array {
  validateParameterTarget(
    'getGPUMultiScaleReliefParameterValues',
    target,
    GPU_MULTI_SCALE_RELIEF_PARAMETER_LENGTH
  );
  const verticalExaggeration = settings.verticalExaggeration ?? 1;
  if (!Number.isFinite(verticalExaggeration)) {
    throw new Error('Multi-scale relief settings must be finite');
  }
  target.fill(0, 0, GPU_MULTI_SCALE_RELIEF_PARAMETER_LENGTH);
  target[0] = verticalExaggeration;
  return target;
}

/** Caller-owned outputs of {@link GPUMultiScaleRelief}; at least one is required. */
export type GPUMultiScaleReliefOutput = {
  /** Optional relief per pixel, in elevation units times exaggeration, NaN where invalid. */
  relief?: GraphDataView<'float32'>;
  /** Optional per-pixel 1 where the center elevation is valid, else 0. */
  outputValidity?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUMultiScaleRelief}.
 *
 * Topology: grid size, elevation format and calibration, the feature-size scales, and which
 * outputs exist. Per-frame: `parameters` (vertical exaggeration) and elevation contents.
 *
 * Cell size model: projected metres. `resolution` and the feature sizes are metres and are
 * converted to pixel radii once at construction; geographic or Web Mercator grids with
 * latitude-dependent spacing are not supported (pass a representative metres-per-pixel).
 */
export type GPUMultiScaleReliefProps = GPUMultiScaleReliefScales & {
  /** Prefix for node and transient IDs. Defaults to `'multi-scale-relief'`. */
  id?: string;
  /** Raster grid: its width and height size the kernels. */
  grid: GPURasterMetadata;
  /** Elevation band, buffer or texture; scale, offset, nodata, and validity are honored. */
  elevation: GPURasterBand;
  /** Per-frame parameters with at least 4 float32 values, see {@link getGPUMultiScaleReliefParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /**
   * Elevation quantum of the exact summed-area table, a power of two. Defaults to `2^-16`
   * (see `GPU_RELIEF_DEFAULT_QUANTUM`). Both filters read one table. See {@link GPUMultiScaleRelief}
   * for the path choice and error bound.
   */
  quantum?: number;
  /** Caller-owned outputs. */
  output: GPUMultiScaleReliefOutput;
};

/**
 * Multi-scale relief model (MSRM): the mean of the differences between consecutive low-pass
 * filtered surfaces, which highlights features between a minimum and a maximum size.
 *
 * References: Orengo and Petrie (2018), "Multi-scale relief model (MSRM): a new algorithm for the
 * visualization of subtle topographic change of variable size in digital elevation models",
 * Earth Surface Processes and Landforms 43(6); Kokalj and Somrak (2019); formulas ported from the
 * Apache-2.0 Relief Visualization Toolbox (`rvt.vis.msrm`).
 *
 * RVT sums the differences of consecutive mean filters with radii `k^s`, `k = i..n`. The sum
 * telescopes: `msrm = (lpf_i - lpf_n) / (n - i)`, where `lpf_k` is the mean filter of radius
 * `k^s` and `lpf` of radius 0 is the elevation itself. So only two filters are computed, whatever
 * the number of scales, and with `d_r = z - lpf_r` (the mean filters return that precise
 * deviation directly) `msrm = ve * (d_last - d_first) / (n - i)`. The test
 * oracle runs RVT's loop literally and proves the equality. Invalid centers receive NaN.
 * `requiredHalo` is the coarsest radius.
 *
 * Two mean filter paths, chosen at declaration by grid size and device limit only, both matching RVT's float64
 * mean filter within their bound. Grids up to `getGPUTerrainSummedAreaMaximumCellCount` cells
 * use one exact 64-bit summed-area table (built once for both filters) of elevations quantized to
 * `quantum`: O(1) per pixel, error at most `quantum / 2` per filter (zero for `|z| >= 128` at the
 * default) plus a few float32 ULP of the deviation; the coarsest radius is then at most 23170.
 * Larger grids use the separable O(radius) float filter. Table-path pixels with
 * `|z - anchor| >= 2^31 * quantum` are invalid.
 */
export class GPUMultiScaleRelief implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUMultiScaleReliefProps;
  /** Filter radii derived from the feature sizes. */
  readonly radii: GPUMultiScaleReliefRadii;
  /** Receptive field in pixels: the coarsest filter radius. */
  readonly requiredHalo: number;

  constructor(props: GPUMultiScaleReliefProps) {
    this.id = props.id ?? 'multi-scale-relief';
    this.props = props;
    const {id} = this;
    const {output} = props;
    const pixelCount = validateRasterGrid(id, props.grid);
    this.radii = getGPUMultiScaleReliefRadii(props);
    this.requiredHalo = this.radii.lastRadius;
    validateReliefMeanFilter(id, props.quantum ?? GPU_RELIEF_DEFAULT_QUANTUM);
    if (!output.relief && !output.outputValidity) {
      throw new Error(`${id} requires at least one output`);
    }
    validateTerrainIlluminationView(id, 'output.relief', output.relief, 'float32', pixelCount);
    validateTerrainIlluminationView(
      id,
      'output.outputValidity',
      output.outputValidity,
      'uint32',
      pixelCount
    );
    validateRasterParameters(id, props.parameters, GPU_MULTI_SCALE_RELIEF_PARAMETER_LENGTH);
    validateRasterBuffersDistinct(
      id,
      [output.relief, output.outputValidity],
      [...getRasterBandViews(props.elevation), props.parameters]
    );
  }

  /** Returns canonicalization, up to two mean filters, and the combine node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, radii} = this;
    const {output} = props;
    const {width, height} = props.grid;
    validateRasterBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, [
      props.parameters,
      output.relief,
      output.outputValidity
    ]);
    const pixelCount = width * height;
    validateReliefMeanFilterRadius(id, graph.device, pixelCount, this.radii.lastRadius);
    const source = getRasterBufferBandNodes(graph, id, props.elevation, props.grid, true);
    const filter = createReliefMeanFilterSetup(graph, {
      id,
      width,
      height,
      band: source.band,
      quantum: props.quantum ?? GPU_RELIEF_DEFAULT_QUANTUM
    });
    const nodes: GPUCommandNode<Parameters>[] = [...source.nodes, ...filter.nodes];
    const coarse = createTransientView(graph, `${id}-coarse-deviation`, 'float32', pixelCount);
    nodes.push(
      ...filter.getFilterNodes({
        id: `${id}-last`,
        operation: 'GPUMultiScaleRelief',
        radius: radii.lastRadius,
        deviation: coarse
      })
    );
    const hasFine = radii.firstRadius > 0;
    const fine = hasFine
      ? createTransientView(graph, `${id}-fine-deviation`, 'float32', pixelCount)
      : undefined;
    if (fine) {
      nodes.push(
        ...filter.getFilterNodes({
          id: `${id}-first`,
          operation: 'GPUMultiScaleRelief',
          radius: radii.firstRadius,
          deviation: fine
        })
      );
    }
    const bindings: WGSLKernelBinding[] = [
      {name: 'coarse', view: coarse, type: 'f32', access: 'read'},
      {name: 'settings', view: props.parameters, type: 'f32', access: 'read'}
    ];
    if (fine) {
      bindings.push({name: 'fine', view: fine, type: 'f32', access: 'read'});
    }
    const target =
      output.relief ?? createTransientView(graph, `${id}-relief`, 'float32', pixelCount);
    bindings.push({name: 'relief', view: target, type: 'f32', access: 'read_write'});
    if (output.outputValidity) {
      bindings.push({
        name: 'validityValues',
        view: output.outputValidity,
        type: 'u32',
        access: 'read_write'
      });
    }
    // The validity of the coarse deviation (non-NaN) is the validity of the centre.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-combine`,
        operation: 'GPUMultiScaleRelief',
        variant: 'combine',
        bindings,
        invocationCount: pixelCount,
        declarations: `${TERRAIN_ILLUMINATION_WGSL_CONSTANTS}
const SCALE_COUNT: f32 = ${radii.lastIndex - radii.firstIndex}.0;`,
        body: `let coarseValue = coarse[coarseOffset + index];
  let isValid = isFiniteValue(coarseValue);
  let fineValue = ${fine ? 'fine[fineOffset + index]' : '0.0'};
  let value = settings[settingsOffset] * (coarseValue - fineValue) / SCALE_COUNT;
  relief[reliefOffset + index] = select(getNaN(index), value, isValid);
  ${output.outputValidity ? 'validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);' : ''}`
      })
    );
    return nodes;
  }
}
