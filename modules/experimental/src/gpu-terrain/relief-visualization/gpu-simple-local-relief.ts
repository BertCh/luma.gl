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
import {validateTerrainIlluminationView} from '../terrain-illumination/terrain-illumination-utils';
import {
  createReliefMeanFilterSetup,
  GPU_RELIEF_DEFAULT_QUANTUM,
  validateReliefMeanFilter,
  validateReliefMeanFilterRadius
} from './relief-visualization-utils';

/** Number of float32 values read from `GPUSimpleLocalReliefProps.parameters`. */
export const GPU_SIMPLE_LOCAL_RELIEF_PARAMETER_LENGTH = 4;

/** Index of the per-frame window radius in the packed parameters. */
const RADIUS_PARAMETER_INDEX = 1;

/** CPU-side description packed by {@link getGPUSimpleLocalReliefParameterValues}. */
export type GPUSimpleLocalReliefSettings = {
  /** Vertical exaggeration applied to the relief. Defaults to 1. */
  verticalExaggeration?: number;
  /**
   * Window half-width in pixels, an integer in `[1, maximumRadius]`. Defaults to
   * `maximumRadius`. The packer throws for a value that is not an integer of at least 1; a value
   * above `maximumRadius` written to the buffer by other means is clamped to `maximumRadius` by
   * the kernel.
   */
  radius?: number;
};

/**
 * Packs settings into the 4-float layout read by {@link GPUSimpleLocalRelief}:
 * `[verticalExaggeration, radius, 0, 0]`. `radius` is 0 when `settings.radius` is omitted, which
 * the kernel reads as `maximumRadius`.
 *
 * @throws If a value is not finite, `radius` is not an integer of at least 1, or `target` holds
 * fewer than 4 values.
 */
export function getGPUSimpleLocalReliefParameterValues(
  settings: GPUSimpleLocalReliefSettings = {},
  target: Float32Array = new Float32Array(GPU_SIMPLE_LOCAL_RELIEF_PARAMETER_LENGTH)
): Float32Array {
  validateParameterTarget(
    'getGPUSimpleLocalReliefParameterValues',
    target,
    GPU_SIMPLE_LOCAL_RELIEF_PARAMETER_LENGTH
  );
  const verticalExaggeration = settings.verticalExaggeration ?? 1;
  if (!Number.isFinite(verticalExaggeration)) {
    throw new Error('Simple local relief settings must be finite');
  }
  if (
    settings.radius !== undefined &&
    (!Number.isSafeInteger(settings.radius) || settings.radius < 1)
  ) {
    throw new Error('Simple local relief radius must be an integer of at least 1');
  }
  target.fill(0, 0, GPU_SIMPLE_LOCAL_RELIEF_PARAMETER_LENGTH);
  target[0] = verticalExaggeration;
  target[RADIUS_PARAMETER_INDEX] = settings.radius ?? 0;
  return target;
}

/** Caller-owned outputs of {@link GPUSimpleLocalRelief}; at least one is required. */
export type GPUSimpleLocalReliefOutput = {
  /** Optional relief per pixel, `verticalExaggeration * (z - mean_r(z))`, NaN where invalid. */
  relief?: GraphDataView<'float32'>;
  /** Optional per-pixel 1 where the center elevation is valid, else 0. */
  outputValidity?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUSimpleLocalRelief}.
 *
 * Topology: grid size, elevation format and calibration, `maximumRadius`, and which outputs exist.
 * Per-frame: `parameters` (vertical exaggeration and window `radius`) and elevation contents.
 *
 * Cell size model: pixel units. The window radius is counted in pixels and the cell size is not
 * used, exactly as in RVT; output is in elevation units times the exaggeration.
 */
export type GPUSimpleLocalReliefProps = {
  /** Prefix for node and transient IDs. Defaults to `'simple-local-relief'`. */
  id?: string;
  /** Raster grid: its width and height size the kernels. */
  grid: GPURasterMetadata;
  /** Elevation band, buffer or texture; scale, offset, nodata, and validity are honored. */
  elevation: GPURasterBand;
  /** Per-frame parameters with at least 4 float32 values, see {@link getGPUSimpleLocalReliefParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /**
   * Largest window half-width in pixels the contributor serves, an integer of at least 1. It
   * sizes `requiredHalo` and bounds the loops; the per-frame `radius` of the settings picks any
   * value in `[1, maximumRadius]` with the same pipelines. RVT restricts the trend radius to
   * 10 to 50 pixels; this contributor accepts any radius (cost grows linearly with it).
   */
  maximumRadius: number;
  /**
   * Elevation quantum of the exact summed-area table, a power of two. Defaults to `2^-16`
   * (see `GPU_RELIEF_DEFAULT_QUANTUM`). It bounds the error of the window mean by `quantum / 2`.
   * Used only for grids up to `getGPUTerrainSummedAreaMaximumCellCount` cells; elevations of
   * offset magnitude from the GPU-selected first valid anchor of `2^31 * quantum` or more (32768 by default) are treated as invalid there.
   */
  quantum?: number;
  /** Caller-owned outputs. */
  output: GPUSimpleLocalReliefOutput;
};

/**
 * Simple local relief model (SLRM): the elevation minus its local mean, which removes the
 * large-scale trend and leaves small features such as banks, ditches and mounds.
 *
 * References: Hesse (2010), "The Lidar Visualisation Toolbox"; Kokalj and Somrak (2019),
 * "Why not a single image? Combining visualizations to facilitate fieldwork and on-screen
 * mapping", Remote Sensing 11(7). Formulas ported from the Apache-2.0 Relief Visualization
 * Toolbox (`rvt.vis.slrm` and `mean_filter`).
 *
 * The mean is taken over the `(2r + 1)^2` window with edge-clamped coordinates, with nodata
 * excluded from the sum and the count. The deviation is computed without large-value
 * cancellation. Invalid centers receive NaN and validity 0. `requiredHalo` is `maximumRadius`;
 * the window radius is the per-frame `radius` setting.
 *
 * Two paths, chosen at declaration by grid size and device limit only, both matching RVT's float64 mean filter
 * (edge replication, nodata excluded) within their bound. Grids up to
 * `getGPUTerrainSummedAreaMaximumCellCount` cells (about 3344 x 3344 on a default device) use one exact 64-bit
 * summed-area table of elevations quantized to `quantum`: O(1) work per pixel independent of the
 * radius, with an error of at most `quantum / 2` (zero for `|z| >= 128` at the default) plus a few
 * float32 ULP of the relief. `maximumRadius` is then at most 23170. Larger grids use the separable
 * O(radius) float filter, accurate to float32 rounding of the relief. Table-path pixels with
 * `|z - anchor| >= 2^31 * quantum` are invalid.
 */
export class GPUSimpleLocalRelief implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSimpleLocalReliefProps;
  /** Receptive field in pixels. */
  readonly requiredHalo: number;

  constructor(props: GPUSimpleLocalReliefProps) {
    this.id = props.id ?? 'simple-local-relief';
    this.props = props;
    const {id} = this;
    const {output} = props;
    const pixelCount = validateRasterGrid(id, props.grid);
    if (!Number.isSafeInteger(props.maximumRadius) || props.maximumRadius < 1) {
      throw new Error(`${id} maximumRadius must be an integer of at least 1`);
    }
    this.requiredHalo = props.maximumRadius;
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
    validateRasterParameters(id, props.parameters, GPU_SIMPLE_LOCAL_RELIEF_PARAMETER_LENGTH);
    validateRasterBuffersDistinct(
      id,
      [output.relief, output.outputValidity],
      [...getRasterBandViews(props.elevation), props.parameters]
    );
  }

  /** Returns canonicalization, the summed-area table (or scratch) and the mean filter nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    const {width, height} = props.grid;
    validateRasterBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, [
      props.parameters,
      output.relief,
      output.outputValidity
    ]);
    const pixelCount = width * height;
    validateReliefMeanFilterRadius(id, graph.device, pixelCount, props.maximumRadius);
    const source = getRasterBufferBandNodes(graph, id, props.elevation, props.grid, true);
    const filter = createReliefMeanFilterSetup(graph, {
      id,
      width,
      height,
      band: source.band,
      quantum: props.quantum ?? GPU_RELIEF_DEFAULT_QUANTUM
    });
    return [
      ...source.nodes,
      ...filter.nodes,
      ...filter.getFilterNodes({
        id: `${id}-mean`,
        operation: 'GPUSimpleLocalRelief',
        radius: props.maximumRadius,
        radiusSlot: {settings: props.parameters, index: RADIUS_PARAMETER_INDEX},
        deviation:
          output.relief ?? createTransientView(graph, `${id}-relief`, 'float32', pixelCount),
        scale: {settings: props.parameters, index: 0},
        validityOutput: output.outputValidity
      })
    ];
  }
}
