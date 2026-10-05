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
import {validateTerrainIlluminationView} from '../terrain-illumination/terrain-illumination-utils';
import {
  createReliefMeanFilterScratch,
  getReliefMeanFilterNodes
} from './relief-visualization-utils';

/** Number of float32 values read from `GPUSimpleLocalReliefProps.settings`. */
export const GPU_SIMPLE_LOCAL_RELIEF_PARAMETER_LENGTH = 4;

/** CPU-side description packed by {@link getGPUSimpleLocalReliefParameterValues}. */
export type GPUSimpleLocalReliefSettings = {
  /** Vertical exaggeration applied to the relief. Defaults to 1. */
  verticalExaggeration?: number;
};

/**
 * Packs settings into the 4-float layout read by {@link GPUSimpleLocalRelief}:
 * `[verticalExaggeration, 0, 0, 0]`.
 *
 * @throws If a value is not finite or `target` holds fewer than 4 values.
 */
export function getGPUSimpleLocalReliefParameterValues(
  settings: GPUSimpleLocalReliefSettings = {},
  target: Float32Array = new Float32Array(GPU_SIMPLE_LOCAL_RELIEF_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_SIMPLE_LOCAL_RELIEF_PARAMETER_LENGTH) {
    throw new Error('Simple local relief settings target must hold 4 values');
  }
  const verticalExaggeration = settings.verticalExaggeration ?? 1;
  if (!Number.isFinite(verticalExaggeration)) {
    throw new Error('Simple local relief settings must be finite');
  }
  target.fill(0, 0, GPU_SIMPLE_LOCAL_RELIEF_PARAMETER_LENGTH);
  target[0] = verticalExaggeration;
  return target;
}

/**
 * Properties for {@link GPUSimpleLocalRelief}.
 *
 * Topology: grid size, elevation format and calibration, `radius`, and which outputs exist.
 * Per-frame: `settings` (vertical exaggeration) and elevation contents.
 *
 * Cell size model: pixel units. The window radius is counted in pixels and the cell size is not
 * used, exactly as in RVT; output is in elevation units times the exaggeration.
 */
export type GPUSimpleLocalReliefProps = {
  /** Prefix for node and transient IDs. Defaults to `'simple-local-relief'`. */
  id?: string;
  /** Grid width in pixels. */
  width: number;
  /** Grid height in pixels. */
  height: number;
  /** Elevation band, buffer or texture; scale, offset, nodata, and validity are honored. */
  elevation: GPURasterBand;
  /** Per-frame settings with at least 4 float32 values, see {@link getGPUSimpleLocalReliefParameterValues}. */
  settings: GraphDataView<'float32'>;
  /**
   * Window half-width in pixels, an integer of at least 1. RVT restricts the trend radius to
   * 10 to 50 pixels; this recipe accepts any radius (cost grows linearly with it).
   */
  radius: number;
  /** Optional relief per pixel, `verticalExaggeration * (z - mean_r(z))`, NaN where invalid. */
  relief?: GraphDataView<'float32'>;
  /** Optional per-pixel 1 where the center elevation is valid, else 0. */
  validity?: GraphDataView<'uint32'>;
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
 * cancellation, see {@link getReliefMeanFilterNodes}. Invalid centers receive NaN and validity 0.
 * `requiredHalo` is `radius`.
 */
export class GPUSimpleLocalRelief implements GPUMapGraphRecipe {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'simple-local-relief';
  /** Validated properties. */
  readonly props: GPUSimpleLocalReliefProps;
  /** Receptive field in pixels. */
  readonly requiredHalo: number;

  constructor(props: GPUSimpleLocalReliefProps) {
    this.id = props.id ?? this.recipe;
    this.props = props;
    const {id} = this;
    const pixelCount = validateTerrainGrid(id, props.width, props.height);
    if (!Number.isSafeInteger(props.radius) || props.radius < 1) {
      throw new Error(`${id} radius must be an integer of at least 1`);
    }
    this.requiredHalo = props.radius;
    if (!props.relief && !props.validity) {
      throw new Error(`${id} requires at least one output`);
    }
    validateTerrainIlluminationView(id, 'relief', props.relief, 'float32', pixelCount);
    validateTerrainIlluminationView(id, 'validity', props.validity, 'uint32', pixelCount);
    validateTerrainSettings(id, props.settings, GPU_SIMPLE_LOCAL_RELIEF_PARAMETER_LENGTH);
    validateTerrainBuffersDistinct(
      id,
      [props.relief, props.validity],
      [...getTerrainBandViews(props.elevation), props.settings]
    );
  }

  /** Returns canonicalization plus the row and column mean filter nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {width, height} = props;
    validateTerrainBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, [props.settings, props.relief, props.validity]);
    const pixelCount = width * height;
    const source = getTerrainElevationNodes(graph, id, props.elevation, width, height, true);
    return [
      ...source.nodes,
      ...getReliefMeanFilterNodes(graph, {
        id: `${id}-mean`,
        operation: 'GPUSimpleLocalRelief',
        width,
        height,
        radius: props.radius,
        values: source.band.storage.values as GraphDataView<'float32'>,
        validity: source.band.validity as GraphDataView<'uint32'>,
        scratch: createReliefMeanFilterScratch(graph, id, pixelCount),
        deviation:
          props.relief ?? createTransientView(graph, `${id}-relief`, 'float32', pixelCount),
        scale: {settings: props.settings, index: 0},
        validityOutput: props.validity
      })
    ];
  }
}
