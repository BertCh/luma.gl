// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {captureGraphCommandNodes} from '../../utils/gpu-contributor-utils';
import type {GPUFloat32Positions, GPUGeospatialPositions} from '../../geospatial/types';
import {GPUProjection, type GPUProjectionValidity} from '../../gpu-project/gpu-projection';
import {compileProjectionPlan} from '../../gpu-project/projection-plan';
import type {ProjectionBounds, ProjectionPlan} from '../../gpu-project/types';
import type {GPULocalFrame} from '../../analysis-core/contracts/local-frame';
import type {GPUSpatialContext} from '../../analysis-core/contracts/spatial-context';

/** Props of {@link GPULocalFrameProjection}. */
export type GPULocalFrameProjectionProps = {
  /** Prefix of generated graph nodes. Defaults to `'gpu-local-frame-projection'`. */
  id?: string;
  /** Frame created with `createGPULocalFrame`. */
  frame: GPULocalFrame;
  /**
   * Longitude/latitude positions in degrees: `float32x2` rows, or raw binary64 rows stored as
   * `uint32x4` (`Float64Array` word order) to keep the full input precision.
   */
  positions: GPUGeospatialPositions;
  /** Caller-owned `float32x2` frame coordinates in meters, one row per position. */
  output: GPUFloat32Positions;
  /**
   * `[minLongitude, minLatitude, maxLongitude, maxLatitude]` covering every position, in the
   * positions' own longitude convention (data crossing the antimeridian must be stored with
   * continuous longitudes, for example 179 to 181). Rows outside the bounds are rejected.
   */
  bounds: ProjectionBounds;
  /** Maximum projection error in meters accepted by the patch compiler. Defaults to `0.01`. */
  tolerance?: number;
  /** Optional `uint32` rows: 1 for projected rows, 0 for rows outside `bounds` (output 0). */
  validity?: GPUProjectionValidity;
};

/**
 * Projects GPU-resident longitude/latitude positions into a `createGPULocalFrame` frame on the GPU,
 * so planar operators can consume them without a CPU readback ("project first").
 *
 * The frame's binary64 projection is compiled on the CPU into error-bounded local polynomial
 * patches over `bounds` and evaluated by `GPUProjection` (`gpu-project`). Output rows are absolute
 * frame meters: the plan is referenced to the frame origin, so float32 rounding is relative to the
 * frame extent. Pass {@link spatialContext} (the frame's planar context) to downstream operators.
 */
export class GPULocalFrameProjection implements GPUCommandNodeProducer {
  readonly id: string;
  readonly frame: GPULocalFrame;
  /** Planar context of the output rows, carrying the frame identity. */
  readonly spatialContext: GPUSpatialContext;
  /** Compiled plan; its destination origin is the frame origin `[0, 0]`. */
  readonly plan: ProjectionPlan;
  private readonly projection: GPUProjection;

  constructor(props: GPULocalFrameProjectionProps) {
    this.id = props.id ?? 'gpu-local-frame-projection';
    this.frame = props.frame;
    this.spatialContext = props.frame.spatialContext;
    const compiled = compileProjectionPlan({
      projection: (coordinates: number[]) => props.frame.project(coordinates[0], coordinates[1]),
      bounds: props.bounds,
      tolerance: props.tolerance ?? 0.01
    });
    // Reference the float32 output to the frame origin instead of the bounds center: packed
    // patch offsets become frame coordinates, so the output rows are absolute frame meters.
    this.plan = {...compiled, destinationOrigin: [0, 0]};
    this.projection = new GPUProjection({
      id: this.id,
      positions: props.positions,
      output: props.output,
      validity: props.validity,
      plan: this.plan
    });
  }

  /** Returns the projection passes; add the contributor with `graph.add(projection)`. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    return captureGraphCommandNodes(graph, () => this.projection.addToGraph(graph));
  }

  /** Releases the owned plan buffer. */
  destroy(): void {
    this.projection.destroy();
  }
}
