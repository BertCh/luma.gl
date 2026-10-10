// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GraphVectorView,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import type {GPUSpatialContext} from '../contracts/index';
import {RHUMB_WGSL} from './rhumb-wgsl';
import {GEODESIC_WGSL} from './geodesic-wgsl';
import {
  getVincentyWGSL,
  GPU_GEODESIC_DEFAULT_ITERATIONS,
  SPHERE_PAIR_WGSL,
  type GPUGeodesicModel
} from './geodesic-kernels';
import {
  resolveGeodesicSpatialContext,
  validateGeodesicColumns,
  validateGeodesicOptions
} from './gpu-geodesic-pairs';

const OPERATION = 'GPUGeodesicDestination';

/** Per-row outputs of {@link GPUGeodesicDestination}. */
export type GPUGeodesicDestinationOutput = {
  /** Destination longitude/latitude, longitude continuous with the origin (may leave ±180). */
  destinations?: GraphDataView<'float32x2'>;
  /** Bearing on arrival, degrees clockwise from north in `(-180, 180]`. */
  finalBearings?: GraphDataView<'float32'>;
  /** `'wgs84'` only: `1` when Vincenty's direct iteration converged. */
  converged?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUGeodesicDestination}.
 *
 * Per-frame (no recompile): the contents of every input column. Compile-time: the row count,
 * `model`, `radius`, `iterations`, and which outputs are present.
 */
export type GPUGeodesicDestinationProps = {
  /** Prefix for generated node IDs. Defaults to `'geodesic-destination'`. */
  id?: string;
  /** Longitude/latitude metric and output-unit contract. */
  spatialContext: GPUSpatialContext;
  /** Origins, longitude/latitude degrees. */
  origins: GraphDataView<'float32x2'>;
  /** Initial bearings in degrees clockwise from north, aligned with `origins`. */
  bearings: GraphDataView<'float32'>;
  /** Distances in radius units (`'sphere'`, `'rhumb'`) or meters (`'wgs84'`), aligned with `origins`. */
  distances: GraphDataView<'float32'>;
  /** Compile-time Vincenty iteration cap for `'wgs84'`. Default 16. */
  iterations?: number;
  /** Per-row outputs; at least one. */
  output: GPUGeodesicDestinationOutput;
};

/**
 * Destination points from origins, bearings and distances (turf `destination`, PostGIS
 * `ST_Project`), for example range rings, sector wedges or projected vessel positions.
 *
 * `'sphere'` uses the shared spherical destination formula; `'wgs84'` runs Vincenty's direct
 * solution in f32 with a compile-time iteration cap (it converges in a few iterations for every
 * distance shorter than half the meridian). One invocation per row; deterministic.
 */
export class GPUGeodesicDestination implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUGeodesicDestinationProps;
  /** Canonical coordinate and metric contract. */
  readonly spatialContext: GPUSpatialContext;
  /** Resolved model. */
  readonly model: GPUGeodesicModel;
  /** Resolved radius. */
  readonly radius: number;
  /** Resolved Vincenty iteration cap. */
  readonly iterations: number;

  constructor(props: GPUGeodesicDestinationProps) {
    this.id = props.id ?? 'geodesic-destination';
    this.props = props;
    this.spatialContext = props.spatialContext;
    const metric = resolveGeodesicSpatialContext(this.id, this.spatialContext);
    this.model = metric.model;
    this.radius = metric.radius;
    this.iterations = props.iterations ?? GPU_GEODESIC_DEFAULT_ITERATIONS;
    validateGeodesicOptions(this.id, this.model, this.radius, this.iterations);
    const {id} = this;
    for (const [name, view, format] of [
      ['origins', props.origins, 'float32x2'],
      ['bearings', props.bearings, 'float32'],
      ['distances', props.distances, 'float32']
    ] as const) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
      validatePackedView(view, [format], `${id} ${name}`);
    }
    const rowCount = props.origins.length;
    if (rowCount < 1 || props.bearings.length !== rowCount || props.distances.length !== rowCount) {
      throw new Error(
        `${id} needs at least one row and bearings and distances aligned with origins`
      );
    }
    validateGeodesicColumns(
      id,
      props.output,
      {destinations: 'float32x2', finalBearings: 'float32', converged: 'uint32'},
      rowCount
    );
    if (props.output.converged && this.model !== 'wgs84') {
      throw new Error(`${id} output.converged requires model 'wgs84'`);
    }
    validateGraphOutputsDisjointFromInputs(id, Object.values(props.output), [
      props.origins,
      props.bearings,
      props.distances
    ]);
  }

  /** Returns the single per-row node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.origins,
      props.bearings,
      props.distances,
      ...Object.values(output)
    ]);
    const bindings: WGSLKernelBinding[] = [
      {name: 'origins', view: props.origins, type: 'f32', access: 'read'},
      {name: 'bearings', view: props.bearings, type: 'f32', access: 'read'},
      {name: 'distances', view: props.distances, type: 'f32', access: 'read'}
    ];
    const statements: string[] = [];
    if (output.destinations) {
      bindings.push({
        name: 'destinations',
        view: output.destinations,
        type: 'f32',
        access: 'read_write'
      });
      statements.push(`destinations[destinationsOffset + 2u * index] = destination.x;
  destinations[destinationsOffset + 2u * index + 1u] = destination.y;`);
    }
    if (output.finalBearings) {
      bindings.push({
        name: 'finalBearings',
        view: output.finalBearings,
        type: 'f32',
        access: 'read_write'
      });
      statements.push('finalBearings[finalBearingsOffset + index] = finalBearing;');
    }
    if (output.converged) {
      bindings.push({name: 'converged', view: output.converged, type: 'u32', access: 'read_write'});
      statements.push('converged[convergedOffset + index] = select(0u, 1u, isConverged);');
    }
    const wgs84 = this.model === 'wgs84';
    const rhumb = this.model === 'rhumb';
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-destination`,
        operation: OPERATION,
        variant: this.model,
        bindings,
        invocationCount: props.origins.length,
        declarations: `${GEODESIC_WGSL}
${wgs84 ? getVincentyWGSL(this.iterations) : ''}
${SPHERE_PAIR_WGSL}
${this.model === 'rhumb' ? RHUMB_WGSL : ''}
const RADIUS: f32 = ${getWGSLFloatLiteral(this.radius)};`,
        body: /* wgsl */ `let origin = vec2<f32>(origins[originsOffset + 2u * index], origins[originsOffset + 2u * index + 1u]);
  let bearing = bearings[bearingsOffset + index];
  let distance = distances[distancesOffset + index];
  ${
    wgs84
      ? `let direct = vincentyDirect(origin, bearing, distance);
  let destination = direct.destination;
  let finalBearing = direct.finalBearing;
  let isConverged = direct.converged;`
      : rhumb
        ? `let rhumbResult = rhumbDestination(origin, bearing, distance / RADIUS);
  let destination = rhumbResult.destination;
  let finalBearing = rhumbResult.finalBearing;
  let isConverged = true;`
        : `let destination = geodesicDestination(origin, bearing, distance / RADIUS);
  let finalBearing = sphereFinalBearingDegrees(origin, destination);
  let isConverged = true;`
  }
  ${statements.join('\n  ')}`
      })
    ];
  }
}
