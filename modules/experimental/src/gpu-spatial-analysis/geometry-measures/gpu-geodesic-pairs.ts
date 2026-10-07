// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GraphVectorView,
  validatePackedUint32View,
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
import {RHUMB_WGSL} from './rhumb-wgsl';
import {GEODESIC_WGSL, GPU_GEODESIC_MEAN_EARTH_RADIUS} from './geodesic-wgsl';
import {
  getVincentyWGSL,
  GPU_GEODESIC_DEFAULT_ITERATIONS,
  SPHERE_PAIR_WGSL,
  type GPUGeodesicModel
} from './geodesic-kernels';

const OPERATION = 'GPUGeodesicPairs';

/** Per-pair outputs of {@link GPUGeodesicPairs}; each optional, one row per pair. */
export type GPUGeodesicPairsOutput = {
  /** Geodesic distance in radius units (`'sphere'`, `'rhumb'`) or meters (`'wgs84'`). */
  distances?: GraphDataView<'float32'>;
  /** Initial bearing at the origin, degrees clockwise from north in `(-180, 180]` (turf `bearing`). */
  initialBearings?: GraphDataView<'float32'>;
  /** Bearing on arrival at the target, degrees clockwise from north in `(-180, 180]`. */
  finalBearings?: GraphDataView<'float32'>;
  /**
   * Great-circle midpoint (rhumb midpoint for `'rhumb'`), longitude continuous with the origin
   * (turf `midpoint`). Spherical in every model.
   */
  midpoints?: GraphDataView<'float32x2'>;
  /** `'wgs84'` only: `1` when Vincenty converged, `0` for near-antipodal pairs (sphere fallback). */
  converged?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUGeodesicPairs}.
 *
 * Per-frame (no recompile): the contents of `origins` and `targets`. Compile-time: the pair count,
 * `model`, `radius`, `iterations`, and which outputs are present.
 */
export type GPUGeodesicPairsProps = {
  /** Prefix for generated node IDs. Defaults to `'geodesic-pairs'`. */
  id?: string;
  /** Origins, longitude/latitude degrees. */
  origins: GraphDataView<'float32x2'>;
  /** Targets aligned with `origins`. */
  targets: GraphDataView<'float32x2'>;
  /**
   * `'sphere'` (default): haversine distance and spherical bearings on a sphere of `radius`.
   * `'wgs84'`: Vincenty's inverse solution on the WGS84 ellipsoid.
   * `'rhumb'`: constant-bearing line on a sphere of `radius` (turf `rhumbDistance`,
   * `rhumbBearing`); both bearings equal the line's bearing and `midpoints` is the rhumb midpoint.
   */
  model?: GPUGeodesicModel;
  /** Sphere radius. Defaults to {@link GPU_GEODESIC_MEAN_EARTH_RADIUS}; also the WGS84 fallback radius. */
  radius?: number;
  /** Compile-time Vincenty iteration cap for `'wgs84'`. Default 16. */
  iterations?: number;
  /** Per-pair outputs; at least one. */
  output: GPUGeodesicPairsOutput;
};

/**
 * Geodesic distance, initial and final bearing, and midpoint between many origin/target pairs
 * (turf `distance`, `bearing`, `midpoint`; PostGIS `ST_Distance(geography)`, `ST_Azimuth`).
 *
 * `'sphere'` uses the shared haversine helpers, which subtract coordinates in degrees before any
 * trigonometry, so meter-scale pairs keep their precision. `'wgs84'` runs Vincenty's inverse in f32
 * with the short-line rewrites described on `getVincentyWGSL` and a compile-time iteration cap;
 * pairs that do not converge (within about 0.5 degrees of antipodal) fall back to the sphere result
 * with `converged = 0`. `'rhumb'` evaluates the Mercator-stretched closed form in `rhumb-wgsl.ts`
 * (east-west limit, antimeridian short way, latitudes clamped at 89.9999 degrees). One invocation per pair; no reductions, so results are deterministic.
 */
export class GPUGeodesicPairs implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUGeodesicPairsProps;
  /** Resolved model. */
  readonly model: GPUGeodesicModel;
  /** Resolved radius. */
  readonly radius: number;
  /** Resolved Vincenty iteration cap. */
  readonly iterations: number;

  constructor(props: GPUGeodesicPairsProps) {
    this.id = props.id ?? 'geodesic-pairs';
    this.props = props;
    this.model = props.model ?? 'sphere';
    this.radius = props.radius ?? GPU_GEODESIC_MEAN_EARTH_RADIUS;
    this.iterations = props.iterations ?? GPU_GEODESIC_DEFAULT_ITERATIONS;
    validateGeodesicOptions(this.id, this.model, this.radius, this.iterations);
    const {id} = this;
    const {output} = props;
    for (const [name, view] of [
      ['origins', props.origins],
      ['targets', props.targets]
    ] as const) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
      validatePackedView(view, ['float32x2'], `${id} ${name}`);
    }
    const pairCount = props.origins.length;
    if (pairCount < 1 || props.targets.length !== pairCount) {
      throw new Error(`${id} needs at least one pair and targets aligned with origins`);
    }
    const formats = {
      distances: 'float32',
      initialBearings: 'float32',
      finalBearings: 'float32',
      midpoints: 'float32x2',
      converged: 'uint32'
    } as const;
    validateGeodesicColumns(id, output, formats, pairCount);
    if (output.converged && this.model !== 'wgs84') {
      throw new Error(`${id} output.converged requires model 'wgs84'`);
    }
    validateGraphOutputsDisjointFromInputs(id, Object.values(output), [
      props.origins,
      props.targets
    ]);
  }

  /** Returns the single per-pair node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.origins,
      props.targets,
      ...Object.values(output)
    ]);
    const bindings: WGSLKernelBinding[] = [
      {name: 'origins', view: props.origins, type: 'f32', access: 'read'},
      {name: 'targets', view: props.targets, type: 'f32', access: 'read'}
    ];
    const statements: string[] = [];
    const rhumb = this.model === 'rhumb';
    const add = (name: keyof GPUGeodesicPairsOutput, type: 'f32' | 'u32', statement: string) => {
      const view = output[name];
      if (view) {
        bindings.push({name, view, type, access: 'read_write'});
        statements.push(statement);
      }
    };
    add('distances', 'f32', 'distances[distancesOffset + index] = distance;');
    add(
      'initialBearings',
      'f32',
      'initialBearings[initialBearingsOffset + index] = initialBearing;'
    );
    add('finalBearings', 'f32', 'finalBearings[finalBearingsOffset + index] = finalBearing;');
    add(
      'midpoints',
      'f32',
      `let midpoint = ${
        rhumb
          ? 'rhumbMidpoint(origin, targetPosition)'
          : 'geodesicInterpolate(origin, targetPosition, geodesicCentralAngle(origin, targetPosition), 0.5)'
      };
  midpoints[midpointsOffset + 2u * index] = midpoint.x;
  midpoints[midpointsOffset + 2u * index + 1u] = midpoint.y;`
    );
    add('converged', 'u32', 'converged[convergedOffset + index] = select(0u, 1u, isConverged);');
    const wgs84 = this.model === 'wgs84';
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-pairs`,
        operation: OPERATION,
        variant: this.model,
        bindings,
        invocationCount: props.origins.length,
        declarations: `${GEODESIC_WGSL}
${wgs84 ? getVincentyWGSL(this.iterations) : ''}
${SPHERE_PAIR_WGSL}
${rhumb ? RHUMB_WGSL : ''}
const RADIUS: f32 = ${getWGSLFloatLiteral(this.radius)};`,
        body: /* wgsl */ `let origin = vec2<f32>(origins[originsOffset + 2u * index], origins[originsOffset + 2u * index + 1u]);
  let targetPosition = vec2<f32>(targets[targetsOffset + 2u * index], targets[targetsOffset + 2u * index + 1u]);
  var distance = ${
    rhumb ? 'rhumbAngle(origin, targetPosition)' : 'geodesicCentralAngle(origin, targetPosition)'
  } * RADIUS;
  var initialBearing = ${
    rhumb
      ? 'rhumbBearingDegrees(origin, targetPosition)'
      : 'geodesicInitialBearingDegrees(origin, targetPosition)'
  };
  var finalBearing = ${
    rhumb ? 'initialBearing' : 'sphereFinalBearingDegrees(origin, targetPosition)'
  };
  var isConverged = true;
  ${
    wgs84
      ? `let inverse = vincentyInverse(origin, targetPosition);
  isConverged = inverse.converged;
  if (isConverged) {
    distance = inverse.distance;
    initialBearing = inverse.initialBearing;
    finalBearing = inverse.finalBearing;
  }`
      : ''
  }
  ${statements.join('\n  ')}`
      })
    ];
  }
}

/** Validates the shared model options of the geodesic column contributors. @internal */
export function validateGeodesicOptions(
  id: string,
  model: GPUGeodesicModel,
  radius: number,
  iterations: number
): void {
  if (model !== 'sphere' && model !== 'wgs84' && model !== 'rhumb') {
    throw new Error(`${id} model must be 'sphere', 'wgs84' or 'rhumb'`);
  }
  if (!Number.isFinite(radius) || radius <= 0) {
    throw new Error(`${id} radius must be a positive finite number`);
  }
  if (!Number.isInteger(iterations) || iterations < 1 || iterations > 200) {
    throw new Error(`${id} iterations must be an integer in [1, 200]`);
  }
}

/** Validates optional per-row output columns against their formats and row count. @internal */
export function validateGeodesicColumns(
  id: string,
  output: Record<string, GraphDataView | undefined>,
  formats: Record<string, 'float32' | 'float32x2' | 'uint32'>,
  rowCount: number
): void {
  let columnCount = 0;
  for (const [name, view] of Object.entries(output)) {
    if (!view) {
      continue;
    }
    const format = formats[name];
    if (!format) {
      throw new Error(`${id} output.${name} is not a known column`);
    }
    if ((view as unknown) instanceof GraphVectorView) {
      throw new Error(`${id} output.${name} must be a single packed view, not a chunked vector`);
    }
    if (format === 'uint32') {
      validatePackedUint32View(view as GraphDataView<'uint32'>, `${id} output.${name}`);
    } else {
      validatePackedView(view, [format], `${id} output.${name}`);
    }
    if (view.length !== rowCount) {
      throw new Error(`${id} output.${name} must hold ${rowCount} rows`);
    }
    columnCount++;
  }
  if (columnCount === 0) {
    throw new Error(`${id} requires at least one output column`);
  }
}
