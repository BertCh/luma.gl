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
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {createSpatialJoinBoundsNode, SPATIAL_JOIN_WGSL_HELPERS} from './spatial-join-passes';
import type {GPUSpatialJoinGeometry} from './spatial-join-types';

const OPERATION = 'GPUSpatialPredicateJoin';

/** Returns the number of features in a predicate-join geometry. @internal */
export function getSpatialJoinFeatureCount(geometry: GPUSpatialJoinGeometry): number {
  switch (geometry.kind) {
    case 'points':
      return geometry.positions.length;
    case 'lines':
      return geometry.lineOffsets.length - 1;
    case 'polygons':
      return geometry.featureOffsets.length - 1;
  }
}

/** Returns every input view of a predicate-join geometry. @internal */
export function getSpatialJoinGeometryViews(geometry: GPUSpatialJoinGeometry): GraphDataView[] {
  if (geometry.kind === 'lines') {
    return [geometry.positions, geometry.lineOffsets];
  }
  if (geometry.kind === 'polygons') {
    return [
      geometry.positions,
      geometry.featureOffsets,
      geometry.polygonOffsets,
      geometry.ringOffsets
    ];
  }
  return [geometry.positions];
}

/** Throws unless the geometry views are packed and the geometry has at least one feature. @internal */
export function validateSpatialJoinGeometry(
  id: string,
  name: string,
  geometry: GPUSpatialJoinGeometry
): void {
  validatePackedView(geometry.positions, ['float32x2'], `${id} ${name}.positions`);
  const offsetViews =
    geometry.kind === 'lines'
      ? [['lineOffsets', geometry.lineOffsets] as const]
      : geometry.kind === 'polygons'
        ? [
            ['featureOffsets', geometry.featureOffsets] as const,
            ['polygonOffsets', geometry.polygonOffsets] as const,
            ['ringOffsets', geometry.ringOffsets] as const
          ]
        : [];
  for (const [offsetName, view] of offsetViews) {
    validatePackedUint32View(view, `${id} ${name}.${offsetName}`);
    if (view.length < 1) {
      throw new Error(`${id} ${name}.${offsetName} requires a terminal entry`);
    }
  }
  if (getSpatialJoinFeatureCount(geometry) < 1) {
    throw new Error(`${id} ${name} must contain at least one feature`);
  }
}

/** Returns whether two geometries are the same kind over the same views. @internal */
export function isSameSpatialJoinGeometry(
  first: GPUSpatialJoinGeometry,
  second: GPUSpatialJoinGeometry
): boolean {
  if (first.kind !== second.kind) {
    return false;
  }
  const firstViews = getSpatialJoinGeometryViews(first);
  const secondViews = getSpatialJoinGeometryViews(second);
  return firstViews.every((view, index) => view === secondViews[index]);
}

/** Writes `(ringStart, ringEnd)` per polygon feature; malformed offsets give an empty range. */
export function createFeatureRingsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: {
    featureCount: number;
    geometry: Extract<GPUSpatialJoinGeometry, {kind: 'polygons'}>;
    featureRings: GraphDataView<'uint32x2'>;
  }
): GPUCommandNode<Parameters> {
  const {geometry} = props;
  return createWGSLKernelNode<Parameters>(graph, {
    id,
    operation: OPERATION,
    variant: 'feature-rings',
    bindings: [
      {name: 'featureOffsets', view: geometry.featureOffsets, type: 'u32', access: 'read'},
      {name: 'polygonOffsets', view: geometry.polygonOffsets, type: 'u32', access: 'read'},
      {name: 'featureRings', view: props.featureRings, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.featureCount,
    declarations: `const POLYGON_COUNT: u32 = ${geometry.polygonOffsets.length - 1}u;
const RING_COUNT: u32 = ${geometry.ringOffsets.length - 1}u;`,
    body: `let polygonStart = featureOffsets[featureOffsetsOffset + index];
  let polygonEnd = featureOffsets[featureOffsetsOffset + index + 1u];
  var ringStart = 0u;
  var ringEnd = 0u;
  if (polygonStart <= polygonEnd && polygonEnd <= POLYGON_COUNT) {
    let first = polygonOffsets[polygonOffsetsOffset + polygonStart];
    let last = polygonOffsets[polygonOffsetsOffset + polygonEnd];
    if (first <= last && last <= RING_COUNT) { ringStart = first; ringEnd = last; }
  }
  featureRings[featureRingsOffset + index * 2u] = ringStart;
  featureRings[featureRingsOffset + index * 2u + 1u] = ringEnd;`
  });
}

/** Writes per-feature bounds for any geometry kind; empty features get inverted bounds. */
export function createBoundsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: {
    featureCount: number;
    geometry: GPUSpatialJoinGeometry;
    featureRings?: GraphDataView<'uint32x2'>;
    minima: GraphDataView<'float32x2'>;
    maxima: GraphDataView<'float32x2'>;
  }
): GPUCommandNode<Parameters> {
  const {geometry} = props;
  if (geometry.kind === 'points') {
    return createSpatialJoinBoundsNode<Parameters>(graph, {
      id,
      operation: OPERATION,
      featureCount: props.featureCount,
      source: {kind: 'points', positions: geometry.positions},
      minima: props.minima,
      maxima: props.maxima
    });
  }
  const bindings: WGSLKernelBinding[] = [
    {name: 'positions', view: geometry.positions, type: 'f32', access: 'read'}
  ];
  let range: string;
  if (geometry.kind === 'lines') {
    bindings.push({name: 'lineOffsets', view: geometry.lineOffsets, type: 'u32', access: 'read'});
    range = `let vertexStart = lineOffsets[lineOffsetsOffset + index];
  let vertexEnd = min(lineOffsets[lineOffsetsOffset + index + 1u], VERTEX_COUNT);`;
  } else {
    bindings.push(
      {
        name: 'featureRings',
        view: props.featureRings as GraphDataView,
        type: 'u32',
        access: 'read'
      },
      {name: 'ringOffsets', view: geometry.ringOffsets, type: 'u32', access: 'read'}
    );
    range = `let ringStart = featureRings[featureRingsOffset + index * 2u];
  let ringEnd = featureRings[featureRingsOffset + index * 2u + 1u];
  var vertexStart = 0u;
  var vertexEnd = 0u;
  if (ringStart < ringEnd) {
    vertexStart = ringOffsets[ringOffsetsOffset + ringStart];
    vertexEnd = min(ringOffsets[ringOffsetsOffset + ringEnd], VERTEX_COUNT);
  }`;
  }
  bindings.push(
    {name: 'featureMinima', view: props.minima, type: 'f32', access: 'read_write'},
    {name: 'featureMaxima', view: props.maxima, type: 'f32', access: 'read_write'}
  );
  return createWGSLKernelNode<Parameters>(graph, {
    id,
    operation: OPERATION,
    variant: `bounds-${geometry.kind}`,
    bindings,
    invocationCount: props.featureCount,
    declarations: `${SPATIAL_JOIN_WGSL_HELPERS}
const VERTEX_COUNT: u32 = ${geometry.positions.length}u;`,
    body: `${range}
  var minimum = vec2f(FLOAT32_MAXIMUM);
  var maximum = vec2f(-FLOAT32_MAXIMUM);
  for (var vertex = vertexStart; vertex < vertexEnd; vertex++) {
    let position = vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
    if (isFiniteValue(position.x) && isFiniteValue(position.y)) {
      minimum = min(minimum, position);
      maximum = max(maximum, position);
    }
  }
  featureMinima[featureMinimaOffset + index * 2u] = minimum.x;
  featureMinima[featureMinimaOffset + index * 2u + 1u] = minimum.y;
  featureMaxima[featureMaximaOffset + index * 2u] = maximum.x;
  featureMaxima[featureMaximaOffset + index * 2u + 1u] = maximum.y;`
  });
}
