// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {GPUGeometryHoleRule} from '../geometry-measures/index';

/** Number of float32 elements in a `GPUShapeDescriptors` parameter buffer. */
export const GPU_SHAPE_DESCRIPTORS_PARAMETER_LENGTH = 4;

/** Inputs shared by the kernels that walk feature rings. @internal */
type FeatureRingInputs = {
  id: string;
  operation: string;
  positions: GraphDataView<'float32x2'>;
  ringOffsets: GraphDataView<'uint32'>;
  featureRingOffsets?: GraphDataView<'uint32'>;
  featureCount: number;
};

function getRingBindings(props: FeatureRingInputs): WGSLKernelBinding[] {
  const bindings: WGSLKernelBinding[] = [
    {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
    {name: 'ringOffsets', view: props.ringOffsets, type: 'u32', access: 'read'}
  ];
  if (props.featureRingOffsets) {
    bindings.push({
      name: 'featureRingOffsets',
      view: props.featureRingOffsets,
      type: 'u32',
      access: 'read'
    });
  }
  return bindings;
}

function getRingDeclarations(props: FeatureRingInputs): string {
  return /* wgsl */ `
const ROW_COUNT: u32 = ${props.positions.length}u;
const RING_COUNT: u32 = ${props.ringOffsets.length - 1}u;
fn getNan(seed: u32) -> f32 {
  // A runtime operand keeps the compiler from folding a constant NaN.
  return bitcast<f32>(0x7fc00000u | (seed & 0u));
}

fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}

fn getRingStart(feature: u32) -> u32 {
  ${props.featureRingOffsets ? 'return min(featureRingOffsets[featureRingOffsetsOffset + feature], RING_COUNT);' : 'return feature;'}
}

fn getRingEnd(feature: u32) -> u32 {
  ${props.featureRingOffsets ? 'return min(featureRingOffsets[featureRingOffsetsOffset + feature + 1u], RING_COUNT);' : 'return feature + 1u;'}
}

fn getRingRowStart(ring: u32) -> u32 {
  return min(ringOffsets[ringOffsetsOffset + ring], ROW_COUNT);
}

fn getRingRowEnd(ring: u32) -> u32 {
  return min(ringOffsets[ringOffsetsOffset + ring + 1u], ROW_COUNT);
}
`;
}

/**
 * Per-feature central second moments about the measured centroid, and the signed area of the
 * first ring. One invocation per feature; Neumaier-compensated sums in row order, deterministic.
 *
 * Output `moments` rows are `[integral of x^2, integral of y^2, integral of xy, first ring signed
 * area]` with coordinates relative to the centroid.
 *
 * @internal
 */
export function createMomentsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: FeatureRingInputs & {
    holeRule: GPUGeometryHoleRule;
    centroids: GraphDataView<'float32x2'>;
    moments: GraphDataView<'float32x4'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'moments',
    bindings: [
      ...getRingBindings(props),
      {name: 'centroids', view: props.centroids, type: 'f32', access: 'read'},
      {name: 'moments', view: props.moments, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.featureCount,
    declarations: `${getRingDeclarations(props)}
const FIRST_RING_EXTERIOR: bool = ${props.holeRule === 'first-ring-exterior'};

// accumulator = (sum, compensation).
fn addNeumaier(accumulator: ptr<function, vec2<f32>>, value: f32) {
  let sum = (*accumulator).x;
  let total = sum + value;
  var compensation = (*accumulator).y;
  if (abs(sum) >= abs(value)) {
    compensation += (sum - total) + value;
  } else {
    compensation += (value - total) + sum;
  }
  *accumulator = vec2<f32>(total, compensation);
}`,
    body: /* wgsl */ `let ringStart = getRingStart(index);
  let ringEnd = max(getRingEnd(index), ringStart);
  let centroid = vec2<f32>(centroids[centroidsOffset + 2u * index], centroids[centroidsOffset + 2u * index + 1u]);
  var totalXX = vec2<f32>(0.0);
  var totalYY = vec2<f32>(0.0);
  var totalXY = vec2<f32>(0.0);
  var totalTwiceArea = 0.0;
  var firstArea = 0.0;
  if (centroid.x == centroid.x && centroid.y == centroid.y) {
    for (var ring = ringStart; ring < ringEnd; ring++) {
      let rowStart = getRingRowStart(ring);
      let rowEnd = getRingRowEnd(ring);
      var twiceArea = vec2<f32>(0.0);
      var ringXX = vec2<f32>(0.0);
      var ringYY = vec2<f32>(0.0);
      var ringXY = vec2<f32>(0.0);
      if (rowEnd > rowStart + 2u) {
        var previous = getPosition(rowEnd - 1u) - centroid;
        for (var row = rowStart; row < rowEnd; row++) {
          let current = getPosition(row) - centroid;
          let cross = previous.x * current.y - current.x * previous.y;
          addNeumaier(&twiceArea, cross);
          addNeumaier(&ringXX, cross * (previous.x * previous.x + previous.x * current.x + current.x * current.x));
          addNeumaier(&ringYY, cross * (previous.y * previous.y + previous.y * current.y + current.y * current.y));
          addNeumaier(&ringXY, cross * (previous.x * current.y + 2.0 * previous.x * previous.y + 2.0 * current.x * current.y + current.x * previous.y));
          previous = current;
        }
      }
      let ringTwiceArea = twiceArea.x + twiceArea.y;
      if (ring == ringStart) {
        firstArea = 0.5 * ringTwiceArea;
      }
      var factor = 1.0;
      if (FIRST_RING_EXTERIOR) {
        factor = select(-1.0, 1.0, ring == ringStart) * sign(ringTwiceArea);
      }
      totalTwiceArea += factor * ringTwiceArea;
      addNeumaier(&totalXX, factor * (ringXX.x + ringXX.y) / 12.0);
      addNeumaier(&totalYY, factor * (ringYY.x + ringYY.y) / 12.0);
      addNeumaier(&totalXY, factor * (ringXY.x + ringXY.y) / 24.0);
    }
  }
  // A clockwise exterior under the winding rule integrates to negative moments; flip them so the
  // moments always describe the covered region.
  let orientationSign = select(1.0, -1.0, totalTwiceArea < 0.0);
  moments[momentsOffset + 4u * index] = orientationSign * (totalXX.x + totalXX.y);
  moments[momentsOffset + 4u * index + 1u] = orientationSign * (totalYY.x + totalYY.y);
  moments[momentsOffset + 4u * index + 2u] = orientationSign * (totalXY.x + totalXY.y);
  moments[momentsOffset + 4u * index + 3u] = firstArea;`
  });
}

/**
 * Per-feature convexity: area divided by the area of the convex hull of all feature vertices.
 * The hull comes from a gift-wrapping march with O(1) memory, so cost is O(vertices * hull
 * vertices) per feature. One invocation per feature, deterministic.
 *
 * @internal
 */
export function createConvexityNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: FeatureRingInputs & {
    areas: GraphDataView<'float32'>;
    convexities: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'convexity',
    bindings: [
      ...getRingBindings(props),
      {name: 'areas', view: props.areas, type: 'f32', access: 'read'},
      {name: 'convexities', view: props.convexities, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.featureCount,
    declarations: getRingDeclarations(props),
    body: /* wgsl */ `let ringStart = getRingStart(index);
  let ringEnd = max(getRingEnd(index), ringStart);
  let rowStart = getRingRowStart(ringStart);
  let rowEnd = max(getRingRowStart(ringEnd), rowStart);
  let area = areas[areasOffset + index];
  var hullArea = 0.0;
  if (rowEnd - rowStart >= 3u && area > 0.0) {
    let origin = getPosition(rowStart);
    // Lowest point, leftmost among ties, lowest row among identical points.
    var startPoint = vec2<f32>(0.0);
    for (var row = rowStart; row < rowEnd; row++) {
      let p = getPosition(row) - origin;
      if (row == rowStart || p.y < startPoint.y || (p.y == startPoint.y && p.x < startPoint.x)) {
        startPoint = p;
      }
    }
    var current = startPoint;
    var twiceArea = 0.0;
    for (var step = 0u; step < rowEnd - rowStart; step++) {
      var best = current;
      var hasBest = false;
      for (var row = rowStart; row < rowEnd; row++) {
        let q = getPosition(row) - origin;
        if (q.x == current.x && q.y == current.y) {
          continue;
        }
        if (!hasBest) {
          best = q;
          hasBest = true;
          continue;
        }
        let toBest = best - current;
        let toQ = q - current;
        let turn = toBest.x * toQ.y - toBest.y * toQ.x;
        // q lies clockwise of best (all others must be left of the hull edge), or farther on a tie.
        if (turn < 0.0 || (turn == 0.0 && dot(toQ, toQ) > dot(toBest, toBest))) {
          best = q;
        }
      }
      if (!hasBest) {
        break;
      }
      twiceArea += current.x * best.y - best.x * current.y;
      current = best;
      if (current.x == startPoint.x && current.y == startPoint.y) {
        break;
      }
    }
    hullArea = 0.5 * twiceArea;
  }
  convexities[convexitiesOffset + index] = select(getNan(index), min(area / hullArea, 1.0), hullArea > 0.0);`
  });
}

/**
 * Per-feature Polsby-Popper and Schwartzberg compactness plus the sliver flag.
 *
 * @internal
 */
export function createCompactnessNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    featureCount: number;
    areas: GraphDataView<'float32'>;
    perimeters: GraphDataView<'float32'>;
    parameters: GraphDataView<'float32'>;
    polsbyPopper?: GraphDataView<'float32'>;
    schwartzberg?: GraphDataView<'float32'>;
    sliver?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'areas', view: props.areas, type: 'f32', access: 'read'},
    {name: 'perimeters', view: props.perimeters, type: 'f32', access: 'read'},
    {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'}
  ];
  if (props.polsbyPopper) {
    bindings.push({
      name: 'polsbyPopper',
      view: props.polsbyPopper,
      type: 'f32',
      access: 'read_write'
    });
  }
  if (props.schwartzberg) {
    bindings.push({
      name: 'schwartzberg',
      view: props.schwartzberg,
      type: 'f32',
      access: 'read_write'
    });
  }
  if (props.sliver) {
    bindings.push({name: 'sliver', view: props.sliver, type: 'u32', access: 'read_write'});
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'compactness',
    bindings,
    invocationCount: props.featureCount,
    declarations: `fn getNan(seed: u32) -> f32 {
  // A runtime operand keeps the compiler from folding a constant NaN.
  return bitcast<f32>(0x7fc00000u | (seed & 0u));
}`,
    body: /* wgsl */ `let area = areas[areasOffset + index];
  let perimeter = perimeters[perimetersOffset + index];
  let isValid = area > 0.0 && perimeter > 0.0;
  // Divide by the perimeter twice so a 1e6 m perimeter does not square into f32 trouble.
  let ratio = 4.0 * 3.14159265358979 * area / perimeter / perimeter;
  ${props.polsbyPopper ? 'polsbyPopper[polsbyPopperOffset + index] = select(getNan(index), ratio, isValid);' : ''}
  ${props.schwartzberg ? 'schwartzberg[schwartzbergOffset + index] = select(getNan(index), sqrt(ratio), isValid);' : ''}
  ${props.sliver ? 'sliver[sliverOffset + index] = select(0u, 1u, perimeter > 0.0 && (area <= 0.0 || ratio < parameters[parametersOffset]));' : ''}`
  });
}

/**
 * Per-feature elongation, orientation and clockwise flag from the central moments.
 *
 * @internal
 */
export function createOrientationNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    featureCount: number;
    moments: GraphDataView<'float32x4'>;
    elongation?: GraphDataView<'float32'>;
    orientation?: GraphDataView<'float32'>;
    clockwise?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'moments', view: props.moments, type: 'f32', access: 'read'}
  ];
  if (props.elongation) {
    bindings.push({name: 'elongation', view: props.elongation, type: 'f32', access: 'read_write'});
  }
  if (props.orientation) {
    bindings.push({
      name: 'orientation',
      view: props.orientation,
      type: 'f32',
      access: 'read_write'
    });
  }
  if (props.clockwise) {
    bindings.push({name: 'clockwise', view: props.clockwise, type: 'u32', access: 'read_write'});
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'orientation',
    bindings,
    invocationCount: props.featureCount,
    declarations: `fn getNan(seed: u32) -> f32 {
  // A runtime operand keeps the compiler from folding a constant NaN.
  return bitcast<f32>(0x7fc00000u | (seed & 0u));
}`,
    body: /* wgsl */ `let xx = moments[momentsOffset + 4u * index];
  let yy = moments[momentsOffset + 4u * index + 1u];
  let xy = moments[momentsOffset + 4u * index + 2u];
  let firstArea = moments[momentsOffset + 4u * index + 3u];
  let mean = 0.5 * (xx + yy);
  let radius = sqrt(0.25 * (xx - yy) * (xx - yy) + xy * xy);
  let major = mean + radius;
  let minor = max(mean - radius, 0.0);
  let isValid = major > 0.0;
  ${props.elongation ? 'elongation[elongationOffset + index] = select(getNan(index), 1.0 - sqrt(minor / major), isValid);' : ''}
  ${props.orientation ? 'orientation[orientationOffset + index] = select(getNan(index), 0.5 * atan2(2.0 * xy, xx - yy), isValid);' : ''}
  ${props.clockwise ? 'clockwise[clockwiseOffset + index] = select(0u, 1u, firstArea < 0.0);' : ''}`
  });
}
