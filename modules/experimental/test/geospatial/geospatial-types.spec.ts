// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expectTypeOf, test} from 'vitest';
import type {GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type {
  GPUHaversineDistanceProps,
  GPUPairwisePointDistanceProps,
  GPUPairwisePointInPolygonProps,
  GPUPairwisePointLinestringNearestProps,
  GPUPairwisePointSegmentDistanceProps
} from '../../src/geospatial';

/** Compile-time coverage for the input/output format correlations of precise planar kernels. */
export function checkGeospatialDistanceTypes(
  float32Positions: GraphDataView<'float32x2'>,
  float64Positions: GraphDataView<'uint32x4'>,
  float32Distances: GraphDataView<'float32'>,
  preciseDistances: GraphDataView<'float32x2'>,
  precisePositions: GraphDataView<'float32x4'>,
  offsets: GraphDataView<'uint32'>,
  classifications: GraphDataView<'uint32'>
): void {
  ({
    left: float32Positions,
    right: float32Positions,
    output: float32Distances
  }) satisfies GPUPairwisePointDistanceProps;

  ({
    left: float32Positions,
    right: float32Positions,
    output: float32Distances
  }) satisfies GPUHaversineDistanceProps;

  ({
    left: float64Positions,
    right: float64Positions,
    output: float32Distances
  }) satisfies GPUHaversineDistanceProps;

  // @ts-expect-error haversine inputs must use the same coordinate format.
  const invalidMixedHaversineInputs: GPUHaversineDistanceProps = {
    left: float32Positions,
    right: float64Positions,
    output: float32Distances
  };

  ({
    left: float64Positions,
    right: float64Positions,
    output: preciseDistances
  }) satisfies GPUPairwisePointDistanceProps;

  // @ts-expect-error f32 point inputs require an f32 scalar output.
  const invalidFloat32PointOutput: GPUPairwisePointDistanceProps = {
    left: float32Positions,
    right: float32Positions,
    output: preciseDistances
  };

  // @ts-expect-error raw binary64 point inputs require a double-single output.
  const invalidFloat64PointOutput: GPUPairwisePointDistanceProps = {
    left: float64Positions,
    right: float64Positions,
    output: float32Distances
  };

  // @ts-expect-error pairwise point inputs must use the same coordinate format.
  const invalidMixedPointInputs: GPUPairwisePointDistanceProps = {
    left: float32Positions,
    right: float64Positions,
    output: float32Distances
  };

  ({
    positions: float32Positions,
    segmentStarts: float32Positions,
    segmentEnds: float32Positions,
    output: float32Distances
  }) satisfies GPUPairwisePointSegmentDistanceProps;

  ({
    positions: float64Positions,
    segmentStarts: float64Positions,
    segmentEnds: float64Positions,
    output: preciseDistances
  }) satisfies GPUPairwisePointSegmentDistanceProps;

  // @ts-expect-error raw binary64 segment inputs require a double-single output.
  const invalidFloat64SegmentOutput: GPUPairwisePointSegmentDistanceProps = {
    positions: float64Positions,
    segmentStarts: float64Positions,
    segmentEnds: float64Positions,
    output: float32Distances
  };

  // @ts-expect-error every point and segment input must use the same coordinate format.
  const invalidMixedSegmentInputs: GPUPairwisePointSegmentDistanceProps = {
    positions: float32Positions,
    segmentStarts: float64Positions,
    segmentEnds: float32Positions,
    output: float32Distances
  };

  ({
    queryPositions: float32Positions,
    positions: float32Positions,
    featureOffsets: offsets,
    polygonOffsets: offsets,
    ringOffsets: offsets,
    output: classifications
  }) satisfies GPUPairwisePointInPolygonProps;

  ({
    queryPositions: float64Positions,
    positions: float64Positions,
    featureOffsets: offsets,
    polygonOffsets: offsets,
    ringOffsets: offsets,
    output: classifications
  }) satisfies GPUPairwisePointInPolygonProps;

  // @ts-expect-error point and polygon storage formats must match.
  const invalidMixedPolygonInputs: GPUPairwisePointInPolygonProps = {
    queryPositions: float32Positions,
    positions: float64Positions,
    featureOffsets: offsets,
    polygonOffsets: offsets,
    ringOffsets: offsets,
    output: classifications
  };

  const invalidPolygonOutput: GPUPairwisePointInPolygonProps = {
    queryPositions: float32Positions,
    positions: float32Positions,
    featureOffsets: offsets,
    polygonOffsets: offsets,
    ringOffsets: offsets,
    // @ts-expect-error point-in-polygon classifications require a uint32 output.
    output: float32Distances
  };

  ({
    queryPositions: float32Positions,
    positions: float32Positions,
    featureOffsets: offsets,
    lineOffsets: offsets,
    output: {
      distances: float32Distances,
      nearestPositions: float32Positions,
      lineIndices: classifications,
      segmentIndices: classifications
    }
  }) satisfies GPUPairwisePointLinestringNearestProps;

  ({
    queryPositions: float64Positions,
    positions: float64Positions,
    featureOffsets: offsets,
    lineOffsets: offsets,
    output: {
      distances: preciseDistances,
      nearestPositions: precisePositions,
      lineIndices: classifications,
      segmentIndices: classifications
    }
  }) satisfies GPUPairwisePointLinestringNearestProps;

  // @ts-expect-error raw binary64 linestring inputs require double-single distance output.
  const invalidFloat64LinestringOutput: GPUPairwisePointLinestringNearestProps = {
    queryPositions: float64Positions,
    positions: float64Positions,
    featureOffsets: offsets,
    lineOffsets: offsets,
    output: {distances: float32Distances}
  };

  const invalidFloat32NearestPointOutput: GPUPairwisePointLinestringNearestProps = {
    queryPositions: float32Positions,
    positions: float32Positions,
    featureOffsets: offsets,
    lineOffsets: offsets,
    // @ts-expect-error local f32 linestring inputs require float32x2 nearest points.
    output: {distances: float32Distances, nearestPositions: precisePositions}
  };

  // @ts-expect-error point and linestring storage formats must match.
  const invalidMixedLinestringInputs: GPUPairwisePointLinestringNearestProps = {
    queryPositions: float32Positions,
    positions: float64Positions,
    featureOffsets: offsets,
    lineOffsets: offsets,
    output: {distances: float32Distances}
  };

  const invalidFloat64NearestPointOutput: GPUPairwisePointLinestringNearestProps = {
    queryPositions: float64Positions,
    positions: float64Positions,
    featureOffsets: offsets,
    lineOffsets: offsets,
    // @ts-expect-error raw binary64 inputs require float32x4 nearest-point rows.
    output: {distances: preciseDistances, nearestPositions: float32Positions}
  };

  void [
    invalidFloat32PointOutput,
    invalidFloat64PointOutput,
    invalidMixedPointInputs,
    invalidMixedHaversineInputs,
    invalidFloat64SegmentOutput,
    invalidMixedSegmentInputs,
    invalidMixedPolygonInputs,
    invalidPolygonOutput,
    invalidFloat64LinestringOutput,
    invalidFloat32NearestPointOutput,
    invalidMixedLinestringInputs,
    invalidFloat64NearestPointOutput
  ];
}

test('geospatial distance props preserve their coordinate/output format correlations', () => {
  expectTypeOf(checkGeospatialDistanceTypes).toBeFunction();
});
