// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {describe, expect, it} from 'vitest';
import {
  assertCompatibleGPUSpatialContexts,
  assertCompatibleGPUPreparedIndexes,
  defineGPUSpatialParameterSchema,
  getGPUCapacityPlan,
  getGPUCapacityRecovery,
  getGPUCellCoverContributorCapacityPlan,
  getGPUCellCapacityPlan,
  getGPUEventCapacityPlan,
  getGPUGeneratedGeometryCapacityPlan,
  getGPULineTopologyCapacityPlan,
  getGPUNeighborhoodCapacityPlan,
  getGPUPartitionMemoryPlan,
  getGPUPairCapacityPlan,
  getGPUSpatialJoinCapacityPlan,
  getGPUSpatialContextCompatibility,
  getGPUSpatialQueryCostPlan,
  getGPUTileSeamOwner,
  getGPUTrajectoryEncounterCapacityPlan,
  isGPUSeamResultOwner,
  packGPUSpatialParameterValues,
  validateGPUCompactPairPort,
  validateGPUFeatureGeometryPort,
  validateGPURecipeStatusPort,
  validateGPUStatusPort,
  validateGPUSpatialContext,
  validateGPUPartitionDescriptor,
  validateGPUTilePartitionDescriptor
} from '../../../src/gpu-spatial-analysis/contracts/index';
import {
  GPU_GEOMETRY_PREDICATES_PARAMETER_SCHEMA,
  GPU_GREAT_CIRCLE_ARCS_PARAMETER_SCHEMA,
  GPU_LINE_DENSITY_PARAMETER_SCHEMA,
  GPU_LINE_SEGMENTIZE_PARAMETER_SCHEMA,
  GPU_NEIGHBOR_SEARCH_PARAMETER_SCHEMA,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_SCHEMA,
  GPU_TRAJECTORY_METRICS_PARAMETER_SCHEMA
} from '../../../src/gpu-spatial-analysis/index';

describe('GPU spatial-analysis contracts', () => {
  it('validates feature geometry, compact pairs and stage-attributed statuses', () => {
    const positions = view('float32x2', 4);
    const lineOffsets = view('uint32', 3);
    const sourceIds = view('uint32', 2);
    expect(() =>
      validateGPUFeatureGeometryPort('lines', {
        kind: 'lines',
        positions,
        lineOffsets,
        sourceIds
      })
    ).not.toThrow();
    expect(() =>
      validateGPUFeatureGeometryPort('bad-lines', {
        kind: 'lines',
        positions,
        lineOffsets,
        sourceIds: view('uint32', 3)
      })
    ).toThrow(/sourceIds length/);

    const count = view('uint32', 1);
    const overflow = view('uint32', 1);
    expect(() =>
      validateGPUCompactPairPort('pairs', {
        leftIds: view('uint32', 8),
        rightIds: view('uint32', 8),
        count,
        requiredCount: count,
        overflow
      })
    ).not.toThrow();
    expect(() => validateGPUStatusPort('status', {count, overflow})).not.toThrow();
    expect(() =>
      validateGPURecipeStatusPort('recipe', {
        stages: [
          {stage: 'join', status: {count, overflow}},
          {stage: 'gather', status: {count, overflow}}
        ]
      })
    ).not.toThrow();
    expect(() =>
      validateGPURecipeStatusPort('recipe', {
        stages: [
          {stage: 'join', status: {count}},
          {stage: 'join', status: {overflow}}
        ]
      })
    ).toThrow(/unique/);
  });

  it('keeps coordinate space, metric parameters and CRS transformation separate', () => {
    const sphere = {
      coordinateSpace: 'longitude-latitude',
      metric: 'great-circle',
      units: 'meters',
      sphereRadius: 6371008.8
    } as const;
    expect(() => validateGPUSpatialContext('sphere', sphere)).not.toThrow();
    expect(getGPUSpatialContextCompatibility(sphere, {...sphere, sphereRadius: undefined})).toEqual(
      {compatible: true}
    );
    expect(getGPUSpatialContextCompatibility(sphere, {...sphere, metric: 'rhumb'})).toEqual({
      compatible: false,
      reason: 'metrics differ (great-circle and rhumb)'
    });
    expect(() =>
      assertCompatibleGPUSpatialContexts('workflow', sphere, {
        coordinateSpace: 'planar',
        metric: 'native',
        units: 'native'
      })
    ).toThrow(/coordinate spaces differ/);
    expect(() =>
      validateGPUSpatialContext('bad-grid', {
        coordinateSpace: 'discrete-grid',
        metric: 'none'
      })
    ).toThrow(/gridFamily/);
  });

  it('validates and packs reusable parameter schemas', () => {
    const schema = defineGPUSpatialParameterSchema({
      id: 'line-segmentize',
      format: 'float32',
      wordLength: 4,
      fields: [
        {
          name: 'maximumSegmentLength',
          format: 'float32',
          wordOffset: 0,
          defaultValue: 1,
          minimum: 0,
          units: 'meters',
          dynamic: true
        }
      ]
    });
    const packed = packGPUSpatialParameterValues(schema, {maximumSegmentLength: 4.5});
    expect(packed).toBeInstanceOf(Float32Array);
    expect(Array.from(packed)).toEqual([4.5, 0, 0, 0]);
    expect(() => packGPUSpatialParameterValues(schema, {unknown: 1})).toThrow(/no parameter/);
    expect(() => packGPUSpatialParameterValues(schema, {maximumSegmentLength: -1})).toThrow(
      /minimum/
    );
  });

  it('publishes schemas from unrelated parameter families', () => {
    const schemas = [
      GPU_GEOMETRY_PREDICATES_PARAMETER_SCHEMA,
      GPU_NEIGHBOR_SEARCH_PARAMETER_SCHEMA,
      GPU_LINE_SEGMENTIZE_PARAMETER_SCHEMA,
      GPU_GREAT_CIRCLE_ARCS_PARAMETER_SCHEMA,
      GPU_SPATIAL_AUTOCORRELATION_PARAMETER_SCHEMA,
      GPU_LINE_DENSITY_PARAMETER_SCHEMA,
      GPU_TRAJECTORY_METRICS_PARAMETER_SCHEMA
    ];
    expect(new Set(schemas.map(({id}) => id)).size).toBe(schemas.length);
    for (const schema of schemas) {
      expect(() => packGPUSpatialParameterValues(schema)).not.toThrow();
      expect(schema.fields.every(field => field.dynamic)).toBe(true);
    }
  });

  it('plans capacity and applies explicit recovery policies', () => {
    expect(
      getGPUCapacityPlan({
        sourceCount: 10,
        expansionFactor: 2.5,
        maximumCapacity: 20,
        outputBytesPerRow: 8,
        transientBytesPerRow: 12
      })
    ).toEqual({
      capacity: 20,
      estimatedRequiredCount: 25,
      limited: true,
      estimatedOutputBytes: 160,
      estimatedPeakTransientBytes: 240
    });
    expect(
      getGPUCapacityRecovery(
        {kind: 'grow-on-next-frame'},
        {output: {capacity: 20, count: 20, requiredCount: 37, overflow: true}}
      )
    ).toEqual({
      action: 'grow',
      complete: false,
      incompleteStages: ['output'],
      nextOutputCapacity: 40
    });
    expect(
      getGPUCapacityRecovery(
        {kind: 'fail-closed'},
        {
          output: {capacity: 20, count: 20, overflow: false},
          candidate: {capacity: 30, requiredCount: 48, overflow: true}
        }
      )
    ).toEqual({action: 'reject', complete: false, incompleteStages: ['candidate']});
    expect(
      getGPUCapacityRecovery(
        {kind: 'fixed-budget'},
        {output: {capacity: 20, count: 10, overflow: false}}
      )
    ).toEqual({action: 'accept', complete: true});
    expect(getGPUPairCapacityPlan({sourceCount: 2}).estimatedOutputBytes).toBe(16);
    expect(getGPUNeighborhoodCapacityPlan({sourceCount: 2}).estimatedOutputBytes).toBe(16);
    expect(getGPUGeneratedGeometryCapacityPlan({sourceCount: 2}).estimatedOutputBytes).toBe(24);
    expect(getGPUCellCapacityPlan({sourceCount: 2}).estimatedOutputBytes).toBe(32);
    expect(getGPUEventCapacityPlan({sourceCount: 2}).estimatedOutputBytes).toBe(40);
  });

  it('plans measured join, cell, topology and encounter contributors by independent stages', () => {
    const join = getGPUSpatialJoinCapacityPlan({
      leftCount: 100,
      rightCount: 50,
      candidateCapacity: 300,
      pairCapacity: 120,
      observedCandidateCount: 275,
      observedRequiredCount: 80
    });
    expect(join.candidates).toMatchObject({capacity: 300, estimatedRequiredCount: 275});
    expect(join.pairs).toMatchObject({capacity: 120, estimatedRequiredCount: 80});
    expect(join.estimatedWorkItems).toBe(425);
    expect(join.estimatedPeakTransientBytes).toBeGreaterThan(join.estimatedOutputBytes);

    const cover = getGPUCellCoverContributorCapacityPlan({
      featureCount: 4,
      vertexCount: 80,
      candidateCapacity: 200,
      cellCapacity: 100,
      observedCandidateCount: 180,
      observedRequiredCount: 90,
      edgeSlabs: true,
      includeCoreColumn: true
    });
    expect(cover.candidates.estimatedRequiredCount).toBe(180);
    expect(cover.cells.estimatedOutputBytes).toBe(1600);
    expect(cover.estimatedEdgeTests).toBe(540);

    const topology = getGPULineTopologyCapacityPlan({
      segmentCount: 20,
      lineCount: 5,
      intersectionCapacity: 50,
      pieceCapacity: 30,
      vertexCapacity: 90,
      observedIntersectionCount: 42,
      observedPieceCount: 22,
      observedVertexCount: 65
    });
    expect(topology.intersections.estimatedRequiredCount).toBe(42);
    expect(topology.pieces.estimatedRequiredCount).toBe(22);
    expect(topology.vertices.estimatedRequiredCount).toBe(65);

    const encounters = getGPUTrajectoryEncounterCapacityPlan({
      trackCount: 10,
      bucketCount: 12,
      latticeCellCount: 48,
      hitCapacity: 250,
      pairCapacity: 40,
      observedHitCount: 220,
      observedRequiredCount: 35
    });
    expect(encounters.sampleCount).toBe(120);
    expect(encounters.hits.estimatedRequiredCount).toBe(220);
    expect(encounters.pairs.estimatedOutputBytes).toBe(960);
  });

  it('grows candidate and final stages independently', () => {
    expect(
      getGPUCapacityRecovery(
        {kind: 'grow-on-next-frame', growthFactor: 1.5},
        {
          output: {capacity: 10, count: 10, requiredCount: 12, overflow: true},
          candidate: {capacity: 20, requiredCount: 41, overflow: true}
        }
      )
    ).toEqual({
      action: 'grow',
      complete: false,
      incompleteStages: ['candidate', 'output'],
      nextOutputCapacity: 15,
      nextCandidateCapacity: 41
    });
    expect(() =>
      getGPUCapacityRecovery(
        {kind: 'fixed-budget'},
        {output: {capacity: 10, count: 10, requiredCount: 11, overflow: false}}
      )
    ).toThrow(/requires overflow/);
  });

  it('preserves partitions, assigns seams and plans query paths without readback', () => {
    const partitioning = {
      length: 10,
      partitions: [
        {id: 7, coreStart: 0, coreEnd: 5, haloStart: 0, haloEnd: 7},
        {id: 3, coreStart: 5, coreEnd: 10, haloStart: 3, haloEnd: 10}
      ],
      seamOwnership: {kind: 'lowest-partition-id'}
    } as const;
    expect(() => validateGPUPartitionDescriptor('tiles', partitioning)).not.toThrow();
    expect(isGPUSeamResultOwner(partitioning, 3, 7, 3)).toBe(true);
    expect(isGPUSeamResultOwner(partitioning, 7, 7, 3)).toBe(false);
    expect(getGPUPartitionMemoryPlan(partitioning, 16)).toEqual({
      packedRows: 10,
      peakPartitionRows: 7,
      packedBytes: 160,
      peakPartitionBytes: 112,
      savedPeakBytes: 48
    });
    expect(() =>
      validateGPUPartitionDescriptor('gap', {
        ...partitioning,
        partitions: [{id: 0, coreStart: 1, coreEnd: 10, haloStart: 0, haloEnd: 10}]
      })
    ).toThrow(/contiguous coverage/);

    expect(
      getGPUSpatialQueryCostPlan({
        sourceCount: 100,
        targetCount: 100,
        overlapRatio: 0.01,
        populatedCellCount: 10,
        preparedBVHDepth: 8
      })
    ).toMatchObject({strategy: 'grid', estimatedComparisons: 310});

    const prepared = {
      spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'native'},
      partitioning,
      revision: 4
    } as const;
    expect(() => assertCompatibleGPUPreparedIndexes('join', prepared, prepared)).not.toThrow();
    expect(() =>
      assertCompatibleGPUPreparedIndexes('join', prepared, {...prepared, revision: 5})
    ).toThrow(/revision/);

    const tiles = {
      tiles: [
        {id: 9, bounds: [0, 0, 1, 1], haloBounds: [-0.1, -0.1, 1.1, 1.1]},
        {id: 4, bounds: [1, 0, 2, 1], haloBounds: [0.9, -0.1, 2.1, 1.1]}
      ],
      seamOwnership: 'lowest-tile-id'
    } as const;
    expect(() => validateGPUTilePartitionDescriptor('map', tiles)).not.toThrow();
    expect(getGPUTileSeamOwner(tiles, [9, 4])).toBe(4);
  });
});

function view<Format extends 'float32x2' | 'uint32'>(
  format: Format,
  length: number
): GraphDataView<Format> {
  return {format, length} as GraphDataView<Format>;
}
