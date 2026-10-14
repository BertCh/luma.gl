// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {describe, expect, it} from 'vitest';
import {
  GPU_SPATIAL_ANALYSIS_CONTRACT_NAMES,
  assertCompatibleGPUPreparedIndexes,
  defineGPUSpatialParameterSchema,
  getGPUCapacityPlan,
  getGPUCapacityRecovery,
  getGPUCellCoverContributorCapacityPlan,
  getGPUPartitionMemoryPlan,
  getGPUSpatialContextCompatibility,
  getGPUSpatialJoinCapacityPlan,
  getGPUSpatialQueryCostPlan,
  getGPUTileSeamOwner,
  isGPUSeamResultOwner,
  packGPUSpatialParameterValues,
  validateGPUCompactPairPort,
  validateGPUFeatureGeometryPort,
  validateGPUPartitionDescriptor,
  validateGPURecipeStatusPort,
  validateGPUSpatialContext,
  validateGPUStatusPort,
  validateGPUTilePartitionDescriptor
} from '../../../src/gpu-spatial-analysis/contracts/index';
import {validateGPUSpatialParameterSchema} from '../../../src/gpu-spatial-analysis/contracts/parameter-schema';

function view<Format extends string>(format: Format, length: number): GraphDataView<never> {
  return {format, length} as unknown as GraphDataView<never>;
}

describe('parameter schemas', () => {
  const field = (overrides = {}) => ({
    name: 'a',
    format: 'float32' as const,
    wordOffset: 0,
    defaultValue: 1,
    dynamic: true,
    ...overrides
  });
  const schema = (fields: object[], overrides = {}): never =>
    ({
      id: 'test-schema',
      format: 'float32' as const,
      wordLength: 4,
      fields,
      ...overrides
    }) as never;

  it('rejects malformed layouts', () => {
    expect(() => validateGPUSpatialParameterSchema(schema([], {id: ''}))).toThrow(/ID/);
    expect(() => validateGPUSpatialParameterSchema(schema([], {wordLength: 0}))).toThrow(/ID/);
    expect(() => validateGPUSpatialParameterSchema(schema([], {wordLength: 1.5}))).toThrow(/ID/);
    expect(() => validateGPUSpatialParameterSchema(schema([field(), field()]))).toThrow(/unique/);
    expect(() => validateGPUSpatialParameterSchema(schema([field({name: ''})]))).toThrow(/unique/);
    expect(() => validateGPUSpatialParameterSchema(schema([field(), field({name: 'b'})]))).toThrow(
      /wordOffset/
    );
    expect(() => validateGPUSpatialParameterSchema(schema([field({wordOffset: 4})]))).toThrow(
      /wordOffset/
    );
    expect(() => validateGPUSpatialParameterSchema(schema([field({wordOffset: -1})]))).toThrow(
      /wordOffset/
    );
    expect(() => validateGPUSpatialParameterSchema(schema([field({format: 'uint32'})]))).toThrow(
      /format/
    );
    expect(() =>
      validateGPUSpatialParameterSchema(schema([field({minimum: 3, maximum: 2, defaultValue: 2})]))
    ).toThrow(/minimum exceeds maximum/);
  });

  it('checks defaults against ranges and formats', () => {
    expect(() => validateGPUSpatialParameterSchema(schema([field({defaultValue: NaN})]))).toThrow(
      /finite/
    );
    expect(() =>
      validateGPUSpatialParameterSchema(schema([field({defaultValue: Infinity})]))
    ).toThrow(/finite/);
    expect(() =>
      validateGPUSpatialParameterSchema(
        schema([field({defaultValue: Infinity, allowNonFinite: true})])
      )
    ).not.toThrow();
    expect(() =>
      validateGPUSpatialParameterSchema(schema([field({defaultValue: 5, maximum: 4})]))
    ).toThrow(/maximum/);
    expect(() =>
      validateGPUSpatialParameterSchema(schema([field({defaultValue: -1, minimum: 0})]))
    ).toThrow(/minimum/);
  });

  it('packs uint32 and sint32 schemas into matching typed arrays and enforces integer ranges', () => {
    const unsigned = defineGPUSpatialParameterSchema({
      id: 'unsigned',
      format: 'uint32',
      wordLength: 2,
      fields: [{name: 'count', format: 'uint32', wordOffset: 1, defaultValue: 7, dynamic: false}]
    });
    const packedUnsigned = packGPUSpatialParameterValues(unsigned);
    expect(packedUnsigned).toBeInstanceOf(Uint32Array);
    expect(Array.from(packedUnsigned)).toEqual([0, 7]);
    expect(Array.from(packGPUSpatialParameterValues(unsigned, {count: 0xffffffff}))).toEqual([
      0, 0xffffffff
    ]);
    for (const bad of [-1, 1.5, 0x100000000, NaN]) {
      expect(() => packGPUSpatialParameterValues(unsigned, {count: bad}), `${bad}`).toThrow();
    }
    const signed = defineGPUSpatialParameterSchema({
      id: 'signed',
      format: 'sint32',
      wordLength: 1,
      fields: [{name: 'offset', format: 'sint32', wordOffset: 0, defaultValue: -3, dynamic: true}]
    });
    expect(packGPUSpatialParameterValues(signed)).toBeInstanceOf(Int32Array);
    expect(Array.from(packGPUSpatialParameterValues(signed, {offset: -0x80000000}))).toEqual([
      -0x80000000
    ]);
    expect(() => packGPUSpatialParameterValues(signed, {offset: 0x80000000})).toThrow(/sint32/);
    expect(() => packGPUSpatialParameterValues(signed, {offset: 0.5})).toThrow(/sint32/);
  });

  it('applies defaults to unset or undefined values and freezes the defined schema', () => {
    const defined = defineGPUSpatialParameterSchema(
      schema([field({name: 'x', defaultValue: 2.5}), field({name: 'y', wordOffset: 3})])
    );
    expect(Array.from(packGPUSpatialParameterValues(defined, {y: undefined}))).toEqual([
      2.5, 0, 0, 1
    ]);
    expect(Object.isFrozen(defined)).toBe(true);
    expect(Object.isFrozen(defined.fields)).toBe(true);
    expect(Object.isFrozen(defined.fields[0])).toBe(true);
    expect(() => packGPUSpatialParameterValues(defined, {x: 1e30, y: Infinity})).toThrow(/finite/);
  });
});

describe('ports and status', () => {
  it('validates point, polygon and trajectory geometry layouts', () => {
    const positions = view('float32x2', 6);
    expect(() =>
      validateGPUFeatureGeometryPort('p', {
        kind: 'points',
        positions,
        sourceIds: view('uint32', 6)
      } as never)
    ).not.toThrow();
    expect(() =>
      validateGPUFeatureGeometryPort('p', {
        kind: 'points',
        positions,
        sourceIds: view('uint32', 5)
      } as never)
    ).toThrow(/sourceIds length/);
    expect(() =>
      validateGPUFeatureGeometryPort('p', {kind: 'points', positions: view('float32', 6)} as never)
    ).toThrow(/positions must use float32x2/);
    const polygons = {
      kind: 'polygons',
      positions,
      featureOffsets: view('uint32', 3),
      polygonOffsets: view('uint32', 4),
      ringOffsets: view('uint32', 5)
    };
    expect(() => validateGPUFeatureGeometryPort('poly', polygons as never)).not.toThrow();
    expect(() =>
      validateGPUFeatureGeometryPort('poly', {...polygons, ringOffsets: view('uint32', 0)} as never)
    ).toThrow(/terminal/);
    expect(() =>
      validateGPUFeatureGeometryPort('poly', {...polygons, sourceIds: view('uint32', 3)} as never)
    ).toThrow(/feature count/);
    expect(() =>
      validateGPUFeatureGeometryPort('poly', {
        ...polygons,
        featureOffsets: view('float32', 3)
      } as never)
    ).toThrow(/featureOffsets must use uint32/);
    const trajectories = {
      kind: 'trajectories',
      positions,
      trackOffsets: view('uint32', 3),
      timestamps: view('float32', 6)
    };
    expect(() => validateGPUFeatureGeometryPort('t', trajectories as never)).not.toThrow();
    expect(() =>
      validateGPUFeatureGeometryPort('t', {...trajectories, timestamps: view('uint32', 6)} as never)
    ).not.toThrow();
    expect(() =>
      validateGPUFeatureGeometryPort('t', {...trajectories, timestamps: view('sint32', 6)} as never)
    ).toThrow(/timestamps must use/);
    expect(() =>
      validateGPUFeatureGeometryPort('t', {
        ...trajectories,
        timestamps: view('float32', 5)
      } as never)
    ).toThrow(/timestamps length/);
    expect(() =>
      validateGPUFeatureGeometryPort('t', {
        ...trajectories,
        trackOffsets: view('uint32', 0)
      } as never)
    ).toThrow(/terminal/);
  });

  it('validates compact pair port formats, capacities and scalars', () => {
    const pairs = {
      leftIds: view('uint32', 4),
      rightIds: view('uint32', 4),
      count: view('uint32', 1),
      overflow: view('uint32', 1)
    };
    expect(() => validateGPUCompactPairPort('pairs', pairs as never)).not.toThrow();
    expect(() =>
      validateGPUCompactPairPort('pairs', {...pairs, rightIds: view('uint32', 5)} as never)
    ).toThrow(/capacities must match/);
    expect(() =>
      validateGPUCompactPairPort('pairs', {...pairs, leftIds: view('float32', 4)} as never)
    ).toThrow(/leftIds must use uint32/);
    expect(() =>
      validateGPUCompactPairPort('pairs', {...pairs, overflow: view('uint32', 0)} as never)
    ).toThrow(/overflow must contain one uint32 row/);
    expect(() =>
      validateGPUCompactPairPort('pairs', {...pairs, candidateOverflow: view('sint32', 1)} as never)
    ).toThrow(/candidateOverflow must use uint32/);
  });

  it('skips absent status fields and rejects non-scalar or non-uint32 ones', () => {
    const scalar = view('uint32', 1);
    expect(() =>
      validateGPUStatusPort('s', {count: scalar, requiredCount: undefined} as never)
    ).not.toThrow();
    expect(() => validateGPUStatusPort('s', {count: view('float32', 1)} as never)).toThrow(
      /count must contain one uint32 row/
    );
    expect(() => validateGPUStatusPort('s', {overflow: view('uint32', 0)} as never)).toThrow(
      /overflow must contain/
    );
    expect(() => validateGPURecipeStatusPort('r', {stages: []})).not.toThrow();
    expect(() => validateGPURecipeStatusPort('r', {stages: [{stage: '', status: {}}]})).toThrow(
      /unique non-empty/
    );
    expect(() =>
      validateGPURecipeStatusPort('r', {
        stages: [{stage: 'join', status: {count: view('float32', 1)} as never}]
      })
    ).toThrow(/stage join count/);
  });

  it('lists each contract name once', () => {
    expect(new Set(GPU_SPATIAL_ANALYSIS_CONTRACT_NAMES).size).toBe(
      GPU_SPATIAL_ANALYSIS_CONTRACT_NAMES.length
    );
  });
});

describe('spatial contexts', () => {
  it('rejects metric, unit, radius, ellipsoid and grid combinations that do not make sense', () => {
    const planar = {coordinateSpace: 'planar', metric: 'native'} as const;
    expect(() => validateGPUSpatialContext('c', planar)).not.toThrow();
    expect(() => validateGPUSpatialContext('c', {...planar, metric: 'great-circle'})).toThrow(
      /incompatible with planar/
    );
    expect(() => validateGPUSpatialContext('c', {...planar, units: 'meters'})).toThrow(
      /native metrics/
    );
    expect(() =>
      validateGPUSpatialContext('c', {
        coordinateSpace: 'longitude-latitude',
        metric: 'great-circle',
        units: 'native'
      })
    ).toThrow(/geographic metrics/);
    const sphere = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;
    expect(() => validateGPUSpatialContext('c', {...sphere, sphereRadius: 0})).toThrow(
      /sphereRadius/
    );
    expect(() => validateGPUSpatialContext('c', {...sphere, sphereRadius: Infinity})).toThrow(
      /sphereRadius/
    );
    expect(() => validateGPUSpatialContext('c', {...planar, sphereRadius: 1})).toThrow(
      /sphereRadius/
    );
    const ellipsoidal = {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal'} as const;
    expect(() =>
      validateGPUSpatialContext('c', {
        ...ellipsoidal,
        ellipsoid: {semiMajorAxis: 6378137, flattening: 1 / 298.257223563}
      })
    ).not.toThrow();
    for (const ellipsoid of [
      {semiMajorAxis: 6378137, flattening: 1},
      {semiMajorAxis: 6378137, flattening: -0.1},
      {semiMajorAxis: 0, flattening: 0.1},
      {semiMajorAxis: 6378137, flattening: NaN}
    ]) {
      expect(() => validateGPUSpatialContext('c', {...ellipsoidal, ellipsoid})).toThrow(
        /ellipsoid/
      );
    }
    expect(() =>
      validateGPUSpatialContext('c', {
        ...sphere,
        ellipsoid: {semiMajorAxis: 6378137, flattening: 0.003}
      })
    ).toThrow(/ellipsoid/);
    const grid = {coordinateSpace: 'discrete-grid', metric: 'none', gridFamily: 'h3'} as const;
    expect(() => validateGPUSpatialContext('c', {...grid, gridResolution: 9})).not.toThrow();
    expect(() => validateGPUSpatialContext('c', {...grid, gridResolution: -1})).toThrow(
      /gridResolution/
    );
    expect(() => validateGPUSpatialContext('c', {...grid, gridResolution: 2.5})).toThrow(
      /gridResolution/
    );
    expect(() =>
      validateGPUSpatialContext('c', {
        coordinateSpace: 'topology-only',
        metric: 'none',
        gridFamily: 'h3'
      })
    ).toThrow(/grid metadata/);
  });

  it('reports the first difference between contexts and treats unspecified fields as compatible', () => {
    const base = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;
    const reason = (source: object, target: object) => {
      const result = getGPUSpatialContextCompatibility(source as never, target as never);
      return result.compatible ? 'compatible' : result.reason;
    };
    expect(reason(base, {...base, units: 'meters'})).toBe('compatible');
    expect(reason({...base, units: 'meters'}, {...base, units: 'kilometers'})).toMatch(
      /units differ/
    );
    expect(reason({...base, sphereRadius: 1}, {...base, sphereRadius: 2})).toBe(
      'sphere radii differ'
    );
    expect(reason({...base, sphereRadius: 1}, base)).toBe('compatible');
    // A 'none' metric is compatible with any metric within the same coordinate space.
    expect(reason({...base, metric: 'none'}, base)).toBe('compatible');
    const ellipsoidal = {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal'} as const;
    expect(
      reason(
        {...ellipsoidal, ellipsoid: {semiMajorAxis: 1, flattening: 0.1}},
        {...ellipsoidal, ellipsoid: {semiMajorAxis: 1, flattening: 0.2}}
      )
    ).toBe('ellipsoids differ');
    const grid = {coordinateSpace: 'discrete-grid', metric: 'none'} as const;
    expect(reason({...grid, gridFamily: 'H3'}, {...grid, gridFamily: 'h3'})).toBe('compatible');
    expect(reason({...grid, gridFamily: 'h3'}, {...grid, gridFamily: 's2'})).toBe(
      'grid families differ'
    );
    expect(
      reason(
        {...grid, gridFamily: 'h3', gridResolution: 8},
        {...grid, gridFamily: 'h3', gridResolution: 9}
      )
    ).toBe('grid resolutions differ');
  });
});

describe('capacity planning', () => {
  it('clamps between minimum and maximum capacity and reports limitation', () => {
    expect(getGPUCapacityPlan({sourceCount: 3, minimumCapacity: 10})).toMatchObject({
      capacity: 10,
      estimatedRequiredCount: 3,
      limited: false
    });
    expect(getGPUCapacityPlan({sourceCount: 0})).toMatchObject({capacity: 0, limited: false});
    // Fractional requirements round up: 3 * 0.5 = 1.5 -> 2 rows.
    expect(getGPUCapacityPlan({sourceCount: 3, expansionFactor: 0.5}).capacity).toBe(2);
    expect(getGPUCapacityPlan({sourceCount: 5, maximumCapacity: 4})).toMatchObject({
      capacity: 4,
      limited: true
    });
  });

  it('rejects invalid planning inputs and saturates instead of overflowing', () => {
    expect(() => getGPUCapacityPlan({sourceCount: -1})).toThrow(/sourceCount/);
    expect(() => getGPUCapacityPlan({sourceCount: 1.5})).toThrow(/sourceCount/);
    expect(() => getGPUCapacityPlan({sourceCount: 1, expansionFactor: -1})).toThrow(
      /expansionFactor/
    );
    expect(() => getGPUCapacityPlan({sourceCount: 1, expansionFactor: Infinity})).toThrow(
      /expansionFactor/
    );
    expect(() =>
      getGPUCapacityPlan({sourceCount: 1, minimumCapacity: 5, maximumCapacity: 4})
    ).toThrow(/maximumCapacity/);
    expect(() => getGPUCapacityPlan({sourceCount: 1, outputBytesPerRow: 0.5})).toThrow(
      /outputBytesPerRow/
    );
    const huge = getGPUCapacityPlan({
      sourceCount: Number.MAX_SAFE_INTEGER,
      expansionFactor: 1e9,
      outputBytesPerRow: 1e9
    });
    expect(huge.estimatedRequiredCount).toBe(Number.MAX_SAFE_INTEGER);
    expect(huge.estimatedOutputBytes).toBe(Number.MAX_SAFE_INTEGER);
  });

  it('uses the default per-left estimates and lets observations override them', () => {
    // Defaults: min(rightCount, 8) candidates and min(rightCount, 1) matches per left row.
    const estimated = getGPUSpatialJoinCapacityPlan({leftCount: 10, rightCount: 3});
    expect(estimated.candidates.estimatedRequiredCount).toBe(30);
    expect(estimated.pairs.estimatedRequiredCount).toBe(10);
    expect(estimated.candidates.limited).toBe(false);
    const limited = getGPUSpatialJoinCapacityPlan({
      leftCount: 10,
      rightCount: 100,
      candidateCapacity: 20
    });
    expect(limited.candidates).toMatchObject({
      estimatedRequiredCount: 80,
      capacity: 20,
      limited: true
    });
    expect(
      getGPUSpatialJoinCapacityPlan({leftCount: 10, rightCount: 3, observedCandidateCount: 0})
        .candidates.estimatedRequiredCount
    ).toBe(0);
    expect(() => getGPUSpatialJoinCapacityPlan({leftCount: -1, rightCount: 3})).toThrow(
      /leftCount/
    );
    expect(() =>
      getGPUSpatialJoinCapacityPlan({leftCount: 1, rightCount: 3, pairCapacity: -1})
    ).toThrow(/capacity/);
    expect(() =>
      getGPUCellCoverContributorCapacityPlan({featureCount: 1, vertexCount: 1.5})
    ).toThrow(/vertexCount/);
  });

  it('rejects inconsistent status observations', () => {
    const policy = {kind: 'fixed-budget'} as const;
    expect(() =>
      getGPUCapacityRecovery(policy, {output: {capacity: 4, overflow: false} as never})
    ).toThrow(/count is required/);
    expect(() =>
      getGPUCapacityRecovery(policy, {output: {capacity: 4, count: 5, overflow: true}})
    ).toThrow(/must not exceed/);
    expect(() =>
      getGPUCapacityRecovery(policy, {
        output: {capacity: 4, count: 4, requiredCount: 3, overflow: true}
      })
    ).toThrow(/below/);
    expect(() =>
      getGPUCapacityRecovery(policy, {output: {capacity: 4.5, count: 1, overflow: false}})
    ).toThrow(/capacity/);
  });

  it('accepts incomplete results under a fixed budget and names the incomplete stages', () => {
    expect(
      getGPUCapacityRecovery(
        {kind: 'fixed-budget'},
        {
          output: {capacity: 4, count: 4, overflow: true},
          candidate: {capacity: 8, overflow: true}
        }
      )
    ).toEqual({action: 'accept', complete: false, incompleteStages: ['candidate', 'output']});
  });

  it('grows to at least the observed requirement, honours maxima and rejects when stuck', () => {
    // Default factor 2: max(requiredCount, ceil(capacity * 2)).
    expect(
      getGPUCapacityRecovery(
        {kind: 'grow-on-next-frame'},
        {output: {capacity: 10, count: 10, requiredCount: 100, overflow: true}}
      )
    ).toMatchObject({action: 'grow', nextOutputCapacity: 100});
    // Without a requirement, growth is capacity * factor, never from a zero capacity stalling.
    expect(
      getGPUCapacityRecovery(
        {kind: 'grow-on-next-frame', growthFactor: 1.5},
        {output: {capacity: 0, count: 0, overflow: true}}
      )
    ).toMatchObject({action: 'grow', nextOutputCapacity: 2});
    // The maximum caps growth.
    expect(
      getGPUCapacityRecovery(
        {kind: 'grow-on-next-frame', maximumOutputCapacity: 30},
        {output: {capacity: 10, count: 10, requiredCount: 100, overflow: true}}
      )
    ).toMatchObject({action: 'grow', nextOutputCapacity: 30});
    // Already at the maximum: nothing to grow into, so the frame is rejected.
    expect(
      getGPUCapacityRecovery(
        {kind: 'grow-on-next-frame', maximumOutputCapacity: 10},
        {output: {capacity: 10, count: 10, overflow: true}}
      )
    ).toEqual({action: 'reject', complete: false, incompleteStages: ['output']});
    expect(
      getGPUCapacityRecovery(
        {kind: 'grow-on-next-frame', maximumCandidateCapacity: 8},
        {
          output: {capacity: 10, count: 3, overflow: false},
          candidate: {capacity: 8, overflow: true}
        }
      )
    ).toEqual({action: 'reject', complete: false, incompleteStages: ['candidate']});
    // A complete output keeps its capacity while only the candidate stage grows.
    expect(
      getGPUCapacityRecovery(
        {kind: 'grow-on-next-frame'},
        {
          output: {capacity: 10, count: 3, overflow: false},
          candidate: {capacity: 8, overflow: true}
        }
      )
    ).toEqual({
      action: 'grow',
      complete: false,
      incompleteStages: ['candidate'],
      nextOutputCapacity: 10,
      nextCandidateCapacity: 16
    });
    for (const growthFactor of [1, 0.5, NaN, Infinity]) {
      expect(() =>
        getGPUCapacityRecovery(
          {kind: 'grow-on-next-frame', growthFactor},
          {output: {capacity: 1, count: 1, overflow: true}}
        )
      ).toThrow(/growthFactor/);
    }
  });
});

describe('partitioning', () => {
  const partition = (
    id: number,
    coreStart: number,
    coreEnd: number,
    haloStart = coreStart,
    haloEnd = coreEnd
  ) => ({
    id,
    coreStart,
    coreEnd,
    haloStart,
    haloEnd
  });
  const descriptor = (length: number, partitions: object[], extra = {}) => ({
    length,
    partitions,
    seamOwnership: {kind: 'lowest-partition-id'} as const,
    ...extra
  });

  it('validates partition descriptors', () => {
    expect(() => validateGPUPartitionDescriptor('d', descriptor(0, []))).not.toThrow();
    expect(() => validateGPUPartitionDescriptor('d', descriptor(5, []))).toThrow(/at least one/);
    expect(() => validateGPUPartitionDescriptor('d', descriptor(-1, []))).toThrow(/length/);
    expect(() =>
      validateGPUPartitionDescriptor('d', descriptor(4, [partition(0, 0, 4)], {haloDistance: -1}))
    ).toThrow(/haloDistance/);
    expect(() =>
      validateGPUPartitionDescriptor('d', descriptor(4, [partition(0, 0, 2), partition(0, 2, 4)]))
    ).toThrow(/unique/);
    expect(() =>
      validateGPUPartitionDescriptor('d', descriptor(4, [partition(0, 0, 2), partition(1, 3, 4)]))
    ).toThrow(/contiguous/);
    expect(() =>
      validateGPUPartitionDescriptor('d', descriptor(5, [partition(0, 0, 2), partition(1, 2, 4)]))
    ).toThrow(/cover partition length/);
    expect(() =>
      validateGPUPartitionDescriptor(
        'd',
        descriptor(4, [partition(0, 0, 2, 1, 2), partition(1, 2, 4)])
      )
    ).toThrow(/halo must contain/);
    expect(() =>
      validateGPUPartitionDescriptor('d', descriptor(4, [partition(0, 0, 4, 0, 5)]))
    ).toThrow(/halo must contain/);
    expect(() =>
      validateGPUPartitionDescriptor('d', descriptor(4, [partition(1.5, 0, 4)]))
    ).toThrow(/integers/);
    // Empty cores are legal as long as they are contiguous.
    expect(() =>
      validateGPUPartitionDescriptor('d', descriptor(4, [partition(0, 0, 0), partition(1, 0, 4)]))
    ).not.toThrow();
  });

  it('assigns seams by lowest partition id or by the source side', () => {
    const lowest = descriptor(4, [partition(0, 0, 4)]);
    expect(isGPUSeamResultOwner(lowest as never, 2, 5, 2)).toBe(true);
    expect(isGPUSeamResultOwner(lowest as never, 5, 5, 2)).toBe(false);
    const sourceLeft = {...lowest, seamOwnership: {kind: 'source-core', sourceSide: 'left'}};
    const sourceRight = {...lowest, seamOwnership: {kind: 'source-core', sourceSide: 'right'}};
    expect(isGPUSeamResultOwner(sourceLeft as never, 5, 5, 2)).toBe(true);
    expect(isGPUSeamResultOwner(sourceRight as never, 2, 5, 2)).toBe(true);
    expect(isGPUSeamResultOwner(sourceRight as never, 5, 5, 2)).toBe(false);
  });

  it('validates tiles and picks the lowest known tile as the seam owner', () => {
    const tile = (id: number, bounds: number[], haloBounds: number[]) => ({id, bounds, haloBounds});
    const ok = tile(1, [0, 0, 1, 1], [0, 0, 1, 1]);
    const check = (tiles: object[]) =>
      validateGPUTilePartitionDescriptor('t', {tiles, seamOwnership: 'lowest-tile-id'} as never);
    expect(() => check([ok])).not.toThrow();
    expect(() => check([ok, {...ok}])).toThrow(/unique/);
    expect(() => check([tile(-1, [0, 0, 1, 1], [0, 0, 1, 1])])).toThrow(/unique/);
    expect(() => check([tile(1, [0, 0, NaN, 1], [0, 0, 1, 1])])).toThrow(/finite/);
    expect(() => check([tile(1, [1, 0, 0, 1], [0, 0, 1, 1])])).toThrow(/ordered/);
    expect(() => check([tile(1, [0, 0, 1, 1], [0.1, 0, 1, 1])])).toThrow(/contain core/);
    const descriptor = {
      tiles: [ok, tile(5, [1, 0, 2, 1], [1, 0, 2, 1])],
      seamOwnership: 'lowest-tile-id'
    } as never;
    expect(getGPUTileSeamOwner(descriptor, [])).toBeUndefined();
    expect(getGPUTileSeamOwner(descriptor, [5])).toBe(5);
    expect(getGPUTileSeamOwner(descriptor, [5, 1, 5])).toBe(1);
    expect(() => getGPUTileSeamOwner(descriptor, [1, 7])).toThrow(/tile 7/);
  });

  it('plans memory from the largest halo and rejects negative row sizes', () => {
    const partitioning = descriptor(10, [partition(0, 0, 4, 0, 6), partition(1, 4, 10, 2, 10)]);
    expect(getGPUPartitionMemoryPlan(partitioning as never, 4)).toEqual({
      packedRows: 10,
      peakPartitionRows: 8,
      packedBytes: 40,
      peakPartitionBytes: 32,
      savedPeakBytes: 8
    });
    expect(() => getGPUPartitionMemoryPlan(partitioning as never, -1)).toThrow(/bytesPerRow/);
    expect(() => getGPUPartitionMemoryPlan(partitioning as never, 0.5)).toThrow(/bytesPerRow/);
  });

  it('prepared-index compatibility compares every descriptor field', () => {
    const context = {coordinateSpace: 'planar', metric: 'native'} as const;
    const prepared = {
      spatialContext: context,
      partitioning: descriptor(4, [partition(0, 0, 4)]),
      revision: 1
    } as never;
    const requested = (overrides: object) => ({...(prepared as object), ...overrides}) as never;
    expect(() => assertCompatibleGPUPreparedIndexes('i', prepared, requested({}))).not.toThrow();
    expect(() =>
      assertCompatibleGPUPreparedIndexes('i', prepared, requested({revision: '1'}))
    ).toThrow(/revision/);
    expect(() =>
      assertCompatibleGPUPreparedIndexes(
        'i',
        prepared,
        requested({spatialContext: {...context, units: 'native'}})
      )
    ).toThrow(/spatial context/);
    expect(() =>
      assertCompatibleGPUPreparedIndexes(
        'i',
        prepared,
        requested({partitioning: descriptor(4, [partition(0, 0, 4)], {haloDistance: 1})})
      )
    ).toThrow(/partitioning/);
    expect(() =>
      assertCompatibleGPUPreparedIndexes(
        'i',
        prepared,
        requested({partitioning: descriptor(4, [partition(0, 0, 4, 0, 4), partition(1, 4, 4)])})
      )
    ).toThrow(/partitioning/);
    expect(() =>
      assertCompatibleGPUPreparedIndexes(
        'i',
        prepared,
        requested({
          partitioning: descriptor(4, [partition(0, 0, 4)], {
            seamOwnership: {kind: 'source-core', sourceSide: 'left'}
          })
        })
      )
    ).toThrow(/partitioning/);
  });

  it('chooses the cheapest query strategy, preferring scan on ties, and validates measurements', () => {
    // Scan is 10 * 10 = 100; grid is 10 + 10 + 5 + 0 = 25; BVH is 10 * 3 + 0 = 30.
    expect(
      getGPUSpatialQueryCostPlan({
        sourceCount: 10,
        targetCount: 10,
        overlapRatio: 0,
        populatedCellCount: 5,
        preparedBVHDepth: 3
      })
    ).toEqual({
      strategy: 'grid',
      estimatedComparisons: 25,
      alternatives: {scan: 100, grid: 25, 'prepared-bvh': 30}
    });
    expect(
      getGPUSpatialQueryCostPlan({
        sourceCount: 10,
        targetCount: 10,
        overlapRatio: 0,
        preparedBVHDepth: 0
      })
    ).toMatchObject({strategy: 'prepared-bvh', estimatedComparisons: 10});
    // Without structure measurements only the scan is available; an equal cost keeps the scan.
    expect(
      getGPUSpatialQueryCostPlan({sourceCount: 2, targetCount: 2, overlapRatio: 1})
    ).toMatchObject({
      strategy: 'scan',
      estimatedComparisons: 4
    });
    expect(
      getGPUSpatialQueryCostPlan({
        sourceCount: 1,
        targetCount: 4,
        overlapRatio: 0,
        populatedCellCount: 0
      }).strategy
    ).toBe('scan');
    for (const bad of [
      {overlapRatio: 1.1},
      {overlapRatio: -0.1},
      {overlapRatio: NaN},
      {sourceCount: -1},
      {targetCount: 1.5},
      {populatedCellCount: -1},
      {preparedBVHDepth: 0.5}
    ]) {
      expect(() =>
        getGPUSpatialQueryCostPlan({sourceCount: 1, targetCount: 1, overlapRatio: 0.5, ...bad})
      ).toThrow();
    }
  });
});
