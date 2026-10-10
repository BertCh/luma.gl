// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUFeatureGeometryPort} from './ports';
import type {GPUSpatialContext} from './spatial-context';

/** One contiguous core range and the halo rows made visible to its worker. */
export type GPUPartitionRange = {
  /** Stable partition identity. IDs must be unique but need not be dense. */
  id: number;
  /** Inclusive first row owned by this partition. */
  coreStart: number;
  /** Exclusive end of the rows owned by this partition. */
  coreEnd: number;
  /** Inclusive first row available for neighborhood operations. */
  haloStart: number;
  /** Exclusive end of the rows available for neighborhood operations. */
  haloEnd: number;
};

/** Deterministic rule used when a result can be emitted by more than one halo. */
export type GPUSeamOwnership =
  | {kind: 'lowest-partition-id'}
  | {kind: 'source-core'; sourceSide: 'left' | 'right'};

/** Logical row partitioning carried beside GPU ports without repacking their storage. */
export type GPUPartitionDescriptor = {
  /** Total number of logical rows covered by the core ranges. */
  length: number;
  /** Core ranges in increasing row order. */
  partitions: readonly GPUPartitionRange[];
  /** Ownership rule that prevents duplicated seam output. */
  seamOwnership: GPUSeamOwnership;
  /** Maximum logical distance represented by each halo, when known. */
  haloDistance?: number;
};

/** A canonical geometry port with its original partition boundaries retained. */
export type GPUPartitionedGeometryPort<Geometry extends GPUFeatureGeometryPort> = {
  geometry: Geometry;
  partitioning: GPUPartitionDescriptor;
};

/** One spatial tile and the larger bounds from which it may read halo features. */
export type GPUTilePartition = {
  id: number;
  /** Core bounds `[minimumX, minimumY, maximumX, maximumY]`. */
  bounds: readonly [number, number, number, number];
  /** Read bounds containing `bounds`. */
  haloBounds: readonly [number, number, number, number];
};

/** Spatial tile identity and deterministic cross-tile result ownership. */
export type GPUTilePartitionDescriptor = {
  tiles: readonly GPUTilePartition[];
  seamOwnership: 'lowest-tile-id';
};

/** Immutable identity required to reuse a prepared index across operators or frames. */
export type GPUPreparedIndexDescriptor = {
  spatialContext: GPUSpatialContext;
  partitioning: GPUPartitionDescriptor;
  /** Caller-managed identity of the indexed values. */
  revision: string | number;
};

/** Measurements available while planning a spatial query without a GPU readback. */
export type GPUSpatialQueryCostMeasurements = {
  sourceCount: number;
  targetCount: number;
  /** Estimated fraction of targets overlapping one source bound, in [0, 1]. */
  overlapRatio: number;
  /** Number of populated grid cells when a grid index is available. */
  populatedCellCount?: number;
  /** Existing BVH depth when a reusable prepared index is available. */
  preparedBVHDepth?: number;
};

/** Query execution strategies selected only from graph-construction measurements. */
export type GPUSpatialQueryStrategy = 'scan' | 'grid' | 'prepared-bvh';

/** Estimated work and selected strategy for a spatial query. */
export type GPUSpatialQueryCostPlan = {
  strategy: GPUSpatialQueryStrategy;
  estimatedComparisons: number;
  alternatives: Readonly<Record<GPUSpatialQueryStrategy, number>>;
};

/** Peak-row and byte comparison between explicit partition execution and implicit packing. */
export type GPUPartitionMemoryPlan = {
  packedRows: number;
  peakPartitionRows: number;
  packedBytes: number;
  peakPartitionBytes: number;
  savedPeakBytes: number;
};

/** Validates complete, non-overlapping core coverage and bounded halo ranges. */
export function validateGPUPartitionDescriptor(
  id: string,
  descriptor: GPUPartitionDescriptor
): void {
  if (!Number.isInteger(descriptor.length) || descriptor.length < 0) {
    throw new Error(`${id} partition length must be a non-negative integer`);
  }
  if (
    descriptor.haloDistance !== undefined &&
    (!Number.isFinite(descriptor.haloDistance) || descriptor.haloDistance < 0)
  ) {
    throw new Error(`${id} haloDistance must be a non-negative finite number`);
  }
  if (descriptor.length > 0 && descriptor.partitions.length === 0) {
    throw new Error(`${id} non-empty data needs at least one partition`);
  }
  const partitionIds = new Set<number>();
  let expectedCoreStart = 0;
  for (const partition of descriptor.partitions) {
    const boundaries = [
      partition.id,
      partition.coreStart,
      partition.coreEnd,
      partition.haloStart,
      partition.haloEnd
    ];
    if (boundaries.some(value => !Number.isInteger(value) || value < 0)) {
      throw new Error(`${id} partition boundaries and IDs must be non-negative integers`);
    }
    if (partitionIds.has(partition.id)) {
      throw new Error(`${id} partition IDs must be unique`);
    }
    partitionIds.add(partition.id);
    if (partition.coreStart !== expectedCoreStart || partition.coreEnd < partition.coreStart) {
      throw new Error(`${id} core ranges must form ordered contiguous coverage`);
    }
    if (
      partition.haloStart > partition.coreStart ||
      partition.haloEnd < partition.coreEnd ||
      partition.haloEnd > descriptor.length
    ) {
      throw new Error(`${id} each halo must contain its core and stay within the input`);
    }
    expectedCoreStart = partition.coreEnd;
  }
  if (expectedCoreStart !== descriptor.length) {
    throw new Error(`${id} core ranges must cover partition length exactly`);
  }
}

/** Returns whether a partition owns a seam result under the descriptor's canonical rule. */
export function isGPUSeamResultOwner(
  descriptor: GPUPartitionDescriptor,
  partitionId: number,
  leftCorePartitionId: number,
  rightCorePartitionId: number
): boolean {
  const {seamOwnership} = descriptor;
  const owner =
    seamOwnership.kind === 'lowest-partition-id'
      ? Math.min(leftCorePartitionId, rightCorePartitionId)
      : seamOwnership.sourceSide === 'left'
        ? leftCorePartitionId
        : rightCorePartitionId;
  return partitionId === owner;
}

/** Validates finite tile bounds, halo containment and unique stable tile IDs. */
export function validateGPUTilePartitionDescriptor(
  id: string,
  descriptor: GPUTilePartitionDescriptor
): void {
  const tileIds = new Set<number>();
  for (const tile of descriptor.tiles) {
    if (!Number.isInteger(tile.id) || tile.id < 0 || tileIds.has(tile.id)) {
      throw new Error(`${id} tile IDs must be unique non-negative integers`);
    }
    tileIds.add(tile.id);
    const [minimumX, minimumY, maximumX, maximumY] = tile.bounds;
    const [haloMinimumX, haloMinimumY, haloMaximumX, haloMaximumY] = tile.haloBounds;
    if (![...tile.bounds, ...tile.haloBounds].every(Number.isFinite)) {
      throw new Error(`${id} tile bounds must be finite`);
    }
    if (minimumX > maximumX || minimumY > maximumY) {
      throw new Error(`${id} tile bounds must be ordered`);
    }
    if (
      haloMinimumX > minimumX ||
      haloMinimumY > minimumY ||
      haloMaximumX < maximumX ||
      haloMaximumY < maximumY
    ) {
      throw new Error(`${id} tile halo bounds must contain core bounds`);
    }
  }
}

/** Selects the one tile allowed to emit a result observed in multiple halos. */
export function getGPUTileSeamOwner(
  descriptor: GPUTilePartitionDescriptor,
  candidateTileIds: readonly number[]
): number | undefined {
  const knownTileIds = new Set(descriptor.tiles.map(tile => tile.id));
  let owner: number | undefined;
  for (const tileId of candidateTileIds) {
    if (!knownTileIds.has(tileId)) {
      throw new Error(`candidate tile ${tileId} is not in the partition descriptor`);
    }
    owner = owner === undefined ? tileId : Math.min(owner, tileId);
  }
  return owner;
}

/** Throws when a prepared index cannot be reused without rebuilding or repartitioning. */
export function assertCompatibleGPUPreparedIndexes(
  id: string,
  prepared: GPUPreparedIndexDescriptor,
  requested: GPUPreparedIndexDescriptor
): void {
  if (prepared.revision !== requested.revision) {
    throw new Error(`${id} prepared index revision differs`);
  }
  if (!equalSpatialContexts(prepared.spatialContext, requested.spatialContext)) {
    throw new Error(`${id} prepared index spatial context differs`);
  }
  if (!equalPartitionDescriptors(prepared.partitioning, requested.partitioning)) {
    throw new Error(`${id} prepared index partitioning differs`);
  }
}

/** Selects scan, populated-grid, or prepared-BVH execution from bounded CPU-side metadata. */
export function getGPUSpatialQueryCostPlan(
  measurements: GPUSpatialQueryCostMeasurements
): GPUSpatialQueryCostPlan {
  validateCostMeasurements(measurements);
  const {sourceCount, targetCount, overlapRatio, populatedCellCount, preparedBVHDepth} =
    measurements;
  const scan = sourceCount * targetCount;
  const expectedCandidates = Math.ceil(scan * overlapRatio);
  const grid =
    populatedCellCount === undefined
      ? Number.POSITIVE_INFINITY
      : sourceCount + targetCount + populatedCellCount + expectedCandidates;
  const preparedBVH =
    preparedBVHDepth === undefined
      ? Number.POSITIVE_INFINITY
      : sourceCount * Math.max(preparedBVHDepth, 1) + expectedCandidates;
  const alternatives = {scan, grid, 'prepared-bvh': preparedBVH} as const;
  let strategy: GPUSpatialQueryStrategy = 'scan';
  for (const candidate of ['grid', 'prepared-bvh'] as const) {
    if (alternatives[candidate] < alternatives[strategy]) {
      strategy = candidate;
    }
  }
  return {strategy, estimatedComparisons: alternatives[strategy], alternatives};
}

/** Measures the peak live input implied by one-partition-at-a-time execution, including halos. */
export function getGPUPartitionMemoryPlan(
  descriptor: GPUPartitionDescriptor,
  bytesPerRow: number
): GPUPartitionMemoryPlan {
  validateGPUPartitionDescriptor('partition memory plan', descriptor);
  if (!Number.isSafeInteger(bytesPerRow) || bytesPerRow < 0) {
    throw new Error('bytesPerRow must be a non-negative safe integer');
  }
  const peakPartitionRows = descriptor.partitions.reduce(
    (peak, partition) => Math.max(peak, partition.haloEnd - partition.haloStart),
    0
  );
  const packedBytes = descriptor.length * bytesPerRow;
  const peakPartitionBytes = peakPartitionRows * bytesPerRow;
  return {
    packedRows: descriptor.length,
    peakPartitionRows,
    packedBytes,
    peakPartitionBytes,
    savedPeakBytes: packedBytes - peakPartitionBytes
  };
}

function validateCostMeasurements(measurements: GPUSpatialQueryCostMeasurements): void {
  for (const [name, value] of [
    ['sourceCount', measurements.sourceCount],
    ['targetCount', measurements.targetCount]
  ] as const) {
    if (!Number.isInteger(value) || value < 0) {
      throw new Error(`${name} must be a non-negative integer`);
    }
  }
  if (
    !Number.isFinite(measurements.overlapRatio) ||
    measurements.overlapRatio < 0 ||
    measurements.overlapRatio > 1
  ) {
    throw new Error('overlapRatio must be between zero and one');
  }
  for (const [name, value] of [
    ['populatedCellCount', measurements.populatedCellCount],
    ['preparedBVHDepth', measurements.preparedBVHDepth]
  ] as const) {
    if (value !== undefined && (!Number.isInteger(value) || value < 0)) {
      throw new Error(`${name} must be a non-negative integer`);
    }
  }
}

function equalSpatialContexts(left: GPUSpatialContext, right: GPUSpatialContext): boolean {
  return (
    left.coordinateSpace === right.coordinateSpace &&
    left.metric === right.metric &&
    left.units === right.units &&
    left.sphereRadius === right.sphereRadius &&
    left.gridFamily === right.gridFamily &&
    left.gridResolution === right.gridResolution &&
    left.ellipsoid?.semiMajorAxis === right.ellipsoid?.semiMajorAxis &&
    left.ellipsoid?.flattening === right.ellipsoid?.flattening
  );
}

function equalPartitionDescriptors(
  left: GPUPartitionDescriptor,
  right: GPUPartitionDescriptor
): boolean {
  return (
    left.length === right.length &&
    left.haloDistance === right.haloDistance &&
    left.seamOwnership.kind === right.seamOwnership.kind &&
    (left.seamOwnership.kind !== 'source-core' ||
      (right.seamOwnership.kind === 'source-core' &&
        left.seamOwnership.sourceSide === right.seamOwnership.sourceSide)) &&
    left.partitions.length === right.partitions.length &&
    left.partitions.every((partition, index) => {
      const other = right.partitions[index];
      return (
        other !== undefined &&
        partition.id === other.id &&
        partition.coreStart === other.coreStart &&
        partition.coreEnd === other.coreEnd &&
        partition.haloStart === other.haloStart &&
        partition.haloEnd === other.haloEnd
      );
    })
  );
}
