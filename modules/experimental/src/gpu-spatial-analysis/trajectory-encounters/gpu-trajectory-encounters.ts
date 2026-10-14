// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCompaction,
  GPUGridIndex,
  GPUSort,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCompactOutput} from '../../utils/gpu-contributor-types';
import {createPublishNode, createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateCompactOutput,
  validateGraphOutputsDisjointFromInputs,
  validateGPUStorageBindingByteLength,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import type {GPUSpatialContext} from '../contracts/spatial-context';
import {
  getTrajectoryGeodesicRadius,
  resolveTrajectorySpatialContext
} from '../trajectory-analysis/trajectory-spatial-context';
import {
  createEncounterClearNode,
  createEncounterCompositeKeyNode,
  createEncounterOverflowNode,
  createEncounterPositionsNode,
  createEncounterReduceNodes,
  createEncounterRunFlagsNode,
  createEncounterScanNode,
  createEncounterSortedHitsNode,
  createEncounterSortKeyNode,
  type EncounterKeyBits,
  type EncounterShape
} from './encounter-kernels';

const OPERATION = 'GPUTrajectoryEncounters';
/** Largest `columns * rows * bucketCount` lattice the contributor builds. */
const MAXIMUM_CELL_COUNT = 1 << 27;

/**
 * Capacity-bounded encounter pair list of {@link GPUTrajectoryEncounters}.
 *
 * `output.ids.length` is the pair capacity. Every column that is present must have exactly that
 * many rows. Pairs are ordered by `(track, partner)` with `track < partner`. Rows at and after
 * `output.count` hold sentinels: `0xffffffff` for IDs and buckets, `0` for counts, distances and
 * times.
 */
export type GPUTrajectoryEncounterOutput = {
  /** Bounded compact result. `ids` holds the lower track index of each pair. */
  output: GPUCompactOutput;
  /** Candidate-stage incompleteness from the hit scratch or grid index. */
  candidateOverflow: GraphDataView<'uint32'>;
  /** Optional higher track index of each pair. */
  partners?: GraphDataView<'uint32'>;
  /** Optional first time bucket in which the pair was within the distance. */
  firstBuckets?: GraphDataView<'uint32'>;
  /** Optional smallest distance over the pair's encounter buckets. */
  minimumDistances?: GraphDataView<'float32'>;
  /** Optional number of buckets in which the pair was within the distance. */
  bucketCounts?: GraphDataView<'uint32'>;
  /** Optional `bucketTimes[firstBucket]`. Requires `bucketTimes` and `firstBuckets`. */
  firstTimes?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUTrajectoryEncounters}.
 *
 * Per-frame (no recompile): the contents of every input buffer, including `distance`. Compile-time:
 * `trackCount`, `bucketCount`, `cellSize`, `bounds`, `hitCapacity`, the output capacity, and which
 * optional views are present.
 */
export type GPUTrajectoryEncountersProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'trajectory-encounters'`. */
  id?: string;
  /**
   * Coordinate space and metric of `samples`, `distance`, `cellSize` and `bounds`. Omitted means
   * native planar coordinates: `distance` and `cellSize` are position units and distances are
   * Euclidean. `{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}` means samples are
   * `(longitude, latitude)` degrees, `distance` is great-circle meters on a sphere of
   * `sphereRadius` (default `GPU_EARTH_MEAN_RADIUS_METERS`), `cellSize` and `bounds` stay in
   * degrees, and the reported distances are meters. Other contexts throw.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Dense `[trackCount * bucketCount]` positions on a common clock: row `track * bucketCount +
   * bucket` is the position of the track at time bucket `bucket`, the layout of
   * `GPUTrajectoryResample.samples`. Samples with a non-finite coordinate, or outside `bounds`, are
   * absent.
   */
  samples: GraphDataView<'float32x2'>;
  /** Number of tracks. */
  trackCount: number;
  /** Number of time buckets per track. */
  bucketCount: number;
  /**
   * Optional per-track flags, `trackCount` rows. Zero ignores the track in every bucket, for
   * example empty tracks that `GPUTrajectoryResample` writes as zeros.
   */
  trackValid?: GraphDataView<'uint32'>;
  /**
   * Per-frame distance, one float32 row. A pair is an encounter in a bucket when the distance of
   * its two samples (planar, or great-circle meters) is at most this value. It is clamped to
   * `cellSize` (planar) or to {@link GPUTrajectoryEncounters.maximumDistance} (great-circle).
   * Negative or NaN matches nothing.
   */
  distance: GraphDataView<'float32'>;
  /**
   * Compile-time grid cell width in position units (degrees for a great-circle context), chosen
   * so that cells are at least as wide as the largest `distance` you will use (planar), or so that
   * {@link GPUTrajectoryEncounters.maximumDistance} covers it (great-circle). The lattice has
   * `floor(extent / cellSize)` columns and rows (at least 1), so cells are never narrower than
   * this. In a great-circle context it must be below 90 degrees, and below 120 degrees when the
   * bounds span all longitudes.
   */
  cellSize: number;
  /**
   * Finite domain `[minX, minY, maxX, maxY]` of the lattice. Samples outside are absent. In a
   * great-circle context longitudes lie in `[-180, 180]`, latitudes in `(-90, 90)`; bounds of
   * exactly `[-180, 180]` in longitude make the lattice wrap across the antimeridian (sample
   * longitudes are folded into that range first), otherwise pairs across the antimeridian are not
   * found.
   */
  bounds: readonly [number, number, number, number];
  /**
   * Maximum `(track, partner, bucket)` hits per encoding. Also the length of the sorted hit
   * scratch, which every encoding sorts.
   */
  hitCapacity: number;
  /** Bounded pair list. */
  pairs: GPUTrajectoryEncounterOutput;
  /** Optional `bucketCount` rows with the time of each bucket, read by `pairs.firstTimes`. */
  bucketTimes?: GraphDataView<'float32'>;
};

/**
 * Finds pairs of tracks that come within a distance of each other in the same time bucket, the
 * MobilityDB `tdwithin` question for resampled trajectories.
 *
 * Samples are indexed in a `GPUGridIndex` with three axes: planar `x` and `y` over `bounds`, and
 * the time bucket as an exact third axis, so only samples of one bucket share a cell. Every
 * sample scans the 3x3 planar cells of its bucket and appends a hit for each higher-index track
 * within `min(distance, cellSize)`. Hits are ordered with three stable `GPUSort` passes (bucket,
 * partner, track) so each `(track, partner)` pair forms one run whose first hit is the first
 * encounter; `GPUCompaction` finds the runs and reduction kernels write the minimum distance and
 * bucket count of each run. Pairs are therefore lowest-ID ordered (`track < partner`),
 * deduplicated across buckets, and sorted by `(track, partner)`.
 *
 * Semantics: this is a discrete approximation. Two tracks that pass within the distance between
 * two buckets are missed; choose `bucketCount` so the distance travelled per bucket is small
 * compared with `distance`, or treat results as encounters sampled at bucket times. Samples on a
 * common clock are required (column `k` means the same instant for every track). Use
 * {@link addClockEncounters} (`GPUTrajectoryResample` with `spacing: 'clock'`) to build that table
 * for tracks with different time windows, or supply your own table and mark absent samples with
 * NaN or `trackValid`.
 *
 * Spatial context: with a great-circle context the test is the great-circle distance in meters
 * (`geodesicHaversineDistance`) between the two samples of the same bucket; there is no
 * segment-closest-approach computation, so the discrete-approximation caveat above is unchanged
 * and no tangent-plane approximation is involved. Candidate pruning is conservative in degrees:
 * two points within angular distance `theta` of each other, one at latitude at most `L`, differ in
 * latitude by at most `theta` and in longitude by at most `asin(sin(theta) / cos(L))` (for
 * `L + theta < 90 degrees`). With `L = max(|minY|, |maxY|)` of `bounds`, the largest searchable
 * distance is therefore `maximumDistance = radius * asin(sin(cellSize) * cos(L))` (scaled by
 * `1 - 1e-6` for rounding) and `distance` is clamped to it. Choose latitude bounds that exclude
 * the polar caps, since `maximumDistance` shrinks to 0 as `L` approaches 90 degrees. Reported
 * `minimumDistances` are meters.
 *
 * Bounds: `pairs.candidateOverflow` is 1 when the hit scratch or grid index overflowed, after
 * which pairs may be missing or have too-small counts. `pairs.output.overflow` is independently 1
 * only when the final pair capacity is too small. The lattice must hold at most 2^27 cells.
 */
export class GPUTrajectoryEncounters implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTrajectoryEncountersProps;
  /** Lattice columns. */
  readonly columns: number;
  /** Lattice rows. */
  readonly rows: number;
  /** Resolved coordinate and metric context; native planar when `props.spatialContext` is omitted. */
  readonly spatialContext: GPUSpatialContext;
  /**
   * Largest `distance` the lattice searches conservatively, in the units of `distance`: `cellSize`
   * for a planar context, meters for a great-circle context. Larger per-frame distances are clamped.
   */
  readonly maximumDistance: number;
  /** Great-circle lattice parameters, or `undefined` for a planar context. */
  private readonly geodesic?: {radius: number; maximumDistance: number; wrapColumns: boolean};

  constructor(props: GPUTrajectoryEncountersProps) {
    this.id = props.id ?? 'trajectory-encounters';
    this.props = props;
    const {id} = this;
    this.spatialContext = resolveTrajectorySpatialContext(id, props.spatialContext);
    const geodesicRadius = getTrajectoryGeodesicRadius(this.spatialContext);
    for (const [name, value] of [
      ['trackCount', props.trackCount],
      ['bucketCount', props.bucketCount],
      ['hitCapacity', props.hitCapacity]
    ] as const) {
      if (!Number.isSafeInteger(value) || value < 1) {
        throw new Error(`${id} ${name} must be a positive integer`);
      }
    }
    const sampleCount = props.trackCount * props.bucketCount;
    if (sampleCount >= 0x7fffffff) {
      throw new Error(`${id} trackCount * bucketCount must be below 2^31`);
    }
    validatePackedView(props.samples, ['float32x2'], `${id} samples`);
    if (props.samples.length !== sampleCount) {
      throw new Error(`${id} samples length must equal trackCount * bucketCount`);
    }
    if (props.trackValid) {
      validatePackedUint32View(props.trackValid, `${id} trackValid`);
      if (props.trackValid.length !== props.trackCount) {
        throw new Error(`${id} trackValid length must equal trackCount`);
      }
    }
    validatePackedView(props.distance, ['float32'], `${id} distance`);
    if (props.distance.length < 1) {
      throw new Error(`${id} distance must contain one float32 row`);
    }
    if (!(Number.isFinite(props.cellSize) && props.cellSize > 0)) {
      throw new Error(`${id} cellSize must be a positive finite number`);
    }
    const [minX, minY, maxX, maxY] = props.bounds;
    if (
      !props.bounds.every(Number.isFinite) ||
      !(Math.fround(minX) < Math.fround(maxX)) ||
      !(Math.fround(minY) < Math.fround(maxY))
    ) {
      throw new Error(`${id} bounds must be finite with minima below maxima`);
    }
    if (geodesicRadius === undefined) {
      this.maximumDistance = props.cellSize;
    } else {
      const maximumLatitude = Math.max(Math.abs(minY), Math.abs(maxY));
      if (minX < -180 || maxX > 180 || maximumLatitude >= 90) {
        throw new Error(
          `${id} great-circle bounds need longitudes within [-180, 180] and latitudes within (-90, 90)`
        );
      }
      const wrapColumns = minX === -180 && maxX === 180;
      if (props.cellSize >= (wrapColumns ? 120 : 90)) {
        throw new Error(
          `${id} great-circle cellSize must be below ${wrapColumns ? 120 : 90} degrees`
        );
      }
      const cellRadians = (props.cellSize * Math.PI) / 180;
      const latitudeRadians = (maximumLatitude * Math.PI) / 180;
      // Conservative: the largest angle whose longitude reach at latitude L fits one cell.
      const maximumAngle = Math.asin(Math.sin(cellRadians) * Math.cos(latitudeRadians));
      this.maximumDistance = geodesicRadius * maximumAngle * (1 - 1e-6);
      this.geodesic = {radius: geodesicRadius, maximumDistance: this.maximumDistance, wrapColumns};
    }
    this.columns = Math.max(
      1,
      Math.floor((Math.fround(maxX) - Math.fround(minX)) / props.cellSize)
    );
    this.rows = Math.max(1, Math.floor((Math.fround(maxY) - Math.fround(minY)) / props.cellSize));
    if (this.columns * this.rows * props.bucketCount > MAXIMUM_CELL_COUNT) {
      throw new Error(`${id} lattice exceeds ${MAXIMUM_CELL_COUNT} cells; raise cellSize`);
    }
    const {pairs} = props;
    validateCompactOutput(id, pairs.output);
    validatePackedUint32View(pairs.candidateOverflow, `${id} pairs.candidateOverflow`);
    if (pairs.candidateOverflow.length < 1) {
      throw new Error(`${id} pairs.candidateOverflow must contain one uint32 row`);
    }
    const capacity = pairs.output.ids.length;
    for (const [name, view, format] of [
      ['partners', pairs.partners, 'uint32'],
      ['firstBuckets', pairs.firstBuckets, 'uint32'],
      ['minimumDistances', pairs.minimumDistances, 'float32'],
      ['bucketCounts', pairs.bucketCounts, 'uint32'],
      ['firstTimes', pairs.firstTimes, 'float32']
    ] as const) {
      if (view) {
        validatePackedView(view, [format], `${id} pairs.${name}`);
        if (view.length !== capacity) {
          throw new Error(`${id} pairs.${name} length must equal the pair capacity`);
        }
      }
    }
    if (pairs.firstTimes && !(props.bucketTimes && pairs.firstBuckets)) {
      throw new Error(`${id} pairs.firstTimes requires bucketTimes and pairs.firstBuckets`);
    }
    if (props.bucketTimes) {
      validatePackedView(props.bucketTimes, ['float32'], `${id} bucketTimes`);
      if (props.bucketTimes.length !== props.bucketCount) {
        throw new Error(`${id} bucketTimes length must equal bucketCount`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        pairs.output.ids,
        pairs.output.count,
        pairs.output.overflow,
        pairs.output.requiredCount,
        pairs.candidateOverflow,
        pairs.partners,
        pairs.firstBuckets,
        pairs.minimumDistances,
        pairs.bucketCounts,
        pairs.firstTimes
      ],
      [props.samples, props.trackValid, props.distance, props.bucketTimes]
    );
  }

  /** Returns the grid build, scan, sort, run detection and reduction nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, columns, rows} = this;
    const {pairs} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.samples,
      props.trackValid,
      props.distance,
      props.bucketTimes,
      pairs.output.ids,
      pairs.output.count,
      pairs.output.overflow,
      pairs.output.requiredCount,
      pairs.candidateOverflow,
      pairs.partners,
      pairs.firstBuckets,
      pairs.minimumDistances,
      pairs.bucketCounts,
      pairs.firstTimes
    ]);
    const bounds = props.bounds.map(Math.fround) as [number, number, number, number];
    const shape: EncounterShape = {
      trackCount: props.trackCount,
      bucketCount: props.bucketCount,
      hitCapacity: props.hitCapacity,
      pairCapacity: pairs.output.ids.length,
      columns,
      rows,
      bounds,
      cellSize: props.cellSize,
      geodesic: this.geodesic
    };
    const sampleCount = props.trackCount * props.bucketCount;
    const capacity = props.hitCapacity;
    const u32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', Math.max(length, 1));
    const f32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'float32', Math.max(length, 1));
    const nodes: GPUCommandNode<Parameters>[] = [];

    const positions = createTransientView(graph, `${id}-positions`, 'float32x3', sampleCount);
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'cell-offsets',
      (columns * rows * props.bucketCount + 1) * 4,
      'raise cellSize or lower bucketCount'
    );
    const cellOffsets = u32('cell-offsets', columns * rows * props.bucketCount + 1);
    const objectIds = u32('object-ids', sampleCount);
    const gridCount = u32('grid-count', 1);
    const gridOverflow = u32('grid-overflow', 1);
    const state = u32('state', 1);
    const hitPairs = createTransientView(graph, `${id}-hit-pairs`, 'uint32x2', capacity);
    const hitBuckets = u32('hit-buckets', capacity);
    const hitDistances = f32('hit-distances', capacity);
    nodes.push(
      createEncounterPositionsNode<Parameters>(graph, {
        id: `${id}-positions`,
        shape,
        samples: props.samples,
        trackValid: props.trackValid,
        positions
      }),
      ...new GPUGridIndex({
        id: `${id}-grid`,
        positions,
        gridSize: [columns, rows, props.bucketCount],
        bounds: [bounds[0], bounds[1], 0, bounds[2], bounds[3], props.bucketCount],
        cellOffsets,
        objectIds,
        count: gridCount,
        overflow: gridOverflow
      }).getCommandNodes(graph),
      createEncounterClearNode<Parameters>(graph, {id: `${id}-clear`, state}),
      createEncounterScanNode<Parameters>(graph, {
        id: `${id}-scan`,
        shape,
        positions,
        cellOffsets,
        objectIds,
        distance: props.distance,
        state,
        hitPairs,
        hitBuckets,
        hitDistances
      })
    );

    // Sort hits into (track, partner, bucket) order. When the three fields fit one 32-bit key, a
    // single radix sort of the packed key replaces the three-pass stable LSD chain (same total key
    // bits, but one key kernel, one sort setup and no per-field gathers). Larger problems fall
    // back to the chain: bucket, then partner, then track.
    const keyBits: EncounterKeyBits = {
      track: getKeyBits(props.trackCount),
      partner: getKeyBits(props.trackCount - 1),
      bucket: getKeyBits(props.bucketCount - 1)
    };
    const isComposite = keyBits.track + keyBits.partner + keyBits.bucket <= 32;
    const identity = u32('identity', capacity);
    const sortedTracks = u32('sorted-tracks', capacity);
    const order = u32('order', capacity);
    let sortedKeys: GraphDataView<'uint32'> | undefined;
    if (isComposite) {
      const compositeKeys = u32('composite-keys', capacity);
      sortedKeys = u32('composite-sorted-keys', capacity);
      nodes.push(
        createEncounterCompositeKeyNode<Parameters>(graph, {
          id: `${id}-composite-keys`,
          shape,
          bits: keyBits,
          state,
          hitPairs,
          hitBuckets,
          keys: compositeKeys,
          identity
        }),
        ...new GPUSort({
          id: `${id}-composite-sort`,
          keys: compositeKeys,
          values: identity,
          outputKeys: sortedKeys,
          outputValues: order,
          keyBits: keyBits.track + keyBits.partner + keyBits.bucket
        }).getCommandNodes(graph)
      );
    } else {
      const bucketKeys = u32('bucket-keys', capacity);
      const bucketSortedKeys = u32('bucket-sorted-keys', capacity);
      const bucketOrder = u32('bucket-order', capacity);
      const partnerKeys = u32('partner-keys', capacity);
      const partnerSortedKeys = u32('partner-sorted-keys', capacity);
      const partnerOrder = u32('partner-order', capacity);
      const trackKeys = u32('track-keys', capacity);
      nodes.push(
        createEncounterSortKeyNode<Parameters>(graph, {
          id: `${id}-bucket-keys`,
          shape,
          variant: 'bucket',
          state,
          hitPairs,
          hitBuckets,
          keys: bucketKeys,
          identity
        }),
        ...new GPUSort({
          id: `${id}-bucket-sort`,
          keys: bucketKeys,
          values: identity,
          outputKeys: bucketSortedKeys,
          outputValues: bucketOrder,
          keyBits: keyBits.bucket
        }).getCommandNodes(graph),
        createEncounterSortKeyNode<Parameters>(graph, {
          id: `${id}-partner-keys`,
          shape,
          variant: 'partner',
          state,
          order: bucketOrder,
          hitPairs,
          keys: partnerKeys
        }),
        ...new GPUSort({
          id: `${id}-partner-sort`,
          keys: partnerKeys,
          values: bucketOrder,
          outputKeys: partnerSortedKeys,
          outputValues: partnerOrder,
          keyBits: keyBits.partner
        }).getCommandNodes(graph),
        createEncounterSortKeyNode<Parameters>(graph, {
          id: `${id}-track-keys`,
          shape,
          variant: 'track',
          state,
          order: partnerOrder,
          hitPairs,
          keys: trackKeys
        }),
        ...new GPUSort({
          id: `${id}-track-sort`,
          keys: trackKeys,
          values: partnerOrder,
          outputKeys: sortedTracks,
          outputValues: order,
          keyBits: keyBits.track
        }).getCommandNodes(graph)
      );
    }

    const sortedPartners = u32('sorted-partners', capacity);
    const sortedBuckets = u32('sorted-buckets', capacity);
    const sortedDistances = f32('sorted-distances', capacity);
    const runFlags = u32('run-flags', capacity);
    const rowIds = u32('row-ids', capacity);
    const runStarts = u32('run-starts', capacity);
    const runCount = u32('run-count', 1);
    const overflowFlag = u32('overflow-flag', 1);
    nodes.push(
      createEncounterSortedHitsNode<Parameters>(graph, {
        id: `${id}-sorted-hits`,
        shape,
        state,
        order,
        hitPairs: isComposite ? undefined : hitPairs,
        hitBuckets: isComposite ? undefined : hitBuckets,
        hitDistances,
        sortedPartners,
        sortedBuckets,
        sortedDistances,
        composite: sortedKeys ? {sortedKeys, sortedTracks, bits: keyBits} : undefined
      }),
      createEncounterRunFlagsNode<Parameters>(graph, {
        id: `${id}-run-flags`,
        shape,
        state,
        sortedTracks,
        sortedPartners,
        flags: runFlags,
        rowIds
      }),
      ...new GPUCompaction({
        id: `${id}-runs`,
        input: rowIds,
        flags: runFlags,
        output: runStarts,
        count: runCount
      }).getCommandNodes(graph),
      ...createEncounterReduceNodes<Parameters>(graph, {
        id: `${id}-reduce`,
        shape,
        state,
        runStarts,
        runCount,
        sortedTracks,
        sortedPartners,
        sortedBuckets,
        sortedDistances,
        ids: pairs.output.ids,
        partners: pairs.partners,
        firstBuckets: pairs.firstBuckets,
        minimumDistances: pairs.minimumDistances,
        bucketCounts: pairs.bucketCounts,
        bucketTimes: props.bucketTimes,
        firstTimes: pairs.firstTimes
      }),
      createEncounterOverflowNode<Parameters>(graph, {
        id: `${id}-overflow`,
        shape,
        state,
        flag: overflowFlag
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-candidate-overflow`,
        operation: OPERATION,
        variant: 'candidate-overflow',
        bindings: [
          {name: 'hitOverflow', view: overflowFlag, type: 'u32', access: 'read'},
          {name: 'gridOverflow', view: gridOverflow, type: 'u32', access: 'read'},
          {
            name: 'candidateOverflow',
            view: pairs.candidateOverflow,
            type: 'u32',
            access: 'read_write'
          }
        ],
        invocationCount: 1,
        body: `candidateOverflow[candidateOverflowOffset] =
  select(0u, 1u, hitOverflow[hitOverflowOffset] != 0u || gridOverflow[gridOverflowOffset] != 0u);`
      }),
      createPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        requiredCount: runCount,
        output: pairs.output,
        overflowSources: []
      })
    );
    return nodes;
  }
}

/** Bits needed to sort keys in `[0, maximumKey]`. */
function getKeyBits(maximumKey: number): number {
  return Math.max(1, 32 - Math.clz32(Math.max(maximumKey, 1)));
}
