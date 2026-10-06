// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {SPATIAL_JOIN_WGSL_HELPERS} from '../spatial-join/spatial-join-passes';

const OPERATION = 'GPUTrajectoryEncounters';

/** Sizes and lattice shared by the encounter kernels. @internal */
export type EncounterShape = {
  trackCount: number;
  bucketCount: number;
  hitCapacity: number;
  pairCapacity: number;
  /** Lattice columns and rows of the `(x, y)` grid; the third axis is the bucket. */
  columns: number;
  rows: number;
  /** f32-rounded domain `[minX, minY, maxX, maxY]`. */
  bounds: readonly [number, number, number, number];
  cellSize: number;
};

function getShapeConstants(shape: EncounterShape): string {
  const [minX, minY, maxX, maxY] = shape.bounds.map(getWGSLFloatLiteral);
  return `const TRACK_COUNT: u32 = ${shape.trackCount}u;
const BUCKET_COUNT: u32 = ${shape.bucketCount}u;
const HIT_CAPACITY: u32 = ${shape.hitCapacity}u;
const PAIR_CAPACITY: u32 = ${shape.pairCapacity}u;
const COLUMNS: u32 = ${shape.columns}u;
const ROWS: u32 = ${shape.rows}u;
const MINIMUM_X: f32 = ${minX};
const MINIMUM_Y: f32 = ${minY};
const MAXIMUM_X: f32 = ${maxX};
const MAXIMUM_Y: f32 = ${maxY};
const CELL_SIZE: f32 = ${getWGSLFloatLiteral(shape.cellSize)};`;
}

/** Builds the `(x, y, bucket + 0.5)` positions of the 3D grid. Invalid tracks become NaN. @internal */
export function createEncounterPositionsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: EncounterShape;
    samples: GraphDataView<'float32x2'>;
    trackValid?: GraphDataView<'uint32'>;
    positions: GraphDataView<'float32x3'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'samples', view: props.samples, type: 'f32', access: 'read'}
  ];
  if (props.trackValid) {
    bindings.push({name: 'trackValid', view: props.trackValid, type: 'u32', access: 'read'});
  }
  bindings.push({name: 'positions', view: props.positions, type: 'f32', access: 'read_write'});
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'positions',
    bindings,
    invocationCount: props.shape.trackCount * props.shape.bucketCount,
    declarations: getShapeConstants(props.shape),
    body: `let track = index / BUCKET_COUNT;
  let bucket = index % BUCKET_COUNT;
  var x = samples[samplesOffset + index * 2u];
  var y = samples[samplesOffset + index * 2u + 1u];
  ${
    props.trackValid
      ? `if (trackValid[trackValidOffset + track] == 0u) {
    // A runtime expression keeps the quiet-NaN bit pattern out of const evaluation.
    x = bitcast<f32>(0x7fc00000u | (index & 0u));
    y = x;
  }`
      : ''
  }
  positions[positionsOffset + index * 3u] = x;
  positions[positionsOffset + index * 3u + 1u] = y;
  positions[positionsOffset + index * 3u + 2u] = f32(bucket) + 0.5;`
  });
}

/**
 * One invocation per sample: scans the 3x3 xy cells of its own bucket and appends a hit for every
 * higher-ID track within the distance. Appended order is unspecified. @internal
 */
export function createEncounterScanNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: EncounterShape;
    positions: GraphDataView<'float32x3'>;
    cellOffsets: GraphDataView<'uint32'>;
    objectIds: GraphDataView<'uint32'>;
    distance: GraphDataView<'float32'>;
    state: GraphDataView<'uint32'>;
    hitPairs: GraphDataView<'uint32x2'>;
    hitBuckets: GraphDataView<'uint32'>;
    hitDistances: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'scan',
    bindings: [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'cellOffsets', view: props.cellOffsets, type: 'u32', access: 'read'},
      {name: 'objectIds', view: props.objectIds, type: 'u32', access: 'read'},
      {name: 'distance', view: props.distance, type: 'f32', access: 'read'},
      {name: 'state', view: props.state, type: 'atomic<u32>', access: 'read_write'},
      {name: 'hitPairs', view: props.hitPairs, type: 'u32', access: 'read_write'},
      {name: 'hitBuckets', view: props.hitBuckets, type: 'u32', access: 'read_write'},
      {name: 'hitDistances', view: props.hitDistances, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.shape.trackCount * props.shape.bucketCount,
    declarations: `${SPATIAL_JOIN_WGSL_HELPERS}
${getShapeConstants(props.shape)}
// Same rule as the GPUGridIndex build, so a point and its query agree on the cell.
fn getCoordinate(value: f32, minimum: f32, maximum: f32, size: u32) -> u32 {
  if (maximum == minimum || value == minimum) { return 0u; }
  if (value == maximum) { return size - 1u; }
  if (minimum < 0.0 && maximum > 0.0) {
    let scale = max(abs(minimum), abs(maximum));
    let scaledValue = value / scale;
    let scaledMinimum = minimum / scale;
    let scaledMaximum = maximum / scale;
    return min(u32((scaledValue - scaledMinimum) / (scaledMaximum - scaledMinimum) * f32(size)), size - 1u);
  }
  return min(u32((value - minimum) / (maximum - minimum) * f32(size)), size - 1u);
}`,
    body: `let track = index / BUCKET_COUNT;
  let bucket = index % BUCKET_COUNT;
  let x = positions[positionsOffset + index * 3u];
  let y = positions[positionsOffset + index * 3u + 1u];
  if (!isFiniteValue(x) || !isFiniteValue(y)) { return; }
  if (x < MINIMUM_X || x > MAXIMUM_X || y < MINIMUM_Y || y > MAXIMUM_Y) { return; }
  let searchDistance = distance[distanceOffset];
  if (!(searchDistance >= 0.0)) { return; }
  let limit = min(searchDistance, CELL_SIZE);
  let column = getCoordinate(x, MINIMUM_X, MAXIMUM_X, COLUMNS);
  let row = getCoordinate(y, MINIMUM_Y, MAXIMUM_Y, ROWS);
  let firstColumn = select(column - 1u, 0u, column == 0u);
  let lastColumn = min(column + 1u, COLUMNS - 1u);
  let firstRow = select(row - 1u, 0u, row == 0u);
  let lastRow = min(row + 1u, ROWS - 1u);
  for (var neighborRow = firstRow; neighborRow <= lastRow; neighborRow++) {
    for (var neighborColumn = firstColumn; neighborColumn <= lastColumn; neighborColumn++) {
      let cell = (bucket * ROWS + neighborRow) * COLUMNS + neighborColumn;
      let cellBegin = cellOffsets[cellOffsetsOffset + cell];
      let cellEnd = cellOffsets[cellOffsetsOffset + cell + 1u];
      for (var entry = cellBegin; entry < cellEnd; entry++) {
        let other = objectIds[objectIdsOffset + entry];
        let otherTrack = other / BUCKET_COUNT;
        if (otherTrack <= track) { continue; }
        let offset = vec2f(
          positions[positionsOffset + other * 3u] - x,
          positions[positionsOffset + other * 3u + 1u] - y
        );
        let separation = length(offset);
        if (separation <= limit) {
          let slot = atomicAdd(&state[stateOffset], 1u);
          if (slot < HIT_CAPACITY) {
            hitPairs[hitPairsOffset + slot * 2u] = track;
            hitPairs[hitPairsOffset + slot * 2u + 1u] = otherTrack;
            hitBuckets[hitBucketsOffset + slot] = bucket;
            hitDistances[hitDistancesOffset + slot] = separation;
          }
        }
      }
    }
  }`
  });
}

/** Zeroes the hit counter. @internal */
export function createEncounterClearNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {id: string; state: GraphDataView<'uint32'>}
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'clear',
    bindings: [{name: 'state', view: props.state, type: 'u32', access: 'read_write'}],
    invocationCount: 1,
    body: 'state[stateOffset] = 0u;'
  });
}

/** Writes the sort key columns of the stable LSD chain (bucket, then partner, then first track). @internal */
export function createEncounterSortKeyNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: EncounterShape;
    variant: 'bucket' | 'partner' | 'track';
    state: GraphDataView<'uint32'>;
    order?: GraphDataView<'uint32'>;
    hitPairs: GraphDataView<'uint32x2'>;
    hitBuckets?: GraphDataView<'uint32'>;
    keys: GraphDataView<'uint32'>;
    identity?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'state', view: props.state, type: 'u32', access: 'read'},
    {name: 'hitPairs', view: props.hitPairs, type: 'u32', access: 'read'}
  ];
  let body: string;
  if (props.variant === 'bucket') {
    bindings.push(
      {name: 'hitBuckets', view: props.hitBuckets!, type: 'u32', access: 'read'},
      {name: 'identity', view: props.identity!, type: 'u32', access: 'read_write'}
    );
    body = `identity[identityOffset + index] = index;
  keys[keysOffset + index] = select(0u, hitBuckets[hitBucketsOffset + index], index < activeCount);`;
  } else {
    bindings.push({name: 'order', view: props.order!, type: 'u32', access: 'read'});
    const component = props.variant === 'partner' ? 1 : 0;
    const sentinel = props.variant === 'partner' ? '0u' : 'TRACK_COUNT';
    body = `let slot = order[orderOffset + index];
  keys[keysOffset + index] = select(${sentinel}, hitPairs[hitPairsOffset + slot * 2u + ${component}u], slot < activeCount);`;
  }
  bindings.push({name: 'keys', view: props.keys, type: 'u32', access: 'read_write'});
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: `sort-key-${props.variant}`,
    bindings,
    invocationCount: props.shape.hitCapacity,
    declarations: getShapeConstants(props.shape),
    body: `let activeCount = min(state[stateOffset], HIT_CAPACITY);
  ${body}`
  });
}

/** Gathers the sorted hit columns. Slots at and after the hit count get sentinels. @internal */
export function createEncounterSortedHitsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: EncounterShape;
    state: GraphDataView<'uint32'>;
    order: GraphDataView<'uint32'>;
    hitPairs: GraphDataView<'uint32x2'>;
    hitBuckets: GraphDataView<'uint32'>;
    hitDistances: GraphDataView<'float32'>;
    sortedPartners: GraphDataView<'uint32'>;
    sortedBuckets: GraphDataView<'uint32'>;
    sortedDistances: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'sorted-hits',
    bindings: [
      {name: 'state', view: props.state, type: 'u32', access: 'read'},
      {name: 'order', view: props.order, type: 'u32', access: 'read'},
      {name: 'hitPairs', view: props.hitPairs, type: 'u32', access: 'read'},
      {name: 'hitBuckets', view: props.hitBuckets, type: 'u32', access: 'read'},
      {name: 'hitDistances', view: props.hitDistances, type: 'f32', access: 'read'},
      {name: 'sortedPartners', view: props.sortedPartners, type: 'u32', access: 'read_write'},
      {name: 'sortedBuckets', view: props.sortedBuckets, type: 'u32', access: 'read_write'},
      {name: 'sortedDistances', view: props.sortedDistances, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.shape.hitCapacity,
    declarations: getShapeConstants(props.shape),
    body: `if (index >= min(state[stateOffset], HIT_CAPACITY)) {
    sortedPartners[sortedPartnersOffset + index] = 0xffffffffu;
    sortedBuckets[sortedBucketsOffset + index] = 0xffffffffu;
    sortedDistances[sortedDistancesOffset + index] = 0.0;
    return;
  }
  let slot = order[orderOffset + index];
  sortedPartners[sortedPartnersOffset + index] = hitPairs[hitPairsOffset + slot * 2u + 1u];
  sortedBuckets[sortedBucketsOffset + index] = hitBuckets[hitBucketsOffset + slot];
  sortedDistances[sortedDistancesOffset + index] = hitDistances[hitDistancesOffset + slot];`
  });
}

/** Flags the first hit of each `(track, partner)` run and writes identity rows. @internal */
export function createEncounterRunFlagsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: EncounterShape;
    state: GraphDataView<'uint32'>;
    sortedTracks: GraphDataView<'uint32'>;
    sortedPartners: GraphDataView<'uint32'>;
    flags: GraphDataView<'uint32'>;
    rowIds: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'run-flags',
    bindings: [
      {name: 'state', view: props.state, type: 'u32', access: 'read'},
      {name: 'sortedTracks', view: props.sortedTracks, type: 'u32', access: 'read'},
      {name: 'sortedPartners', view: props.sortedPartners, type: 'u32', access: 'read'},
      {name: 'flags', view: props.flags, type: 'u32', access: 'read_write'},
      {name: 'rowIds', view: props.rowIds, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.shape.hitCapacity,
    declarations: getShapeConstants(props.shape),
    body: `rowIds[rowIdsOffset + index] = index;
  var isStart = index < min(state[stateOffset], HIT_CAPACITY);
  if (isStart && index > 0u) {
    isStart = sortedTracks[sortedTracksOffset + index] != sortedTracks[sortedTracksOffset + index - 1u]
      || sortedPartners[sortedPartnersOffset + index] != sortedPartners[sortedPartnersOffset + index - 1u];
  }
  flags[flagsOffset + index] = select(0u, 1u, isStart);`
  });
}

/** Run reduction inputs and outputs for {@link createEncounterReduceNodes}. @internal */
export type EncounterReduceProps = {
  id: string;
  shape: EncounterShape;
  state: GraphDataView<'uint32'>;
  runStarts: GraphDataView<'uint32'>;
  runCount: GraphDataView<'uint32'>;
  sortedTracks: GraphDataView<'uint32'>;
  sortedPartners: GraphDataView<'uint32'>;
  sortedBuckets: GraphDataView<'uint32'>;
  sortedDistances: GraphDataView<'float32'>;
  ids: GraphDataView<'uint32'>;
  partners?: GraphDataView<'uint32'>;
  firstBuckets?: GraphDataView<'uint32'>;
  minimumDistances?: GraphDataView<'float32'>;
  bucketCounts?: GraphDataView<'uint32'>;
  bucketTimes?: GraphDataView<'float32'>;
  firstTimes?: GraphDataView<'float32'>;
};

/** Writes the per-pair output columns from the sorted runs. Every slot is rewritten. @internal */
export function createEncounterReduceNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: EncounterReduceProps
): GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const declarations = getShapeConstants(props.shape);
  const live = `let isLive = index < min(runCount[runCountOffset], PAIR_CAPACITY);
  let start = select(0u, runStarts[runStartsOffset + index], isLive);`;
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-identity`,
      operation: OPERATION,
      variant: 'reduce-identity',
      bindings: [
        {name: 'runStarts', view: props.runStarts, type: 'u32', access: 'read'},
        {name: 'runCount', view: props.runCount, type: 'u32', access: 'read'},
        {name: 'sortedTracks', view: props.sortedTracks, type: 'u32', access: 'read'},
        {name: 'sortedPartners', view: props.sortedPartners, type: 'u32', access: 'read'},
        {name: 'ids', view: props.ids, type: 'u32', access: 'read_write'},
        ...(props.partners
          ? [
              {
                name: 'partners',
                view: props.partners,
                type: 'u32' as const,
                access: 'read_write' as const
              }
            ]
          : [])
      ],
      invocationCount: props.shape.pairCapacity,
      declarations,
      body: `${live}
  ids[idsOffset + index] = select(0xffffffffu, sortedTracks[sortedTracksOffset + start], isLive);
  ${props.partners ? 'partners[partnersOffset + index] = select(0xffffffffu, sortedPartners[sortedPartnersOffset + start], isLive);' : ''}`
    })
  );
  if (props.firstBuckets || props.minimumDistances || props.bucketCounts) {
    const bindings: WGSLKernelBinding[] = [
      {name: 'state', view: props.state, type: 'u32', access: 'read'},
      {name: 'runStarts', view: props.runStarts, type: 'u32', access: 'read'},
      {name: 'runCount', view: props.runCount, type: 'u32', access: 'read'},
      {name: 'sortedBuckets', view: props.sortedBuckets, type: 'u32', access: 'read'},
      {name: 'sortedDistances', view: props.sortedDistances, type: 'f32', access: 'read'}
    ];
    for (const [name, view] of [
      ['firstBuckets', props.firstBuckets],
      ['minimumDistances', props.minimumDistances],
      ['bucketCounts', props.bucketCounts]
    ] as const) {
      if (view) {
        bindings.push({
          name,
          view,
          type: name === 'minimumDistances' ? 'f32' : 'u32',
          access: 'read_write'
        });
      }
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${props.id}-statistics`,
        operation: OPERATION,
        variant: 'reduce-statistics',
        bindings,
        invocationCount: props.shape.pairCapacity,
        declarations,
        body: `${live}
  let hitTotal = min(state[stateOffset], HIT_CAPACITY);
  var end = start;
  if (isLive) {
    end = hitTotal;
    if (index + 1u < min(runCount[runCountOffset], PAIR_CAPACITY)) { end = runStarts[runStartsOffset + index + 1u]; }
  }
  // Hits of one pair are in bucket order, so the run start is the first encounter.
  var minimumDistance = sortedDistances[sortedDistancesOffset + start];
  for (var hit = start + 1u; hit < end; hit++) {
    minimumDistance = min(minimumDistance, sortedDistances[sortedDistancesOffset + hit]);
  }
  ${props.firstBuckets ? 'firstBuckets[firstBucketsOffset + index] = select(0xffffffffu, sortedBuckets[sortedBucketsOffset + start], isLive);' : ''}
  ${props.minimumDistances ? 'minimumDistances[minimumDistancesOffset + index] = select(0.0, minimumDistance, isLive);' : ''}
  ${props.bucketCounts ? 'bucketCounts[bucketCountsOffset + index] = select(0u, end - start, isLive);' : ''}`
      })
    );
  }
  if (props.firstTimes && props.bucketTimes && props.firstBuckets) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${props.id}-first-times`,
        operation: OPERATION,
        variant: 'first-times',
        bindings: [
          {name: 'firstBuckets', view: props.firstBuckets, type: 'u32', access: 'read'},
          {name: 'bucketTimes', view: props.bucketTimes, type: 'f32', access: 'read'},
          {name: 'firstTimes', view: props.firstTimes, type: 'f32', access: 'read_write'}
        ],
        invocationCount: props.shape.pairCapacity,
        declarations,
        body: `let bucket = firstBuckets[firstBucketsOffset + index];
  firstTimes[firstTimesOffset + index] = select(0.0, bucketTimes[bucketTimesOffset + min(bucket, BUCKET_COUNT - 1u)], bucket < BUCKET_COUNT);`
      })
    );
  }
  return nodes;
}

/** Writes the overflow flag: hit scratch exceeded. @internal */
export function createEncounterOverflowNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: EncounterShape;
    state: GraphDataView<'uint32'>;
    flag: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'overflow',
    bindings: [
      {name: 'state', view: props.state, type: 'u32', access: 'read'},
      {name: 'flag', view: props.flag, type: 'u32', access: 'read_write'}
    ],
    invocationCount: 1,
    declarations: getShapeConstants(props.shape),
    body: 'flag[flagOffset] = select(0u, 1u, state[stateOffset] > HIT_CAPACITY);'
  });
}
