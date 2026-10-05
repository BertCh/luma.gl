// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GraphDataView,
  GraphVectorView
} from '@luma.gl/gpgpu/gpu-core';
import {getGraphViewChunks} from '../../utils/gpu-contributor-utils';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {GPU_POINT_DENSITY_HEXAGON_WGSL} from '../../gpu-spatial-analysis/point-density/point-density-hexagon';

const OPERATION = 'GPUFlowAggregation';
const SEGMENT_WORKGROUP_SIZE = 256;

/** Bounds resolved to either a literal or one GPU `float32x4` row. @internal */
export type FlowResolvedBounds =
  | readonly [number, number, number, number]
  | GraphDataView<'float32x4'>;

/**
 * Writes one zone ID per position of a chunk into `zones[zoneStart + index]`, or the no-zone
 * marker for rejected rows. Grid cells mirror `GPUGridBinning`; hexagons mirror point-density.
 *
 * @internal
 */
export function createFlowZoneNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    kind: 'grid' | 'hexagon';
    positions: GraphDataView;
    zones: GraphDataView<'uint32'>;
    zoneStart: number;
    gridSize: readonly [number, number];
    bounds: FlowResolvedBounds;
    radius?: number | GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
    {name: 'zones', view: props.zones, type: 'u32', access: 'read_write'}
  ];
  const boundsIsView = !Array.isArray(props.bounds);
  if (boundsIsView) {
    bindings.push({
      name: 'boundsValues',
      view: props.bounds as GraphDataView,
      type: 'f32',
      access: 'read'
    });
  }
  if (props.kind === 'hexagon' && typeof props.radius !== 'number' && props.radius) {
    bindings.push({
      name: 'radiusValues',
      view: props.radius,
      type: 'f32',
      access: 'read'
    });
  }
  const literal = boundsIsView
    ? undefined
    : (props.bounds as readonly number[]).map(getWGSLFloatLiteral);
  const boundsSource = literal
    ? `let minimumX = ${literal[0]}; let minimumY = ${literal[1]};
  let maximumX = ${literal[2]}; let maximumY = ${literal[3]};`
    : `let minimumX = boundsValues[boundsValuesOffset];
  let minimumY = boundsValues[boundsValuesOffset + 1u];
  let maximumX = boundsValues[boundsValuesOffset + 2u];
  let maximumY = boundsValues[boundsValuesOffset + 3u];`;
  const radiusSource =
    typeof props.radius === 'number'
      ? `let radius = ${getWGSLFloatLiteral(props.radius)};`
      : 'let radius = radiusValues[radiusValuesOffset];';
  const zoneSource =
    props.kind === 'hexagon'
      ? `${radiusSource}
  let validRadius = radius > 0.0 && radius <= MAXIMUM_FLOAT;
  if (finite && inside && validRadius) {
    let cell = getPointDensityHexagonCell(x, y, minimumX, minimumY, radius);
    if (cell.x >= 0 && cell.y >= 0 && u32(cell.x) < COLUMNS && u32(cell.y) < ROWS) {
      zone = u32(cell.y) * COLUMNS + u32(cell.x);
    }
  }`
      : `if (finite && inside) {
    zone = getCoordinate(y, minimumY, maximumY, ROWS) * COLUMNS +
      getCoordinate(x, minimumX, maximumX, COLUMNS);
  }`;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: `${props.kind}-zones`,
    bindings,
    invocationCount: props.positions.length,
    declarations: `const ZONE_START: u32 = ${props.zoneStart}u;
const COLUMNS: u32 = ${props.gridSize[0]}u;
const ROWS: u32 = ${props.gridSize[1]}u;
const NO_ZONE: u32 = 0xffffffffu;
const MAXIMUM_FLOAT: f32 = 3.402823466e+38;
${props.kind === 'hexagon' ? GPU_POINT_DENSITY_HEXAGON_WGSL : ''}
// Same cell formula as GPUGridBinning: inclusive bounds, maximum edge in the last cell.
fn getCoordinate(value: f32, minimum: f32, maximum: f32, size: u32) -> u32 {
  if (maximum == minimum || value == minimum) { return 0u; }
  if (value == maximum) { return size - 1u; }
  return min(u32((value - minimum) / (maximum - minimum) * f32(size)), size - 1u);
}`,
    body: `${boundsSource}
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  var zone = NO_ZONE;
  let finite = x == x && y == y && abs(x) <= MAXIMUM_FLOAT && abs(y) <= MAXIMUM_FLOAT;
  let inX = x >= minimumX && x <= maximumX && (maximumX != minimumX || x == minimumX);
  let inY = y >= minimumY && y <= maximumY && (maximumY != minimumY || y == minimumY);
  let inside = maximumX >= minimumX && maximumY >= minimumY && inX && inY;
  ${zoneSource}
  zones[zonesOffset + ZONE_START + index] = zone;`
  });
}

/**
 * Combines zones, mask, and time mask into one pair key per row (`0xffffffff` when rejected) and
 * writes accepted origin and destination zones (no-zone when rejected).
 *
 * @internal
 */
export function createFlowPairKeysNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    originZones: GraphDataView<'uint32'>;
    destinationZones: GraphDataView<'uint32'>;
    mask?: GraphDataView<'uint32'>;
    timeMask?: GraphDataView<'uint32'>;
    pairKeys: GraphDataView<'uint32'>;
    acceptedOriginZones: GraphDataView<'uint32'>;
    acceptedDestinationZones: GraphDataView<'uint32'>;
    zoneCount: number;
    excludeSelfFlows: boolean;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {
      name: 'originZones',
      view: props.originZones,
      type: 'u32',
      access: 'read'
    },
    {
      name: 'destinationZones',
      view: props.destinationZones,
      type: 'u32',
      access: 'read'
    }
  ];
  if (props.mask) {
    bindings.push({
      name: 'rowMask',
      view: props.mask,
      type: 'u32',
      access: 'read'
    });
  }
  if (props.timeMask) {
    bindings.push({
      name: 'timeMask',
      view: props.timeMask,
      type: 'u32',
      access: 'read'
    });
  }
  bindings.push(
    {
      name: 'pairKeys',
      view: props.pairKeys,
      type: 'u32',
      access: 'read_write'
    },
    {
      name: 'acceptedOriginZones',
      view: props.acceptedOriginZones,
      type: 'u32',
      access: 'read_write'
    },
    {
      name: 'acceptedDestinationZones',
      view: props.acceptedDestinationZones,
      type: 'u32',
      access: 'read_write'
    }
  );
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'pair-keys',
    bindings,
    invocationCount: props.pairKeys.length,
    declarations: `const ZONE_COUNT: u32 = ${props.zoneCount}u;
const NO_ZONE: u32 = 0xffffffffu;`,
    body: `let origin = originZones[originZonesOffset + index];
  let destination = destinationZones[destinationZonesOffset + index];
  var accepted = origin < ZONE_COUNT && destination < ZONE_COUNT;
  ${props.excludeSelfFlows ? 'accepted = accepted && origin != destination;' : ''}
  ${props.mask ? 'accepted = accepted && rowMask[rowMaskOffset + index] != 0u;' : ''}
  ${props.timeMask ? 'accepted = accepted && timeMask[timeMaskOffset + index] != 0u;' : ''}
  pairKeys[pairKeysOffset + index] = select(NO_ZONE, origin * ZONE_COUNT + destination, accepted);
  acceptedOriginZones[acceptedOriginZonesOffset + index] = select(NO_ZONE, origin, accepted);
  acceptedDestinationZones[acceptedDestinationZonesOffset + index] =
    select(NO_ZONE, destination, accepted);`
  });
}

/** slotKeys[slot] = tableKeys[slot] and slotIndices[slot] = slot, one row per hash slot. @internal */
export function createFlowSlotKeysNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    tableKeys: GraphDataView<'uint32'>;
    slotKeys: GraphDataView<'uint32'>;
    slotIndices: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'slot-keys',
    bindings: [
      {name: 'tableKeys', view: props.tableKeys, type: 'u32', access: 'read'},
      {
        name: 'slotKeys',
        view: props.slotKeys,
        type: 'u32',
        access: 'read_write'
      },
      {
        name: 'slotIndices',
        view: props.slotIndices,
        type: 'u32',
        access: 'read_write'
      }
    ],
    invocationCount: props.tableKeys.length,
    body: `slotKeys[slotKeysOffset + index] = tableKeys[tableKeysOffset + index];
  slotIndices[slotIndicesOffset + index] = index;`
  });
}

/**
 * Writes one ascending sort key per pair-key-ordered slot such that ascending order is descending
 * weight (or count). The key is `0xffffffff` only for empty slots: counts are at least one and
 * finite weight sums never map to the all-ones ordered pattern.
 *
 * @internal
 */
export function createFlowWeightKeysNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    slotsByPair: GraphDataView<'uint32'>;
    tableKeys: GraphDataView<'uint32'>;
    tableValues: GraphDataView<'uint32'>;
    rowCounts: GraphDataView<'uint32'>;
    rowWeights?: GraphDataView<'float32'>;
    weightKeys: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {
      name: 'slotsByPair',
      view: props.slotsByPair,
      type: 'u32',
      access: 'read'
    },
    {name: 'tableKeys', view: props.tableKeys, type: 'u32', access: 'read'},
    {
      name: 'tableValues',
      view: props.tableValues,
      type: 'u32',
      access: 'read'
    }
  ];
  // Only the binding that the generated WGSL uses may be declared: unused bindings are dropped
  // from the pipeline layout.
  if (props.rowWeights) {
    bindings.push({
      name: 'rowWeights',
      view: props.rowWeights,
      type: 'f32',
      access: 'read'
    });
  } else {
    bindings.push({
      name: 'rowCounts',
      view: props.rowCounts,
      type: 'u32',
      access: 'read'
    });
  }
  bindings.push({
    name: 'weightKeys',
    view: props.weightKeys,
    type: 'u32',
    access: 'read_write'
  });
  const keySource = props.rowWeights
    ? `var weight = rowWeights[rowWeightsOffset + row];
      weight = select(weight, 0.0, weight == 0.0);
      let bits = bitcast<u32>(weight);
      let ordered = bits ^ select(0x80000000u, 0xffffffffu, (bits >> 31u) != 0u);
      key = ~ordered;`
    : 'key = ~rowCounts[rowCountsOffset + row];';
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'weight-keys',
    bindings,
    invocationCount: props.slotsByPair.length,
    body: `let slot = slotsByPair[slotsByPairOffset + index];
  var key = 0xffffffffu;
  if (tableKeys[tableKeysOffset + slot] != 0xffffffffu) {
    let row = tableValues[tableValuesOffset + slot];
    ${keySource}
  }
  weightKeys[weightKeysOffset + index] = key;`
  });
}

/** Writes the top-K pair keys of `output.ids`, sentinel for rows past the count. @internal */
export function createFlowGatherIdsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    statistics: GraphDataView<'uint32'>;
    rankedSlots: GraphDataView<'uint32'>;
    tableKeys: GraphDataView<'uint32'>;
    ids: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'gather-ids',
    bindings: [
      {
        name: 'statistics',
        view: props.statistics,
        type: 'u32',
        access: 'read'
      },
      {
        name: 'rankedSlots',
        view: props.rankedSlots,
        type: 'u32',
        access: 'read'
      },
      {name: 'tableKeys', view: props.tableKeys, type: 'u32', access: 'read'},
      {name: 'ids', view: props.ids, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.ids.length,
    declarations: `const CAPACITY: u32 = ${props.ids.length}u;
const SLOT_COUNT: u32 = ${props.rankedSlots.length}u;`,
    body: `let count = min(statistics[statisticsOffset], min(CAPACITY, SLOT_COUNT));
  var id = 0xffffffffu;
  if (index < count) {
    id = tableKeys[tableKeysOffset + rankedSlots[rankedSlotsOffset + index]];
  }
  ids[idsOffset + index] = id;`
  });
}

/** Writes the top-K flow counts and weights, zero for rows past the count. @internal */
export function createFlowGatherValuesNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    capacity: number;
    statistics: GraphDataView<'uint32'>;
    rankedSlots: GraphDataView<'uint32'>;
    tableValues: GraphDataView<'uint32'>;
    rowCounts: GraphDataView<'uint32'>;
    rowWeights?: GraphDataView<'float32'>;
    flowCounts?: GraphDataView<'uint32'>;
    flowWeights?: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'statistics', view: props.statistics, type: 'u32', access: 'read'},
    {
      name: 'rankedSlots',
      view: props.rankedSlots,
      type: 'u32',
      access: 'read'
    },
    {
      name: 'tableValues',
      view: props.tableValues,
      type: 'u32',
      access: 'read'
    }
  ];
  const usesRowWeights = Boolean(props.rowWeights && props.flowWeights);
  if (props.flowCounts || (props.flowWeights && !usesRowWeights)) {
    bindings.push({
      name: 'rowCounts',
      view: props.rowCounts,
      type: 'u32',
      access: 'read'
    });
  }
  if (props.rowWeights && usesRowWeights) {
    bindings.push({
      name: 'rowWeights',
      view: props.rowWeights,
      type: 'f32',
      access: 'read'
    });
  }
  if (props.flowCounts) {
    bindings.push({
      name: 'flowCounts',
      view: props.flowCounts,
      type: 'u32',
      access: 'read_write'
    });
  }
  if (props.flowWeights) {
    bindings.push({
      name: 'flowWeights',
      view: props.flowWeights,
      type: 'f32',
      access: 'read_write'
    });
  }
  const weightSource = usesRowWeights
    ? 'rowWeights[rowWeightsOffset + row]'
    : 'f32(rowCounts[rowCountsOffset + row])';
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'gather-values',
    bindings,
    invocationCount: props.capacity,
    declarations: `const CAPACITY: u32 = ${props.capacity}u;
const SLOT_COUNT: u32 = ${props.rankedSlots.length}u;`,
    body: `let count = min(statistics[statisticsOffset], min(CAPACITY, SLOT_COUNT));
  var flowCount = 0u;
  var flowWeight = 0.0;
  if (index < count) {
    let row = tableValues[tableValuesOffset + rankedSlots[rankedSlotsOffset + index]];
    ${props.flowCounts ? 'flowCount = rowCounts[rowCountsOffset + row];' : ''}
    ${props.flowWeights ? `flowWeight = ${weightSource};` : ''}
  }
  ${props.flowCounts ? 'flowCounts[flowCountsOffset + index] = flowCount;' : ''}
  ${props.flowWeights ? 'flowWeights[flowWeightsOffset + index] = flowWeight;' : ''}`
  });
}

/** Splits output pair keys into origin and destination zone columns. @internal */
export function createFlowDecodeZonesNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    zoneCount: number;
    ids: GraphDataView<'uint32'>;
    originZoneIds?: GraphDataView<'uint32'>;
    destinationZoneIds?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'ids', view: props.ids, type: 'u32', access: 'read'}
  ];
  if (props.originZoneIds) {
    bindings.push({
      name: 'originOut',
      view: props.originZoneIds,
      type: 'u32',
      access: 'read_write'
    });
  }
  if (props.destinationZoneIds) {
    bindings.push({
      name: 'destinationOut',
      view: props.destinationZoneIds,
      type: 'u32',
      access: 'read_write'
    });
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'decode-zones',
    bindings,
    invocationCount: props.ids.length,
    declarations: `const ZONE_COUNT: u32 = ${props.zoneCount}u;`,
    body: `let key = ids[idsOffset + index];
  let valid = key != 0xffffffffu;
  ${props.originZoneIds ? 'originOut[originOutOffset + index] = select(0xffffffffu, key / ZONE_COUNT, valid);' : ''}
  ${props.destinationZoneIds ? 'destinationOut[destinationOutOffset + index] = select(0xffffffffu, key % ZONE_COUNT, valid);' : ''}`
  });
}

/** pairOverflow[0] = 1 when any accepted row's pair could not be stored. @internal */
export function createFlowPairOverflowNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    statistics: GraphDataView<'uint32'>;
    pairOverflow: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'pair-overflow',
    bindings: [
      {
        name: 'statistics',
        view: props.statistics,
        type: 'u32',
        access: 'read'
      },
      {
        name: 'pairOverflow',
        view: props.pairOverflow,
        type: 'u32',
        access: 'read_write'
      }
    ],
    invocationCount: 1,
    body: 'pairOverflow[pairOverflowOffset] = select(0u, 1u, statistics[statisticsOffset + 2u] != 0u);'
  });
}

/**
 * Writes one finite-or-zero weight per source row into `contributions` (one node per weight chunk),
 * so non-finite weights add 0 to every sorted sum, matching the atomic path that skips them.
 *
 * @internal
 */
export function createFlowContributionNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    weights: GraphDataView<'float32'> | GraphVectorView<'float32'>;
    contributions: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const chunks = getGraphViewChunks(props.weights);
  let rowStart = 0;
  for (const [chunkIndex, chunk] of chunks.entries()) {
    if (chunk.length > 0) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${props.id}-contributions${chunks.length > 1 ? `-chunk-${chunkIndex}` : ''}`,
          operation: OPERATION,
          variant: 'contributions',
          bindings: [
            {name: 'weights', view: chunk, type: 'f32', access: 'read'},
            {
              name: 'contributions',
              view: props.contributions,
              type: 'f32',
              access: 'read_write'
            }
          ],
          invocationCount: chunk.length,
          declarations: `const ROW_START: u32 = ${rowStart}u;`,
          body: `let weight = weights[weightsOffset + index];
  let isFinite = weight == weight && abs(weight) <= 3.402823466e+38;
  contributions[contributionsOffset + ROW_START + index] = select(0.0, weight, isFinite);`
        })
      );
    }
    rowStart += chunk.length;
  }
  return nodes;
}

/** Sort keys for the per-pair sum: representative row (clamped to `rowCount`) and row index. @internal */
export function createFlowPairSortPrepareNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    representativeRows: GraphDataView<'uint32'>;
    sortKeys: GraphDataView<'uint32'>;
    sortIndices: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'pair-sum-sort-prepare',
    bindings: [
      {
        name: 'representativeRows',
        view: props.representativeRows,
        type: 'u32',
        access: 'read'
      },
      {
        name: 'sortKeys',
        view: props.sortKeys,
        type: 'u32',
        access: 'read_write'
      },
      {
        name: 'sortIndices',
        view: props.sortIndices,
        type: 'u32',
        access: 'read_write'
      }
    ],
    invocationCount: props.representativeRows.length,
    declarations: `const ROW_COUNT: u32 = ${props.representativeRows.length}u;`,
    // Rows whose pair was not stored have representative row 0xffffffff and sort last.
    body: `sortKeys[sortKeysOffset + index] = min(representativeRows[representativeRowsOffset + index], ROW_COUNT);
  sortIndices[sortIndicesOffset + index] = index;`
  });
}

/** sortedContributions[i] = contributions[sortedIndices[i]]. @internal */
export function createFlowSortedGatherNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    sortedIndices: GraphDataView<'uint32'>;
    contributions: GraphDataView<'float32'>;
    sortedContributions: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'pair-sum-gather',
    bindings: [
      {
        name: 'sortedIndices',
        view: props.sortedIndices,
        type: 'u32',
        access: 'read'
      },
      {
        name: 'contributions',
        view: props.contributions,
        type: 'f32',
        access: 'read'
      },
      {
        name: 'sortedContributions',
        view: props.sortedContributions,
        type: 'f32',
        access: 'read_write'
      }
    ],
    invocationCount: props.sortedIndices.length,
    body: `sortedContributions[sortedContributionsOffset + index] =
    contributions[contributionsOffset + sortedIndices[sortedIndicesOffset + index]];`
  });
}

/**
 * Sums each stored pair's contributions with one 256-thread workgroup per table slot: a strided
 * per-thread partial sum, then a fixed binary tree (the same order as the zonal-statistics segment
 * sum). The pair's rows are the segment `[rowOffsets[r], rowOffsets[r] + rowCounts[r])` of the
 * sorted contributions, where `r` is the pair's representative row; the sum is written to
 * `rowWeights[r]`. The order depends only on those bounds, so the result is bitwise reproducible.
 *
 * @internal
 */
export function createFlowPairSumNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    tableKeys: GraphDataView<'uint32'>;
    tableValues: GraphDataView<'uint32'>;
    rowCounts: GraphDataView<'uint32'>;
    rowOffsets: GraphDataView<'uint32'>;
    sortedContributions: GraphDataView<'float32'>;
    rowWeights: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'pair-sum',
    bindings: [
      {name: 'tableKeys', view: props.tableKeys, type: 'u32', access: 'read'},
      {
        name: 'tableValues',
        view: props.tableValues,
        type: 'u32',
        access: 'read'
      },
      {name: 'rowCounts', view: props.rowCounts, type: 'u32', access: 'read'},
      {
        name: 'rowOffsets',
        view: props.rowOffsets,
        type: 'u32',
        access: 'read'
      },
      {
        name: 'sortedContributions',
        view: props.sortedContributions,
        type: 'f32',
        access: 'read'
      },
      {
        name: 'rowWeights',
        view: props.rowWeights,
        type: 'f32',
        access: 'read_write'
      }
    ],
    workgroupSize: SEGMENT_WORKGROUP_SIZE,
    invocationCount: props.tableKeys.length * SEGMENT_WORKGROUP_SIZE,
    guardIndex: false,
    declarations: `const INPUT_COUNT: u32 = ${props.sortedContributions.length}u;
var<workgroup> partialSums: array<f32, ${SEGMENT_WORKGROUP_SIZE}>;`,
    // No early return: every invocation of a workgroup must reach the barriers.
    body: `let slot = index / ${SEGMENT_WORKGROUP_SIZE}u;
  let isInRange = index < INVOCATION_COUNT;
  var partialSum = 0.0;
  var row = 0u;
  var isStored = false;
  if (isInRange && tableKeys[tableKeysOffset + slot] != 0xffffffffu) {
    isStored = true;
    row = tableValues[tableValuesOffset + slot];
    let begin = rowOffsets[rowOffsetsOffset + row];
    let end = min(begin + rowCounts[rowCountsOffset + row], INPUT_COUNT);
    for (var position = begin + localInvocationIndex; position < end; position += ${SEGMENT_WORKGROUP_SIZE}u) {
      partialSum += sortedContributions[sortedContributionsOffset + position];
    }
  }
  partialSums[localInvocationIndex] = partialSum;
  workgroupBarrier();
  for (var stride = ${SEGMENT_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (isStored && localInvocationIndex == 0u) {
    rowWeights[rowWeightsOffset + row] = partialSums[0];
  }`
  });
}
