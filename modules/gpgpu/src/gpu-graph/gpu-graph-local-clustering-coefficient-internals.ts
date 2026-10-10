// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileComment: Independently implemented for WebGPU; inspired by NVIDIA RAPIDS cuGraph.

import {type Binding} from '@luma.gl/core';
import {Computation} from '@luma.gl/engine';
import type {GPUCommandGraph, GraphBufferUse, GraphDataView} from '../gpu-core/gpu-command-graph';
import {
  type GPUBoundedDispatchLayout,
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource
} from '../gpu-core/gpu-dispatch-utils';
import {
  createTransientView,
  getViewBinding,
  getViewElementOffset
} from '../gpu-core/graph-data-view-utils';
import {GPUScan} from '../gpu-core/gpu-scan';
import {GPUSort} from '../gpu-core/gpu-sort';
import type {GPUGraphLocalClusteringCoefficient} from './gpu-graph-local-clustering-coefficient';
import {addDirectLocalClusteringToGraph} from './gpu-graph-local-clustering-coefficient-direct';

const LOCAL_CLUSTERING_WORKGROUP_SIZE = 256;
const INVALID_TRIANGLE_COUNT = 0xffffffff;
/** Above this physical slot density, sort/compact cost is amortized by shorter intersections. */
const DIRECT_MAXIMUM_SLOTS_PER_VERTEX = 8;

type ImportedAdjacency = {
  id: string;
  offsets: GraphDataView<'uint32'>;
  neighbors: GraphDataView<'uint32'>;
  overflow: GraphDataView<'uint32'>;
};

type CanonicalAdjacency = {
  rows: GraphDataView<'uint32'>;
  neighbors: GraphDataView<'uint32'>;
};

type ImportedLocalClustering = {
  id: string;
  vertexCount: number;
  directed: boolean;
  forward: ImportedAdjacency;
  reverse?: ImportedAdjacency;
  output: GraphDataView<'float32'>;
  triangles?: GraphDataView<'uint32'>;
};

type LocalClusteringBinding = {
  view: GraphDataView<'uint32'> | GraphDataView<'float32'>;
  usage: GraphBufferUse['usage'];
};

/**
 * Declares exact Graphalytics clustering through operation-owned sorted adjacency scratch.
 *
 * The canonicalization is deliberately local to this operation: caller-owned CSR order and
 * duplicate slots remain untouched. Each orientation is stably sorted by neighbor and then row,
 * yielding lexicographic `(row, neighbor)` order before a flag-scan-scatter removes equal pairs.
 * The final pass intersects sorted unique rows, reducing a hub from cubic membership scans to
 * quadratic merge work in its distinct degree. @internal
 */
export function addGPUGraphLocalClusteringCoefficientToGraphWithDispatchLimit<Parameters>(
  clustering: GPUGraphLocalClusteringCoefficient,
  commandGraph: GPUCommandGraph<Parameters>,
  maxComputeWorkgroupsPerDimension: number
): void {
  const vertexCount = clustering.topology.graph.vertexCount;
  if (vertexCount === 0) return;

  const topology = clustering.topology;
  const reverse = topology.graph.directed ? topology.reverse : undefined;
  const algorithm = getGPUGraphLocalClusteringCoefficientAlgorithm(
    clustering.algorithm,
    vertexCount,
    topology.forward.neighbors.length,
    reverse?.neighbors.length ?? 0
  );
  if (algorithm === 'direct') {
    addDirectLocalClusteringToGraph(clustering, commandGraph, maxComputeWorkgroupsPerDimension);
    return;
  }
  const importAdjacency = (id: string, adjacency: typeof topology.forward): ImportedAdjacency => ({
    id,
    offsets: commandGraph.importGPUVector(`${id}-offsets`, adjacency.offsets).data[0],
    neighbors: commandGraph.importGPUVector(`${id}-neighbors`, adjacency.neighbors).data[0],
    overflow: commandGraph.importGPUVector(`${id}-overflow`, adjacency.overflow).data[0]
  });
  const state: ImportedLocalClustering = {
    id: clustering.id,
    vertexCount,
    directed: topology.graph.directed,
    forward: importAdjacency(`${clustering.id}-forward`, topology.forward),
    ...(reverse ? {reverse: importAdjacency(`${clustering.id}-reverse`, reverse)} : {}),
    output: commandGraph.importGPUVector(`${clustering.id}-output`, clustering.output).data[0],
    ...(clustering.triangles
      ? {
          triangles: commandGraph.importGPUVector(
            `${clustering.id}-triangles`,
            clustering.triangles
          ).data[0]
        }
      : {})
  };

  const forward = addCanonicalAdjacency(
    commandGraph,
    state.forward,
    vertexCount,
    maxComputeWorkgroupsPerDimension
  );
  const canonicalReverse = state.reverse
    ? addCanonicalAdjacency(
        commandGraph,
        state.reverse,
        vertexCount,
        maxComputeWorkgroupsPerDimension
      )
    : undefined;
  addClusteringPass(
    commandGraph,
    state,
    forward,
    canonicalReverse,
    maxComputeWorkgroupsPerDimension
  );
}

/** Resolves the compile-time sparse/dense crossover without inspecting or reading GPU data. */
export function getGPUGraphLocalClusteringCoefficientAlgorithm(
  requested: 'auto' | 'direct' | 'canonical',
  vertexCount: number,
  forwardCapacity: number,
  reverseCapacity: number
): 'direct' | 'canonical' {
  if (requested !== 'auto') return requested;
  const totalCapacity = forwardCapacity + reverseCapacity;
  return totalCapacity <= vertexCount * DIRECT_MAXIMUM_SLOTS_PER_VERTEX ? 'direct' : 'canonical';
}

/** Builds lexicographically sorted `(row, neighbor)` scratch without modifying source CSR. */
function addCanonicalAdjacency<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  adjacency: ImportedAdjacency,
  vertexCount: number,
  maxComputeWorkgroupsPerDimension: number
): CanonicalAdjacency {
  const capacity = adjacency.neighbors.length;
  const inputRows = createTransientView(
    commandGraph,
    `${adjacency.id}-canonical-input-rows`,
    'uint32',
    capacity
  );
  const neighborSorted = createTransientView(
    commandGraph,
    `${adjacency.id}-canonical-neighbor-sorted`,
    'uint32',
    capacity
  );
  const neighborSortedRows = createTransientView(
    commandGraph,
    `${adjacency.id}-canonical-neighbor-sorted-rows`,
    'uint32',
    capacity
  );
  const sortedRows = createTransientView(
    commandGraph,
    `${adjacency.id}-canonical-sorted-rows`,
    'uint32',
    capacity
  );
  const sortedNeighbors = createTransientView(
    commandGraph,
    `${adjacency.id}-canonical-sorted-neighbors`,
    'uint32',
    capacity
  );
  const rows = createTransientView(
    commandGraph,
    `${adjacency.id}-canonical-rows`,
    'uint32',
    capacity
  );
  const neighbors = createTransientView(
    commandGraph,
    `${adjacency.id}-canonical-neighbors`,
    'uint32',
    capacity
  );

  if (capacity === 0) return {rows, neighbors};

  const dispatchLayout = getBoundedDispatchLayout(
    'GPUGraphLocalClusteringCoefficient canonicalization',
    capacity,
    LOCAL_CLUSTERING_WORKGROUP_SIZE,
    maxComputeWorkgroupsPerDimension
  );
  const source = /* wgsl */ `
const VERTEX_COUNT: u32 = ${vertexCount}u;
const CAPACITY: u32 = ${capacity}u;
const OFFSETS_OFFSET: u32 = ${getViewElementOffset(adjacency.offsets)}u;
const NEIGHBORS_OFFSET: u32 = ${getViewElementOffset(adjacency.neighbors)}u;
const ROWS_OFFSET: u32 = ${getViewElementOffset(inputRows)}u;
@group(0) @binding(0) var<storage, read> offsets: array<u32>;
@group(0) @binding(1) var<storage, read> sourceNeighbors: array<u32>;
@group(0) @binding(2) var<storage, read_write> rows: array<u32>;

@compute @workgroup_size(${LOCAL_CLUSTERING_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(dispatchLayout, LOCAL_CLUSTERING_WORKGROUP_SIZE)}
  if (index >= CAPACITY) { return; }

  var row = VERTEX_COUNT;
  let used = min(offsets[OFFSETS_OFFSET + VERTEX_COUNT], CAPACITY);
  if (index < used) {
    var low = 0u;
    var high = VERTEX_COUNT;
    while (low < high) {
      let middle = (low + high) / 2u;
      let nextOffset = min(offsets[OFFSETS_OFFSET + middle + 1u], CAPACITY);
      if (nextOffset <= index) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    let candidate = sourceNeighbors[NEIGHBORS_OFFSET + index];
    if (low < VERTEX_COUNT && candidate < VERTEX_COUNT && candidate != low) {
      row = low;
    }
  }
  rows[ROWS_OFFSET + index] = row;
}`;
  commandGraph.addComputePass({
    id: `${adjacency.id}-canonical-materialize`,
    resources: [
      {buffer: adjacency.offsets, usage: 'storage-read'},
      {buffer: adjacency.neighbors, usage: 'storage-read'},
      {buffer: inputRows, usage: 'storage-write'}
    ],
    compile: ({device}) => {
      const computation = new Computation(device, {
        id: `${adjacency.id}-canonical-materialize`,
        source,
        shaderLayout: {
          bindings: [
            {name: 'offsets', type: 'storage', group: 0, location: 0},
            {name: 'sourceNeighbors', type: 'storage', group: 0, location: 1},
            {name: 'rows', type: 'storage', group: 0, location: 2}
          ]
        }
      });
      return {
        encode: ({computePass, getBuffer}) => {
          computation.setBindings({
            offsets: getViewBinding(adjacency.offsets, getBuffer),
            sourceNeighbors: getViewBinding(adjacency.neighbors, getBuffer),
            rows: getViewBinding(inputRows, getBuffer)
          });
          computation.dispatch(computePass, dispatchLayout.x, dispatchLayout.y, dispatchLayout.z);
        },
        destroy: () => computation.destroy()
      };
    }
  });

  const keyBits = Math.max(1, Math.ceil(Math.log2(vertexCount + 1)));
  commandGraph.add(
    new GPUSort({
      id: `${adjacency.id}-canonical-sort-neighbor`,
      keys: adjacency.neighbors,
      values: inputRows,
      outputKeys: neighborSorted,
      outputValues: neighborSortedRows,
      keyBits
    })
  );
  // The sort is stable, so sorting neighbor-ordered pairs by row produces lexicographic order.
  commandGraph.add(
    new GPUSort({
      id: `${adjacency.id}-canonical-sort-row`,
      keys: neighborSortedRows,
      values: neighborSorted,
      outputKeys: sortedRows,
      outputValues: sortedNeighbors,
      keyBits
    })
  );
  addUniqueCompaction(
    commandGraph,
    adjacency.id,
    vertexCount,
    sortedRows,
    sortedNeighbors,
    rows,
    neighbors,
    maxComputeWorkgroupsPerDimension
  );
  return {rows, neighbors};
}

/** Compacts adjacent equal pairs and sentinel-fills unused capacity for bounded binary searches. */
function addUniqueCompaction<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  id: string,
  vertexCount: number,
  sortedRows: GraphDataView<'uint32'>,
  sortedNeighbors: GraphDataView<'uint32'>,
  rows: GraphDataView<'uint32'>,
  neighbors: GraphDataView<'uint32'>,
  maxComputeWorkgroupsPerDimension: number
): void {
  const capacity = sortedRows.length;
  const flags = createTransientView(
    commandGraph,
    `${id}-canonical-unique-flags`,
    'uint32',
    capacity
  );
  const positions = createTransientView(
    commandGraph,
    `${id}-canonical-unique-positions`,
    'uint32',
    capacity
  );
  const dispatchLayout = getBoundedDispatchLayout(
    'GPUGraphLocalClusteringCoefficient unique compaction',
    capacity,
    LOCAL_CLUSTERING_WORKGROUP_SIZE,
    maxComputeWorkgroupsPerDimension
  );
  const constants = `const VERTEX_COUNT: u32 = ${vertexCount}u;
const CAPACITY: u32 = ${capacity}u;
const SORTED_ROWS_OFFSET: u32 = ${getViewElementOffset(sortedRows)}u;
const SORTED_NEIGHBORS_OFFSET: u32 = ${getViewElementOffset(sortedNeighbors)}u;
const FLAGS_OFFSET: u32 = ${getViewElementOffset(flags)}u;
const POSITIONS_OFFSET: u32 = ${getViewElementOffset(positions)}u;
const ROWS_OFFSET: u32 = ${getViewElementOffset(rows)}u;
const NEIGHBORS_OFFSET: u32 = ${getViewElementOffset(neighbors)}u;`;
  addCanonicalKernelPass(commandGraph, {
    id: `${id}-canonical-unique-flags`,
    source: `${constants}
@group(0) @binding(0) var<storage, read> sortedRows: array<u32>;
@group(0) @binding(1) var<storage, read> sortedNeighbors: array<u32>;
@group(0) @binding(2) var<storage, read_write> flags: array<u32>;
@group(0) @binding(3) var<storage, read_write> rows: array<u32>;
@group(0) @binding(4) var<storage, read_write> neighbors: array<u32>;
@compute @workgroup_size(${LOCAL_CLUSTERING_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(dispatchLayout, LOCAL_CLUSTERING_WORKGROUP_SIZE)}
  if (index >= CAPACITY) { return; }
  let row = sortedRows[SORTED_ROWS_OFFSET + index];
  let neighbor = sortedNeighbors[SORTED_NEIGHBORS_OFFSET + index];
  var unique = row < VERTEX_COUNT;
  if (unique && index > 0u) {
    unique = sortedRows[SORTED_ROWS_OFFSET + index - 1u] != row ||
      sortedNeighbors[SORTED_NEIGHBORS_OFFSET + index - 1u] != neighbor;
  }
  flags[FLAGS_OFFSET + index] = select(0u, 1u, unique);
  rows[ROWS_OFFSET + index] = VERTEX_COUNT;
  neighbors[NEIGHBORS_OFFSET + index] = VERTEX_COUNT;
}`,
    resources: [
      {buffer: sortedRows, usage: 'storage-read'},
      {buffer: sortedNeighbors, usage: 'storage-read'},
      {buffer: flags, usage: 'storage-write'},
      {buffer: rows, usage: 'storage-write'},
      {buffer: neighbors, usage: 'storage-write'}
    ],
    bindings: {sortedRows, sortedNeighbors, flags, rows, neighbors},
    dispatchLayout
  });
  commandGraph.add(
    new GPUScan({id: `${id}-canonical-unique-scan`, input: flags, output: positions})
  );
  addCanonicalKernelPass(commandGraph, {
    id: `${id}-canonical-unique-scatter`,
    source: `${constants}
@group(0) @binding(0) var<storage, read> sortedRows: array<u32>;
@group(0) @binding(1) var<storage, read> sortedNeighbors: array<u32>;
@group(0) @binding(2) var<storage, read> flags: array<u32>;
@group(0) @binding(3) var<storage, read> positions: array<u32>;
@group(0) @binding(4) var<storage, read_write> rows: array<u32>;
@group(0) @binding(5) var<storage, read_write> neighbors: array<u32>;
@compute @workgroup_size(${LOCAL_CLUSTERING_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(dispatchLayout, LOCAL_CLUSTERING_WORKGROUP_SIZE)}
  if (index >= CAPACITY || flags[FLAGS_OFFSET + index] == 0u) { return; }
  let outputIndex = positions[POSITIONS_OFFSET + index];
  rows[ROWS_OFFSET + outputIndex] = sortedRows[SORTED_ROWS_OFFSET + index];
  neighbors[NEIGHBORS_OFFSET + outputIndex] = sortedNeighbors[SORTED_NEIGHBORS_OFFSET + index];
}`,
    resources: [
      {buffer: sortedRows, usage: 'storage-read'},
      {buffer: sortedNeighbors, usage: 'storage-read'},
      {buffer: flags, usage: 'storage-read'},
      {buffer: positions, usage: 'storage-read'},
      {buffer: rows, usage: 'storage-write'},
      {buffer: neighbors, usage: 'storage-write'}
    ],
    bindings: {sortedRows, sortedNeighbors, flags, positions, rows, neighbors},
    dispatchLayout
  });
}

/** Adds one fixed-binding canonicalization kernel. */
function addCanonicalKernelPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    source: string;
    resources: GraphBufferUse[];
    bindings: Record<string, GraphDataView<'uint32'>>;
    dispatchLayout: GPUBoundedDispatchLayout;
  }
): void {
  commandGraph.addComputePass({
    id: props.id,
    resources: props.resources,
    compile: ({device}) => {
      const computation = new Computation(device, {
        id: props.id,
        source: props.source,
        shaderLayout: {
          bindings: Object.keys(props.bindings).map((name, location) => ({
            name,
            type: 'storage' as const,
            group: 0,
            location
          }))
        }
      });
      return {
        encode: ({computePass, getBuffer}) => {
          const shaderBindings: Record<string, Binding> = {};
          for (const [name, view] of Object.entries(props.bindings)) {
            shaderBindings[name] = getViewBinding(view, getBuffer);
          }
          computation.setBindings(shaderBindings);
          computation.dispatch(
            computePass,
            props.dispatchLayout.x,
            props.dispatchLayout.y,
            props.dispatchLayout.z
          );
        },
        destroy: () => computation.destroy()
      };
    }
  });
}

/** Counts closures with sorted row intersections and publishes one result per vertex. */
function addClusteringPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  state: ImportedLocalClustering,
  forward: CanonicalAdjacency,
  reverse: CanonicalAdjacency | undefined,
  maxComputeWorkgroupsPerDimension: number
): void {
  const bindings: Record<string, LocalClusteringBinding> = {
    forwardRows: {view: forward.rows, usage: 'storage-read'},
    forwardNeighbors: {view: forward.neighbors, usage: 'storage-read'},
    forwardOverflow: {view: state.forward.overflow, usage: 'storage-read'},
    ...(reverse
      ? {
          reverseRows: {view: reverse.rows, usage: 'storage-read' as const},
          reverseNeighbors: {view: reverse.neighbors, usage: 'storage-read' as const},
          reverseOverflow: {view: state.reverse!.overflow, usage: 'storage-read' as const}
        }
      : {}),
    output: {view: state.output, usage: 'storage-write'},
    ...(state.triangles ? {triangles: {view: state.triangles, usage: 'storage-write'}} : {})
  };
  const dispatchLayout = getGPUGraphLocalClusteringCoefficientDispatchLayout(
    state.vertexCount,
    maxComputeWorkgroupsPerDimension
  );
  const source = getLocalClusteringSource(state, bindings, forward, reverse, dispatchLayout);

  commandGraph.addComputePass({
    id: `${state.id}-calculate`,
    resources: Object.values(bindings).map(({view, usage}) => ({buffer: view, usage})),
    compile: ({device}) => {
      const computation = new Computation(device, {
        id: `${state.id}-calculate`,
        source,
        shaderLayout: {
          bindings: Object.keys(bindings).map((name, location) => ({
            name,
            type: 'storage' as const,
            group: 0,
            location
          }))
        }
      });
      return {
        encode: ({computePass, getBuffer}) => {
          const shaderBindings: Record<string, Binding> = {};
          for (const [name, binding] of Object.entries(bindings)) {
            shaderBindings[name] = getViewBinding(binding.view, getBuffer);
          }
          computation.setBindings(shaderBindings);
          computation.dispatch(computePass, dispatchLayout.x, dispatchLayout.y, dispatchLayout.z);
        },
        destroy: () => computation.destroy()
      };
    }
  });
}

/** Generates the merge-intersection pass using at most eight baseline storage bindings. */
function getLocalClusteringSource(
  state: ImportedLocalClustering,
  bindings: Record<string, LocalClusteringBinding>,
  forward: CanonicalAdjacency,
  reverse: CanonicalAdjacency | undefined,
  dispatchLayout: GPUBoundedDispatchLayout
): string {
  const hasReverse = Boolean(reverse);
  const reverseConstants = hasReverse
    ? `const REVERSE_CAPACITY: u32 = ${reverse!.neighbors.length}u;
const REVERSE_ROWS_OFFSET: u32 = ${getViewElementOffset(reverse!.rows)}u;
const REVERSE_NEIGHBORS_OFFSET: u32 = ${getViewElementOffset(reverse!.neighbors)}u;
const REVERSE_OVERFLOW_OFFSET: u32 = ${getViewElementOffset(state.reverse!.overflow)}u;`
    : '';
  const reverseOverflow = hasReverse ? ' || reverseOverflow[REVERSE_OVERFLOW_OFFSET] != 0u' : '';
  const reverseRangeHelpers = hasReverse
    ? `
fn getReverseRowFirst(row: u32) -> u32 {
  var low = 0u;
  var high = REVERSE_CAPACITY;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (reverseRows[REVERSE_ROWS_OFFSET + middle] < row) { low = middle + 1u; } else { high = middle; }
  }
  return low;
}

fn getReverseRowEnd(row: u32) -> u32 {
  var low = 0u;
  var high = REVERSE_CAPACITY;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (reverseRows[REVERSE_ROWS_OFFSET + middle] <= row) { low = middle + 1u; } else { high = middle; }
  }
  return low;
}`
    : '';
  const reverseIntersectionInitialization = hasReverse
    ? `var reverseCursor = getReverseRowFirst(center);
  let reverseEnd = getReverseRowEnd(center);`
    : '';
  const reverseIntersectionCondition = hasReverse ? ' || reverseCursor < reverseEnd' : '';
  const reverseIntersectionCandidate = hasReverse
    ? `if (reverseCursor < reverseEnd) {
      weakNeighbor = min(weakNeighbor, reverseNeighbors[REVERSE_NEIGHBORS_OFFSET + reverseCursor]);
    }`
    : '';
  const reverseIntersectionAdvance = hasReverse
    ? `while (reverseCursor < reverseEnd &&
           reverseNeighbors[REVERSE_NEIGHBORS_OFFSET + reverseCursor] == weakNeighbor) {
      reverseCursor++;
    }`
    : '';
  const reverseMainInitialization = hasReverse
    ? `var reverseCursor = getReverseRowFirst(index);
  let reverseEnd = getReverseRowEnd(index);`
    : '';
  const reverseMainCondition = hasReverse ? ' || reverseCursor < reverseEnd' : '';
  const reverseMainCandidate = hasReverse
    ? `if (reverseCursor < reverseEnd) {
      neighbor = min(neighbor, reverseNeighbors[REVERSE_NEIGHBORS_OFFSET + reverseCursor]);
    }`
    : '';
  const reverseMainAdvance = hasReverse
    ? `while (reverseCursor < reverseEnd &&
           reverseNeighbors[REVERSE_NEIGHBORS_OFFSET + reverseCursor] == neighbor) {
      reverseCursor++;
    }`
    : '';
  const triangleOffset = state.triangles
    ? `const TRIANGLES_OFFSET: u32 = ${getViewElementOffset(state.triangles)}u;`
    : '';
  const publishInvalidTriangles = state.triangles
    ? `triangles[TRIANGLES_OFFSET + index] = ${INVALID_TRIANGLE_COUNT}u;`
    : '';
  const publishTriangles = state.triangles
    ? `triangles[TRIANGLES_OFFSET + index] = ${state.directed ? 'closureCount' : 'closureCount / 2u'};`
    : '';
  const declarations = Object.entries(bindings)
    .map(([name, binding], location) => {
      const access = binding.usage === 'storage-read' ? 'read' : 'read_write';
      const element = binding.view.format === 'float32' ? 'f32' : 'u32';
      return `@group(0) @binding(${location}) var<storage, ${access}> ${name}: array<${element}>;`;
    })
    .join('\n');

  return /* wgsl */ `
const VERTEX_COUNT: u32 = ${state.vertexCount}u;
const FORWARD_CAPACITY: u32 = ${forward.neighbors.length}u;
const FORWARD_ROWS_OFFSET: u32 = ${getViewElementOffset(forward.rows)}u;
const FORWARD_NEIGHBORS_OFFSET: u32 = ${getViewElementOffset(forward.neighbors)}u;
const FORWARD_OVERFLOW_OFFSET: u32 = ${getViewElementOffset(state.forward.overflow)}u;
const OUTPUT_OFFSET: u32 = ${getViewElementOffset(state.output)}u;
${reverseConstants}
${triangleOffset}
${declarations}

fn getForwardRowFirst(row: u32) -> u32 {
  var low = 0u;
  var high = FORWARD_CAPACITY;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (forwardRows[FORWARD_ROWS_OFFSET + middle] < row) { low = middle + 1u; } else { high = middle; }
  }
  return low;
}

fn getForwardRowEnd(row: u32) -> u32 {
  var low = 0u;
  var high = FORWARD_CAPACITY;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (forwardRows[FORWARD_ROWS_OFFSET + middle] <= row) { low = middle + 1u; } else { high = middle; }
  }
  return low;
}
${reverseRangeHelpers}

// Intersects the center's sorted weak-neighbor union with one neighbor's sorted outgoing row.
// Equal runs advance together, so duplicate source slots never inflate the closure count.
fn countNeighborClosures(center: u32, neighbor: u32) -> u32 {
  var forwardCursor = getForwardRowFirst(center);
  let forwardEnd = getForwardRowEnd(center);
  ${reverseIntersectionInitialization}
  var edgeCursor = getForwardRowFirst(neighbor);
  let edgeEnd = getForwardRowEnd(neighbor);
  var count = 0u;

  while (edgeCursor < edgeEnd && (forwardCursor < forwardEnd${reverseIntersectionCondition})) {
    var weakNeighbor = 0xffffffffu;
    if (forwardCursor < forwardEnd) {
      weakNeighbor = forwardNeighbors[FORWARD_NEIGHBORS_OFFSET + forwardCursor];
    }
    ${reverseIntersectionCandidate}
    let edgeNeighbor = forwardNeighbors[FORWARD_NEIGHBORS_OFFSET + edgeCursor];
    if (weakNeighbor == edgeNeighbor) { count++; }

    if (weakNeighbor <= edgeNeighbor) {
      while (forwardCursor < forwardEnd &&
             forwardNeighbors[FORWARD_NEIGHBORS_OFFSET + forwardCursor] == weakNeighbor) {
        forwardCursor++;
      }
      ${reverseIntersectionAdvance}
    }
    if (edgeNeighbor <= weakNeighbor) {
      while (edgeCursor < edgeEnd &&
             forwardNeighbors[FORWARD_NEIGHBORS_OFFSET + edgeCursor] == edgeNeighbor) {
        edgeCursor++;
      }
    }
  }
  return count;
}

@compute @workgroup_size(${LOCAL_CLUSTERING_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(dispatchLayout, LOCAL_CLUSTERING_WORKGROUP_SIZE)}
  if (index >= VERTEX_COUNT) { return; }
  if (forwardOverflow[FORWARD_OVERFLOW_OFFSET] != 0u${reverseOverflow}) {
    output[OUTPUT_OFFSET + index] = 0.0;
    ${publishInvalidTriangles}
    return;
  }

  var forwardCursor = getForwardRowFirst(index);
  let forwardEnd = getForwardRowEnd(index);
  ${reverseMainInitialization}
  var degree = 0u;
  var closureCount = 0u;
  while (forwardCursor < forwardEnd${reverseMainCondition}) {
    var neighbor = 0xffffffffu;
    if (forwardCursor < forwardEnd) {
      neighbor = forwardNeighbors[FORWARD_NEIGHBORS_OFFSET + forwardCursor];
    }
    ${reverseMainCandidate}
    while (forwardCursor < forwardEnd &&
           forwardNeighbors[FORWARD_NEIGHBORS_OFFSET + forwardCursor] == neighbor) {
      forwardCursor++;
    }
    ${reverseMainAdvance}
    degree++;

    let increment = countNeighborClosures(index, neighbor);
    if (closureCount >= ${INVALID_TRIANGLE_COUNT}u - increment) {
      output[OUTPUT_OFFSET + index] = 0.0;
      ${publishInvalidTriangles}
      return;
    }
    closureCount += increment;
  }

  output[OUTPUT_OFFSET + index] = select(
    0.0,
    f32(closureCount) / (f32(degree) * f32(degree - 1u)),
    degree >= 2u
  );
  ${publishTriangles}
}`;
}

/** Plans bounded true three-dimensional vertex-clustering dispatch. @internal */
export function getGPUGraphLocalClusteringCoefficientDispatchLayout(
  elementCount: number,
  maxComputeWorkgroupsPerDimension: number
): GPUBoundedDispatchLayout {
  return getBoundedDispatchLayout(
    'GPUGraphLocalClusteringCoefficient',
    elementCount,
    LOCAL_CLUSTERING_WORKGROUP_SIZE,
    maxComputeWorkgroupsPerDimension
  );
}
