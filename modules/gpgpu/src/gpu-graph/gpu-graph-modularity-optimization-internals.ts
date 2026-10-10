// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileComment: Independently implemented for WebGPU; inspired by NVIDIA RAPIDS cuGraph.

import {type Binding} from '@luma.gl/core';
import {Computation} from '@luma.gl/engine';
import type {
  GPUCommandGraph,
  GraphBufferUse,
  GraphDataView,
  GraphVectorView
} from '../gpu-core/gpu-command-graph';
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
import {GPUGraphModularity} from './gpu-graph-modularity';
import type {GPUGraphModularityOptimization} from './gpu-graph-modularity-optimization';

const OPTIMIZATION_WORKGROUP_SIZE = 256;
const INVALID_COMMUNITY = 0xffffffff;
const INVALID_STATUS = 1;
const CONVERGED_STATUS = 2;

type OptimizationView =
  | GraphDataView<'uint32'>
  | GraphDataView<'float32'>
  | GraphDataView<'uint32x2'>
  | GraphDataView<'float32x2'>
  | GraphDataView<'float32x4'>;

type OptimizationBinding = {
  view: OptimizationView;
  usage: GraphBufferUse['usage'];
  atomic?: boolean;
};

type OptimizationPass = {
  id: string;
  source: string;
  bindings: Record<string, OptimizationBinding>;
  dispatchLayout: GPUBoundedDispatchLayout;
};

type ImportedOptimization = {
  id: string;
  vertexCount: number;
  directed: boolean;
  resolution: number;
  minimumGain: number;
  iterations: number;
  sourceVertices: GraphVectorView<'uint32'>;
  targetVertices: GraphVectorView<'uint32'>;
  edgeWeights?: GraphVectorView<'float32'>;
  forwardOffsets: GraphDataView<'uint32'>;
  forwardNeighbors: GraphDataView<'uint32'>;
  forwardWeights?: GraphDataView<'float32'>;
  forwardOverflow: GraphDataView<'uint32'>;
  packedForward: GraphDataView<'uint32x2'>;
  reverseOffsets?: GraphDataView<'uint32'>;
  reverseNeighbors?: GraphDataView<'uint32'>;
  reverseWeights?: GraphDataView<'float32'>;
  reverseOverflow?: GraphDataView<'uint32'>;
  packedReverse?: GraphDataView<'uint32x2'>;
  initialCommunities?: GraphDataView<'uint32'>;
  output: GraphDataView<'uint32'>;
  statistics: GraphDataView<'float32x4'>;
  aggregatedCommunities?: GraphDataView<'uint32'>;
  aggregatedWeights?: GraphDataView<'float32x2'>;
  aggregationRanges?: GraphDataView<'uint32x2'>;
  candidates: GraphDataView<'uint32x2'>;
  control: GraphDataView<'uint32'>;
  converged?: GraphDataView<'uint32'>;
  maxComputeWorkgroupsPerDimension: number;
};

type AggregatedOptimization = ImportedOptimization & {
  aggregatedCommunities: GraphDataView<'uint32'>;
  aggregatedWeights: GraphDataView<'float32x2'>;
  aggregationRanges: GraphDataView<'uint32x2'>;
};

/** Composes single-level modularity local moving with stable ties for computed GPU gains. */
export function addGPUGraphModularityOptimizationToGraphWithDispatchLimit<Parameters>(
  optimization: GPUGraphModularityOptimization,
  commandGraph: GPUCommandGraph<Parameters>,
  maxComputeWorkgroupsPerDimension: number
): void {
  const topology = optimization.topology;
  const vertexCount = topology.graph.vertexCount;
  const reverse = topology.graph.directed ? topology.reverse : undefined;
  const aggregationCapacity = getCommunityAggregationCapacity(
    vertexCount,
    topology.forward.neighbors.length,
    reverse?.neighbors.length ?? 0
  );
  const maximumScratchBindingByteLength = Math.min(
    commandGraph.device.limits.maxBufferSize,
    commandGraph.device.limits.maxStorageBufferBindingSize
  );
  const supportedAggregationCapacity =
    aggregationCapacity !== undefined &&
    aggregationCapacity * 8 <= maximumScratchBindingByteLength &&
    vertexCount * 8 <= maximumScratchBindingByteLength
      ? aggregationCapacity
      : undefined;
  const state: ImportedOptimization = {
    id: optimization.id,
    vertexCount,
    directed: topology.graph.directed,
    resolution: Math.fround(optimization.resolution),
    minimumGain: Math.fround(optimization.minimumGain),
    iterations: optimization.iterations,
    sourceVertices: commandGraph.importGPUVector(
      `${optimization.id}-source-vertices`,
      topology.graph.sourceVertices
    ),
    targetVertices: commandGraph.importGPUVector(
      `${optimization.id}-target-vertices`,
      topology.graph.targetVertices
    ),
    ...(topology.graph.edgeWeights
      ? {
          edgeWeights: commandGraph.importGPUVector(
            `${optimization.id}-edge-weights`,
            topology.graph.edgeWeights
          )
        }
      : {}),
    forwardOffsets: commandGraph.importGPUVector(
      `${optimization.id}-forward-offsets`,
      topology.forward.offsets
    ).data[0],
    forwardNeighbors: commandGraph.importGPUVector(
      `${optimization.id}-forward-neighbors`,
      topology.forward.neighbors
    ).data[0],
    ...(topology.forward.edgeWeights
      ? {
          forwardWeights: commandGraph.importGPUVector(
            `${optimization.id}-forward-weights`,
            topology.forward.edgeWeights
          ).data[0]
        }
      : {}),
    forwardOverflow: commandGraph.importGPUVector(
      `${optimization.id}-forward-overflow`,
      topology.forward.overflow
    ).data[0],
    packedForward: createTransientView(
      commandGraph,
      `${optimization.id}-packed-forward`,
      'uint32x2',
      topology.forward.neighbors.length
    ),
    ...(reverse
      ? {
          reverseOffsets: commandGraph.importGPUVector(
            `${optimization.id}-reverse-offsets`,
            reverse.offsets
          ).data[0],
          reverseNeighbors: commandGraph.importGPUVector(
            `${optimization.id}-reverse-neighbors`,
            reverse.neighbors
          ).data[0],
          ...(reverse.edgeWeights
            ? {
                reverseWeights: commandGraph.importGPUVector(
                  `${optimization.id}-reverse-weights`,
                  reverse.edgeWeights
                ).data[0]
              }
            : {}),
          reverseOverflow: commandGraph.importGPUVector(
            `${optimization.id}-reverse-overflow`,
            reverse.overflow
          ).data[0],
          packedReverse: createTransientView(
            commandGraph,
            `${optimization.id}-packed-reverse`,
            'uint32x2',
            reverse.neighbors.length
          )
        }
      : {}),
    ...(optimization.initialCommunities
      ? {
          initialCommunities: commandGraph.importGPUVector(
            `${optimization.id}-initial-communities`,
            optimization.initialCommunities
          ).data[0]
        }
      : {}),
    output: commandGraph.importGPUVector(`${optimization.id}-output`, optimization.output).data[0],
    statistics: createTransientView(
      commandGraph,
      `${optimization.id}-vertex-community-statistics`,
      'float32x4',
      vertexCount
    ),
    ...(supportedAggregationCapacity !== undefined
      ? {
          aggregatedCommunities: createTransientView(
            commandGraph,
            `${optimization.id}-aggregated-communities`,
            'uint32',
            supportedAggregationCapacity
          ),
          aggregatedWeights: createTransientView(
            commandGraph,
            `${optimization.id}-aggregated-community-weights`,
            'float32x2',
            supportedAggregationCapacity
          ),
          aggregationRanges: createTransientView(
            commandGraph,
            `${optimization.id}-community-aggregation-ranges`,
            'uint32x2',
            vertexCount
          )
        }
      : {}),
    candidates: createTransientView(
      commandGraph,
      `${optimization.id}-candidate-moves`,
      'uint32x2',
      vertexCount
    ),
    control: createTransientView(
      commandGraph,
      `${optimization.id}-control`,
      'uint32',
      vertexCount + 5
    ),
    ...(optimization.converged
      ? {
          converged: commandGraph.importGPUVector(
            `${optimization.id}-converged`,
            optimization.converged
          ).data[0]
        }
      : {}),
    maxComputeWorkgroupsPerDimension
  };

  addControlInitializationPass(commandGraph, state);
  if (vertexCount > 0) addCommunityInitializationPass(commandGraph, state);
  if (state.forwardNeighbors.length > 0) {
    addAdjacencyPackingPass(commandGraph, {
      state,
      direction: 'forward',
      neighbors: state.forwardNeighbors,
      weights: state.forwardWeights,
      packed: state.packedForward
    });
  }
  if (state.reverseNeighbors && state.packedReverse && state.reverseNeighbors.length > 0) {
    addAdjacencyPackingPass(commandGraph, {
      state,
      direction: 'reverse',
      neighbors: state.reverseNeighbors,
      weights: state.reverseWeights,
      packed: state.packedReverse
    });
  }

  for (let iteration = 0; iteration < Math.max(optimization.iterations, 1); iteration++) {
    addIterationResetPass(commandGraph, {state, iteration});
    if (vertexCount > 0) {
      addStatisticsInitializationPass(commandGraph, {state, iteration});
      if (iteration < optimization.iterations) {
        addCommunityOccupancyPass(commandGraph, {state, iteration});
        addAvailableCommunitySelectionPass(commandGraph, {state, iteration});
      }
    }
    for (const [chunkIndex, sources] of state.sourceVertices.data.entries()) {
      if (sources.length === 0) continue;
      addEdgeStatisticsPass(commandGraph, {
        state,
        iteration,
        chunkIndex,
        sources,
        targets: state.targetVertices.data[chunkIndex],
        weights: state.edgeWeights?.data[chunkIndex]
      });
    }
    if (iteration < optimization.iterations && vertexCount > 0) {
      if (hasCommunityAggregation(state)) {
        addCommunityAggregationPass(commandGraph, {state, iteration});
        addAggregatedCandidatePass(commandGraph, {state, iteration});
      } else {
        addRescanCandidatePass(commandGraph, {state, iteration});
      }
      addWinnerSelectionPass(commandGraph, {state, iteration});
      addWinnerApplicationPass(commandGraph, {state, iteration});
    }
  }

  addFailureFinalizationPass(commandGraph, state);
  new GPUGraphModularity({
    id: `${optimization.id}-final-modularity`,
    graph: topology.graph,
    communities: optimization.output,
    output: optimization.modularity,
    resolution: optimization.resolution,
    ...(optimization.valid ? {valid: optimization.valid} : {})
  }).addToGraph(commandGraph);
}

/** Initializes compact scalar control and rejects overflow before any communities are consumed. */
function addControlInitializationPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  state: ImportedOptimization
): void {
  const bindings: Record<string, OptimizationBinding> = {
    control: {view: state.control, usage: 'storage-read-write', atomic: true},
    forwardOverflow: {view: state.forwardOverflow, usage: 'storage-read'},
    ...(state.reverseOverflow
      ? {reverseOverflow: {view: state.reverseOverflow, usage: 'storage-read' as const}}
      : {}),
    ...(state.converged
      ? {converged: {view: state.converged, usage: 'storage-write' as const}}
      : {})
  };
  const reverseOverflow = state.reverseOverflow
    ? ` || reverseOverflow[${getViewElementOffset(state.reverseOverflow)}u] != 0u`
    : '';
  const initializeConvergence = state.converged
    ? `converged[${getViewElementOffset(state.converged)}u] = select(0u, 1u, VERTEX_COUNT == 0u && !hasOverflow);`
    : '';
  const source = /* wgsl */ `
const VERTEX_COUNT: u32 = ${state.vertexCount}u;
${getBindingDeclarations(bindings)}

@compute @workgroup_size(1)
fn main() {
  let hasOverflow = forwardOverflow[${getViewElementOffset(state.forwardOverflow)}u] != 0u${reverseOverflow};
  atomicStore(&control[0u], 0u);
  atomicStore(&control[1u], select(0u, ${INVALID_STATUS}u, hasOverflow));
  atomicStore(&control[2u], 0u);
  atomicStore(&control[3u], ${INVALID_COMMUNITY}u);
  atomicStore(&control[4u], ${INVALID_COMMUNITY}u);
  ${initializeConvergence}
}`;
  addOptimizationPass(commandGraph, {
    id: `${state.id}-initialize-control`,
    source,
    bindings,
    dispatchLayout: {x: 1, y: 1, z: 1}
  });
}

/** Copies stable caller-owned assignments or initializes one identity community per vertex. */
function addCommunityInitializationPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  state: ImportedOptimization
): void {
  const bindings: Record<string, OptimizationBinding> = {
    output: {view: state.output, usage: 'storage-write'},
    ...(state.initialCommunities
      ? {initialCommunities: {view: state.initialCommunities, usage: 'storage-read' as const}}
      : {}),
    control: {view: state.control, usage: 'storage-read-write', atomic: true}
  };
  const initialCommunity = state.initialCommunities
    ? `initialCommunities[${getViewElementOffset(state.initialCommunities)}u + index]`
    : 'index';
  const dispatchLayout = getDispatchLayout(state, state.vertexCount);
  const source = /* wgsl */ `
const VERTEX_COUNT: u32 = ${state.vertexCount}u;
${getBindingDeclarations(bindings)}

@compute @workgroup_size(${OPTIMIZATION_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(dispatchLayout, OPTIMIZATION_WORKGROUP_SIZE)}
  if (index >= VERTEX_COUNT) { return; }
  let community = ${initialCommunity};
  output[${getViewElementOffset(state.output)}u + index] = community;
  if (community >= VERTEX_COUNT) {
    atomicOr(&control[1u], ${INVALID_STATUS}u);
  }
}`;
  addOptimizationPass(commandGraph, {
    id: `${state.id}-initialize-communities`,
    source,
    bindings,
    dispatchLayout
  });
}

/** Packs one existing CSR neighbor and its optional float weight into a single storage binding. */
function addAdjacencyPackingPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: {
    state: ImportedOptimization;
    direction: 'forward' | 'reverse';
    neighbors: GraphDataView<'uint32'>;
    weights?: GraphDataView<'float32'>;
    packed: GraphDataView<'uint32x2'>;
  }
): void {
  const bindings: Record<string, OptimizationBinding> = {
    neighbors: {view: props.neighbors, usage: 'storage-read'},
    ...(props.weights ? {weights: {view: props.weights, usage: 'storage-read' as const}} : {}),
    packed: {view: props.packed, usage: 'storage-write'}
  };
  const weight = props.weights ? `weights[${getViewElementOffset(props.weights)}u + index]` : '1.0';
  const dispatchLayout = getDispatchLayout(props.state, props.neighbors.length);
  const source = /* wgsl */ `
const ADJACENCY_CAPACITY: u32 = ${props.neighbors.length}u;
${getBindingDeclarations(bindings)}

@compute @workgroup_size(${OPTIMIZATION_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(dispatchLayout, OPTIMIZATION_WORKGROUP_SIZE)}
  if (index >= ADJACENCY_CAPACITY) { return; }
  packed[index] = vec2<u32>(
    neighbors[${getViewElementOffset(props.neighbors)}u + index],
    bitcast<u32>(${weight})
  );
}`;
  addOptimizationPass(commandGraph, {
    id: `${props.state.id}-pack-${props.direction}-adjacency`,
    source,
    bindings,
    dispatchLayout
  });
}

/** Clears move selection and total volume while preserving sticky invalid and converged states. */
function addIterationResetPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: {state: ImportedOptimization; iteration: number}
): void {
  const bindings: Record<string, OptimizationBinding> = {
    control: {view: props.state.control, usage: 'storage-read-write', atomic: true}
  };
  const source = /* wgsl */ `
${getBindingDeclarations(bindings)}

@compute @workgroup_size(1)
fn main() {
  if (atomicLoad(&control[1u]) != 0u) { return; }
  atomicStore(&control[0u], 0u);
  atomicStore(&control[2u], 0u);
  atomicStore(&control[3u], ${INVALID_COMMUNITY}u);
  atomicStore(&control[4u], ${INVALID_COMMUNITY}u);
}`;
  addOptimizationPass(commandGraph, {
    id: `${props.state.id}-iteration-${props.iteration}-reset`,
    source,
    bindings,
    dispatchLayout: {x: 1, y: 1, z: 1}
  });
}

/** Clears four packed float lanes per vertex and validates the current immutable partition. */
function addStatisticsInitializationPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: {state: ImportedOptimization; iteration: number}
): void {
  const {state} = props;
  const bindings: Record<string, OptimizationBinding> = {
    output: {view: state.output, usage: 'storage-read'},
    statistics: {view: state.statistics, usage: 'storage-read-write', atomic: true},
    control: {view: state.control, usage: 'storage-read-write', atomic: true}
  };
  const dispatchLayout = getDispatchLayout(state, state.vertexCount);
  const source = /* wgsl */ `
const VERTEX_COUNT: u32 = ${state.vertexCount}u;
${getBindingDeclarations(bindings)}

@compute @workgroup_size(${OPTIMIZATION_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(dispatchLayout, OPTIMIZATION_WORKGROUP_SIZE)}
  if (index >= VERTEX_COUNT || atomicLoad(&control[1u]) != 0u) { return; }
  atomicStore(&statistics[4u * index], 0u);
  atomicStore(&statistics[4u * index + 1u], 0u);
  atomicStore(&statistics[4u * index + 2u], 0u);
  atomicStore(&statistics[4u * index + 3u], 0u);
  atomicStore(&control[5u + index], 0u);
  if (output[${getViewElementOffset(state.output)}u + index] >= VERTEX_COUNT) {
    atomicOr(&control[1u], ${INVALID_STATUS}u);
  }
}`;
  addOptimizationPass(commandGraph, {
    id: `${state.id}-iteration-${props.iteration}-initialize-statistics`,
    source,
    bindings,
    dispatchLayout
  });
}

/** Marks every occupied stable community, including communities containing only isolates. */
function addCommunityOccupancyPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: {state: ImportedOptimization; iteration: number}
): void {
  const {state} = props;
  const bindings: Record<string, OptimizationBinding> = {
    communities: {view: state.output, usage: 'storage-read'},
    control: {view: state.control, usage: 'storage-read-write', atomic: true}
  };
  const dispatchLayout = getDispatchLayout(state, state.vertexCount);
  const source = /* wgsl */ `
const VERTEX_COUNT: u32 = ${state.vertexCount}u;
${getBindingDeclarations(bindings)}

@compute @workgroup_size(${OPTIMIZATION_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(dispatchLayout, OPTIMIZATION_WORKGROUP_SIZE)}
  if (index >= VERTEX_COUNT || atomicLoad(&control[1u]) != 0u) { return; }
  let community = communities[${getViewElementOffset(state.output)}u + index];
  if (community >= VERTEX_COUNT) {
    atomicOr(&control[1u], ${INVALID_STATUS}u);
    return;
  }
  atomicStore(&control[5u + community], 1u);
}`;
  addOptimizationPass(commandGraph, {
    id: `${state.id}-iteration-${props.iteration}-mark-community-occupancy`,
    source,
    bindings,
    dispatchLayout
  });
}

/** Selects the globally lowest genuinely unoccupied stable community in linear GPU work. */
function addAvailableCommunitySelectionPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: {state: ImportedOptimization; iteration: number}
): void {
  const {state} = props;
  const bindings: Record<string, OptimizationBinding> = {
    control: {view: state.control, usage: 'storage-read-write', atomic: true}
  };
  const dispatchLayout = getDispatchLayout(state, state.vertexCount);
  const source = /* wgsl */ `
const VERTEX_COUNT: u32 = ${state.vertexCount}u;
${getBindingDeclarations(bindings)}

@compute @workgroup_size(${OPTIMIZATION_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(dispatchLayout, OPTIMIZATION_WORKGROUP_SIZE)}
  if (index >= VERTEX_COUNT || atomicLoad(&control[1u]) != 0u) { return; }
  if (atomicLoad(&control[5u + index]) == 0u) {
    atomicMin(&control[4u], index);
  }
}`;
  addOptimizationPass(commandGraph, {
    id: `${state.id}-iteration-${props.iteration}-select-unused-community`,
    source,
    bindings,
    dispatchLayout
  });
}

/** Accumulates exact directed degrees and community volumes from one original source chunk. */
function addEdgeStatisticsPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: {
    state: ImportedOptimization;
    iteration: number;
    chunkIndex: number;
    sources: GraphDataView<'uint32'>;
    targets: GraphDataView<'uint32'>;
    weights?: GraphDataView<'float32'>;
  }
): void {
  const {state} = props;
  const bindings: Record<string, OptimizationBinding> = {
    sourceVertices: {view: props.sources, usage: 'storage-read'},
    targetVertices: {view: props.targets, usage: 'storage-read'},
    ...(props.weights ? {edgeWeights: {view: props.weights, usage: 'storage-read' as const}} : {}),
    communities: {view: state.output, usage: 'storage-read'},
    statistics: {view: state.statistics, usage: 'storage-read-write', atomic: true},
    control: {view: state.control, usage: 'storage-read-write', atomic: true}
  };
  const weight = props.weights
    ? `edgeWeights[${getViewElementOffset(props.weights)}u + index]`
    : '1.0';
  const symmetricAccumulation = state.directed
    ? ''
    : `
  atomicAddFloat(&statistics[4u * targetVertex], edgeWeight);
  atomicAddFloat(&statistics[4u * sourceVertex + 1u], edgeWeight);
  atomicAddFloat(&statistics[4u * targetCommunity + 2u], edgeWeight);
  atomicAddFloat(&statistics[4u * sourceCommunity + 3u], edgeWeight);
  atomicAddFloat(&control[0u], edgeWeight);`;
  const dispatchLayout = getDispatchLayout(state, props.sources.length);
  const source = /* wgsl */ `
const EDGE_COUNT: u32 = ${props.sources.length}u;
const VERTEX_COUNT: u32 = ${state.vertexCount}u;
${getBindingDeclarations(bindings)}

${getAtomicFloatAdditionSource()}

@compute @workgroup_size(${OPTIMIZATION_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(dispatchLayout, OPTIMIZATION_WORKGROUP_SIZE)}
  if (index >= EDGE_COUNT || atomicLoad(&control[1u]) != 0u) { return; }
  let sourceVertex = sourceVertices[${getViewElementOffset(props.sources)}u + index];
  let targetVertex = targetVertices[${getViewElementOffset(props.targets)}u + index];
  if (sourceVertex >= VERTEX_COUNT || targetVertex >= VERTEX_COUNT) { return; }

  let edgeWeight = ${weight};
  if ((bitcast<u32>(edgeWeight) & 0x7fffffffu) >= 0x7f800000u || edgeWeight < 0.0) {
    atomicOr(&control[1u], ${INVALID_STATUS}u);
    return;
  }

  let sourceCommunity = communities[${getViewElementOffset(state.output)}u + sourceVertex];
  let targetCommunity = communities[${getViewElementOffset(state.output)}u + targetVertex];
  if (sourceCommunity >= VERTEX_COUNT || targetCommunity >= VERTEX_COUNT) {
    atomicOr(&control[1u], ${INVALID_STATUS}u);
    return;
  }

  atomicAddFloat(&statistics[4u * sourceVertex], edgeWeight);
  atomicAddFloat(&statistics[4u * targetVertex + 1u], edgeWeight);
  atomicAddFloat(&statistics[4u * sourceCommunity + 2u], edgeWeight);
  atomicAddFloat(&statistics[4u * targetCommunity + 3u], edgeWeight);
  atomicAddFloat(&control[0u], edgeWeight);
  ${symmetricAccumulation}
}`;
  addOptimizationPass(commandGraph, {
    id: `${state.id}-iteration-${props.iteration}-accumulate-chunk-${props.chunkIndex}`,
    source,
    bindings,
    dispatchLayout
  });
}

/** Narrows optional large scratch after device-limit-aware planning. */
function hasCommunityAggregation(state: ImportedOptimization): state is AggregatedOptimization {
  return Boolean(state.aggregatedCommunities && state.aggregatedWeights && state.aggregationRanges);
}

/** Aggregates each vertex's weak-neighbor weights once into a private open-addressed segment. */
function addCommunityAggregationPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: {state: AggregatedOptimization; iteration: number}
): void {
  const {state} = props;
  const bindings: Record<string, OptimizationBinding> = {
    forwardOffsets: {view: state.forwardOffsets, usage: 'storage-read'},
    packedForward: {view: state.packedForward, usage: 'storage-read'},
    ...(state.reverseOffsets && state.packedReverse
      ? {
          reverseOffsets: {view: state.reverseOffsets, usage: 'storage-read' as const},
          packedReverse: {view: state.packedReverse, usage: 'storage-read' as const}
        }
      : {}),
    communities: {view: state.output, usage: 'storage-read'},
    aggregatedCommunities: {view: state.aggregatedCommunities, usage: 'storage-write'},
    aggregatedWeights: {view: state.aggregatedWeights, usage: 'storage-write'},
    aggregationRanges: {view: state.aggregationRanges, usage: 'storage-write'}
  };
  const dispatchLayout = getDispatchLayout(state, state.vertexCount);
  const source = /* wgsl */ `
const VERTEX_COUNT: u32 = ${state.vertexCount}u;
const COMMUNITY_OFFSET: u32 = ${getViewElementOffset(state.output)}u;
const FORWARD_OFFSETS_OFFSET: u32 = ${getViewElementOffset(state.forwardOffsets)}u;
const FORWARD_CAPACITY: u32 = ${state.packedForward.length}u;
const DIRECTION_COUNT: u32 = ${state.directed ? 2 : 1}u;
${
  state.reverseOffsets && state.packedReverse
    ? `const REVERSE_OFFSETS_OFFSET: u32 = ${getViewElementOffset(state.reverseOffsets)}u;
const REVERSE_CAPACITY: u32 = ${state.packedReverse.length}u;`
    : ''
}
${getBindingDeclarations(bindings)}

${getNeighborAccessSource(state)}

fn getAggregationRange(vertex: u32) -> vec2<u32> {
  let forwardStart = getNeighborStart(vertex, 0u);
  let forwardDegree = getNeighborEnd(vertex, 0u) - forwardStart;
  var weakStart = forwardStart;
  var weakDegree = forwardDegree;
  ${
    state.directed
      ? `let reverseStart = getNeighborStart(vertex, 1u);
  weakStart += reverseStart;
  weakDegree += getNeighborEnd(vertex, 1u) - reverseStart;`
      : ''
  }
  let first = 2u * weakStart + vertex;
  return vec2<u32>(first, first + 2u * weakDegree + 1u);
}

fn hashCommunity(community: u32) -> u32 {
  var hash = community;
  hash = (hash ^ (hash >> 16u)) * 0x7feb352du;
  hash = (hash ^ (hash >> 15u)) * 0x846ca68bu;
  return hash ^ (hash >> 16u);
}

@compute @workgroup_size(${OPTIMIZATION_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(dispatchLayout, OPTIMIZATION_WORKGROUP_SIZE)}
  if (index >= VERTEX_COUNT) { return; }

  let range = getAggregationRange(index);
  aggregationRanges[index] = range;
  let segmentCapacity = range.y - range.x;
  for (var slot = range.x; slot < range.y; slot++) {
    aggregatedCommunities[slot] = ${INVALID_COMMUNITY}u;
    aggregatedWeights[slot] = vec2<f32>(0.0);
  }

  for (var direction = 0u; direction < DIRECTION_COUNT; direction++) {
    let first = getNeighborStart(index, direction);
    let last = getNeighborEnd(index, direction);
    for (var neighborSlot = first; neighborSlot < last; neighborSlot++) {
      let neighbor = getNeighbor(neighborSlot, direction);
      if (neighbor.x >= VERTEX_COUNT || neighbor.x == index) { continue; }
      let neighborCommunity = communities[COMMUNITY_OFFSET + neighbor.x];
      let edgeWeight = bitcast<f32>(neighbor.y);
      let contribution = select(
        vec2<f32>(edgeWeight, 0.0),
        vec2<f32>(0.0, edgeWeight),
        direction == 1u
      );
      var aggregateSlot = range.x + hashCommunity(neighborCommunity) % segmentCapacity;
      for (var attempt = 0u; attempt < segmentCapacity; attempt++) {
        let aggregateCommunity = aggregatedCommunities[aggregateSlot];
        if (aggregateCommunity == ${INVALID_COMMUNITY}u) {
          aggregatedCommunities[aggregateSlot] = neighborCommunity;
          aggregatedWeights[aggregateSlot] = contribution;
          break;
        }
        if (aggregateCommunity == neighborCommunity) {
          aggregatedWeights[aggregateSlot] += contribution;
          break;
        }
        aggregateSlot = range.x + (aggregateSlot - range.x + 1u) % segmentCapacity;
      }
    }
  }
}`;
  addOptimizationPass(commandGraph, {
    id: `${state.id}-iteration-${props.iteration}-aggregate-neighbor-communities`,
    source,
    bindings,
    dispatchLayout
  });
}

/** Scores each unique neighboring and unused community with at most seven storage bindings. */
function addAggregatedCandidatePass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: {state: AggregatedOptimization; iteration: number}
): void {
  const {state} = props;
  const bindings: Record<string, OptimizationBinding> = {
    communities: {view: state.output, usage: 'storage-read'},
    statistics: {view: state.statistics, usage: 'storage-read'},
    aggregatedCommunities: {view: state.aggregatedCommunities, usage: 'storage-read'},
    aggregatedWeights: {view: state.aggregatedWeights, usage: 'storage-read'},
    aggregationRanges: {view: state.aggregationRanges, usage: 'storage-read'},
    candidates: {view: state.candidates, usage: 'storage-write'},
    control: {view: state.control, usage: 'storage-read-write', atomic: true}
  };
  const dispatchLayout = getDispatchLayout(state, state.vertexCount);
  const source = /* wgsl */ `
const VERTEX_COUNT: u32 = ${state.vertexCount}u;
const COMMUNITY_OFFSET: u32 = ${getViewElementOffset(state.output)}u;
const RESOLUTION: f32 = ${state.resolution.toExponential()};
const MINIMUM_GAIN: f32 = ${state.minimumGain.toExponential()};
${getBindingDeclarations(bindings)}

fn isFiniteValue(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u;
}

fn isFiniteVector(value: vec4<f32>) -> bool {
  let magnitude = bitcast<vec4<u32>>(value) & vec4<u32>(0x7fffffffu);
  return all(magnitude < vec4<u32>(0x7f800000u));
}

@compute @workgroup_size(${OPTIMIZATION_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(dispatchLayout, OPTIMIZATION_WORKGROUP_SIZE)}
  if (index >= VERTEX_COUNT || atomicLoad(&control[1u]) != 0u) { return; }

  let total = bitcast<f32>(atomicLoad(&control[0u]));
  if (!isFiniteValue(total) || total <= 0.0) {
    atomicOr(&control[1u], ${INVALID_STATUS}u);
    return;
  }

  let currentCommunity = communities[COMMUNITY_OFFSET + index];
  if (currentCommunity >= VERTEX_COUNT) {
    atomicOr(&control[1u], ${INVALID_STATUS}u);
    return;
  }

  let vertexStatistics = statistics[index];
  let currentStatistics = statistics[currentCommunity];
  if (!isFiniteVector(vertexStatistics) || !isFiniteVector(currentStatistics)) {
    atomicOr(&control[1u], ${INVALID_STATUS}u);
    return;
  }

  let aggregationRange = aggregationRanges[index];
  var currentWeights = vec2<f32>(0.0);
  for (var slot = aggregationRange.x; slot < aggregationRange.y; slot++) {
    if (aggregatedCommunities[slot] == currentCommunity) {
      currentWeights = aggregatedWeights[slot];
      break;
    }
  }
  let currentOutgoingWeight = currentWeights.x;
  let currentIncomingWeight = ${state.directed ? 'currentWeights.y' : 'currentWeights.x'};
  var bestCommunity = currentCommunity;
  var bestGain = 0.0;

  for (var slot = aggregationRange.x; slot < aggregationRange.y; slot++) {
    let candidateCommunity = aggregatedCommunities[slot];
    if (candidateCommunity >= VERTEX_COUNT || candidateCommunity == currentCommunity) {
      continue;
    }

    let candidateStatistics = statistics[candidateCommunity];
    if (!isFiniteVector(candidateStatistics)) {
      atomicOr(&control[1u], ${INVALID_STATUS}u);
      return;
    }

    let weights = aggregatedWeights[slot];
    let outgoingWeight = weights.x;
    let incomingWeight = ${state.directed ? 'weights.y' : 'weights.x'};
    let observedGain =
      ((outgoingWeight - currentOutgoingWeight) +
       (incomingWeight - currentIncomingWeight)) / total;
    let expectedGain = RESOLUTION * (
      (vertexStatistics.y / total) *
        ((candidateStatistics.z - (currentStatistics.z - vertexStatistics.x)) / total) +
      (vertexStatistics.x / total) *
        ((candidateStatistics.w - (currentStatistics.w - vertexStatistics.y)) / total)
    );
    let gain = observedGain - expectedGain;
    if (!isFiniteValue(gain)) {
      atomicOr(&control[1u], ${INVALID_STATUS}u);
      return;
    }
    if (gain > MINIMUM_GAIN &&
        (gain > bestGain || (gain == bestGain && candidateCommunity < bestCommunity))) {
      bestGain = gain;
      bestCommunity = candidateCommunity;
    }
  }

  let availableCommunity = atomicLoad(&control[4u]);
  if (availableCommunity < VERTEX_COUNT && availableCommunity != currentCommunity) {
    let availableStatistics = statistics[availableCommunity];
    if (!isFiniteVector(availableStatistics)) {
      atomicOr(&control[1u], ${INVALID_STATUS}u);
      return;
    }
    let observedGain =
      ((0.0 - currentOutgoingWeight) + (0.0 - currentIncomingWeight)) / total;
    let expectedGain = RESOLUTION * (
      (vertexStatistics.y / total) *
        ((availableStatistics.z - (currentStatistics.z - vertexStatistics.x)) / total) +
      (vertexStatistics.x / total) *
        ((availableStatistics.w - (currentStatistics.w - vertexStatistics.y)) / total)
    );
    let gain = observedGain - expectedGain;
    if (!isFiniteValue(gain)) {
      atomicOr(&control[1u], ${INVALID_STATUS}u);
      return;
    }
    if (gain > MINIMUM_GAIN &&
        (gain > bestGain || (gain == bestGain && availableCommunity < bestCommunity))) {
      bestGain = gain;
      bestCommunity = availableCommunity;
    }
  }

  candidates[index] = vec2<u32>(bestCommunity, bitcast<u32>(bestGain));
  if (bestGain > MINIMUM_GAIN) {
    atomicMax(&control[2u], bitcast<u32>(bestGain));
  }
}`;
  addOptimizationPass(commandGraph, {
    id: `${state.id}-iteration-${props.iteration}-evaluate-candidates`,
    source,
    bindings,
    dispatchLayout
  });
}

/** Preserves the bounded no-extra-scratch path when a device cannot bind aggregation storage. */
function addRescanCandidatePass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: {state: ImportedOptimization; iteration: number}
): void {
  const {state} = props;
  const bindings: Record<string, OptimizationBinding> = {
    forwardOffsets: {view: state.forwardOffsets, usage: 'storage-read'},
    packedForward: {view: state.packedForward, usage: 'storage-read'},
    ...(state.reverseOffsets && state.packedReverse
      ? {
          reverseOffsets: {view: state.reverseOffsets, usage: 'storage-read' as const},
          packedReverse: {view: state.packedReverse, usage: 'storage-read' as const}
        }
      : {}),
    communities: {view: state.output, usage: 'storage-read'},
    statistics: {view: state.statistics, usage: 'storage-read'},
    candidates: {view: state.candidates, usage: 'storage-write'},
    control: {view: state.control, usage: 'storage-read-write', atomic: true}
  };
  const dispatchLayout = getDispatchLayout(state, state.vertexCount);
  const source = /* wgsl */ `
const VERTEX_COUNT: u32 = ${state.vertexCount}u;
const COMMUNITY_OFFSET: u32 = ${getViewElementOffset(state.output)}u;
const FORWARD_OFFSETS_OFFSET: u32 = ${getViewElementOffset(state.forwardOffsets)}u;
const FORWARD_CAPACITY: u32 = ${state.packedForward.length}u;
const DIRECTION_COUNT: u32 = ${state.directed ? 2 : 1}u;
const RESOLUTION: f32 = ${state.resolution.toExponential()};
const MINIMUM_GAIN: f32 = ${state.minimumGain.toExponential()};
${
  state.reverseOffsets && state.packedReverse
    ? `const REVERSE_OFFSETS_OFFSET: u32 = ${getViewElementOffset(state.reverseOffsets)}u;
const REVERSE_CAPACITY: u32 = ${state.packedReverse.length}u;`
    : ''
}
${getBindingDeclarations(bindings)}

fn isFiniteValue(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u;
}

fn isFiniteVector(value: vec4<f32>) -> bool {
  let magnitude = bitcast<vec4<u32>>(value) & vec4<u32>(0x7fffffffu);
  return all(magnitude < vec4<u32>(0x7f800000u));
}

${getNeighborAccessSource(state)}

fn getCommunityWeight(vertex: u32, community: u32, direction: u32) -> f32 {
  var weight = 0.0;
  let first = getNeighborStart(vertex, direction);
  let last = getNeighborEnd(vertex, direction);
  for (var slot = first; slot < last; slot++) {
    let neighbor = getNeighbor(slot, direction);
    if (neighbor.x < VERTEX_COUNT && neighbor.x != vertex &&
        communities[COMMUNITY_OFFSET + neighbor.x] == community) {
      weight += bitcast<f32>(neighbor.y);
    }
  }
  return weight;
}

@compute @workgroup_size(${OPTIMIZATION_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(dispatchLayout, OPTIMIZATION_WORKGROUP_SIZE)}
  if (index >= VERTEX_COUNT || atomicLoad(&control[1u]) != 0u) { return; }

  let total = bitcast<f32>(atomicLoad(&control[0u]));
  if (!isFiniteValue(total) || total <= 0.0) {
    atomicOr(&control[1u], ${INVALID_STATUS}u);
    return;
  }

  let currentCommunity = communities[COMMUNITY_OFFSET + index];
  if (currentCommunity >= VERTEX_COUNT) {
    atomicOr(&control[1u], ${INVALID_STATUS}u);
    return;
  }

  let vertexStatistics = statistics[index];
  let currentStatistics = statistics[currentCommunity];
  if (!isFiniteVector(vertexStatistics) || !isFiniteVector(currentStatistics)) {
    atomicOr(&control[1u], ${INVALID_STATUS}u);
    return;
  }

  let currentOutgoingWeight = getCommunityWeight(index, currentCommunity, 0u);
  let currentIncomingWeight = ${
    state.directed ? 'getCommunityWeight(index, currentCommunity, 1u)' : 'currentOutgoingWeight'
  };
  var bestCommunity = currentCommunity;
  var bestGain = 0.0;

  for (var direction = 0u; direction < DIRECTION_COUNT; direction++) {
    let first = getNeighborStart(index, direction);
    let last = getNeighborEnd(index, direction);
    for (var slot = first; slot < last; slot++) {
      let neighborVertex = getNeighbor(slot, direction).x;
      if (neighborVertex >= VERTEX_COUNT || neighborVertex == index) { continue; }
      let candidateCommunity = communities[COMMUNITY_OFFSET + neighborVertex];
      if (candidateCommunity >= VERTEX_COUNT || candidateCommunity == currentCommunity) {
        continue;
      }

      let candidateStatistics = statistics[candidateCommunity];
      if (!isFiniteVector(candidateStatistics)) {
        atomicOr(&control[1u], ${INVALID_STATUS}u);
        return;
      }

      let outgoingWeight = getCommunityWeight(index, candidateCommunity, 0u);
      let incomingWeight = ${
        state.directed ? 'getCommunityWeight(index, candidateCommunity, 1u)' : 'outgoingWeight'
      };
      let observedGain =
        ((outgoingWeight - currentOutgoingWeight) +
         (incomingWeight - currentIncomingWeight)) / total;
      let expectedGain = RESOLUTION * (
        (vertexStatistics.y / total) *
          ((candidateStatistics.z - (currentStatistics.z - vertexStatistics.x)) / total) +
        (vertexStatistics.x / total) *
          ((candidateStatistics.w - (currentStatistics.w - vertexStatistics.y)) / total)
      );
      let gain = observedGain - expectedGain;
      if (!isFiniteValue(gain)) {
        atomicOr(&control[1u], ${INVALID_STATUS}u);
        return;
      }
      if (gain > MINIMUM_GAIN &&
          (gain > bestGain || (gain == bestGain && candidateCommunity < bestCommunity))) {
        bestGain = gain;
        bestCommunity = candidateCommunity;
      }
    }
  }

  let availableCommunity = atomicLoad(&control[4u]);
  if (availableCommunity < VERTEX_COUNT && availableCommunity != currentCommunity) {
    let availableStatistics = statistics[availableCommunity];
    if (!isFiniteVector(availableStatistics)) {
      atomicOr(&control[1u], ${INVALID_STATUS}u);
      return;
    }
    let outgoingWeight = getCommunityWeight(index, availableCommunity, 0u);
    let incomingWeight = ${
      state.directed ? 'getCommunityWeight(index, availableCommunity, 1u)' : 'outgoingWeight'
    };
    let observedGain =
      ((outgoingWeight - currentOutgoingWeight) +
       (incomingWeight - currentIncomingWeight)) / total;
    let expectedGain = RESOLUTION * (
      (vertexStatistics.y / total) *
        ((availableStatistics.z - (currentStatistics.z - vertexStatistics.x)) / total) +
      (vertexStatistics.x / total) *
        ((availableStatistics.w - (currentStatistics.w - vertexStatistics.y)) / total)
    );
    let gain = observedGain - expectedGain;
    if (!isFiniteValue(gain)) {
      atomicOr(&control[1u], ${INVALID_STATUS}u);
      return;
    }
    if (gain > MINIMUM_GAIN &&
        (gain > bestGain || (gain == bestGain && availableCommunity < bestCommunity))) {
      bestGain = gain;
      bestCommunity = availableCommunity;
    }
  }

  candidates[index] = vec2<u32>(bestCommunity, bitcast<u32>(bestGain));
  if (bestGain > MINIMUM_GAIN) {
    atomicMax(&control[2u], bitcast<u32>(bestGain));
  }
}`;
  addOptimizationPass(commandGraph, {
    id: `${state.id}-iteration-${props.iteration}-evaluate-candidates-by-rescan`,
    source,
    bindings,
    dispatchLayout
  });
}

/** Resolves equal globally maximal positive gains by selecting the smallest stable vertex. */
function addWinnerSelectionPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: {state: ImportedOptimization; iteration: number}
): void {
  const {state} = props;
  const bindings: Record<string, OptimizationBinding> = {
    candidates: {view: state.candidates, usage: 'storage-read'},
    control: {view: state.control, usage: 'storage-read-write', atomic: true}
  };
  const dispatchLayout = getDispatchLayout(state, state.vertexCount);
  const source = /* wgsl */ `
const VERTEX_COUNT: u32 = ${state.vertexCount}u;
${getBindingDeclarations(bindings)}

@compute @workgroup_size(${OPTIMIZATION_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(dispatchLayout, OPTIMIZATION_WORKGROUP_SIZE)}
  if (index >= VERTEX_COUNT || atomicLoad(&control[1u]) != 0u) { return; }
  let maximumGain = atomicLoad(&control[2u]);
  if (maximumGain != 0u && candidates[index].y == maximumGain) {
    atomicMin(&control[3u], index);
  }
}`;
  addOptimizationPass(commandGraph, {
    id: `${state.id}-iteration-${props.iteration}-select-winner`,
    source,
    bindings,
    dispatchLayout
  });
}

/** Applies one race-free strictly improving move or records an irreversible local optimum. */
function addWinnerApplicationPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: {state: ImportedOptimization; iteration: number}
): void {
  const {state} = props;
  const bindings: Record<string, OptimizationBinding> = {
    communities: {view: state.output, usage: 'storage-read-write'},
    candidates: {view: state.candidates, usage: 'storage-read'},
    control: {view: state.control, usage: 'storage-read-write', atomic: true},
    ...(state.converged
      ? {converged: {view: state.converged, usage: 'storage-write' as const}}
      : {})
  };
  const publishConverged = state.converged
    ? `converged[${getViewElementOffset(state.converged)}u] = 1u;`
    : '';
  const publishUnconverged = state.converged
    ? `converged[${getViewElementOffset(state.converged)}u] = 0u;`
    : '';
  const source = /* wgsl */ `
${getBindingDeclarations(bindings)}

@compute @workgroup_size(1)
fn main() {
  if (atomicLoad(&control[1u]) != 0u) { return; }
  let winner = atomicLoad(&control[3u]);
  if (atomicLoad(&control[2u]) == 0u || winner == ${INVALID_COMMUNITY}u) {
    atomicOr(&control[1u], ${CONVERGED_STATUS}u);
    ${publishConverged}
    return;
  }
  communities[${getViewElementOffset(state.output)}u + winner] = candidates[winner].x;
  ${publishUnconverged}
}`;
  addOptimizationPass(commandGraph, {
    id: `${state.id}-iteration-${props.iteration}-apply-winner`,
    source,
    bindings,
    dispatchLayout: {x: 1, y: 1, z: 1}
  });
}

/** Publishes sentinel communities for invalid topology, weights, labels, or total edge volume. */
function addFailureFinalizationPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  state: ImportedOptimization
): void {
  const bindings: Record<string, OptimizationBinding> = {
    communities: {view: state.output, usage: 'storage-read-write'},
    control: {view: state.control, usage: 'storage-read-write', atomic: true},
    ...(state.converged
      ? {converged: {view: state.converged, usage: 'storage-read-write' as const}}
      : {})
  };
  const clearConvergence = state.converged
    ? `if (index == 0u && VERTEX_COUNT > 0u) {
    converged[${getViewElementOffset(state.converged)}u] = 0u;
  }`
    : '';
  const dispatchLayout = getDispatchLayout(state, Math.max(state.vertexCount, 1));
  const source = /* wgsl */ `
const VERTEX_COUNT: u32 = ${state.vertexCount}u;
${getBindingDeclarations(bindings)}

fn isFiniteValue(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u;
}

@compute @workgroup_size(${OPTIMIZATION_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(dispatchLayout, OPTIMIZATION_WORKGROUP_SIZE)}
  if (index >= max(VERTEX_COUNT, 1u)) { return; }
  let volume = bitcast<f32>(atomicLoad(&control[0u]));
  let invalid = (atomicLoad(&control[1u]) & ${INVALID_STATUS}u) != 0u ||
    !isFiniteValue(volume) || volume <= 0.0;
  if (!invalid) { return; }
  if (index < VERTEX_COUNT) {
    communities[${getViewElementOffset(state.output)}u + index] = ${INVALID_COMMUNITY}u;
  }
  ${clearConvergence}
}`;
  addOptimizationPass(commandGraph, {
    id: `${state.id}-finalize-failures`,
    source,
    bindings,
    dispatchLayout
  });
}

/** Exposes packed direction-independent CSR access while preserving existing slice offsets. */
function getNeighborAccessSource(state: ImportedOptimization): string {
  const reverse = state.reverseOffsets && state.packedReverse;
  return /* wgsl */ `
fn getNeighborStart(vertex: u32, direction: u32) -> u32 {
  ${
    reverse
      ? `if (direction == 1u) {
    return min(reverseOffsets[REVERSE_OFFSETS_OFFSET + vertex], REVERSE_CAPACITY);
  }`
      : ''
  }
  return min(forwardOffsets[FORWARD_OFFSETS_OFFSET + vertex], FORWARD_CAPACITY);
}

fn getNeighborEnd(vertex: u32, direction: u32) -> u32 {
  ${
    reverse
      ? `if (direction == 1u) {
    return min(reverseOffsets[REVERSE_OFFSETS_OFFSET + vertex + 1u], REVERSE_CAPACITY);
  }`
      : ''
  }
  return min(forwardOffsets[FORWARD_OFFSETS_OFFSET + vertex + 1u], FORWARD_CAPACITY);
}

fn getNeighbor(slot: u32, direction: u32) -> vec2<u32> {
  ${
    reverse
      ? `if (direction == 1u) {
    return packedReverse[slot];
  }`
      : ''
  }
  return packedForward[slot];
}`;
}

/** Implements portable floating-point addition with universally supported unsigned atomics. */
function getAtomicFloatAdditionSource(): string {
  return /* wgsl */ `
fn atomicAddFloat(destination: ptr<storage, atomic<u32>, read_write>, value: f32) {
  var previousBits = atomicLoad(destination);
  loop {
    let nextBits = bitcast<u32>(bitcast<f32>(previousBits) + value);
    let exchange = atomicCompareExchangeWeak(destination, previousBits, nextBits);
    if (exchange.exchanged) { return; }
    previousBits = exchange.old_value;
  }
}`;
}

/** Declares packed scalar, tuple, float-vector, and portable atomic storage bindings. */
function getBindingDeclarations(bindings: Record<string, OptimizationBinding>): string {
  return Object.entries(bindings)
    .map(([name, binding], location) => {
      const access = binding.usage === 'storage-read' ? 'read' : 'read_write';
      let element = 'u32';
      if (binding.atomic) {
        element = 'atomic<u32>';
      } else if (binding.view.format === 'float32') {
        element = 'f32';
      } else if (binding.view.format === 'float32x2') {
        element = 'vec2<f32>';
      } else if (binding.view.format === 'uint32x2') {
        element = 'vec2<u32>';
      } else if (binding.view.format === 'float32x4') {
        element = 'vec4<f32>';
      }
      return `@group(0) @binding(${location}) var<storage, ${access}> ${name}: array<${element}>;`;
    })
    .join('\n');
}

/** Compiles one bounded GPU graph node without queue submission, mapping, or CPU readback. */
function addOptimizationPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: OptimizationPass
): void {
  commandGraph.addComputePass({
    id: props.id,
    resources: Object.values(props.bindings).map(({view, usage}) => ({buffer: view, usage})),
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
          for (const [name, binding] of Object.entries(props.bindings)) {
            shaderBindings[name] = getViewBinding(binding.view, getBuffer);
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

/** Resolves one operation-local multidimensional dispatch using its imported device limit. */
function getDispatchLayout(
  state: ImportedOptimization,
  elementCount: number
): GPUBoundedDispatchLayout {
  return getGPUGraphModularityOptimizationDispatchLayout(
    elementCount,
    state.maxComputeWorkgroupsPerDimension
  );
}

/** Sizes disjoint half-full hash segments without overflowing WGSL's uint32 addressing. */
export function getCommunityAggregationCapacity(
  vertexCount: number,
  forwardCapacity: number,
  reverseCapacity: number
): number | undefined {
  if (
    !Number.isSafeInteger(vertexCount) ||
    vertexCount < 0 ||
    !Number.isSafeInteger(forwardCapacity) ||
    forwardCapacity < 0 ||
    !Number.isSafeInteger(reverseCapacity) ||
    reverseCapacity < 0
  ) {
    throw new Error('Community aggregation sizes must be non-negative safe integers');
  }
  const capacity = 2 * (forwardCapacity + reverseCapacity) + vertexCount;
  if (!Number.isSafeInteger(capacity) || capacity > 0xffffffff) {
    return undefined;
  }
  return capacity;
}

/** Plans bounded, true three-dimensional adjacency, edge, vertex, or finalization work. */
export function getGPUGraphModularityOptimizationDispatchLayout(
  elementCount: number,
  maxComputeWorkgroupsPerDimension: number
): GPUBoundedDispatchLayout {
  return getBoundedDispatchLayout(
    'GPUGraphModularityOptimization',
    elementCount,
    OPTIMIZATION_WORKGROUP_SIZE,
    maxComputeWorkgroupsPerDimension
  );
}
