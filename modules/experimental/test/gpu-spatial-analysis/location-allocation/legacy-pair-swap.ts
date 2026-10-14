// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {IS_NAN_BITS_WGSL} from '../../../src/utils/wgsl-common';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../../src/utils/wgsl-kernel-nodes';

/**
 * The swap phase of `GPULocationAllocation` before the Whitaker / Resende-Werneck decomposition,
 * kept only as the "old path" of the swap A/B spec: one thread per (selected slot, candidate) pair
 * scans every demand row (`O(maximumFacilityCount * facilityCount * demandCount)` per iteration),
 * with the same gating, tolerance and tie rules. Dense `p-median` only. Not part of the library.
 */
export function createLegacyPairSwapNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    costs: GraphDataView<'float32'>;
    demandWeights?: GraphDataView<'float32'>;
    facilityCount: number;
    demandCount: number;
    maximumFacilityCount: number;
    maximumSwapIterations: number;
    selectedFacilities: GraphDataView<'uint32'>;
    status: GraphDataView<'uint32'>;
  }
): {nodes: GPUCommandNode<Parameters>[]; swapState: GraphDataView<'uint32'>} {
  const {id, facilityCount, demandCount, costs, selectedFacilities, status} = props;
  const slots = props.maximumFacilityCount;
  const pairCount = slots * facilityCount;
  const workgroupSize = 256;
  const declarations = `const DEMAND_COUNT: u32 = ${demandCount}u;
const FACILITY_COUNT: u32 = ${facilityCount}u;
const NONE: u32 = 0xffffffffu;
const LARGE_COST: f32 = 1e30;
const SWAP_SLOTS: u32 = ${slots}u;
const SWAP_TOLERANCE: f32 = 1e-6;
${IS_NAN_BITS_WGSL}
fn isReachable(cost: f32) -> bool { return cost < LARGE_COST; }`;
  const view = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, `${id}-${name}`, format, length);
  const demandData = view('demand-data', 'uint32', demandCount * 3);
  const selectedList = view('selected-list', 'uint32', slots);
  const pairScores = view('pair-scores', 'float32', 2 * pairCount);
  const swapState = view('swap-state', 'uint32', 3);
  const weightBinding: WGSLKernelBinding[] = props.demandWeights
    ? [{name: 'demandWeights', view: props.demandWeights, type: 'f32', access: 'read'}]
    : [];
  const weight = props.demandWeights ? 'demandWeights[demandWeightsOffset + demand]' : '1.0';
  const gate = createWGSLDispatchGate(graph, {
    id: `${id}-dispatch`,
    operation: 'LegacyPairSwap',
    shapes: [[demandCount], [1], [pairCount], [workgroupSize, workgroupSize]]
  });
  const operation = 'LegacyPairSwap';
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode(graph, {
      id: `${id}-init`,
      operation,
      invocationCount: 1,
      bindings: [
        {name: 'swapState', view: swapState, type: 'u32', access: 'read_write'},
        gate.binding
      ],
      declarations,
      body: `swapState[swapStateOffset] = 0u;
  swapState[swapStateOffset + 1u] = 0u;
  swapState[swapStateOffset + 2u] = NONE;
  ${gate.armWGSL}`
    })
  ];
  for (let iteration = 0; iteration < props.maximumSwapIterations; iteration++) {
    const iterationId = `${id}-${iteration}`;
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${iterationId}-prepare`,
        operation,
        invocationCount: demandCount,
        ...gate.getGate(0),
        bindings: [
          {name: 'costs', view: costs, type: 'f32', access: 'read'},
          {name: 'selected', view: selectedFacilities, type: 'u32', access: 'read'},
          {name: 'swapState', view: swapState, type: 'u32', access: 'read'},
          {name: 'demandData', view: demandData, type: 'u32', access: 'read_write'}
        ],
        declarations,
        body: `if (swapState[swapStateOffset] != 0u) {
  return;
}
let demand = index;
var nearestCost = LARGE_COST;
var secondCost = LARGE_COST;
var nearestFacility = NONE;
for (var facility = 0u; facility < FACILITY_COUNT; facility++) {
  if (selected[selectedOffset + facility] == 0u) { continue; }
  let cost = min(costs[costsOffset + demand * FACILITY_COUNT + facility], LARGE_COST);
  if (cost < nearestCost) {
    secondCost = nearestCost;
    nearestCost = cost;
    nearestFacility = facility;
  } else if (cost < secondCost) {
    secondCost = cost;
  }
}
demandData[demandDataOffset + demand * 3u] = bitcast<u32>(nearestCost);
demandData[demandDataOffset + demand * 3u + 1u] = bitcast<u32>(secondCost);
demandData[demandDataOffset + demand * 3u + 2u] = nearestFacility;`
      }),
      createWGSLKernelNode(graph, {
        id: `${iterationId}-list`,
        operation,
        invocationCount: 1,
        ...gate.getGate(1),
        bindings: [
          {name: 'selected', view: selectedFacilities, type: 'u32', access: 'read'},
          {name: 'status', view: status, type: 'u32', access: 'read'},
          {name: 'swapState', view: swapState, type: 'u32', access: 'read'},
          {name: 'selectedList', view: selectedList, type: 'u32', access: 'read_write'}
        ],
        declarations,
        body: `for (var slot = 0u; slot < SWAP_SLOTS; slot++) {
  selectedList[selectedListOffset + slot] = NONE;
}
if (swapState[swapStateOffset] != 0u || status[statusOffset + 3u] != 0u) {
  return;
}
var slot = 0u;
for (var facility = 0u; facility < FACILITY_COUNT; facility++) {
  if (selected[selectedOffset + facility] != 0u && slot < SWAP_SLOTS) {
    selectedList[selectedListOffset + slot] = facility;
    slot++;
  }
}`
      }),
      createWGSLKernelNode(graph, {
        id: `${iterationId}-score`,
        operation,
        invocationCount: pairCount,
        ...gate.getGate(2),
        bindings: [
          {name: 'costs', view: costs, type: 'f32', access: 'read'},
          {name: 'selected', view: selectedFacilities, type: 'u32', access: 'read'},
          {name: 'demandData', view: demandData, type: 'u32', access: 'read'},
          {name: 'selectedList', view: selectedList, type: 'u32', access: 'read'},
          {name: 'pairScores', view: pairScores, type: 'f32', access: 'read_write'},
          ...weightBinding
        ],
        declarations,
        body: `let slot = index / FACILITY_COUNT;
let candidate = index % FACILITY_COUNT;
let removed = selectedList[selectedListOffset + slot];
pairScores[pairScoresOffset + 2u * index] = -1.0;
pairScores[pairScoresOffset + 2u * index + 1u] = -1.0;
if (removed == NONE || selected[selectedOffset + candidate] != 0u) {
  return;
}
var improvement = 0.0;
var scale = 0.0;
var reachGain = 0.0;
var totalWeight = 0.0;
for (var demand = 0u; demand < DEMAND_COUNT; demand++) {
  let weight = ${weight};
  let candidateCost = costs[costsOffset + demand * FACILITY_COUNT + candidate];
  let current = bitcast<f32>(demandData[demandDataOffset + demand * 3u]);
  let second = bitcast<f32>(demandData[demandDataOffset + demand * 3u + 1u]);
  let base = select(current, second, demandData[demandDataOffset + demand * 3u + 2u] == removed);
  let next = min(base, min(candidateCost, LARGE_COST));
  let currentReachable = isReachable(current);
  let nextReachable = isReachable(next);
  reachGain += weight * (select(0.0, 1.0, nextReachable) - select(0.0, 1.0, currentReachable));
  improvement += select(0.0, weight * current, currentReachable) - select(0.0, weight * next, nextReachable);
  totalWeight += weight;
  if (currentReachable) { scale += weight * current; }
}
if (reachGain > SWAP_TOLERANCE * totalWeight) {
  pairScores[pairScoresOffset + 2u * index] = reachGain;
  pairScores[pairScoresOffset + 2u * index + 1u] = improvement;
} else if (abs(reachGain) <= SWAP_TOLERANCE * totalWeight && improvement > SWAP_TOLERANCE * scale) {
  pairScores[pairScoresOffset + 2u * index] = 0.0;
  pairScores[pairScoresOffset + 2u * index + 1u] = improvement;
}`
      }),
      createWGSLKernelNode(graph, {
        id: `${iterationId}-choose`,
        operation,
        invocationCount: workgroupSize,
        workgroupSize,
        guardIndex: false,
        ...gate.getGate(3),
        bindings: [
          {name: 'pairScores', view: pairScores, type: 'f32', access: 'read'},
          {name: 'swapState', view: swapState, type: 'u32', access: 'read_write'}
        ],
        declarations: `${declarations}
const PAIR_COUNT: u32 = ${pairCount}u;
const SWAP_WORKGROUP_SIZE: u32 = ${workgroupSize}u;
var<workgroup> sharedScore: array<f32, ${workgroupSize}>;
var<workgroup> sharedSecondary: array<f32, ${workgroupSize}>;
var<workgroup> sharedPair: array<u32, ${workgroupSize}>;
fn isBetterPair(
  score: f32, secondary: f32, pair: u32, bestScore: f32, bestSecondary: f32, bestPair: u32
) -> bool {
  if (score != bestScore) { return score > bestScore; }
  if (secondary != bestSecondary) { return secondary > bestSecondary; }
  return pair < bestPair;
}`,
        body: `var bestPair = NONE;
var bestScore = -1.0;
var bestSecondary = -1.0;
if (swapState[swapStateOffset] == 0u) {
  for (var pair = localInvocationIndex; pair < PAIR_COUNT; pair += SWAP_WORKGROUP_SIZE) {
    let score = pairScores[pairScoresOffset + 2u * pair];
    let secondary = pairScores[pairScoresOffset + 2u * pair + 1u];
    if (isBetterPair(score, secondary, pair, bestScore, bestSecondary, bestPair)) {
      bestPair = pair;
      bestScore = score;
      bestSecondary = secondary;
    }
  }
}
sharedScore[localInvocationIndex] = bestScore;
sharedSecondary[localInvocationIndex] = bestSecondary;
sharedPair[localInvocationIndex] = bestPair;
workgroupBarrier();
for (var stride = SWAP_WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
  if (localInvocationIndex < stride &&
      isBetterPair(
        sharedScore[localInvocationIndex + stride],
        sharedSecondary[localInvocationIndex + stride],
        sharedPair[localInvocationIndex + stride],
        sharedScore[localInvocationIndex],
        sharedSecondary[localInvocationIndex],
        sharedPair[localInvocationIndex])) {
    sharedScore[localInvocationIndex] = sharedScore[localInvocationIndex + stride];
    sharedSecondary[localInvocationIndex] = sharedSecondary[localInvocationIndex + stride];
    sharedPair[localInvocationIndex] = sharedPair[localInvocationIndex + stride];
  }
  workgroupBarrier();
}
if (localInvocationIndex == 0u) {
  swapState[swapStateOffset + 2u] = select(NONE, sharedPair[0], sharedScore[0] >= 0.0);
}`
      }),
      createWGSLKernelNode(graph, {
        id: `${iterationId}-apply`,
        operation,
        invocationCount: 1,
        bindings: [
          {name: 'selected', view: selectedFacilities, type: 'u32', access: 'read_write'},
          {name: 'selectedList', view: selectedList, type: 'u32', access: 'read'},
          {name: 'swapState', view: swapState, type: 'u32', access: 'read_write'},
          gate.binding
        ],
        declarations,
        body: `if (swapState[swapStateOffset] != 0u) {
  return;
}
let pair = swapState[swapStateOffset + 2u];
if (pair == NONE) {
  swapState[swapStateOffset] = 1u;
  ${gate.disarmWGSL}
  return;
}
selected[selectedOffset + selectedList[selectedListOffset + pair / FACILITY_COUNT]] = 0u;
selected[selectedOffset + pair % FACILITY_COUNT] = 1u;
swapState[swapStateOffset + 1u] = swapState[swapStateOffset + 1u] + 1u;`
      })
    );
  }
  return {nodes, swapState};
}
