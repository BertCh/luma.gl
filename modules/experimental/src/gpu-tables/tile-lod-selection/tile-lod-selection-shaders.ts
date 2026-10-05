// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {GPU_TILE_LOD_VIEW_OFFSETS as VIEW} from './tile-lod-view';

const OPERATION = 'GPUTileLODSelection';

/** uint32 words per node record. @internal */
export const TILE_LOD_RECORD_WORDS = 4;
/** uint32 words in the shared state block. @internal */
export const TILE_LOD_STATE_LENGTH = 12;
/** uint32 words of budget scratch per level: cost, count, threshold, 32 buckets x 4. @internal */
export const TILE_LOD_LEVEL_SCRATCH_WORDS = 131;

/** Flag bits, state slots, and helpers shared by every tile LOD kernel. */
const TILE_LOD_CONSTANTS = /* wgsl */ `
const F32_MAX: f32 = 3.402823466e+38;
const MINIMUM_SURFACE_DISTANCE: f32 = 1e-6;
const FLAG_RESIDENT: u32 = 1u;
const FLAG_ENABLED: u32 = 2u;
const FLAG_VISIBLE: u32 = 4u;
const FLAG_WANTS_REFINE: u32 = 8u;
const FLAG_REFINED: u32 = 16u;
const FLAG_CHILDREN_RESIDENT: u32 = 32u;
const FLAG_BUDGET_REJECTED: u32 = 64u;
const BUCKET_SHIFT: u32 = 8u;
const STATE_DESIRED_COUNT: u32 = 0u;
const STATE_DESIRED_COST: u32 = 1u;
const STATE_DRAWN_COUNT: u32 = 2u;
const STATE_DRAWN_COST: u32 = 3u;
const STATE_REQUESTED_COUNT: u32 = 4u;
const STATE_BUDGET_EXHAUSTED: u32 = 5u;
const STATE_VISIBLE_COUNT: u32 = 6u;
const STATE_COMMITTED_COUNT: u32 = 8u;
const STATE_COMMITTED_COST: u32 = 9u;
fn addSaturated(left: u32, right: u32) -> u32 {
  let sum = left + right;
  return select(sum, 0xffffffffu, sum < left);
}`;

/** Static per-recipe configuration used to generate kernels. @internal */
export type TileLODShaderConfig = {
  id: string;
  nodeCount: number;
  rootCount: number;
  levelOffsets: readonly number[];
  refinementAdd: boolean;
  hierarchy: {
    sphereBounds: GraphDataView<'float32x4'>;
    geometricErrors: GraphDataView<'float32'>;
    children: GraphDataView<'uint32x2'>;
    nodeCosts?: GraphDataView<'uint32'>;
  };
  view: GraphDataView<'float32'>;
  nodeRecords: GraphDataView<'uint32'>;
  state: GraphDataView<'uint32'>;
  levelScratch?: GraphDataView<'uint32'>;
};

/** Returns level topology constants, matching `GPUVirtualGeometrySelection`. */
function getLevelConstants(config: TileLODShaderConfig, level: number): string {
  const {levelOffsets} = config;
  const levelCount = levelOffsets.length - 1;
  const hasNextLevel = level + 1 < levelCount;
  return `const FIRST_NODE: u32 = ${levelOffsets[level]}u;
const HAS_NEXT_LEVEL: bool = ${hasNextLevel};
const NEXT_LEVEL_FIRST: u32 = ${hasNextLevel ? levelOffsets[level + 1] : 0}u;
const NEXT_LEVEL_END: u32 = ${hasNextLevel ? levelOffsets[level + 2] : 0}u;
const REFINEMENT_ADD: bool = ${config.refinementAdd};
const LEVEL_BASE: u32 = ${level * TILE_LOD_LEVEL_SCRATCH_WORDS}u;`;
}

/** WGSL functions for visibility, projected error, foveation, and node evaluation. */
function getEvaluationSource(hasCosts: boolean): string {
  return /* wgsl */ `
fn finite(value: f32) -> bool { return value == value && abs(value) <= F32_MAX; }
fn readView(slot: u32) -> f32 { return view[viewOffset + slot]; }
fn getCenter(node: u32) -> vec3f {
  let base = sphereBoundsOffset + node * 4u;
  return vec3f(sphereBounds[base], sphereBounds[base + 1u], sphereBounds[base + 2u]);
}
fn getRadius(node: u32) -> f32 { return sphereBounds[sphereBoundsOffset + node * 4u + 3u]; }
fn hasValidBounds(node: u32) -> bool {
  let center = getCenter(node);
  let radius = getRadius(node);
  return finite(center.x) && finite(center.y) && finite(center.z) && finite(radius) && radius >= 0.0;
}
fn getCamera() -> vec3f {
  return vec3f(readView(${VIEW.cameraPosition}u), readView(${VIEW.cameraPosition + 1}u), readView(${VIEW.cameraPosition + 2}u));
}
fn isSphereVisible(center: vec3f, radius: f32) -> bool {
  for (var planeIndex = 0u; planeIndex < 6u; planeIndex++) {
    let base = ${VIEW.frustumPlanes}u + planeIndex * 4u;
    let plane = vec4f(readView(base), readView(base + 1u), readView(base + 2u), readView(base + 3u));
    let validPlane = finite(plane.x) && finite(plane.y) && finite(plane.z) && finite(plane.w);
    if (validPlane && dot(plane.xyz, center) + plane.w < -radius) { return false; }
  }
  return true;
}
fn readFlags(node: u32) -> u32 {
  return atomicLoad(&nodeRecords[nodeRecordsOffset + node * 4u + 1u]);
}
fn isNodeVisible(node: u32) -> bool {
  if ((readFlags(node) & FLAG_ENABLED) == 0u) { return false; }
  return !hasValidBounds(node) || isSphereVisible(getCenter(node), getRadius(node));
}
fn getNodeCost(node: u32) -> u32 { return ${hasCosts ? 'nodeCosts[nodeCostsOffset + node]' : '0u'}; }

struct ErrorValue { value: f32, valid: bool }

fn getProjectedError(node: u32) -> ErrorValue {
  let center = getCenter(node);
  let radius = getRadius(node);
  let camera = getCamera();
  let scale = readView(${VIEW.pixelProjectionScale}u);
  let geometricError = geometricErrors[geometricErrorsOffset + node];
  let validView = hasValidBounds(node) && finite(camera.x) && finite(camera.y) && finite(camera.z) &&
    finite(scale) && scale >= 0.0 && finite(geometricError) && geometricError >= 0.0;
  if (!validView) { return ErrorValue(F32_MAX, false); }
  let surface = distance(camera, center) - radius;
  if (surface <= MINIMUM_SURFACE_DISTANCE) { return ErrorValue(F32_MAX, false); }
  let error = geometricError * scale / surface;
  return ErrorValue(error, finite(error));
}

fn getWeightedError(node: u32, projected: ErrorValue) -> ErrorValue {
  if (!projected.valid) { return projected; }
  var error = projected.value;
  let center = getCenter(node);
  let radius = getRadius(node);
  let camera = getCamera();
  let strength = readView(${VIEW.foveationStrength}u);
  if (finite(strength) && strength > 0.0) {
    let cameraDistance = distance(camera, center);
    let m = ${VIEW.viewProjectionMatrix}u;
    let clip = vec4f(readView(m), readView(m + 1u), readView(m + 2u), readView(m + 3u)) * center.x +
      vec4f(readView(m + 4u), readView(m + 5u), readView(m + 6u), readView(m + 7u)) * center.y +
      vec4f(readView(m + 8u), readView(m + 9u), readView(m + 10u), readView(m + 11u)) * center.z +
      vec4f(readView(m + 12u), readView(m + 13u), readView(m + 14u), readView(m + 15u));
    if (cameraDistance > radius && clip.w > 0.0) {
      let width = max(readView(${VIEW.viewportSize}u), 1.0);
      let height = max(readView(${VIEW.viewportSize + 1}u), 1.0);
      let screen = vec2f(clip.x / clip.w * 0.5 + 0.5, 0.5 - clip.y / clip.w * 0.5);
      let offset = screen - vec2f(readView(${VIEW.foveationCenter}u), readView(${VIEW.foveationCenter + 1}u));
      let gaze = length(offset);
      let projectedRadius = radius * readView(${VIEW.pixelProjectionScale}u) / (cameraDistance * height);
      let aspect = width / height;
      var toward = projectedRadius;
      if (gaze > 0.0) {
        toward = projectedRadius / length(vec2f(offset.x / gaze * aspect, offset.y / gaze));
      }
      let peripheral = max(gaze - toward - max(readView(${VIEW.foveationRadius}u), 0.0), 0.0);
      error = error / (1.0 + peripheral * strength);
    }
  }
  let focus = readView(${VIEW.focusDistance}u);
  let falloff = readView(${VIEW.distanceFalloff}u);
  if (finite(falloff) && falloff > 0.0 && finite(focus) && focus > 0.0) {
    let surface = distance(camera, center) - radius;
    error = error / pow(max(surface / focus, 1.0), falloff);
  }
  return ErrorValue(error, finite(error));
}

fn getPriorityBucket(error: ErrorValue) -> u32 {
  if (!error.valid) { return 31u; }
  return u32(clamp(i32(floor(log2(max(error.value, 0.00390625)))) + 8, 0, 31));
}

struct Evaluation {
  wantsRefine: bool,
  childrenResident: bool,
  visibleChildCount: u32,
  visibleChildCost: u32,
  priority: f32,
  bucket: u32,
  firstChild: u32,
  childCount: u32
}

// Evaluates a visible active node: refinement intent, child residency, and priority.
fn evaluateNode(node: u32) -> Evaluation {
  let firstChild = children[childrenOffset + node * 2u];
  let childCount = children[childrenOffset + node * 2u + 1u];
  let validChildRange = HAS_NEXT_LEVEL && childCount > 0u &&
    firstChild >= NEXT_LEVEL_FIRST && firstChild <= NEXT_LEVEL_END &&
    childCount <= NEXT_LEVEL_END - firstChild;
  let weighted = getWeightedError(node, getProjectedError(node));
  let threshold = readView(${VIEW.maximumScreenSpaceError}u);
  let refineForView = !weighted.valid || !finite(threshold) || threshold < 0.0 || weighted.value > threshold;
  var childrenResident = true;
  var visibleChildCount = 0u;
  var visibleChildCost = 0u;
  if (validChildRange) {
    for (var child = firstChild; child < firstChild + childCount; child++) {
      if (isNodeVisible(child)) {
        visibleChildCount += 1u;
        visibleChildCost = addSaturated(visibleChildCost, getNodeCost(child));
        if ((readFlags(child) & FLAG_RESIDENT) == 0u) { childrenResident = false; }
      }
    }
  }
  return Evaluation(
    validChildRange && refineForView,
    childrenResident,
    visibleChildCount,
    visibleChildCost,
    select(F32_MAX, weighted.value, weighted.valid),
    getPriorityBucket(weighted),
    select(0u, firstChild, validChildRange),
    select(0u, childCount, validChildRange)
  );
}

// Stores the evaluation flags and priority word of a visible node.
fn writeEvaluation(node: u32, evaluation: Evaluation) {
  var flags = FLAG_VISIBLE | (evaluation.bucket << BUCKET_SHIFT);
  if (evaluation.wantsRefine) { flags |= FLAG_WANTS_REFINE; }
  if (evaluation.childrenResident) { flags |= FLAG_CHILDREN_RESIDENT; }
  atomicOr(&nodeRecords[nodeRecordsOffset + node * 4u + 1u], flags);
  atomicStore(&nodeRecords[nodeRecordsOffset + node * 4u + 2u], bitcast<u32>(evaluation.priority));
}`;
}

/** WGSL that applies one refinement decision and activates children. */
function getDecisionSource(budgeted: boolean): string {
  return /* wgsl */ `
fn applyDecision(node: u32, wantsRefine: bool, childrenResident: bool, firstChild: u32, childCount: u32, refine: bool) {
  let recordBase = nodeRecordsOffset + node * 4u;
  if (wantsRefine && !refine) { atomicOr(&nodeRecords[recordBase + 1u], FLAG_BUDGET_REJECTED); }
  if (refine) {
    atomicOr(&nodeRecords[recordBase + 1u], FLAG_REFINED);
    let nodeState = atomicLoad(&nodeRecords[recordBase]);
    let childState = select(1u, 2u, nodeState == 2u && (REFINEMENT_ADD || childrenResident));
    for (var child = firstChild; child < firstChild + childCount; child++) {
      atomicMax(&nodeRecords[nodeRecordsOffset + child * 4u], childState);
    }
  }
  ${
    budgeted
      ? `if (!refine || REFINEMENT_ADD) {
    atomicAdd(&state[stateOffset + STATE_COMMITTED_COUNT], 1u);
    atomicAdd(&state[stateOffset + STATE_COMMITTED_COST], getNodeCost(node));
  }`
      : ''
  }
}`;
}

/** Builds the prepare node that resets records, state, and budget scratch. @internal */
export function createTileLODPrepareNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  config: TileLODShaderConfig,
  residency?: GraphDataView<'uint32'>,
  enabledNodes?: GraphDataView<'uint32'>
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'nodeRecords', view: config.nodeRecords, type: 'u32', access: 'read_write'},
    {name: 'state', view: config.state, type: 'u32', access: 'read_write'}
  ];
  if (config.levelScratch) {
    bindings.push({
      name: 'levelScratch',
      view: config.levelScratch,
      type: 'u32',
      access: 'read_write'
    });
  }
  if (residency) bindings.push({name: 'residency', view: residency, type: 'u32', access: 'read'});
  if (enabledNodes) {
    bindings.push({name: 'enabledNodes', view: enabledNodes, type: 'u32', access: 'read'});
  }
  const scratchLength = config.levelScratch?.length ?? 0;
  return createWGSLKernelNode<Parameters>(graph, {
    id: `${config.id}-prepare`,
    operation: OPERATION,
    variant: 'prepare',
    bindings,
    invocationCount: Math.max(config.nodeCount, scratchLength, TILE_LOD_STATE_LENGTH),
    declarations: TILE_LOD_CONSTANTS,
    body: `if (index < ${config.nodeCount}u) {
    let recordBase = nodeRecordsOffset + index * 4u;
    var flags = 0u;
    if (${residency ? 'residency[residencyOffset + index] != 0u' : 'true'}) { flags |= FLAG_RESIDENT; }
    if (${enabledNodes ? 'enabledNodes[enabledNodesOffset + index] != 0u' : 'true'}) { flags |= FLAG_ENABLED; }
    nodeRecords[recordBase] = select(0u, 2u, index < ${config.rootCount}u);
    nodeRecords[recordBase + 1u] = flags;
    nodeRecords[recordBase + 2u] = 0u;
    nodeRecords[recordBase + 3u] = 0u;
  }
  if (index < ${TILE_LOD_STATE_LENGTH}u) { state[stateOffset + index] = 0u; }
  ${
    config.levelScratch
      ? `if (index < ${scratchLength}u) {
    levelScratch[levelScratchOffset + index] = select(0u, 32u, index % ${TILE_LOD_LEVEL_SCRATCH_WORDS}u == 2u);
  }`
      : ''
  }`
  });
}

/** Returns the read-only hierarchy bindings shared by evaluation kernels. */
function getHierarchyBindings(
  config: TileLODShaderConfig,
  withCosts: boolean
): WGSLKernelBinding[] {
  const {hierarchy} = config;
  const bindings: WGSLKernelBinding[] = [
    {name: 'sphereBounds', view: hierarchy.sphereBounds, type: 'f32', access: 'read'},
    {name: 'geometricErrors', view: hierarchy.geometricErrors, type: 'f32', access: 'read'},
    {name: 'children', view: hierarchy.children, type: 'u32', access: 'read'},
    {name: 'view', view: config.view, type: 'f32', access: 'read'}
  ];
  if (withCosts && hierarchy.nodeCosts) {
    bindings.push({name: 'nodeCosts', view: hierarchy.nodeCosts, type: 'u32', access: 'read'});
  }
  return bindings;
}

/** Builds the fused evaluate-and-decide pass for one level without a budget. @internal */
export function createTileLODLevelNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  config: TileLODShaderConfig,
  level: number
): GPUCommandNode<Parameters> {
  const levelNodeCount = config.levelOffsets[level + 1] - config.levelOffsets[level];
  return createWGSLKernelNode<Parameters>(graph, {
    id: `${config.id}-level-${level}`,
    operation: OPERATION,
    variant: 'level',
    bindings: [
      ...getHierarchyBindings(config, false),
      {name: 'nodeRecords', view: config.nodeRecords, type: 'atomic<u32>', access: 'read_write'}
    ],
    invocationCount: levelNodeCount,
    declarations: `${TILE_LOD_CONSTANTS}
${getLevelConstants(config, level)}
${getEvaluationSource(false)}
${getDecisionSource(false)}`,
    body: `let node = FIRST_NODE + index;
  if (atomicLoad(&nodeRecords[nodeRecordsOffset + node * 4u]) == 0u) { return; }
  if (!isNodeVisible(node)) { return; }
  let evaluation = evaluateNode(node);
  writeEvaluation(node, evaluation);
  applyDecision(node, evaluation.wantsRefine, evaluation.childrenResident, evaluation.firstChild, evaluation.childCount, evaluation.wantsRefine);`
  });
}

/** Builds the budgeted evaluate pass that accumulates per-bucket level sums. @internal */
export function createTileLODEvaluateNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  config: TileLODShaderConfig,
  level: number
): GPUCommandNode<Parameters> {
  const levelNodeCount = config.levelOffsets[level + 1] - config.levelOffsets[level];
  return createWGSLKernelNode<Parameters>(graph, {
    id: `${config.id}-level-${level}-evaluate`,
    operation: OPERATION,
    variant: 'evaluate',
    bindings: [
      ...getHierarchyBindings(config, true),
      {name: 'nodeRecords', view: config.nodeRecords, type: 'atomic<u32>', access: 'read_write'},
      {
        name: 'levelScratch',
        view: config.levelScratch as GraphDataView,
        type: 'atomic<u32>',
        access: 'read_write'
      }
    ],
    invocationCount: levelNodeCount,
    declarations: `${TILE_LOD_CONSTANTS}
${getLevelConstants(config, level)}
${getEvaluationSource(Boolean(config.hierarchy.nodeCosts))}`,
    body: `let node = FIRST_NODE + index;
  if (atomicLoad(&nodeRecords[nodeRecordsOffset + node * 4u]) == 0u) { return; }
  if (!isNodeVisible(node)) { return; }
  let evaluation = evaluateNode(node);
  writeEvaluation(node, evaluation);
  let scratchBase = levelScratchOffset + LEVEL_BASE;
  atomicAdd(&levelScratch[scratchBase], getNodeCost(node));
  atomicAdd(&levelScratch[scratchBase + 1u], 1u);
  if (evaluation.wantsRefine) {
    let bucketBase = scratchBase + 3u + evaluation.bucket * 4u;
    atomicAdd(&levelScratch[bucketBase], evaluation.visibleChildCost);
    atomicAdd(&levelScratch[bucketBase + 2u], evaluation.visibleChildCount);
    if (!REFINEMENT_ADD) {
      atomicAdd(&levelScratch[bucketBase + 1u], getNodeCost(node));
      atomicAdd(&levelScratch[bucketBase + 3u], 1u);
    }
  }`
  });
}

/**
 * Builds the one-invocation budget pass: accepts whole priority buckets from the highest down
 * and stops at the first bucket that does not fit. @internal
 */
export function createTileLODBudgetNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  config: TileLODShaderConfig,
  level: number,
  budget: GraphDataView<'uint32'>
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: `${config.id}-level-${level}-budget`,
    operation: OPERATION,
    variant: 'budget',
    bindings: [
      {
        name: 'levelScratch',
        view: config.levelScratch as GraphDataView,
        type: 'u32',
        access: 'read_write'
      },
      {name: 'state', view: config.state, type: 'u32', access: 'read_write'},
      {name: 'budget', view: budget, type: 'u32', access: 'read'}
    ],
    invocationCount: 1,
    declarations: `${TILE_LOD_CONSTANTS}
const LEVEL_BASE: u32 = ${level * TILE_LOD_LEVEL_SCRATCH_WORDS}u;`,
    body: `let scratchBase = levelScratchOffset + LEVEL_BASE;
  let baseCost = addSaturated(state[stateOffset + STATE_COMMITTED_COST], levelScratch[scratchBase]);
  let baseCount = addSaturated(state[stateOffset + STATE_COMMITTED_COUNT], levelScratch[scratchBase + 1u]);
  let maximumCost = budget[budgetOffset];
  let maximumCount = budget[budgetOffset + 1u];
  var accepted = vec4<u32>(0u);
  var threshold = 32u;
  for (var bucket = 31i; bucket >= 0i; bucket--) {
    let bucketBase = scratchBase + 3u + u32(bucket) * 4u;
    let entry = vec4<u32>(
      levelScratch[bucketBase], levelScratch[bucketBase + 1u],
      levelScratch[bucketBase + 2u], levelScratch[bucketBase + 3u]
    );
    // Empty buckets hold no refinement candidates and never stop the walk.
    if (all(entry == vec4<u32>(0u))) { continue; }
    let next = vec4<u32>(
      addSaturated(accepted.x, entry.x), addSaturated(accepted.y, entry.y),
      addSaturated(accepted.z, entry.z), addSaturated(accepted.w, entry.w)
    );
    let fitsCost = addSaturated(baseCost, next.x) <= addSaturated(maximumCost, next.y);
    let fitsCount = addSaturated(baseCount, next.z) <= addSaturated(maximumCount, next.w);
    if (fitsCost && fitsCount) {
      accepted = next;
      threshold = u32(bucket);
    } else {
      state[stateOffset + STATE_BUDGET_EXHAUSTED] = 1u;
      break;
    }
  }
  levelScratch[scratchBase + 2u] = threshold;`
  });
}

/** Builds the budgeted decide pass. @internal */
export function createTileLODDecideNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  config: TileLODShaderConfig,
  level: number
): GPUCommandNode<Parameters> {
  const levelNodeCount = config.levelOffsets[level + 1] - config.levelOffsets[level];
  const bindings: WGSLKernelBinding[] = [
    {name: 'children', view: config.hierarchy.children, type: 'u32', access: 'read'}
  ];
  if (config.hierarchy.nodeCosts) {
    bindings.push({
      name: 'nodeCosts',
      view: config.hierarchy.nodeCosts,
      type: 'u32',
      access: 'read'
    });
  }
  bindings.push(
    {name: 'levelScratch', view: config.levelScratch as GraphDataView, type: 'u32', access: 'read'},
    {name: 'nodeRecords', view: config.nodeRecords, type: 'atomic<u32>', access: 'read_write'},
    {name: 'state', view: config.state, type: 'atomic<u32>', access: 'read_write'}
  );
  return createWGSLKernelNode<Parameters>(graph, {
    id: `${config.id}-level-${level}-decide`,
    operation: OPERATION,
    variant: 'decide',
    bindings,
    invocationCount: levelNodeCount,
    declarations: `${TILE_LOD_CONSTANTS}
${getLevelConstants(config, level)}
fn getNodeCost(node: u32) -> u32 { return ${config.hierarchy.nodeCosts ? 'nodeCosts[nodeCostsOffset + node]' : '0u'}; }
${getDecisionSource(true)}`,
    body: `let node = FIRST_NODE + index;
  let recordBase = nodeRecordsOffset + node * 4u;
  if (atomicLoad(&nodeRecords[recordBase]) == 0u) { return; }
  let flags = atomicLoad(&nodeRecords[recordBase + 1u]);
  if ((flags & FLAG_VISIBLE) == 0u) { return; }
  let wantsRefine = (flags & FLAG_WANTS_REFINE) != 0u;
  let bucket = (flags >> BUCKET_SHIFT) & 0xffu;
  let threshold = levelScratch[levelScratchOffset + LEVEL_BASE + 2u];
  let refine = wantsRefine && bucket >= threshold;
  var firstChild = 0u;
  var childCount = 0u;
  if (wantsRefine) {
    firstChild = children[childrenOffset + node * 2u];
    childCount = children[childrenOffset + node * 2u + 1u];
  }
  applyDecision(node, wantsRefine, (flags & FLAG_CHILDREN_RESIDENT) != 0u, firstChild, childCount, refine);`
  });
}

/** Builds the emit pass that writes node-aligned masks and statistics. @internal */
export function createTileLODEmitNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  config: TileLODShaderConfig,
  outputs: {
    drawMask: GraphDataView<'uint32'>;
    desiredMask?: GraphDataView<'uint32'>;
    requestMask?: GraphDataView<'uint32'>;
    priorityBits?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'nodeRecords', view: config.nodeRecords, type: 'u32', access: 'read'}
  ];
  if (config.hierarchy.nodeCosts) {
    bindings.push({
      name: 'nodeCosts',
      view: config.hierarchy.nodeCosts,
      type: 'u32',
      access: 'read'
    });
  }
  bindings.push(
    {name: 'state', view: config.state, type: 'atomic<u32>', access: 'read_write'},
    {name: 'drawMask', view: outputs.drawMask, type: 'u32', access: 'read_write'}
  );
  if (outputs.desiredMask) {
    bindings.push({
      name: 'desiredMask',
      view: outputs.desiredMask,
      type: 'u32',
      access: 'read_write'
    });
  }
  if (outputs.requestMask) {
    bindings.push({
      name: 'requestMask',
      view: outputs.requestMask,
      type: 'u32',
      access: 'read_write'
    });
  }
  if (outputs.priorityBits) {
    bindings.push({
      name: 'priorityBits',
      view: outputs.priorityBits,
      type: 'u32',
      access: 'read_write'
    });
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: `${config.id}-emit`,
    operation: OPERATION,
    variant: 'emit',
    bindings,
    invocationCount: config.nodeCount,
    declarations: `${TILE_LOD_CONSTANTS}
const REFINEMENT_ADD: bool = ${config.refinementAdd};`,
    body: `let recordBase = nodeRecordsOffset + index * 4u;
  let nodeState = nodeRecords[recordBase];
  let flags = nodeRecords[recordBase + 1u];
  let visible = nodeState > 0u && (flags & FLAG_VISIBLE) != 0u;
  let resident = (flags & FLAG_RESIDENT) != 0u;
  let refined = (flags & FLAG_REFINED) != 0u;
  let childrenResident = (flags & FLAG_CHILDREN_RESIDENT) != 0u;
  let desired = visible && (REFINEMENT_ADD || !refined);
  let drawn = visible && resident && nodeState == 2u && (REFINEMENT_ADD || !refined || !childrenResident);
  let requested = visible && !resident;
  let cost = ${config.hierarchy.nodeCosts ? 'nodeCosts[nodeCostsOffset + index]' : '0u'};
  drawMask[drawMaskOffset + index] = select(0u, 1u, drawn);
  ${outputs.desiredMask ? 'desiredMask[desiredMaskOffset + index] = select(0u, 1u, desired);' : ''}
  ${outputs.requestMask ? 'requestMask[requestMaskOffset + index] = select(0u, 1u, requested);' : ''}
  ${outputs.priorityBits ? 'priorityBits[priorityBitsOffset + index] = select(0u, nodeRecords[recordBase + 2u], visible);' : ''}
  if (visible) { atomicAdd(&state[stateOffset + STATE_VISIBLE_COUNT], 1u); }
  if (desired) {
    atomicAdd(&state[stateOffset + STATE_DESIRED_COUNT], 1u);
    atomicAdd(&state[stateOffset + STATE_DESIRED_COST], cost);
  }
  if (drawn) {
    atomicAdd(&state[stateOffset + STATE_DRAWN_COUNT], 1u);
    atomicAdd(&state[stateOffset + STATE_DRAWN_COST], cost);
  }
  if (requested) { atomicAdd(&state[stateOffset + STATE_REQUESTED_COUNT], 1u); }`
  });
}

/** Builds the one-invocation pass that writes indirect records and statistics. @internal */
export function createTileLODIndirectNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    drawTotal: GraphDataView<'uint32'>;
    state: GraphDataView<'uint32'>;
    outputCapacity: number;
    drawWords?: GraphDataView<'uint32'>;
    drawWordIndex?: number;
    dispatchCommand?: GraphDataView<'uint32'>;
    workgroupSize?: number;
    statistics?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'drawTotal', view: props.drawTotal, type: 'u32', access: 'read'},
    {name: 'state', view: props.state, type: 'u32', access: 'read'}
  ];
  if (props.drawWords) {
    bindings.push({name: 'drawWords', view: props.drawWords, type: 'u32', access: 'read_write'});
  }
  if (props.dispatchCommand) {
    bindings.push({
      name: 'dispatchCommand',
      view: props.dispatchCommand,
      type: 'u32',
      access: 'read_write'
    });
  }
  if (props.statistics) {
    bindings.push({name: 'statistics', view: props.statistics, type: 'u32', access: 'read_write'});
  }
  const workgroupSize = props.workgroupSize ?? 1;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'indirect',
    bindings,
    invocationCount: 1,
    body: `let retained = min(drawTotal[drawTotalOffset], ${props.outputCapacity}u);
  ${props.drawWords ? `drawWords[drawWordsOffset + ${props.drawWordIndex ?? 1}u] = retained;` : ''}
  ${
    props.dispatchCommand
      ? `dispatchCommand[dispatchCommandOffset] = retained / ${workgroupSize}u + select(0u, 1u, retained % ${workgroupSize}u != 0u);
  dispatchCommand[dispatchCommandOffset + 1u] = 1u;
  dispatchCommand[dispatchCommandOffset + 2u] = 1u;`
      : ''
  }
  ${
    props.statistics
      ? `for (var row = 0u; row < 7u; row++) {
    statistics[statisticsOffset + row] = state[stateOffset + row];
  }
  statistics[statisticsOffset + 7u] = 0u;`
      : ''
  }`
  });
}
