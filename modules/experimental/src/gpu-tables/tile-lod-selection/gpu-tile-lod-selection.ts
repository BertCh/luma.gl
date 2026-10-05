// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  doGraphDataViewsOverlap,
  GPUAncestorProjection,
  GPUCompaction,
  GPUVisibilityWorkflow,
  makeGPUVirtualGeometrySelectionPlan,
  validatePackedUint32View,
  validatePackedView,
  type DrawCommandBufferView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUVirtualGeometrySelectionPlan,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createPublishNode} from '../../utils/wgsl-kernel-nodes';
import type {GPUCompactOutput} from '../../utils/gpu-contributor-types';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphViewsBelongToGraph,
  validateCompactOutput
} from '../../utils/gpu-contributor-utils';
import {
  createTileLODBudgetNode,
  createTileLODDecideNode,
  createTileLODEmitNode,
  createTileLODEvaluateNode,
  createTileLODIndirectNode,
  createTileLODLevelNode,
  createTileLODPrepareNode,
  TILE_LOD_LEVEL_SCRATCH_WORDS,
  TILE_LOD_RECORD_WORDS,
  TILE_LOD_STATE_LENGTH,
  type TileLODShaderConfig
} from './tile-lod-selection-shaders';
import {GPU_TILE_LOD_VIEW_LENGTH} from './tile-lod-view';

/** uint32 rows in the optional statistics output. */
export const GPU_TILE_LOD_STATISTICS_LENGTH = 8;
/** uint32 rows in the optional budget input: `[maximumCost, maximumNodeCount]`. */
export const GPU_TILE_LOD_BUDGET_LENGTH = 2;
/** Budget value meaning "no limit". */
export const GPU_TILE_LOD_UNLIMITED = 0xffffffff;
/** Sentinel written to `drawnAncestors` for nodes with no drawn ancestor. */
export const GPU_TILE_LOD_INVALID_NODE = 0xffffffff;
/** Number of fixed priority buckets used by budgeted refinement. */
export const GPU_TILE_LOD_PRIORITY_BUCKET_COUNT = 32;

/** GPU-resident, breadth-level-ordered tile hierarchy consumed by {@link GPUTileLODSelection}. */
export type GPUTileLODHierarchy = {
  /** Bounding sphere per node: center XYZ and radius, in the same space as the view. */
  sphereBounds: GraphDataView<'float32x4'>;
  /** World-space geometric error per node. */
  geometricErrors: GraphDataView<'float32'>;
  /** `[firstChild, childCount]` per node. Children must lie in the immediately next level. */
  children: GraphDataView<'uint32x2'>;
  /** CPU-known level offsets: start at 0, strictly increasing, end at the node count. Topology. */
  levelOffsets: readonly number[];
  /** Optional stable tile IDs; zero-based node indices are used when omitted. */
  tileIds?: GraphDataView<'uint32'>;
  /** Optional parent node per node (roots: any value `>=` node count). Required by `drawnAncestors`. */
  parents?: GraphDataView<'uint32'>;
  /** Optional per-node cost in the budget unit (bytes, KiB, rows). Missing means cost 0. */
  nodeCosts?: GraphDataView<'uint32'>;
};

/** Desired tiles that are not resident, in breadth (coarse-first) order. */
export type GPUTileLODRequestOutput = GPUCompactOutput & {
  /** Optional weighted screen-space error per request, aligned with `ids`. */
  priorities?: GraphDataView<'float32'>;
};

/** Indirect draw record whose instance count receives the drawn count. */
export type GPUTileLODIndirectDraw = {
  /** View from `DrawCommandBuffer.importToGraph(graph)`. */
  commands: DrawCommandBufferView;
  /** Record index. Defaults to 0. Topology. */
  commandIndex?: number;
};

/** Indirect dispatch record receiving `[ceil(drawnCount / workgroupSize), 1, 1]`. */
export type GPUTileLODIndirectDispatch = {
  /** Packed uint32 view with at least 3 rows. */
  command: GraphDataView<'uint32'>;
  /** Invocations per workgroup of the consumer shader. Topology. */
  workgroupSize: number;
};

/**
 * Properties for {@link GPUTileLODSelection}.
 *
 * Topology: node count, `levelOffsets`, `refinement`, which optional views exist, every output
 * capacity, `indirectDraw.commandIndex`, and `indirectDispatch.workgroupSize`. Per-frame: `view`,
 * `budget`, `residency`, `enabledNodes`, and the hierarchy contents.
 */
export type GPUTileLODSelectionProps = {
  /** Prefix for node and transient IDs. Defaults to `'tile-lod-selection'`. */
  id?: string;
  /** Tile hierarchy. */
  hierarchy: GPUTileLODHierarchy;
  /** Per-frame packed view of `GPU_TILE_LOD_VIEW_LENGTH` floats, see `getGPUTileLODViewParameterValues`. */
  view: GraphDataView<'float32'>;
  /** `'replace'` (default) draws children instead of parents; `'add'` draws both. */
  refinement?: 'replace' | 'add';
  /** Per-node nonzero when content is resident. Omitted means all resident. */
  residency?: GraphDataView<'uint32'>;
  /** Per-node zero prunes the node and its subtree. */
  enabledNodes?: GraphDataView<'uint32'>;
  /** Per-frame `[maximumCost, maximumNodeCount]`; `GPU_TILE_LOD_UNLIMITED` disables a cap. */
  budget?: GraphDataView<'uint32'>;
  /** Drawn tile IDs. */
  output: GPUCompactOutput;
  /** Optional caller-visible node-aligned drawn mask (0/1). */
  drawMask?: GraphDataView<'uint32'>;
  /** Optional node-aligned desired-frontier mask (0/1). */
  desiredMask?: GraphDataView<'uint32'>;
  /** Optional requested tile IDs with priorities. */
  requests?: GPUTileLODRequestOutput;
  /** Optional node-aligned nearest drawn ancestor-or-self node index. */
  drawnAncestors?: GraphDataView<'uint32'>;
  /** Optional statistics: desired count and cost, drawn count and cost, requested count, budget exhausted, visible count, reserved. */
  statistics?: GraphDataView<'uint32'>;
  /** Optional indirect draw record. */
  indirectDraw?: GPUTileLODIndirectDraw;
  /** Optional indirect dispatch record. */
  indirectDispatch?: GPUTileLODIndirectDispatch;
};

/**
 * Selects the tiles to draw and to load from a GPU-resident tile hierarchy, entirely on the GPU.
 *
 * Each level pass refines a node while its weighted screen-space error
 * `geometricError * pixelProjectionScale / distanceToSphere`, optionally relaxed by foveation and
 * focus-distance falloff, exceeds `maximumScreenSpaceError`. With a budget, refinement is accepted
 * a whole priority bucket at a time from the highest error down and stops at the first bucket that
 * does not fit, which keeps the result deterministic. A resident refined node stands in for its
 * subtree until all visible children are resident; every visible non-resident node is requested.
 * Without foveation, budget, and residency the drawn frontier equals `GPUVirtualGeometrySelection`.
 */
export class GPUTileLODSelection implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTileLODSelectionProps;
  /** Frozen level topology. */
  readonly plan: Readonly<GPUVirtualGeometrySelectionPlan>;
  /** Resolved refinement mode. */
  readonly refinement: 'replace' | 'add';

  constructor(props: GPUTileLODSelectionProps) {
    this.id = props.id ?? 'tile-lod-selection';
    this.props = props;
    this.refinement = props.refinement ?? 'replace';
    const {id} = this;
    const {hierarchy} = props;
    this.plan = makeGPUVirtualGeometrySelectionPlan(
      hierarchy.levelOffsets,
      hierarchy.sphereBounds.length
    );
    const nodeCount = this.plan.nodeCount;
    validatePackedView(hierarchy.sphereBounds, ['float32x4'], `${id} sphereBounds`);
    validatePackedView(hierarchy.geometricErrors, ['float32'], `${id} geometricErrors`);
    validatePackedView(hierarchy.children, ['uint32x2'], `${id} children`);
    validatePackedView(props.view, ['float32'], `${id} view`);
    const nodeAligned = [
      ['geometricErrors', hierarchy.geometricErrors],
      ['children', hierarchy.children],
      ['tileIds', hierarchy.tileIds],
      ['parents', hierarchy.parents],
      ['nodeCosts', hierarchy.nodeCosts],
      ['residency', props.residency],
      ['enabledNodes', props.enabledNodes],
      ['drawMask', props.drawMask],
      ['desiredMask', props.desiredMask],
      ['drawnAncestors', props.drawnAncestors]
    ] as const;
    for (const [name, view] of nodeAligned) {
      if (!view) {
        continue;
      }
      if (name !== 'geometricErrors' && name !== 'children') {
        validatePackedUint32View(view, `${id} ${name}`);
      }
      if (view.length !== nodeCount) {
        throw new Error(`${id} ${name} length must match sphereBounds`);
      }
    }
    if (props.view.length !== GPU_TILE_LOD_VIEW_LENGTH) {
      throw new Error(`${id} view must contain ${GPU_TILE_LOD_VIEW_LENGTH} float32 values`);
    }
    if (props.budget) {
      validatePackedUint32View(props.budget, `${id} budget`);
      if (props.budget.length !== GPU_TILE_LOD_BUDGET_LENGTH) {
        throw new Error(`${id} budget must contain ${GPU_TILE_LOD_BUDGET_LENGTH} uint32 values`);
      }
    }
    if (props.statistics) {
      validatePackedUint32View(props.statistics, `${id} statistics`);
      if (props.statistics.length < GPU_TILE_LOD_STATISTICS_LENGTH) {
        throw new Error(`${id} statistics must contain ${GPU_TILE_LOD_STATISTICS_LENGTH} rows`);
      }
    }
    validateCompactOutput(id, props.output);
    if (props.requests) {
      validateCompactOutput(`${id} requests`, props.requests);
      if (props.requests.priorities) {
        validatePackedView(props.requests.priorities, ['float32'], `${id} requests.priorities`);
        if (props.requests.priorities.length !== props.requests.ids.length) {
          throw new Error(`${id} requests.priorities length must equal requests.ids length`);
        }
      }
    }
    if (props.drawnAncestors && !hierarchy.parents) {
      throw new Error(`${id} drawnAncestors requires hierarchy.parents`);
    }
    if (!['replace', 'add'].includes(this.refinement)) {
      throw new Error(`${id} refinement must be replace or add`);
    }
    if (props.indirectDraw) {
      const commandIndex = props.indirectDraw.commandIndex ?? 0;
      if (
        !Number.isSafeInteger(commandIndex) ||
        commandIndex < 0 ||
        commandIndex >= props.indirectDraw.commands.capacity
      ) {
        throw new Error(`${id} indirectDraw.commandIndex is out of range`);
      }
    }
    if (props.indirectDispatch) {
      validatePackedUint32View(props.indirectDispatch.command, `${id} indirectDispatch.command`);
      const {workgroupSize} = props.indirectDispatch;
      if (props.indirectDispatch.command.length < 3) {
        throw new Error(`${id} indirectDispatch.command must contain 3 rows`);
      }
      if (!Number.isSafeInteger(workgroupSize) || workgroupSize < 1 || workgroupSize > 0xffffffff) {
        throw new Error(`${id} indirectDispatch.workgroupSize must be a positive integer`);
      }
    }
    const outputs = getOutputViews(props);
    const inputs = getInputViews(props);
    for (const [outputIndex, output] of outputs.entries()) {
      if (
        inputs.some(input => doGraphDataViewsOverlap(input, output)) ||
        outputs.some(
          (other, otherIndex) =>
            otherIndex !== outputIndex && doGraphDataViewsOverlap(other, output)
        )
      ) {
        throw new Error(`${id} output views must not overlap each other or inputs`);
      }
    }
  }

  /** Returns prepare, per-level, emit, compaction, publish, and optional output nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, plan} = this;
    const {hierarchy, output, requests} = props;
    validateGraphViewsBelongToGraph(id, graph, [...getInputViews(props), ...getOutputViews(props)]);
    const nodeCount = plan.nodeCount;
    const budgeted = Boolean(props.budget);
    const config: TileLODShaderConfig = {
      id,
      nodeCount,
      rootCount: plan.rootCount,
      levelOffsets: plan.levelOffsets,
      refinementAdd: this.refinement === 'add',
      hierarchy,
      view: props.view,
      nodeRecords: createTransientView(
        graph,
        `${id}-node-records`,
        'uint32',
        nodeCount * TILE_LOD_RECORD_WORDS
      ),
      state: createTransientView(graph, `${id}-state`, 'uint32', TILE_LOD_STATE_LENGTH),
      levelScratch: budgeted
        ? createTransientView(
            graph,
            `${id}-level-scratch`,
            'uint32',
            plan.levelCount * TILE_LOD_LEVEL_SCRATCH_WORDS
          )
        : undefined
    };
    const nodes: GPUCommandNode<Parameters>[] = [
      createTileLODPrepareNode<Parameters>(graph, config, props.residency, props.enabledNodes)
    ];
    for (let level = 0; level < plan.levelCount; level++) {
      if (props.budget) {
        nodes.push(
          createTileLODEvaluateNode<Parameters>(graph, config, level),
          createTileLODBudgetNode<Parameters>(graph, config, level, props.budget),
          createTileLODDecideNode<Parameters>(graph, config, level)
        );
      } else {
        nodes.push(createTileLODLevelNode<Parameters>(graph, config, level));
      }
    }

    const drawMask =
      props.drawMask ?? createTransientView(graph, `${id}-draw-mask`, 'uint32', nodeCount);
    const requestMask = requests
      ? createTransientView(graph, `${id}-request-mask`, 'uint32', nodeCount)
      : undefined;
    const priorityBits = requests?.priorities
      ? createTransientView(graph, `${id}-priority-bits`, 'uint32', nodeCount)
      : undefined;
    nodes.push(
      createTileLODEmitNode<Parameters>(graph, config, {
        drawMask,
        desiredMask: props.desiredMask,
        requestMask,
        priorityBits
      })
    );

    const drawIds = createTransientView(graph, `${id}-draw-ids`, 'uint32', nodeCount);
    const drawTotal = createTransientView(graph, `${id}-draw-total`, 'uint32', 1);
    nodes.push(
      ...new GPUVisibilityWorkflow({
        id: `${id}-draw-visibility`,
        predicates: [{kind: ['bounds', 'lod'], mask: drawMask}],
        outputMask: drawMask,
        sourceIds: hierarchy.tileIds,
        output: drawIds,
        count: drawTotal
      }).getCommandNodes(graph),
      createPublishNode<Parameters>(graph, {
        id: `${id}-draw-publish`,
        operation: 'GPUTileLODSelection',
        totalCount: drawTotal,
        compactIds: drawIds,
        output
      })
    );
    if (props.indirectDraw || props.indirectDispatch || props.statistics) {
      const commands = props.indirectDraw?.commands;
      nodes.push(
        createTileLODIndirectNode<Parameters>(graph, {
          id: `${id}-indirect`,
          drawTotal,
          state: config.state,
          outputCapacity: output.ids.length,
          drawWords: commands?.words,
          drawWordIndex: commands
            ? (props.indirectDraw?.commandIndex ?? 0) * (commands.recordByteLength / 4) + 1
            : undefined,
          dispatchCommand: props.indirectDispatch?.command,
          workgroupSize: props.indirectDispatch?.workgroupSize,
          statistics: props.statistics
        })
      );
    }

    if (requests && requestMask) {
      const requestIds = createTransientView(graph, `${id}-request-ids`, 'uint32', nodeCount);
      const requestTotal = createTransientView(graph, `${id}-request-total`, 'uint32', 1);
      nodes.push(
        ...new GPUVisibilityWorkflow({
          id: `${id}-request-visibility`,
          predicates: [{kind: 'selection', mask: requestMask}],
          outputMask: requestMask,
          sourceIds: hierarchy.tileIds,
          output: requestIds,
          count: requestTotal
        }).getCommandNodes(graph)
      );
      let requestPriorities: GraphDataView<'uint32'> | undefined;
      if (priorityBits && requests.priorities) {
        requestPriorities = createTransientView(
          graph,
          `${id}-request-priorities`,
          'uint32',
          nodeCount
        );
        nodes.push(
          ...new GPUCompaction({
            id: `${id}-request-priorities`,
            input: priorityBits,
            flags: requestMask,
            output: requestPriorities,
            count: createTransientView(graph, `${id}-request-priority-count`, 'uint32', 1)
          }).getCommandNodes(graph)
        );
      }
      nodes.push(
        createPublishNode<Parameters>(graph, {
          id: `${id}-request-publish`,
          operation: 'GPUTileLODSelection',
          totalCount: requestTotal,
          compactIds: requestIds,
          output: requests,
          extraColumn:
            requestPriorities && requests.priorities
              ? {source: requestPriorities, destination: requests.priorities}
              : undefined
        })
      );
    }

    if (props.drawnAncestors && hierarchy.parents) {
      nodes.push(
        ...new GPUAncestorProjection({
          id: `${id}-drawn-ancestors`,
          parents: hierarchy.parents,
          visibility: drawMask,
          output: props.drawnAncestors,
          maxDepth: plan.levelCount,
          invalidValue: GPU_TILE_LOD_INVALID_NODE
        }).getCommandNodes(graph)
      );
    }
    return nodes;
  }
}

/** Returns every read-only view of the contributor. */
function getInputViews(props: GPUTileLODSelectionProps): GraphDataView[] {
  const {hierarchy} = props;
  return [
    hierarchy.sphereBounds,
    hierarchy.geometricErrors,
    hierarchy.children,
    hierarchy.tileIds,
    hierarchy.parents,
    hierarchy.nodeCosts,
    props.view,
    props.residency,
    props.enabledNodes,
    props.budget
  ].filter(view => view !== undefined);
}

/** Returns every writable view of the contributor. */
function getOutputViews(props: GPUTileLODSelectionProps): GraphDataView[] {
  return [
    props.output.ids,
    props.output.count,
    props.output.overflow,
    props.output.totalCount,
    props.requests?.ids,
    props.requests?.count,
    props.requests?.overflow,
    props.requests?.totalCount,
    props.requests?.priorities,
    props.drawMask,
    props.desiredMask,
    props.drawnAncestors,
    props.statistics,
    props.indirectDraw?.commands.words,
    props.indirectDispatch?.command
  ].filter(view => view !== undefined);
}
