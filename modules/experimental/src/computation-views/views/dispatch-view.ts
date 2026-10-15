// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {renderRasterImage} from '../encoding/raster-image';
import {CARD_COLORS} from '../encoding/card-tokens';
import {resolveValueScale} from './value-strip';
import {assert} from '@luma.gl/core';
import type {GPUComputationNode} from '@luma.gl/gpgpu';
import type {LensRecord, ViewContext, ViewRegistration} from '../registry/view-registration';
import type {MarkProvenance} from '../types';
import {h, viewNodeToDom, type ViewElementNode} from '../view-node';
import {provenanceMarkAttributes, provenancePatternDefs} from '../encoding/provenance';
import {getSelectionInk} from '../encoding/selection-ink';
import {
  getAllCaptures,
  getBoundNode,
  getMostConcreteGraph,
  getPinnedEntity,
  getDatumKeySpace,
  getAdapterLabel,
  getCaptureValue,
  getCaptureProvenance
} from './record-queries';
import {buildKernelSourceModel, renderKernelSource, type KernelSourceModel} from './kernel-source';

/** One dispatch with at most one expanded workgroup. */
export type DispatchViewModel = {
  nodeId: string;
  label: string;
  namespace: string;
  workgroups: [number, number, number];
  workgroupSize: [number, number, number];
  lanesPerWorkgroup: number;
  workgroupCount: number;
  totalInvocations: number;
  invocationCount?: number;
  idleLaneCount: number;
  indirect: boolean;
  expandedWorkgroup: number;
  expandedReason: 'bound' | 'pinned-entity' | 'default';
  lanes: {index: number; globalIndex: number; active: boolean; pinned: boolean}[];
  pinnedLane?: {globalIndex: number; workgroup: number; lane: number} | {unknown: string};
  idProbe:
    | {captureId: string; values: number[]; provenance: MarkProvenance}
    | {status: 'not-recorded'};
  workgroupGrid: {drawn: number; aggregated: boolean};
  source: KernelSourceModel;
};

function getDispatchNode(context: ViewContext): GPUComputationNode | undefined {
  const bound = getBoundNode(context)?.node;
  return bound?.dispatch && bound.workgroupSize
    ? bound
    : getMostConcreteGraph(context.record)?.nodes.find(
        node => node.kind === 'compute' && node.dispatch && node.workgroupSize
      );
}

function mapPinnedLane(
  context: ViewContext,
  node: GPUComputationNode,
  total: number
): DispatchViewModel['pinnedLane'] {
  const entity = getPinnedEntity(context);
  if (entity === undefined) return undefined;
  let globalIndex = entity;
  if (!node.datum || node.datum.keySpace !== getDatumKeySpace(context))
    return {unknown: 'datum key space does not match the dispatch'};
  if (node.datum.relation) {
    const relation = context.record.axes.relations.find(
      candidate => candidate.id === node.datum?.relation
    );
    const reference = relation?.ref;
    if (
      relation?.kind === 'function' &&
      reference &&
      'function' in reference &&
      reference.function === 'identity'
    ) {
      globalIndex = entity;
    } else if (
      relation?.kind === 'function' &&
      reference &&
      'function' in reference &&
      reference.function === 'divide' &&
      Number.isFinite(reference.params?.['divisor']) &&
      reference.params!['divisor'] > 0
    ) {
      globalIndex = Math.floor(entity / reference.params!['divisor']);
    } else {
      return {
        unknown: `relation ${node.datum.relation} (${relation?.kind ?? 'missing'}) not evaluable`
      };
    }
  }
  if (globalIndex >= total) return {unknown: 'mapped lane is outside the recorded dispatch shape'};
  const workgroups = node.dispatch!.workgroups;
  const size = node.workgroupSize!;
  const globalCoordinates = unflattenIndex(globalIndex, [
    workgroups[0] * size[0],
    workgroups[1] * size[1],
    workgroups[2] * size[2]
  ]);
  const workgroupCoordinates = globalCoordinates.map((coordinate, axis) =>
    Math.floor(coordinate / size[axis])
  );
  const laneCoordinates = globalCoordinates.map((coordinate, axis) => coordinate % size[axis]);
  return {
    globalIndex,
    workgroup: flattenIndex(workgroupCoordinates, workgroups),
    lane: flattenIndex(laneCoordinates, size)
  };
}

function unflattenIndex(index: number, shape: readonly number[]): number[] {
  return [
    index % shape[0],
    Math.floor(index / shape[0]) % shape[1],
    Math.floor(index / (shape[0] * shape[1]))
  ];
}

function flattenIndex(coordinates: readonly number[], shape: readonly number[]): number {
  return coordinates[0] + shape[0] * (coordinates[1] + shape[1] * coordinates[2]);
}

/** Builds dispatch geometry and separates captured probe values from computed lane ids. */
export function buildDispatchViewModel(context: ViewContext): DispatchViewModel {
  const node = getDispatchNode(context);
  // The registration only builds a node with recorded geometry.
  assert(node?.dispatch && node.workgroupSize);
  const workgroups: [number, number, number] = [...node.dispatch.workgroups];
  const workgroupSize: [number, number, number] = [...node.workgroupSize];
  const lanesPerWorkgroup = workgroupSize.reduce((product, value) => product * value, 1);
  const workgroupCount = workgroups.reduce((product, value) => product * value, 1);
  const totalInvocations = workgroupCount * lanesPerWorkgroup;
  const pinnedLane = mapPinnedLane(context, node, totalInvocations);
  const bound = context.bindings.workgroup;
  const validBound =
    typeof bound === 'number' && Number.isInteger(bound) && bound >= 0 && bound < workgroupCount;
  const expandedWorkgroup = validBound
    ? bound
    : pinnedLane && 'workgroup' in pinnedLane
      ? pinnedLane.workgroup
      : 0;
  const expandedReason = validBound
    ? 'bound'
    : pinnedLane && 'workgroup' in pinnedLane
      ? 'pinned-entity'
      : 'default';
  const lanes = Array.from({length: workgroupCount ? lanesPerWorkgroup : 0}, (_, index) => {
    const workgroupCoordinates = unflattenIndex(expandedWorkgroup, workgroups);
    const laneCoordinates = unflattenIndex(index, workgroupSize);
    const globalCoordinates = workgroupCoordinates.map(
      (coordinate, axis) => coordinate * workgroupSize[axis] + laneCoordinates[axis]
    );
    const globalIndex = flattenIndex(
      globalCoordinates,
      workgroups.map((count, axis) => count * workgroupSize[axis])
    );
    return {
      index,
      globalIndex,
      active:
        node.dispatch!.invocationCount === undefined ||
        globalIndex < node.dispatch!.invocationCount,
      pinned: Boolean(
        pinnedLane && 'globalIndex' in pinnedLane && pinnedLane.globalIndex === globalIndex
      )
    };
  });
  const graph = context.record.graphs.find(candidate => candidate.nodes.includes(node));
  const probes = getAllCaptures(context.record).filter(
    capture =>
      (capture.graph === undefined || capture.graph === graph?.id) &&
      (capture.name === 'id-probe' || capture.meaning?.includes('global_invocation_id'))
  );
  const probe =
    probes.find(capture => capture.afterNode === node.id) ??
    probes.find(capture => capture.afterNode === undefined);
  return {
    nodeId: node.id,
    label: node.label,
    namespace: node.namespace,
    workgroups,
    workgroupSize,
    lanesPerWorkgroup,
    workgroupCount,
    totalInvocations,
    invocationCount: node.dispatch.invocationCount,
    idleLaneCount:
      node.dispatch.invocationCount === undefined
        ? 0
        : Math.max(0, totalInvocations - node.dispatch.invocationCount),
    indirect: node.dispatch.indirect ?? false,
    expandedWorkgroup,
    expandedReason,
    lanes,
    pinnedLane,
    idProbe: probe
      ? {
          captureId: probe.id,
          values: lanes.map(lane => getCaptureValue(probe, lane.globalIndex) ?? NaN),
          provenance: getCaptureProvenance(probe)
        }
      : {status: 'not-recorded'},
    workgroupGrid: {
      drawn: workgroupCount > 1024 ? 1 : workgroupCount,
      aggregated: workgroupCount > 1024
    },
    source: buildKernelSourceModel(context.record, node)
  };
}

/** Draws the dispatch, the expanded lane grid, and retained kernel source. */
export function renderDispatchView(
  model: DispatchViewModel,
  context: ViewContext
): ViewElementNode {
  const ink = getSelectionInk(context.ground);
  const metadata = provenanceMarkAttributes('metadata', 'currentColor', context.idPrefix);
  const probe = 'captureId' in model.idProbe ? model.idProbe : undefined;
  const capture =
    probe && getAllCaptures(context.record).find(candidate => candidate.id === probe.captureId);
  const scale = capture ? resolveValueScale(context, capture.name)?.scale : undefined;
  const compact = model.lanes.length > 32;
  const labelledLanes = compact
    ? model.lanes.filter(lane => lane.pinned || lane.index < 15)
    : model.lanes;
  return h(
    'section',
    {class: 'lens-view lens-dispatch-view'},
    h(
      'h3',
      null,
      `${model.label} · ${model.workgroups.join('×')} workgroups × ${model.workgroupSize.join('×')} lanes = ${model.totalInvocations} invocations`
    ),
    model.indirect && h('p', null, 'Indirect dispatch: recorded shape is an upper bound.'),
    model.invocationCount === undefined &&
      h('p', {class: 'lens-unknown'}, 'invocation guard unknown'),
    model.pinnedLane &&
      'unknown' in model.pinnedLane &&
      h('p', {class: 'lens-unknown'}, `unknown — ${model.pinnedLane.unknown}`),
    h(
      'svg',
      {
        class: 'lens-workgroups',
        viewBox: `0 0 640 ${model.workgroupGrid.aggregated ? 40 : Math.max(24, Math.ceil(model.workgroupGrid.drawn / 32) * 20)}`,
        role: 'img',
        'aria-label': 'Workgroups'
      },
      provenancePatternDefs(context.idPrefix, context.ground),
      model.workgroupGrid.aggregated
        ? [
            h('rect', {
              x: 1,
              y: 1,
              width: 638,
              height: 36,
              ...metadata,
              'data-aggregate': 'workgroups'
            }),
            h('text', {x: 8, y: 24, fill: 'currentColor'}, `${model.workgroupCount} workgroups`)
          ]
        : Array.from({length: model.workgroupGrid.drawn}, (_, index) =>
            h('rect', {
              x: (index % 32) * 20 + 1,
              y: Math.floor(index / 32) * 20 + 1,
              width: 17,
              height: 17,
              ...metadata,
              stroke: index === model.expandedWorkgroup ? ink.core : 'currentColor',
              'stroke-width': index === model.expandedWorkgroup ? 3 : 1,
              'data-state': index === model.expandedWorkgroup ? 'selection' : undefined,
              'data-action': 'bind',
              'data-axis': 'workgroup',
              'data-key': index
            })
          )
    ),
    h('h4', null, `Workgroup ${model.expandedWorkgroup} (${model.expandedReason})`),
    h(
      'svg',
      {
        class: 'lens-lanes',
        viewBox: `0 0 1280 ${Math.max(32, Math.ceil(model.lanes.length / 32) * 32)}`,
        role: 'img',
        'aria-label': 'Lanes in expanded workgroup'
      },
      provenancePatternDefs(`${context.idPrefix}-lanes`, context.ground),
      compact &&
        renderRasterImage({
          columns: 32,
          rows: Math.ceil(model.lanes.length / 32),
          limit: 160,
          valueAt: index => (!model.lanes[index]?.active ? NaN : probe ? probe.values[index] : 1),
          categorical: !probe,
          colorFor: value =>
            !Number.isFinite(value)
              ? CARD_COLORS[context.ground]['no-data']
              : (scale?.colorFor(value) ?? CARD_COLORS[context.ground]['input']),
          attributes: {
            width: 1280,
            height: Math.ceil(model.lanes.length / 32) * 32,
            'data-provenance': probe?.provenance ?? 'metadata'
          }
        }),
      labelledLanes.map(lane => {
        const geometry = {
          x: (lane.index % 32) * 40 + 2,
          y: Math.floor(lane.index / 32) * 32 + 2,
          width: 36,
          height: 28
        };
        const value = probe?.values[lane.index];
        const provenance =
          probe && value !== undefined && Number.isFinite(value) ? probe.provenance : 'metadata';
        return h(
          'g',
          {'data-action': 'pick', 'data-axis': 'lane', 'data-key': lane.globalIndex},
          !compact &&
            h(
              'rect',
              {
                ...geometry,
                ...provenanceMarkAttributes(
                  provenance,
                  'currentColor',
                  `${context.idPrefix}-lanes`
                ),
                ...(lane.active ? {} : {fill: `url(#${context.idPrefix}-lanes-hatched)`})
              },
              h(
                'title',
                null,
                lane.active ? `lane ${lane.globalIndex}` : 'idle: beyond the invocation guard'
              )
            ),
          h(
            'text',
            {
              x: geometry.x + 3,
              y: geometry.y + 19,
              fill:
                provenance === 'readback' && lane.active
                  ? context.ground === 'light'
                    ? '#ffffff'
                    : '#13171c'
                  : 'currentColor',
              'data-provenance': provenance
            },
            probe ? (Number.isFinite(value) ? String(value) : 'unknown') : String(lane.globalIndex)
          ),
          lane.pinned && [
            h('rect', {
              ...geometry,
              fill: 'none',
              stroke: ink.halo,
              'stroke-width': 5,
              'data-state': 'selection-halo'
            }),
            h('rect', {
              ...geometry,
              fill: 'none',
              stroke: ink.core,
              'stroke-width': 2,
              'data-state': 'selection'
            })
          ]
        );
      })
    ),
    compact &&
      h(
        'p',
        null,
        `${model.lanes.length} lanes; first 15 and pinned lane labelled. Grey cells are idle or unavailable.`
      ),
    h(
      'p',
      null,
      probe
        ? `id probe ${probe.captureId} · ${getAdapterLabel(context.record)}`
        : `id probe not recorded; lane ids are computed from the dispatch shape · ${getAdapterLabel(context.record)}`
    ),
    renderKernelSource(model.source)
  );
}

/** Generic kernel-rung dispatch view. */
export const DISPATCH_VIEW: ViewRegistration<DispatchViewModel> = {
  id: 'dispatch-view',
  label: 'Dispatch',
  rung: 'kernel',
  recordType: 'dispatch',
  order: 50,
  signature: {
    binds: ['node'],
    free: ['workgroup', 'lane'],
    layout: {workgroup: 'position', lane: 'position'}
  },
  descriptor: {audience: 'expert', task: 'explore', interaction: 'high'},
  level: 'execution',
  grain: 'kernel',
  satisfies: context =>
    getDispatchNode(context)
      ? {ok: true}
      : {ok: false, reason: 'no compute node with a recorded dispatch shape and workgroup size'},
  build: buildDispatchViewModel,
  render: renderDispatchView,
  describe: (model, context) => ({
    subject: `Dispatch of ${model.label}`,
    keyNumber: {label: 'invocations', value: model.totalInvocations},
    facts: [
      `workgroups ${model.workgroupCount}`,
      `lanes per workgroup ${model.lanesPerWorkgroup}`,
      ...(model.lanes.length > 32
        ? ['Lane grid rasterized; first 15 and pinned lane labelled']
        : []),
      model.invocationCount === undefined
        ? 'idle lanes unknown; invocation guard unknown'
        : `idle lanes ${model.idleLaneCount}`,
      `expanded workgroup ${model.expandedWorkgroup}`,
      'status' in model.idProbe ? 'id probe not recorded' : `id probe ${model.idProbe.captureId}`
    ],
    provenance: 'status' in model.idProbe ? 'metadata' : model.idProbe.provenance,
    adapterLabel: getAdapterLabel(context.record)
  })
};

let mountCount = 0;

/** Mounts the dispatch view in a host-owned element; no DOM is accessed until called. */
export function mountDispatchView(
  element: HTMLElement,
  record: LensRecord,
  options: Partial<Omit<ViewContext, 'record'>> = {}
): {update(record: LensRecord): void; destroy(): void} {
  const defaults = {
    bindings: {},
    ground: 'light' as const,
    idPrefix: `lens-dispatch-${++mountCount}`,
    reducedMotion: false,
    ...options
  };
  let destroyed = false;
  const update = (nextRecord: LensRecord) => {
    if (destroyed) return;
    const context: ViewContext = {...defaults, record: nextRecord};
    const satisfaction = DISPATCH_VIEW.satisfies(context);
    const tree = !('reason' in satisfaction)
      ? renderDispatchView(buildDispatchViewModel(context), context)
      : h('p', {class: 'lens-disabled'}, satisfaction.reason);
    element.replaceChildren(viewNodeToDom(tree, element.ownerDocument));
  };
  update(record);
  return {
    update,
    destroy: () => {
      destroyed = true;
      element.replaceChildren();
    }
  };
}
