// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPUComputationGraphForm} from '@luma.gl/gpgpu';
import type {LensRecord, ViewContext, ViewRegistration} from '../registry/view-registration';
import {h, viewNodeToDom, type ViewElementNode} from '../view-node';
import {provenanceMarkAttributes} from '../encoding/provenance';
import {getGraphByForm} from './record-queries';
import {foldLoops, layerGraph} from './graph-layering';

/** Declared dependency structure, independent of execution timings. */
export type SkeletonViewModel = {
  graphId: string;
  label: string;
  form: GPUComputationGraphForm;
  nodes: {
    id: string;
    label: string;
    layer: number;
    loop?: {iterationCount: number; maximumIterations?: number};
  }[];
  edges: ReturnType<typeof foldLoops>['edges'];
  layers: number;
  work: number;
  workUnit: 'nodes' | 'invocations';
  depth: number;
  caption: string;
};

/** Selects the most abstract available graph and folds explicit loop coordinates. */
export function buildSkeletonViewModel(context: ViewContext): SkeletonViewModel {
  const graph = getGraphByForm(context.record, ['intent', 'lowered', 'compiled']);
  // The registration requires an available declared graph.
  assert(graph);
  const folded = foldLoops(graph);
  const layering = layerGraph(folded.nodes, folded.edges);
  const layerById = new Map(
    layering.layers.flatMap((layer, index) => layer.map(id => [id, index] as const))
  );
  const hasWorkload =
    graph.nodes.length > 0 &&
    graph.nodes.every(node => node.workload?.maximumInvocationCount !== undefined);
  return {
    graphId: graph.id,
    label: graph.label ?? graph.id,
    form: graph.form,
    nodes: folded.nodes.map(node => {
      const source = graph.nodes.find(candidate => candidate.id === node.memberIds[0]);
      const maximumIterations =
        node.maximumIterations ??
        (typeof source?.attrs['maximumIterations'] === 'number'
          ? source.attrs['maximumIterations']
          : undefined);
      return {
        id: node.id,
        label: node.label,
        layer: layerById.get(node.id) ?? 0,
        loop:
          node.loopId || source?.kind === 'loop'
            ? {iterationCount: node.iterationCount, maximumIterations}
            : undefined
      };
    }),
    edges: folded.edges,
    layers: layering.depth,
    depth: layering.depth,
    work: hasWorkload
      ? graph.nodes.reduce((sum, node) => sum + node.workload!.maximumInvocationCount!, 0)
      : graph.nodes.length,
    workUnit: hasWorkload ? 'invocations' : 'nodes',
    caption: "Declared structure, not this run's schedule"
  };
}

/** Renders outline-only structure, summarizing large graphs by layer. */
export function renderSkeletonView(
  model: SkeletonViewModel,
  context: ViewContext
): ViewElementNode {
  const summary = model.nodes.length > 40;
  const layers = Array.from({length: model.layers}, (_, index) =>
    model.nodes.filter(node => node.layer === index)
  );
  const width = Math.max(160, model.layers * 180);
  const height = Math.max(80, ...layers.map(layer => (summary ? 1 : layer.length) * 70 + 10));
  const attributes = provenanceMarkAttributes('metadata', 'currentColor', context.idPrefix);
  const positions = new Map(
    model.nodes.map(node => [
      node.id,
      {x: node.layer * 180 + 10, y: (summary ? 0 : layers[node.layer].indexOf(node)) * 70 + 10}
    ])
  );
  return h(
    'section',
    {class: 'lens-view lens-skeleton-view'},
    h('h3', null, model.label),
    h(
      'figure',
      null,
      h(
        'svg',
        {
          viewBox: `0 0 ${width} ${height}`,
          role: 'img',
          'aria-label': 'Declared dependency layers'
        },
        summary
          ? layers.map((layer, index) =>
              h(
                'g',
                {'data-aggregate': 'layer'},
                h('rect', {x: index * 180 + 10, y: 10, width: 150, height: 50, ...attributes}),
                h(
                  'text',
                  {x: index * 180 + 20, y: 40, fill: 'currentColor'},
                  `${layer.length} nodes`
                )
              )
            )
          : [
              model.edges.map(edge => {
                const from = positions.get(edge.from);
                const to = positions.get(edge.to);
                return from && to
                  ? h('line', {
                      x1: from.x + 150,
                      y1: from.y + 25,
                      x2: to.x,
                      y2: to.y + 25,
                      ...attributes,
                      opacity: 0.2
                    })
                  : null;
              }),
              model.nodes.map(node => {
                const position = positions.get(node.id)!;
                return h(
                  'g',
                  {'data-action': 'bind', 'data-axis': 'node', 'data-key': node.id},
                  h('rect', {...position, width: 150, height: 50, ...attributes}),
                  h(
                    'text',
                    {x: position.x + 8, y: position.y + 20, fill: 'currentColor'},
                    node.label
                  ),
                  node.loop &&
                    h(
                      'text',
                      {x: position.x + 8, y: position.y + 40, fill: 'currentColor'},
                      node.loop.iterationCount
                        ? `×${node.loop.iterationCount}`
                        : node.loop.maximumIterations !== undefined
                          ? `≤ ${node.loop.maximumIterations}`
                          : 'iterations unknown'
                    )
                );
              })
            ]
      ),
      h('figcaption', null, model.caption)
    )
  );
}

/** Generic declared-structure iteration view. */
export const SKELETON_VIEW: ViewRegistration<SkeletonViewModel> = {
  id: 'skeleton-view',
  label: 'Declared structure',
  rung: 'iterations',
  recordType: 'graph',
  order: 60,
  signature: {
    binds: [],
    free: ['step', 'element'],
    layout: {step: 'position', element: 'position'}
  },
  descriptor: {audience: 'analyst', task: 'explain', interaction: 'medium'},
  level: 'skeleton',
  grain: 'contributor',
  satisfies: context =>
    getGraphByForm(context.record, ['intent', 'lowered', 'compiled'])
      ? {ok: true}
      : {ok: false, reason: 'no graph form to draw a declared structure from'},
  build: buildSkeletonViewModel,
  render: renderSkeletonView,
  describe: model => ({
    subject: `Declared structure of ${model.label}`,
    keyNumber: {label: 'depth', value: model.layers},
    facts: [
      `work ${model.work} ${model.workUnit}`,
      `loops ${model.nodes.filter(node => node.loop).length}`
    ],
    provenance: 'metadata'
  })
};

let mountCount = 0;

/** Mounts the skeleton view in a host-owned element; no DOM is accessed until called. */
export function mountSkeletonView(
  element: HTMLElement,
  record: LensRecord,
  options: Partial<Omit<ViewContext, 'record'>> = {}
): {update(record: LensRecord): void; destroy(): void} {
  const defaults = {
    bindings: {},
    ground: 'light' as const,
    idPrefix: `lens-skeleton-${++mountCount}`,
    reducedMotion: false,
    ...options
  };
  let destroyed = false;
  const update = (nextRecord: LensRecord) => {
    if (destroyed) return;
    const context: ViewContext = {...defaults, record: nextRecord};
    const satisfaction = SKELETON_VIEW.satisfies(context);
    const tree = !('reason' in satisfaction)
      ? renderSkeletonView(buildSkeletonViewModel(context), context)
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
