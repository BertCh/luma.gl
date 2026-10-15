// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationRecord} from './computation-record-types';

/** Model Explorer collection, containing one graph per recorded form. */
export type ModelExplorerGraphCollection = {label: string; graphs: ModelExplorerGraph[]};
/** A Model Explorer graph with stable node ids. */
export type ModelExplorerGraph = {id: string; nodes: ModelExplorerNode[]};
/** Model Explorer input_graph.ts node shape. */
export type ModelExplorerNode = {
  id: string;
  label: string;
  namespace: string;
  attrs?: {key: string; value: string}[];
  incomingEdges?: {sourceNodeId: string; sourceNodeOutputId: string; targetNodeInputId: string}[];
};

/** Exports all forms, hierarchy, attributes and incoming dependencies as Model Explorer JSON. */
export function toModelExplorerGraph(record: GPUComputationRecord): ModelExplorerGraphCollection {
  return {
    label: record.id,
    graphs: record.graphs.map(graph => ({
      id: graph.id,
      nodes: graph.nodes.map(node => {
        const attrs: {key: string; value: string}[] = [];
        const add = (key: string, value: unknown): void => {
          if (value !== undefined)
            attrs.push({key, value: typeof value === 'string' ? value : JSON.stringify(value)});
        };
        add('kind', node.kind);
        add('op', node.op);
        add('form', graph.form);
        add('derivedFrom', node.derivedFrom?.join(','));
        if (node.loop) add('loop', `${node.loop.id}#${node.loop.iteration}`);
        add('reads', node.reads.join(','));
        add('writes', node.writes.join(','));
        for (const key of Object.keys(node.workload ?? {}).sort())
          add(`workload.${key}`, node.workload?.[key as keyof typeof node.workload]);
        for (const key of Object.keys(node.attrs).sort()) add(key, node.attrs[key]);
        return {
          id: node.id,
          label: node.label,
          namespace: node.namespace,
          attrs,
          incomingEdges: graph.edges
            .filter(edge => edge.to === node.id)
            .map(edge => ({
              sourceNodeId: edge.from,
              sourceNodeOutputId: '0',
              targetNodeInputId: `${edge.kind}${edge.resource === undefined ? '' : `:${edge.resource}`}`
            }))
        };
      })
    }))
  };
}
