// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationRecord, GPUComputationNode} from './computation-record-types';

/** Exports one graph (the last by default) as monochrome DOT with nested namespace clusters. */
export function toGraphvizDOT(
  record: GPUComputationRecord,
  options: {graph?: string; includeResources?: boolean} = {}
): string {
  const graph =
    options.graph === undefined
      ? record.graphs.at(-1)
      : record.graphs.find(candidate => candidate.id === options.graph);
  const quote = (value: string): string => JSON.stringify(value);
  if (!graph) return 'digraph computation {\n}';
  const lines = [`digraph ${quote(graph.id)} {`];
  type Group = {nodes: GPUComputationNode[]; children: Map<string, Group>};
  const root: Group = {nodes: [], children: new Map()};
  for (const node of graph.nodes) {
    let group = root;
    for (const segment of node.namespace.split('/').filter(Boolean)) {
      let child = group.children.get(segment);
      if (!child) {
        child = {nodes: [], children: new Map()};
        group.children.set(segment, child);
      }
      group = child;
    }
    group.nodes.push(node);
  }
  let clusterIndex = 0;
  const appendGroup = (group: Group, indent: string): void => {
    for (const node of group.nodes)
      lines.push(`${indent}${quote(node.id)} [label=${quote(node.label)}];`);
    for (const [segment, child] of group.children) {
      lines.push(
        `${indent}subgraph cluster_${clusterIndex++} {`,
        `${indent}  label=${quote(segment)};`
      );
      appendGroup(child, `${indent}  `);
      lines.push(`${indent}}`);
    }
  };
  appendGroup(root, '  ');
  const styles = {
    raw: 'solid',
    war: 'dashed',
    waw: 'bold',
    explicit: 'dotted',
    dataflow: 'solid',
    indirect: 'dashed, arrowhead=odiamond'
  };
  for (const edge of graph.edges) {
    lines.push(
      `  ${quote(edge.from)} -> ${quote(edge.to)} [style=${styles[edge.kind]}${edge.resource === undefined ? '' : `, label=${quote(edge.resource)}`}];`
    );
  }
  if (options.includeResources) {
    const usedIds = new Set(graph.nodes.map(node => node.id));
    for (const [index, resource] of (graph.resources ?? []).entries()) {
      let resourceId = `resource_${index}`;
      while (usedIds.has(resourceId)) resourceId = `_${resourceId}`;
      usedIds.add(resourceId);
      lines.push(`  ${quote(resourceId)} [shape=note, label=${quote(resource.id)}];`);
      for (const node of graph.nodes) {
        if (node.reads.includes(resource.id))
          lines.push(`  ${quote(resourceId)} -> ${quote(node.id)} [label="read", color=gray];`);
        if (node.writes.includes(resource.id))
          lines.push(`  ${quote(node.id)} -> ${quote(resourceId)} [label="write", color=gray];`);
      }
    }
  }
  lines.push('}');
  return lines.join('\n');
}
