// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationGraph, GPUComputationRecord} from './computation-record-types';

/** Prints a graph for golden tests and logs. The text format is NOT an API promise. */
export function printGPUComputationGraph(graph: GPUComputationGraph): string {
  const lines = [
    `graph ${graph.id} form=${graph.form} domain=${graph.domain} nodes=${graph.nodes.length} edges=${graph.edges.length}`
  ];
  for (const node of graph.nodes) {
    const indent = '  '.repeat(1 + node.namespace.split('/').filter(Boolean).length);
    const segments = [
      `${indent}[${node.kind}] ${node.id}`,
      `ns=${node.namespace || '-'}`,
      `op=${node.op ?? '-'}`,
      `loop=${node.loop ? `${node.loop.id}#${node.loop.iteration}` : '-'}`
    ];
    if (node.reads.length) segments.push(`r=${node.reads.join(',')}`);
    if (node.writes.length) segments.push(`w=${node.writes.join(',')}`);
    if (node.derivedFrom?.length) segments.push(`<- ${node.derivedFrom.join(',')}`);
    lines.push(segments.join(' '));
  }
  if (graph.edges.length) {
    lines.push('edges:');
    for (const edge of graph.edges)
      lines.push(
        `  ${edge.from} -${edge.kind}${edge.resource === undefined ? '' : `[${edge.resource}]`}-> ${edge.to}`
      );
  }
  if (graph.resources?.length) {
    lines.push('resources:');
    for (const resource of graph.resources)
      lines.push(
        `  ${resource.id} ${resource.ownership} ${resource.bytes}B alloc=${resource.allocation ?? '-'} life=${resource.firstNode ?? '-'}..${resource.lastNode ?? '-'}`
      );
  }
  return lines.join('\n');
}

/** Prints a record for golden tests and logs. The text format is NOT an API promise. */
export function printGPUComputationRecord(record: GPUComputationRecord): string {
  const adapter = [record.adapter.backend, record.adapter.variant]
    .filter(value => value !== undefined)
    .join('/');
  const lines = [
    `record ${record.id} run=${record.runId} adapter=${adapter} domains=${record.domains.join(',')}`,
    ...record.graphs.map(printGPUComputationGraph)
  ];
  if (record.decisions.length) {
    lines.push('decisions:');
    for (const decision of record.decisions) {
      lines.push(`  ${decision.id} chosen=${decision.chosen}`, '    id supported score reason');
      for (const candidate of decision.candidates)
        lines.push(
          `    ${candidate.id} ${candidate.supported} ${candidate.score ?? '-'} ${candidate.reason}`
        );
    }
  }
  if (record.choices.length) {
    lines.push('choices:');
    for (const choice of record.choices)
      lines.push(`  ${choice.stage}.${choice.name}=${JSON.stringify(choice.value)}`);
  }
  if (record.axes.axes.length || record.axes.keySpaces.length) {
    lines.push('axes:');
    for (const axis of record.axes.axes)
      lines.push(
        `  ${axis.id} kind=${axis.kind} parent=${axis.parent ?? '-'} keySpace=${axis.keySpace ?? '-'}`
      );
    for (const space of record.axes.keySpaces)
      lines.push(
        `  keySpace ${space.id} label=${JSON.stringify(space.label)} size=${space.size ?? '-'}`
      );
  }
  if (record.axes.relations.length) {
    lines.push('relations:');
    for (const relation of record.axes.relations)
      lines.push(
        `  ${relation.id} ${relation.from}->${relation.to} ${relation.cardinality} kind=${relation.kind} timeDependent=${relation.timeDependent}`
      );
  }
  return lines.join('\n');
}
