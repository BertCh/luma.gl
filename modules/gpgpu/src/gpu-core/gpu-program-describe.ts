// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUComputationGraph,
  GPUComputationNode,
  GPUComputationEdge,
  GPUComputationSourceRef,
  GPUDecisionRecord
} from '../computation-record/computation-record-types';
import type {GPUProgram} from './gpu-program';
import type {GPUProgramCompilation} from './gpu-program-compiler';
import type {GPUOperationTree} from './gpu-operation';

/** Semantic and optional compiled forms, with source references and lowering decisions. */
export type GPUProgramDescription = {
  graphs: readonly GPUComputationGraph[];
  sources: readonly GPUComputationSourceRef[];
  decisions: readonly GPUDecisionRecord[];
};

/** Returns the stable graph identity shared by intent nodes and lowering provenance. */
export function getGPUProgramIntentGraphId(program: GPUProgram): string {
  return `${program.id}-intent`;
}

/** @internal Allocates an intent node id in pre-order; repeated names receive a numeric suffix. */
export function getGPUProgramIntentNodeId(id: string, counts: Map<string, number>): string {
  let count = counts.get(id) ?? 0;
  let uniqueId = count ? `${id}#${count}` : id;
  while (counts.has(uniqueId)) uniqueId = `${id}#${++count}`;
  counts.set(id, count + 1);
  if (uniqueId !== id) counts.set(uniqueId, 1);
  return uniqueId;
}

/**
 * Describes declared program intent and the backend's lowering choices.
 * Only the chosen lowering rule is known today; candidates do not imply an exhaustive search.
 */
export function describeGPUProgramCompilation<Parameters>(
  compilation: GPUProgramCompilation<Parameters>,
  options: {
    compiled?: {
      describe(options?: {graphId?: string}): {
        graph: GPUComputationGraph;
        sources: readonly GPUComputationSourceRef[];
      };
    };
    domain?: string;
  } = {}
): GPUProgramDescription {
  const graphId = getGPUProgramIntentGraphId(compilation.program);
  const nodes: GPUComputationNode[] = [];
  const edges: GPUComputationEdge[] = [];
  const counts = new Map<string, number>();
  const lastWriters = new Map<string, string>();
  const visit = (operation: GPUOperationTree, ancestors: readonly string[]): void => {
    const id = getGPUProgramIntentNodeId(operation.id, counts);
    const kind = ['composite', 'loop', 'conditional'].includes(operation.type)
      ? operation.type
      : 'operation';
    const opaque = operation.type === 'command-node-producer';
    const reads = operation.metadata?.inputs?.map(resource => resource.name) ?? [];
    const writes = operation.metadata?.outputs?.map(resource => resource.name) ?? [];
    const attrs: Record<string, string | number | boolean> = {...operation.metadata?.workload};
    if (operation.loop) Object.assign(attrs, operation.loop);
    if (operation.conditional) Object.assign(attrs, operation.conditional);
    if (opaque) attrs['opaque'] = true;
    nodes.push({
      id,
      label: operation.id,
      namespace: ancestors.join('/'),
      kind,
      ...(kind === 'operation' && !opaque ? {op: operation.type} : {}),
      reads,
      writes,
      attrs
    });
    for (const resource of reads) {
      const from = lastWriters.get(resource);
      if (from !== undefined) edges.push({from, to: id, kind: 'dataflow', resource});
    }
    for (const resource of writes) lastWriters.set(resource, id);
    for (const child of operation.children ?? []) visit(child, [...ancestors, id]);
  };
  for (const operation of compilation.lowering.operations) visit(operation, []);
  const graphs: GPUComputationGraph[] = [
    {
      id: graphId,
      form: 'intent',
      domain: options.domain ?? 'gpu-core',
      label: compilation.program.id,
      nodes,
      edges
    }
  ];
  const compiled = options.compiled?.describe({graphId: `${compilation.program.id}-compiled`});
  if (compiled) graphs.push(compiled.graph);
  const decisionCounts = new Map<string, number>();
  const decisions = compilation.lowering.decisions.map(decision => {
    const count = decisionCounts.get(decision.operationId) ?? 0;
    decisionCounts.set(decision.operationId, count + 1);
    return {
      id: `lowering:${decision.operationId}#${count}`,
      subject: `${graphId}:${decision.operationId}`,
      kind: 'lowering',
      chosen: decision.lowering,
      reason: decision.reason,
      method: 'rule',
      candidates: [{id: decision.lowering, supported: true, reason: decision.reason}]
    } satisfies GPUDecisionRecord;
  });
  return {graphs, sources: compiled?.sources ?? [], decisions};
}
