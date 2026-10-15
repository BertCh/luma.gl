// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationGraph, GPUComputationEdgeKind} from '@luma.gl/gpgpu';

/** A declared node, optionally representing several unrolled loop copies. */
export type FoldedNode = {
  id: string;
  label: string;
  memberIds: string[];
  loopId?: string;
  iterationCount: number;
  maximumIterations?: number;
};

/** Longest-path layers in compiler order; backward and self edges are counted and ignored. */
export function layerGraph(
  nodes: readonly {id: string}[],
  edges: readonly {from: string; to: string}[]
): {layers: string[][]; depth: number; order: Map<string, number>; backEdgeCount: number} {
  const order = new Map(nodes.map((node, index) => [node.id, index]));
  const predecessors = new Map<string, string[]>();
  let backEdgeCount = 0;
  for (const edge of edges) {
    const from = order.get(edge.from);
    const to = order.get(edge.to);
    if (from === undefined || to === undefined) continue;
    if (from >= to) {
      backEdgeCount++;
      continue;
    }
    const incoming = predecessors.get(edge.to) ?? [];
    incoming.push(edge.from);
    predecessors.set(edge.to, incoming);
  }
  const depths = new Map<string, number>();
  const layers: string[][] = [];
  for (const node of nodes) {
    let depth = 0;
    for (const predecessor of predecessors.get(node.id) ?? []) {
      depth = Math.max(depth, (depths.get(predecessor) ?? 0) + 1);
    }
    depths.set(node.id, depth);
    layers[depth] ??= [];
    layers[depth].push(node.id);
  }
  return {layers, depth: layers.length, order, backEdgeCount};
}

/** Folds loop copies by explicit loop identity and operation name, preserving declaration order. */
export function foldLoops(graph: GPUComputationGraph): {
  nodes: FoldedNode[];
  edges: {from: string; to: string; kinds: GPUComputationEdgeKind[]}[];
} {
  const nodes = new Map<string, FoldedNode>();
  const membership = new Map<string, string>();
  const iterations = new Map<string, Set<number>>();
  for (const node of graph.nodes) {
    const label = node.dedupName ?? node.label;
    const id = node.loop ? `loop:${node.loop.id}/${label}` : node.id;
    membership.set(node.id, id);
    let folded = nodes.get(id);
    if (!folded) {
      folded = {id, label, memberIds: [], loopId: node.loop?.id, iterationCount: 0};
      nodes.set(id, folded);
    }
    folded.memberIds.push(node.id);
    if (node.loop) {
      const seen = iterations.get(id) ?? new Set<number>();
      seen.add(node.loop.iteration);
      iterations.set(id, seen);
      folded.iterationCount = seen.size;
      if (node.loop.maximumIterations !== undefined) {
        folded.maximumIterations = Math.max(
          folded.maximumIterations ?? 0,
          node.loop.maximumIterations
        );
      }
    }
  }
  const edges = new Map<string, {from: string; to: string; kinds: GPUComputationEdgeKind[]}>();
  for (const edge of graph.edges) {
    const from = membership.get(edge.from);
    const to = membership.get(edge.to);
    if (!from || !to || from === to) continue;
    const key = JSON.stringify([from, to]);
    const folded = edges.get(key) ?? {from, to, kinds: []};
    if (!folded.kinds.includes(edge.kind)) folded.kinds.push(edge.kind);
    edges.set(key, folded);
  }
  return {nodes: [...nodes.values()], edges: [...edges.values()]};
}
