// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationEdgeKind, GPUComputationGraph, GPUComputationNode} from '@luma.gl/gpgpu';
import {getNamespaceSegments} from './record-queries';

/** A visible leaf or a derived namespace/loop aggregate, in compiler order. */
export type GraphUnit = {
  id: string;
  kind: 'node' | 'group' | 'loop';
  label: string;
  memberIds: readonly string[];
  depth: number;
  order: number;
  doi: number;
  tier: 'subject' | 'input' | 'context';
  iterationCount?: number;
};

/** A partition of the graph leaves with aggregated dependencies between visible units. */
export type FoldedGraph = {
  units: GraphUnit[];
  edges: {
    from: string;
    to: string;
    kinds: Partial<Record<GPUComputationEdgeKind, number>>;
    resources: string[];
  }[];
  budget: number;
  /** Leaves not visible as their own unit, including when mandatory expansion exceeds budget. */
  hiddenNodeCount: number;
  focusUnitId?: string;
};

type UnitTree = {unit: GraphUnit; path: string[]; children: UnitTree[]};

/**
 * Derives a namespace tree and greedily expands API − tree-distance in schedule order.
 * Mandatory focus/reader expansions can exceed the budget. Wide root frontiers are grouped
 * into explicit contiguous schedule ranges, without inventing semantic namespaces.
 * Loop copies are placed at their common namespace ancestor so each leaf occurs exactly once.
 */
export function foldGraphByDegreeOfInterest(
  graph: GPUComputationGraph,
  options: {focus?: string; budget?: number; foldLoops?: boolean; expanded?: readonly string[]} = {}
): FoldedGraph {
  const budget = Number.isFinite(options.budget) ? Math.max(1, Math.floor(options.budget!)) : 24;
  const nodes = new Map(graph.nodes.map(node => [node.id, node]));
  const orders = new Map(graph.nodes.map((node, index) => [node.id, index]));
  const imported = new Set(
    graph.resources
      ?.filter(resource => resource.ownership === 'imported')
      .map(resource => resource.id)
  );
  const groups = new Map<string, UnitTree>();
  const roots: UnitTree[] = [];
  const expanded = new Set(
    (options.expanded ?? []).map(path =>
      path.startsWith('group:') || path.startsWith('loop:') ? path : `group:${path}`
    )
  );
  const focusNode = options.focus ? nodes.get(options.focus) : undefined;
  const focusPath = focusNode
    ? [...getNamespaceSegments(focusNode.namespace), `node:${focusNode.id}`]
    : options.focus?.startsWith('group:')
      ? getNamespaceSegments(options.focus.slice(6))
      : undefined;

  function makeUnit(
    kind: GraphUnit['kind'],
    id: string,
    label: string,
    members: readonly GPUComputationNode[],
    path: string[]
  ): UnitTree {
    const depth = kind === 'group' ? path.length - 1 : path.length;
    return {
      unit: {
        id,
        kind,
        label,
        memberIds: members.map(node => node.id),
        depth,
        order: Math.min(...members.map(node => orders.get(node.id)!)),
        doi: 0,
        tier: 'subject'
      },
      path,
      children: []
    };
  }

  function getChildren(path: string[]): UnitTree[] {
    if (!path.length) return roots;
    const key = path.join('/');
    let group = groups.get(key);
    if (!group) {
      group = makeUnit('group', `group:${key}`, key, [], path);
      groups.set(key, group);
      getChildren(path.slice(0, -1)).push(group);
    }
    return group.children;
  }

  const loops = new Map<string, GPUComputationNode[]>();
  for (const node of graph.nodes) {
    if (options.foldLoops !== false && node.loop && node.dedupName) {
      const key = `loop:${node.loop.id}/${node.dedupName}`;
      const members = loops.get(key) ?? [];
      members.push(node);
      loops.set(key, members);
    }
  }
  const loopMembers = new Set([...loops.values()].flat().map(node => node.id));
  for (const node of graph.nodes) {
    if (!loopMembers.has(node.id)) {
      const path = getNamespaceSegments(node.namespace);
      getChildren(path).push(
        makeUnit('node', `node:${node.id}`, node.label || node.id, [node], path)
      );
    }
  }
  for (const [id, members] of loops) {
    const path = getNamespaceSegments(members[0].namespace);
    for (const member of members) {
      const memberPath = getNamespaceSegments(member.namespace);
      path.splice(commonLength(path, memberPath));
    }
    const iterationCount = new Set(members.map(node => node.loop!.iteration)).size;
    const tree = makeUnit('loop', id, `${members[0].dedupName} ×${iterationCount}`, members, path);
    tree.unit.iterationCount = iterationCount;
    tree.children = members.map(node =>
      makeUnit(
        'node',
        `node:${node.id}`,
        node.label || node.id,
        [node],
        getNamespaceSegments(node.namespace)
      )
    );
    getChildren(path).push(tree);
  }

  function finishTree(tree: UnitTree): void {
    tree.children.forEach(finishTree);
    tree.children.sort((left, right) => left.unit.order - right.unit.order);
    if (tree.unit.kind === 'group') {
      tree.unit.memberIds = tree.children
        .flatMap(child => child.unit.memberIds)
        .sort((left, right) => orders.get(left)! - orders.get(right)!);
      tree.unit.order = Math.min(...tree.children.map(child => child.unit.order));
    }
    const members = tree.unit.memberIds.map(id => nodes.get(id)!);
    const api =
      -tree.unit.depth +
      Number(members.some(node => node.publication)) +
      (members.some(node => node.writes.some(id => imported.has(id))) ? 0.5 : 0);
    const unitPath = tree.unit.kind === 'group' ? tree.path : [...tree.path, tree.unit.id];
    const distance = focusPath
      ? unitPath.length + focusPath.length - 2 * commonLength(unitPath, focusPath)
      : 0;
    tree.unit.doi = api - distance;
  }
  roots.forEach(finishTree);
  let visible = roots.sort((left, right) => left.unit.order - right.unit.order);
  if (visible.length > budget && !options.focus && !expanded.size) {
    const span = Math.ceil(visible.length / budget);
    const ranges: UnitTree[] = [];
    for (let start = 0; start < visible.length; start += span) {
      const children = visible.slice(start, start + span);
      const members = children.flatMap(child => child.unit.memberIds);
      ranges.push({
        unit: {
          id: `group:schedule/${start}`,
          kind: 'group',
          label: `Schedule range: ${members.length} nodes`,
          memberIds: members,
          depth: 0,
          order: children[0].unit.order,
          doi: 0,
          tier: 'context'
        },
        path: [],
        children
      });
    }
    visible = ranges;
  }
  const expandedLoopMembers = new Set(
    [...loops]
      .filter(([id]) => expanded.has(id))
      .flatMap(([, members]) => members.map(node => node.id))
  );
  const mandatory = (tree: UnitTree): boolean => {
    if (tree.unit.kind === 'node') return false;
    if (expanded.has(tree.unit.id)) return true;
    if (tree.unit.memberIds.some(id => expandedLoopMembers.has(id))) return true;
    if (focusNode && tree.unit.memberIds.includes(focusNode.id)) return true;
    if (tree.unit.kind !== 'group') return false;
    const prefix = `${tree.unit.id}/`;
    return (
      Boolean(options.focus?.startsWith(prefix)) || [...expanded].some(id => id.startsWith(prefix))
    );
  };
  function expand(tree: UnitTree): void {
    visible = visible.flatMap(candidate => (candidate === tree ? tree.children : [candidate]));
    visible.sort((left, right) => left.unit.order - right.unit.order);
  }
  for (let candidate = visible.find(mandatory); candidate; candidate = visible.find(mandatory))
    expand(candidate);
  while (true) {
    const candidate = visible
      .filter(
        tree =>
          tree.unit.kind === 'group' &&
          tree.unit.id !== options.focus &&
          visible.length - 1 + tree.children.length <= budget
      )
      .sort(
        (left, right) => right.unit.doi - left.unit.doi || left.unit.order - right.unit.order
      )[0];
    if (!candidate) break;
    expand(candidate);
  }

  const units = visible.map(tree => tree.unit);
  const owner = new Map(units.flatMap(unit => unit.memberIds.map(id => [id, unit.id] as const)));
  const focusUnitId = focusNode
    ? owner.get(focusNode.id)
    : units.find(unit => unit.id === options.focus)?.id;
  const focusMembers = new Set(
    focusNode
      ? [focusNode.id]
      : options.focus?.startsWith('group:')
        ? groups.get(options.focus.slice(6))?.unit.memberIds
        : []
  );
  const neighbors = new Set<string>();
  for (const edge of graph.edges) {
    if (focusMembers.has(edge.from)) neighbors.add(edge.to);
    if (focusMembers.has(edge.to)) neighbors.add(edge.from);
  }
  for (const unit of units) {
    unit.tier =
      !focusMembers.size || unit.memberIds.some(id => focusMembers.has(id))
        ? 'subject'
        : unit.memberIds.some(id => neighbors.has(id))
          ? 'input'
          : 'context';
  }
  const edges = new Map<string, FoldedGraph['edges'][number]>();
  for (const edge of graph.edges) {
    const from = owner.get(edge.from);
    const to = owner.get(edge.to);
    if (!from || !to || from === to) continue;
    const key = JSON.stringify([from, to]);
    const aggregate = edges.get(key) ?? {from, to, kinds: {}, resources: []};
    aggregate.kinds[edge.kind] = (aggregate.kinds[edge.kind] ?? 0) + 1;
    if (edge.resource && !aggregate.resources.includes(edge.resource))
      aggregate.resources.push(edge.resource);
    edges.set(key, aggregate);
  }
  return {
    units,
    edges: [...edges.values()],
    budget,
    hiddenNodeCount: graph.nodes.length - units.filter(unit => unit.kind === 'node').length,
    focusUnitId
  };
}

function commonLength(left: readonly string[], right: readonly string[]): number {
  let length = 0;
  while (length < left.length && length < right.length && left[length] === right[length]) length++;
  return length;
}
