// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationGraph} from '@luma.gl/gpgpu';

/** Alternate projections of the same ordered command graph. */
export type CommandGraphPreset = 'algorithm' | 'pass-resource' | 'memory-aliasing';

/** Pass/resource incidence matrix; undeclared resources retain unknown ownership. */
export type PassResourceLadder = {
  passes: {nodeId: string; label: string; order: number}[];
  resources: {id: string; ownership: 'imported' | 'transient' | 'unknown'; allocation?: string}[];
  cells: {nodeId: string; resourceId: string; access: 'read' | 'write' | 'read-write'}[];
};

/** Physical allocation rows and inclusive logical lifetimes in compiler schedule coordinates. */
export type MemoryAliasing = {
  allocations: {
    allocation: string;
    bytes: number;
    allocationKnown: boolean;
    resources: {
      id: string;
      firstIndex?: number;
      lastIndex?: number;
      bytes: number;
      retained: boolean;
      lifetime: 'known' | 'unknown';
    }[];
  }[];
  imported: {id: string; bytes: number}[];
  scheduleLength: number;
  /** Peak of live physical allocation bytes; retained resources remain live through the schedule. */
  peakTransientBytes: number;
  aliasedSavingsBytes: number;
  /** Whether all allocation sizes and lifetimes are recorded; otherwise totals are partial. */
  totalsKnown: boolean;
  /** Savings need allocation sizes, independently of lifetime availability. */
  savingsKnown: boolean;
};

/** Builds one cell per node/resource pair, merging reads and writes without duplicate cells. */
export function buildPassResourceLadder(graph: GPUComputationGraph): PassResourceLadder {
  const resources: PassResourceLadder['resources'] = (graph.resources ?? []).map(
    ({id, ownership, allocation}) => ({id, ownership, allocation})
  );
  const resourceIds = new Set(resources.map(resource => resource.id));
  const cells: PassResourceLadder['cells'] = [];
  for (const node of graph.nodes) {
    const reads = new Set(node.reads);
    const writes = new Set(node.writes);
    for (const resourceId of new Set([...reads, ...writes])) {
      if (!resourceIds.has(resourceId)) {
        resourceIds.add(resourceId);
        resources.push({id: resourceId, ownership: 'unknown'});
      }
      cells.push({
        nodeId: node.id,
        resourceId,
        access: reads.has(resourceId) ? (writes.has(resourceId) ? 'read-write' : 'read') : 'write'
      });
    }
  }
  return {
    passes: graph.nodes.map((node, order) => ({
      nodeId: node.id,
      label: node.label || node.id,
      order
    })),
    resources,
    cells
  };
}

/** Groups recorded allocations without inferring lifetimes from accesses or node id spelling. */
export function buildMemoryAliasing(graph: GPUComputationGraph): MemoryAliasing {
  const orders = new Map(graph.nodes.map((node, index) => [node.id, index]));
  const allocations: MemoryAliasing['allocations'] = [];
  const byAllocation = new Map<string, MemoryAliasing['allocations'][number]>();
  const imported: MemoryAliasing['imported'] = [];
  let logicalBytes = 0;
  let totalsKnown = graph.resources !== undefined;
  for (const resource of graph.resources ?? []) {
    if (resource.ownership === 'imported') {
      imported.push({id: resource.id, bytes: resource.bytes});
      continue;
    }
    logicalBytes += resource.bytes;
    let allocation = resource.allocation ? byAllocation.get(resource.allocation) : undefined;
    if (!allocation) {
      allocation = {
        allocation: resource.allocation ?? 'unknown',
        bytes: 0,
        allocationKnown: Boolean(resource.allocation && resource.allocationBytes !== undefined),
        resources: []
      };
      allocations.push(allocation);
      if (resource.allocation) byAllocation.set(resource.allocation, allocation);
    }
    allocation.bytes = Math.max(allocation.bytes, resource.allocationBytes ?? resource.bytes);
    allocation.allocationKnown &&= resource.allocationBytes !== undefined;
    const firstIndex =
      resource.firstNode === undefined ? undefined : orders.get(resource.firstNode);
    const lastIndex = resource.lastNode === undefined ? undefined : orders.get(resource.lastNode);
    const lifetime =
      firstIndex !== undefined && lastIndex !== undefined && lastIndex >= firstIndex
        ? 'known'
        : 'unknown';
    allocation.resources.push({
      id: resource.id,
      firstIndex: lifetime === 'known' ? firstIndex : undefined,
      lastIndex: lifetime === 'known' ? lastIndex : undefined,
      bytes: resource.bytes,
      retained: resource.retained ?? false,
      lifetime
    });
    totalsKnown &&= allocation.allocationKnown && lifetime === 'known';
  }
  let peakTransientBytes = 0;
  for (let index = 0; index < graph.nodes.length; index++) {
    const liveBytes = allocations.reduce(
      (total, allocation) =>
        total +
        (allocation.resources.some(
          resource =>
            resource.firstIndex !== undefined &&
            resource.firstIndex <= index &&
            (resource.retained || resource.lastIndex! >= index)
        )
          ? allocation.bytes
          : 0),
      0
    );
    peakTransientBytes = Math.max(peakTransientBytes, liveBytes);
  }
  return {
    allocations,
    imported,
    scheduleLength: graph.nodes.length,
    peakTransientBytes,
    aliasedSavingsBytes:
      logicalBytes - allocations.reduce((total, allocation) => total + allocation.bytes, 0),
    totalsKnown,
    savingsKnown:
      graph.resources !== undefined && allocations.every(allocation => allocation.allocationKnown)
  };
}
