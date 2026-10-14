// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createFillNode, createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';

/**
 * Gives every zone without a raster cell a fallback cell, in a transient copy of the zone raster.
 *
 * Rules (deterministic): zones are visited by ascending row on one invocation. A zone with no cell
 * and a valid anchor cell claims that cell when the cell belongs to no zone, or to a zone that
 * keeps at least one other cell. A claimed cell stays claimed, so when several zones share an
 * anchor the lowest zone row wins. Zones whose anchor is out of range or cannot be claimed stay
 * without cells. The input raster is never written.
 *
 * Cost: one pass over the cells plus a serial loop over the zones (a few microseconds per thousand
 * zones).
 *
 * @internal
 */
export function getZoneAnchorNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    zones: GraphDataView<'uint32'>;
    zoneCount: number;
    /** `zoneCount` rows: the fallback cell index of each zone. Out of range means none. */
    anchorCells: GraphDataView<'uint32'>;
  }
): {zones: GraphDataView<'uint32'>; nodes: GPUCommandNode<Parameters>[]} {
  const {id, operation, zones, zoneCount, anchorCells} = props;
  const cellCount = zones.length;
  const effectiveZones = createTransientView(graph, `${id}-anchored-zones`, 'uint32', cellCount);
  const counts = createTransientView(graph, `${id}-anchor-counts`, 'uint32', zoneCount);
  const declarations = `const ZONE_COUNT: u32 = ${zoneCount}u;
const CELL_COUNT: u32 = ${cellCount}u;`;
  const nodes: GPUCommandNode<Parameters>[] = [
    createFillNode<Parameters>(graph, {
      id: `${id}-anchor-counts-clear`,
      operation,
      view: counts,
      type: 'u32',
      value: '0u'
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-anchor-copy`,
      operation,
      variant: 'anchor-copy',
      bindings: [
        {name: 'zones', view: zones, type: 'u32', access: 'read'},
        {name: 'effectiveZones', view: effectiveZones, type: 'u32', access: 'read_write'},
        {name: 'counts', view: counts, type: 'atomic<u32>', access: 'read_write'}
      ],
      invocationCount: cellCount,
      declarations,
      body: `let zone = zones[zonesOffset + index];
  effectiveZones[effectiveZonesOffset + index] = zone;
  if (zone < ZONE_COUNT) {
    atomicAdd(&counts[countsOffset + zone], 1u);
  }`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-anchor-claim`,
      operation,
      variant: 'anchor-claim',
      bindings: [
        {name: 'anchorCells', view: anchorCells, type: 'u32', access: 'read'},
        {name: 'effectiveZones', view: effectiveZones, type: 'u32', access: 'read_write'},
        {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
      ],
      invocationCount: 1,
      declarations,
      body: `for (var zone = 0u; zone < ZONE_COUNT; zone++) {
    if (counts[countsOffset + zone] != 0u) {
      continue;
    }
    let anchor = anchorCells[anchorCellsOffset + zone];
    if (anchor >= CELL_COUNT) {
      continue;
    }
    let owner = effectiveZones[effectiveZonesOffset + anchor];
    if (owner < ZONE_COUNT) {
      // The owner must keep a cell; a cell claimed earlier belongs to a one-cell zone.
      if (counts[countsOffset + owner] < 2u) {
        continue;
      }
      counts[countsOffset + owner] -= 1u;
    }
    effectiveZones[effectiveZonesOffset + anchor] = zone;
    counts[countsOffset + zone] = 1u;
  }`
    })
  ];
  return {zones: effectiveZones, nodes};
}
