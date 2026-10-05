// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Zone ID written for rejected or unused rows. */
export const GPU_FLOW_AGGREGATION_NO_ZONE = 0xffffffff;

/** Largest zone count whose pair keys `originZone * zoneCount + destinationZone` fit in a uint32. */
export const GPU_FLOW_AGGREGATION_MAXIMUM_ZONE_COUNT = 65535;

/**
 * Pair key of a flow: `originZone * zoneCount + destinationZone`.
 *
 * Keys are always below `0xffffffff`, which the hash table reserves as its empty marker.
 */
export function getGPUFlowPairKey(
  originZone: number,
  destinationZone: number,
  zoneCount: number
): number {
  return originZone * zoneCount + destinationZone;
}

/** Inverse of {@link getGPUFlowPairKey}: `[originZone, destinationZone]`. */
export function getGPUFlowPairZones(pairKey: number, zoneCount: number): [number, number] {
  return [Math.floor(pairKey / zoneCount), pairKey % zoneCount];
}
