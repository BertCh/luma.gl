// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ZoneKind} from './b12-zones';

/**
 * Names of the harbor zones, cleaned once for every place that prints one (tooltip, ranking
 * chart, notes, legends), and the grouping of split polygons into the one anchorage or channel
 * they belong to. Pure TypeScript.
 *
 * NOAA publishes some anchorages in several pieces ("Lower Bay Anchorage Area Number 28" appears
 * three times) and the dissolved channels repeat one fairway name; the pieces are disjoint, so
 * the story treats them as one named zone: values add up, areas add up, one fill colour.
 */

/** The zone kinds that are hand-drawn approximations rather than NOAA boundaries. */
export const APPROXIMATE_ZONE_KINDS: readonly ZoneKind[] = ['terminal', 'gate', 'tourist', 'ferry'];

/** Which zones a story step looks at. */
export type ZoneKindFilter = 'all' | 'anchorage' | 'channel' | 'approximate';

/** Whether a zone of `kind` is inside `filter`. */
export function matchesZoneKindFilter(filter: ZoneKindFilter, kind: ZoneKind): boolean {
  if (filter === 'all') return true;
  if (filter === 'approximate') return APPROXIMATE_ZONE_KINDS.includes(kind);
  return filter === kind;
}

/** Whether a zone of `kind` is hand-drawn. */
export function isApproximateZoneKind(kind: ZoneKind): boolean {
  return APPROXIMATE_ZONE_KINDS.includes(kind);
}

const ANCHORAGE_PATTERN = /^(.*?)\s*Anchorage(?:\s+Area)?(?:\s+Number)?\s*(.*)$/i;

type ParsedName = {
  /** Identity of the zone: pieces of one named zone share it. */
  key: string;
  /** Printable name before disambiguation. */
  base: string;
  /** Region of an anchorage (first words of the NOAA name), for the rare number collision. */
  region: string | null;
};

function normaliseSpaces(text: string): string {
  return text.replace(/\s+,/g, ',').replace(/\s+/g, ' ').trim();
}

function parseZoneName(rawName: string, kind: ZoneKind): ParsedName {
  const name = normaliseSpaces(rawName);
  if (kind === 'anchorage') {
    const match = ANCHORAGE_PATTERN.exec(name);
    if (match) {
      const region = match[1].split(',')[0].trim();
      const identifier = match[2].trim();
      if (!identifier) {
        return {key: `anchorage|${region}|`, base: `${region} anchorage`, region};
      }
      const isNaval = /naval/i.test(region);
      const base = isNaval
        ? `${region.charAt(0).toUpperCase()}${region.slice(1).toLowerCase()} anchorage ${identifier}`
        : `Anchorage ${identifier}`;
      return {key: `anchorage|${region}|${identifier}`, base, region: isNaval ? null : region};
    }
  }
  const cleaned = normaliseSpaces(name.replace(/\s*\(maintained channel\)\s*$/i, ''));
  return {key: `${kind}|${cleaned}`, base: cleaned, region: null};
}

/** Zones grouped into named zones. */
export type ZoneGroups = {
  groupCount: number;
  /** Group row of each zone row. */
  groupOfZone: Uint32Array;
  /** Zone rows of each group. */
  members: readonly (readonly number[])[];
  /** Cleaned, printable name of each group. */
  names: readonly string[];
  /** Kind of each group. */
  kinds: readonly ZoneKind[];
};

/**
 * Groups zone rows by cleaned name. Names are cleaned like this: "(maintained channel)" and stray
 * spaces go, "Upper Bay Anchorage Area Number 21B" becomes "Anchorage 21B", "(approx.)" stays as
 * a suffix. When two different anchorages share a number the region is appended
 * ("Anchorage 9, East River").
 */
export function groupZones(rawNames: readonly string[], kinds: readonly ZoneKind[]): ZoneGroups {
  const groupIndexByKey = new Map<string, number>();
  const members: number[][] = [];
  const parsed: ParsedName[] = [];
  const groupKinds: ZoneKind[] = [];
  const groupOfZone = new Uint32Array(rawNames.length);
  rawNames.forEach((rawName, zone) => {
    const info = parseZoneName(rawName, kinds[zone]);
    let group = groupIndexByKey.get(info.key);
    if (group === undefined) {
      group = members.length;
      groupIndexByKey.set(info.key, group);
      members.push([]);
      parsed.push(info);
      groupKinds.push(kinds[zone]);
    }
    members[group].push(zone);
    groupOfZone[zone] = group;
  });
  const baseCounts = new Map<string, number>();
  for (const info of parsed) baseCounts.set(info.base, (baseCounts.get(info.base) ?? 0) + 1);
  const names = parsed.map(info =>
    (baseCounts.get(info.base) ?? 0) > 1 && info.region ? `${info.base}, ${info.region}` : info.base
  );
  return {groupCount: members.length, groupOfZone, members, names, kinds: groupKinds};
}
