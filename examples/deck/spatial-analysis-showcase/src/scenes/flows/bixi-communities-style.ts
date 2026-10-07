// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {MapGround} from '../../cartography/hue-registry';
import type {ZoomStops} from '../../cartography/zoom';
import type {PaletteColor} from '../../engine/ramps';
import type {LegendSpec} from '../scene';
import {BETWEEN_GROUPS_INK, getGroupPalette} from './flows-style';

/**
 * Symbolisation of the bixi-communities scene: station and link sizes by zoom, the inks of the
 * borough outlines, hulls and seam rings, and the legends. Pure data: no luma.gl, so the scene
 * file can import it.
 */

/** Station radius in CSS pixels by zoom: 3 px at the island scale, 4.5 px from zoom 13. */
export const STATION_RADIUS_STOPS: ZoomStops = [
  [10, 3],
  [13, 4.5]
];

/** Link width in CSS pixels by zoom: 1, 1.4 and 2 px. */
export const LINK_WIDTH_STOPS: ZoomStops = [
  [10, 1],
  [12, 1.4],
  [14, 2]
];

/** Alpha (0-255) of a group's hull wash and of its 0.75 px outline. */
export const HULL_FILL_ALPHA = 26;
export const HULL_STROKE_ALPHA = 128;

/** Borough outline: 0.9 px ink at 0.6 over a white casing (the zone-boundary spec). */
export const BOROUGH_INK: Record<MapGround, PaletteColor> = {
  light: [58, 63, 75, 153],
  dark: [224, 228, 235, 153]
};
export const BOROUGH_CASING: Record<MapGround, PaletteColor> = {
  light: [255, 255, 255, 200],
  dark: [14, 17, 22, 200]
};

/** Seam ring: achromatic ink, so it never competes with a group hue. */
export const SEAM_INK: Record<MapGround, PaletteColor> = {
  light: [17, 24, 39, 255],
  dark: [244, 241, 232, 255]
};

/** The stroke around a station, in the ground colour. */
export const STATION_HALO: Record<MapGround, PaletteColor> = {
  light: [255, 255, 255, 235],
  dark: [14, 17, 22, 235]
};

/** Alpha (0-255) of the between-group links: faint, drawn first. */
export const BETWEEN_LINK_ALPHA = 46;

/** Group palette with a uniform alpha (slot 7 is the grey "other"). */
export function getCommunityPalette(ground: MapGround, alpha = 255): PaletteColor[] {
  return getGroupPalette(ground, alpha);
}

/** What the compute module publishes for the legends. */
export type CommunityLegendData = {
  entries: {slot: number; label: string; stations: number}[];
  greyGroups: number;
  greyStations: number;
  ground: MapGround;
  seamCount: number;
};

/** The options the legends read. */
export type CommunityLegendState = {
  partition: 'optimized' | 'propagation' | 'boroughs';
  showBetween: boolean;
  showBoroughs: boolean;
  showSeams: boolean;
  showHulls: boolean;
};

/** Legends: the groups (interactive, hover isolates), and the line and ring keys. */
export function getCommunityLegends(
  state: CommunityLegendState,
  data: Readonly<Record<string, unknown>>
): LegendSpec[] {
  const legend = data['communities'] as CommunityLegendData | undefined;
  const ground = legend?.ground ?? 'light';
  const palette = getCommunityPalette(ground);
  const entries = (legend?.entries ?? []).map(entry => ({
    color: palette[entry.slot],
    label: entry.label,
    count: entry.stations,
    shape: 'dot' as const
  }));
  if (legend && legend.greyGroups > 0) {
    entries.push({
      color: palette[7],
      label: `${legend.greyGroups} smaller ${legend.greyGroups === 1 ? 'group' : 'groups'}`,
      count: legend.greyStations,
      shape: 'dot' as const
    });
  }
  if (entries.length === 0) {
    entries.push({color: palette[7], label: 'Computing', count: 0, shape: 'dot' as const});
  }
  const unit = state.partition === 'boroughs' ? 'borough' : 'riding group';
  const legends: LegendSpec[] = [
    {
      kind: 'categories',
      id: 'communities',
      title: `Stations by ${unit} (stations)`,
      entries,
      interactive: true,
      layout: 'list',
      note: 'Hue is identity: it follows a group when the partition, the links or the resolution change. Only the seven largest groups of eight or more stations take a hue; lines inside a group take its hue.'
    }
  ];
  const keys: {color: PaletteColor; label: string; shape: 'line' | 'ring'}[] = [];
  if (state.showBetween) {
    keys.push({
      color: BETWEEN_GROUPS_INK[ground],
      label: 'Link between groups',
      shape: 'line'
    });
  }
  if (state.showBoroughs) {
    keys.push({color: BOROUGH_INK[ground], label: 'Borough boundary', shape: 'line'});
  }
  if (state.showSeams) {
    keys.push({
      color: SEAM_INK[ground],
      label: `Seam station (${legend?.seamCount ?? 0})`,
      shape: 'ring'
    });
  }
  if (keys.length > 0) {
    legends.push({
      kind: 'categories',
      title: 'Lines and rings',
      entries: keys,
      layout: 'list',
      note: state.showSeams
        ? 'A seam is a station whose riding group is mostly in another borough.'
        : undefined
    });
  }
  return legends;
}
