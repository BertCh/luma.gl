// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Types, constants, colours and legends of the summits story. Everything here is plain data and
 * pure functions (no GPU, no engine imports) so the light scene file can read it.
 */

import {hexToRgba} from '../../cartography/class-table';
import type {PaletteColor} from '../../engine/ramps';
import {CONTOUR_STYLE} from '../../engine/relief';
import type {LegendSpec} from '../scene';
import {
  getSnapStatusColor,
  getSnapStatusLegend,
  getSummitInk,
  SNAP_STATUS,
  SUMMIT_DROP_CLASSES,
  type SnapStatusId,
  type TerrainGroundTone
} from './terrain-palettes';

/** Option state of the summits scene. */
export type SummitsOptions = {
  summitRadius: number;
  summitMinimumDrop: number;
  summitMaximumRadius: '16' | '24' | '32';
  incompleteNeighborhood: 'reject' | 'ignore';
  catalogueError: number;
  snapRadius: number;
  snapMaximumMove: number;
  snapMaximumHeightChange: number;
  snapInterior: boolean;
  snapCatalogueHeights: boolean;
  snapDistanceRule: boolean;
  connectivity: '8' | '6';
  contourInterval: number;
  contourIndexEvery: number;
  showSummits: boolean;
  showRejected: boolean;
  showProbe: boolean;
  showEdge: boolean;
  showSnap: boolean;
  showCritical: boolean;
  showContours: boolean;
};

/** The three summit definitions of the scale ladder (`Boulder`, `Horn`, `Massif`). */
export const SCALE_PRESETS = [
  {label: 'Boulder', radius: 150, drop: 10},
  {label: 'Horn', radius: 400, drop: 100},
  {label: 'Massif', radius: 800, drop: 250}
] as const;

/** Ground metres per cell of the 4 x 4 averaged analysis grid (the DEM's 6.6 m cells times four). */
export const ANALYSIS_CELL_METERS = 26.6;

/** The radii, in metres, at which the count-by-radius curve runs the summit kernel. */
export const RADIUS_LADDER_METERS: readonly number[] = [100, 200, 300, 400, 600, 800];

/** Camera of the establishing shot, the radius lesson and the explore step. */
export const SUMMIT_FRAMES = {
  /** The Matterhorn, Zermatt and the Gornergrat (the chapter's home frame). */
  home: {longitude: 7.742, latitude: 45.985, zoom: 11.9, pitch: 0, bearing: 0},
  /** The Matterhorn, Obergabelhorn and Zmutt side: the drop test at the north-west corner. */
  drop: {longitude: 7.7, latitude: 45.995, zoom: 12.4, pitch: 0, bearing: 0},
  /** The west edge of the DEM, 350 m from the Matterhorn. */
  edge: {longitude: 7.67, latitude: 45.975, zoom: 12.6, pitch: 0, bearing: 0},
  /** The catalogue peaks of the whole tile. */
  snap: {longitude: 7.74, latitude: 45.975, zoom: 12.1, pitch: 0, bearing: 0},
  /** The Breithorn and Klein Matterhorn saddle, where contours and critical points meet. */
  critical: {longitude: 7.735, latitude: 45.94, zoom: 13, pitch: 0, bearing: 0}
} as const;

/** Zoom from which the critical-point glyphs fade in. */
export const CRITICAL_MINIMUM_ZOOM = 12.5;

/** Half-width of the fade-in, in zoom levels. */
export const CRITICAL_FADE_ZOOM = 0.15;

/** A catalogue peak within this ground distance of a candidate summit lends it its name. */
export const NAME_MATCH_METERS = 150;

/** Candidates the GPU may list: radius 100 m with no drop test finds about ten thousand on this tile. */
export const SUMMIT_LIST_CAPACITY = 32768;

/** `1,234` with a thin grouping, for tooltips and legends. */
export const formatMeters = (value: number): string =>
  `${Math.round(value).toLocaleString('en-US')}`;

/** The snap status of a `GPU_TERRAIN_PEAK_SNAP_STATUS` code, or `null` when it has no mark. */
export function getSnapStatusId(code: number): SnapStatusId | null {
  switch (code) {
    case 0:
      return 'unchanged';
    case 1:
      return 'snapped';
    case 2:
      return 'flank';
    case 3:
      return 'too-far';
    case 4:
      return 'height-mismatch';
    default:
      return null;
  }
}

/** The category palette of the snap result discs, indexed by status code (transparent beyond 4). */
export function getSnapPalette(ground: TerrainGroundTone): PaletteColor[] {
  const ids: (SnapStatusId | null)[] = [0, 1, 2, 3, 4, 5, 6, 7].map(getSnapStatusId);
  return ids.map(id => (id ? getSnapStatusColor(id, ground) : ([0, 0, 0, 0] as PaletteColor)));
}

/** Ink, halo and context colours of the summit layers on a ground. */
export function getSummitColors(ground: TerrainGroundTone) {
  const ink = getSummitInk(ground);
  const dark = ground === 'dark';
  return {
    ...ink,
    /** Catalogue positions: paper ring with an ink edge. */
    paper: ink.outline,
    /** Connector from the catalogue position to the snapped cell, 70 % ink. */
    connector: [ink.summit[0], ink.summit[1], ink.summit[2], 179] as const,
    /** Hatch of the incomplete-disc band, ink at 35 %. */
    hatch: [ink.summit[0], ink.summit[1], ink.summit[2], 89] as const,
    /** Peaks of the critical-point layer. */
    criticalPeak: ink.summit,
    dark
  };
}

/** The colour, width and opacity of a contour class on a ground (`CONTOUR_STYLE`). */
export function getContourClass(ground: TerrainGroundTone, isIndex: boolean) {
  const style = CONTOUR_STYLE[ground][isIndex ? 'index' : 'intermediate'];
  return {
    color: hexToRgba(style.color, Math.round(style.opacity * 255)),
    widthPixels: style.widthPixels
  };
}

/**
 * The class labels of the size legend for the classes that can occur at a minimum drop: the first
 * class starts at the minimum, and classes that end below it are left out.
 */
export function getDropClassEntries(minimumDrop: number) {
  const lows = [SUMMIT_DROP_CLASSES.minimumDropMeters, ...SUMMIT_DROP_CLASSES.breaks];
  const highs = [...SUMMIT_DROP_CLASSES.breaks, Number.POSITIVE_INFINITY];
  const entries: {radiusPixels: number; label: string}[] = [];
  SUMMIT_DROP_CLASSES.sizesPixels.forEach((size, index) => {
    if (highs[index] <= minimumDrop) return;
    const low = index === 0 ? minimumDrop : Math.max(lows[index], minimumDrop);
    entries.push({
      radiusPixels: size / 2,
      label: Number.isFinite(highs[index]) ? `${low}-${highs[index]}` : `over ${low}`
    });
  });
  return entries;
}

/** What the legends need from the scene's run time state. */
export type SummitsLegendData = {
  ground?: TerrainGroundTone;
  snapCounts?: Partial<Record<SnapStatusId, number>>;
};

/** The legends of the summits story for an option state. */
export function getSummitsLegends(
  state: SummitsOptions,
  data: Readonly<Record<string, unknown>>
): LegendSpec[] {
  const legendData = data as SummitsLegendData;
  const ground = legendData.ground ?? 'light';
  const colors = getSummitColors(ground);
  const legends: LegendSpec[] = [];
  if (state.showSummits) {
    legends.push({
      kind: 'size',
      title: 'Summit drop',
      unit: 'm drop',
      layout: 'nested',
      color: colors.summit,
      entries: getDropClassEntries(state.summitMinimumDrop),
      note: 'Highest within the radius and this far above the ring around it'
    });
  }
  const categories: Extract<LegendSpec, {kind: 'categories'}>['entries'][number][] = [];
  if (state.showRejected) {
    categories.push({
      color: colors.rejected,
      label: 'Highest in its disc, drop too small',
      shape: 'ring'
    });
  }
  if (state.showEdge) {
    categories.push({color: colors.hatch, label: 'Disc would leave the DEM', shape: 'hatch'});
  }
  if (categories.length) {
    legends.push({kind: 'categories', title: 'Test result', layout: 'list', entries: categories});
  }
  if (state.showSnap) {
    const base = getSnapStatusLegend(ground, legendData.snapCounts);
    legends.push(
      base.kind === 'categories'
        ? {
            ...base,
            note: 'Paper ring: where the catalogue puts the peak. The line joins it to its cell.'
          }
        : base
    );
  }
  if (state.showCritical) {
    legends.push({
      kind: 'categories',
      title: 'Critical points',
      layout: 'list',
      entries: [
        {color: colors.criticalPeak, label: 'Peak: every neighbour lower', shape: 'dot'},
        {color: colors.saddle, label: 'Saddle: lower and higher alternate', shape: 'dot'},
        {color: colors.pit, label: 'Pit: every neighbour higher', shape: 'ring'}
      ],
      note: `Drawn from zoom ${CRITICAL_MINIMUM_ZOOM} on`
    });
  }
  if (state.showContours) {
    const intermediate = getContourClass(ground, false);
    const index = getContourClass(ground, true);
    legends.push({
      kind: 'line',
      title: 'Contours',
      entries: [
        {
          color: intermediate.color,
          widthPixels: intermediate.widthPixels,
          label: `${state.contourInterval} m contour`
        },
        {
          color: index.color,
          widthPixels: index.widthPixels,
          label: `${state.contourInterval * state.contourIndexEvery} m index contour`
        }
      ]
    });
  }
  return legends;
}

/** The status list of the snap outcomes, in legend order, for the stacked chart. */
export const SNAP_STATUS_IDS: readonly SnapStatusId[] = SNAP_STATUS.map(status => status.id);
