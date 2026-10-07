// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The legends and class tables of the viewshed story that are not in the chapter palette file: the
 * pyramid blocks, the sight line, the cells that flip between earth models and the lookouts. The
 * scene's legends and the layers read the same objects from here.
 */

import {getClassTableLegend, hexToRgba, makeClassTable} from '../../cartography/class-table';
import {MAP_INK, NO_DATA_COLOR} from '../../cartography/hue-registry';
import type {ClassTable} from '../../cartography/types';
import type {LegendSpec} from '../scene';
import {
  ELEVATION_BREAKS,
  getObserverColors,
  makeCumulativeTable,
  type TerrainGroundTone
} from './terrain-palettes';

/** Class value of a cell with no data: the raster layer's uint32 no-data sentinel. */
export const NO_DATA_CLASS = 0xffffffff;

/** Value of the lookout-count raster for a cell beyond the reach of every lookout. */
export const BEYOND_REACH_CLASS = 7;

/** Slots of the lookout count and observers buffers. */
export const LOOKOUT_SLOTS = 6;

/** Fine cells per coarse cell of the unseen mask. */
export const COARSE_STRIDE = 8;

/** Size of the coarse unseen mask of a grid. */
export function getCoarseSize(width: number, height: number): {width: number; height: number} {
  return {width: Math.ceil(width / COARSE_STRIDE), height: Math.ceil(height / COARSE_STRIDE)};
}

/**
 * The highest ground of each pyramid block in neutral greys (a height must not borrow the veil, the
 * lookout counts or the relief tints), on the same 7 elevation classes as the relief legend.
 */
export function makePyramidTable(ground: TerrainGroundTone): ClassTable {
  return makeClassTable({
    breaks: ELEVATION_BREAKS,
    scheme: 'Greys',
    ground,
    alpha: 200,
    unit: 'm',
    method: 'Highest ground in each block of the extrema pyramid',
    noData: {label: 'No data', color: NO_DATA_COLOR[ground]}
  });
}

/**
 * The lookout-count table as the layer draws it: the chapter table plus the clear class for cells
 * beyond every lookout's reach (class value {@link BEYOND_REACH_CLASS}).
 */
export function getCumulativeLayerTable(ground: TerrainGroundTone, observerCount: number) {
  const table = makeCumulativeTable(ground, observerCount);
  return {
    breaks: [...table.breaks, BEYOND_REACH_CLASS],
    colors: [
      ...table.colors.map(color => [color[0], color[1], color[2], color[3] ?? 255] as const),
      [0, 0, 0, 0] as const
    ]
  };
}

/** The legend of the lookout counts: classes 0 to n, a count in lookouts, the clear class noted. */
export function getCumulativeLegend(ground: TerrainGroundTone, observerCount: number): LegendSpec {
  const table = makeCumulativeTable(ground, observerCount);
  return getClassTableLegend(
    {
      ...table,
      noData: {label: "Beyond every lookout's reach", color: [0, 0, 0, 0]}
    },
    {
      title: 'Lookouts that see the ground',
      id: 'lookouts',
      layout: 'list',
      note: 'Dark: seen from none of them. Clear: beyond the reach of every lookout.'
    }
  );
}

/** Sight-line key: solid when visible, dashed when marginal, dotted when hidden (ink, never hue). */
export function getSightLineLegend(ground: TerrainGroundTone): LegendSpec {
  const ink = hexToRgba(MAP_INK[ground].ink);
  return {
    kind: 'line',
    title: 'Sight line to the teal target',
    entries: [
      {color: ink, widthPixels: 2.2, label: 'Visible (solid)'},
      {color: ink, widthPixels: 2.2, label: 'Marginal (dashed)', dashed: true},
      {color: ink, widthPixels: 2.2, label: 'Hidden (dotted)', dashed: true}
    ]
  };
}

/** The cells that flip class when the earth is flat instead of curved. */
export function getFlipLegend(ground: TerrainGroundTone): LegendSpec {
  return {
    kind: 'categories',
    title: 'Curved against flat earth',
    entries: [
      {
        color: getObserverColors(ground).changed,
        label: 'Cell changes class',
        shape: 'swatch'
      }
    ],
    note: 'Each mark is one cell, drawn three cells wide so it shows at this zoom.'
  };
}

/** Names the lookouts of the cumulative step: the gold eye is the first, the numbers follow. */
export function getLookoutLegend(ground: TerrainGroundTone, names: readonly string[]): LegendSpec {
  const colors = getObserverColors(ground);
  return {
    kind: 'categories',
    title: 'Lookouts',
    layout: 'list',
    entries: names.map((name, index) => ({
      color: colors.lookout,
      label: index === 0 ? `${name} (gold eye, draggable)` : `${index + 1}  ${name}`,
      shape: 'dot' as const
    }))
  };
}
