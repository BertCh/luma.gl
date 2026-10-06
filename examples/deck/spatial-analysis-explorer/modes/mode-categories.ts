// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** A navigation category of explorer modes. */
export type SpatialAnalysisModeCategory = {
  /** Stable category id. */
  id: string;
  /** Label shown in the category selector. */
  title: string;
  /** Mode ids in the category, in tab order. */
  modeIds: readonly string[];
};

/** Id of the fallback category that collects modes missing from `SPATIAL_ANALYSIS_MODE_CATEGORIES`. */
export const OTHER_CATEGORY_ID = 'other';

/**
 * Explorer navigation categories, in display order. Every registered mode should be listed once;
 * modes that are not listed appear in an automatic "Other" category so they never disappear.
 */
export const SPATIAL_ANALYSIS_MODE_CATEGORIES: readonly SpatialAnalysisModeCategory[] = [
  {
    id: 'points',
    title: 'Points & density',
    modeIds: ['density', 'time', 'lasso', 'clusters', 'line-density', 'point-pattern']
  },
  {
    id: 'joins',
    title: 'Joins & geometry',
    modeIds: [
      'polygon-join',
      'nearest-join',
      'zonal',
      'buffer',
      'distance-field',
      'geometry',
      'geometry-tools',
      'relate',
      'validity',
      'crossings',
      'simplification'
    ]
  },
  {
    id: 'statistics',
    title: 'Weights & statistics',
    modeIds: [
      'spatial-weights',
      'hot-spots',
      'group-statistics',
      'classification',
      'rates',
      'segregation'
    ]
  },
  {
    id: 'regression',
    title: 'Regression & models',
    modeIds: ['regression', 'trade-areas']
  },
  {
    id: 'cells',
    title: 'Cells (DGGS)',
    modeIds: ['cells', 'cell-pyramid', 'tile-lod']
  },
  {
    id: 'networks',
    title: 'Networks',
    modeIds: ['network-analysis', 'reachability', 'accessibility', 'isochrones', 'coverage']
  },
  {
    id: 'time',
    title: 'Trajectories & time',
    modeIds: [
      'trajectories',
      'playhead',
      'flows',
      'flow-field',
      'space-time',
      'encounters',
      'zone-events',
      'knox',
      'markov'
    ]
  },
  {
    id: 'support',
    title: 'Change of support',
    modeIds: ['areal-interpolation', 'interpolation', 'dot-density', 'raster-join']
  },
  {
    id: 'raster',
    title: 'Raster & terrain',
    modeIds: [
      'terrain',
      'relief',
      'relief-visualization',
      'geomorphometry',
      'terrain-features',
      'visibility',
      'hydrology',
      'drainage',
      'cost-distance',
      'contours',
      'suitability',
      'raster-zonal',
      'patches'
    ]
  },
  {
    id: 'recipes',
    title: 'Recipes',
    modeIds: ['recipes']
  }
];

/** Category plus the tabs it contains. */
export type SpatialAnalysisCategoryGroup<T extends {id: string}> = {
  id: string;
  title: string;
  tabs: T[];
};

/**
 * Groups tabs by category in display order, dropping empty categories. Tabs whose id is in no
 * category go to a trailing "Other" group, so newly registered modes are always reachable.
 */
export function groupModesByCategory<T extends {id: string}>(
  tabs: readonly T[]
): SpatialAnalysisCategoryGroup<T>[] {
  const tabsById = new Map(tabs.map(tab => [tab.id, tab]));
  const placed = new Set<string>();
  const groups: SpatialAnalysisCategoryGroup<T>[] = [];
  for (const category of SPATIAL_ANALYSIS_MODE_CATEGORIES) {
    const categoryTabs: T[] = [];
    for (const modeId of category.modeIds) {
      const tab = tabsById.get(modeId);
      if (tab && !placed.has(modeId)) {
        placed.add(modeId);
        categoryTabs.push(tab);
      }
    }
    if (categoryTabs.length)
      groups.push({id: category.id, title: category.title, tabs: categoryTabs});
  }
  const others = tabs.filter(tab => !placed.has(tab.id));
  if (others.length) groups.push({id: OTHER_CATEGORY_ID, title: 'Other', tabs: others});
  return groups;
}
