// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** One metric of the group-statistics map. */
export type MetricDefinition = {
  id: string;
  label: string;
  unit: string;
  /** `slot` reads one metrics slot, `percentile` indexes the percentile block, `derived` the joined columns. */
  kind: 'slot' | 'unsigned' | 'percentile' | 'derived' | 'category';
  slot?: number;
  index?: number;
  /** Multiplier for display (rates are fractions). */
  scale?: number;
  help: string;
};

export const ZONE_COUNT = 77;
export const ZONE_CAPACITY = 128;
export const PERCENTILE_COUNT = 3;
export const SLOT = {
  counts: 0,
  hourMean: 1,
  hourMedian: 2,
  hourSd: 3,
  hourPercentiles: 4,
  hourMode: 6,
  hourUnique: 7,
  hourSkew: 8,
  hourKurtosis: 9,
  hourMinimum: 10,
  hourMaximum: 11,
  gradeSum: 12,
  gradeMean: 13,
  categoryMode: 14,
  categoryUnique: 15,
  introducedMean: 16,
  speciesUnique: 17
} as const;
export const SLOT_COUNT = 18;

/** The metrics offered on the map. */
export const GROUP_METRICS: readonly MetricDefinition[] = [
  {
    id: 'count',
    label: 'Observations',
    unit: 'observations',
    kind: 'unsigned',
    slot: SLOT.counts,
    help: 'GPUGroupStatistics counts: rows per group.'
  },
  {
    id: 'perThousand',
    label: 'Observations per 1,000 residents',
    unit: 'per 1,000',
    kind: 'derived',
    index: 0,
    help: 'Observation counts divided by the population GPUKeyJoin sums over the tracts of each area.'
  },
  {
    id: 'researchShare',
    label: 'Research-grade share',
    unit: '%',
    kind: 'slot',
    slot: SLOT.gradeMean,
    scale: 100,
    help: 'Mean of the 0/1 research-grade column: observations whose identification the community confirmed.'
  },
  {
    id: 'introducedShare',
    label: 'Introduced share',
    unit: '%',
    kind: 'slot',
    slot: SLOT.introducedMean,
    scale: 100,
    help: 'Mean of the 0/1 introduced flag: non-native taxa.'
  },
  {
    id: 'hourMean',
    label: 'Mean hour of day',
    unit: 'hour',
    kind: 'slot',
    slot: SLOT.hourMean,
    help: 'Mean of the observation hour (0 to 23).'
  },
  {
    id: 'hourMedian',
    label: 'Median hour of day',
    unit: 'hour',
    kind: 'slot',
    slot: SLOT.hourMedian,
    help: 'Linear-interpolated 0.5 quantile of the hour.'
  },
  {
    id: 'hourLow',
    label: 'Lower percentile of the hour',
    unit: 'hour',
    kind: 'percentile',
    index: 0,
    help: 'Percentile set by the lower-fraction slider.'
  },
  {
    id: 'hourHigh',
    label: 'Upper percentile of the hour',
    unit: 'hour',
    kind: 'percentile',
    index: 2,
    help: 'Percentile set by the upper-fraction slider.'
  },
  {
    id: 'hourSd',
    label: 'Spread of the hour (std. deviation)',
    unit: 'hours',
    kind: 'slot',
    slot: SLOT.hourSd,
    help: 'Sample or population deviation, per the variance option.'
  },
  {
    id: 'hourSkew',
    label: 'Skewness of the hour',
    unit: '',
    kind: 'slot',
    slot: SLOT.hourSkew,
    help: 'Fisher-Pearson g1: positive when late hours are rarer than early ones.'
  },
  {
    id: 'hourKurtosis',
    label: 'Excess kurtosis of the hour',
    unit: '',
    kind: 'slot',
    slot: SLOT.hourKurtosis,
    help: 'g2: peaked (positive) or flat (negative) distribution of hours.'
  },
  {
    id: 'hourMode',
    label: 'Most common hour (mode)',
    unit: 'hour',
    kind: 'slot',
    slot: SLOT.hourMode,
    help: 'Most frequent hour; ties go to the smallest.'
  },
  {
    id: 'richness',
    label: 'Species richness (distinct taxa)',
    unit: 'taxa',
    kind: 'unsigned',
    slot: SLOT.speciesUnique,
    help: 'Unique count of the taxon column: how many different species were seen in the area.'
  },
  {
    id: 'distinctGroups',
    label: 'Distinct groups',
    unit: 'groups',
    kind: 'unsigned',
    slot: SLOT.categoryUnique,
    help: 'Unique count of the category column.'
  },
  {
    id: 'modalGroup',
    label: 'Most common group (mode)',
    unit: '',
    kind: 'category',
    slot: SLOT.categoryMode,
    help: 'Mode of the category column, drawn with a categorical palette.'
  },
  {
    id: 'researchCount',
    label: 'Research-grade observations',
    unit: 'observations',
    kind: 'slot',
    slot: SLOT.gradeSum,
    help: 'Sum of the research-grade column.'
  },
  {
    id: 'income',
    label: 'Mean tract per-capita income',
    unit: 'US dollars',
    kind: 'derived',
    index: 1,
    help: 'GPUKeyJoin mean over the tracts inside each area.'
  },
  {
    id: 'poverty',
    label: 'Mean tract poverty',
    unit: '%',
    kind: 'derived',
    index: 2,
    help: 'GPUKeyJoin mean of poverty below 150% of the line.'
  },
  {
    id: 'population',
    label: 'Population',
    unit: 'residents',
    kind: 'derived',
    index: 4,
    help: 'GPUKeyJoin sum of tract populations.'
  },
  {
    id: 'tractCount',
    label: 'Census tracts',
    unit: 'tracts',
    kind: 'derived',
    index: 3,
    help: 'GPUKeyJoin 1:n count of the matching tracts.'
  }
];
