// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** The question a metric answers; it decides the colour scheme of its class table. */
export type MetricFamily = 'rate' | 'share' | 'richness' | 'hour' | 'income';

/** One metric of the group-statistics map. */
export type MetricDefinition = {
  id: string;
  label: string;
  unit: string;
  /**
   * `slot` reads one float slot of the metrics buffer, `unsigned` one uint32 slot, `percentile`
   * indexes the percentile block and `derived` the columns `GPUKeyJoin` attached to the areas.
   */
  kind: 'slot' | 'unsigned' | 'percentile' | 'derived';
  family: MetricFamily;
  slot?: number;
  index?: number;
  /** Multiplier for display (shares are fractions). */
  scale?: number;
  /**
   * True when the value rests on the observations of the area, so an area under the minimum is
   * withheld. Population and income do not: they come from the tracts, not from the records.
   */
  suppressed: boolean;
  help: string;
};

export const ZONE_COUNT = 77;
export const ZONE_CAPACITY = 128;
export const PERCENTILE_COUNT = 3;
/**
 * Slot numbers of the metrics buffer. Every contributor output of the 77-row dense table lands in
 * its own slot of `ZONE_CAPACITY` rows, so one display kernel can pick a slot by number. The
 * percentile block spans two slots; slots of statistics the scene no longer asks for stay unwritten.
 */
export const SLOT = {
  counts: 0,
  hourMean: 1,
  hourMedian: 2,
  hourSd: 3,
  hourPercentiles: 4,
  hourMode: 6,
  gradeSum: 12,
  gradeMean: 13,
  categoryMode: 14,
  categoryUnique: 15,
  introducedMean: 16,
  speciesUnique: 17
} as const;
export const SLOT_COUNT = 18;

/** Derived column indexes of the join result. */
export const DERIVED = {perThousand: 0, income: 1, population: 2} as const;

/**
 * The statistics offered on the map, grouped by the question they answer. Raw counts are not on
 * the list: they are drawn as proportional circles (the `view` option), never as a fill.
 */
export const GROUP_METRICS: readonly MetricDefinition[] = [
  {
    id: 'perThousand',
    label: 'Observations per 1,000 residents',
    unit: 'per 1,000',
    kind: 'derived',
    family: 'rate',
    index: DERIVED.perThousand,
    suppressed: true,
    help: 'Rows per group divided by the residents GPUKeyJoin sums over the tracts of each area.'
  },
  {
    id: 'researchShare',
    label: 'Research-grade share',
    unit: '%',
    kind: 'slot',
    family: 'share',
    slot: SLOT.gradeMean,
    scale: 100,
    suppressed: true,
    help: 'Mean of the 0/1 research-grade column: the share of records whose identification the community confirmed.'
  },
  {
    id: 'introducedShare',
    label: 'Introduced (non-native) share',
    unit: '%',
    kind: 'slot',
    family: 'share',
    slot: SLOT.introducedMean,
    scale: 100,
    suppressed: true,
    help: 'Mean of the 0/1 introduced flag: the share of records of non-native taxa.'
  },
  {
    id: 'richness',
    label: 'Distinct taxa',
    unit: 'taxa',
    kind: 'unsigned',
    family: 'richness',
    slot: SLOT.speciesUnique,
    suppressed: true,
    help: 'Unique count of the taxon column per area. It climbs with the number of records.'
  },
  {
    id: 'hourMedian',
    label: 'Median hour of day',
    unit: 'hour',
    kind: 'slot',
    family: 'hour',
    slot: SLOT.hourMedian,
    suppressed: true,
    help: 'The 0.5 quantile of the observation hour, linearly interpolated like numpy.'
  },
  {
    id: 'income',
    label: 'Per-capita income (population-weighted)',
    unit: 'US dollars',
    kind: 'derived',
    family: 'income',
    index: DERIVED.income,
    suppressed: false,
    help: 'GPUKeyJoin sums income times residents over the tracts of each area; the map divides by the residents with a known income.'
  }
];
