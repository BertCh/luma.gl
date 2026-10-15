// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {VariableKind} from '../../computation-views/types';

/**
 * Named variable kinds (Stevens level plus extensive/intensive) used by spatial operator
 * entries. Extensive quantities may be summed over supports; intensive ones may not.
 */
export const SPATIAL_VARIABLE_KINDS = {
  /** Counts of events or population: summable over supports. */
  count: {level: 'ratio', extent: 'extensive'},
  /** Rates, densities, proportions and means: not summable over supports. */
  rate: {level: 'ratio', extent: 'intensive'},
  /** Interval quantities such as z-scores, standardised values and residuals. */
  score: {level: 'interval', extent: 'intensive'},
  /** Probabilities and p-values in [0, 1]. */
  probability: {level: 'ratio', extent: 'intensive'},
  /** Metric lengths, areas and distances. */
  measure: {level: 'ratio', extent: 'extensive'},
  /** Category labels: cluster classes, region ids, cell ids. */
  category: {level: 'nominal'},
  /** Ordered classes such as ranks or quantile bins. */
  rank: {level: 'ordinal'},
  /** Directions and angles. */
  direction: {level: 'cyclic'},
  /** Positions (x, y or lon, lat): interval values with no meaningful zero; never summed. */
  coordinate: {level: 'interval', extent: 'intensive'}
} as const satisfies Readonly<Record<string, VariableKind>>;

/** Name of a shared spatial variable kind. */
export type SpatialVariableKindName = keyof typeof SPATIAL_VARIABLE_KINDS;
