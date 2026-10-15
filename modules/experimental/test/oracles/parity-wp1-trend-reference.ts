// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import fixtureJson from './fixtures/parity-wp1-trend-reference.json';
import {decodeRound4Fixture} from './parity-wp1-round4-fixture';

type TrendCell = {
  baseStatistic: number;
  groups: {indices: number[]; variants: Record<string, [number, number]>}[];
  lookup: Record<string, [number, number]>;
};
const fixture = decodeRound4Fixture<{
  cases: Record<string, number>;
  templates: TrendCell[][];
}>(fixtureJson);

/** Select independent SciPy entries by Gi* ranks, never by GPU trend outputs. */
export async function getConditionalTrendReference(
  name: string,
  scores: number[],
  sliceCount: number
): Promise<{trendS: number[]; trendZ: number[]; trendP: number[]}> {
  const reference = await fixture;
  const cells = reference.templates[reference.cases[name]];
  const result = {trendS: [] as number[], trendZ: [] as number[], trendP: [] as number[]};
  for (const [cellIndex, cell] of cells.entries()) {
    let statistic = cell.baseStatistic;
    let tieTerm = 0;
    for (const group of cell.groups) {
      const values = group.indices.map(index => scores[cellIndex * sliceCount + index]);
      const levels = [...new Set(values)].sort((first, second) => first - second);
      const key = values.map(value => levels.indexOf(value)).join(',');
      const contribution = group.variants[key];
      if (!contribution) throw new Error(`Missing independent Kendall ordering: ${key}`);
      statistic += contribution[0];
      tieTerm += contribution[1];
    }
    const trend = cell.lookup[`${statistic},${tieTerm}`];
    if (!trend) throw new Error('Missing independent Kendall outcome');
    result.trendS.push(statistic);
    result.trendZ.push(trend[0]);
    result.trendP.push(trend[1]);
  }
  return result;
}
