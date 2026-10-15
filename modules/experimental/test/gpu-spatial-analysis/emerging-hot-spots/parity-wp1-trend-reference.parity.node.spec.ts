// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import fixture from '../../oracles/fixtures/parity-wp1-trend-reference.json';
import {decodeRound4Fixture} from '../../oracles/parity-wp1-round4-fixture';
import {getConditionalTrendReference} from '../../oracles/parity-wp1-trend-reference';

type Probe = {
  name: string;
  sliceCount: number;
  scores: number[];
  expected: {trendS: number[]; trendZ: number[]; trendP: number[]};
};

it('conditional Kendall table matches full-series SciPy for original and perturbed stage inputs', async () => {
  const {probes} = await decodeRound4Fixture<{probes: Probe[]}>(fixture);
  expect(probes).toHaveLength(36);
  for (const probe of probes) {
    const actual = await getConditionalTrendReference(probe.name, probe.scores, probe.sliceCount);
    expect(actual, probe.name).toEqual(probe.expected);
  }
});

it('conditional Kendall reference pins its source fixtures and independent library versions', () => {
  expect(fixture.versions.scipy).toBeTruthy();
  for (const [name, hash] of Object.entries(fixture.sourceSha256)) {
    const bytes = readFileSync(new URL(`../../oracles/fixtures/${name}`, import.meta.url));
    expect(createHash('sha256').update(bytes).digest('hex'), name).toBe(hash);
  }
});
