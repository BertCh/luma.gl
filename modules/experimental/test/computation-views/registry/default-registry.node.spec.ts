// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {mkdirSync, writeFileSync} from 'node:fs';

import {describe, expect, test} from 'vitest';

import {toRecordedLensFile} from '../../../src/computation-lenses/registry';
import {
  createDefaultViewRegistry,
  getDefaultViewPacks,
  registerDefaultViewPack
} from '../../../src/computation-views/registry/default-registry';
import type {LensRecord} from '../../../src/computation-views/registry/view-registration';
import {createViewContext} from '../../../src/computation-views/registry/view-registry';
import {
  buildLensTwin,
  renderLensTwinHtml,
  validateLensTwin
} from '../../../src/computation-views/twin/static-twin';
import {LENS_RUNG_IDS, type LensRungId} from '../../../src/computation-views/types';
import {makeScanRecord} from '../fixtures/scan-record';
import {makeRecord as makeV4Record} from '../fixtures/v4-record';
import {makeRecord as makeV5Record} from '../fixtures/v5-record';
import {makeV6Record} from '../fixtures/v6-command-graph';

/** Set `DUMP_LENS_TWINS=<dir>` to write every twin's HTML for a manual look. */
const DUMP_DIRECTORY = process.env['DUMP_LENS_TWINS'];

const FIXTURES: Array<{name: string; make: () => LensRecord; datum?: number}> = [
  {name: 'scan', make: makeScanRecord, datum: 3},
  {name: 'v4', make: makeV4Record, datum: 1},
  {name: 'v5', make: makeV5Record, datum: 2},
  {name: 'v6', make: makeV6Record}
];

function recordFile(record: LensRecord) {
  return toRecordedLensFile(record, {
    scenarioId: 'gpu-core.fixture',
    recordedAt: '2026-10-14',
    adapterLabel: 'Fixture adapter'
  });
}

describe('default view registry', () => {
  test('registers one generic view per rung (plus alternatives) and the table', () => {
    const registry = createDefaultViewRegistry();
    for (const rung of LENS_RUNG_IDS) {
      expect(registry.viewsForRung(rung).length, rung).toBeGreaterThan(0);
    }
    expect(registry.getTableView().id).toBe('table');
  });

  test('default packs registered by side effect join every new registry', () => {
    const before = getDefaultViewPacks().length;
    registerDefaultViewPack({id: 'test-pack', analysisStages: ['support', 'operator']});
    const registry = createDefaultViewRegistry();
    expect(registry.getPack('test-pack')?.analysisStages).toEqual(['support', 'operator']);
    expect(getDefaultViewPacks().length).toBe(before + 1);
  });

  for (const fixture of FIXTURES) {
    test(`every rung yields a valid, deterministic twin for fixture ${fixture.name}`, () => {
      const enabledRungs: LensRungId[] = [];
      for (const rung of LENS_RUNG_IDS) {
        const directive = {
          scenario: 'gpu-core.fixture',
          rung,
          ...(fixture.datum === undefined ? {} : {datum: String(fixture.datum)})
        };
        const twin = buildLensTwin(recordFile(fixture.make()), directive);
        const again = buildLensTwin(recordFile(fixture.make()), directive);
        expect(validateLensTwin(twin), `${fixture.name}/${rung}`).toEqual([]);
        expect(JSON.stringify(twin)).toBe(JSON.stringify(again));
        const html = renderLensTwinHtml(twin);
        expect(html, `${fixture.name}/${rung} has an empty cell`).not.toMatch(/<td[^>]*><\/td>/);
        expect(html).not.toMatch(/undefined|NaN undefined|\[object Object\]/);
        if (!twin.disabled) {
          enabledRungs.push(rung);
          expect(twin.viewId).not.toBe(rung === 'table' ? '' : 'table');
        } else {
          expect(twin.disabled.reason.length).toBeGreaterThan(8);
        }
        if (DUMP_DIRECTORY) {
          mkdirSync(DUMP_DIRECTORY, {recursive: true});
          writeFileSync(`${DUMP_DIRECTORY}/${fixture.name}-${rung}.html`, html);
        }
      }
      expect(enabledRungs).toContain('table');
      expect(enabledRungs.length).toBeGreaterThan(1);
    });
  }

  test('rung rail reports a reason for every disabled rung', () => {
    const registry = createDefaultViewRegistry();
    const context = createViewContext(makeScanRecord());
    for (const rung of registry.listRungs(context)) {
      if (rung.status === 'disabled') {
        expect(rung.reason).toBeTruthy();
      }
    }
  });
});
