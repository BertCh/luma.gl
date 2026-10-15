// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync, mkdirSync, readdirSync, readFileSync, unlinkSync, writeFileSync} from 'node:fs';
import {join, relative} from 'node:path';
import {describe, expect, it} from 'vitest';
// Registers scenario descriptors (example paths, elements for pack guards) and their domain packs.
import '../../src/computation-lenses/scenarios/index';
import {
  fromRecordedLensFile,
  getLensExamplePath,
  getLensScenario,
  type RecordedLensFile
} from '../../src/computation-lenses/index';
import {
  buildLensTwin,
  parseLensDirective,
  validateLensTwin,
  type LensTwin
} from '../../src/computation-views/index';
import {
  DOCS_ROOT,
  REPOSITORY_ROOT,
  listDocsMarkdownFiles,
  scanLensDirectives,
  twinKey
} from './lens-directives';

// Regenerate with UPDATE_LENS_TWINS=1 npx vitest run --project node
// modules/experimental/test/computation-docs/lens-twins.node.spec.ts.
// Only the capture run writes source records; this spec writes derived twins when opted in.
const RECORDS_DIRECTORY = join(DOCS_ROOT, 'lens-records');
const TWINS_DIRECTORY = join(RECORDS_DIRECTORY, 'twins');
const LENSES_SOURCE_DIRECTORY = join(
  REPOSITORY_ROOT,
  'modules/experimental/src/computation-lenses'
);

/** Source text of a scenario's `.example.ts` (CARD-DESIGN §1: printed under the result figure). */
function readLensExampleSource(scenarioId: string): string | undefined {
  const examplePath = join(
    LENSES_SOURCE_DIRECTORY,
    getLensScenario(scenarioId)?.example?.path ?? getLensExamplePath(scenarioId)
  );
  return existsSync(examplePath) ? readFileSync(examplePath, 'utf8') : undefined;
}

/** Catalog thumbnails exist for plain `rung=result` directives (CARD-DESIGN §7). */
function getThumbnailTwinFile(attributes: Record<string, string>): string | undefined {
  return attributes['rung'] === 'result' &&
    attributes['datum'] === undefined &&
    attributes['iteration'] === undefined &&
    attributes['view'] === undefined
    ? `${twinKey(attributes)}~thumbnail.json`
    : undefined;
}

function serializeTwin(
  recorded: RecordedLensFile,
  attributes: Record<string, string>,
  size: 'full' | 'thumbnail' = 'full'
): string {
  const directive = {
    ...attributes,
    scenario: attributes['scenario'],
    rung: attributes['rung'],
    // Author captions belong to hosts and must not affect a shared key's generated twin.
    caption: undefined
  };
  const parsed = parseLensDirective(directive);
  const envelope = {
    schema: 'luma.lens-twin/1',
    scenario: parsed.scenarioId,
    rung: parsed.rung,
    ...(attributes['datum'] === undefined ? {} : {datum: Number(attributes['datum'])}),
    ...(attributes['iteration'] === undefined ? {} : {iteration: Number(attributes['iteration'])}),
    ...(attributes['view'] === undefined ? {} : {view: attributes['view']})
  };
  const example =
    size === 'full' && parsed.rung === 'result'
      ? readLensExampleSource(parsed.scenarioId)
      : undefined;
  const twin = buildLensTwin(recorded, directive, {
    size,
    ...(example === undefined ? {} : {example})
  });
  return `${JSON.stringify({...envelope, ...twin}, null, 2)}\n`;
}

function listTwinFiles(): string[] {
  return existsSync(TWINS_DIRECTORY)
    ? readdirSync(TWINS_DIRECTORY, {withFileTypes: true})
        .filter(file => file.isFile() && file.name.endsWith('.json'))
        .map(file => file.name)
        .sort()
    : [];
}

function collectTwinDrift(
  expected: ReadonlyMap<string, string>,
  existing: ReadonlyMap<string, string>,
  referenced: ReadonlySet<string>
): string[] {
  const failures: string[] = [];
  for (const [file, text] of expected) {
    if (!existing.has(file)) failures.push(`missing twin: ${file}`);
    else if (existing.get(file) !== text) failures.push(`stale twin: ${file}`);
  }
  for (const file of existing.keys()) {
    if (!referenced.has(file)) failures.push(`orphan twin: ${file}`);
  }
  return failures.sort();
}

function collectTwinContractFailures(twin: LensTwin): string[] {
  const required = {
    caption: twin?.caption,
    'keyNumber.label': twin?.keyNumber?.label,
    'keyNumber.value': twin?.keyNumber?.value,
    description: twin?.description,
    'provenance.adapter': twin?.provenance?.adapter,
    'provenance.date': twin?.provenance?.date,
    'provenance.runId': twin?.provenance?.runId
  };
  const failures = Object.entries(required)
    .filter(([, value]) => typeof value !== 'string' || value.trim().length === 0)
    .map(([name]) => `${name} must be a non-empty string`);
  try {
    failures.push(...validateLensTwin(twin));
  } catch (error) {
    failures.push(`validateLensTwin: ${String(error)}`);
  }
  return failures;
}

describe('documentation lens twins', () => {
  it('materializes canonical twins and detects missing, stale and orphan files', () => {
    const files = listDocsMarkdownFiles();
    expect(files.length).toBeGreaterThan(0);
    const expected = new Map<string, string>();
    const referenced = new Set<string>();
    const failures: string[] = [];
    for (const file of files) {
      try {
        for (const directive of scanLensDirectives(file)) {
          const attributes = directive.attributes;
          const twinFile = `${twinKey(attributes)}.json`;
          referenced.add(twinFile);
          const thumbnailFile = getThumbnailTwinFile(attributes);
          if (thumbnailFile) referenced.add(thumbnailFile);
          const recordFile = join(RECORDS_DIRECTORY, `${attributes['scenario']}.json`);
          if (!existsSync(recordFile)) continue;
          try {
            const recorded = fromRecordedLensFile(JSON.parse(readFileSync(recordFile, 'utf8')));
            const serialized = serializeTwin(recorded, attributes);
            if (expected.has(twinFile) && expected.get(twinFile) !== serialized) {
              failures.push(`${twinFile}: directives with the same key produced different twins`);
            }
            expected.set(twinFile, serialized);
            const thumbnailFile = getThumbnailTwinFile(attributes);
            if (thumbnailFile) {
              expected.set(thumbnailFile, serializeTwin(recorded, attributes, 'thumbnail'));
            }
          } catch (error) {
            failures.push(`${relative(REPOSITORY_ROOT, file)}:${directive.line}: ${String(error)}`);
          }
        }
      } catch (error) {
        failures.push(String(error));
      }
    }
    // Never update or prune after a failed scan or build: the reference set may be incomplete.
    expect(failures).toEqual([]);
    const existingFiles = listTwinFiles();
    if (process.env['UPDATE_LENS_TWINS'] === '1') {
      if (expected.size > 0) mkdirSync(TWINS_DIRECTORY, {recursive: true});
      for (const [file, text] of expected) {
        if (
          !existsSync(join(TWINS_DIRECTORY, file)) ||
          readFileSync(join(TWINS_DIRECTORY, file), 'utf8') !== text
        )
          writeFileSync(join(TWINS_DIRECTORY, file), text);
      }
      for (const file of existingFiles) {
        if (!referenced.has(file)) unlinkSync(join(TWINS_DIRECTORY, file));
      }
    } else {
      expect(
        [...expected]
          .flatMap(([file, text]) =>
            !existsSync(join(TWINS_DIRECTORY, file))
              ? [`missing twin: ${file}`]
              : readFileSync(join(TWINS_DIRECTORY, file), 'utf8') !== text
                ? [`stale twin: ${file}`]
                : []
          )
          .concat(
            existingFiles.filter(file => !referenced.has(file)).map(file => `orphan twin: ${file}`)
          )
          .sort(),
        'Run with UPDATE_LENS_TWINS=1 to regenerate docs/lens-records/twins'
      ).toEqual([]);
    }
  });

  it('every existing twin satisfies the accessible documentation contract and size budget', () => {
    const failures: string[] = [];
    for (const file of listTwinFiles()) {
      try {
        const text = readFileSync(join(TWINS_DIRECTORY, file), 'utf8');
        const size = Buffer.byteLength(text);
        const budget = file.endsWith('~thumbnail.json') ? 48 * 1024 : 256 * 1024;
        if (size > budget) failures.push(`${file}: ${size} bytes exceeds ${budget} bytes`);
        const twin: LensTwin = JSON.parse(text);
        failures.push(...collectTwinContractFailures(twin).map(failure => `${file}: ${failure}`));
      } catch (error) {
        failures.push(`${file}: ${String(error)}`);
      }
    }
    expect(failures).toEqual([]);
  });
});

describe('twin helpers', () => {
  const recorded: RecordedLensFile = {
    schema: 'luma.lens-record/1',
    scenarioId: 'test.example',
    recordedAt: '2026-10-14T00:00:00.000Z',
    adapterLabel: 'Fixture adapter',
    record: {
      schema: 'luma.computation/1',
      id: 'test.example',
      runId: 'fixture-run',
      createdAt: '2026-10-14T00:00:00.000Z',
      adapter: {backend: 'webgpu'},
      domains: ['test'],
      graphs: [],
      axes: {axes: [], keySpaces: [], relations: []},
      refs: {formulas: [], sources: []},
      decisions: [],
      choices: []
    }
  };

  it('materializes a catalog thumbnail only for plain result directives', () => {
    expect(getThumbnailTwinFile({scenario: 'test.example', rung: 'result'})).toBe(
      'test.example~result~thumbnail.json'
    );
    expect(getThumbnailTwinFile({scenario: 'test.example', rung: 'datum'})).toBeUndefined();
    expect(
      getThumbnailTwinFile({scenario: 'test.example', rung: 'result', datum: '3'})
    ).toBeUndefined();
    const restored = fromRecordedLensFile(JSON.parse(JSON.stringify(recorded)));
    const thumbnail = JSON.parse(
      serializeTwin(restored, {scenario: 'test.example', rung: 'result'}, 'thumbnail')
    );
    expect(thumbnail).toMatchObject({rung: 'result', size: 'thumbnail'});
    expect(thumbnail.receipt).toBeUndefined();
    expect(thumbnail.example).toBeUndefined();
  });

  it('keeps zero pins in canonical keys and omits absent pins', () => {
    expect(twinKey({scenario: 'test.example', rung: 'table'})).toBe('test.example~table');
    expect(twinKey({scenario: 'test.example', rung: 'table', datum: '0', iteration: '00'})).toBe(
      'test.example~table~d0~i0'
    );
    expect(twinKey({scenario: 'test.example', rung: 'datum', view: 'neighbourhood-view'})).toBe(
      'test.example~datum~vneighbourhood-view'
    );
  });

  it('builds deterministic twins with numeric pins and host-independent captions', () => {
    const attributes = {scenario: 'test.example', rung: 'table', datum: '0', iteration: '0'};
    const restored = fromRecordedLensFile(JSON.parse(JSON.stringify(recorded)));
    const serialized = serializeTwin(restored, {...attributes, caption: 'First host caption'});
    expect(serialized).toBe(
      serializeTwin(restored, {...attributes, caption: 'Second host caption'})
    );
    const twin = JSON.parse(serialized);
    expect(twin).toMatchObject({
      schema: 'luma.lens-twin/1',
      scenario: 'test.example',
      rung: 'table',
      datum: 0,
      iteration: 0
    });
    expect(collectTwinContractFailures(twin)).toEqual([]);
    expect(serialized).toBe(`${JSON.stringify(twin, null, 2)}\n`);
    const withoutPins = JSON.parse(
      serializeTwin(restored, {scenario: 'test.example', rung: 'table'})
    );
    expect(withoutPins).not.toHaveProperty('datum');
    expect(withoutPins).not.toHaveProperty('iteration');
  });

  it('reports all drift and preserves referenced twins whose source records are pending', () => {
    expect(
      collectTwinDrift(
        new Map([
          ['missing.json', 'new'],
          ['stale.json', 'new']
        ]),
        new Map([
          ['stale.json', 'old'],
          ['orphan.json', 'old'],
          ['pending.json', 'existing']
        ]),
        new Set(['missing.json', 'stale.json', 'pending.json'])
      )
    ).toEqual(['missing twin: missing.json', 'orphan twin: orphan.json', 'stale twin: stale.json']);
  });

  it('rejects empty required contract fields', () => {
    const twin: LensTwin = JSON.parse(
      serializeTwin(recorded, {scenario: 'test.example', rung: 'table'})
    );
    twin.keyNumber.label = ' ';
    twin.description = '';
    twin.provenance.adapter = '';
    twin.provenance.date = '';
    expect(collectTwinContractFailures(twin)).toEqual([
      'keyNumber.label must be a non-empty string',
      'description must be a non-empty string',
      'provenance.adapter must be a non-empty string',
      'provenance.date must be a non-empty string'
    ]);
  });
});
