// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {findViewNodes} from '../../../../src/computation-views/view-node';
import {renderGPUCoreArrayResult} from '../../../../src/computation-views/packs/gpu-core/array-result-view';
import {getArrayCellBounds} from '../../../../src/computation-views/packs/gpu-core/array-strip-render';
import {mkdirSync, writeFileSync} from 'node:fs';
import {describe, expect, test} from 'vitest';
import type {GPUComputationCapture} from '@luma.gl/gpgpu';
import {toRecordedLensFile} from '../../../../src/computation-lenses/registry';
import {
  GPU_CORE_ARRAY_RESULT_VIEW,
  buildGPUCoreArrayResultModel
} from '../../../../src/computation-views/packs/gpu-core';
import {loadGPUCorePackData} from '../../../../src/computation-views/packs/gpu-core/pack-data';
import {createViewContext} from '../../../../src/computation-views/registry/view-registry';
import {
  buildLensTwin,
  renderLensTwinHtml,
  validateLensTwin
} from '../../../../src/computation-views/twin/static-twin';
import type {LensRecord} from '../../../../src/computation-views/registry/view-registration';
import {makeScanRecord} from '../../fixtures/scan-record';

const DUMP_DIRECTORY = process.env['DUMP_LENS_TWINS'];
const range = (count: number) => Array.from({length: count}, (_, index) => index + 1);
function capture(
  name: string,
  role: GPUComputationCapture['role'],
  values: number[],
  meaning = '',
  extra: Partial<GPUComputationCapture> = {}
): GPUComputationCapture {
  return {
    id: name,
    name,
    role,
    values,
    meaning,
    elementType: 'f32',
    keySpace: 'row',
    provenance: 'gpu-readback',
    ...extra
  };
}
function record(
  captures: GPUComputationCapture[],
  shapes: Record<string, [number, number]> = {},
  size = 16
): LensRecord {
  const base = makeScanRecord();
  return {
    ...base,
    metrics: [],
    choices: [],
    axes: {
      ...base.axes,
      keySpaces: [
        {id: 'row', label: 'row', size},
        ...Object.entries(shapes).map(([id, shape]) => ({
          id,
          label: id,
          size: shape[0] * shape[1],
          shape
        }))
      ]
    },
    captures: {
      schema: 'luma.computation-captures/1',
      runId: base.runId,
      snapshots: captures,
      stamps: [],
      replays: []
    }
  };
}
const points = capture('points', 'input', [0.1, 0.2, 1.8, 1.3], '', {
  components: 2,
  geometry: {kind: 'points', coordinateModel: 'projected', unit: 'm'}
});
const gridRecord = record(
  [points, capture('cells', 'output', [1, 0, 0, 1], 'grid', {keySpace: 'grid'})],
  {grid: [2, 2]}
);
gridRecord.choices = [
  {name: 'gridOriginX', value: 0, unit: 'm', stage: 'input', sweepable: false},
  {name: 'gridOriginY', value: 0, unit: 'm', stage: 'input', sweepable: false},
  {name: 'gridCellSize', value: 1, unit: 'm', stage: 'input', sweepable: false}
];
const FIXTURES: {name: string; record: LensRecord; tokens: string[]}[] = [
  {
    name: 'scan',
    record: record(
      [
        capture('input', 'input', range(16)),
        capture(
          'output',
          'output',
          range(16).map(value => (value * (value + 1)) / 2),
          'prefix',
          {highlight: {key: 9, label: 'prefix'}}
        )
      ],
      {},
      1024
    ),
    tokens: ['prefix', 'cv-mark--highlight', '1008 more']
  },
  {
    name: 'segmented-scan',
    record: record(
      [
        capture('input', 'input', range(8), 'first:8'),
        capture('heads', 'context', [0, 0, 1, 0, 0, 1, 0, 0], 'segments-flags first:8'),
        capture('output', 'output', [2, 3, 1, 2, 3, 1, 2, 3], 'prefix first:8', {
          highlight: {key: 12, label: 'segment prefix'}
        })
      ],
      {},
      32
    ),
    tokens: ['segments', 'data-prefix-start="10"', 'data-key="8"']
  },
  {
    name: 'reduction',
    record: record(
      [capture('input', 'input', [1, 2, 3, 4]), capture('output', 'output', [10])],
      {},
      4
    ),
    tokens: ['fan']
  },
  {
    name: 'sort',
    record: record([
      capture('input', 'input', [8, 2, 5, 3], 'bars'),
      capture('indices', 'context', [1, 3, 2, 500], 'index-gather'),
      capture('output', 'output', [2, 3, 5, 6], 'bars')
    ]),
    tokens: ['bars', 'index-gather', 'data-stub="true"', '>500<']
  },
  {
    name: 'segmented-sort',
    record: record(
      [
        capture('input', 'input', [3, 1, 8, 6], 'bars'),
        capture('segments', 'context', [0, 2, 4], 'segments-offsets'),
        capture('output', 'output', [1, 3, 6, 8], 'bars')
      ],
      {},
      4
    ),
    tokens: ['segments', 'bars']
  },
  {
    name: 'compaction',
    record: record(
      [
        capture('input', 'input', [8, 2, 5, 3]),
        capture('flags', 'context', [1, 0, 1, 0], 'flags'),
        capture('output', 'output', [8, 5], 'compact', {keySpace: 'row'}),
        capture('count', 'output', [2], 'count')
      ],
      {},
      4
    ),
    tokens: ['flags', 'compact']
  },
  {
    name: 'gather',
    record: record(
      [
        capture('input', 'input', [9, 8, 7]),
        capture('indices', 'context', [2, 0], 'index-gather'),
        capture('output', 'output', [7, 9])
      ],
      {},
      3
    ),
    tokens: ['index-gather']
  },
  {
    name: 'scatter',
    record: record(
      [
        capture('input', 'input', [9, 8, 7]),
        capture('indices', 'context', [2, 0, 1], 'index-scatter'),
        capture('output', 'output', [8, 7, 9])
      ],
      {},
      3
    ),
    tokens: ['index-scatter']
  },
  {
    name: 'histogram',
    record: record(
      [
        capture('input', 'input', [1, 2, 4, 5]),
        capture('output', 'output', [2, 2], 'bars edges:0,3,6')
      ],
      {},
      4
    ),
    tokens: ['bars', 'edges', '0–3', '3–6']
  },
  {
    name: 'group-aggregation',
    record: record(
      [
        capture('keys', 'input', range(10), 'column:Key'),
        capture('values', 'input', range(10), 'column:Rain'),
        capture('groups', 'output', [1, 2, 3], 'column:Group'),
        capture('totals', 'output', [4, 7, 11], 'column:Sum')
      ],
      {},
      10
    ),
    tokens: ['cv-array-table--output', 'more rows']
  },
  {
    name: 'grid',
    record: gridRecord,
    tokens: ['data-token="grid"', 'data-token="points"', 'cv-seq-']
  },
  {
    name: 'mat-vec',
    record: record(
      [
        capture('matrix', 'input', range(64), '', {keySpace: 'matrix'}),
        capture('vector', 'input', range(8), 'operator:×', {keySpace: 'vector'}),
        capture('output', 'output', range(8), 'operator:=', {keySpace: 'vector'})
      ],
      {matrix: [8, 8], vector: [1, 8]}
    ),
    tokens: ['>×<', '>=<', 'cv-array-heatmap', 'cv-seq-']
  },
  {
    name: 'convolution',
    record: record(
      [
        capture('input', 'input', range(9), '', {keySpace: 'tile'}),
        capture('weights', 'context', [0, 1, 0, 1, 4, 1, 0, 1, 0], 'stencil operator:×', {
          keySpace: 'tile'
        }),
        capture('output', 'output', range(9), 'operator:=', {keySpace: 'tile'})
      ],
      {tile: [3, 3]}
    ),
    tokens: ['stencil', 'cv-mark--context', 'cv-seq-']
  },
  {
    name: 'fft1d',
    record: record(
      [
        capture('input', 'input', [0, 1, 0, -1], 'signal'),
        capture('output', 'output', [3, 4, 0, 2, 0, 0, 0, -2], 'bars complex-magnitude', {
          components: 2
        })
      ],
      {},
      4
    ),
    tokens: ['signal', 'bars', 'data-provenance="derived"']
  },
  {
    name: 'fft2d',
    record: record(
      [
        capture('input', 'input', range(4096), '', {keySpace: 'tile'}),
        capture(
          'output',
          'output',
          range(4096).flatMap(value => [value, value / 2]),
          'complex-magnitude log',
          {components: 2, keySpace: 'tile'}
        )
      ],
      {tile: [64, 64]}
    ),
    tokens: ['cv-array-image-tile', 'cv-seq-', 'data:image/png;base64,']
  },
  {
    name: 'mask',
    record: record(
      [
        capture('left', 'input', [0, 1, 1], 'flags'),
        capture('right', 'input', [1, 0, 1], 'flags operator:+'),
        capture('output', 'output', [1, 1, 1], 'flags operator:=')
      ],
      {},
      3
    ),
    tokens: ['flags', '>+<', '>=<']
  },
  {
    name: 'flag-offsets',
    record: record(
      [
        capture('flags', 'input', [1, 0, 1, 1], 'flags'),
        capture('output', 'output', [0, 1, 1, 2]),
        capture('count', 'output', [3], 'count')
      ],
      {},
      4
    ),
    tokens: ['flags']
  }
];

function twinFile(value: LensRecord, name: string) {
  return toRecordedLensFile(value, {
    scenarioId: `gpu-core.${name}`,
    recordedAt: '2026-10-14',
    adapterLabel: 'Hand-built fixture (no GPU)'
  });
}

describe('GPU Core array result view', () => {
  for (const fixture of FIXTURES)
    test(`${fixture.name}: selected, deterministic full and thumbnail twins`, () => {
      const file = twinFile(fixture.record, fixture.name);
      const directive = {scenario: file.scenarioId, rung: 'result'};
      const twin = buildLensTwin(file, directive);
      expect(twin.viewId).toBe(GPU_CORE_ARRAY_RESULT_VIEW.id);
      expect(twin.disabled).toBeUndefined();
      expect(validateLensTwin(twin)).toEqual([]);
      expect(twin).toEqual(buildLensTwin(file, directive));
      expect(twin.figure.html).not.toMatch(/<td[^>]*>\s*<\/td>|undefined|\[object Object\]/);
      expect(twin.figure.html).toContain('cv-mark--output');
      expect(twin.figure.html).toContain('cv-mark--input');
      expect((twin.figure.html.match(/data-numbered="true"/g) ?? []).length).toBeLessThanOrEqual(
        16
      );
      for (const token of fixture.tokens) expect(twin.figure.html).toContain(token);
      const thumbnail = buildLensTwin(file, directive, {size: 'thumbnail'});
      expect(validateLensTwin(thumbnail)).toEqual([]);
      expect(thumbnail.viewId).toBe(GPU_CORE_ARRAY_RESULT_VIEW.id);
      expect(thumbnail.figure.width).toBe(160);
      expect(thumbnail.figure.height).toBe(96);
      expect(thumbnail.figure.html).not.toMatch(/<text|<td|cv-legend/);
      expect(thumbnail).toEqual(buildLensTwin(file, directive, {size: 'thumbnail'}));
      if (DUMP_DIRECTORY) {
        mkdirSync(DUMP_DIRECTORY, {recursive: true});
        writeFileSync(`${DUMP_DIRECTORY}/${fixture.name}-result.html`, renderLensTwinHtml(twin));
        writeFileSync(
          `${DUMP_DIRECTORY}/${fixture.name}-thumbnail.html`,
          renderLensTwinHtml(thumbnail)
        );
      }
    });
  test('geometry output is rejected with a specific reason and generic ResultView wins', () => {
    const value = record([
      points,
      capture('output', 'output', [0, 0, 1, 1], '', {
        components: 2,
        geometry: {kind: 'points', coordinateModel: 'projected', unit: 'm'}
      })
    ]);
    expect(GPU_CORE_ARRAY_RESULT_VIEW.satisfies(createViewContext(value))).toMatchObject({
      ok: false,
      reason: expect.stringContaining('geometry')
    });
    const file = twinFile(value, 'geometry');
    expect(buildLensTwin(file, {scenario: file.scenarioId, rung: 'result'}).viewId).toBe(
      'result-view'
    );
  });
  test('missing output disables arrays; geometry inputs require a grid output', () => {
    expect(
      GPU_CORE_ARRAY_RESULT_VIEW.satisfies(
        createViewContext(record([capture('input', 'input', [1])]))
      )
    ).toMatchObject({ok: false, reason: expect.stringContaining('role=output')});
    expect(
      GPU_CORE_ARRAY_RESULT_VIEW.satisfies(
        createViewContext(record([points, capture('output', 'output', [1])]))
      )
    ).toMatchObject({ok: false});
  });
  test('magnitude, logarithm, block means, counts, windows, and unknown grid metadata are explicit', () => {
    const magnitude = buildGPUCoreArrayResultModel(
      createViewContext(FIXTURES.find(fixture => fixture.name === 'fft1d')!.record)
    );
    expect(magnitude.panels[1].values).toEqual([5, 2, 0, 2]);
    const fft = FIXTURES.find(fixture => fixture.name === 'fft2d')!;
    const model = buildGPUCoreArrayResultModel(
      createViewContext(fft.record, {options: {size: 'thumbnail'}})
    );
    expect(model.panels[1].values[0]).toBeCloseTo(Math.log10(1 + Math.hypot(1, 0.5)));
    expect(model.panels[1].blockSize).toBe(1);
    expect(model.notes.join(' ')).toContain('native resolution');
    const count = FIXTURES.find(fixture => fixture.name === 'flag-offsets')!;
    expect(buildGPUCoreArrayResultModel(createViewContext(count.record)).keyNumber).toEqual({
      label: 'count',
      value: 3
    });
    const unknown = buildGPUCoreArrayResultModel(createViewContext({...gridRecord, choices: []}));
    expect(unknown.grid).toBeUndefined();
    expect(unknown.notes.join(' ')).toContain('unknown');
    const long = buildGPUCoreArrayResultModel(
      createViewContext(
        record(
          [capture('input', 'input', range(30)), capture('output', 'output', range(30))],
          {},
          30
        )
      )
    );
    expect(long.panels[0].cells).toEqual([
      ...range(8).map(value => ({index: value - 1, key: value - 1})),
      {gap: '… 14 more …'},
      ...range(8).map(value => ({index: value + 21, key: value + 21}))
    ]);
  });
  test('declared diverging midpoint uses the shared scale and suppresses highlighting', () => {
    const value = record(
      [
        capture('output', 'output', [-2, -1, 1, 2], '', {
          keySpace: 'tile',
          highlight: {key: 0, label: 'answer'}
        })
      ],
      {tile: [2, 2]}
    );
    const kind = {level: 'interval' as const, diverging: true, midpoint: 0};
    const context = createViewContext(value, {
      packs: [{id: 'gpu-core', variableKinds: {output: kind}}]
    });
    const model = buildGPUCoreArrayResultModel(context);
    expect(model.panels[0].palette).toBe('div');
    expect(model.panels[0].scale?.midpoint).toBe(0);
    expect(model.panels[0].scale).toBe(buildGPUCoreArrayResultModel(context).panels[0].scale);
  });
  test('each data mark has provenance and stencil values take the numeric-label budget first', () => {
    for (const fixture of FIXTURES) {
      const file = twinFile(fixture.record, fixture.name);
      const html = buildLensTwin(file, {scenario: file.scenarioId, rung: 'result'}).figure.html;
      for (const tag of html.match(/<[^>]+class="cv-mark [^>]+>/g) ?? [])
        expect(tag).toContain('data-provenance=');
    }
    const stencil = FIXTURES.find(fixture => fixture.name === 'convolution')!;
    const model = buildGPUCoreArrayResultModel(createViewContext(stencil.record));
    expect(model.panels.find(panel => panel.tokens.includes('stencil'))!.numbered).toHaveLength(9);
    const compact = FIXTURES.find(fixture => fixture.name === 'compaction')!;
    const file = twinFile(compact.record, compact.name);
    const twin = buildLensTwin(file, {scenario: file.scenarioId, rung: 'result'});
    expect(twin.figure.html).toContain('data-token="compact"');
    expect(twin.figure.html).not.toContain('data-panel="count"');
    expect(twin.keyNumber).toEqual({label: 'count', value: '2'});
  });
  test('csr-offsets grid cells draw per-cell counts and say so', () => {
    const offsets = record(
      [
        points,
        capture('cellOffsets', 'output', [0, 2, 2, 3, 4], 'grid csr-offsets', {keySpace: 'grid'})
      ],
      {grid: [2, 2]}
    );
    offsets.choices = gridRecord.choices;
    const model = buildGPUCoreArrayResultModel(createViewContext(offsets));
    expect(model.panels[0].values).toEqual([2, 0, 1, 1]);
    expect(model.panels[0].provenance).toBe('derived');
    expect(model.notes.join(' ')).toContain('offsets[c + 1] − offsets[c]');
    expect(model.keyNumber).toEqual({label: 'max cellOffsets', value: 2});
  });
  test('underscores in column labels render as spaces', () => {
    const value = record(
      [
        capture('rain', 'input', range(8), 'column:rainfall_(mm)'),
        capture('total', 'output', range(8), 'column:total_rainfall_(mm)')
      ],
      {},
      8
    );
    const file = twinFile(value, 'underscores');
    const html = buildLensTwin(file, {scenario: file.scenarioId, rung: 'result'}).figure.html;
    expect(html).toContain('rainfall (mm)');
    expect(html).toContain('total rainfall (mm)');
    expect(html).not.toContain('rainfall_(mm)');
  });
  test.each([
    'segments-offsets',
    'segments-flags'
  ])('three segment START ticks respect key spaces and gap widths: %s', meaning => {
    const source = record([
      capture('input', 'input', range(24), '', {keySpace: 'stations'}),
      capture(
        'starts',
        'context',
        meaning === 'segments-offsets'
          ? [0, 4, 20, 24]
          : range(24).map((_, index) => Number([0, 4, 20].includes(index))),
        meaning,
        {keySpace: meaning === 'segments-offsets' ? 'offsets' : 'stations'}
      ),
      capture('output', 'output', range(24), '', {keySpace: 'stations'}),
      capture('totals', 'output', [10, 20, 30], '', {keySpace: 'states'})
    ]);
    const context = createViewContext(source);
    const model = buildGPUCoreArrayResultModel(context);
    const tree = renderGPUCoreArrayResult(model, context);
    for (const panel of model.panels) {
      const group = findViewNodes(
        tree,
        node => node.attributes['data-panel'] === panel.capture.name
      )[0];
      const ticks = findViewNodes(group, node => node.attributes['data-token'] === 'segments');
      if (panel.capture.keySpace === 'states') expect(ticks).toHaveLength(0);
      else {
        expect(ticks).toHaveLength(3);
        expect(ticks.map(tick => tick.attributes['d'])).toEqual(
          [0, 4, 20].map(key => {
            const position = panel.cells.findIndex(cell => 'key' in cell && cell.key === key);
            return `M${getArrayCellBounds(panel, position).left},${panel.y}v${panel.height}`;
          })
        );
      }
    }
  });
  test('pure pack data is available without rendering or devices', () => {
    expect(loadGPUCorePackData().id).toBe('gpu-core');
    expect(Object.keys(loadGPUCorePackData().operators ?? {})).toHaveLength(23);
  });
});
