// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  compactLensRecordEnvelope,
  expandLensRecordEnvelope
} from '../../src/computation-lenses/record-encoding';

function base64Words(words: number[]): string {
  const bytes = new Uint8Array(words.length * 4);
  const view = new DataView(bytes.buffer);
  words.forEach((word, index) => view.setUint32(index * 4, word >>> 0, true));
  return `b64:${Buffer.from(bytes).toString('base64')}`;
}

/** The parts of a compacted or expanded envelope these specs read. */
type EnvelopeView = {
  encoding: {sourceLines: string[]; strings: string[]};
  record: {
    graphs: unknown[];
    refs: {sources: {textLines?: unknown}[]};
    captures: {snapshots: {values: string}[]; stamps: {values: string}[]};
  };
};

const sentinelTail = [7, 8, 9, ...new Array(40).fill(0xffffffff)];
const envelope = {
  schema: 'luma.lens-record/1',
  scenarioId: 'gpu-core.scan',
  record: {
    graphs: [
      {
        id: 'graph',
        nodes: [
          {id: 'scan-level-0', label: 'scan-level-0', reads: ['~tilde', '~~double'], attrs: {}},
          {id: 'scan-level-1', label: 'scan-level-1', reads: ['~tilde', 'values'], attrs: {}}
        ],
        edges: [{from: 'scan-level-0', to: 'scan-level-1', resource: 'values'}],
        list: [['~:not-a-shape', 1], []]
      }
    ],
    refs: {
      formulas: [],
      sources: [
        {id: 'wgsl:a', text: 'const a = 0u;\nfn main() {\n  body();\n}'},
        {id: 'wgsl:b', text: 'const a = 4u;\nfn main() {\n  body();\n}'},
        {id: 'wgsl:c'}
      ]
    },
    captures: {
      snapshots: [
        {id: 'first', elementType: 'u32', values: base64Words(sentinelTail)},
        {id: 'copy', elementType: 'u32', values: base64Words(sentinelTail)},
        {id: 'short', elementType: 'f32', values: base64Words([1, 2, 3])},
        {id: 'numbers', values: [1, 'NaN']}
      ],
      stamps: [{id: 'stamp', elementType: 'u32', values: base64Words(sentinelTail)}],
      replays: []
    }
  }
};

test('lens record compaction round-trips exactly', () => {
  const compacted = JSON.parse(JSON.stringify(compactLensRecordEnvelope(envelope)));
  expect(expandLensRecordEnvelope(compacted)).toEqual(envelope);
});

test('lens record compaction encodes runs, repeats, lines and graph strings', () => {
  const compacted = compactLensRecordEnvelope(envelope) as unknown as EnvelopeView;
  const [first, copy, short] = compacted.record.captures.snapshots;
  expect(first.values).toMatch(/^rle:[^,]+,\*40:ffffffff$/);
  expect(copy.values).toBe('same:snapshots.0');
  expect(short.values).toBe(envelope.record.captures.snapshots[2].values);
  expect(compacted.record.captures.stamps[0].values).toBe('same:snapshots.0');
  expect(compacted.encoding.sourceLines).toEqual([
    'const a = 0u;',
    'fn main() {',
    '  body();',
    '}',
    'const a = 4u;'
  ]);
  expect(compacted.record.refs.sources[1].textLines).toEqual([4, [1, 3]]);
  expect(compacted.record.refs.sources[2]).toEqual({id: 'wgsl:c'});
  expect(compacted.encoding.strings).toContain('~tilde');
  expect(JSON.stringify(compacted.record.graphs)).not.toContain('scan-level-0');
});

test('undefined graph members stay absent, as in plain JSON', () => {
  const withUndefined = {
    ...envelope,
    record: {...envelope.record, graphs: [{id: 'graph', allocation: undefined, nodes: []}]}
  };
  const json = JSON.stringify(compactLensRecordEnvelope(withUndefined));
  expect(json).not.toContain('null');
  expect(
    (expandLensRecordEnvelope(JSON.parse(json)) as unknown as EnvelopeView).record.graphs
  ).toEqual([{id: 'graph', nodes: []}]);
});

test('lens records without an encoding table are read unchanged', () => {
  const {record, ...rest} = envelope;
  expect(expandLensRecordEnvelope({...rest, record})).toEqual(envelope);
});

test('interned strings inside plain graph objects are restored', () => {
  const expanded = expandLensRecordEnvelope({
    schema: 'luma.lens-record/1',
    encoding: {strings: ['scan-level-0']},
    record: {graphs: [{id: '~0', nodes: [{id: '~0', label: '~~literal'}]}]}
  }) as unknown as EnvelopeView;
  expect(expanded.record.graphs[0]).toEqual({
    id: 'scan-level-0',
    nodes: [{id: 'scan-level-0', label: '~literal'}]
  });
});
