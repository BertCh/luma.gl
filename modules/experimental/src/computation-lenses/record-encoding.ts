// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Lossless on-disk compaction of a JSON-safe lens record envelope (`docs/lens-records/*.json`).
 *
 * The record keeps its schema shape; only these representations change, and
 * {@link expandLensRecordEnvelope} restores the exact JSON-safe envelope:
 * - typed capture `values` (`b64:` text) that equal an earlier capture's are written as
 *   `same:<list>.<index>`, and runs of at least {@link MINIMUM_RUN_WORDS} identical 32-bit words
 *   (unused capacity, sentinels) are written as `rle:` segments;
 * - WGSL source `text` is split into lines stored once per record (`encoding.sourceLines`) and
 *   referenced by `textLines` (a line index, or `[first, count]` for consecutive lines);
 * - strings in `record.graphs` that repeat are interned (`encoding.strings`, written `~<base36>`;
 *   literal strings starting with `~` are escaped as `~~…`), and every graph object is written as
 *   an array of its values headed by its key list (`encoding.shapes`, written `~:<base36>`);
 *   packed arrays of at least {@link MINIMUM_LIST_LENGTH} JSON characters that repeat (bindings,
 *   dispatch shapes, edges) are stored once (`encoding.lists`, written `~^<base36>`).
 */

const BASE64_PREFIX = 'b64:';
const RUN_PREFIX = 'rle:';
const SAME_PREFIX = 'same:';
const MINIMUM_RUN_WORDS = 16;
const CAPTURE_LISTS = ['snapshots', 'stamps', 'replays'] as const;

/** Tables shared by the compacted parts of one envelope. */
export type LensRecordEncoding = {
  /** Interned graph strings; `~<base36 index>` refers to an entry. */
  strings?: string[];
  /** Distinct WGSL source lines referenced by `textLines`. */
  sourceLines?: string[];
  /** Key lists of graph objects; `['~:<base36 index>', ...values]` is one object. */
  shapes?: string[][];
  /** Packed graph arrays that repeat; `~^<base36 index>` stands for an entry. */
  lists?: unknown[][];
};

type JSONObject = Record<string, unknown>;

/** Compacts a JSON-safe envelope (typed values already `b64:` text). Returns a new object. */
export function compactLensRecordEnvelope(envelope: JSONObject): JSONObject {
  const record = {...(envelope['record'] as JSONObject)};
  const encoding: LensRecordEncoding = {};
  compactCaptureValues(record);
  const sourceLines = compactSourceTexts(record);
  if (sourceLines) encoding.sourceLines = sourceLines;
  if (Array.isArray(record['graphs'])) {
    const packed = packGraphs(record['graphs']);
    record['graphs'] = packed.graphs;
    encoding.strings = packed.strings;
    encoding.shapes = packed.shapes;
    encoding.lists = packed.lists;
  }
  // Always present on compacted envelopes: it marks graph strings as escaped or interned.
  return {...envelope, encoding, record};
}

/** Restores the JSON-safe envelope written by {@link compactLensRecordEnvelope}. */
export function expandLensRecordEnvelope(envelope: JSONObject): JSONObject {
  const {encoding, ...rest} = envelope as JSONObject & {encoding?: LensRecordEncoding};
  if (encoding === undefined) {
    return rest;
  }
  if (!isObject(encoding)) throw new Error('Invalid lens record encoding');
  const record = {...(rest['record'] as JSONObject)};
  if (Array.isArray(record['graphs'])) record['graphs'] = unpackGraphs(record['graphs'], encoding);
  if (encoding.sourceLines !== undefined) {
    if (!isStringArray(encoding.sourceLines))
      throw new Error('Invalid lens record encoding sourceLines');
    expandSourceTexts(record, encoding.sourceLines);
  }
  expandCaptureValues(record);
  return {...rest, record};
}

// Captures

function compactCaptureValues(record: JSONObject): void {
  if (!isObject(record['captures'])) return;
  const captures = {...record['captures']};
  const firstByValues = new Map<string, string>();
  for (const list of CAPTURE_LISTS) {
    if (!Array.isArray(captures[list])) continue;
    captures[list] = (captures[list] as unknown[]).map((capture, index) => {
      if (!isObject(capture) || typeof capture['values'] !== 'string') return capture;
      const values = capture['values'];
      if (!values.startsWith(BASE64_PREFIX)) return capture;
      const first = firstByValues.get(values);
      if (first !== undefined) return {...capture, values: `${SAME_PREFIX}${first}`};
      firstByValues.set(values, `${list}.${index}`);
      return {...capture, values: compactRuns(values)};
    });
  }
  record['captures'] = captures;
}

function expandCaptureValues(record: JSONObject): void {
  if (!isObject(record['captures'])) return;
  const captures = {...record['captures']};
  for (const list of CAPTURE_LISTS) {
    if (!Array.isArray(captures[list])) continue;
    captures[list] = (captures[list] as unknown[]).map(capture =>
      isObject(capture) &&
      typeof capture['values'] === 'string' &&
      capture['values'].startsWith(RUN_PREFIX)
        ? {...capture, values: expandRuns(capture['values'])}
        : capture
    );
  }
  // References point at earlier captures, which are already expanded.
  for (const list of CAPTURE_LISTS) {
    if (!Array.isArray(captures[list])) continue;
    captures[list] = (captures[list] as unknown[]).map(capture => {
      if (
        !isObject(capture) ||
        typeof capture['values'] !== 'string' ||
        !capture['values'].startsWith(SAME_PREFIX)
      )
        return capture;
      const [targetList, targetIndex] = capture['values'].slice(SAME_PREFIX.length).split('.');
      const target = (captures[targetList] as unknown[] | undefined)?.[Number(targetIndex)];
      if (!isObject(target) || typeof target['values'] !== 'string')
        throw new Error(`Invalid capture reference: ${capture['values']}`);
      return {...capture, values: target['values']};
    });
  }
  record['captures'] = captures;
}

/** Splits `b64:` words into base64 segments and `*<count>:<hex word>` runs. */
function compactRuns(values: string): string {
  const bytes = decodeBase64(values.slice(BASE64_PREFIX.length));
  if (bytes.length % 4 !== 0) return values;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const wordCount = bytes.length / 4;
  const segments: string[] = [];
  let literalStart = 0;
  let index = 0;
  while (index < wordCount) {
    const word = view.getUint32(index * 4, true);
    let end = index + 1;
    while (end < wordCount && view.getUint32(end * 4, true) === word) end++;
    if (end - index >= MINIMUM_RUN_WORDS) {
      if (literalStart < index)
        segments.push(encodeBase64(bytes.subarray(literalStart * 4, index * 4)));
      segments.push(`*${end - index}:${word.toString(16).padStart(8, '0')}`);
      literalStart = end;
    }
    index = end;
  }
  if (!segments.length) return values;
  if (literalStart < wordCount) segments.push(encodeBase64(bytes.subarray(literalStart * 4)));
  return `${RUN_PREFIX}${segments.join(',')}`;
}

function expandRuns(values: string): string {
  const parts: Uint8Array[] = [];
  for (const segment of values.slice(RUN_PREFIX.length).split(',')) {
    const run = /^\*(\d+):([0-9a-f]{8})$/.exec(segment);
    if (!run) {
      parts.push(decodeBase64(segment));
      continue;
    }
    const count = Number(run[1]);
    const word = parseInt(run[2], 16);
    const bytes = new Uint8Array(count * 4);
    const view = new DataView(bytes.buffer);
    for (let index = 0; index < count; index++) view.setUint32(index * 4, word, true);
    parts.push(bytes);
  }
  const bytes = new Uint8Array(parts.reduce((total, part) => total + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    bytes.set(part, offset);
    offset += part.length;
  }
  return `${BASE64_PREFIX}${encodeBase64(bytes)}`;
}

// Sources

function compactSourceTexts(record: JSONObject): string[] | undefined {
  const refs = record['refs'];
  if (!isObject(refs) || !Array.isArray(refs['sources'])) return undefined;
  const lines = new Map<string, number>();
  const sources = refs['sources'].map(source => {
    if (!isObject(source) || typeof source['text'] !== 'string') return source;
    const {text, ...rest} = source;
    const indices = (text as string).split('\n').map(line => {
      let index = lines.get(line);
      if (index === undefined) {
        index = lines.size;
        lines.set(line, index);
      }
      return index;
    });
    return {...rest, textLines: toRuns(indices)};
  });
  if (!lines.size) return undefined;
  record['refs'] = {...refs, sources};
  return [...lines.keys()];
}

function expandSourceTexts(record: JSONObject, lines: readonly string[]): void {
  const refs = record['refs'];
  if (!isObject(refs) || !Array.isArray(refs['sources'])) return;
  record['refs'] = {
    ...refs,
    sources: refs['sources'].map(source => {
      if (!isObject(source) || !Array.isArray(source['textLines'])) return source;
      const {textLines, ...rest} = source;
      const text = fromRuns(textLines as unknown[])
        .map(index => {
          const line = lines[index];
          if (line === undefined) throw new Error('Invalid source line reference');
          return line;
        })
        .join('\n');
      return {...rest, text};
    })
  };
}

/** Writes consecutive indices as `[first, count]`. */
function toRuns(indices: readonly number[]): (number | [number, number])[] {
  const runs: (number | [number, number])[] = [];
  let index = 0;
  while (index < indices.length) {
    let end = index + 1;
    while (end < indices.length && indices[end] === indices[end - 1] + 1) end++;
    runs.push(end - index > 1 ? [indices[index], end - index] : indices[index]);
    index = end;
  }
  return runs;
}

function fromRuns(runs: readonly unknown[]): number[] {
  const indices: number[] = [];
  for (const run of runs) {
    if (typeof run === 'number') indices.push(run);
    else if (Array.isArray(run) && typeof run[0] === 'number' && typeof run[1] === 'number')
      for (let offset = 0; offset < run[1]; offset++) indices.push(run[0] + offset);
    else throw new Error('Invalid source line run');
  }
  return indices;
}

// Graphs

const SHAPE_PREFIX = '~:';
const LIST_PREFIX = '~^';
const MINIMUM_LIST_LENGTH = 10;

function packGraphs(graphs: unknown[]): {
  graphs: unknown;
  strings: string[];
  shapes: string[][];
  lists: unknown[][];
} {
  const strings = getInternedStrings(graphs);
  const indices = new Map(strings.map((text, index) => [text, index]));
  const shapes: string[][] = [];
  const shapeIndices = new Map<string, number>();
  const pack = (value: unknown): unknown => {
    if (typeof value === 'string') {
      const index = indices.get(value);
      if (index !== undefined) return `~${index.toString(36)}`;
      return value.startsWith('~') ? `~${value}` : value;
    }
    if (Array.isArray(value)) return value.map(pack);
    if (!isObject(value)) return value;
    // JSON omits undefined members; a positional array would turn them into null.
    const keys = Object.keys(value).filter(key => value[key] !== undefined);
    const shapeKey = JSON.stringify(keys);
    let shape = shapeIndices.get(shapeKey);
    if (shape === undefined) {
      shape = shapes.length;
      shapes.push(keys);
      shapeIndices.set(shapeKey, shape);
    }
    return [`${SHAPE_PREFIX}${shape.toString(36)}`, ...keys.map(key => pack(value[key]))];
  };
  const packed = graphs.map(pack);
  // Repeated packed arrays (below the graph list) are stored once; entries stay unsubstituted.
  const counts = new Map<string, number>();
  const count = (value: unknown): void => {
    if (!Array.isArray(value)) return;
    value.forEach(count);
    const key = JSON.stringify(value);
    if (key.length >= MINIMUM_LIST_LENGTH) counts.set(key, (counts.get(key) ?? 0) + 1);
  };
  packed.forEach(graph => (Array.isArray(graph) ? graph.forEach(count) : undefined));
  const listKeys = [...counts]
    .filter(([, occurrences]) => occurrences > 1)
    .sort((first, second) => second[1] - first[1] || (first[0] < second[0] ? -1 : 1))
    .map(([key]) => key);
  const listIndices = new Map(listKeys.map((key, index) => [key, index]));
  const substitute = (value: unknown): unknown => {
    if (!Array.isArray(value)) return value;
    const index = listIndices.get(JSON.stringify(value));
    return index !== undefined ? `${LIST_PREFIX}${index.toString(36)}` : value.map(substitute);
  };
  return {
    graphs: packed.map(graph => (Array.isArray(graph) ? graph.map(substitute) : graph)),
    strings,
    shapes,
    lists: listKeys.map(key => JSON.parse(key) as unknown[])
  };
}

function unpackGraphs(graphs: unknown[], encoding: LensRecordEncoding): unknown[] {
  const {strings = [], shapes = []} = encoding;
  if (!isStringArray(strings)) throw new Error('Invalid lens record encoding strings');
  if (!Array.isArray(shapes) || !shapes.every(isStringArray))
    throw new Error('Invalid lens record encoding shapes');
  const {lists = []} = encoding;
  if (!Array.isArray(lists) || !lists.every(Array.isArray))
    throw new Error('Invalid lens record encoding lists');
  const unpack = (value: unknown): unknown => {
    if (typeof value === 'string') {
      if (!value.startsWith('~')) return value;
      if (value.startsWith(LIST_PREFIX)) {
        const list = lists[parseInt(value.slice(LIST_PREFIX.length), 36)];
        if (list === undefined) throw new Error(`Invalid repeated list: ${value}`);
        return unpack(list);
      }
      if (value.startsWith('~~')) return value.slice(1);
      const text = strings[parseInt(value.slice(1), 36)];
      if (text === undefined) throw new Error(`Invalid interned string: ${value}`);
      return text;
    }
    if (isObject(value))
      return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, unpack(item)]));
    if (!Array.isArray(value)) return value;
    const [head, ...values] = value;
    if (typeof head !== 'string' || !head.startsWith(SHAPE_PREFIX)) return value.map(unpack);
    const keys = shapes[parseInt(head.slice(SHAPE_PREFIX.length), 36)];
    if (!keys || keys.length !== values.length) throw new Error(`Invalid graph shape: ${head}`);
    return Object.fromEntries(keys.map((key, index) => [key, unpack(values[index])]));
  };
  return graphs.map(unpack);
}

/** Strings that occur at least twice and are longer than their token, most frequent first. */
function getInternedStrings(graphs: unknown): string[] {
  const counts = new Map<string, number>();
  const count = (value: unknown): void => {
    if (typeof value === 'string') counts.set(value, (counts.get(value) ?? 0) + 1);
    else if (Array.isArray(value)) value.forEach(count);
    else if (isObject(value)) Object.values(value).forEach(count);
  };
  count(graphs);
  return [...counts]
    .filter(([text, occurrences]) => occurrences > 1 && text.length > 3)
    .sort((first, second) => second[1] - first[1] || (first[0] < second[0] ? -1 : 1))
    .map(([text]) => text);
}

// Utilities

function isObject(value: unknown): value is JSONObject {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every(item => typeof item === 'string');
}

function decodeBase64(text: string): Uint8Array {
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index++) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

function encodeBase64(bytes: Uint8Array): string {
  let binary = '';
  // Chunked to stay under argument-count limits of String.fromCharCode.
  for (let offset = 0; offset < bytes.length; offset += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
  }
  return btoa(binary);
}
