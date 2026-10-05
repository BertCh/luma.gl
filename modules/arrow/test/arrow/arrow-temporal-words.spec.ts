// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  convertArrowTemporalToGPUVector,
  getArrowTemporalVectorInfo,
  makeArrowTemporalWordGPUVector,
  readArrowGPUVectorAsync,
  TEMPORAL_KIND_METADATA_KEY,
  TEMPORAL_UNIT_METADATA_KEY
} from '@luma.gl/arrow';
import {NullDevice, getWebGPUTestDevice} from '@luma.gl/test-utils';
import {DynamicBuffer} from '@luma.gl/engine';
import type {Device} from '@luma.gl/core';
import * as arrow from 'apache-arrow';

const EPOCH_MS = 1_700_000_000_000n;
const WRAP_VALUES = new BigInt64Array([
  0n,
  1n,
  0xffffffffn,
  0x100000000n,
  0x100000001n,
  -1n,
  -0x100000000n,
  -0x100000001n,
  -86_400_000n * 365n * 30n,
  EPOCH_MS,
  EPOCH_MS + 0xffffffffn,
  2n ** 62n + 12345n,
  -(2n ** 63n),
  2n ** 63n - 1n
]);

function makeInt64Field(kind?: string, unit?: string): arrow.Field {
  const metadata = new Map<string, string>();
  if (kind) metadata.set(TEMPORAL_KIND_METADATA_KEY, kind);
  if (unit) metadata.set(TEMPORAL_UNIT_METADATA_KEY, unit);
  return new arrow.Field('time', new arrow.Int64(), false, metadata);
}

function makeVector<T extends arrow.DataType>(
  type: T,
  values: BigInt64Array | Int32Array
): arrow.Vector<T> {
  // Date(ms) stores two int32 words per row.
  const length = type instanceof arrow.DateMillisecond ? values.length / 2 : values.length;
  return new arrow.Vector([
    arrow.makeData({
      type,
      length,
      data: values
    } as any) as arrow.Data<T>
  ]);
}

async function readWords(device: Device, vector: {data: any[]}): Promise<BigInt64Array[]> {
  const chunks: BigInt64Array[] = [];
  for (const chunk of vector.data) {
    const buffer = chunk.buffer instanceof DynamicBuffer ? chunk.buffer.buffer : chunk.buffer;
    const bytes = await buffer.readAsync(0, chunk.length * 8);
    chunks.push(new BigInt64Array(bytes.slice().buffer));
  }
  void device;
  return chunks;
}

it('getArrowTemporalVectorInfo recognizes Int64 only with temporal metadata', () => {
  const vector = makeVector(new arrow.Int64(), new BigInt64Array([1n, 2n]));
  expect(getArrowTemporalVectorInfo(vector), 'no field').toBeNull();
  expect(getArrowTemporalVectorInfo(vector, makeInt64Field()), 'no metadata').toBeNull();
  const info = getArrowTemporalVectorInfo(vector, makeInt64Field('timestamp', 'millisecond'));
  expect(info?.kind).toBe('timestamp');
  expect(info?.unit).toBe('millisecond');
  expect(info?.bitWidth).toBe(64);
  expect(info?.variableLength).toBe(false);
  const uint = makeVector(new arrow.Uint32(), new Int32Array([1, 2]) as any);
  expect(getArrowTemporalVectorInfo(uint, makeInt64Field('timestamp', 'millisecond'))).toBeNull();
});

it('getArrowTemporalVectorInfo rejects invalid Int64 temporal metadata', () => {
  const vector = makeVector(new arrow.Int64(), new BigInt64Array([1n]));
  expect(() =>
    getArrowTemporalVectorInfo(vector, makeInt64Field('instant', 'millisecond'))
  ).toThrow(/Invalid visgl:temporal-kind/);
  expect(() =>
    getArrowTemporalVectorInfo(vector, makeInt64Field('timestamp', 'fortnight'))
  ).toThrow(/Invalid visgl:temporal-unit/);
  expect(() => getArrowTemporalVectorInfo(vector, makeInt64Field('timestamp'))).toThrow(
    /Invalid visgl:temporal-unit/
  );
  expect(() => getArrowTemporalVectorInfo(vector, makeInt64Field('timestamp', 'day'))).toThrow(
    /only valid for kind "date"/
  );
});

it('getArrowTemporalVectorInfo recognizes Int32 date/time metadata only', () => {
  const vector = makeVector(new arrow.Int32(), new Int32Array([1, 2]));
  const field = (kind: string, unit: string) =>
    new arrow.Field(
      't',
      new arrow.Int32(),
      false,
      new Map([
        [TEMPORAL_KIND_METADATA_KEY, kind],
        [TEMPORAL_UNIT_METADATA_KEY, unit]
      ])
    );
  expect(getArrowTemporalVectorInfo(vector, field('date', 'day'))?.bitWidth).toBe(32);
  expect(() => getArrowTemporalVectorInfo(vector, field('timestamp', 'second'))).toThrow(/Int32/);
});

it('convertArrowTemporalToGPUVector on Int64 + metadata matches the Timestamp equivalent', async () => {
  const device = new NullDevice({});
  const values = new BigInt64Array([1000n, 1005n, 2000n]);
  const int64 = await convertArrowTemporalToGPUVector(
    device,
    makeVector(new arrow.Int64(), values) as any,
    {preferGPU: false, field: makeInt64Field('timestamp', 'millisecond')}
  );
  const timestamp = await convertArrowTemporalToGPUVector(
    device,
    makeVector(new arrow.TimestampMillisecond(), values),
    {preferGPU: false}
  );
  const a = await readArrowGPUVectorAsync(int64.temporal);
  const b = await readArrowGPUVectorAsync(timestamp.temporal);
  expect(Array.from(a.toArray())).toEqual(Array.from(b.toArray()));
  expect(int64.temporalInfo.origin).toBe(1000n);
  expect(int64.field.metadata.get(TEMPORAL_UNIT_METADATA_KEY)).toBe('millisecond');
  int64.destroy();
  timestamp.destroy();
});

it('convertArrowTemporalToGPUVector rejects Int64 without metadata', async () => {
  const device = new NullDevice({});
  await expect(
    convertArrowTemporalToGPUVector(
      device,
      makeVector(new arrow.Int64(), new BigInt64Array([1n])) as any,
      {
        preferGPU: false
      }
    )
  ).rejects.toThrow(/requires Date, Time, Timestamp, Duration, Int64/);
});

it('makeArrowTemporalWordGPUVector round-trips exact words and reports the unit', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const sources: Array<[string, arrow.Vector<any>, arrow.Field | undefined, string]> = [
    [
      'timestamp',
      makeVector(new arrow.TimestampMillisecond(), WRAP_VALUES),
      undefined,
      'millisecond'
    ],
    [
      'duration',
      makeVector(new arrow.DurationMicrosecond(), WRAP_VALUES),
      undefined,
      'microsecond'
    ],
    ['time64', makeVector(new arrow.TimeNanosecond(), WRAP_VALUES), undefined, 'nanosecond'],
    [
      'int64',
      makeVector(new arrow.Int64(), WRAP_VALUES),
      makeInt64Field('timestamp', 'millisecond'),
      'millisecond'
    ]
  ];
  for (const [label, source, field, unit] of sources) {
    const words = makeArrowTemporalWordGPUVector(device, source, {field});
    const [chunk] = await readWords(device, words.vector);
    expect(Array.from(chunk!), label).toEqual(Array.from(WRAP_VALUES));
    expect(words.temporalInfo.unit, label).toBe(unit);
    expect(words.temporalInfo.origin, `${label} origin`).toBeUndefined();
    expect(words.vector.format).toBe('uint32x2');
    expect(words.vector.data[0]!.byteStride).toBe(8);
    expect(words.vector.data[0]!.rowByteLength).toBe(8);
    words.destroy();
  }
});

it('makeArrowTemporalWordGPUVector handles Date(ms) int32 pairs', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const expected = new BigInt64Array([EPOCH_MS, -5n, 0x100000000n]);
  const pairs = new Int32Array(expected.buffer.slice(0));
  const words = makeArrowTemporalWordGPUVector(
    device,
    makeVector(new arrow.DateMillisecond(), pairs)
  );
  const [chunk] = await readWords(device, words.vector);
  expect(Array.from(chunk!)).toEqual(Array.from(expected));
  expect(words.temporalInfo.kind).toBe('date');
  words.destroy();
});

it('makeArrowTemporalWordGPUVector keeps batches and slices exact', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const type = new arrow.TimestampMillisecond();
  const first = arrow.makeData({
    type,
    length: 4,
    data: WRAP_VALUES.slice(0, 4)
  });
  const second = arrow.makeData({
    type,
    length: 5,
    data: WRAP_VALUES.slice(4, 9)
  });
  const multiBatch = new arrow.Vector([first, second]);
  const words = makeArrowTemporalWordGPUVector(device, multiBatch);
  const chunks = await readWords(device, words.vector);
  expect(words.vector.data.length, 'one chunk per batch').toBe(2);
  expect(chunks.map(chunk => Array.from(chunk))).toEqual([
    Array.from(WRAP_VALUES.slice(0, 4)),
    Array.from(WRAP_VALUES.slice(4, 9))
  ]);
  words.destroy();

  const sliced = new arrow.Vector([
    arrow.makeData({type, length: 8, data: WRAP_VALUES.slice(0, 8)})
  ]).slice(2, 7);
  expect(sliced.data[0]!.offset, 'slice keeps a nonzero data offset').toBe(2);
  const slicedWords = makeArrowTemporalWordGPUVector(device, sliced);
  const [slicedChunk] = await readWords(device, slicedWords.vector);
  expect(Array.from(slicedChunk!)).toEqual(Array.from(WRAP_VALUES.slice(2, 7)));
  slicedWords.destroy();
});

it('makeArrowTemporalWordGPUVector uploads a view of the Arrow bytes without copying', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const source = makeVector(new arrow.TimestampMillisecond(), WRAP_VALUES).slice(3, 9);
  const arrowValues = source.data[0]!.values as BigInt64Array;
  const uploads: unknown[] = [];
  const createBuffer = device.createBuffer.bind(device);
  device.createBuffer = ((props: any) => {
    if (props?.data) {
      uploads.push(props.data);
    }
    return createBuffer(props);
  }) as typeof device.createBuffer;
  try {
    const words = makeArrowTemporalWordGPUVector(device, source);
    expect(uploads.length).toBe(1);
    const uploaded = uploads[0] as Uint32Array;
    expect(uploaded.buffer, 'shares the Arrow ArrayBuffer').toBe(arrowValues.buffer);
    expect(uploaded.byteOffset, 'starts at the sliced first row').toBe(arrowValues.byteOffset);
    expect(uploaded.length).toBe(6 * 2);
    expect(arrowValues.byteOffset, 'slice is a nonzero view of the source').toBe(3 * 8);
    words.destroy();
  } finally {
    device.createBuffer = createBuffer;
  }
});

it('makeArrowTemporalWordGPUVector rejects nulls, 32-bit, lists, and non-temporal input', () => {
  const device = new NullDevice({});
  const nullable = new arrow.Vector([
    arrow.makeData({
      type: new arrow.TimestampMillisecond(),
      length: 2,
      nullCount: 1,
      nullBitmap: new Uint8Array([0b10]),
      data: new BigInt64Array([1n, 2n])
    })
  ]);
  expect(() => makeArrowTemporalWordGPUVector(device, nullable)).toThrow(/nullable temporal rows/);
  expect(() =>
    makeArrowTemporalWordGPUVector(device, makeVector(new arrow.DateDay(), new Int32Array([1, 2])))
  ).toThrow(/64-bit source/);
  expect(() =>
    makeArrowTemporalWordGPUVector(
      device,
      makeVector(new arrow.Int64(), new BigInt64Array([1n])) as any
    )
  ).toThrow(/requires Date, Time, Timestamp, Duration, or Int64/);
  const child = arrow.makeData({
    type: new arrow.TimestampMillisecond(),
    length: 2,
    data: new BigInt64Array([1n, 2n])
  });
  const list = new arrow.Vector([
    arrow.makeData({
      type: new arrow.List(new arrow.Field('v', new arrow.TimestampMillisecond(), false)),
      length: 1,
      nullCount: 0,
      nullBitmap: null,
      valueOffsets: new Int32Array([0, 2]),
      child
    })
  ]);
  expect(() => makeArrowTemporalWordGPUVector(device, list as any)).toThrow(
    /does not support List/
  );
});

it('convertArrowTemporalToGPUVector reads sliced scalar Int64 rows', async () => {
  const device = new NullDevice({});
  const sliced = makeVector(
    new arrow.Int64(),
    new BigInt64Array([9n, 10n, 1000n, 1010n, 1025n, 7n])
  ).slice(2, 5);
  const prepared = await convertArrowTemporalToGPUVector(device, sliced as any, {
    preferGPU: false,
    field: makeInt64Field('timestamp', 'millisecond')
  });
  const result = await readArrowGPUVectorAsync(prepared.temporal);
  expect(Array.from(result.toArray())).toEqual([0, 10, 25]);
  expect(prepared.temporalInfo.origin).toBe(1000n);
  prepared.destroy();
});
