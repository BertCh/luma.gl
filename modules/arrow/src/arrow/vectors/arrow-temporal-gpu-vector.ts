// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Binding, type Device, type ShaderLayout} from '@luma.gl/core';
import {Computation, DynamicBuffer} from '@luma.gl/engine';
import {GPUData, GPUVector} from '@luma.gl/gpgpu/gpu-data';
import {
  Data,
  DataType,
  DateUnit,
  Date_,
  Duration,
  Field,
  Float32,
  Int32,
  Int64,
  List,
  Time,
  TimeUnit,
  Timestamp,
  Vector,
  makeData
} from 'apache-arrow';
import {makeGPUVectorFromArrow} from '../gpu/arrow-gpu-table-adapters';
import {getRequiredArrowGPUVectorDataType} from '../gpu/arrow-gpu-data';

/** Supported Arrow temporal logical kinds. */
export type ArrowTemporalKind = 'date' | 'time' | 'timestamp' | 'duration';
/** Supported Arrow temporal source units. */
export type ArrowTemporalUnit = 'day' | 'second' | 'millisecond' | 'microsecond' | 'nanosecond';
/** Temporal origin selection policy retained in output metadata. */
export type ArrowTemporalOriginPolicy = 'first-valid' | 'zero';
/**
 * Supported non-interval Arrow temporal leaf types.
 *
 * Plain `Int64` and `Int32` columns count as temporal only when their field carries
 * `visgl:temporal-kind` and `visgl:temporal-unit` metadata (see {@link TEMPORAL_KIND_METADATA_KEY}).
 */
export type ArrowTemporalType = Date_ | Time | Timestamp | Duration | Int64 | Int32;
/** Supported scalar or variable-length Arrow temporal columns. */
export type ArrowTemporalColumnType = ArrowTemporalType | List<ArrowTemporalType>;
/** Prepared relative temporal output types. */
export type ArrowRelativeTemporalType = Float32 | List<Float32>;
type RelativeTemporalFormat = 'float32' | 'vertex-list<float32>';

/** Arrow field metadata key for the prepared temporal logical kind. */
export const TEMPORAL_KIND_METADATA_KEY = 'visgl:temporal-kind';
/** Arrow field metadata key for the prepared temporal source unit. */
export const TEMPORAL_UNIT_METADATA_KEY = 'visgl:temporal-unit';
/** Arrow field metadata key for the prepared temporal source origin. */
export const TEMPORAL_ORIGIN_METADATA_KEY = 'visgl:temporal-origin';
/** Arrow field metadata key for the prepared temporal origin policy. */
export const TEMPORAL_ORIGIN_POLICY_METADATA_KEY = 'visgl:temporal-origin-policy';
/** Arrow field metadata key for the prepared timestamp timezone, when present. */
export const TEMPORAL_TIMEZONE_METADATA_KEY = 'visgl:temporal-timezone';

/** Metadata recovered from one supported Arrow temporal column. */
export type ArrowTemporalVectorInfo = {
  /** Logical Arrow temporal kind. */
  kind: ArrowTemporalKind;
  /** Source Arrow temporal unit retained by prepared values. */
  unit: ArrowTemporalUnit;
  /** Whether the source/output rows are variable-length list leaves. */
  variableLength: boolean;
  /** Physical source Arrow temporal scalar width. */
  bitWidth: 32 | 64;
  /** Timestamp timezone when the source Arrow type carries one. */
  timezone?: string | null;
  /** Persisted or selected relative temporal origin in source units. */
  origin?: number | bigint;
  /** Persisted or selected origin policy. */
  originPolicy?: ArrowTemporalOriginPolicy;
};

/** Options used when converting one Arrow temporal column for GPU consumption. */
export type ConvertArrowTemporalToGPUVectorOptions = {
  /** Stable prepared GPU vector name. Defaults to `temporal`. */
  name?: string;
  /** Stable resource id prefix. Defaults to the vector name. */
  id?: string;
  /** Optional Arrow field carrying previously persisted temporal metadata. */
  field?: Field;
  /** Override or seed the relative-time origin in source units. */
  origin?: number | bigint | string;
  /** Prefer WebGPU compute when available. Defaults to `true` on WebGPU devices. */
  preferGPU?: boolean;
};

/** Options used when converting several named Arrow temporal columns together. */
export type ConvertArrowTemporalToGPUVectorsOptions = {
  /** Per-column conversion options keyed by source column name. */
  columns?: Record<string, Omit<ConvertArrowTemporalToGPUVectorOptions, 'name'>>;
};

/** Prepared relative Float32 temporal GPU vector plus persisted metadata. */
export type PreparedArrowTemporalGPUVector<
  Format extends RelativeTemporalFormat = RelativeTemporalFormat
> = {
  /** Prepared relative Float32 GPU vector. */
  temporal: GPUVector<Format>;
  /** Alias for callers that prefer the generic vector name. */
  vector: GPUVector<Format>;
  /** Output Arrow field carrying prepared temporal metadata. */
  field: Field;
  /** Recovered source temporal metadata and chosen origin. */
  temporalInfo: ArrowTemporalVectorInfo & {
    origin: number | bigint;
    originPolicy: ArrowTemporalOriginPolicy;
  };
  /** Releases owned GPU resources. */
  destroy: () => void;
};

type ArrowTemporalSource = Vector<ArrowTemporalColumnType> | GPUVector;
type PreparedArrowTemporalFormatForSource<Source> =
  Source extends Vector<List<ArrowTemporalType>>
    ? 'vertex-list<float32>'
    : Source extends Vector<ArrowTemporalType>
      ? 'float32'
      : RelativeTemporalFormat;
type PreparedArrowTemporalGPUVectorMap<SourceVectors extends Record<string, ArrowTemporalSource>> =
  {
    [Name in keyof SourceVectors]: PreparedArrowTemporalGPUVector<
      PreparedArrowTemporalFormatForSource<SourceVectors[Name]>
    >;
  };

type TemporalListGPUReadbackMetadata = {
  kind: 'temporal-list';
  valueOffsets: Int32Array;
  valueByteLength: number;
};

const makeFloat32Data = makeData as (props: {
  type: Float32;
  length: number;
  data: Float32Array;
}) => Data<Float32>;

const makeFloat32ListData = makeData as (props: {
  type: List<Float32>;
  length: number;
  nullCount: number;
  nullBitmap: null;
  valueOffsets: Int32Array;
  child: Data<Float32>;
}) => Data<List<Float32>>;

const TEMPORAL_CONVERSION_SHADER_LAYOUT: ShaderLayout = {
  bindings: [
    {
      name: 'sourceTemporalValues',
      type: 'read-only-storage',
      group: 0,
      location: 0
    },
    {
      name: 'temporalConversionConfig',
      type: 'read-only-storage',
      group: 0,
      location: 1
    },
    {name: 'preparedTemporalValues', type: 'storage', group: 0, location: 2}
  ],
  attributes: []
};

/**
 * Recover supported temporal metadata from one Arrow scalar or list column.
 *
 * True Arrow temporal types (Date, Time, Timestamp, Duration) are always recognized. A plain
 * `Int64` (or `Int32`) column is recognized only when its field, or its list child field, carries
 * `visgl:temporal-kind` and `visgl:temporal-unit` metadata; otherwise this returns `null`.
 *
 * @param vector - Arrow vector or adapter-backed GPU vector.
 * @param field - Optional Arrow field of a scalar column; list columns use their child field.
 * @returns Temporal info, or `null` when the column is not temporal.
 * @throws If integer metadata is present but has an invalid kind, unit, or bit width.
 */
export function getArrowTemporalVectorInfo(
  vector: Pick<Vector, 'type'> | Pick<GPUVector, 'dataType'>,
  field?: Field
): ArrowTemporalVectorInfo | null {
  const type = getArrowTemporalSourceType(vector);
  const leafType = getArrowTemporalLeafType(type);
  const leafField = getArrowTemporalLeafField(type, field);
  const metadata = leafField?.metadata;
  const leafInfo = leafType ? getArrowTemporalLeafInfo(leafType, metadata) : null;
  if (!leafInfo) {
    return null;
  }
  return {
    ...leafInfo,
    variableLength: DataType.isList(type),
    ...(metadata?.has(TEMPORAL_ORIGIN_METADATA_KEY)
      ? {
          origin: parseTemporalOrigin(
            metadata.get(TEMPORAL_ORIGIN_METADATA_KEY)!,
            leafInfo.bitWidth
          )
        }
      : {}),
    ...(metadata?.has(TEMPORAL_ORIGIN_POLICY_METADATA_KEY)
      ? {
          originPolicy: metadata.get(
            TEMPORAL_ORIGIN_POLICY_METADATA_KEY
          ) as ArrowTemporalOriginPolicy
        }
      : {})
  };
}

/** Convert one scalar or variable-length Arrow temporal column to relative Float32 GPU values. */
export function convertArrowTemporalToGPUVector(
  device: Device,
  source: Vector<List<ArrowTemporalType>>,
  options?: ConvertArrowTemporalToGPUVectorOptions
): Promise<PreparedArrowTemporalGPUVector<'vertex-list<float32>'>>;
export function convertArrowTemporalToGPUVector(
  device: Device,
  source: Vector<ArrowTemporalType>,
  options?: ConvertArrowTemporalToGPUVectorOptions
): Promise<PreparedArrowTemporalGPUVector<'float32'>>;
export function convertArrowTemporalToGPUVector(
  device: Device,
  source: Vector<ArrowTemporalColumnType>,
  options?: ConvertArrowTemporalToGPUVectorOptions
): Promise<PreparedArrowTemporalGPUVector>;
export function convertArrowTemporalToGPUVector(
  device: Device,
  source: GPUVector,
  options?: ConvertArrowTemporalToGPUVectorOptions
): Promise<PreparedArrowTemporalGPUVector>;
export async function convertArrowTemporalToGPUVector(
  device: Device,
  source: ArrowTemporalSource,
  options: ConvertArrowTemporalToGPUVectorOptions = {}
): Promise<PreparedArrowTemporalGPUVector> {
  const sourceInfo = getRequiredArrowTemporalVectorInfo(source, options.field);
  const origin = resolveTemporalOrigin(source, sourceInfo, options);
  const temporalInfo = {
    ...sourceInfo,
    origin,
    originPolicy: sourceInfo.kind === 'duration' ? 'zero' : 'first-valid'
  } satisfies PreparedArrowTemporalGPUVector['temporalInfo'];
  const field = makePreparedArrowTemporalField(source, temporalInfo, options.field);
  const name = options.name || 'temporal';
  const id = options.id || name;
  const preferGPU = options.preferGPU ?? device.type === 'webgpu';

  if (preferGPU && device.type === 'webgpu') {
    const sourceVector =
      source instanceof GPUVector
        ? source
        : makeArrowTemporalSourceGPUVector(device, source, {
            name: `${name}-source`,
            id
          });
    const ownsSourceVector = !(source instanceof GPUVector);
    try {
      return await convertArrowTemporalToGPUVectorOnGPU(device, sourceVector, temporalInfo, field, {
        name,
        id
      });
    } finally {
      if (ownsSourceVector) {
        sourceVector.destroy();
      }
    }
  }

  if (source instanceof GPUVector) {
    throw new Error(
      'convertArrowTemporalToGPUVector requires WebGPU for GPU-resident temporal input'
    );
  }

  const preparedVector = makePreparedArrowTemporalVector(source, temporalInfo, field);
  const temporal = makeGPUVectorFromArrow(device, preparedVector, {
    name,
    id,
    format: getPreparedArrowTemporalFormat(temporalInfo)
  });
  return createPreparedArrowTemporalGPUVector(temporal, field, temporalInfo, true);
}

/** Convert several named temporal columns with the same scalar/list normalization rules. */
export async function convertArrowTemporalToGPUVectors<
  SourceVectors extends Record<string, ArrowTemporalSource>
>(
  device: Device,
  sourceVectors: SourceVectors,
  options: ConvertArrowTemporalToGPUVectorsOptions = {}
): Promise<PreparedArrowTemporalGPUVectorMap<SourceVectors>> {
  assertArrowTemporalVectorAlignment(sourceVectors);
  const entries = await Promise.all(
    Object.entries(sourceVectors).map(async ([name, source]) => {
      const columnOptions = {name, ...(options.columns?.[name] || {})};
      const prepared =
        source instanceof GPUVector
          ? await convertArrowTemporalToGPUVector(device, source, columnOptions)
          : await convertArrowTemporalToGPUVector(device, source, columnOptions);
      return [name, prepared] as const;
    })
  );
  return Object.fromEntries(entries) as unknown as PreparedArrowTemporalGPUVectorMap<SourceVectors>;
}

/** Options for {@link makeArrowTemporalWordGPUVector}. */
export type MakeArrowTemporalWordGPUVectorOptions = {
  /** Stable GPU vector name. Defaults to `temporal-words`. */
  name?: string;
  /** Stable resource id prefix. Defaults to the vector name. */
  id?: string;
  /** Optional Arrow field; required to recognize a plain `Int64` column via `visgl:temporal-*` metadata. */
  field?: Field;
};

/** Exact Int64 temporal words as a `uint32x2` GPU vector, plus the source temporal metadata. */
export type ArrowTemporalWordGPUVector = {
  /**
   * One `uint32x2` row `(low, high)` per source row: the little-endian words of the signed 64-bit
   * source value, in source units, with no origin subtracted. One chunk per Arrow batch.
   */
  vector: GPUVector<'uint32x2'>;
  /**
   * Source temporal metadata. `unit` is the unit of the values (for example `millisecond`), so
   * callers scale windows and playheads to it. No `origin` is set: values are absolute.
   */
  temporalInfo: ArrowTemporalVectorInfo;
  /** The Arrow field passed in options, when any. */
  field?: Field;
  /** Releases the uploaded buffers. */
  destroy: () => void;
};

const IS_LITTLE_ENDIAN = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;

/**
 * Upload a scalar 64-bit Arrow temporal column as exact `uint32x2` words, with no CPU per-row pass.
 *
 * Supports `Timestamp`, `Date` (millisecond), `Duration`, `Time64`, and plain `Int64` columns whose
 * field carries `visgl:temporal-kind` and `visgl:temporal-unit` metadata. Each Arrow batch is
 * uploaded from a `Uint32Array` view over the batch's own values bytes (a sliced vector uploads only its rows),
 * so a sliced vector uploads only its rows and no values are copied or converted on the CPU. Batch
 * boundaries are preserved as GPUData chunks, which `graph.importGPUVector` turns into a vector
 * view. A GPU kernel then subtracts a playhead exactly with a borrow-subtract.
 *
 * Use {@link convertArrowTemporalToGPUVector} instead for relative float32 values or 32-bit
 * sources.
 *
 * @param device - Device that owns the uploaded buffers.
 * @param source - Scalar 64-bit temporal Arrow vector.
 * @param options - Name, id prefix, and optional Arrow field.
 * @returns The word vector, source temporal info, and a `destroy` function.
 * @throws If the source is not temporal, is a list, is 32-bit, has nulls, or the platform is
 * big-endian.
 */
export function makeArrowTemporalWordGPUVector(
  device: Device,
  source: Vector<ArrowTemporalType>,
  options: MakeArrowTemporalWordGPUVectorOptions = {}
): ArrowTemporalWordGPUVector {
  if (!IS_LITTLE_ENDIAN) {
    throw new Error('makeArrowTemporalWordGPUVector requires a little-endian platform');
  }
  const temporalInfo = getArrowTemporalVectorInfo(source, options.field);
  if (!temporalInfo) {
    throw new Error(
      'makeArrowTemporalWordGPUVector requires Date, Time, Timestamp, Duration, or Int64 with visgl:temporal-* metadata'
    );
  }
  if (temporalInfo.variableLength) {
    throw new Error('makeArrowTemporalWordGPUVector does not support List temporal columns');
  }
  if (temporalInfo.bitWidth !== 64) {
    throw new Error(
      'makeArrowTemporalWordGPUVector requires a 64-bit source; use convertArrowTemporalToGPUVector for 32-bit temporal values'
    );
  }
  const name = options.name || 'temporal-words';
  const id = options.id || name;
  const bytesPerRow = BigInt64Array.BYTES_PER_ELEMENT;

  const data: GPUData<'uint32x2'>[] = [];
  try {
    for (const [chunkIndex, sourceData] of source.data.entries()) {
      validateArrowTemporalData(sourceData as Data<ArrowTemporalColumnType>);
      const values = sourceData.values as ArrayBufferView | undefined;
      // A view over the Arrow bytes: no element loop, no copy. Arrow's `Data.slice` already
      // advances `values` to the first row, so `data.offset` must not be applied again.
      // Date(ms) stores two int32 per row, hence the Uint32Array over the raw bytes.
      const words = values
        ? new Uint32Array(values.buffer, values.byteOffset, sourceData.length * 2)
        : undefined;
      const buffer = new DynamicBuffer(device, {
        id: `${id}-temporal-words-${chunkIndex}`,
        usage: Buffer.VERTEX | Buffer.STORAGE | Buffer.COPY_DST | Buffer.COPY_SRC,
        ...(words && words.length > 0 ? {data: words} : {byteLength: bytesPerRow})
      });
      data.push(
        new GPUData({
          buffer,
          dataType: sourceData.type,
          format: 'uint32x2',
          length: sourceData.length,
          byteStride: bytesPerRow,
          rowByteLength: bytesPerRow,
          ownsBuffer: true
        })
      );
    }
  } catch (error) {
    for (const chunk of data) {
      chunk.destroy();
    }
    throw error;
  }

  const vector = new GPUVector({
    type: 'data',
    name,
    dataType: source.type,
    format: 'uint32x2',
    data,
    byteStride: bytesPerRow,
    rowByteLength: bytesPerRow,
    ownsData: true
  });
  let destroyed = false;
  return {
    vector,
    temporalInfo,
    ...(options.field ? {field: options.field} : {}),
    destroy: () => {
      if (!destroyed) {
        destroyed = true;
        vector.destroy();
      }
    }
  };
}

function assertArrowTemporalVectorAlignment(
  sourceVectors: Record<string, ArrowTemporalSource>
): void {
  const entries = Object.entries(sourceVectors);
  const [referenceName, referenceVector] = entries[0] || [];
  if (!referenceName || !referenceVector) {
    return;
  }
  for (const [name, vector] of entries.slice(1)) {
    if (vector.length !== referenceVector.length) {
      throw new Error(
        `convertArrowTemporalToGPUVectors ${name} rows must match ${referenceName} rows (${vector.length} !== ${referenceVector.length})`
      );
    }
    if (vector.data.length !== referenceVector.data.length) {
      throw new Error(
        `convertArrowTemporalToGPUVectors ${name} batch count must match ${referenceName} batch count`
      );
    }
  }
}

function createPreparedArrowTemporalGPUVector<Format extends RelativeTemporalFormat>(
  temporal: GPUVector<Format>,
  field: Field,
  temporalInfo: PreparedArrowTemporalGPUVector['temporalInfo'],
  ownsTemporal: boolean
): PreparedArrowTemporalGPUVector<Format> {
  let destroyed = false;
  return {
    temporal,
    vector: temporal,
    field,
    temporalInfo,
    destroy: () => {
      if (!ownsTemporal || destroyed) {
        return;
      }
      destroyed = true;
      temporal.destroy();
    }
  };
}

function getPreparedArrowTemporalFormat(
  temporalInfo: Pick<ArrowTemporalVectorInfo, 'variableLength'>
): RelativeTemporalFormat {
  return temporalInfo.variableLength ? 'vertex-list<float32>' : 'float32';
}

async function convertArrowTemporalToGPUVectorOnGPU(
  device: Device,
  source: GPUVector,
  temporalInfo: PreparedArrowTemporalGPUVector['temporalInfo'],
  field: Field,
  options: Required<Pick<ConvertArrowTemporalToGPUVectorOptions, 'name' | 'id'>>
): Promise<PreparedArrowTemporalGPUVector> {
  const outputType = getPreparedArrowTemporalType(field);
  const outputFormat = getPreparedArrowTemporalFormat(temporalInfo);
  const sourceType = getArrowTemporalSourceType(source);
  const outputData: GPUData<typeof outputFormat>[] = [];
  const transientResources: Array<{destroy: () => void}> = [];

  for (const [chunkIndex, sourceData] of source.data.entries()) {
    const temporalValueOffsets = getTemporalValueOffsets(sourceType, sourceData);
    const scalarCount = temporalValueOffsets
      ? (temporalValueOffsets[temporalValueOffsets.length - 1] ?? 0)
      : sourceData.length;
    const configBuffer = device.createBuffer({
      id: `${options.id}-temporal-config-${chunkIndex}`,
      usage: Buffer.STORAGE | Buffer.COPY_DST | Buffer.COPY_SRC,
      data: makeTemporalConversionConfig(temporalInfo, scalarCount)
    });
    const outputBuffer = new DynamicBuffer(device, {
      id: `${options.id}-temporal-values-${chunkIndex}`,
      usage: Buffer.VERTEX | Buffer.STORAGE | Buffer.COPY_DST | Buffer.COPY_SRC,
      byteLength: Math.max(
        Float32Array.BYTES_PER_ELEMENT,
        scalarCount * Float32Array.BYTES_PER_ELEMENT
      )
    });
    dispatchArrowTemporalConversion(device, temporalInfo, {
      id: options.id,
      chunkIndex,
      scalarCount,
      sourceTemporalValues: getGPUDataBinding(sourceData, getTemporalSourceByteLength(sourceData)),
      temporalConversionConfig: configBuffer,
      preparedTemporalValues: outputBuffer
    });
    outputData.push(
      new GPUData({
        buffer: outputBuffer,
        dataType: outputType,
        format: outputFormat,
        length: sourceData.length,
        stride: 1,
        byteStride: Float32Array.BYTES_PER_ELEMENT,
        rowByteLength: Float32Array.BYTES_PER_ELEMENT,
        ownsBuffer: true,
        ...(temporalValueOffsets
          ? {
              valueOffsets: temporalValueOffsets,
              valueByteLength: scalarCount * Float32Array.BYTES_PER_ELEMENT,
              readbackMetadata: {
                kind: 'variable-length-attribute',
                valueOffsets: temporalValueOffsets,
                nullCount: 0,
                valueByteLength: scalarCount * Float32Array.BYTES_PER_ELEMENT
              }
            }
          : {})
      })
    );
    transientResources.push(configBuffer);
  }

  await waitForSubmittedWork(device);
  for (const resource of transientResources) {
    resource.destroy();
  }

  const temporal = new GPUVector({
    type: 'data',
    name: options.name,
    dataType: outputType,
    format: outputFormat,
    data: outputData,
    stride: 1,
    byteStride: Float32Array.BYTES_PER_ELEMENT,
    rowByteLength: Float32Array.BYTES_PER_ELEMENT,
    ownsData: true
  });
  return createPreparedArrowTemporalGPUVector(temporal, field, temporalInfo, true);
}

function makeArrowTemporalSourceGPUVector(
  device: Device,
  source: Vector<ArrowTemporalColumnType>,
  options: Required<Pick<ConvertArrowTemporalToGPUVectorOptions, 'name' | 'id'>>
): GPUVector {
  const sourceInfo = getRequiredArrowTemporalVectorInfo(source);
  const data = source.data.map((sourceData, chunkIndex) => {
    validateArrowTemporalData(sourceData);
    const sourceValues = getArrowTemporalDataBufferSource(sourceData);
    const byteStride = sourceInfo.bitWidth / 8;
    return new GPUData({
      buffer: new DynamicBuffer(device, {
        id: `${options.id}-temporal-source-values-${chunkIndex}`,
        usage: Buffer.STORAGE | Buffer.COPY_DST | Buffer.COPY_SRC,
        data: sourceValues
      }),
      dataType: sourceData.type as ArrowTemporalColumnType,
      format: DataType.isList(sourceData.type) ? 'vertex-list<float32>' : 'float32',
      length: sourceData.length,
      stride: 1,
      byteStride,
      rowByteLength: byteStride,
      ownsBuffer: true,
      ...(DataType.isList(sourceData.type)
        ? {
            valueOffsets: getNormalizedArrowValueOffsets(sourceData as Data<List<any>>),
            valueByteLength: sourceValues.byteLength,
            readbackMetadata: {
              kind: 'temporal-list',
              valueOffsets: getNormalizedArrowValueOffsets(sourceData as Data<List<any>>),
              valueByteLength: sourceValues.byteLength
            } satisfies TemporalListGPUReadbackMetadata
          }
        : {})
    });
  });
  return new GPUVector({
    type: 'data',
    name: options.name,
    dataType: source.type,
    format: DataType.isList(source.type) ? 'vertex-list<float32>' : 'float32',
    data,
    stride: 1,
    byteStride: sourceInfo.bitWidth / 8,
    rowByteLength: sourceInfo.bitWidth / 8,
    ownsData: true
  });
}

function makePreparedArrowTemporalVector(
  source: Vector<ArrowTemporalColumnType>,
  temporalInfo: PreparedArrowTemporalGPUVector['temporalInfo'],
  field: Field
): Vector<ArrowRelativeTemporalType> {
  const outputType = getPreparedArrowTemporalType(field);
  const outputData = source.data.map(sourceData => {
    validateArrowTemporalData(sourceData);
    const sourceValues = getArrowTemporalDataBufferSource(sourceData);
    const preparedValues = new Float32Array(sourceValues.length);
    for (let valueIndex = 0; valueIndex < sourceValues.length; valueIndex++) {
      preparedValues[valueIndex] = getRelativeTemporalValue(
        sourceValues[valueIndex]!,
        temporalInfo.origin
      );
    }

    if (DataType.isList(outputType)) {
      const childData = makeFloat32Data({
        type: new Float32(),
        length: preparedValues.length,
        data: preparedValues
      });
      return makeFloat32ListData({
        type: outputType,
        length: sourceData.length,
        nullCount: 0,
        nullBitmap: null,
        valueOffsets: getNormalizedArrowValueOffsets(sourceData as Data<List<any>>),
        child: childData
      });
    }

    return makeFloat32Data({
      type: outputType,
      length: preparedValues.length,
      data: preparedValues
    });
  });
  return new Vector<ArrowRelativeTemporalType>(outputData as Data<ArrowRelativeTemporalType>[]);
}

function makePreparedArrowTemporalField(
  source: ArrowTemporalSource,
  temporalInfo: PreparedArrowTemporalGPUVector['temporalInfo'],
  sourceField?: Field
): Field {
  const sourceType = getArrowTemporalSourceType(source);
  const metadata = makeArrowTemporalMetadata(
    temporalInfo,
    getArrowTemporalLeafField(sourceType, sourceField)?.metadata
  );
  const sourceLeafField = getArrowTemporalLeafField(sourceType, sourceField);
  const outputType = makePreparedArrowTemporalType(
    sourceType,
    new Field(sourceLeafField?.name || 'value', new Float32(), false, metadata)
  );
  return new Field(sourceField?.name || 'temporal', outputType, false, metadata);
}

function makePreparedArrowTemporalType(
  sourceType: ArrowTemporalColumnType,
  leafField: Field
): ArrowRelativeTemporalType {
  if (DataType.isList(sourceType)) {
    return new List(leafField);
  }
  return new Float32();
}

function getPreparedArrowTemporalType(field: Field): ArrowRelativeTemporalType {
  return field.type as ArrowRelativeTemporalType;
}

function makeArrowTemporalMetadata(
  temporalInfo: PreparedArrowTemporalGPUVector['temporalInfo'],
  sourceMetadata?: Map<string, string>
): Map<string, string> {
  const metadata = new Map(sourceMetadata);
  metadata.set(TEMPORAL_KIND_METADATA_KEY, temporalInfo.kind);
  metadata.set(TEMPORAL_UNIT_METADATA_KEY, temporalInfo.unit);
  metadata.set(TEMPORAL_ORIGIN_METADATA_KEY, temporalInfo.origin.toString());
  metadata.set(TEMPORAL_ORIGIN_POLICY_METADATA_KEY, temporalInfo.originPolicy);
  if (temporalInfo.timezone) {
    metadata.set(TEMPORAL_TIMEZONE_METADATA_KEY, temporalInfo.timezone);
  }
  return metadata;
}

function resolveTemporalOrigin(
  source: ArrowTemporalSource,
  temporalInfo: ArrowTemporalVectorInfo,
  options: ConvertArrowTemporalToGPUVectorOptions
): number | bigint {
  if (options.origin !== undefined) {
    return parseTemporalOrigin(options.origin.toString(), temporalInfo.bitWidth);
  }
  if (temporalInfo.origin !== undefined) {
    return temporalInfo.origin;
  }
  if (temporalInfo.kind === 'duration') {
    return temporalInfo.bitWidth === 64 ? 0n : 0;
  }
  if (source instanceof GPUVector) {
    throw new Error(
      'GPU-resident absolute temporal input requires an explicit or persisted origin'
    );
  }
  return getFirstArrowTemporalValue(source) ?? (temporalInfo.bitWidth === 64 ? 0n : 0);
}

function getFirstArrowTemporalValue(
  source: Vector<ArrowTemporalColumnType>
): number | bigint | undefined {
  for (const data of source.data) {
    validateArrowTemporalData(data);
    const values = getArrowTemporalDataBufferSource(data);
    if (values.length > 0) {
      return values[0]!;
    }
  }
  return undefined;
}

function getRequiredArrowTemporalVectorInfo(
  vector: Pick<Vector, 'type'> | Pick<GPUVector, 'dataType'>,
  field?: Field
): ArrowTemporalVectorInfo {
  const temporalInfo = getArrowTemporalVectorInfo(vector, field);
  if (!temporalInfo) {
    throw new Error(
      'convertArrowTemporalToGPUVector requires Date, Time, Timestamp, Duration, Int64 with visgl:temporal-* metadata, or List thereof'
    );
  }
  return temporalInfo;
}

function getArrowTemporalSourceType(
  source: Pick<Vector, 'type'> | Pick<GPUVector, 'dataType'>
): ArrowTemporalColumnType {
  return 'type' in source
    ? (source.type as ArrowTemporalColumnType)
    : getRequiredArrowGPUVectorDataType<ArrowTemporalColumnType>(source);
}

function getArrowTemporalLeafType(type: DataType): ArrowTemporalType | null {
  const leafType = DataType.isList(type) ? type.children[0]?.type : type;
  return leafType &&
    (DataType.isDate(leafType) ||
      DataType.isTime(leafType) ||
      DataType.isTimestamp(leafType) ||
      DataType.isDuration(leafType) ||
      DataType.isInt(leafType))
    ? (leafType as ArrowTemporalType)
    : null;
}

function getArrowTemporalLeafField(type: DataType, field?: Field): Field | undefined {
  return DataType.isList(type) ? type.children[0] : field;
}

const TEMPORAL_KINDS: readonly ArrowTemporalKind[] = ['date', 'time', 'timestamp', 'duration'];
const TEMPORAL_UNITS: readonly ArrowTemporalUnit[] = [
  'day',
  'second',
  'millisecond',
  'microsecond',
  'nanosecond'
];

function getArrowTemporalLeafInfo(
  type: ArrowTemporalType,
  metadata?: Map<string, string>
): Omit<ArrowTemporalVectorInfo, 'variableLength'> | null {
  if (DataType.isInt(type)) {
    return getArrowIntegerTemporalLeafInfo(type as Int32 | Int64, metadata);
  }
  if (DataType.isDate(type)) {
    return {
      kind: 'date',
      unit: type.unit === DateUnit.DAY ? 'day' : 'millisecond',
      bitWidth: type.unit === DateUnit.DAY ? 32 : 64
    };
  }
  if (DataType.isTime(type)) {
    return {
      kind: 'time',
      unit: getArrowTimeUnit(type.unit),
      bitWidth: type.bitWidth
    };
  }
  if (DataType.isTimestamp(type)) {
    return {
      kind: 'timestamp',
      unit: getArrowTimeUnit(type.unit),
      bitWidth: 64,
      timezone: type.timezone
    };
  }
  return {
    kind: 'duration',
    unit: getArrowTimeUnit(type.unit),
    bitWidth: 64
  };
}

function getArrowTimeUnit(unit: TimeUnit): Exclude<ArrowTemporalUnit, 'day'> {
  switch (unit) {
    case TimeUnit.SECOND:
      return 'second';
    case TimeUnit.MILLISECOND:
      return 'millisecond';
    case TimeUnit.MICROSECOND:
      return 'microsecond';
    case TimeUnit.NANOSECOND:
      return 'nanosecond';
  }
}

/**
 * Reads temporal info for a plain integer column from `visgl:temporal-*` field metadata.
 * Returns `null` without a kind or unit (not temporal) and throws on invalid values.
 */
function getArrowIntegerTemporalLeafInfo(
  type: Int32 | Int64,
  metadata?: Map<string, string>
): Omit<ArrowTemporalVectorInfo, 'variableLength'> | null {
  if (!type.isSigned || (type.bitWidth !== 32 && type.bitWidth !== 64)) {
    return null;
  }
  const kind = metadata?.get(TEMPORAL_KIND_METADATA_KEY);
  const unit = metadata?.get(TEMPORAL_UNIT_METADATA_KEY);
  if (kind === undefined && unit === undefined) {
    return null;
  }
  if (!TEMPORAL_KINDS.includes(kind as ArrowTemporalKind)) {
    throw new Error(
      `Invalid ${TEMPORAL_KIND_METADATA_KEY} "${kind}"; expected one of ${TEMPORAL_KINDS.join(', ')}`
    );
  }
  if (!TEMPORAL_UNITS.includes(unit as ArrowTemporalUnit)) {
    throw new Error(
      `Invalid ${TEMPORAL_UNIT_METADATA_KEY} "${unit}"; expected one of ${TEMPORAL_UNITS.join(', ')}`
    );
  }
  if (unit === 'day' && kind !== 'date') {
    throw new Error(`${TEMPORAL_UNIT_METADATA_KEY} "day" is only valid for kind "date"`);
  }
  const bitWidth = type.bitWidth as 32 | 64;
  if (bitWidth === 32 && (kind === 'timestamp' || kind === 'duration')) {
    throw new Error(`Int32 columns cannot carry ${TEMPORAL_KIND_METADATA_KEY} "${kind}"`);
  }
  return {
    kind: kind as ArrowTemporalKind,
    unit: unit as ArrowTemporalUnit,
    bitWidth,
    ...(kind === 'timestamp' && metadata?.has(TEMPORAL_TIMEZONE_METADATA_KEY)
      ? {timezone: metadata.get(TEMPORAL_TIMEZONE_METADATA_KEY)}
      : {})
  };
}

function parseTemporalOrigin(origin: string, bitWidth: 32 | 64): number | bigint {
  return bitWidth === 64 ? BigInt(origin) : Number(origin);
}

function getRelativeTemporalValue(value: number | bigint, origin: number | bigint): number {
  return typeof value === 'bigint'
    ? Number(value - (origin as bigint))
    : value - (origin as number);
}

function validateArrowTemporalData(data: Data<ArrowTemporalColumnType>): void {
  if (data.nullCount > 0) {
    throw new Error('convertArrowTemporalToGPUVector does not support nullable temporal rows');
  }
  if (DataType.isList(data.type) && (data.children[0]?.nullCount ?? 0) > 0) {
    throw new Error('convertArrowTemporalToGPUVector does not support nullable temporal values');
  }
}

function getArrowTemporalDataBufferSource(
  data: Data<ArrowTemporalColumnType>
): Int32Array | BigInt64Array {
  if (DataType.isList(data.type)) {
    const childData = data.children[0] as Data<ArrowTemporalType> | undefined;
    if (!childData) {
      return new Int32Array(0);
    }
    const values = childData.values as Int32Array | BigInt64Array | undefined;
    if (!values) {
      return childData.type.ArrayType === BigInt64Array ? new BigInt64Array(0) : new Int32Array(0);
    }
    const valueOffsets = data.valueOffsets as Int32Array | undefined;
    if (!valueOffsets) {
      throw new Error('convertArrowTemporalToGPUVector list input requires Arrow value offsets');
    }
    // Arrow exposes valueOffsets as the logical row slice for data.offset; only child values
    // still need their physical child offset applied when copying flattened temporal leaves.
    const firstValueOffset = valueOffsets[0] ?? 0;
    const lastValueOffset = valueOffsets[data.length] ?? firstValueOffset;
    const childValueOffset = childData.offset ?? 0;
    return values.subarray(
      childValueOffset + firstValueOffset,
      childValueOffset + lastValueOffset
    ) as Int32Array | BigInt64Array;
  }
  return getArrowTemporalScalarDataBufferSource(data as Data<ArrowTemporalType>);
}

function getArrowTemporalScalarDataBufferSource(
  data: Data<ArrowTemporalType>
): Int32Array | BigInt64Array {
  const values = data.values as Int32Array | BigInt64Array | undefined;
  if (!values) {
    return data.type.ArrayType === BigInt64Array ? new BigInt64Array(0) : new Int32Array(0);
  }
  // Arrow's `Data.slice` already advances `values` to the first row; `data.offset` only applies
  // to the validity bitmap and offset buffers, so it is not added here.
  return values.subarray(0, data.length) as Int32Array | BigInt64Array;
}

function getNormalizedArrowValueOffsets(data: Data<List<any>>): Int32Array {
  const valueOffsets = data.valueOffsets as Int32Array | undefined;
  if (!valueOffsets) {
    throw new Error('convertArrowTemporalToGPUVector list input requires Arrow value offsets');
  }
  // Arrow exposes valueOffsets as the logical row slice for data.offset.
  const firstValueOffset = valueOffsets[0] ?? 0;
  return Int32Array.from(valueOffsets, valueOffset => valueOffset - firstValueOffset);
}

function getTemporalValueOffsets(
  type: ArrowTemporalColumnType,
  data: GPUData
): Int32Array | undefined {
  if (!DataType.isList(type)) {
    return undefined;
  }
  const metadata = data.readbackMetadata as TemporalListGPUReadbackMetadata | undefined;
  if (metadata?.kind !== 'temporal-list') {
    throw new Error('GPU-resident temporal list input requires copied Arrow value offsets');
  }
  return metadata.valueOffsets;
}

function getTemporalSourceByteLength(data: GPUData): number {
  const metadata = data.readbackMetadata as TemporalListGPUReadbackMetadata | undefined;
  return metadata?.kind === 'temporal-list'
    ? metadata.valueByteLength
    : data.length * data.byteStride;
}

function makeTemporalConversionConfig(
  temporalInfo: PreparedArrowTemporalGPUVector['temporalInfo'],
  scalarCount: number
): Uint32Array {
  const originWords = toTemporalOriginWords(temporalInfo.origin, temporalInfo.bitWidth);
  return new Uint32Array([scalarCount, originWords[0], originWords[1]]);
}

function toTemporalOriginWords(origin: number | bigint, bitWidth: 32 | 64): [number, number] {
  if (bitWidth === 32) {
    return [new Uint32Array(new Int32Array([origin as number]).buffer)[0]!, 0];
  }
  const normalizedOrigin = BigInt.asUintN(64, origin as bigint);
  return [Number(normalizedOrigin & 0xffffffffn), Number((normalizedOrigin >> 32n) & 0xffffffffn)];
}

function dispatchArrowTemporalConversion(
  device: Device,
  temporalInfo: PreparedArrowTemporalGPUVector['temporalInfo'],
  props: {
    id: string;
    chunkIndex: number;
    scalarCount: number;
    sourceTemporalValues: Binding;
    temporalConversionConfig: Binding;
    preparedTemporalValues: Binding;
  }
): void {
  const computation = new Computation(device, {
    id: `${props.id}-temporal-conversion-${props.chunkIndex}`,
    source: getArrowTemporalConversionSource(temporalInfo.bitWidth),
    shaderLayout: TEMPORAL_CONVERSION_SHADER_LAYOUT,
    bindings: {
      sourceTemporalValues: props.sourceTemporalValues,
      temporalConversionConfig: props.temporalConversionConfig,
      preparedTemporalValues: props.preparedTemporalValues
    }
  });
  if (props.scalarCount > 0) {
    const computePass = device.beginComputePass({});
    computation.dispatch(computePass, Math.ceil(props.scalarCount / 64));
    computePass.end();
    device.submit();
  }
  computation.destroy();
}

function getArrowTemporalConversionSource(bitWidth: 32 | 64): string {
  const sourceType = bitWidth === 64 ? 'array<vec2<u32>>' : 'array<i32>';
  const readRelativeValue =
    bitWidth === 64
      ? `
fn readRelativeTemporalValue(scalarIndex : u32) -> f32 {
  let valueBits = sourceTemporalValues[scalarIndex];
  let originBits = vec2<u32>(temporalConversionConfig[1], temporalConversionConfig[2]);
  let deltaLow = valueBits.x - originBits.x;
  let borrow = select(0u, 1u, valueBits.x < originBits.x);
  let deltaHigh = valueBits.y - originBits.y - borrow;
  let isNegative = (deltaHigh & 0x80000000u) != 0u;
  var magnitudeLow = deltaLow;
  var magnitudeHigh = deltaHigh;
  if (isNegative) {
    magnitudeLow = ~deltaLow + 1u;
    let carry = select(0u, 1u, magnitudeLow == 0u);
    magnitudeHigh = ~deltaHigh + carry;
  }
  let magnitude = f32(magnitudeHigh) * 4294967296.0 + f32(magnitudeLow);
  return select(magnitude, -magnitude, isNegative);
}
`
      : `
fn readRelativeTemporalValue(scalarIndex : u32) -> f32 {
  return f32(sourceTemporalValues[scalarIndex] - bitcast<i32>(temporalConversionConfig[1]));
}
`;
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read> sourceTemporalValues : ${sourceType};
@group(0) @binding(1) var<storage, read> temporalConversionConfig : array<u32>;
@group(0) @binding(2) var<storage, read_write> preparedTemporalValues : array<f32>;

${readRelativeValue}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) globalInvocationId : vec3<u32>) {
  let scalarIndex = globalInvocationId.x;
  if (scalarIndex >= temporalConversionConfig[0]) {
    return;
  }
  preparedTemporalValues[scalarIndex] = readRelativeTemporalValue(scalarIndex);
}
`;
}

function getGPUDataBinding(data: GPUData, size: number): Binding {
  return {
    buffer: getGPUDataBuffer(data),
    offset: data.byteOffset,
    ...(size > 0 ? {size} : {})
  };
}

function getGPUDataBuffer(data: GPUData): Buffer {
  return data.buffer instanceof DynamicBuffer ? data.buffer.buffer : data.buffer;
}

async function waitForSubmittedWork(device: Device): Promise<void> {
  const queue = (
    device as Device & {
      handle?: {queue?: {onSubmittedWorkDone?: () => Promise<void>}};
    }
  ).handle?.queue;
  await queue?.onSubmittedWorkDone?.();
}
