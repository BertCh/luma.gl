// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Scalar formats supported by packed spatial-analysis parameter schemas. */
export type GPUSpatialParameterFormat = 'float32' | 'uint32' | 'sint32';

/** One named scalar in a packed GPU parameter layout. */
export type GPUSpatialParameterField<Format extends GPUSpatialParameterFormat> = {
  name: string;
  format: Format;
  wordOffset: number;
  defaultValue: number;
  minimum?: number;
  maximum?: number;
  units?: string;
  /** Accept NaN or infinity when the kernel gives those values an explicit meaning. */
  allowNonFinite?: boolean;
  /** Whether the value may change between graph encodings without recompilation. */
  dynamic: boolean;
};

/** Declarative metadata for one homogeneous packed parameter view. */
export type GPUSpatialParameterSchema<Format extends GPUSpatialParameterFormat> = {
  id: string;
  format: Format;
  wordLength: number;
  fields: readonly GPUSpatialParameterField<Format>[];
};

/** Values accepted by a schema packer, keyed by field name. */
export type GPUSpatialParameterValues = Readonly<Record<string, number | undefined>>;

/** Typed packed array corresponding to a schema format. */
export type GPUSpatialParameterArray<Format extends GPUSpatialParameterFormat> =
  Format extends 'float32' ? Float32Array : Format extends 'uint32' ? Uint32Array : Int32Array;

/** Validates and freezes a parameter schema for reuse by contributors and documentation. */
export function defineGPUSpatialParameterSchema<Format extends GPUSpatialParameterFormat>(
  schema: GPUSpatialParameterSchema<Format>
): GPUSpatialParameterSchema<Format> {
  validateGPUSpatialParameterSchema(schema);
  return Object.freeze({
    ...schema,
    fields: Object.freeze(schema.fields.map(field => Object.freeze({...field})))
  });
}

/** Validates schema names, formats, offsets, defaults and ranges. */
export function validateGPUSpatialParameterSchema<Format extends GPUSpatialParameterFormat>(
  schema: GPUSpatialParameterSchema<Format>
): void {
  if (!schema.id || !Number.isInteger(schema.wordLength) || schema.wordLength < 1) {
    throw new Error('parameter schema needs an ID and a positive wordLength');
  }
  const names = new Set<string>();
  const offsets = new Set<number>();
  for (const field of schema.fields) {
    if (!field.name || names.has(field.name)) {
      throw new Error(`${schema.id} parameter field names must be unique and non-empty`);
    }
    if (field.format !== schema.format) {
      throw new Error(`${schema.id} field ${field.name} format must match the schema`);
    }
    if (
      !Number.isInteger(field.wordOffset) ||
      field.wordOffset < 0 ||
      field.wordOffset >= schema.wordLength ||
      offsets.has(field.wordOffset)
    ) {
      throw new Error(`${schema.id} field ${field.name} needs a unique in-range wordOffset`);
    }
    if (
      field.minimum !== undefined &&
      field.maximum !== undefined &&
      field.minimum > field.maximum
    ) {
      throw new Error(`${schema.id} field ${field.name} minimum exceeds maximum`);
    }
    validateFieldValue(schema.id, field, field.defaultValue);
    names.add(field.name);
    offsets.add(field.wordOffset);
  }
}

/** Packs defaults plus supplied overrides into the schema's GPU-ready typed array. */
export function packGPUSpatialParameterValues<Format extends GPUSpatialParameterFormat>(
  schema: GPUSpatialParameterSchema<Format>,
  values: GPUSpatialParameterValues = {}
): GPUSpatialParameterArray<Format> {
  validateGPUSpatialParameterSchema(schema);
  const fieldByName = new Map(schema.fields.map(field => [field.name, field]));
  for (const name of Object.keys(values)) {
    if (!fieldByName.has(name)) {
      throw new Error(`${schema.id} has no parameter named ${name}`);
    }
  }
  const packed = createParameterArray(schema.format, schema.wordLength);
  for (const field of schema.fields) {
    const value = values[field.name] ?? field.defaultValue;
    validateFieldValue(schema.id, field, value);
    packed[field.wordOffset] = value;
  }
  return packed as GPUSpatialParameterArray<Format>;
}

function createParameterArray(
  format: GPUSpatialParameterFormat,
  length: number
): Float32Array | Uint32Array | Int32Array {
  switch (format) {
    case 'float32':
      return new Float32Array(length);
    case 'uint32':
      return new Uint32Array(length);
    case 'sint32':
      return new Int32Array(length);
  }
}

function validateFieldValue<Format extends GPUSpatialParameterFormat>(
  schemaId: string,
  field: GPUSpatialParameterField<Format>,
  value: number
): void {
  if (!field.allowNonFinite && !Number.isFinite(value)) {
    throw new Error(`${schemaId} field ${field.name} must be finite`);
  }
  if (field.format === 'uint32' && (!Number.isInteger(value) || value < 0 || value > 0xffffffff)) {
    throw new Error(`${schemaId} field ${field.name} must be a uint32`);
  }
  if (
    field.format === 'sint32' &&
    (!Number.isInteger(value) || value < -0x80000000 || value > 0x7fffffff)
  ) {
    throw new Error(`${schemaId} field ${field.name} must be a sint32`);
  }
  if (field.minimum !== undefined && value < field.minimum) {
    throw new Error(`${schemaId} field ${field.name} is below its minimum`);
  }
  if (field.maximum !== undefined && value > field.maximum) {
    throw new Error(`${schemaId} field ${field.name} exceeds its maximum`);
  }
}
