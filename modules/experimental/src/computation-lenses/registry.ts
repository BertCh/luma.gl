// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {LENS_RUNG_IDS} from '../computation-views/types';
import {compactLensRecordEnvelope, expandLensRecordEnvelope} from './record-encoding';
import type {LensScenarioDescriptor, LensScenarioResult, RecordedLensFile} from './types';

const SCENARIO_ID =
  /^[a-z0-9]+(?:-[a-z0-9]+)*\.[a-z0-9]+(?:-[a-z0-9]+)*(?:\.[a-z0-9]+(?:-[a-z0-9]+)*)?$/;
const scenarios = new Map<string, LensScenarioDescriptor>();

/** Checks the public scenario-id syntax without loading GPU code. */
export function isLensScenarioId(id: string): boolean {
  return SCENARIO_ID.test(id);
}

/**
 * Registers a lazy descriptor. Scenario modules call this at module scope, then hosts import
 * those registration modules explicitly; the registry never imports or executes GPU scenarios.
 */
export function registerLensScenario(descriptor: LensScenarioDescriptor): void {
  if (!isLensScenarioId(descriptor.id)) throw new Error(`Invalid scenario id: ${descriptor.id}`);
  const [entry] = descriptor.id.split('.');
  if (entry !== descriptor.entry) throw new Error('Scenario entry must match the first id segment');
  // The element is the documented export (class name, or function name for recipes), so the
  // coverage contract can join scenarios to pack operators and formulas by that name.
  if (!/^[A-Za-z_$][\w$]*$/.test(descriptor.element))
    throw new Error(`Scenario element must be an exported identifier: ${descriptor.element}`);
  if (
    !descriptor.rungs.includes(descriptor.defaultRung) ||
    descriptor.rungs.some(rung => !LENS_RUNG_IDS.includes(rung))
  )
    throw new Error('Scenario defaultRung must belong to valid rungs');
  if (scenarios.has(descriptor.id)) throw new Error(`Duplicate scenario id: ${descriptor.id}`);
  scenarios.set(descriptor.id, descriptor);
}

/** Looks up a scenario descriptor without loading it. */
export function getLensScenario(id: string): LensScenarioDescriptor | undefined {
  return scenarios.get(id);
}

/** Lists scenarios deterministically by id. */
export function listLensScenarios(): LensScenarioDescriptor[] {
  return [...scenarios.values()].sort((first, second) =>
    first.id < second.id ? -1 : first.id > second.id ? 1 : 0
  );
}

/**
 * Copies a result to a JSON-safe, compacted envelope: non-finite values become explicit tokens,
 * typed captures base64, and the record is losslessly compacted (`record-encoding.ts`).
 */
export function toRecordedLensFile(
  result: LensScenarioResult,
  options: {scenarioId: string; recordedAt: string; adapterLabel: string; variant?: string}
): RecordedLensFile {
  if (!isLensScenarioId(options.scenarioId))
    throw new Error(`Invalid scenarioId: ${options.scenarioId}`);
  // The record schema is numeric in memory; only this serialization boundary uses string tokens.
  // Lossless compaction (see record-encoding.ts) keeps docs records within their size budget.
  return compactLensRecordEnvelope({
    schema: 'luma.lens-record/1',
    ...options,
    record: transformValues(result, false) as LensScenarioResult
  }) as RecordedLensFile;
}

/** Validates an envelope and restores capture arrays, including NaN and infinities. */
export function fromRecordedLensFile(json: unknown): RecordedLensFile {
  if (!isObject(json) || json['schema'] !== 'luma.lens-record/1')
    throw new Error('Invalid lens record schema');
  if (typeof json['scenarioId'] !== 'string' || !isLensScenarioId(json['scenarioId']))
    throw new Error('Invalid lens record scenarioId');
  if (typeof json['recordedAt'] !== 'string' || !Number.isFinite(Date.parse(json['recordedAt'])))
    throw new Error('Invalid lens record recordedAt');
  if (typeof json['adapterLabel'] !== 'string' || !json['adapterLabel'].trim())
    throw new Error('Invalid lens record adapterLabel');
  if (json['variant'] !== undefined && typeof json['variant'] !== 'string')
    throw new Error('Invalid lens record variant');
  if (
    !isObject(json['record']) ||
    json['record']['schema'] !== 'luma.computation/1' ||
    typeof json['record']['runId'] !== 'string' ||
    !json['record']['runId'] ||
    !Array.isArray(json['record']['graphs'])
  )
    throw new Error('Invalid lens record record');
  // Validation and recursive conversion are localized to the external JSON boundary.
  return transformValues(expandLensRecordEnvelope(json), true) as RecordedLensFile;
}

function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function transformValues(value: unknown, restore: boolean): unknown {
  if (Array.isArray(value)) return value.map(item => transformValues(item, restore));
  if (!isObject(value)) return value;
  const result: Record<string, unknown> = {};
  for (const [key, item] of Object.entries(value)) {
    const isArrayLike =
      item !== null &&
      typeof item === 'object' &&
      'length' in item &&
      typeof item.length === 'number';
    // Free-form attributes may also be named "values" without representing a capture.
    if (key !== 'values' || (!isArrayLike && value['elementType'] === undefined)) {
      result[key] = transformValues(item, restore);
      continue;
    }
    const elementType = value['elementType'];
    const isTyped = elementType === 'f32' || elementType === 'u32' || elementType === 'i32';
    // Typed captures travel as base64 of their little-endian bytes: exact, and about a third the
    // size of decimal text. Number arrays are still accepted on restore.
    if (!restore && isTyped) {
      result[key] = encodeCaptureValues(item as ArrayLike<number>, elementType);
      continue;
    }
    if (restore && isTyped && typeof item === 'string') {
      result[key] = decodeCaptureValues(item, elementType);
      continue;
    }
    if (
      item === null ||
      typeof item !== 'object' ||
      !('length' in item) ||
      typeof item.length !== 'number'
    )
      throw new Error('Invalid record values: expected an array');
    const values = Array.from(item as ArrayLike<unknown>, entry => {
      if (restore) {
        if (entry === 'NaN') return NaN;
        if (entry === 'Infinity') return Infinity;
        if (entry === '-Infinity') return -Infinity;
        if (typeof entry !== 'number')
          throw new Error('Invalid record values: expected numbers or non-finite tokens');
        return entry;
      }
      if (typeof entry !== 'number') throw new Error('Invalid record values: expected numbers');
      return Number.isFinite(entry) ? entry : String(entry);
    });
    if (!restore || value['elementType'] === undefined) {
      result[key] = values;
      continue;
    }
    // Restoring produces numeric entries; the element type selects the capture storage.
    const numbers = values as number[];
    switch (value['elementType']) {
      case 'f32':
        result[key] = new Float32Array(numbers);
        break;
      case 'u32':
      case 'i32': {
        const minimum = value['elementType'] === 'u32' ? 0 : -2147483648;
        const maximum = value['elementType'] === 'u32' ? 4294967295 : 2147483647;
        if (
          numbers.some(number => !Number.isInteger(number) || number < minimum || number > maximum)
        )
          throw new Error(`Invalid ${value['elementType']} values`);
        result[key] =
          value['elementType'] === 'u32' ? new Uint32Array(numbers) : new Int32Array(numbers);
        break;
      }
      default:
        throw new Error(`Invalid capture elementType: ${String(value['elementType'])}`);
    }
  }
  return result;
}

const BASE64_PREFIX = 'b64:';

function encodeCaptureValues(
  values: ArrayLike<number>,
  elementType: 'f32' | 'u32' | 'i32'
): string {
  const typed =
    elementType === 'f32'
      ? Float32Array.from(values)
      : elementType === 'u32'
        ? Uint32Array.from(values)
        : Int32Array.from(values);
  const bytes = new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength);
  let binary = '';
  // Chunked to stay under argument-count limits of String.fromCharCode.
  for (let offset = 0; offset < bytes.length; offset += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
  }
  return `${BASE64_PREFIX}${btoa(binary)}`;
}

function decodeCaptureValues(
  text: string,
  elementType: 'f32' | 'u32' | 'i32'
): Float32Array | Uint32Array | Int32Array {
  if (!text.startsWith(BASE64_PREFIX)) throw new Error('Invalid record values: expected b64 text');
  const binary = atob(text.slice(BASE64_PREFIX.length));
  if (binary.length % 4 !== 0) throw new Error(`Invalid ${elementType} values`);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index++) bytes[index] = binary.charCodeAt(index);
  if (elementType === 'f32') return new Float32Array(bytes.buffer);
  return elementType === 'u32' ? new Uint32Array(bytes.buffer) : new Int32Array(bytes.buffer);
}
