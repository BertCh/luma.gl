// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect} from 'vitest';
import {importGraphBuffer} from '../../src/utils/gpu-contributor-utils';
import {createInputBuffer, createOutputBuffer} from '../utils/gpu-contributor-test-utils';

/** Scoped graph buffers for the independent fixture specs. */
export function createParityGraph(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const input = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    name: string,
    values: number[],
    format: Format
  ) => {
    const buffer = createInputBuffer(
      device,
      format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values)
    );
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(
        graph,
        name,
        buffer,
        format,
        values.length / (format === 'float32x2' ? 2 : 1)
      )
    };
  };
  const output = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    name: string,
    length: number,
    format: Format
  ) => {
    const buffer = createOutputBuffer(device, length * (format === 'float32x2' ? 2 : 1));
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
  };
  return {
    graph,
    input,
    output,
    destroy: () => {
      for (const buffer of buffers) buffer.destroy();
    }
  };
}

/** Variable analytic bounds, preserving measured errors in native degrees in PARITY logs. */
export function compareGeodesicAngles(
  className: string,
  caseName: string,
  metric: string,
  actual: number[],
  expected: number[],
  bounds: number[],
  circular: boolean,
  singular: number[] = []
): void {
  let maxAbs = 0;
  let maxRel = 0;
  let mismatches = Math.abs(actual.length - expected.length);
  let compared = 0;
  const failures: {index: number; actual: string; expected: number; bound: number}[] = [];
  for (let index = 0; index < expected.length; index++) {
    if (singular[index]) continue;
    compared++;
    const difference = circular
      ? Math.abs(((actual[index] - expected[index] + 540) % 360) - 180)
      : Math.abs(actual[index] - expected[index]);
    if (!Number.isFinite(difference) || difference > bounds[index]) {
      failures.push({
        index,
        actual: String(actual[index]),
        expected: expected[index],
        bound: bounds[index]
      });
    }
    if (!Number.isFinite(difference)) {
      mismatches++;
      continue;
    }
    maxAbs = Math.max(maxAbs, difference);
    maxRel = Math.max(maxRel, difference / Math.max(Math.abs(expected[index]), 1e-30));
    mismatches += Number(difference > bounds[index]);
  }
  console.log(
    'PARITY',
    JSON.stringify({
      class: className,
      case: caseName,
      reference: 'pyproj.Geod.inv/fwd (PROJ)',
      n: compared,
      metric,
      maxAbs,
      maxRel,
      mismatches,
      tolerance: {
        absolute: {minimum: Math.min(...bounds), maximum: Math.max(...bounds)},
        relative: 0,
        perElement: 'fixture analytic bound; degrees',
        excludedSingular: expected.length - compared,
        failures
      }
    })
  );
  expect(mismatches, `${className}/${caseName}/${metric}`).toBe(0);
}
