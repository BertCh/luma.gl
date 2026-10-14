// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {validatePackedView} from '../../utils/gpu-contributor-utils';
import type {GPULineGeometryOutput} from './line-segmentize-types';

/** Returns every view of a line output, for aliasing and graph-membership checks. @internal */
export function getLineGeometryOutputViews(
  output: GPULineGeometryOutput
): (GraphDataView | undefined)[] {
  return [
    output.positions,
    output.lineOffsets,
    output.status.count,
    output.status.overflow,
    output.status.requiredCount,
    output.status.lineCount,
    output.sourceLineIndices,
    output.sourceRows,
    output.measures
  ];
}

/**
 * Validates the formats and lengths of a {@link GPULineGeometryOutput}.
 *
 * @param id Contributor ID for error messages.
 * @param output Output to validate.
 * @param lineCapacity Required number of output lines (`lineOffsets.length - 1`).
 * @internal
 */
export function validateLineGeometryOutput(
  id: string,
  output: GPULineGeometryOutput,
  lineCapacity: number
): void {
  validatePackedView(output.positions, ['float32x2'], `${id} output.positions`);
  if (output.positions.length < 1) {
    throw new Error(`${id} output.positions must hold at least one row`);
  }
  const capacity = output.positions.length;
  validatePackedView(output.lineOffsets, ['uint32'], `${id} output.lineOffsets`);
  if (output.lineOffsets.length !== lineCapacity + 1) {
    throw new Error(`${id} output.lineOffsets must hold ${lineCapacity + 1} rows`);
  }
  for (const [name, scalar] of [
    ['count', output.status.count],
    ['overflow', output.status.overflow],
    ['requiredCount', output.status.requiredCount],
    ['lineCount', output.status.lineCount]
  ] as const) {
    if (scalar) {
      validatePackedView(scalar, ['uint32'], `${id} output.status.${name}`);
      if (scalar.length < 1) {
        throw new Error(`${id} output.status.${name} must contain one uint32 row`);
      }
    }
  }
  if (output.sourceLineIndices) {
    validatePackedView(output.sourceLineIndices, ['uint32'], `${id} output.sourceLineIndices`);
    if (output.sourceLineIndices.length !== lineCapacity) {
      throw new Error(`${id} output.sourceLineIndices must hold ${lineCapacity} rows`);
    }
  }
  if (output.sourceRows) {
    validatePackedView(output.sourceRows, ['uint32'], `${id} output.sourceRows`);
    if (output.sourceRows.length !== capacity) {
      throw new Error(`${id} output.sourceRows length must equal output.positions length`);
    }
  }
  if (output.measures) {
    validatePackedView(output.measures, ['float32'], `${id} output.measures`);
    if (output.measures.length !== capacity) {
      throw new Error(`${id} output.measures length must equal output.positions length`);
    }
  }
}
