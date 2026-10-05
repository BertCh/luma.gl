// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GraphVectorView,
  validatePackedUint32View,
  validatePackedView,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPULinePathOutput} from './line-segmentize-types';

/** Returns every view of a path output, for aliasing and graph-membership checks. @internal */
export function getLinePathOutputViews(output: GPULinePathOutput): (GraphDataView | undefined)[] {
  return [
    output.positions,
    output.pathOffsets,
    output.count,
    output.overflow,
    output.totalCount,
    output.pathCount,
    output.sourcePaths,
    output.sourceRows,
    output.measures
  ];
}

/**
 * Validates the formats and lengths of a {@link GPULinePathOutput}.
 *
 * @param id Contributor ID for error messages.
 * @param output Output to validate.
 * @param pathCapacity Required number of output paths (`pathOffsets.length - 1`).
 * @internal
 */
export function validateLinePathOutput(
  id: string,
  output: GPULinePathOutput,
  pathCapacity: number
): void {
  for (const [name, view] of Object.entries(output)) {
    if ((view as unknown) instanceof GraphVectorView) {
      throw new Error(`${id} output.${name} must be a single packed view, not a chunked vector`);
    }
  }
  validatePackedView(output.positions, ['float32x2'], `${id} output.positions`);
  if (output.positions.length < 1) {
    throw new Error(`${id} output.positions must hold at least one row`);
  }
  const capacity = output.positions.length;
  validatePackedUint32View(output.pathOffsets, `${id} output.pathOffsets`);
  if (output.pathOffsets.length !== pathCapacity + 1) {
    throw new Error(`${id} output.pathOffsets must hold ${pathCapacity + 1} rows`);
  }
  for (const [name, scalar] of [
    ['count', output.count],
    ['overflow', output.overflow],
    ['totalCount', output.totalCount],
    ['pathCount', output.pathCount]
  ] as const) {
    if (scalar) {
      validatePackedUint32View(scalar, `${id} output.${name}`);
      if (scalar.length < 1) {
        throw new Error(`${id} output.${name} must contain one uint32 row`);
      }
    }
  }
  if (output.sourcePaths) {
    validatePackedUint32View(output.sourcePaths, `${id} output.sourcePaths`);
    if (output.sourcePaths.length !== pathCapacity) {
      throw new Error(`${id} output.sourcePaths must hold ${pathCapacity} rows`);
    }
  }
  if (output.sourceRows) {
    validatePackedUint32View(output.sourceRows, `${id} output.sourceRows`);
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
