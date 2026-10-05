// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_BIVARIATE_CLASSIFICATION_NO_CLASS} from '../../../src/map-graphs/column-classification/bivariate-classification-parameters';

/** Input of {@link computeBivariateClassificationOnCPU}. */
export type BivariateOracleInput = {
  valuesX: ArrayLike<number>;
  valuesY: ArrayLike<number>;
  mask?: ArrayLike<number>;
  breaksX: ArrayLike<number>;
  breaksY: ArrayLike<number>;
  classCountX: number;
  classCountY: number;
  maximumClassCount: number;
  palette: ArrayLike<number>;
  alphaValues?: ArrayLike<number>;
  noDataColor?: number;
  valueByAlpha?: {domain: readonly [number, number]; minimumAlpha: number};
};

/** Output of {@link computeBivariateClassificationOnCPU}. */
export type BivariateOracleResult = {
  classIds: Uint32Array;
  colors: Uint32Array;
  classCounts: Uint32Array;
};

/** Brute-force reference: counts inner edges `<= value` with a linear scan. */
export function computeBivariateClassificationOnCPU(
  input: BivariateOracleInput
): BivariateOracleResult {
  const rows = input.valuesX.length;
  const classIds = new Uint32Array(rows).fill(GPU_BIVARIATE_CLASSIFICATION_NO_CLASS);
  const colors = new Uint32Array(rows);
  const classCounts = new Uint32Array(input.maximumClassCount ** 2);
  const noDataColor = (input.noDataColor ?? 0) >>> 0;
  const countX = Math.min(input.classCountX, input.maximumClassCount);
  const countY = Math.min(input.classCountY, input.maximumClassCount);
  const getAxisClass = (value: number, breaks: ArrayLike<number>, count: number) => {
    let result = 0;
    for (let edge = 1; edge <= count - 1; edge++) {
      if (breaks[edge] <= value) {
        result++;
      }
    }
    return result;
  };
  for (let row = 0; row < rows; row++) {
    colors[row] = noDataColor;
    const x = input.valuesX[row];
    const y = input.valuesY[row];
    if (
      (input.mask && input.mask[row] === 0) ||
      Number.isNaN(x) ||
      Number.isNaN(y) ||
      countX === 0 ||
      countY === 0
    ) {
      continue;
    }
    const classId =
      getAxisClass(y, input.breaksY, countY) * countX + getAxisClass(x, input.breaksX, countX);
    classIds[row] = classId;
    classCounts[classId]++;
    let color = input.palette[classId] >>> 0;
    const alpha = input.valueByAlpha;
    if (alpha && input.alphaValues) {
      const [low, high] = alpha.domain;
      const alphaValue = input.alphaValues[row];
      let ramp = high > low ? Math.min(Math.max((alphaValue - low) / (high - low), 0), 1) : 1;
      if (Number.isNaN(alphaValue)) {
        ramp = 0;
      }
      const factor = alpha.minimumAlpha + (1 - alpha.minimumAlpha) * ramp;
      const scaled = Math.min(Math.floor((color >>> 24) * factor + 0.5), 255);
      color = ((color & 0x00ffffff) | (scaled << 24)) >>> 0;
    }
    colors[row] = color;
  }
  return {classIds, colors, classCounts};
}
