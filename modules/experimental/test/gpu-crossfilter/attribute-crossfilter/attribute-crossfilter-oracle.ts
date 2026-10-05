// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUAttributeCrossfilterDimensionState} from '../../../src/gpu-crossfilter/attribute-crossfilter';

/** One oracle dimension: column values plus compile-time shape. */
export type AttributeCrossfilterOracleDimension = {
  values: ArrayLike<number>;
  binCount: number;
  domain?: 'auto' | 'parameters';
  isUint32?: boolean;
};

/** Oracle result mirroring the recipe outputs. */
export type AttributeCrossfilterOracleResult = {
  histograms: number[];
  domains: number[];
  selectedCount: number;
  liveCount: number;
  selection: number[];
  selectedIds: number[];
};

const isFinite32 = (value: number) => Number.isFinite(value);

/** CPU reference with the same f32 binning math as the WGSL kernels. */
export function computeAttributeCrossfilterOracle(
  dimensions: readonly AttributeCrossfilterOracleDimension[],
  states: readonly GPUAttributeCrossfilterDimensionState[],
  liveMask?: ArrayLike<number>
): AttributeCrossfilterOracleResult {
  const rows = dimensions[0].values.length;
  const live = (row: number) => (liveMask ? liveMask[row] !== 0 : true);
  const value = (dimensionIndex: number, row: number) =>
    Math.fround(dimensions[dimensionIndex].values[row]);
  const finite = (dimensionIndex: number, row: number) => isFinite32(value(dimensionIndex, row));

  const domains: number[] = [];
  for (const [dimensionIndex, dimension] of dimensions.entries()) {
    if ((dimension.domain ?? 'auto') === 'parameters') {
      domains.push(
        Math.fround(states[dimensionIndex].domain?.[0] ?? 0),
        Math.fround(states[dimensionIndex].domain?.[1] ?? 0)
      );
      continue;
    }
    let minimum = Infinity;
    let maximum = -Infinity;
    for (let row = 0; row < rows; row++) {
      if (live(row) && finite(dimensionIndex, row)) {
        minimum = Math.min(minimum, value(dimensionIndex, row));
        maximum = Math.max(maximum, value(dimensionIndex, row));
      }
    }
    domains.push(...(minimum > maximum ? [0, 0] : [minimum, maximum]));
  }

  const fails = (dimensionIndex: number, row: number) => {
    if (!finite(dimensionIndex, row)) {
      return true;
    }
    const brush = states[dimensionIndex].brush;
    const v = value(dimensionIndex, row);
    return Boolean(brush) && !(v >= Math.fround(brush![0]) && v < Math.fround(brush![1]));
  };

  const histograms: number[] = [];
  for (const [dimensionIndex, dimension] of dimensions.entries()) {
    const bins = new Array<number>(dimension.binCount).fill(0);
    const minimum = domains[2 * dimensionIndex];
    const maximum = domains[2 * dimensionIndex + 1];
    for (let row = 0; row < rows; row++) {
      if (!live(row) || !finite(dimensionIndex, row)) {
        continue;
      }
      if (dimensions.some((_, other) => other !== dimensionIndex && fails(other, row))) {
        continue;
      }
      const v = value(dimensionIndex, row);
      if (!(v >= minimum && v <= maximum)) {
        continue;
      }
      let bin = 0;
      if (maximum > minimum) {
        const scale = Math.fround(dimension.binCount / Math.fround(maximum - minimum));
        const scaled = Math.fround(Math.fround(v - minimum) * scale);
        bin = Math.min(Math.floor(scaled), dimension.binCount - 1);
      }
      bins[bin]++;
    }
    histograms.push(...bins);
  }

  const selection: number[] = [];
  const selectedIds: number[] = [];
  let liveCount = 0;
  for (let row = 0; row < rows; row++) {
    const isLive = live(row);
    liveCount += isLive ? 1 : 0;
    const selected = isLive && !dimensions.some((_, index) => fails(index, row));
    selection.push(selected ? 1 : 0);
    if (selected) {
      selectedIds.push(row);
    }
  }
  return {
    histograms,
    domains,
    selectedCount: selectedIds.length,
    liveCount,
    selection,
    selectedIds
  };
}
