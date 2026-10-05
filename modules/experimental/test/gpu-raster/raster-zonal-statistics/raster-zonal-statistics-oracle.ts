// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Options mirroring `GPURasterZonalStatisticsProps` topology and band calibration. */
export type ZonalStatisticsOracleOptions = {
  zoneCapacity: number;
  /** Defaults to 0xffffffff. */
  ignoredZone?: number;
  /** Raw nodata value compared before calibration. */
  noDataValue?: number;
  /** Optional per-cell validity flags. */
  validity?: ArrayLike<number>;
  scale?: number;
  offset?: number;
};

/** Per-zone reference statistics. Empty zones follow the GPU convention (NaN mean/min/max). */
export type ZonalStatisticsOracleResult = {
  cellCounts: number[];
  valueCounts: number[];
  sums: number[];
  means: number[];
  minimums: number[];
  maximums: number[];
  overflow: number;
};

/** Computes per-zone statistics on the CPU with the same validity rules as the GPU recipe. */
export function computeZonalStatistics(
  zones: ArrayLike<number>,
  rawValues: ArrayLike<number>,
  options: ZonalStatisticsOracleOptions
): ZonalStatisticsOracleResult {
  const {zoneCapacity} = options;
  const ignoredZone = options.ignoredZone ?? 0xffffffff;
  const scale = options.scale ?? 1;
  const offset = options.offset ?? 0;
  const cellCounts = new Array<number>(zoneCapacity).fill(0);
  const valueCounts = new Array<number>(zoneCapacity).fill(0);
  const sums = new Array<number>(zoneCapacity).fill(0);
  const minimums = new Array<number>(zoneCapacity).fill(Number.NaN);
  const maximums = new Array<number>(zoneCapacity).fill(Number.NaN);
  let overflow = 0;
  for (let index = 0; index < zones.length; index++) {
    const zone = zones[index];
    if (zone === ignoredZone) {
      continue;
    }
    if (zone >= zoneCapacity) {
      overflow = 1;
      continue;
    }
    cellCounts[zone]++;
    const raw = rawValues[index];
    if (options.validity && options.validity[index] === 0) {
      continue;
    }
    if (options.noDataValue !== undefined && raw === options.noDataValue) {
      continue;
    }
    const value = Math.fround(Math.fround(Math.fround(raw) * scale) + offset);
    if (!Number.isFinite(value)) {
      continue;
    }
    valueCounts[zone]++;
    sums[zone] = Math.fround(sums[zone] + value);
    minimums[zone] = Number.isNaN(minimums[zone]) ? value : Math.min(minimums[zone], value);
    maximums[zone] = Number.isNaN(maximums[zone]) ? value : Math.max(maximums[zone], value);
  }
  const means = sums.map((sum, zone) => (valueCounts[zone] ? sum / valueCounts[zone] : Number.NaN));
  return {cellCounts, valueCounts, sums, means, minimums, maximums, overflow};
}
