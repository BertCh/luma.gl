// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * CPU model of the summed-area mean filter of `getReliefSummedAreaFilterNodes`: the same
 * quantization, the same box queries with edge-replication weights, the same exact 64-bit
 * modular integers (BigInt, wrapped to 64 bits after every operation), and the same float32
 * conversions in the same order.
 */

const MODULUS = 2n ** 64n;
const wrap64 = (value: bigint): bigint => BigInt.asUintN(64, value);
const toSigned = (value: bigint): bigint => BigInt.asIntN(64, value);

/** WGSL round-to-nearest-even of a float32 product. */
function roundTiesToEven(value: number): number {
  const floor = Math.floor(value);
  return value - floor === 0.5 ? floor + (floor % 2 === 0 ? 0 : 1) : Math.round(value);
}

/** Maximum quantized magnitude the quantize kernel accepts. */
export const MODEL_MAXIMUM_QUANTIZED = 2147483520;

/** Modular 2D summed-area table over quantized elevations and counts. */
export type ReliefSummedAreaModel = {
  width: number;
  height: number;
  quantum: number;
  anchor: number;
  quantized: Int32Array;
  isValid: Uint8Array;
  sums: bigint[];
  counts: number[];
};

/** Quantizes (invalid or unquantizable cells contribute zero) and builds the table. */
export function buildReliefSummedAreaModel(
  elevation: ArrayLike<number>,
  validity: ArrayLike<number>,
  width: number,
  height: number,
  quantum: number
): ReliefSummedAreaModel {
  const quantized = new Int32Array(width * height);
  const isValid = new Uint8Array(width * height);
  let anchor = 0;
  for (let index = 0; index < width * height; index++) {
    if (validity[index] && Number.isFinite(elevation[index])) {
      anchor = Math.fround(elevation[index]);
      if (Math.abs(anchor) < quantum * 2 ** 23)
        anchor = roundTiesToEven(Math.fround(anchor / quantum)) * quantum;
      break;
    }
  }
  for (let index = 0; index < width * height; index++) {
    const scaled = Math.fround(
      Math.fround(Math.fround(elevation[index]) - anchor) * Math.fround(1 / quantum)
    );
    if (validity[index] && Number.isFinite(scaled) && Math.abs(scaled) <= MODEL_MAXIMUM_QUANTIZED) {
      isValid[index] = 1;
      quantized[index] = roundTiesToEven(scaled);
    }
  }
  const sums = new Array<bigint>(width * height).fill(0n);
  const counts = new Array<number>(width * height).fill(0);
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      const index = row * width + column;
      const up = row > 0 ? index - width : -1;
      const left = column > 0 ? index - 1 : -1;
      const diagonal = row > 0 && column > 0 ? index - width - 1 : -1;
      sums[index] = wrap64(
        BigInt(quantized[index]) +
          (up >= 0 ? sums[up] : 0n) +
          (left >= 0 ? sums[left] : 0n) -
          (diagonal >= 0 ? sums[diagonal] : 0n)
      );
      counts[index] =
        isValid[index] +
        (up >= 0 ? counts[up] : 0) +
        (left >= 0 ? counts[left] : 0) -
        (diagonal >= 0 ? counts[diagonal] : 0);
    }
  }
  return {width, height, quantum, anchor, quantized, isValid, sums, counts};
}

/** Exact modular sum and count over the inclusive in-raster box. */
export function queryReliefBox(
  model: ReliefSummedAreaModel,
  firstColumn: number,
  lastColumn: number,
  firstRow: number,
  lastRow: number
): {sum: bigint; count: number} {
  const at = (column: number, row: number) =>
    column < 0 || row < 0 ? undefined : row * model.width + column;
  const corner = (column: number, row: number, sign: bigint) => {
    const index = at(column, row);
    return {
      sum: index === undefined ? 0n : sign * model.sums[index],
      count: index === undefined ? 0 : Number(sign) * model.counts[index]
    };
  };
  const parts = [
    corner(lastColumn, lastRow, 1n),
    corner(firstColumn - 1, lastRow, -1n),
    corner(lastColumn, firstRow - 1, -1n),
    corner(firstColumn - 1, firstRow - 1, 1n)
  ];
  return {
    sum: wrap64(parts.reduce((total, part) => total + part.sum, 0n)),
    count: parts.reduce((total, part) => total + part.count, 0)
  };
}

/** WGSL `signed64ToFloat` in float32 arithmetic. */
export function signed64ToFloatModel(value: bigint): number {
  const negative = value >= MODULUS / 2n;
  const magnitude = negative ? wrap64(MODULUS - value) : value;
  const high = Number(magnitude >> 32n);
  const low = Number(magnitude & 0xffffffffn);
  const result = Math.fround(Math.fround(Math.fround(high) * 4294967296) + Math.fround(low));
  return negative ? -result : result;
}

/** Result of {@link modelReliefDeviation}. */
export type ReliefDeviationModel = {
  /** float32 deviation `z - mean`, NaN at invalid centres (before the exaggeration). */
  deviation: Float64Array;
  /** Exact `D = N q_c - S` per pixel as a signed integer (0n for invalid centres). */
  difference: bigint[];
  /** Exact weighted valid count `N` per pixel (0 for invalid centres). */
  count: number[];
};

/** Runs the kernel body for square windows, or optional per-row rectangular half-widths. */
export function modelReliefDeviation(
  elevation: ArrayLike<number>,
  validity: ArrayLike<number>,
  width: number,
  height: number,
  radius: number,
  quantum: number,
  rowRadii?: readonly (readonly [number, number])[]
): ReliefDeviationModel {
  const model = buildReliefSummedAreaModel(elevation, validity, width, height, quantum);
  const deviation = new Float64Array(width * height).fill(NaN);
  const difference = new Array<bigint>(width * height).fill(0n);
  const countOutput = new Array<number>(width * height).fill(0);
  for (let row = 0; row < height; row++) {
    const [radiusX, radiusY] = rowRadii?.[row] ?? [radius, radius];
    for (let column = 0; column < width; column++) {
      const index = row * width + column;
      const centre = queryReliefBox(model, column, column, row, row);
      if (centre.count !== 1) {
        continue;
      }
      const firstColumn = Math.max(column - radiusX, 0);
      const lastColumn = Math.min(column + radiusX, width - 1);
      const firstRow = Math.max(row - radiusY, 0);
      const lastRow = Math.min(row + radiusY, height - 1);
      const overhangLeft = Math.max(radiusX - column, 0);
      const overhangRight = Math.max(column + radiusX - (width - 1), 0);
      const overhangTop = Math.max(radiusY - row, 0);
      const overhangBottom = Math.max(row + radiusY - (height - 1), 0);
      let sum = 0n;
      let count = 0;
      const add = (box: {sum: bigint; count: number}, weight: number) => {
        sum = wrap64(sum + wrap64(box.sum * BigInt(weight)));
        count = (count + box.count * weight) >>> 0;
      };
      add(queryReliefBox(model, firstColumn, lastColumn, firstRow, lastRow), 1);
      if (overhangLeft > 0) add(queryReliefBox(model, 0, 0, firstRow, lastRow), overhangLeft);
      if (overhangRight > 0) {
        add(queryReliefBox(model, width - 1, width - 1, firstRow, lastRow), overhangRight);
      }
      if (overhangTop > 0) {
        add(queryReliefBox(model, firstColumn, lastColumn, 0, 0), overhangTop);
        if (overhangLeft > 0) add(queryReliefBox(model, 0, 0, 0, 0), overhangLeft * overhangTop);
        if (overhangRight > 0) {
          add(queryReliefBox(model, width - 1, width - 1, 0, 0), overhangRight * overhangTop);
        }
      }
      if (overhangBottom > 0) {
        add(queryReliefBox(model, firstColumn, lastColumn, height - 1, height - 1), overhangBottom);
        if (overhangLeft > 0) {
          add(queryReliefBox(model, 0, 0, height - 1, height - 1), overhangLeft * overhangBottom);
        }
        if (overhangRight > 0) {
          add(
            queryReliefBox(model, width - 1, width - 1, height - 1, height - 1),
            overhangRight * overhangBottom
          );
        }
      }
      const quantizedCentre = model.quantized[index];
      const difference64 = wrap64(BigInt(count) * BigInt(quantizedCentre) - sum);
      const scaled = Math.fround(
        Math.fround(Math.fround(elevation[index]) - model.anchor) * Math.fround(1 / quantum)
      );
      const residual = Math.fround(Math.fround(scaled - quantizedCentre) * Math.fround(quantum));
      const term = Math.fround(
        Math.fround(
          Math.fround(signed64ToFloatModel(difference64) / Math.fround(count)) *
            Math.fround(quantum)
        )
      );
      deviation[index] = Math.fround(residual + term);
      difference[index] = toSigned(difference64);
      countOutput[index] = count;
    }
  }
  return {deviation, difference, count: countOutput};
}

/**
 * Literal RVT window mean in float64: every window coordinate clamps to the grid
 * (`numpy.pad(mode='edge')`), invalid samples are excluded from the sum and the count. Returns
 * `z - mean` per pixel (NaN at invalid centres) and the integer window count.
 */
export function computeLiteralReliefDeviation(
  elevation: ArrayLike<number>,
  validity: ArrayLike<number>,
  width: number,
  height: number,
  radius: number
): {deviation: Float64Array; count: number[]} {
  const deviation = new Float64Array(width * height).fill(NaN);
  const countOutput = new Array<number>(width * height).fill(0);
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      const index = row * width + column;
      if (!validity[index]) continue;
      let sum = 0;
      let count = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const sample =
            Math.min(Math.max(row + dy, 0), height - 1) * width +
            Math.min(Math.max(column + dx, 0), width - 1);
          if (validity[sample]) {
            sum += Math.fround(elevation[sample]);
            count++;
          }
        }
      }
      deviation[index] = Math.fround(elevation[index]) - sum / count;
      countOutput[index] = count;
    }
  }
  return {deviation, count: countOutput};
}
