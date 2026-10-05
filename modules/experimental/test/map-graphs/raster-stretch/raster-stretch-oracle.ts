// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

const fround = Math.fround;

/** CPU description of a {@link GPURasterStretch} input. */
export type RasterStretchScene = {
  values: Float32Array;
  width: number;
  height: number;
  noDataValue?: number;
  validity?: Uint32Array;
  regionMask?: Uint32Array;
  binCount: number;
  lutSize: number;
};

/** Statistics and tables derived from a scene and a packed parameter vector. */
export type RasterStretchStatistics = {
  /** Included valid finite cells (window and mask applied). */
  validCount: number;
  domainMin: number;
  domainMax: number;
  /** Bin scale `binCount / (domainMax - domainMin)` in f32, 0 for an empty range. */
  scale: number;
  binWidth: number;
  histogram: Uint32Array;
  cdf: Uint32Array;
  /** Cells inside the domain (last CDF value). */
  total: number;
  lo: number;
  hi: number;
  /** True for an automatic domain without included cells (statistics report NaN). */
  isEmpty: boolean;
};

/** True when a cell is nodata (NaN, sentinel, or zero validity). */
export function isNoDataOnCPU(scene: RasterStretchScene, cell: number): boolean {
  const value = scene.values[cell];
  return (
    Number.isNaN(value) ||
    (scene.noDataValue !== undefined && value === fround(scene.noDataValue)) ||
    (scene.validity !== undefined && scene.validity[cell] === 0)
  );
}

/** Returns the bin of an in-domain value, replicating the f32 arithmetic of the GPU kernel. */
export function getBinOnCPU(
  value: number,
  stats: RasterStretchStatistics,
  binCount: number
): number {
  const position = fround(fround(value - stats.domainMin) * stats.scale);
  return Math.min(Math.floor(position), binCount - 1);
}

/**
 * Returns the percentile bound: first bin with positive cumulative count reaching `target`, then
 * linear interpolation inside it. Evaluated in f64 (the GPU uses f32 division).
 */
export function getPercentileValueOnCPU(
  target: number,
  cdf: Uint32Array,
  domainMin: number,
  domainMax: number,
  binWidth: number
): number {
  let bin = cdf.length - 1;
  for (let index = 0; index < cdf.length; index++) {
    if (cdf[index] > 0 && fround(cdf[index]) >= target) {
      bin = index;
      break;
    }
  }
  const before = bin > 0 ? cdf[bin - 1] : 0;
  const count = cdf[bin] - before;
  const fraction = count > 0 ? Math.min(Math.max((target - before) / count, 0), 1) : 0;
  return Math.min(Math.max(domainMin + (bin + fraction) * binWidth, domainMin), domainMax);
}

/** Computes domain, histogram, CDF and stretch bounds (steps 1 to 5 of the pipeline). */
export function computeRasterStretchStatistics(
  scene: RasterStretchScene,
  parameters: Float32Array
): RasterStretchStatistics {
  const {width, height, binCount} = scene;
  const column0 = Math.min(parameters[0], width);
  const row0 = Math.min(parameters[1], height);
  const column1 = Math.min(parameters[2], width);
  const row1 = Math.min(parameters[3], height);
  const included: number[] = [];
  for (let cell = 0; cell < width * height; cell++) {
    const column = cell % width;
    const row = Math.floor(cell / width);
    if (column < column0 || column >= column1 || row < row0 || row >= row1) {
      continue;
    }
    if (scene.regionMask && scene.regionMask[cell] === 0) {
      continue;
    }
    if (isNoDataOnCPU(scene, cell) || !Number.isFinite(scene.values[cell])) {
      continue;
    }
    included.push(cell);
  }
  const explicit = parameters[4] !== 0;
  let domainMin = 0;
  let domainMax = 0;
  if (explicit) {
    domainMin = parameters[5];
    domainMax = parameters[6];
  } else if (included.length > 0) {
    domainMin = Infinity;
    domainMax = -Infinity;
    for (const cell of included) {
      domainMin = Math.min(domainMin, scene.values[cell]);
      domainMax = Math.max(domainMax, scene.values[cell]);
    }
  }
  const range = fround(domainMax - domainMin);
  const hasRange = domainMax > domainMin;
  const stats: RasterStretchStatistics = {
    validCount: included.length,
    domainMin,
    domainMax,
    scale: hasRange ? fround(binCount / range) : 0,
    binWidth: hasRange ? fround(range / binCount) : 0,
    histogram: new Uint32Array(binCount),
    cdf: new Uint32Array(binCount),
    total: 0,
    lo: domainMin,
    hi: domainMax,
    isEmpty: !explicit && included.length === 0
  };
  for (const cell of included) {
    const value = scene.values[cell];
    if (value < domainMin || value > domainMax) {
      continue;
    }
    stats.histogram[getBinOnCPU(value, stats, binCount)]++;
  }
  let running = 0;
  for (let bin = 0; bin < binCount; bin++) {
    running += stats.histogram[bin];
    stats.cdf[bin] = running;
  }
  stats.total = running;
  const mode = parameters[7];
  if (mode !== 0 && stats.total === 0) {
    stats.hi = stats.lo;
  } else if (mode === 1) {
    stats.lo = getPercentileValueOnCPU(
      fround(parameters[8] * stats.total),
      stats.cdf,
      domainMin,
      domainMax,
      stats.binWidth
    );
    stats.hi = getPercentileValueOnCPU(
      fround(parameters[9] * stats.total),
      stats.cdf,
      domainMin,
      domainMax,
      stats.binWidth
    );
  }
  return stats;
}

/** Applies mode, gamma and sigmoid to one finite or infinite value (f64 math on f32 inputs). */
export function stretchValueOnCPU(
  value: number,
  parameters: Float32Array,
  stats: RasterStretchStatistics,
  binCount: number
): number {
  const {lo, hi} = stats;
  if (hi <= lo) {
    return value < lo ? 0 : value > hi ? 1 : 0.5;
  }
  let unit: number;
  if (parameters[7] === 2) {
    if (value <= stats.domainMin) {
      unit = 0;
    } else if (value >= stats.domainMax) {
      unit = 1;
    } else {
      const position = fround(fround(value - stats.domainMin) * stats.scale);
      const bin = Math.min(Math.floor(position), binCount - 1);
      const fraction = Math.min(Math.max(position - bin, 0), 1);
      const before = bin > 0 ? stats.cdf[bin - 1] : 0;
      const count = stats.cdf[bin] - before;
      unit = (before + fraction * count) / stats.total;
    }
  } else {
    unit = Math.min(Math.max((value - lo) / (hi - lo), 0), 1);
  }
  const gamma = parameters[10];
  if (gamma !== 1 && unit > 0 && unit < 1) {
    unit = unit ** gamma;
  }
  const contrast = parameters[11];
  if (contrast > 0) {
    const midpoint = parameters[12];
    const low = 1 / (1 + Math.exp(contrast * midpoint));
    const high = 1 / (1 + Math.exp(contrast * (midpoint - 1)));
    if (high - low > 0) {
      unit = (1 / (1 + Math.exp(contrast * (midpoint - unit))) - low) / (high - low);
    }
    unit = Math.min(Math.max(unit, 0), 1);
  }
  return unit;
}

/** Stretches every cell (NaN for nodata). */
export function stretchRasterOnCPU(
  scene: RasterStretchScene,
  parameters: Float32Array,
  stats: RasterStretchStatistics
): Float32Array {
  const output = new Float32Array(scene.width * scene.height);
  for (let cell = 0; cell < output.length; cell++) {
    output[cell] = isNoDataOnCPU(scene, cell)
      ? NaN
      : stretchValueOnCPU(scene.values[cell], parameters, stats, scene.binCount);
  }
  return output;
}

/** Samples the stretch curve at `lutSize` values across `[lo, hi]`. */
export function computeRasterStretchLutOnCPU(
  scene: RasterStretchScene,
  parameters: Float32Array,
  stats: RasterStretchStatistics
): Float32Array {
  const lut = new Float32Array(scene.lutSize);
  for (let index = 0; index < scene.lutSize; index++) {
    const value = stats.lo + ((index + 0.5) / scene.lutSize) * (stats.hi - stats.lo);
    lut[index] = stretchValueOnCPU(value, parameters, stats, scene.binCount);
  }
  return lut;
}

/** Palette lookup of a stretched value; NaN gives 0 (transparent). Linear rounds half up. */
export function getPaletteColorOnCPU(t: number, palette: Uint32Array, linear: boolean): number {
  if (Number.isNaN(t)) {
    return 0;
  }
  const size = palette.length;
  if (!linear || size === 1) {
    return palette[Math.min(Math.floor(fround(t * size)), size - 1)];
  }
  const position = fround(t * (size - 1));
  const lower = Math.min(Math.floor(position), size - 2);
  const fraction = position - lower;
  let color = 0;
  for (let channel = 0; channel < 4; channel++) {
    const first = (palette[lower] >>> (8 * channel)) & 255;
    const second = (palette[lower + 1] >>> (8 * channel)) & 255;
    const value = Math.min(Math.max(Math.floor(first + (second - first) * fraction + 0.5), 0), 255);
    color |= value << (8 * channel);
  }
  return color >>> 0;
}

/** Packs an rgba8 colour as `r | g << 8 | b << 16 | a << 24`. */
export function packColor(red: number, green: number, blue: number, alpha: number): number {
  return (red | (green << 8) | (blue << 16) | (alpha << 24)) >>> 0;
}
