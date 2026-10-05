// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Float64 oracles written straight from the formulas of the Relief Visualization Toolbox (RVT,
 * Apache-2.0; Kokalj and Somrak 2019). They deliberately share no code with the GPU recipes:
 * the mean filter uses RVT's padded integral image, MSRM runs RVT's loop over every scale, local
 * dominance rolls the padded array, and blending follows `render_all_images` top-last.
 */

/** Python `round`: ties go to the nearest even integer. */
export function pythonRound(value: number): number {
  const floor = Math.floor(value);
  const difference = value - floor;
  if (difference !== 0.5) {
    return Math.round(value);
  }
  return floor % 2 === 0 ? floor : floor + 1;
}

/** `numpy.pad(mode='edge')` with `before` leading and `after` trailing samples on both axes. */
function padEdge(
  values: ArrayLike<number>,
  width: number,
  height: number,
  before: number,
  after: number
): {data: Float64Array; width: number; height: number} {
  const paddedWidth = width + before + after;
  const paddedHeight = height + before + after;
  const data = new Float64Array(paddedWidth * paddedHeight);
  for (let row = 0; row < paddedHeight; row++) {
    const sourceRow = Math.min(Math.max(row - before, 0), height - 1);
    for (let column = 0; column < paddedWidth; column++) {
      const sourceColumn = Math.min(Math.max(column - before, 0), width - 1);
      data[row * paddedWidth + column] = values[sourceRow * width + sourceColumn];
    }
  }
  return {data, width: paddedWidth, height: paddedHeight};
}

/** Summed-area table with the origin in the upper left corner (`rvt.vis.integral_image`). */
function integralImage(values: Float64Array, width: number, height: number): Float64Array {
  const result = new Float64Array(values.length);
  for (let row = 0; row < height; row++) {
    let rowTotal = 0;
    for (let column = 0; column < width; column++) {
      rowTotal += values[row * width + column];
      result[row * width + column] = rowTotal + (row > 0 ? result[(row - 1) * width + column] : 0);
    }
  }
  return result;
}

/**
 * RVT `mean_filter`: pad `(r + 1, r)` with edge values, treat NaN as 0 with a zero count, take
 * rolled integral-image differences, divide, and restore NaN at invalid centers. `numpy.roll`
 * wraps, so indices wrap here exactly as in numpy. Returned in float64 (RVT casts to float32).
 */
export function computeMeanFilterRVT(
  dem: ArrayLike<number>,
  width: number,
  height: number,
  radius: number
): Float64Array {
  if (radius === 0) {
    return Float64Array.from(dem as ArrayLike<number>);
  }
  const padded = padEdge(dem, width, height, radius + 1, radius);
  const {width: paddedWidth, height: paddedHeight} = padded;
  const counts = new Float64Array(padded.data.length);
  for (let index = 0; index < padded.data.length; index++) {
    if (Number.isNaN(padded.data[index])) {
      padded.data[index] = 0;
    } else {
      counts[index] = 1;
    }
  }
  const sums = integralImage(padded.data, paddedWidth, paddedHeight);
  const countImage = integralImage(counts, paddedWidth, paddedHeight);
  const wrap = (value: number, size: number) => ((value % size) + size) % size;
  // np.roll(a, shift)[i] = a[i - shift].
  const rolled = (image: Float64Array, rowShift: number, columnShift: number, i: number, j: number) =>
    image[wrap(i - rowShift, paddedHeight) * paddedWidth + wrap(j - columnShift, paddedWidth)];
  const window = (image: Float64Array, i: number, j: number) =>
    rolled(image, radius, radius, i, j) +
    rolled(image, -radius - 1, -radius - 1, i, j) -
    rolled(image, -radius - 1, radius, i, j) -
    rolled(image, radius, -radius - 1, i, j);
  const output = new Float64Array(width * height);
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      const i = row + radius;
      const j = column + radius;
      output[row * width + column] = Number.isNaN(dem[row * width + column])
        ? NaN
        : window(sums, i, j) / window(countImage, i, j);
    }
  }
  return output;
}

/** RVT `slrm`: `ve * dem - mean_filter(ve * dem)`. */
export function computeSimpleLocalReliefRVT(options: {
  width: number;
  height: number;
  elevation: ArrayLike<number>;
  radius: number;
  verticalExaggeration?: number;
}): number[] {
  const {width, height, radius} = options;
  const ve = options.verticalExaggeration ?? 1;
  const dem = Array.from(options.elevation, value => value * ve);
  const mean = computeMeanFilterRVT(dem, width, height, radius);
  return dem.map((value, index) => value - mean[index]);
}

/** RVT `msrm`, running the loop over every scale literally. */
export function computeMultiScaleReliefRVT(options: {
  width: number;
  height: number;
  elevation: ArrayLike<number>;
  resolution: number;
  featureMinimum: number;
  featureMaximum: number;
  scalingFactor: number;
  verticalExaggeration?: number;
}): number[] {
  const {width, height, resolution, featureMaximum} = options;
  const ve = options.verticalExaggeration ?? 1;
  const dem = Array.from(options.elevation, value => value * ve);
  const featureMinimum = Math.max(options.featureMinimum, resolution);
  const scalingFactor = Math.trunc(options.scalingFactor);
  const first = Math.floor(
    ((featureMinimum - resolution) / (2 * resolution)) ** (1 / scalingFactor)
  );
  const last = Math.ceil(((featureMaximum - resolution) / (2 * resolution)) ** (1 / scalingFactor));
  const sum = new Float64Array(dem.length);
  let count = 0;
  let previous: Float64Array | undefined;
  for (let k = first; k <= last; k++) {
    const surface = computeMeanFilterRVT(dem, width, height, k ** scalingFactor);
    if (k !== first) {
      for (let index = 0; index < sum.length; index++) {
        sum[index] += previous![index] - surface[index];
      }
      count++;
    }
    previous = surface;
  }
  return Array.from(sum, value => value / count);
}

/**
 * RVT `local_dominance` with the padded-array roll loop and Python rounding. Invalid centers are
 * NaN (RVT leaves them 0).
 */
export function computeLocalDominanceRVT(options: {
  width: number;
  height: number;
  elevation: ArrayLike<number>;
  minimumRadius?: number;
  maximumRadius?: number;
  radiusIncrement?: number;
  angularResolution?: number;
  observerHeight?: number;
  verticalExaggeration?: number;
}): number[] {
  const {width, height} = options;
  const minimum = options.minimumRadius ?? 10;
  const maximum = options.maximumRadius ?? 20;
  const increment = options.radiusIncrement ?? 1;
  const angular = options.angularResolution ?? 15;
  const observer = options.observerHeight ?? 1.7;
  const ve = options.verticalExaggeration ?? 1;
  const padded = padEdge(options.elevation, width, height, maximum, maximum);
  const dem = padded.data.map(value => value * ve);
  const {width: paddedWidth, height: paddedHeight} = padded;
  const distanceCount = Math.trunc((maximum - minimum) / increment + 1);
  const distances = Array.from({length: distanceCount}, (_, k) => minimum + increment * k);
  const angleCount = Math.trunc(359 / angular + 1);
  const angles = Array.from({length: angleCount}, (_, j) => angular * j);
  const norma =
    distances.reduce((total, d) => total + (observer / d) * (2 * d + increment), 0) * angleCount;
  const output = new Float64Array(dem.length);
  const wrap = (value: number, size: number) => ((value % size) + size) % size;
  for (const angle of angles) {
    const radians = (angle * Math.PI) / 180;
    for (const distance of distances) {
      const rowShift = pythonRound(Math.sin(radians) * distance);
      const columnShift = pythonRound(Math.cos(radians) * distance);
      const factor = 2 * distance + increment;
      for (let row = 0; row < paddedHeight; row++) {
        for (let column = 0; column < paddedWidth; column++) {
          const moved =
            dem[
              wrap(row - rowShift, paddedHeight) * paddedWidth + wrap(column - columnShift, paddedWidth)
            ];
          const here = dem[row * paddedWidth + column];
          if (here + observer > moved) {
            output[row * paddedWidth + column] += ((here + observer - moved) / distance) * factor;
          }
        }
      }
    }
  }
  const result: number[] = [];
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      result.push(
        Number.isNaN(options.elevation[row * width + column])
          ? NaN
          : output[(row + maximum) * paddedWidth + column + maximum] / norma
      );
    }
  }
  return result;
}

/** One RVT layer, as listed in a `BlenderCombination` (layer 1 is the TOP layer). */
export type RVTBlendLayer = {
  image: ArrayLike<number>;
  minimum: number;
  maximum: number;
  /** RVT inverts slope gradient and negative openness. */
  invert: boolean;
  blendMode: 'normal' | 'multiply' | 'screen' | 'overlay' | 'soft_light' | 'luminosity';
  /** 0 to 1. */
  opacity: number;
};

function normalizeLinear(value: number, minimum: number, maximum: number): number {
  const clipped = Math.min(Math.max(value, minimum), maximum);
  return Math.min(Math.max((clipped - minimum) / (maximum - minimum), 0), 1);
}

/** RVT's single-band blend equations (`blend_func.py`), without the in-place aliasing. */
export function blendRVT(mode: RVTBlendLayer['blendMode'], active: number, background: number): number {
  switch (mode) {
    case 'multiply':
      return active * background;
    case 'screen':
      return 1 - (1 - active) * (1 - background);
    case 'overlay':
      return background > 0.5
        ? 1 - (1 - 2 * (background - 0.5)) * (1 - active)
        : 2 * background * active;
    case 'soft_light':
      return active < 0.5
        ? 2 * background * active + background ** 2 * (1 - 2 * active)
        : 2 * background * (1 - active) + Math.sqrt(background) * (2 * active - 1);
    default:
      // normal, and luminosity of greyscale images (`lum(img)` is the image).
      return active;
  }
}

/**
 * RVT render loop (`BlenderCombination.render_all_images`): layers run from the LAST (bottom) to
 * the FIRST (top); the first rendered layer is used as is.
 *
 * @param layers RVT order, top layer first.
 */
export function computeBlendRVT(layers: readonly RVTBlendLayer[]): number[] {
  const pixelCount = layers[0].image.length;
  const rendered: number[] = new Array(pixelCount);
  for (let pixel = 0; pixel < pixelCount; pixel++) {
    let background = NaN;
    for (let layerIndex = layers.length - 1; layerIndex >= 0; layerIndex--) {
      const layer = layers[layerIndex];
      let value = normalizeLinear(layer.image[pixel], layer.minimum, layer.maximum);
      if (layer.invert) {
        value = 1 - value;
      }
      if (layerIndex === layers.length - 1) {
        background = value;
      } else {
        const top = blendRVT(layer.blendMode, value, background);
        background = top * layer.opacity + background * (1 - layer.opacity);
      }
    }
    rendered[pixel] = background;
  }
  return rendered;
}
