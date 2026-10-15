// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Display-space colour match; the input PNGs are already tone-mapped RGB. */
export type CloudRepairStyle = {
  gain: number[];
  offset: number[];
  residualLimit: number;
  sampleCount: number;
};

const GUARD_RADIUS = 6;
const FRINGE_RADIUS = 48;
const STYLE_SAMPLE_COUNT = 16384;
const COARSE_CELL_SIZE = 32;

/** Chessboard distance, including diagonal cloud edges, in two linear passes. */
function getMaskDistance(mask: Uint8Array, width: number, height: number): Float32Array {
  const distance = new Float32Array(mask.length).fill(Infinity);
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      const cell = row * width + column;
      if (mask[cell]) distance[cell] = 0;
      else {
        if (column > 0) distance[cell] = Math.min(distance[cell], distance[cell - 1] + 1);
        if (row > 0) {
          for (
            let neighbor = Math.max(0, column - 1);
            neighbor <= Math.min(width - 1, column + 1);
            neighbor++
          ) {
            distance[cell] = Math.min(distance[cell], distance[(row - 1) * width + neighbor] + 1);
          }
        }
      }
    }
  }
  for (let row = height - 1; row >= 0; row--) {
    for (let column = width - 1; column >= 0; column--) {
      const cell = row * width + column;
      if (column + 1 < width) distance[cell] = Math.min(distance[cell], distance[cell + 1] + 1);
      if (row + 1 < height) {
        for (
          let neighbor = Math.max(0, column - 1);
          neighbor <= Math.min(width - 1, column + 1);
          neighbor++
        ) {
          distance[cell] = Math.min(distance[cell], distance[(row + 1) * width + neighbor] + 1);
        }
      }
    }
  }
  return distance;
}

function getMedian(values: number[]): number {
  values.sort((left, right) => left - right);
  return values[Math.floor(values.length / 2)] ?? 0;
}

function getResidual(
  target: Uint8Array,
  prior: Uint8Array,
  cell: number,
  style: CloudRepairStyle
): number {
  let residual = 0;
  for (let channel = 0; channel < 3; channel++) {
    const pixel = cell * 4 + channel;
    residual = Math.max(
      residual,
      Math.abs(
        target[pixel] / 255 - ((prior[pixel] / 255) * style.gain[channel] + style.offset[channel])
      )
    );
  }
  return residual;
}

/** Robust paired regression excludes the mask, its surroundings, saturation and RGB outliers. */
function fitStyle(target: Uint8Array, prior: Uint8Array, distance: Float32Array): CloudRepairStyle {
  const samples: number[] = [];
  const stride = Math.max(1, Math.ceil(distance.length / STYLE_SAMPLE_COUNT));
  for (let cell = 0; cell < distance.length; cell += stride) {
    if (distance[cell] <= GUARD_RADIUS * 2) continue;
    let valid = true;
    for (let channel = 0; channel < 3; channel++) {
      const pixel = cell * 4 + channel;
      valid &&= target[pixel] > 2 && target[pixel] < 253 && prior[pixel] > 2 && prior[pixel] < 253;
    }
    if (valid) samples.push(cell);
  }
  const style: CloudRepairStyle = {
    gain: [1, 1, 1],
    offset: [0, 0, 0],
    residualLimit: 0.04,
    sampleCount: samples.length
  };
  if (samples.length < 16) return style;
  for (let channel = 0; channel < 3; channel++) {
    style.offset[channel] = getMedian(
      samples.map(cell => (target[cell * 4 + channel] - prior[cell * 4 + channel]) / 255)
    );
  }
  for (let iteration = 0; iteration < 5; iteration++) {
    const errors = samples.map(cell => getResidual(target, prior, cell, style));
    const cutoff = Math.max(0.025, 4 * getMedian([...errors]));
    for (let channel = 0; channel < 3; channel++) {
      let weightSum = 0;
      let priorSum = 0;
      let targetSum = 0;
      let priorSquaredSum = 0;
      let productSum = 0;
      for (let sample = 0; sample < samples.length; sample++) {
        const ratio = errors[sample] / cutoff;
        const weight = ratio < 1 ? (1 - ratio * ratio) ** 2 : 0;
        const pixel = samples[sample] * 4 + channel;
        const before = prior[pixel] / 255;
        const after = target[pixel] / 255;
        weightSum += weight;
        priorSum += weight * before;
        targetSum += weight * after;
        priorSquaredSum += weight * before * before;
        productSum += weight * before * after;
      }
      if (weightSum < 16) continue;
      const variance = priorSquaredSum - (priorSum * priorSum) / weightSum;
      // Flat scenes only constrain an offset; avoid amplifying quantization noise.
      style.gain[channel] =
        variance > weightSum * 0.0001
          ? Math.max(0.5, Math.min(2, (productSum - (priorSum * targetSum) / weightSum) / variance))
          : 1;
      style.offset[channel] = (targetSum - style.gain[channel] * priorSum) / weightSum;
    }
  }
  style.residualLimit = Math.max(
    0.04,
    Math.min(0.12, 4 * getMedian(samples.map(cell => getResidual(target, prior, cell, style))))
  );
  return style;
}

/**
 * Match the prior to reliable current-day pixels, then seed local residuals on a coarse grid.
 * Masked target colours never enter the fit or residual field. Large holes are styled immediately.
 */
export function prepareCloudRepairStyle(
  target: Uint8Array,
  prior: Uint8Array,
  mask: Uint8Array,
  width: number,
  height: number
) {
  const originalDistance = getMaskDistance(mask, width, height);
  const style = fitStyle(target, prior, originalDistance);
  const repairMask = mask.slice();
  for (let cell = 0; cell < repairMask.length; cell++) {
    if (
      !mask[cell] &&
      originalDistance[cell] <= FRINGE_RADIUS &&
      getResidual(target, prior, cell, style) > style.residualLimit
    ) {
      repairMask[cell] = 3;
    }
  }
  const distance = getMaskDistance(repairMask, width, height);
  for (let cell = 0; cell < repairMask.length; cell++) {
    if (!repairMask[cell] && distance[cell] <= GUARD_RADIUS) repairMask[cell] = 3;
  }

  const coarseWidth = Math.ceil(width / COARSE_CELL_SIZE);
  const coarseHeight = Math.ceil(height / COARSE_CELL_SIZE);
  const coarseCount = coarseWidth * coarseHeight;
  const counts = new Uint32Array(coarseCount);
  let coarseResidual = new Float32Array(coarseCount * 3);
  let nextResidual = new Float32Array(coarseCount * 3);
  const priorValues = new Float32Array(mask.length * 3);
  for (let cell = 0; cell < mask.length; cell++) {
    const coarseCell =
      Math.floor(Math.floor(cell / width) / COARSE_CELL_SIZE) * coarseWidth +
      Math.floor((cell % width) / COARSE_CELL_SIZE);
    const reliable =
      distance[cell] > GUARD_RADIUS * 2 &&
      getResidual(target, prior, cell, style) <= style.residualLimit;
    if (reliable) counts[coarseCell]++;
    for (let channel = 0; channel < 3; channel++) {
      const matched =
        (prior[cell * 4 + channel] / 255) * style.gain[channel] + style.offset[channel];
      priorValues[cell * 3 + channel] = matched;
      if (reliable)
        coarseResidual[coarseCell * 3 + channel] += target[cell * 4 + channel] / 255 - matched;
    }
  }
  for (let cell = 0; cell < coarseCount; cell++) {
    for (let channel = 0; channel < 3; channel++)
      coarseResidual[cell * 3 + channel] /= Math.max(1, counts[cell]);
  }
  for (let iteration = 0; iteration < 128; iteration++) {
    for (let row = 0; row < coarseHeight; row++) {
      for (let column = 0; column < coarseWidth; column++) {
        const cell = row * coarseWidth + column;
        for (let channel = 0; channel < 3; channel++) {
          const offset = cell * 3 + channel;
          nextResidual[offset] = counts[cell]
            ? coarseResidual[offset]
            : (coarseResidual[(row * coarseWidth + Math.max(0, column - 1)) * 3 + channel] +
                coarseResidual[
                  (row * coarseWidth + Math.min(coarseWidth - 1, column + 1)) * 3 + channel
                ] +
                coarseResidual[(Math.max(0, row - 1) * coarseWidth + column) * 3 + channel] +
                coarseResidual[
                  (Math.min(coarseHeight - 1, row + 1) * coarseWidth + column) * 3 + channel
                ]) *
              0.25;
        }
      }
    }
    [coarseResidual, nextResidual] = [nextResidual, coarseResidual];
  }
  const state = new Float32Array(mask.length * 4);
  for (let row = 0; row < height; row++) {
    const coarseRow = Math.max(0, Math.min(coarseHeight - 1, (row + 0.5) / COARSE_CELL_SIZE - 0.5));
    const lowerRow = Math.floor(coarseRow);
    const rowFraction = coarseRow - lowerRow;
    for (let column = 0; column < width; column++) {
      const cell = row * width + column;
      const coarseColumn = Math.max(
        0,
        Math.min(coarseWidth - 1, (column + 0.5) / COARSE_CELL_SIZE - 0.5)
      );
      const lowerColumn = Math.floor(coarseColumn);
      const columnFraction = coarseColumn - lowerColumn;
      for (let channel = 0; channel < 3; channel++) {
        let residual = 0;
        for (let rowOffset = 0; rowOffset < 2; rowOffset++) {
          for (let columnOffset = 0; columnOffset < 2; columnOffset++) {
            const coarseCell =
              Math.min(coarseHeight - 1, lowerRow + rowOffset) * coarseWidth +
              Math.min(coarseWidth - 1, lowerColumn + columnOffset);
            residual +=
              coarseResidual[coarseCell * 3 + channel] *
              (rowOffset ? rowFraction : 1 - rowFraction) *
              (columnOffset ? columnFraction : 1 - columnFraction);
          }
        }
        state[cell * 4 + channel] = repairMask[cell]
          ? residual
          : target[cell * 4 + channel] / 255 - priorValues[cell * 3 + channel];
      }
      // This is an immutable observation flag, not a confidence channel to diffuse.
      state[cell * 4 + 3] = repairMask[cell] ? 0 : 1;
    }
  }
  return {style, repairMask, priorValues, state};
}
