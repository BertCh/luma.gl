// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_EDGE_BUNDLING_DEFAULTS,
  GPU_EDGE_BUNDLING_WORK_BOX_PADDING,
  getGPUEdgeBundlingFixedPointExponent
} from '../../../src/gpu-network/edge-bundling';

const f = Math.fround;

const DEAD_LIMIT = -1.0e7;
const LENGTH_EPSILON = f(1e-9);
const GRADIENT_QUANTA = 4;
const RADIUS_EPSILON = f(1e-9);

/** Inputs of {@link bundleEdgesOracle}; mirrors the recipe props and per-frame parameters. */
export type EdgeBundlingOracleInput = {
  positions: Float32Array;
  sources: ArrayLike<number>;
  targets: ArrayLike<number>;
  mask?: ArrayLike<number>;
  pointsPerEdge: number;
  /** Compiled iteration maximum. */
  iterations: number;
  densityResolution: number;
  activeIterations?: number;
  kernelRadius?: number;
  lambda?: number;
  smoothing?: number;
  stepScale?: number;
};

/** Result of {@link bundleEdgesOracle}. */
export type EdgeBundlingOracleResult = {
  /** `edgeCount * pointsPerEdge * 2` floats in the caller's coordinates. */
  paths: Float32Array;
  /** Work box `[originX, originY, side]`. */
  box: [number, number, number];
  /** Fixed-point density of the last executed iteration, or empty. */
  density: Float64Array;
};

/**
 * CPU reference of `GPUEdgeBundling`: the same schedule, f32 rounding, fixed-point density and
 * work-box rule.
 *
 * The schedule is chaotic: points that sit near their own density peak or a saddle take a
 * full-radius step in a direction set by tiny differences, so ulp-level differences between this
 * oracle and the device (fused multiply-add, sqrt rounding) grow roughly 1000x per iteration. The
 * oracle itself moves 3e-3 of the box after two iterations when its inputs are nudged by 1e-7 of
 * the box. Parity is therefore tight (below 1e-5 of the box) for one iteration and statistical
 * afterwards.
 */
export function bundleEdgesOracle(input: EdgeBundlingOracleInput): EdgeBundlingOracleResult {
  const {positions, sources, targets, mask, pointsPerEdge: P, densityResolution: R} = input;
  const edgeCount = sources.length;
  const vertexCount = positions.length / 2;
  const exponent = getGPUEdgeBundlingFixedPointExponent(edgeCount * P);
  const fixedScale = 2 ** exponent;
  const fixedInverse = 2 ** -exponent;
  const live = (edge: number): boolean => {
    const s = sources[edge];
    const t = targets[edge];
    if (s >= vertexCount || t >= vertexCount) return false;
    if (mask && mask[edge] === 0) return false;
    return [positions[s * 2], positions[s * 2 + 1], positions[t * 2], positions[t * 2 + 1]].every(
      value => Math.abs(value) < 3.0e38
    );
  };

  // Work box.
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let anyLive = false;
  for (let edge = 0; edge < edgeCount; edge++) {
    if (!live(edge)) continue;
    anyLive = true;
    const s = sources[edge];
    const t = targets[edge];
    minX = Math.min(minX, positions[s * 2], positions[t * 2]);
    minY = Math.min(minY, positions[s * 2 + 1], positions[t * 2 + 1]);
    maxX = Math.max(maxX, positions[s * 2], positions[t * 2]);
    maxY = Math.max(maxY, positions[s * 2 + 1], positions[t * 2 + 1]);
  }
  let box: [number, number, number] = [0, 0, 1];
  if (anyLive) {
    const extent = Math.max(f(maxX - minX), f(maxY - minY));
    const side = extent > 0 ? f(extent * f(1 + 2 * GPU_EDGE_BUNDLING_WORK_BOX_PADDING)) : 1;
    box = [
      f(f(0.5 * f(minX + maxX)) - f(0.5 * side)),
      f(f(0.5 * f(minY + maxY)) - f(0.5 * side)),
      side
    ];
  }

  // Initialize.
  const work = new Float64Array(edgeCount * P * 2);
  for (let edge = 0; edge < edgeCount; edge++) {
    for (let i = 0; i < P; i++) {
      const index = (edge * P + i) * 2;
      if (!live(edge)) {
        work[index] = work[index + 1] = -1.0e8;
        continue;
      }
      const s = sources[edge];
      const t = targets[edge];
      const a = [0, 1].map(c => f(f(positions[s * 2 + c] - box[c]) / box[2]));
      const b = [0, 1].map(c => f(f(positions[t * 2 + c] - box[c]) / box[2]));
      const u = f(i / (P - 1));
      for (let c = 0; c < 2; c++) {
        work[index + c] = i === 0 ? a[c] : i === P - 1 ? b[c] : f(a[c] + f(f(b[c] - a[c]) * u));
      }
    }
  }

  // Per-frame parameters.
  const clamp = (value: number, low: number, high: number) => Math.min(Math.max(value, low), high);
  const active = Math.min(input.activeIterations ?? input.iterations, input.iterations);
  const initialRadius = Math.max(
    f(input.kernelRadius ?? GPU_EDGE_BUNDLING_DEFAULTS.kernelRadius),
    0
  );
  const lambda = clamp(f(input.lambda ?? GPU_EDGE_BUNDLING_DEFAULTS.lambda), 0.5, 0.9);
  const smoothing = clamp(f(input.smoothing ?? GPU_EDGE_BUNDLING_DEFAULTS.smoothing), 0, 1);
  const stepScale = Math.max(f(input.stepScale ?? GPU_EDGE_BUNDLING_DEFAULTS.stepScale), 0);

  const density = new Float64Array(R * R);
  const RF = f(R);
  let lastDensity = new Float64Array(0);
  for (let iteration = 0; iteration < input.iterations; iteration++) {
    if (iteration >= active) break;
    let radius = initialRadius;
    for (let round = 0; round < iteration; round++) radius = f(radius * lambda);
    if (!(radius > RADIUS_EPSILON)) break;
    density.fill(0);
    // Splat.
    const radiusCells = f(radius * RF);
    const reach = Math.max(1, Math.ceil(radiusCells));
    const inverseSquare = f(1 / f(radiusCells * radiusCells));
    for (let p = 0; p < edgeCount * P; p++) {
      const px = work[p * 2];
      const py = work[p * 2 + 1];
      if (px < DEAD_LIMIT) continue;
      const gx = f(px * RF);
      const gy = f(py * RF);
      const cx = Math.floor(gx);
      const cy = Math.floor(gy);
      for (let y = Math.max(cy - reach, 0); y <= Math.min(cy + reach, R - 1); y++) {
        const dy = f(f(y + 0.5) - gy);
        for (let x = Math.max(cx - reach, 0); x <= Math.min(cx + reach, R - 1); x++) {
          const dx = f(f(x + 0.5) - gx);
          const q = f(f(f(dx * dx) + f(dy * dy)) * inverseSquare);
          if (q < 1) {
            const weight = Math.floor(f(f(f(1 - q) * fixedScale) + 0.5));
            if (weight > 0) density[y * R + x] += weight;
          }
        }
      }
    }
    lastDensity = Float64Array.from(density);
    const readDensity = (x: number, y: number) => f(f(density[y * R + x]) * fixedInverse);
    const sample = (gx: number, gy: number): number => {
      const cx = clamp(gx, 0, R - 1);
      const cy = clamp(gy, 0, R - 1);
      const x0 = Math.floor(cx);
      const y0 = Math.floor(cy);
      const x1 = Math.min(x0 + 1, R - 1);
      const y1 = Math.min(y0 + 1, R - 1);
      const fx = f(cx - x0);
      const fy = f(cy - y0);
      const term = (value: number, wx: number, wy: number) => f(f(value * wx) * wy);
      return f(
        f(
          f(
            term(readDensity(x0, y0), f(1 - fx), f(1 - fy)) +
              term(readDensity(x1, y0), fx, f(1 - fy))
          ) + term(readDensity(x0, y1), f(1 - fx), fy)
        ) + term(readDensity(x1, y1), fx, fy)
      );
    };
    // Update per edge.
    const step = f(radius * stepScale);
    for (let edge = 0; edge < edgeCount; edge++) {
      const base = edge * P;
      if (work[base * 2] < DEAD_LIMIT) continue;
      const points: number[][] = [];
      for (let i = 0; i < P; i++) points.push([work[(base + i) * 2], work[(base + i) * 2 + 1]]);
      for (let i = 1; i < P - 1; i++) {
        const gx = f(f(points[i][0] * RF) - 0.5);
        const gy = f(f(points[i][1] * RF) - 0.5);
        if (gx < 1 || gy < 1 || gx >= RF - 2 || gy >= RF - 2) continue;
        const gradientX = f(sample(f(gx + 1), gy) - sample(f(gx - 1), gy));
        const gradientY = f(sample(gx, f(gy + 1)) - sample(gx, f(gy - 1)));
        const length = f(Math.sqrt(f(f(gradientX * gradientX) + f(gradientY * gradientY))));
        if (length < f(GRADIENT_QUANTA * fixedInverse)) continue;
        points[i] = [
          f(points[i][0] + f(f(gradientX / length) * step)),
          f(points[i][1] + f(f(gradientY / length) * step))
        ];
      }
      const segmentLengthOf = (i: number) => {
        const dx = f(points[i][0] - points[i - 1][0]);
        const dy = f(points[i][1] - points[i - 1][1]);
        return f(Math.sqrt(f(f(dx * dx) + f(dy * dy))));
      };
      let total = 0;
      for (let i = 1; i < P; i++) total = f(total + segmentLengthOf(i));
      let segment = 1;
      let consumed = 0;
      let segmentLength = segmentLengthOf(1);
      const resampled: number[][] = [];
      for (let i = 0; i < P; i++) {
        const arcTarget = f(f(total * i) / (P - 1));
        while (segment < P - 1 && f(consumed + segmentLength) < arcTarget) {
          consumed = f(consumed + segmentLength);
          segment++;
          segmentLength = segmentLengthOf(segment);
        }
        const fraction =
          segmentLength < LENGTH_EPSILON ? 0 : f(f(arcTarget - consumed) / segmentLength);
        const previous = points[segment - 1];
        resampled.push(
          [0, 1].map(c => f(previous[c] + f(f(points[segment][c] - previous[c]) * fraction)))
        );
      }
      resampled[0] = points[0];
      resampled[P - 1] = points[P - 1];
      for (let i = 0; i < P; i++) {
        let out = resampled[i];
        if (i > 0 && i < P - 1) {
          out = [0, 1].map(c => {
            const midpoint = f(
              f(0.5 * f(resampled[i - 1][c] + resampled[i + 1][c])) - resampled[i][c]
            );
            return f(resampled[i][c] + f(smoothing * midpoint));
          });
        }
        work[(base + i) * 2] = out[0];
        work[(base + i) * 2 + 1] = out[1];
      }
    }
  }

  // Finalize.
  const paths = new Float32Array(edgeCount * P * 2);
  for (let edge = 0; edge < edgeCount; edge++) {
    const s = sources[edge];
    for (let i = 0; i < P; i++) {
      const index = (edge * P + i) * 2;
      let point = [0, 0];
      if (!live(edge)) {
        if (s < vertexCount) point = [positions[s * 2], positions[s * 2 + 1]];
      } else if (i === 0) {
        point = [positions[s * 2], positions[s * 2 + 1]];
      } else if (i === P - 1) {
        const t = targets[edge];
        point = [positions[t * 2], positions[t * 2 + 1]];
      } else {
        point = [0, 1].map(c => f(box[c] + f(work[index + c] * box[2])));
      }
      paths[index] = point[0];
      paths[index + 1] = point[1];
    }
  }
  return {paths, box, density: lastDensity};
}
