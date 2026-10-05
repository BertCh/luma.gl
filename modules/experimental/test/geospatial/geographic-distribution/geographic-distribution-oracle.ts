// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Inputs of {@link computeGeographicDistribution}, mirroring `GPUGeographicDistribution`. */
export type GeographicDistributionOracleInput = {
  /** Interleaved `[x, y]` pairs. */
  positions: ArrayLike<number>;
  weights?: ArrayLike<number>;
  groupIds?: ArrayLike<number>;
  groupCount: number;
  mask?: ArrayLike<number>;
  /** Interleaved `[x, y]` line ends. */
  lineEnds?: ArrayLike<number>;
  origin?: readonly [number, number];
  standardDeviations?: number;
  ellipseConvention?: 'arcgis' | 'standard';
  orientationOnly?: boolean;
  medianTolerance?: number;
  medianIterations?: number;
  vertexCount?: number;
};

/** Per-group results of {@link computeGeographicDistribution}; empty groups hold NaN. */
export type GeographicDistributionOracleResult = {
  counts: number[];
  weightSums: number[];
  meanCenters: number[];
  medianCenters: number[];
  medianConverged: number[];
  standardDistances: number[];
  ellipses: number[];
  directionalMeans: number[];
  ellipseVertices: number[];
  circleVertices: number[];
};

const COINCIDENT_DISTANCE_SQUARED = 1e-12;

/**
 * Float64 reference implementation of the documented geographic distribution definitions: sequential
 * sums, two-pass central moments, fixed-iteration Weiszfeld median and circular statistics.
 */
export function computeGeographicDistribution(
  input: GeographicDistributionOracleInput
): GeographicDistributionOracleResult {
  const {groupCount} = input;
  const rowCount = input.positions.length / 2;
  const [originX, originY] = input.origin ?? [0, 0];
  const multiplier = input.standardDeviations ?? 1;
  const scale = (input.ellipseConvention ?? 'arcgis') === 'arcgis' ? Math.SQRT2 : 1;
  const iterations = input.medianIterations ?? 24;
  const tolerance = input.medianTolerance ?? 1e-3;
  const vertexCount = input.vertexCount ?? 64;
  const result: GeographicDistributionOracleResult = {
    counts: new Array(groupCount).fill(0),
    weightSums: new Array(groupCount).fill(0),
    meanCenters: new Array(groupCount * 2).fill(NaN),
    medianCenters: new Array(groupCount * 2).fill(NaN),
    medianConverged: new Array(groupCount).fill(0),
    standardDistances: new Array(groupCount).fill(NaN),
    ellipses: new Array(groupCount * 3).fill(NaN),
    directionalMeans: new Array(groupCount * 3).fill(NaN),
    ellipseVertices: new Array(groupCount * vertexCount * 2).fill(NaN),
    circleVertices: new Array(groupCount * vertexCount * 2).fill(NaN)
  };

  for (let group = 0; group < groupCount; group++) {
    const rows: {x: number; y: number; w: number; row: number}[] = [];
    for (let row = 0; row < rowCount; row++) {
      const x = input.positions[row * 2] - originX;
      const y = input.positions[row * 2 + 1] - originY;
      const w = input.weights ? input.weights[row] : 1;
      const rowGroup = input.groupIds ? input.groupIds[row] : 0;
      const valid =
        Number.isFinite(x) &&
        Number.isFinite(y) &&
        Number.isFinite(w) &&
        w > 0 &&
        rowGroup === group &&
        (!input.mask || input.mask[row] !== 0);
      if (valid) {
        rows.push({x, y, w, row});
      }
    }
    result.counts[group] = rows.length;
    const total = rows.reduce((sum, r) => sum + r.w, 0);
    result.weightSums[group] = total;
    if (!(total > 0)) {
      continue;
    }
    const meanX = rows.reduce((sum, r) => sum + r.w * r.x, 0) / total;
    const meanY = rows.reduce((sum, r) => sum + r.w * r.y, 0) / total;
    result.meanCenters[group * 2] = meanX + originX;
    result.meanCenters[group * 2 + 1] = meanY + originY;
    const varianceX = rows.reduce((sum, r) => sum + r.w * (r.x - meanX) ** 2, 0) / total;
    const varianceY = rows.reduce((sum, r) => sum + r.w * (r.y - meanY) ** 2, 0) / total;
    const covariance =
      rows.reduce((sum, r) => sum + r.w * (r.x - meanX) * (r.y - meanY), 0) / total;
    const standardDistance = multiplier * Math.sqrt(Math.max(varianceX + varianceY, 0));
    result.standardDistances[group] = standardDistance;

    const halfSum = 0.5 * (varianceX + varianceY);
    const halfDifference = 0.5 * (varianceX - varianceY);
    const radius = Math.hypot(halfDifference, covariance);
    const majorAngle = radius > 0 ? 0.5 * Math.atan2(covariance, halfDifference) : 0;
    let angle = majorAngle + Math.PI / 2;
    if (angle > Math.PI / 2) {
      angle -= Math.PI;
    }
    const sigmaX = multiplier * scale * Math.sqrt(Math.max(halfSum - radius, 0));
    const sigmaY = multiplier * scale * Math.sqrt(Math.max(halfSum + radius, 0));
    result.ellipses.splice(group * 3, 3, angle, sigmaX, sigmaY);

    for (let vertex = 0; vertex < vertexCount; vertex++) {
      const t = (2 * Math.PI * vertex) / vertexCount;
      const localX = sigmaX * Math.cos(t);
      const localY = sigmaY * Math.sin(t);
      const offset = (group * vertexCount + vertex) * 2;
      result.ellipseVertices[offset] =
        meanX + originX + localX * Math.cos(angle) - localY * Math.sin(angle);
      result.ellipseVertices[offset + 1] =
        meanY + originY + localX * Math.sin(angle) + localY * Math.cos(angle);
      result.circleVertices[offset] = meanX + originX + standardDistance * Math.cos(t);
      result.circleVertices[offset + 1] = meanY + originY + standardDistance * Math.sin(t);
    }

    let medianX = meanX;
    let medianY = meanY;
    let lastMove = 0;
    for (let iteration = 0; iteration < iterations; iteration++) {
      let sumWeight = 0;
      let sumX = 0;
      let sumY = 0;
      for (const r of rows) {
        const dx = r.x - medianX;
        const dy = r.y - medianY;
        const distanceSquared = dx * dx + dy * dy;
        if (distanceSquared > COINCIDENT_DISTANCE_SQUARED) {
          const inverseWeight = r.w / Math.sqrt(distanceSquared);
          sumWeight += inverseWeight;
          sumX += inverseWeight * dx;
          sumY += inverseWeight * dy;
        }
      }
      const stepX = sumWeight > 0 ? sumX / sumWeight : 0;
      const stepY = sumWeight > 0 ? sumY / sumWeight : 0;
      medianX += stepX;
      medianY += stepY;
      lastMove = Math.hypot(stepX, stepY);
    }
    result.medianCenters[group * 2] = medianX + originX;
    result.medianCenters[group * 2 + 1] = medianY + originY;
    result.medianConverged[group] = lastMove <= tolerance ? 1 : 0;

    if (input.lineEnds) {
      let cosine = 0;
      let sine = 0;
      let length = 0;
      let lineWeight = 0;
      for (const r of rows) {
        const dx = input.lineEnds[r.row * 2] - input.positions[r.row * 2];
        const dy = input.lineEnds[r.row * 2 + 1] - input.positions[r.row * 2 + 1];
        const lengthSquared = dx * dx + dy * dy;
        if (!Number.isFinite(lengthSquared) || !(lengthSquared > 0)) {
          continue;
        }
        const lineLength = Math.sqrt(lengthSquared);
        lineWeight += r.w;
        length += r.w * lineLength;
        if (input.orientationOnly) {
          cosine += (r.w * (dx * dx - dy * dy)) / lengthSquared;
          sine += (r.w * 2 * dx * dy) / lengthSquared;
        } else {
          cosine += (r.w * dx) / lineLength;
          sine += (r.w * dy) / lineLength;
        }
      }
      if (lineWeight > 0) {
        const resultant = Math.min(Math.hypot(cosine, sine) / lineWeight, 1);
        const hasAngle = cosine !== 0 || sine !== 0;
        let meanAngle = Math.atan2(sine, cosine);
        if (input.orientationOnly) {
          meanAngle *= 0.5;
        }
        result.directionalMeans.splice(
          group * 3,
          3,
          hasAngle ? meanAngle : NaN,
          1 - resultant,
          length / lineWeight
        );
      }
    }
  }
  return result;
}
