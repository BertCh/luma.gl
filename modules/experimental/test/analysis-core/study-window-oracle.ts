// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {StudyWindowKernel} from '../../src/utils/study-window-wgsl';

/** Binary64 GeoArrow polygon fixture. Ring roles are positional, independent of winding. */
export type OracleStudyWindow = {
  positions: Float64Array;
  featureOffsets?: Uint32Array;
  polygonOffsets: Uint32Array;
  ringOffsets: Uint32Array;
};
/** Cartesian point in the operator's metre frame. */
export type OraclePoint = readonly [number, number];
/** Oriented edge of a non-degenerate ring. */
export type OracleEdge = readonly [number, number, number, number];
/** Bounds in minimum-x, minimum-y, maximum-x, maximum-y order. */
export type OracleBounds = readonly [number, number, number, number];

function makeWindow(polygons: number[][][]): OracleStudyWindow {
  const positions: number[] = [];
  const polygonOffsets = [0];
  const ringOffsets = [0];
  for (const polygon of polygons) {
    for (const ring of polygon) {
      positions.push(...ring);
      ringOffsets.push(positions.length / 2);
    }
    polygonOffsets.push(ringOffsets.length - 1);
  }
  return {
    positions: new Float64Array(positions),
    polygonOffsets: new Uint32Array(polygonOffsets),
    ringOffsets: new Uint32Array(ringOffsets),
    featureOffsets: new Uint32Array([0, polygons.length])
  };
}

/** 2000 by 1000 metre rectangle. */
export const ORACLE_RECTANGLE = makeWindow([[[0, 0, 2000, 0, 2000, 1000, 0, 1000]]]);
/** Re-entrant corner at (700, 700). */
export const ORACLE_L_SHAPE = makeWindow([
  [[0, 0, 2000, 0, 2000, 700, 700, 700, 700, 2000, 0, 2000]]
]);
/** Square with a central square hole. */
export const ORACLE_SQUARE_WITH_HOLE = makeWindow([
  [
    [0, 0, 2000, 0, 2000, 2000, 0, 2000],
    [700, 700, 1300, 700, 1300, 1300, 700, 1300]
  ]
]);
/** One feature with two disjoint polygons, separated by a 500 metre gap. */
export const ORACLE_TWO_ISLANDS = makeWindow([
  [[0, 0, 1000, 0, 1000, 1000, 0, 1000]],
  [[1500, 0, 2500, 0, 2500, 1000, 1500, 1000]]
]);
/** Clockwise shell and counterclockwise hole require both orientation reversals. */
export const ORACLE_CLOCKWISE_SHELL_CCW_HOLE = makeWindow([
  [
    [0, 0, 0, 2000, 2000, 2000, 2000, 0],
    [700, 700, 1300, 700, 1300, 1300, 700, 1300]
  ]
]);

/** Adds the empty and collapsed slots produced by polygon clipping without changing the window. */
export function addOracleDegenerateSlots(window: OracleStudyWindow): OracleStudyWindow {
  const minimumX = getOracleWindowBounds(window)[0];
  const vertexCount = window.positions.length / 2;
  const ringCount = window.ringOffsets.length - 1;
  return {
    positions: new Float64Array([
      ...window.positions,
      minimumX,
      -10,
      minimumX,
      20,
      minimumX,
      -5,
      minimumX,
      10,
      -50,
      -50,
      50,
      50,
      99,
      99,
      99,
      99,
      99,
      99
    ]),
    ringOffsets: new Uint32Array([
      ...window.ringOffsets,
      vertexCount,
      vertexCount + 4,
      vertexCount + 6,
      vertexCount + 9
    ]),
    // Empty polygon, then four polygons containing the empty and collapsed rings.
    polygonOffsets: new Uint32Array([
      ...window.polygonOffsets,
      ringCount,
      ringCount + 1,
      ringCount + 2,
      ringCount + 3,
      ringCount + 4
    ]),
    featureOffsets: new Uint32Array([0, window.polygonOffsets.length - 1 + 5])
  };
}

/** Normalizes interior to the left, omitting degenerate rings. */
export function getOracleOrientedEdges(window: OracleStudyWindow): OracleEdge[] {
  const edges: OracleEdge[] = [];
  for (let polygon = 0; polygon + 1 < window.polygonOffsets.length; polygon++) {
    for (
      let ring = window.polygonOffsets[polygon];
      ring < window.polygonOffsets[polygon + 1];
      ring++
    ) {
      const begin = window.ringOffsets[ring];
      const end = window.ringOffsets[ring + 1];
      const originX = window.positions[2 * begin];
      const originY = window.positions[2 * begin + 1];
      let nonzeroOrientations = 0;
      for (let vertex = begin + 1; vertex + 1 < end; vertex++) {
        const startX = window.positions[2 * vertex] - originX;
        const startY = window.positions[2 * vertex + 1] - originY;
        const endX = window.positions[2 * (vertex + 1)] - originX;
        const endY = window.positions[2 * (vertex + 1) + 1] - originY;
        nonzeroOrientations += Number(startX * endY - startY * endX !== 0);
      }
      if (end - begin < 3 || nonzeroOrientations === 0) {
        continue;
      }
      const edgeBegin = edges.length;
      let area = 0;
      for (let vertex = begin; vertex < end; vertex++) {
        const next = vertex + 1 === end ? begin : vertex + 1;
        const edge: OracleEdge = [
          window.positions[2 * vertex],
          window.positions[2 * vertex + 1],
          window.positions[2 * next],
          window.positions[2 * next + 1]
        ];
        edges.push(edge);
        area += edge[0] * edge[3] - edge[1] * edge[2];
      }
      const role = ring === window.polygonOffsets[polygon] ? 1 : -1;
      if (area * role < 0) {
        for (let vertex = edgeBegin; vertex < edges.length; vertex++) {
          const edge = edges[vertex];
          edges[vertex] = [edge[2], edge[3], edge[0], edge[1]];
        }
      }
    }
  }
  return edges;
}

/** Polygon area, subtracting holes by their positional role. */
export function getOracleWindowArea(window: OracleStudyWindow): number {
  return (
    getOracleOrientedEdges(window).reduce(
      (sum, edge) => sum + edge[0] * edge[3] - edge[1] * edge[2],
      0
    ) / 2
  );
}
/** Total boundary length including holes. */
export function getOracleWindowPerimeter(window: OracleStudyWindow): number {
  return getOracleOrientedEdges(window).reduce(
    (sum, edge) => sum + Math.hypot(edge[2] - edge[0], edge[3] - edge[1]),
    0
  );
}
/** Axis-aligned bounds of non-degenerate rings. */
export function getOracleWindowBounds(window: OracleStudyWindow): OracleBounds {
  const bounds: [number, number, number, number] = [Infinity, Infinity, -Infinity, -Infinity];
  for (const edge of getOracleOrientedEdges(window)) {
    bounds[0] = Math.min(bounds[0], edge[0]);
    bounds[1] = Math.min(bounds[1], edge[1]);
    bounds[2] = Math.max(bounds[2], edge[0]);
    bounds[3] = Math.max(bounds[3], edge[1]);
  }
  if (bounds[0] > bounds[2]) {
    return [NaN, NaN, NaN, NaN];
  }
  return bounds;
}
function isInsideEdges(edges: readonly OracleEdge[], point: OraclePoint): boolean {
  if (!point.every(Number.isFinite)) {
    return false;
  }
  let parity = false;
  for (const [startX, startY, endX, endY] of edges) {
    if (startY >= point[1] === endY >= point[1]) {
      continue;
    }
    const determinant =
      (endX - startX) * (point[1] - startY) - (endY - startY) * (point[0] - startX);
    const side =
      Math.sign(determinant) ||
      (endX !== startX ? -Math.sign(endX - startX) : Math.sign(endY - startY));
    if (side === Math.sign(endY - startY)) {
      parity = !parity;
    }
  }
  return parity;
}
/** Even-odd membership with the exact helper's downward/leftward SoS boundary convention. */
export function isOracleInsideWindow(window: OracleStudyWindow, point: OraclePoint): boolean {
  return isInsideEdges(getOracleOrientedEdges(window), point);
}
function edgeDistance(edge: OracleEdge, point: OraclePoint): number {
  const directionX = edge[2] - edge[0];
  const directionY = edge[3] - edge[1];
  const squaredLength = directionX ** 2 + directionY ** 2;
  const fraction =
    squaredLength > 0
      ? Math.max(
          0,
          Math.min(
            1,
            ((point[0] - edge[0]) * directionX + (point[1] - edge[1]) * directionY) / squaredLength
          )
        )
      : 0;
  return Math.hypot(
    point[0] - edge[0] - fraction * directionX,
    point[1] - edge[1] - fraction * directionY
  );
}
/** Nearest boundary distance, or infinity when every edge is beyond limit. */
export function getOracleBoundaryDistance(
  window: OracleStudyWindow,
  point: OraclePoint,
  limit = Infinity
): number {
  const distance = Math.min(
    ...getOracleOrientedEdges(window).map(edge => edgeDistance(edge, point))
  );
  return distance <= limit ? distance : Infinity;
}
function getFan(edge: OracleEdge, centre: OraclePoint) {
  const startX = edge[0] - centre[0];
  const startY = edge[1] - centre[1];
  const endX = edge[2] - centre[0];
  const endY = edge[3] - centre[1];
  const length = Math.hypot(endX - startX, endY - startY);
  const cross = startX * endY - startY * endX;
  const distance = length > 0 ? Math.abs(cross) / length : 0;
  const start = length > 0 ? (startX * (endX - startX) + startY * (endY - startY)) / length : 0;
  const end = start + length;
  return {
    distance,
    start,
    end,
    sign: Math.sign(cross),
    angle: cross === 0 ? 0 : Math.atan2(cross, startX * endX + startY * endY)
  };
}
function clampFraction(value: number): number {
  return Math.max(0, Math.min(1, value));
}

/** Signed fan sum, including the boundary angular limit and outside centres. */
export function getOracleCircleFractionFan(
  window: OracleStudyWindow,
  centre: OraclePoint,
  radius: number
): number {
  let total = 0;
  for (const edge of getOracleOrientedEdges(window)) {
    const fan = getFan(edge, centre);
    total += fan.angle;
    if (!(fan.distance > 0 && fan.distance < radius)) {
      continue;
    }
    const angle = Math.acos(Math.min(1, fan.distance / radius));
    total -=
      fan.sign *
      Math.max(
        0,
        Math.min(Math.atan2(fan.end, fan.distance), angle) -
          Math.max(Math.atan2(fan.start, fan.distance), -angle)
      );
  }
  return clampFraction(total / (2 * Math.PI));
}
/** Independent circle midpoint-angle membership integration. */
export function getOracleCircleFractionBrute(
  window: OracleStudyWindow,
  centre: OraclePoint,
  radius: number,
  angleCount = 1e6
): number {
  const edges = getOracleOrientedEdges(window);
  let count = 0;
  for (let index = 0; index < angleCount; index++) {
    const angle = ((index + 0.5) * 2 * Math.PI) / angleCount;
    count += Number(
      isInsideEdges(edges, [
        centre[0] + radius * Math.cos(angle),
        centre[1] + radius * Math.sin(angle)
      ])
    );
  }
  return count / angleCount;
}
/** Uncapped rectangle side-arc formula (the existing Ripley oracle caps its reciprocal at 100). */
export function getOracleRectangleIsotropicFraction(
  rectangle: OracleBounds,
  centre: OraclePoint,
  radius: number
): number {
  const angles = [
    centre[0] - rectangle[0],
    centre[1] - rectangle[1],
    rectangle[2] - centre[0],
    rectangle[3] - centre[1]
  ].map(distance =>
    radius > distance
      ? Math.atan2(Math.sqrt(Math.max(0, (radius - distance) * (radius + distance))), distance)
      : 0
  );
  let fraction = 1 - angles.reduce((sum, angle) => sum + angle, 0) / Math.PI;
  for (let side = 0; side < 4; side++) {
    fraction += Math.max(0, angles[side] + angles[(side + 1) % 4] - Math.PI / 2) / (2 * Math.PI);
  }
  return clampFraction(fraction);
}
function outsideMass(radius: number, kernel: StudyWindowKernel): number {
  return kernel === 'gaussian'
    ? Math.exp((-radius * radius) / 2)
    : Math.max(0, 1 - radius * radius) ** (kernel === 'quartic' ? 3 : 2);
}
function integrateAdaptive(
  evaluate: (value: number) => number,
  low: number,
  high: number,
  tolerance = 1e-12
): number {
  const middle = (low + high) / 2;
  const first = evaluate(low);
  const centre = evaluate(middle);
  const last = evaluate(high);
  const whole = ((high - low) * (first + 4 * centre + last)) / 6;
  function refine(
    begin: number,
    end: number,
    left: number,
    mid: number,
    right: number,
    estimate: number,
    error: number,
    depth: number
  ): number {
    const centrePoint = (begin + end) / 2;
    const leftMid = evaluate((begin + centrePoint) / 2);
    const rightMid = evaluate((centrePoint + end) / 2);
    const leftIntegral = ((centrePoint - begin) * (left + 4 * leftMid + mid)) / 6;
    const rightIntegral = ((end - centrePoint) * (mid + 4 * rightMid + right)) / 6;
    const difference = leftIntegral + rightIntegral - estimate;
    if (depth === 0 || Math.abs(difference) <= 15 * error) {
      return leftIntegral + rightIntegral + difference / 15;
    }
    return (
      refine(begin, centrePoint, left, leftMid, mid, leftIntegral, error / 2, depth - 1) +
      refine(centrePoint, end, mid, rightMid, right, rightIntegral, error / 2, depth - 1)
    );
  }
  return high > low ? refine(low, high, first, centre, last, whole, tolerance, 40) : 0;
}

// Positive nodes and weights of 16-point Gauss-Legendre. Deliberately independent of the WGSL source.
const GAUSS_NODES = [
  0.09501250983763744, 0.2816035507792589, 0.4580167776572274, 0.6178762444026438,
  0.755404408355003, 0.8656312023878318, 0.9445750230732326, 0.9894009349916499
];
const GAUSS_WEIGHTS = [
  0.1894506104550685, 0.1826034150449236, 0.16915651939500254, 0.14959598881657673,
  0.12462897125553387, 0.09515851168249278, 0.06225352393864789, 0.027152459411754096
];
function gaussianCorrection(ratio: number, low: number, high: number): number {
  let total = 0;
  // Split at p/h as well as each half-bandwidth. Subtracting the constant integrand removes
  // the near-pi/2 boundary layer when p/h is tiny; the remaining integral is smooth in x.
  for (let panel = 0; panel < 10; panel++) {
    const begin = Math.max(low, panel / 2);
    const end = Math.min(high, (panel + 1) / 2);
    const split = Math.max(begin, Math.min(end, ratio));
    for (const [left, right] of [
      [begin, split],
      [split, end]
    ]) {
      if (!(right > left)) {
        continue;
      }
      const middle = (left + right) / 2;
      const half = (right - left) / 2;
      for (let node = 0; node < 8; node++) {
        for (const sign of [-1, 1]) {
          const value = middle + sign * half * GAUSS_NODES[node];
          total +=
            (half * GAUSS_WEIGHTS[node] * (1 - Math.exp((-value * value) / 2)) * ratio) /
            (ratio * ratio + value * value);
        }
      }
    }
  }
  return total;
}
function schemeIntegral(
  ratio: number,
  start: number,
  end: number,
  kernel: StudyWindowKernel
): number {
  const radius = kernel === 'gaussian' ? 5 : 1;
  if (!(ratio < radius)) {
    return 0;
  }
  const extent = Math.sqrt(Math.max(0, radius * radius - ratio * ratio));
  const low = Math.max(start, -extent);
  const high = Math.min(end, extent);
  if (!(high > low)) {
    return 0;
  }
  const angle = Math.atan2(high, ratio) - Math.atan2(low, ratio);
  if (kernel === 'gaussian') {
    const correction =
      gaussianCorrection(ratio, Math.max(0, low), Math.max(0, high)) +
      gaussianCorrection(ratio, Math.max(0, -high), Math.max(0, -low));
    return Math.exp((-ratio * ratio) / 2) * (angle - correction);
  }
  const squaredRatio = ratio * ratio;
  const polynomial = (value: number) =>
    kernel === 'epanechnikov'
      ? (-2 + squaredRatio) * value + value ** 3 / 3
      : (-3 + 3 * squaredRatio - squaredRatio ** 2) * value +
        (1 - (2 * squaredRatio) / 3) * value ** 3 -
        value ** 5 / 5;
  return angle + ratio * (polynomial(high) - polynomial(low));
}
function kernelFan(
  window: OracleStudyWindow,
  centre: OraclePoint,
  bandwidth: number,
  kernel: StudyWindowKernel,
  scheme: boolean
): number {
  let total = 0;
  for (const edge of getOracleOrientedEdges(window)) {
    const fan = getFan(edge, centre);
    total += fan.angle;
    if (!(fan.distance > 0)) {
      continue;
    }
    const ratio = fan.distance / bandwidth;
    const supportAngle = kernel === 'gaussian' ? Math.PI / 2 : Math.acos(Math.min(1, ratio));
    const low = Math.max(-supportAngle, Math.atan2(fan.start, fan.distance));
    const high = Math.min(supportAngle, Math.atan2(fan.end, fan.distance));
    // Split at the maximum and support endpoints so Simpson cannot miss a narrow compact lobe.
    const split = Math.max(low, Math.min(high, 0));
    const evaluate = (angle: number) => outsideMass(ratio / Math.cos(angle), kernel);
    const integral = scheme
      ? schemeIntegral(ratio, fan.start / bandwidth, fan.end / bandwidth, kernel)
      : integrateAdaptive(evaluate, low, split, 5e-13) +
        integrateAdaptive(evaluate, split, high, 5e-13);
    total -= fan.sign * integral;
  }
  return clampFraction(total / (2 * Math.PI));
}
/** Exact (untruncated Gaussian) fan mass, adaptive Simpson in angle to 1e-12. */
export function getOracleKernelMassFan(
  window: OracleStudyWindow,
  centre: OraclePoint,
  bandwidth: number,
  kernel: StudyWindowKernel
): number {
  return kernelFan(window, centre, bandwidth, kernel, false);
}
/** Binary64 mirror of the WGSL's compact closed forms and fixed Gaussian quadrature. */
export function getOracleKernelMassScheme(
  window: OracleStudyWindow,
  centre: OraclePoint,
  bandwidth: number,
  kernel: StudyWindowKernel
): number {
  return kernelFan(window, centre, bandwidth, kernel, true);
}
/** Independent polar integration: ray intersections give exact radial mass per angular midpoint. */
export function getOracleKernelMassBrute(
  window: OracleStudyWindow,
  centre: OraclePoint,
  bandwidth: number,
  kernel: StudyWindowKernel
): number {
  const edges = getOracleOrientedEdges(window);
  const angleCount = 262144;
  let total = 0;
  for (let index = 0; index < angleCount; index++) {
    const angle = ((index + 0.5) * 2 * Math.PI) / angleCount;
    const directionX = Math.cos(angle);
    const directionY = Math.sin(angle);
    const crossings = [0];
    for (const edge of edges) {
      const edgeX = edge[2] - edge[0];
      const edgeY = edge[3] - edge[1];
      const offsetX = edge[0] - centre[0];
      const offsetY = edge[1] - centre[1];
      const cross = directionX * edgeY - directionY * edgeX;
      if (cross === 0) {
        continue;
      }
      const distance = (offsetX * edgeY - offsetY * edgeX) / cross;
      const fraction = (offsetX * directionY - offsetY * directionX) / cross;
      if (distance > 0 && fraction >= 0 && fraction < 1) {
        crossings.push(distance);
      }
    }
    crossings.sort((first, second) => first - second);
    for (let interval = 0; interval + 1 < crossings.length; interval++) {
      const low = crossings[interval];
      const high = crossings[interval + 1];
      const middle = (low + high) / 2;
      if (
        isInsideEdges(edges, [centre[0] + middle * directionX, centre[1] + middle * directionY])
      ) {
        total += outsideMass(low / bandwidth, kernel) - outsideMass(high / bandwidth, kernel);
      }
    }
  }
  return total / angleCount;
}
/** Fine midpoint raster estimate of the area at least radius from the boundary. */
export function getOracleErodedArea(
  window: OracleStudyWindow,
  radius: number,
  resolution: number
): number {
  const bounds = getOracleWindowBounds(window);
  const edges = getOracleOrientedEdges(window);
  const width = (bounds[2] - bounds[0]) / resolution;
  const height = (bounds[3] - bounds[1]) / resolution;
  let count = 0;
  for (let row = 0; row < resolution; row++) {
    for (let column = 0; column < resolution; column++) {
      const point: OraclePoint = [
        bounds[0] + (column + 0.5) * width,
        bounds[1] + (row + 0.5) * height
      ];
      if (isInsideEdges(edges, point) && edges.every(edge => edgeDistance(edge, point) >= radius)) {
        count++;
      }
    }
  }
  return count * width * height;
}
/** Fine midpoint raster per row-major quadrat (resolution samples on each quadrat axis). */
export function getOracleQuadratAreas(
  window: OracleStudyWindow,
  bounds: OracleBounds,
  quadratGrid: readonly [number, number],
  resolution: number
): number[] {
  const edges = getOracleOrientedEdges(window);
  const width = (bounds[2] - bounds[0]) / quadratGrid[0];
  const height = (bounds[3] - bounds[1]) / quadratGrid[1];
  return Array.from({length: quadratGrid[0] * quadratGrid[1]}, (_, quadrat) => {
    let count = 0;
    for (let row = 0; row < resolution; row++) {
      for (let column = 0; column < resolution; column++) {
        count += Number(
          isInsideEdges(edges, [
            bounds[0] + ((quadrat % quadratGrid[0]) + (column + 0.5) / resolution) * width,
            bounds[1] + (Math.floor(quadrat / quadratGrid[0]) + (row + 0.5) / resolution) * height
          ])
        );
      }
    }
    return (count * width * height) / resolution ** 2;
  });
}
