// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** CPU oracle for `GPUSegmentIntersection`: exact BigInt arithmetic over f32 coordinates. */

export type OraclePoint = [number, number];

/** One linestring or polygon feature of a test scene. */
export type OracleSegmentFeature =
  | {kind: 'lines'; vertices: OraclePoint[]}
  | {kind: 'polygons'; polygons: OraclePoint[][][]};

export type OracleSegment = {
  id: number;
  a: OraclePoint;
  b: OraclePoint;
  ring: number;
  feature: number;
  successor: number;
};

export type OracleHit = {
  left: number;
  right: number;
  kind: 'proper' | 'touch' | 'collinearTouch' | 'overlap';
  point: OraclePoint;
  endPoint: OraclePoint;
  leftFeature: number;
  rightFeature: number;
  leftRing: number;
  rightRing: number;
};

export const KIND_CODES = {proper: 1, touch: 2, collinearTouch: 3, overlap: 4} as const;

/** Flat arrays in the GeoArrow layout consumed by the contributor. */
export type OracleArrays = {
  kind: 'lines' | 'polygons';
  positions: Float32Array;
  lineOffsets: Uint32Array;
  featureOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  ringOffsets: Uint32Array;
};

export function buildArrays(features: OracleSegmentFeature[]): OracleArrays {
  const kind = features[0]?.kind ?? 'lines';
  const positions: number[] = [];
  const lineOffsets = [0];
  const featureOffsets = [0];
  const polygonOffsets = [0];
  const ringOffsets = [0];
  for (const feature of features) {
    if (feature.kind === 'lines') {
      positions.push(...feature.vertices.flat());
      lineOffsets.push(positions.length / 2);
    } else {
      for (const polygon of feature.polygons) {
        for (const ring of polygon) {
          positions.push(...ring.flat());
          ringOffsets.push(positions.length / 2);
        }
        polygonOffsets.push(ringOffsets.length - 1);
      }
      featureOffsets.push(polygonOffsets.length - 1);
    }
  }
  return {
    kind,
    positions: Float32Array.from(positions),
    lineOffsets: Uint32Array.from(lineOffsets),
    featureOffsets: Uint32Array.from(featureOffsets),
    polygonOffsets: Uint32Array.from(polygonOffsets),
    ringOffsets: Uint32Array.from(ringOffsets)
  };
}

const f32 = (value: number) => Math.fround(value);

/** Exact scaled integer of an f32 value: value * 2^149. */
function toScaled(value: number): bigint {
  const view = new DataView(new ArrayBuffer(4));
  view.setFloat32(0, value);
  const bits = view.getUint32(0);
  const negative = bits >>> 31 === 1;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x7fffff;
  const magnitude =
    exponent === 0 ? BigInt(mantissa) : BigInt(mantissa | 0x800000) << BigInt(exponent - 1);
  return negative ? -magnitude : magnitude;
}

/** Exact sign of cross(b - a, c - a). */
export function orientExact(a: OraclePoint, b: OraclePoint, c: OraclePoint): number {
  const [ax, ay, bx, by, cx, cy] = [a[0], a[1], b[0], b[1], c[0], c[1]].map(toScaled);
  const determinant = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
  return determinant > 0n ? 1 : determinant < 0n ? -1 : 0;
}

type Ring = {vertices: OraclePoint[]; feature: number; closedLine: boolean; cyclic: boolean};

function isFinitePoint(point: OraclePoint): boolean {
  return Number.isFinite(point[0]) && Number.isFinite(point[1]);
}

/** Builds the segment list in vertex order: segment ID is the start-vertex index. */
export function buildSegments(features: OracleSegmentFeature[]): {
  segments: Map<number, OracleSegment>;
  vertexCount: number;
} {
  const rings: Ring[] = [];
  features.forEach((feature, featureRow) => {
    if (feature.kind === 'lines') {
      const {vertices} = feature;
      const first = vertices[0];
      const last = vertices[vertices.length - 1];
      rings.push({
        vertices,
        feature: featureRow,
        cyclic: false,
        closedLine: vertices.length >= 3 && first[0] === last[0] && first[1] === last[1]
      });
    } else {
      for (const polygon of feature.polygons) {
        for (const ring of polygon) {
          rings.push({vertices: ring, feature: featureRow, cyclic: true, closedLine: false});
        }
      }
    }
  });
  const segments = new Map<number, OracleSegment>();
  let vertexBase = 0;
  rings.forEach((ring, ringRow) => {
    const {vertices} = ring;
    const count = vertices.length;
    const minimum = ring.cyclic ? 3 : 2;
    if (count >= minimum) {
      const slotCount = ring.cyclic ? count : count - 1;
      const usable: number[] = [];
      for (let slot = 0; slot < slotCount; slot++) {
        const a = vertices[slot];
        const b = vertices[(slot + 1) % count];
        if (isFinitePoint(a) && isFinitePoint(b) && (a[0] !== b[0] || a[1] !== b[1])) {
          usable.push(slot);
        }
      }
      const wraps = ring.cyclic || ring.closedLine;
      usable.forEach((slot, position) => {
        let successor = -1;
        if (position + 1 < usable.length) {
          successor = vertexBase + usable[position + 1];
        } else if (wraps) {
          successor = vertexBase + usable[0];
        }
        segments.set(vertexBase + slot, {
          id: vertexBase + slot,
          a: vertices[slot],
          b: vertices[(slot + 1) % count],
          ring: ringRow,
          feature: ring.feature,
          successor
        });
      });
    }
    vertexBase += count;
  });
  return {segments, vertexCount: vertexBase};
}

type Classified = {
  kind: OracleHit['kind'];
  point: OraclePoint;
  endPoint: OraclePoint;
} | null;

function classify(a: OraclePoint, b: OraclePoint, c: OraclePoint, d: OraclePoint): Classified {
  if (
    Math.max(a[0], b[0]) < Math.min(c[0], d[0]) ||
    Math.max(c[0], d[0]) < Math.min(a[0], b[0]) ||
    Math.max(a[1], b[1]) < Math.min(c[1], d[1]) ||
    Math.max(c[1], d[1]) < Math.min(a[1], b[1])
  ) {
    return null;
  }
  const o1 = orientExact(a, b, c);
  const o2 = orientExact(a, b, d);
  const o3 = orientExact(c, d, a);
  const o4 = orientExact(c, d, b);
  if (o1 === 0 && o2 === 0 && o3 === 0 && o4 === 0) {
    const useX = a[0] !== b[0];
    const axis = (p: OraclePoint) => (useX ? p[0] : p[1]);
    const points = [a, b, c, d];
    const low = Math.max(Math.min(axis(a), axis(b)), Math.min(axis(c), axis(d)));
    const high = Math.min(Math.max(axis(a), axis(b)), Math.max(axis(c), axis(d)));
    if (low > high) {
      return null;
    }
    const at = (value: number) => points.find(p => axis(p) === value) as OraclePoint;
    return {
      kind: low === high ? 'collinearTouch' : 'overlap',
      point: at(low),
      endPoint: at(high)
    };
  }
  if (o1 * o2 > 0 || o3 * o4 > 0) {
    return null;
  }
  if (o1 !== 0 && o2 !== 0 && o3 !== 0 && o4 !== 0) {
    // Crossing point in double precision: the GPU rounds to f32, so compare with a tolerance.
    const rx = b[0] - a[0];
    const ry = b[1] - a[1];
    const sx = d[0] - c[0];
    const sy = d[1] - c[1];
    const t = ((c[0] - a[0]) * sy - (c[1] - a[1]) * sx) / (rx * sy - ry * sx);
    const point: OraclePoint = [a[0] + rx * t, a[1] + ry * t];
    return {kind: 'proper', point, endPoint: point};
  }
  const point = o1 === 0 ? c : o2 === 0 ? d : o3 === 0 ? a : b;
  return {kind: 'touch', point, endPoint: point};
}

/** Brute-force intersections of `left` with `right`, or with itself when `right` is omitted. */
export function intersectWithOracle(
  left: OracleSegmentFeature[],
  right?: OracleSegmentFeature[],
  options: {sameFeatureOnly?: boolean} = {}
): OracleHit[] {
  const leftSide = buildSegments(left);
  const rightSide = right ? buildSegments(right) : leftSide;
  const hits: OracleHit[] = [];
  const leftIds = [...leftSide.segments.keys()].sort((x, y) => x - y);
  const rightIds = [...rightSide.segments.keys()].sort((x, y) => x - y);
  for (const leftId of leftIds) {
    const l = leftSide.segments.get(leftId) as OracleSegment;
    for (const rightId of rightIds) {
      const r = rightSide.segments.get(rightId) as OracleSegment;
      if (!right && rightId <= leftId) {
        continue;
      }
      if (options.sameFeatureOnly && l.feature !== r.feature) {
        continue;
      }
      const result = classify(l.a, l.b, r.a, r.b);
      if (!result) {
        continue;
      }
      if (
        !right &&
        (result.kind === 'touch' || result.kind === 'collinearTouch') &&
        (l.successor === rightId || r.successor === leftId)
      ) {
        continue;
      }
      hits.push({
        left: leftId,
        right: rightId,
        ...result,
        leftFeature: l.feature,
        rightFeature: r.feature,
        leftRing: l.ring,
        rightRing: r.ring
      });
    }
  }
  return hits;
}

/** Tiny seeded generator (mulberry32) so failures reproduce. */
export function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Rounds to f32 so the oracle and the GPU see the same coordinates. */
export function roundPoint(point: OraclePoint): OraclePoint {
  return [f32(point[0]), f32(point[1])];
}
