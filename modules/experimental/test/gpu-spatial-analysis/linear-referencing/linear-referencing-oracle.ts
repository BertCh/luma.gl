// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Flat lines used by the linear-referencing tests (f32-rounded values). */
export type ReferenceLines = {
  positions: Float32Array;
  lineOffsets: Uint32Array;
};

/** f64 projection of one point onto one segment. */
export type SegmentProjection = {
  line: number;
  segmentRow: number;
  segmentIndex: number;
  fraction: number;
  foot: [number, number];
  distance: number;
  measure: number;
  side: number;
};

/** Returns f64 cumulative measures per vertex row (`NaN` outside lines). */
export function getVertexMeasures(lines: ReferenceLines): number[] {
  const rowCount = lines.positions.length / 2;
  const measures = new Array<number>(rowCount).fill(NaN);
  for (let line = 0; line + 1 < lines.lineOffsets.length; line++) {
    const start = lines.lineOffsets[line];
    const end = lines.lineOffsets[line + 1];
    let measure = 0;
    for (let row = start; row < end; row++) {
      if (row > start) {
        measure += Math.hypot(
          lines.positions[2 * row] - lines.positions[2 * row - 2],
          lines.positions[2 * row + 1] - lines.positions[2 * row - 1]
        );
      }
      measures[row] = measure;
    }
  }
  return measures;
}

/** Projects `point` onto the segment starting at `segmentRow` (f64). */
export function projectOntoSegment(
  lines: ReferenceLines,
  measures: number[],
  point: readonly number[],
  segmentRow: number
): SegmentProjection {
  const {positions, lineOffsets} = lines;
  const ax = positions[2 * segmentRow];
  const ay = positions[2 * segmentRow + 1];
  const dx = positions[2 * segmentRow + 2] - ax;
  const dy = positions[2 * segmentRow + 3] - ay;
  const rx = point[0] - ax;
  const ry = point[1] - ay;
  const lengthSquared = dx * dx + dy * dy;
  const fraction =
    lengthSquared > 0 ? Math.min(1, Math.max(0, (rx * dx + ry * dy) / lengthSquared)) : 0;
  const distance = Math.hypot(rx - dx * fraction, ry - dy * fraction);
  const cross = dx * ry - dy * rx;
  let line = 0;
  while (lineOffsets[line + 1] <= segmentRow) {
    line++;
  }
  return {
    line,
    segmentRow,
    segmentIndex: segmentRow - lineOffsets[line],
    fraction,
    foot: [ax + dx * fraction, ay + dy * fraction],
    distance,
    measure: measures[segmentRow] + fraction * Math.sqrt(lengthSquared),
    side: distance > 0 ? Math.sign(cross) || 0 : 0
  };
}

/** Brute-force nearest segment within `radius`; ties go to the smallest segment row. */
export function findNearestSegment(
  lines: ReferenceLines,
  measures: number[],
  point: readonly number[],
  radius: number
): SegmentProjection | null {
  let best: SegmentProjection | null = null;
  for (let line = 0; line + 1 < lines.lineOffsets.length; line++) {
    for (let row = lines.lineOffsets[line]; row + 1 < lines.lineOffsets[line + 1]; row++) {
      const projection = projectOntoSegment(lines, measures, point, row);
      if (projection.distance <= radius && (!best || projection.distance < best.distance)) {
        best = projection;
      }
    }
  }
  return best;
}

/** f64 reference of `GPULineLocate` for one event. */
export function locateAlong(
  lines: ReferenceLines,
  measures: number[],
  line: number,
  eventMeasure: number,
  options: {fraction?: boolean; offset?: number} = {}
): {position: [number, number]; segmentIndex: number; tangent: [number, number]; status: number} {
  const lineCount = lines.lineOffsets.length - 1;
  const start = line < lineCount ? lines.lineOffsets[line] : 0;
  const end = line < lineCount ? lines.lineOffsets[line + 1] : 0;
  if (end <= start) {
    return {position: [NaN, NaN], segmentIndex: 0xffffffff, tangent: [NaN, NaN], status: 2};
  }
  const total = measures[end - 1];
  let measure = options.fraction ? eventMeasure * total : eventMeasure;
  let status = 0;
  if (measure < 0 || measure > total) {
    measure = Math.min(Math.max(measure, 0), total);
    status = 1;
  }
  const {positions} = lines;
  let position: [number, number] = [positions[2 * start], positions[2 * start + 1]];
  let tangent: [number, number] = [0, 0];
  let segmentIndex = 0;
  if (end - start >= 2) {
    let upper = start + 1;
    while (upper < end && measures[upper] <= measure) {
      upper++;
    }
    const row = Math.min(upper, end - 1) - 1;
    const dx = positions[2 * row + 2] - positions[2 * row];
    const dy = positions[2 * row + 3] - positions[2 * row + 1];
    const segmentMeasure = measures[row + 1] - measures[row];
    const fraction =
      segmentMeasure > 0 ? Math.min(1, Math.max(0, (measure - measures[row]) / segmentMeasure)) : 0;
    position = [positions[2 * row] + dx * fraction, positions[2 * row + 1] + dy * fraction];
    const length = Math.hypot(dx, dy);
    tangent = length > 0 ? [dx / length, dy / length] : [0, 0];
    segmentIndex = row - start;
  }
  const offset = options.offset ?? 0;
  return {
    position: [position[0] - tangent[1] * offset, position[1] + tangent[0] * offset],
    segmentIndex,
    tangent,
    status
  };
}
