// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared assertions of the snap-rounded noding specs (executed on the CPU and on the GPU): a run of
// GPUSnapRoundedNoding must equal the BigInt reference piece for piece and satisfy the
// snap-rounding guarantee.

import {expect} from 'vitest';
import {
  checkSnapRoundingGuarantee,
  referenceLatticeFrame,
  referenceSnapRound,
  type ReferencePiece
} from './snap-rounding-reference';

/** What a spec reads back from one run of `GPUSnapRoundedNoding`. */
export type NodingRun = {
  /** Input vertices (x, y pairs, float32). */
  positions: Float32Array;
  /** The input lines, vertex by vertex (NaN allowed). */
  lines: readonly (readonly (readonly [number, number])[])[];
  gridSize: number;
  /** `output.lattice`: origin x, origin y, cell, 0. */
  lattice: number[];
  count: number;
  required: number;
  candidateOverflow: number;
  uncertain: number;
  /** Pieces in output order (lattice endpoints and source segment). */
  pieces: ReferencePiece[];
  /** `output.endpoints` rows, four floats per piece. */
  worldRows: Float32Array;
};

/** Compares a run with the BigInt reference and checks the snap-rounding guarantee. */
export function expectNodingMatchesReference(result: NodingRun, label: string) {
  const frame = referenceLatticeFrame(result.positions, result.gridSize);
  let row = 0;
  const referenceLines = result.lines.map(line => {
    const points = frame.points.slice(row, row + line.length);
    row += line.length;
    return points;
  });
  const reference = referenceSnapRound(referenceLines);
  expect(result.lattice, `${label} lattice`).toEqual([...frame.origin, frame.cell, 0]);
  expect(result.pieces, `${label} pieces`).toEqual(reference.pieces);
  expect(result.count, `${label} count`).toBe(reference.pieces.length);
  expect(result.required, `${label} required`).toBe(reference.pieces.length);
  expect(result.candidateOverflow, `${label} candidate overflow`).toBe(0);
  expect(result.uncertain, `${label} uncertain`).toBe(0);
  expect(
    checkSnapRoundingGuarantee(result.pieces, reference.hotPixels),
    `${label} guarantee`
  ).toEqual([]);
  // World endpoints: origin + k * cell, rounded once to float32.
  result.pieces.forEach((piece, index) => {
    const world = [
      frame.origin[0] + piece.from[0] * frame.cell,
      frame.origin[1] + piece.from[1] * frame.cell,
      frame.origin[0] + piece.to[0] * frame.cell,
      frame.origin[1] + piece.to[1] * frame.cell
    ].map(Math.fround);
    expect(
      Array.from(result.worldRows.slice(4 * index, 4 * index + 4)),
      `${label} world ${index}`
    ).toEqual(world);
  });
  return reference;
}
