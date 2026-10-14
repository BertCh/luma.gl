// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {OverlayOperand} from '../overlay-gpu-harness';

const STAR_VERTICES = 20;
const CELL_SPACING = 10;

type Scene = {left: OverlayOperand; right: OverlayOperand; perimeterBound: number; extent: number};

/** Deterministic grid of stars; no randomness, so both sides and every run see the same input. */
export function createOverlayABScene(vertexCount: number): Scene {
  const starCount = Math.max(1, Math.round(vertexCount / STAR_VERTICES));
  const columns = Math.ceil(Math.sqrt(starCount));
  const makeOperand = (offset: number, phase: number): OverlayOperand => {
    const operand: OverlayOperand = [];
    for (let star = 0; star < starCount; star++) {
      const centerX = (star % columns) * CELL_SPACING + offset;
      const centerY = Math.floor(star / columns) * CELL_SPACING + offset;
      const ring: [number, number][] = [];
      for (let vertex = 0; vertex < STAR_VERTICES; vertex++) {
        const radius = vertex % 2 === 0 ? 8.5 : 3.5;
        const angle = phase + (2 * Math.PI * vertex) / STAR_VERTICES;
        // 1/64 multiples: exact in f32 and a generic position relative to the lattice.
        ring.push([
          Math.round((centerX + radius * Math.cos(angle)) * 64) / 64,
          Math.round((centerY + radius * Math.sin(angle)) * 64) / 64
        ]);
      }
      operand.push([[ring]]);
    }
    return operand;
  };
  const left = makeOperand(0, 0.1);
  const right = makeOperand(CELL_SPACING / 2, 0.4);
  let perimeterBound = 0;
  for (const operand of [left, right]) {
    for (const feature of operand) {
      const ring = feature[0][0];
      ring.forEach((vertex, i) => {
        const next = ring[(i + 1) % ring.length];
        perimeterBound += Math.hypot(next[0] - vertex[0], next[1] - vertex[1]);
      });
    }
  }
  return {
    left,
    right,
    perimeterBound,
    extent: (columns + 1) * CELL_SPACING
  };
}
