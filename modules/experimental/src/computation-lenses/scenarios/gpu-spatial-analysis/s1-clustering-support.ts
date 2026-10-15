// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPULocalFrame} from '../../../analysis-core/contracts/local-frame';
import {createS2ChicagoFrame, projectS2Positions} from './s2-lens-run';

/** iNaturalist observations of Chicago's North Side, projected to local metres. */
export type NaturePoints = {
  /** Local equal-distance frame the positions were projected with. */
  frame: GPULocalFrame;
  /** Interleaved `[x, y]` metres, one pair per kept observation. */
  positions: Float32Array;
  /** Padded `[minX, minY, maxX, maxY]` extent of the positions, in metres. */
  extent: [number, number, number, number];
  /** Kept observations. */
  count: number;
  /** Zero-based day of 2023 of each kept observation. */
  days: Uint32Array;
  /** Local wall-clock hour of each kept observation. */
  hours: Uint32Array;
};

/**
 * Loads `s2-chicago-north-nature.json` and keeps every `step`-th observation, projected with the
 * azimuthal-equidistant North Side frame. `step` above one only shrinks the unrolled graphs of the
 * hierarchical clustering; the captions say so.
 */
export async function loadNaturePoints(options: {step?: number; padding?: number} = {}) {
  const step = options.step ?? 1;
  assert(Number.isInteger(step) && step >= 1);
  const nature = (await import('../../data/s2-chicago-north-nature.json')).default;
  const total = nature.positions.length / 2;
  const kept: number[] = [];
  for (let observation = 0; observation < total; observation += step) kept.push(observation);
  const frame = createS2ChicagoFrame();
  const projected = projectS2Positions(
    frame,
    Float64Array.from(
      kept.flatMap(observation => [
        nature.positions[2 * observation],
        nature.positions[2 * observation + 1]
      ])
    ),
    options.padding ?? 250
  );
  return {
    frame,
    positions: projected.positions,
    extent: projected.extent,
    count: kept.length,
    days: Uint32Array.from(kept, observation => nature.day[observation]),
    hours: Uint32Array.from(kept, observation => nature.hour[observation])
  } satisfies NaturePoints;
}

/**
 * Event times of the observations as little-endian Int64 words of epoch milliseconds, the wall-clock
 * day and hour read as UTC (the source holds local wall-clock time without a timezone).
 */
export function getNatureEventTimes(points: NaturePoints): Uint32Array {
  const words = new Uint32Array(points.count * 2);
  for (let index = 0; index < points.count; index++) {
    const milliseconds = Date.UTC(2023, 0, 1 + points.days[index], points.hours[index]);
    words[2 * index] = milliseconds % 2 ** 32;
    words[2 * index + 1] = Math.floor(milliseconds / 2 ** 32);
  }
  return words;
}

/** Calendar month (0 to 11) of every observation, the CPU twin of the recipe's month slice. */
export function getNatureMonths(points: NaturePoints): Uint32Array {
  return Uint32Array.from(points.days, day => new Date(Date.UTC(2023, 0, 1 + day)).getUTCMonth());
}

/** An 800 m lattice over the observation extent: columns, rows and the exact covered bounds. */
export function getNatureLattice(
  points: NaturePoints,
  cellSize = 800
): {width: number; height: number; cellSize: number; bounds: [number, number, number, number]} {
  const [minX, minY, maxX, maxY] = points.extent;
  const width = Math.ceil((maxX - minX) / cellSize);
  const height = Math.ceil((maxY - minY) / cellSize);
  return {
    width,
    height,
    cellSize,
    bounds: [minX, minY, minX + width * cellSize, minY + height * cellSize]
  };
}

/**
 * Observation counts per lattice cell and calendar month (`cell * 12 + month`, the cube layout of
 * `GPUEmergingHotSpots`). Binned on the CPU: it is the input table of the emerging hot spot card.
 */
export function getNatureCube(
  points: NaturePoints,
  lattice: ReturnType<typeof getNatureLattice>
): Uint32Array {
  const months = getNatureMonths(points);
  const cube = new Uint32Array(lattice.width * lattice.height * 12);
  for (let index = 0; index < points.count; index++) {
    const column = Math.floor((points.positions[2 * index] - lattice.bounds[0]) / lattice.cellSize);
    const row = Math.floor(
      (points.positions[2 * index + 1] - lattice.bounds[1]) / lattice.cellSize
    );
    cube[(row * lattice.width + column) * 12 + months[index]]++;
  }
  return cube;
}
