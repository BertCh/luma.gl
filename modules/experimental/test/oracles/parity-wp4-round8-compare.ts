// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LICCase} from './parity-wp4-round2-fixtures';
import {compareRasterParity} from './parity-wp4-compare';

export type LICModeCase = LICCase & {lower: number[]; upper: number[]; ambiguousTaps: number[]};

/** Per-tap floor ambiguity follows the generator's gamma_k bound; L stays unchanged. */
export function compareLICMode(scene: LICModeCase, actual: ArrayLike<number>, prefix = '') {
  const projected = Float64Array.from(actual, (value, index) =>
    Math.max(scene.lower[index], Math.min(scene.upper[index], value))
  );
  compareRasterParity(
    'GPULineIntegralConvolution',
    `${prefix}${scene.name}`,
    scene.reference,
    'every-value-within-independent-f32-discrete-tap-interval',
    actual,
    projected,
    'L'
  );
  const unambiguous = scene.ambiguousTaps.flatMap((count, index) => (count === 0 ? [index] : []));
  compareRasterParity(
    'GPULineIntegralConvolution',
    `${prefix}${scene.name}`,
    scene.reference,
    'all-unambiguous-values',
    unambiguous.map(index => actual[index]),
    unambiguous.map(index => scene.values[index]),
    'L'
  );
}
