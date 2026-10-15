// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ProvisionalState} from '../types';
import {h, type ViewElementNode} from '../view-node';

/** SVG blur filters encoding provisional and unconverged state without changing colour. */
export function crispnessDefs(idPrefix: string): ViewElementNode {
  return h(
    'defs',
    null,
    h('filter', {id: `${idPrefix}-provisional`}, h('feGaussianBlur', {stdDeviation: 1.2})),
    h('filter', {id: `${idPrefix}-unconverged`}, h('feGaussianBlur', {stdDeviation: 2.4}))
  );
}

/** Crispness-only attributes: final marks retain their original sharpness and fill. */
export function crispnessAttributes(
  state: ProvisionalState,
  idPrefix: string
): Record<string, string> {
  return state === 'final' ? {} : {filter: `url(#${idPrefix}-${state})`, 'data-provisional': state};
}

/** Coarsens text/table precision to two provisional or one unconverged significant digits. */
export function coarsenValues(values: ArrayLike<number>, state: ProvisionalState): number[] {
  return Array.from(values, value =>
    state === 'final' || !Number.isFinite(value)
      ? value
      : Number(value.toPrecision(state === 'provisional' ? 2 : 1))
  );
}
