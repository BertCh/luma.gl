// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getCardMarkAttributes, type CardMarkShape} from '../../encoding/card-tokens';
import {provenanceMarkAttributes} from '../../encoding/provenance';
import type {ViewContext} from '../../registry/view-registration';
import {h, type ViewElementNode} from '../../view-node';
import type {ArrayResultPanel} from './array-result-model';

/** Role, provenance, and delegated pick attributes for every captured data mark. */
export function getArrayMarkAttributes(
  panel: ArrayResultPanel,
  context: ViewContext,
  index: number,
  shape: CardMarkShape = 'cell',
  classed = false
): Record<string, string | number> {
  const value = panel.values[index];
  const attributes = getCardMarkAttributes(panel.role, shape, {
    ...(classed && panel.role === 'output'
      ? {palette: panel.palette, classIndex: panel.scale?.classFor(value) ?? 0}
      : {}),
    noData: !Number.isFinite(value)
  });
  const provenance = provenanceMarkAttributes(
    panel.provenance,
    attributes['fill'],
    context.idPrefix
  );
  return {
    ...attributes,
    ...provenance,
    ...(shape === 'line' ? {fill: 'none'} : {}),
    ...(classed && panel.role === 'input' && Number.isFinite(value)
      ? {'fill-opacity': 0.18 + 0.16 * (panel.scale?.classFor(value) ?? 2)}
      : {}),
    'data-capture': panel.capture.id,
    ...(panel.role === 'context'
      ? {}
      : {
          'data-action': 'pick',
          'data-axis': 'entity',
          'data-key': panel.first + index,
          'data-key-space': panel.capture.keySpace ?? ''
        })
  };
}

/** Muted card label; callers suppress labels entirely in thumbnails. */
export function renderArrayLabel(
  text: string,
  x: number,
  y: number,
  attributes: Record<string, string | number> = {}
): ViewElementNode {
  return h(
    'text',
    {class: 'cv-label', x, y, fill: '#5c636e', 'font-size': 11, ...attributes},
    text
  );
}

/** Context path, with metadata provenance unless backed by a context capture. */
export function renderArrayContextPath(
  path: string,
  context: ViewContext,
  attributes: Record<string, string | number> = {}
): ViewElementNode {
  return h('path', {
    ...getCardMarkAttributes('context', 'line'),
    ...provenanceMarkAttributes('metadata', '#d4d4d0', context.idPrefix),
    fill: 'none',
    d: path,
    ...attributes
  });
}
