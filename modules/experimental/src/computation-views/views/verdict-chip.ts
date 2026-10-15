// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GuardVerdict} from '../guards/guards';
import {h} from '../view-node';
import type {ViewElementNode} from '../view-node';

/** Prints the status and reason as accessible text, independent of colour. */
export function renderVerdictChip(verdict: GuardVerdict): ViewElementNode {
  const text =
    verdict.status === 'pass'
      ? `✓ ${verdict.label}`
      : verdict.status === 'fail'
        ? `✕ ${verdict.label}: ${verdict.reason}`
        : `? ${verdict.label}: unknown — ${verdict.reason}`;
  return h(
    'span',
    {class: `lens-verdict lens-verdict--${verdict.status}`, role: 'note', title: verdict.reason},
    text
  );
}

/** Withholds only the misleading glyph, leaving surrounding content available. */
export function renderBlockedGlyph(verdict: GuardVerdict, glyphLabel: string): ViewElementNode {
  return h(
    'div',
    {class: 'lens-blocked-glyph', 'data-guard': verdict.guardId},
    `${glyphLabel} withheld `,
    renderVerdictChip(verdict)
  );
}
