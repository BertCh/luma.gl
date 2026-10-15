// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ViewContext} from '../../registry/view-registration';
import {getCardMarkAttributes} from '../../encoding/card-tokens';
import {h, type ViewElementNode} from '../../view-node';
import {formatViewNumber} from '../../views/record-queries';
import {getArrayTokenValue, type ArrayResultPanel} from './array-result-model';
import {
  getArrayMarkAttributes,
  renderArrayLabel,
  renderArrayContextPath
} from './array-render-utils';

/** Placement of a reusable SVG table within a card figure. */
export type ArrayTableLayout = {x: number; y: number; width: number};

/**
 * Reusable dataframe table renderer: same-role columns, six recorded rows, explicit omissions.
 * Each numbered row is one compound mark; cells retain their capture binding and provenance.
 */
export function renderArrayTable(
  panels: readonly ArrayResultPanel[],
  context: ViewContext,
  layout: ArrayTableLayout = {x: 16, y: 32, width: 608}
): ViewElementNode {
  const role = panels[0]?.role ?? 'context';
  const first = Math.min(...panels.map(panel => panel.first));
  const capturedEnd = Math.max(first, ...panels.map(panel => panel.first + panel.values.length));
  const totalEnd = Math.max(capturedEnd, ...panels.map(panel => panel.total ?? capturedEnd));
  const visibleCount = Math.min(6, capturedEnd - first);
  const columnWidth = layout.width / Math.max(1, panels.length);
  const children: ViewElementNode[] = [];
  if (role === 'output') {
    children.push(
      h('rect', {
        ...getCardMarkAttributes('output', 'area'),
        'data-provenance': 'metadata',
        x: layout.x,
        y: layout.y - 20,
        width: layout.width,
        height: 24 + visibleCount * 28,
        fill: '#0072b2',
        'fill-opacity': 0.1,
        stroke: 'none'
      })
    );
    children.push(
      renderArrayContextPath(`M${layout.x},${layout.y - 20}h${layout.width}`, context, {
        stroke: '#0072b2',
        'stroke-width': 1.5
      })
    );
  }
  children.push(
    h(
      'g',
      {role: 'row'},
      panels.map((panel, column) =>
        renderArrayLabel(
          `${getArrayTokenValue(panel.tokens, 'operator') ?? ''} ${getArrayTokenValue(panel.tokens, 'column')?.replace(/_/g, ' ') || panel.capture.name}`.trim(),
          layout.x + column * columnWidth + 4,
          layout.y - 5,
          {role: 'columnheader'}
        )
      )
    )
  );
  for (let row = 0; row < visibleCount; row++) {
    children.push(
      h(
        'g',
        {role: 'row', 'data-key': first + row, 'data-numbered': 'true'},
        panels.map((panel, column) => {
          const index = first + row - panel.first;
          const value = panel.values[index];
          return h(
            'g',
            {...getArrayMarkAttributes(panel, context, index), role: 'cell'},
            renderArrayLabel(
              Number.isFinite(value) ? formatViewNumber(value) : 'unknown',
              layout.x + column * columnWidth + 4,
              layout.y + row * 28 + 18
            )
          );
        })
      )
    );
  }
  if (totalEnd > first + visibleCount)
    children.push(
      renderArrayLabel(
        `… ${totalEnd - first - visibleCount} more rows`,
        layout.x + 4,
        layout.y + visibleCount * 28 + 18
      )
    );
  if (!visibleCount)
    children.push(renderArrayLabel('no captured rows', layout.x + 4, layout.y + 18));
  return h(
    'g',
    {
      role: 'table',
      class: `cv-array-table cv-array-table--${role}`,
      'aria-label': `${role} columns`
    },
    children
  );
}
