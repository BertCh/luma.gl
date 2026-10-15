// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {AxisHeaderModel} from '../registry/view-registration';
import {h} from '../view-node';
import type {ViewElementNode, ViewNode} from '../view-node';

/** Renders the accessible axis header; pins expose delegated free-axis actions. */
export function renderAxisHeader(model: AxisHeaderModel): ViewElementNode {
  const children: ViewNode[] = [];
  for (const chip of model.chips) {
    if (children.length) children.push(' · ');
    children.push(
      chip.state === 'bound'
        ? h(
            'button',
            {
              type: 'button',
              class: 'lens-axis-chip lens-axis-chip--bound',
              'data-action': 'free',
              'data-axis': chip.axis,
              'aria-label': `Free ${chip.axis}`
            },
            `${chip.label} ${chip.valueText} pinned`
          )
        : h(
            'span',
            {class: `lens-axis-chip lens-axis-chip--${chip.state}`},
            chip.state === 'free'
              ? `${chip.label} free (${chip.valueText})`
              : `${chip.label} ${chip.valueText}`
          )
    );
  }
  if (children.length) children.push(' · ');
  children.push(h('span', {class: 'lens-axis-adapter'}, model.adapterLabel));
  return h('div', {class: 'lens-axis-header', role: 'group', 'aria-label': 'Axes'}, children);
}
