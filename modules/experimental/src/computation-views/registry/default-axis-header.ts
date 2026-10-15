// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {AxisHeaderModel, ViewContext, ViewRegistration} from './view-registration';
import {getAdapterLabel} from '../views/view-utils';

/** Derives explicit pins and wildcards from a view signature. */
export function buildDefaultAxisHeader<Model>(
  view: ViewRegistration<Model>,
  context: ViewContext
): AxisHeaderModel {
  const labelFor = (axis: string) =>
    context.record.axes.axes.find(candidate => candidate.id === axis)?.label ?? axis;
  return {
    chips: [
      ...view.signature.binds.map(axis => {
        const value = context.bindings[axis];
        const label = labelFor(axis);
        if (value === undefined) {
          // The view chose a default instance itself; say so instead of claiming a pin.
          return {axis, label, state: 'context' as const, valueText: 'default (not pinned)'};
        }
        const valueText = Array.isArray(value) ? `{${value.length} keys}` : String(value);
        return {
          axis,
          label,
          state: 'bound' as const,
          valueText,
          action: 'free' as const
        };
      }),
      ...view.signature.free.map(axis => {
        const definition = context.record.axes.axes.find(candidate => candidate.id === axis);
        const label = labelFor(axis);
        const wildcard = `all ${definition?.size === undefined ? '' : `${definition.size} `}${label}s`;
        const role = view.signature.layout[axis]?.replace(/-/g, ' ');
        return {
          axis,
          label,
          state: 'free' as const,
          valueText: `${wildcard}${role ? `, ${role}` : ''}`
        };
      })
    ],
    adapterLabel: getAdapterLabel(context.record)
  };
}

/** Uses a view-specific header when supplied, otherwise the signature-derived header. */
export function getAxisHeader<Model>(
  view: ViewRegistration<Model>,
  model: Model,
  context: ViewContext
): AxisHeaderModel {
  return view.axisHeader?.(model, context) ?? buildDefaultAxisHeader(view, context);
}
