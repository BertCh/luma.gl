// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getRampCssGradient} from '../engine/ramps';
import type {LegendColor, LegendSpec} from '../scenes/scene';
import {h} from './dom';

const cssColor = (color: LegendColor) =>
  `rgb(${color[0]} ${color[1]} ${color[2]} / ${((color[3] ?? 255) / 255).toFixed(2)})`;

const defaultFormat = (value: number): string =>
  Math.abs(value) >= 1000 || (value !== 0 && Math.abs(value) < 0.01)
    ? value.toPrecision(3)
    : Number(value.toFixed(2)).toString();

/**
 * Renders legends. Ramp gradients come from the same `RAMP_STOPS` table as the WGSL shader, and
 * `'gpu'` extents show what the scene last reported through `ctx.setLegendExtent`.
 */
export function renderLegends(
  container: HTMLElement,
  specs: readonly LegendSpec[],
  gpuExtents: ReadonlyMap<string, readonly [number, number]>
): void {
  container.replaceChildren(...specs.map(spec => renderLegend(spec, gpuExtents)));
}

function renderLegend(
  spec: LegendSpec,
  gpuExtents: ReadonlyMap<string, readonly [number, number]>
): HTMLElement {
  if (spec.kind === 'ramp') {
    const extent = spec.extent === 'gpu' ? gpuExtents.get(spec.id ?? spec.title) : spec.extent;
    const format = spec.format ?? defaultFormat;
    const low = spec.labels?.[0] ?? (extent ? format(extent[0]) : '—');
    const high = spec.labels?.[1] ?? (extent ? format(extent[1]) : '—');
    const middle = extent && !spec.labels ? format(extent[0] + (extent[1] - extent[0]) * 0.5) : '';
    const logTicks =
      spec.scale === 'log' && extent && !spec.labels && extent[0] > 0 && extent[1] > extent[0]
        ? renderLogTicks(extent)
        : null;
    const background = spec.colors
      ? getColorsCssGradient(spec.colors)
      : getRampCssGradient(spec.ramp, {sqrtScale: spec.sqrtScale});
    return h(
      'figure',
      {class: 'legend legend-ramp'},
      h(
        'figcaption',
        {},
        spec.title,
        spec.unit ? h('span', {class: 'muted'}, ` (${spec.unit})`) : null
      ),
      h('div', {
        class: 'legend-gradient',
        role: 'img',
        'aria-label': `${spec.ramp} color ramp from ${low} to ${high}`,
        style: {background}
      }),
      logTicks ??
        h(
          'div',
          {class: 'legend-ticks'},
          h('span', {}, low),
          h('span', {}, middle),
          h('span', {}, high)
        ),
      spec.extent === 'gpu' && !extent
        ? h('div', {class: 'legend-note'}, 'Range read from the GPU…')
        : null,
      spec.extent === 'gpu' && extent && !spec.labels
        ? h(
            'div',
            {class: 'legend-note'},
            `${spec.sqrtScale ? 'Square-root scale. ' : ''}${spec.scale === 'log' ? 'Log scale. ' : ''}Range read from the GPU.`
          )
        : null
    );
  }
  if (spec.kind === 'categories') {
    return h(
      'figure',
      {class: 'legend legend-categories'},
      h('figcaption', {}, spec.title),
      h(
        'ul',
        {},
        spec.entries.map(entry =>
          h(
            'li',
            {},
            h('span', {class: 'swatch', style: {background: cssColor(entry.color)}}),
            entry.label
          )
        )
      ),
      spec.note ? h('div', {class: 'legend-note'}, spec.note) : null
    );
  }
  return h(
    'figure',
    {class: 'legend legend-size'},
    h('figcaption', {}, spec.title),
    h(
      'ul',
      {},
      spec.entries.map(entry =>
        h(
          'li',
          {},
          h('span', {
            class: 'dot',
            style: {
              width: `${entry.radiusPixels * 2}px`,
              height: `${entry.radiusPixels * 2}px`,
              background: cssColor(spec.color ?? [120, 130, 150, 255])
            }
          }),
          entry.label
        )
      )
    )
  );
}

/** Evenly spaced CSS gradient through custom legend colors, low value first. */
function getColorsCssGradient(colors: readonly LegendColor[]): string {
  const stops = colors.map(
    (color, index) =>
      `${cssColor(color)} ${((index / Math.max(1, colors.length - 1)) * 100).toFixed(1)}%`
  );
  return `linear-gradient(to right, ${stops.join(', ')})`;
}

/** Decade ticks (`1`, `10`, `10^2`, ...) placed by log position between a positive extent. */
function renderLogTicks(extent: readonly [number, number]): HTMLElement {
  const lowLog = Math.log10(extent[0]);
  const highLog = Math.log10(extent[1]);
  const ticks: HTMLElement[] = [];
  for (let power = Math.ceil(lowLog - 1e-9); power <= highLog + 1e-9; power++) {
    const fraction = (power - lowLog) / (highLog - lowLog);
    ticks.push(
      h(
        'span',
        {class: 'legend-tick', style: {left: `${(fraction * 100).toFixed(2)}%`}},
        power >= 0 && power <= 1
          ? String(10 ** power)
          : h('span', {}, '10', h('sup', {}, String(power)))
      )
    );
  }
  if (ticks.length === 0) {
    const format = (value: number) => defaultFormat(value);
    ticks.push(
      h('span', {class: 'legend-tick', style: {left: '0%'}}, format(extent[0])),
      h('span', {class: 'legend-tick', style: {left: '100%'}}, format(extent[1]))
    );
  }
  return h('div', {class: 'legend-ticks legend-ticks-log'}, ticks);
}
