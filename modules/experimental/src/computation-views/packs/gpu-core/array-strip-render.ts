// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ViewContext} from '../../registry/view-registration';
import {getCardMarkAttributes} from '../../encoding/card-tokens';
import {getCaptureProvenance, formatViewNumber} from '../../views/record-queries';
import {h, type ViewElementNode} from '../../view-node';
import {
  getArrayTokens,
  getArrayTokenValue,
  type ArrayResultPanel,
  type GPUCoreArrayResultModel
} from './array-result-model';
import {
  getArrayMarkAttributes,
  renderArrayLabel,
  renderArrayContextPath
} from './array-render-utils';

/** Cell placement reserves enough room for an explicit head-window remainder. */
export function getArrayCellBounds(
  panel: ArrayResultPanel,
  position: number
): {left: number; width: number} {
  const gaps = panel.cells.filter(cell => 'gap' in cell).length;
  const gapWidth = Math.min(90, panel.width / 3);
  const cellWidth = (panel.width - gaps * gapWidth) / Math.max(1, panel.cells.length - gaps);
  const precedingGaps = panel.cells.slice(0, position).filter(cell => 'gap' in cell).length;
  return {
    left: panel.x + (position - precedingGaps) * cellWidth + precedingGaps * gapWidth,
    width: 'gap' in panel.cells[position] ? gapWidth : cellWidth
  };
}

/** Visible key position; undefined deliberately produces a connector stub instead of invented data. */
export function getArrayCellPosition(panel: ArrayResultPanel, key: number): number | undefined {
  const position = panel.cells.findIndex(cell => 'key' in cell && cell.key === key);
  if (position < 0) return undefined;
  const bounds = getArrayCellBounds(panel, position);
  return bounds.left + bounds.width / 2;
}

/** Renders value, flag, bar, or signal rows from recorded values, with a shared numeric-label budget. */
export function renderArrayStrip(
  panel: ArrayResultPanel,
  model: GPUCoreArrayResultModel,
  context: ViewContext
): ViewElementNode {
  const bars = panel.tokens.includes('bars') || panel.tokens.includes('complex-magnitude');
  const flags = panel.tokens.includes('flags');
  const signal = panel.tokens.includes('signal');
  const finite = panel.values.filter(Number.isFinite);
  const minimum = finite.reduce((previous, value) => Math.min(previous, value), 0);
  const maximum = finite.reduce((previous, value) => Math.max(previous, value), 0);
  const span = maximum - minimum || 1;
  const valueY = (value: number) =>
    panel.y + ((maximum - value) / span) * (panel.height - (model.thumbnail ? 0 : 10));
  const edges = getArrayTokenValue(panel.tokens, 'edges')?.split(',');
  const children: ViewElementNode[] = [];
  if (!model.thumbnail)
    children.push(
      renderArrayLabel(`${panel.capture.name} (${panel.values.length})`, 16, panel.y + 17)
    );
  const glyph = getArrayTokenValue(panel.tokens, 'operator');
  if (glyph && !model.thumbnail)
    children.push(renderArrayLabel(glyph, panel.x - 12, panel.y + panel.height / 2));
  if (signal) {
    // Break paths at omissions and nonfinite values; never bridge an unseen interval.
    let path = '';
    let connected = false;
    for (let position = 0; position < panel.cells.length; position++) {
      const cell = panel.cells[position];
      if (!('index' in cell) || !Number.isFinite(panel.values[cell.index])) {
        connected = false;
        continue;
      }
      const bounds = getArrayCellBounds(panel, position);
      path += `${connected ? 'L' : 'M'}${bounds.left + bounds.width / 2},${valueY(panel.values[cell.index])} `;
      connected = true;
    }
    children.push(
      h('path', {
        ...getArrayMarkAttributes(panel, context, 0, 'line'),
        d: path,
        'data-token': 'signal',
        'stroke-width': 1
      })
    );
  }
  panel.cells.forEach((cell, position) => {
    const {left, width: cellWidth} = getArrayCellBounds(panel, position);
    if ('gap' in cell) {
      if (!model.thumbnail)
        children.push(
          renderArrayLabel(cell.gap, left + cellWidth / 2, panel.y + 18, {
            'text-anchor': 'middle',
            'font-size': cell.gap === '…' ? 11 : 9
          })
        );
      return;
    }
    const value = panel.values[cell.index];
    const center = left + cellWidth / 2;
    if (!signal) {
      const baseline = valueY(0);
      const top = bars ? Math.min(baseline, valueY(value)) : panel.y;
      const height = bars
        ? Math.max(1, Math.abs(valueY(value) - baseline))
        : flags
          ? Math.min(14, panel.height)
          : panel.height;
      const attributes = getArrayMarkAttributes(panel, context, cell.index);
      children.push(
        h(
          'rect',
          {
            ...attributes,
            x: flags ? center - height / 2 : left + 1,
            y: Number.isFinite(top) ? top : panel.y,
            width: flags ? height : Math.max(1, cellWidth - 2),
            height: Number.isFinite(height) ? height : panel.height,
            ...(flags && value === 0 ? {fill: 'none', stroke: attributes['fill']} : {}),
            'data-token': flags ? 'flags' : bars ? 'bars' : 'value'
          },
          h('title', null, `${panel.capture.name}[${cell.key}] = ${formatViewNumber(value)}`)
        )
      );
    }
    if (panel.numbered.includes(cell.index) && !model.thumbnail) {
      children.push(
        renderArrayLabel(
          formatViewNumber(value),
          center,
          panel.y + (bars || signal ? panel.height : 18),
          {'text-anchor': 'middle', 'data-numbered': 'true', 'data-key': cell.key}
        )
      );
    }
    if (panel.capture.highlight?.key === cell.key && panel.palette !== 'div') {
      children.push(
        h('rect', {
          ...getCardMarkAttributes('highlight', 'line'),
          'data-provenance': panel.provenance,
          'data-capture': panel.capture.id,
          x: left + 1,
          y: panel.y,
          width: Math.max(1, cellWidth - 2),
          height: panel.height,
          'stroke-width': 2.5,
          'data-action': 'pick',
          'data-axis': 'entity',
          'data-key': cell.key
        })
      );
    }
    if (edges && !model.thumbnail)
      children.push(
        renderArrayLabel(
          `${edges[cell.index] ?? '?'}–${edges[cell.index + 1] ?? '?'}`,
          center,
          panel.y + panel.height + 10,
          {'text-anchor': 'middle', 'data-token': 'edges', 'font-size': 9}
        )
      );
  });
  const segmentInput = model.panels.find(
    candidate => candidate.role === 'input' && !candidate.tokens.includes('flags')
  );
  const sameKeySpace = segmentInput && panel.capture.keySpace === segmentInput.capture.keySpace;
  for (const segment of sameKeySpace ? model.segments : []) {
    const position = panel.cells.findIndex(cell => 'key' in cell && cell.key === segment);
    if (position >= 0)
      children.push(
        renderArrayContextPath(
          `M${getArrayCellBounds(panel, position).left},${panel.y}v${panel.height}`,
          context,
          {
            'stroke-width': 2,
            'data-token': 'segments',
            'data-provenance': model.segmentCapture
              ? getCaptureProvenance(model.segmentCapture)
              : 'metadata'
          }
        )
      );
  }
  return h('g', {'data-panel': panel.capture.name}, children);
}

/** Links only visible captured keys; off-window references receive explicit index stubs. */
export function renderArrayConnections(
  model: GPUCoreArrayResultModel,
  context: ViewContext
): ViewElementNode {
  const children: ViewElementNode[] = [];
  const input = model.panels.find(
    panel => panel.role === 'input' && !panel.tokens.includes('flags')
  );
  const output = model.panels.find(panel => panel.role === 'output');
  if (!input || !output || model.tiled) return h('g', null);
  const connect = (sourceKey: number, targetKey: number, token: string, provenance: string) => {
    const sourceX = getArrayCellPosition(input, sourceKey);
    const targetX = getArrayCellPosition(output, targetKey);
    if (sourceX === undefined && targetX === undefined) return;
    const startY = input.y + input.height;
    const endY = output.y;
    const startX = sourceX ?? targetX!;
    const endX = targetX ?? sourceX!;
    const stub = sourceX === undefined || targetX === undefined;
    const fromY = sourceX === undefined ? endY - 8 : startY;
    const toY = targetX === undefined ? startY + 8 : endY;
    children.push(
      renderArrayContextPath(`M${startX},${fromY}L${endX},${toY}m-2,-3l2,3l2,-3`, context, {
        'data-token': token,
        'data-source-key': sourceKey,
        'data-target-key': targetKey,
        'data-provenance': provenance,
        ...(stub ? {'data-stub': 'true'} : {})
      })
    );
    if (stub && !model.thumbnail)
      children.push(
        renderArrayLabel(
          String(sourceX === undefined ? sourceKey : targetKey),
          endX + 3,
          sourceX === undefined ? fromY : toY,
          {'data-stub-label': 'true'}
        )
      );
  };
  for (const indices of model.panels.filter(
    panel => panel.tokens.includes('index-gather') || panel.tokens.includes('index-scatter')
  )) {
    for (const cell of indices.cells) {
      if (!('index' in cell)) continue;
      const gather = indices.tokens.includes('index-gather');
      connect(
        gather ? indices.values[cell.index] : cell.key,
        gather ? cell.key : indices.values[cell.index],
        gather ? 'index-gather' : 'index-scatter',
        indices.provenance
      );
    }
  }
  if (output.tokens.includes('compact')) {
    const flags = model.captures.find(capture => getArrayTokens(capture).includes('flags'));
    if (flags) {
      const first = Number(getArrayTokenValue(getArrayTokens(flags), 'first') ?? 0);
      // A nonzero-start flag window cannot locate global compacted slots without earlier flags.
      if (first === 0) {
        let slot = 0;
        Array.from(flags.values, Number).forEach((flag, index) => {
          if (flag !== 0) connect(index, slot, 'compact', getCaptureProvenance(flags));
          slot = (slot + flag) >>> 0;
        });
      }
    }
  }
  if (output.values.length === 1 && !output.tokens.includes('compact')) {
    for (const cell of input.cells)
      if ('key' in cell) connect(cell.key, output.first, 'fan', output.provenance);
  }
  if (output.tokens.includes('prefix') && output.capture.highlight) {
    const key = output.capture.highlight.key;
    const start = model.segments
      .filter(segment => segment <= key)
      .reduce((previous, segment) => Math.max(previous, segment), 0);
    const visible = input.cells.filter(
      cell => 'key' in cell && cell.key >= start && cell.key <= key
    );
    const first = visible[0];
    const last = visible.at(-1);
    if (first && last && 'key' in first && 'key' in last) {
      const left = getArrayCellPosition(input, first.key)!;
      const right = getArrayCellPosition(input, last.key)!;
      children.push(
        renderArrayContextPath(`M${left},${input.y + input.height + 1}v4H${right}v-4`, context, {
          'data-token': 'prefix',
          'data-prefix-start': start,
          'data-prefix-end': key
        })
      );
    }
  }
  return h('g', {class: 'cv-array-connections'}, children);
}
