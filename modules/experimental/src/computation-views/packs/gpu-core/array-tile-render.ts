// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {renderRasterImage} from '../../encoding/raster-image';
import type {ViewContext} from '../../registry/view-registration';
import {getCardMarkAttributes} from '../../encoding/card-tokens';
import {provenanceMarkAttributes} from '../../encoding/provenance';
import {getCaptureProvenance, formatViewNumber} from '../../views/record-queries';
import {h, type ViewElementNode} from '../../view-node';
import {
  getArrayTokens,
  getArrayTokenValue,
  type ArrayResultPanel,
  type GPUCoreArrayResultModel
} from './array-result-model';
import {getArrayMarkAttributes, renderArrayLabel} from './array-render-utils';

/** Draws small heatmaps or bounded SVG image tiles, preserving class styling and pick provenance. */
export function renderArrayTile(
  panel: ArrayResultPanel,
  model: GPUCoreArrayResultModel,
  context: ViewContext
): ViewElementNode {
  const [columns, rows] = panel.shape ?? [Math.max(1, panel.values.length), 1];
  const cellSize = Math.min(panel.width / columns, panel.height / rows);
  const width = columns * cellSize;
  const height = rows * cellSize;
  const left = panel.x + (panel.width - width) / 2;
  const top = panel.y + (panel.height - height) / 2;
  const grid = panel.tokens.includes('grid');
  const children: ViewElementNode[] = [];
  const glyph = getArrayTokenValue(panel.tokens, 'operator');
  if (!model.thumbnail) {
    children.push(renderArrayLabel(panel.capture.name, left, top - 8));
    if (glyph) children.push(renderArrayLabel(glyph, panel.x - 14, top + height / 2));
  }
  const style = getArrayMarkAttributes(panel, context, 0, 'area', true);
  children.push(
    renderRasterImage({
      columns,
      rows,
      limit: model.thumbnail ? 64 : 160,
      valueAt: key => panel.values[key - panel.first],
      categorical: panel.palette === 'nom',
      flipRows: grid,
      colorFor: value =>
        Number.isFinite(value)
          ? (panel.scale?.colorFor(value) ?? '#808080')
          : (panel.scale?.noDataColor ?? '#d9d9d9'),
      attributes: {
        ...style,
        'data-action': undefined,
        'data-key': undefined,
        x: left,
        y: top,
        width,
        height,
        'data-token': grid ? 'grid' : panel.tokens.includes('stencil') ? 'stencil' : 'tile'
      }
    })
  );
  for (const key of panel.numbered.map(index => index + panel.first)) {
    if (model.thumbnail || Math.max(columns, rows) > 6) break;
    const column = key % columns;
    const row = Math.floor(key / columns);
    children.push(
      renderArrayLabel(
        formatViewNumber(panel.values[key - panel.first]),
        left + (column + 0.5) * cellSize,
        top + ((grid ? rows - row - 1 : row) + 0.5) * cellSize + 3,
        {'text-anchor': 'middle', 'data-numbered': 'true'}
      )
    );
  }
  const highlight = panel.capture.highlight;
  if (highlight && panel.palette !== 'div') {
    const row = Math.floor(highlight.key / columns);
    children.push(
      h('rect', {
        ...getCardMarkAttributes('highlight', 'line'),
        'data-provenance': panel.provenance,
        'data-capture': panel.capture.id,
        x: left + (highlight.key % columns) * cellSize,
        y: top + (grid ? rows - row - 1 : row) * cellSize,
        width: cellSize,
        height: cellSize,
        'data-key': highlight.key
      })
    );
  }
  if (grid) {
    if (model.grid) {
      const coordinates = model.grid;
      for (const capture of model.captures.filter(
        candidate => candidate.role === 'input' && candidate.geometry?.kind === 'points'
      )) {
        const components = capture.components ?? 2;
        const pointCount = Math.floor(capture.values.length / components);
        const pointStride = Math.max(1, Math.ceil(pointCount / (model.thumbnail ? 64 : 400)));
        for (let index = 0; index < pointCount; index += pointStride) {
          const column =
            (Number(capture.values[index * components]) - coordinates.originX) /
            coordinates.cellSize;
          const row =
            (Number(capture.values[index * components + 1]) - coordinates.originY) /
            coordinates.cellSize;
          if (
            !Number.isFinite(column) ||
            !Number.isFinite(row) ||
            column < 0 ||
            column > columns ||
            row < 0 ||
            row > rows
          )
            continue;
          const attributes = getCardMarkAttributes('input', 'point');
          children.push(
            h('circle', {
              ...attributes,
              ...provenanceMarkAttributes(
                getCaptureProvenance(capture),
                attributes['fill'],
                context.idPrefix
              ),
              cx: left + column * cellSize,
              cy: top + height - row * cellSize,
              'data-capture': capture.id,
              'data-action': 'pick',
              'data-axis': 'entity',
              'data-key': index + Number(getArrayTokenValue(getArrayTokens(capture), 'first') ?? 0),
              'data-key-space': capture.keySpace,
              'data-token': 'points'
            })
          );
        }
      }
    }
  }
  // Raster values use the run scale; labels and highlights remain vector overlays.
  return h(
    'g',
    {
      class: Math.max(columns, rows) > 8 ? 'cv-array-image-tile' : 'cv-array-heatmap',
      role: 'img',
      'aria-label': panel.capture.name
    },
    children
  );
}
