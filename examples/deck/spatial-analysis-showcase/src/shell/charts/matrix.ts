// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {RAMP_STOPS, type RampName, getClassColors, sampleRamp} from '../../engine/ramps';
import type {ChartColor, MatrixChartData} from '../../scenes/chart-types';
import {
  type BuildContext,
  type ChartBuild,
  FONT_TICK,
  colorToCss,
  createRoot,
  estimateTextWidth,
  formatChartNumber,
  getLuminance,
  svg,
  truncateText
} from './core';

/** Resolves a ramp name, falling back to a default for an unknown or missing name. */
export function resolveRamp(name: string | undefined, fallback: RampName): RampName {
  return name && name in RAMP_STOPS ? (name as RampName) : fallback;
}

/** Builds a heat matrix with labels, optional marginal bars, diagonal and highlight. */
export function buildMatrix(data: MatrixChartData, context: BuildContext): ChartBuild {
  const {rows, columns} = data;
  const width = context.width;
  if (!(rows > 0 && columns > 0)) {
    return {
      svg: createRoot('matrix', width, 20, data.title, data.description, 'Matrix'),
      table: null
    };
  }
  const title = context.title ?? data.title;
  const formatCell = data.formatCell ?? formatChartNumber;
  const marginals = data.marginals ?? 'none';
  const rowMarginal = marginals === 'rows' || marginals === 'both';
  const columnMarginal = marginals === 'columns' || marginals === 'both';
  const showText = rows <= 8 && columns <= 8;

  const rowLabelWidth = data.rowLabels
    ? Math.min(72, Math.max(...data.rowLabels.map(label => estimateTextWidth(label))) + 6)
    : 0;
  const left = 4 + (data.yLabel && !context.compact ? 13 : 0) + rowLabelWidth;
  const right = 6 + (rowMarginal ? 40 : 0);
  const cellWidth = Math.max(4, (width - left - right) / columns);
  const gridWidth = cellWidth * columns;
  const top =
    6 +
    (title ? 16 : 0) +
    (data.xLabel && !context.compact ? 12 : 0) +
    (data.columnLabels ? 14 : 0);

  const cellColors = (() => {
    const values = data.values;
    const finite: number[] = [];
    for (let index = 0; index < rows * columns; index++) {
      if (Number.isFinite(values[index])) finite.push(values[index]);
    }
    const min = finite.length ? Math.min(...finite) : 0;
    const max = finite.length ? Math.max(...finite) : 1;
    const name = resolveRamp(data.ramp, data.midpoint !== undefined ? 'diverging' : 'viridis');
    const classColors = data.breaks
      ? getClassColors(name, data.breaks.length + 1, data.reverse)
      : null;
    const colorFor = (value: number, index: number): ChartColor | null => {
      if (data.cellColors?.[index]) return data.cellColors[index];
      if (!Number.isFinite(value)) return null;
      if (classColors && data.breaks) {
        const classIndex = data.breaks.filter(breakValue => breakValue <= value).length;
        return classColors[Math.min(classIndex, classColors.length - 1)];
      }
      let t: number;
      if (data.midpoint !== undefined) {
        const half = Math.max(Math.abs(min - data.midpoint), Math.abs(max - data.midpoint)) || 1;
        t = 0.5 + (value - data.midpoint) / (2 * half);
      } else t = max > min ? (value - min) / (max - min) : 0.5;
      return sampleRamp(name, t, data.reverse);
    };
    return {colorFor, min, max, name, classColors};
  })();

  // Marginal sums.
  const rowSums = Array.from({length: rows}, (_, row) => {
    let sum = 0;
    for (let column = 0; column < columns; column++) {
      const value = data.values[row * columns + column];
      if (Number.isFinite(value)) sum += value;
    }
    return sum;
  });
  const columnSums = Array.from({length: columns}, (_, column) => {
    let sum = 0;
    for (let row = 0; row < rows; row++) {
      const value = data.values[row * columns + column];
      if (Number.isFinite(value)) sum += value;
    }
    return sum;
  });
  const bottomMarginal = columnMarginal ? 30 : 0;
  const keyHeight = data.cellColors || context.compact ? 0 : 28;
  const cellHeight = data.height
    ? Math.max(6, (data.height - top - bottomMarginal - keyHeight - 4) / rows)
    : Math.min(24, Math.max(9, cellWidth));
  const gridHeight = cellHeight * rows;
  const height = top + gridHeight + bottomMarginal + keyHeight + 6;

  const interactive = !!data.onCellClick;
  const root = createRoot('matrix', width, height, title, data.description, 'Matrix', interactive);
  if (title) root.append(svg('text', {class: 'chart-title', x: 2, y: 13}, title));

  // Axis labels.
  if (data.xLabel && !context.compact) {
    root.append(
      svg(
        'text',
        {
          class: 'chart-label',
          x: left + gridWidth / 2,
          y: top - (data.columnLabels ? 17 : 4),
          'text-anchor': 'middle'
        },
        data.xLabel
      )
    );
  }
  if (data.yLabel && !context.compact) {
    root.append(
      svg(
        'text',
        {
          class: 'chart-label',
          transform: `translate(10 ${top + gridHeight / 2}) rotate(-90)`,
          'text-anchor': 'middle'
        },
        data.yLabel
      )
    );
  }
  // Column labels, thinned to fit.
  if (data.columnLabels) {
    const maxWidth = Math.max(...data.columnLabels.map(label => estimateTextWidth(label)));
    const stride = Math.max(1, Math.ceil((maxWidth + 4) / cellWidth));
    for (let column = 0; column < columns; column += stride) {
      const bold = data.highlight?.column === column;
      root.append(
        svg(
          'text',
          {
            class: `chart-tick${bold ? ' is-strong' : ''}`,
            x: left + (column + 0.5) * cellWidth,
            y: top - 4,
            'text-anchor': 'middle'
          },
          truncateText(data.columnLabels[column] ?? '', cellWidth * stride - 1)
        )
      );
    }
  }
  if (data.rowLabels) {
    const stride = Math.max(1, Math.ceil((FONT_TICK + 2) / cellHeight));
    for (let row = 0; row < rows; row += stride) {
      const bold = data.highlight?.row === row;
      root.append(
        svg(
          'text',
          {
            class: `chart-tick${bold ? ' is-strong' : ''}`,
            x: left - 5,
            y: top + (row + 0.5) * cellHeight + 3.5,
            'text-anchor': 'end'
          },
          truncateText(data.rowLabels[row] ?? '', rowLabelWidth - 4)
        )
      );
    }
  }

  // Cells.
  const cells = svg('g', {class: 'chart-cells'});
  const texts = svg('g', {class: 'chart-cell-texts'});
  root.append(cells, texts);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const index = row * columns + column;
      const value = data.values[index];
      const color = cellColors.colorFor(value, index);
      const rowName = data.rowLabels?.[row] ?? `Row ${row + 1}`;
      const columnName = data.columnLabels?.[column] ?? `column ${column + 1}`;
      const label = `${rowName}, ${columnName}: ${Number.isFinite(value) ? formatCell(value) : 'no data'}`;
      const rect = svg(
        'rect',
        {
          class: `chart-cell${color ? '' : ' is-empty'}${interactive ? ' is-input' : ''}`,
          x: (left + column * cellWidth).toFixed(2),
          y: (top + row * cellHeight).toFixed(2),
          width: cellWidth.toFixed(2),
          height: cellHeight.toFixed(2),
          style: color ? `fill:${colorToCss(color)}` : undefined,
          tabindex: interactive ? 0 : undefined,
          role: interactive ? 'button' : undefined,
          'aria-label': interactive ? label : undefined
        },
        svg('title', {}, label)
      );
      if (data.onCellClick) {
        const onClick = data.onCellClick;
        rect.addEventListener('click', () => onClick(row, column));
        rect.addEventListener('keydown', event => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            onClick(row, column);
          }
        });
      }
      cells.append(rect);
      if (showText && color && Number.isFinite(value) && cellHeight >= 13 && cellWidth >= 18) {
        texts.append(
          svg(
            'text',
            {
              class: `chart-cell-text ${getLuminance(color) > 0.4 ? 'on-light' : 'on-dark'}`,
              x: (left + (column + 0.5) * cellWidth).toFixed(2),
              y: (top + (row + 0.5) * cellHeight + 3.5).toFixed(2),
              'text-anchor': 'middle'
            },
            truncateText(formatCell(value), cellWidth - 2)
          )
        );
      }
    }
  }

  // Diagonal and highlight outlines.
  if (data.diagonal) {
    for (let index = 0; index < Math.min(rows, columns); index++) {
      root.append(
        svg('rect', {
          class: 'chart-cell-outline',
          x: left + index * cellWidth,
          y: top + index * cellHeight,
          width: cellWidth,
          height: cellHeight
        })
      );
    }
  }
  const highlight = data.highlight;
  if (highlight && (highlight.row !== undefined || highlight.column !== undefined)) {
    const row = highlight.row;
    const column = highlight.column;
    root.append(
      svg('rect', {
        class: 'chart-cell-highlight',
        x: left + (column ?? 0) * cellWidth,
        y: top + (row ?? 0) * cellHeight,
        width: column === undefined ? gridWidth : cellWidth,
        height: row === undefined ? gridHeight : cellHeight
      })
    );
  }

  // Marginal bars.
  const marginalLabel = (value: number) => formatChartNumber(value);
  if (rowMarginal) {
    const maxSum = Math.max(...rowSums, 1e-9);
    for (let row = 0; row < rows; row++) {
      const barWidth = (rowSums[row] / maxSum) * 26;
      const barHeight = Math.max(2, cellHeight * 0.55);
      root.append(
        svg(
          'rect',
          {
            class: 'chart-marginal',
            x: left + gridWidth + 5,
            y: top + (row + 0.5) * cellHeight - barHeight / 2,
            width: Math.max(0.5, barWidth),
            height: barHeight
          },
          svg(
            'title',
            {},
            `${data.rowLabels?.[row] ?? `Row ${row + 1}`}: ${marginalLabel(rowSums[row])}`
          )
        )
      );
    }
    root.append(
      svg(
        'text',
        {class: 'chart-note', x: left + gridWidth + 5, y: top + gridHeight + 11},
        `max ${marginalLabel(maxSum)}`
      )
    );
  }
  if (columnMarginal) {
    const maxSum = Math.max(...columnSums, 1e-9);
    for (let column = 0; column < columns; column++) {
      const barHeight = (columnSums[column] / maxSum) * 20;
      const barWidth = Math.max(2, cellWidth * 0.55);
      root.append(
        svg(
          'rect',
          {
            class: 'chart-marginal',
            x: left + (column + 0.5) * cellWidth - barWidth / 2,
            y: top + gridHeight + 4,
            width: barWidth,
            height: Math.max(0.5, barHeight)
          },
          svg(
            'title',
            {},
            `${data.columnLabels?.[column] ?? `Column ${column + 1}`}: ${marginalLabel(columnSums[column])}`
          )
        )
      );
    }
    root.append(
      svg(
        'text',
        {class: 'chart-note', x: left - 5, y: top + gridHeight + 14, 'text-anchor': 'end'},
        `max ${marginalLabel(maxSum)}`
      )
    );
  }

  // Colour key.
  if (keyHeight) {
    const keyTop = top + gridHeight + bottomMarginal + 8;
    const keyWidth = Math.min(120, gridWidth);
    const {classColors} = cellColors;
    if (classColors && data.breaks) {
      const swatchWidth = keyWidth / classColors.length;
      classColors.forEach((color, index) => {
        root.append(
          svg('rect', {
            class: 'chart-key-swatch',
            x: left + index * swatchWidth,
            y: keyTop,
            width: swatchWidth,
            height: 7,
            style: `fill:${colorToCss(color)}`
          })
        );
      });
      const stride = Math.max(1, Math.ceil(30 / swatchWidth));
      data.breaks.forEach((breakValue, index) => {
        if (index % stride) return;
        root.append(
          svg(
            'text',
            {
              class: 'chart-tick',
              x: left + (index + 1) * swatchWidth,
              y: keyTop + 18,
              'text-anchor': 'middle'
            },
            formatCell(breakValue)
          )
        );
      });
    } else {
      const steps = 32;
      for (let step = 0; step < steps; step++) {
        const t = (step + 0.5) / steps;
        const color = sampleRamp(cellColors.name, t, data.reverse);
        root.append(
          svg('rect', {
            class: 'chart-key-swatch',
            x: left + (t - 0.5 / steps) * keyWidth,
            y: keyTop,
            width: keyWidth / steps + 0.4,
            height: 7,
            style: `fill:rgb(${color[0]},${color[1]},${color[2]})`
          })
        );
      }
      const lowText = formatCell(cellColors.min);
      const highText = formatCell(cellColors.max);
      root.append(
        svg(
          'text',
          {class: 'chart-tick', x: left, y: keyTop + 18, 'text-anchor': 'start'},
          lowText
        ),
        svg(
          'text',
          {class: 'chart-tick', x: left + keyWidth, y: keyTop + 18, 'text-anchor': 'end'},
          highText
        )
      );
      if (data.midpoint !== undefined) {
        root.append(
          svg(
            'text',
            {class: 'chart-tick', x: left + keyWidth / 2, y: keyTop + 18, 'text-anchor': 'middle'},
            formatCell(data.midpoint)
          )
        );
      }
    }
  }

  const table =
    rows * columns <= 400
      ? {
          headers: [
            '',
            ...Array.from(
              {length: columns},
              (_, column) => data.columnLabels?.[column] ?? String(column + 1)
            )
          ],
          rows: Array.from({length: rows}, (_, row) => [
            data.rowLabels?.[row] ?? String(row + 1),
            ...Array.from({length: columns}, (_, column) => {
              const value = data.values[row * columns + column];
              return Number.isFinite(value) ? formatCell(value) : '';
            })
          ])
        }
      : null;
  return {svg: root, table};
}
