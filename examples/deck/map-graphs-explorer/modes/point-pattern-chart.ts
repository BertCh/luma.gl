// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * A tiny canvas chart for the Pattern mode's panel: Ripley's L(r) - r, the empirical and fitted
 * semivariogram, and the correlogram are drawn from small GPU readbacks (tens of values), so a
 * 2D canvas is enough and no chart dependency is added.
 */

/** One drawable series. */
export type MiniChartSeries = {
  kind: 'line' | 'points' | 'bars';
  x: ArrayLike<number>;
  y: ArrayLike<number>;
  color: string;
  /** Point radius in CSS pixels (`points`), or one radius per point. */
  radius?: number | ArrayLike<number>;
  /** Draws hollow markers for entries where this returns false (`points`). */
  filled?: (index: number) => boolean;
};

/** A horizontal reference line. */
export type MiniChartReferenceLine = {y: number; color: string; dashed?: boolean};

/** Options of {@link MiniChart.update}. */
export type MiniChartUpdate = {
  series: readonly MiniChartSeries[];
  referenceLines?: readonly MiniChartReferenceLine[];
  /** Fixed x range. Defaults to the extent of the series. */
  xRange?: readonly [number, number];
  /** Caption drawn in the top-left corner. */
  caption?: string;
};

const WIDTH = 296;
const HEIGHT = 112;
const MARGIN = {left: 38, right: 8, top: 16, bottom: 18};

/** A small line, point and bar chart on a canvas appended to the panel. */
export class MiniChart {
  readonly element: HTMLDivElement;
  private readonly canvas: HTMLCanvasElement;
  private readonly title: string;
  private readonly xLabel: string;

  constructor(title: string, xLabel: string) {
    this.title = title;
    this.xLabel = xLabel;
    this.element = document.createElement('div');
    this.element.style.cssText = 'margin-top:10px';
    this.canvas = document.createElement('canvas');
    const scale = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.round(WIDTH * scale);
    this.canvas.height = Math.round(HEIGHT * scale);
    this.canvas.style.cssText = `width:${WIDTH}px;height:${HEIGHT}px;max-width:100%;border-radius:8px;background:rgba(19,32,63,.7)`;
    this.element.appendChild(this.canvas);
    this.canvas.getContext('2d')?.scale(scale, scale);
    this.update({series: []});
  }

  /** Inserts the chart before `anchor` in the panel. */
  insertBefore(anchor: Element | null): void {
    anchor?.insertAdjacentElement('beforebegin', this.element);
  }

  /** Removes the chart from the panel. */
  destroy(): void {
    this.element.remove();
  }

  /** Redraws the chart. Non-finite values are skipped. */
  update({series, referenceLines = [], xRange, caption}: MiniChartUpdate): void {
    const context = this.canvas.getContext('2d');
    if (!context) return;
    context.save();
    context.setTransform(this.canvas.width / WIDTH, 0, 0, this.canvas.height / HEIGHT, 0, 0);
    context.clearRect(0, 0, WIDTH, HEIGHT);
    let minimumX = Infinity;
    let maximumX = -Infinity;
    let minimumY = Infinity;
    let maximumY = -Infinity;
    for (const entry of series) {
      for (let index = 0; index < entry.x.length; index++) {
        const x = entry.x[index];
        const y = entry.y[index];
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        minimumX = Math.min(minimumX, x);
        maximumX = Math.max(maximumX, x);
        minimumY = Math.min(minimumY, y);
        maximumY = Math.max(maximumY, y);
      }
    }
    for (const line of referenceLines) {
      if (Number.isFinite(line.y)) {
        minimumY = Math.min(minimumY, line.y);
        maximumY = Math.max(maximumY, line.y);
      }
    }
    if (series.some(entry => entry.kind === 'bars')) {
      minimumY = Math.min(minimumY, 0);
      maximumY = Math.max(maximumY, 0);
    }
    if (xRange) [minimumX, maximumX] = xRange;
    context.font = '10px system-ui, sans-serif';
    context.fillStyle = '#a9b8d0';
    context.textBaseline = 'alphabetic';
    context.fillText(this.title, MARGIN.left, 11);
    if (!(maximumX > minimumX) || !(maximumY >= minimumY)) {
      context.fillText('no data', MARGIN.left + 6, HEIGHT / 2);
      context.restore();
      return;
    }
    if (maximumY === minimumY) maximumY = minimumY + 1;
    const padding = (maximumY - minimumY) * 0.08;
    minimumY -= padding;
    maximumY += padding;
    const plotWidth = WIDTH - MARGIN.left - MARGIN.right;
    const plotHeight = HEIGHT - MARGIN.top - MARGIN.bottom;
    const toX = (x: number) => MARGIN.left + ((x - minimumX) / (maximumX - minimumX)) * plotWidth;
    const toY = (y: number) =>
      MARGIN.top + (1 - (y - minimumY) / (maximumY - minimumY)) * plotHeight;

    context.strokeStyle = '#2a3c66';
    context.lineWidth = 1;
    context.strokeRect(MARGIN.left, MARGIN.top, plotWidth, plotHeight);
    context.fillStyle = '#7f90ad';
    context.textAlign = 'right';
    context.fillText(formatTick(maximumY), MARGIN.left - 3, MARGIN.top + 8);
    context.fillText(formatTick(minimumY), MARGIN.left - 3, MARGIN.top + plotHeight);
    context.textAlign = 'left';
    context.fillText(formatTick(minimumX), MARGIN.left, HEIGHT - 5);
    context.textAlign = 'right';
    context.fillText(`${formatTick(maximumX)} ${this.xLabel}`, WIDTH - MARGIN.right, HEIGHT - 5);
    if (caption) {
      context.textAlign = 'right';
      context.fillStyle = '#f1c96b';
      context.fillText(caption, WIDTH - MARGIN.right, 11);
    }

    for (const line of referenceLines) {
      if (!Number.isFinite(line.y)) continue;
      context.strokeStyle = line.color;
      context.setLineDash(line.dashed ? [4, 3] : []);
      context.beginPath();
      context.moveTo(MARGIN.left, toY(line.y));
      context.lineTo(MARGIN.left + plotWidth, toY(line.y));
      context.stroke();
    }
    context.setLineDash([]);

    for (const entry of series) {
      context.strokeStyle = entry.color;
      context.fillStyle = entry.color;
      context.lineWidth = 1.5;
      if (entry.kind === 'line') {
        context.beginPath();
        let drawing = false;
        for (let index = 0; index < entry.x.length; index++) {
          const x = entry.x[index];
          const y = entry.y[index];
          if (!Number.isFinite(x) || !Number.isFinite(y)) {
            drawing = false;
            continue;
          }
          if (drawing) context.lineTo(toX(x), toY(y));
          else context.moveTo(toX(x), toY(y));
          drawing = true;
        }
        context.stroke();
      } else if (entry.kind === 'bars') {
        const barWidth = Math.max(1, (plotWidth / entry.x.length) * 0.7);
        for (let index = 0; index < entry.x.length; index++) {
          const y = entry.y[index];
          if (!Number.isFinite(y)) continue;
          context.fillRect(
            toX(entry.x[index]) - barWidth / 2,
            toY(Math.max(y, 0)),
            barWidth,
            Math.abs(toY(y) - toY(0))
          );
        }
      } else {
        for (let index = 0; index < entry.x.length; index++) {
          const x = entry.x[index];
          const y = entry.y[index];
          if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
          const radius =
            typeof entry.radius === 'object' ? entry.radius[index] : (entry.radius ?? 2.5);
          context.beginPath();
          context.arc(toX(x), toY(y), radius, 0, Math.PI * 2);
          if (entry.filled && !entry.filled(index)) context.stroke();
          else context.fill();
        }
      }
    }
    context.restore();
  }
}

function formatTick(value: number): string {
  const magnitude = Math.abs(value);
  if (magnitude >= 1000) return `${(value / 1000).toFixed(magnitude >= 10000 ? 0 : 1)}k`;
  if (magnitude >= 100) return value.toFixed(0);
  if (magnitude >= 1) return value.toFixed(1);
  if (magnitude === 0) return '0';
  return value.toFixed(magnitude >= 0.1 ? 2 : 3);
}
