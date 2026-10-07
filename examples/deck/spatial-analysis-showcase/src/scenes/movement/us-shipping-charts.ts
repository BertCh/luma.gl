// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {DiagramChartData} from '../scene';
import {
  getCellBounds,
  type ClipPiece,
  type ClipWalkSegment,
  type DensityGrid
} from './us-shipping-density';

/**
 * The two hand-drawn diagrams of the `us-shipping-day` story: one real segment walked cell by
 * cell, and the hourly gate crossings as stacked bars. Both are SVG children for the shell's
 * `diagram` chart, styled with its `diagram-*` classes so they follow the page theme.
 */

const DIAGRAM_WIDTH = 320;
const DEGREES = Math.PI / 180;

/** Escapes text for SVG. */
function escapeText(text: string): string {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/**
 * One segment of the day drawn the way `GPULineDensity` sees it: the block of grid cells around
 * it, the cells it crosses shaded, the cut points where it leaves one cell for the next, and the
 * length of each piece in kilometres. The pieces add up to the segment's great-circle length.
 */
export function getClipWalkDiagram(
  grid: DensityGrid,
  segment: ClipWalkSegment,
  pieces: readonly ClipPiece[]
): DiagramChartData {
  if (pieces.length === 0) {
    return {
      kind: 'diagram',
      width: DIAGRAM_WIDTH,
      height: 40,
      description: 'The chosen segment lies outside the grid.',
      svg: `<text class="diagram-muted" x="160" y="24" text-anchor="middle">The segment is outside the grid</text>`
    };
  }
  const columns = pieces.map(piece => piece.column);
  const rows = pieces.map(piece => piece.row);
  const firstColumn = Math.min(...columns);
  const lastColumn = Math.max(...columns);
  const firstRow = Math.min(...rows);
  const lastRow = Math.max(...rows);
  const columnCount = lastColumn - firstColumn + 1;
  const rowCount = lastRow - firstRow + 1;
  const [west, south, east, north] = grid.bounds;
  const cellWidth = (east - west) / grid.columns;
  const cellHeight = (north - south) / grid.rows;
  const cosLatitude = Math.cos(((segment.from[1] + segment.to[1]) / 2) * DEGREES);
  const blockWidth = columnCount * cellWidth * cosLatitude;
  const blockHeight = rowCount * cellHeight;
  const scale = Math.min(292 / blockWidth, 104 / blockHeight);
  const originX = (DIAGRAM_WIDTH - blockWidth * scale) / 2;
  const originY = 10;
  const blockWest = west + firstColumn * cellWidth;
  const blockNorth = south + (lastRow + 1) * cellHeight;
  const toX = (longitude: number) => originX + (longitude - blockWest) * cosLatitude * scale;
  const toY = (latitude: number) => originY + (blockNorth - latitude) * scale;
  const crossed = new Set(pieces.map(piece => piece.row * grid.columns + piece.column));
  const parts: string[] = [];
  for (let row = firstRow; row <= lastRow; row++) {
    for (let column = firstColumn; column <= lastColumn; column++) {
      const cell = row * grid.columns + column;
      const [cellWest, cellSouth, cellEast, cellNorth] = getCellBounds(grid, cell);
      const attributes = `x="${toX(cellWest).toFixed(1)}" y="${toY(cellNorth).toFixed(1)}" width="${(
        (cellEast - cellWest) * cosLatitude * scale
      ).toFixed(1)}" height="${((cellNorth - cellSouth) * scale).toFixed(1)}"`;
      if (crossed.has(cell)) parts.push(`<rect class="diagram-fill" ${attributes}/>`);
      parts.push(`<rect class="diagram-muted" ${attributes}/>`);
    }
  }
  parts.push(
    `<line class="diagram-signal" x1="${toX(segment.from[0]).toFixed(1)}" y1="${toY(segment.from[1]).toFixed(1)}" x2="${toX(segment.to[0]).toFixed(1)}" y2="${toY(segment.to[1]).toFixed(1)}"/>`
  );
  // The cut points: where the walk leaves one cell for the next.
  for (let index = 1; index < pieces.length; index++) {
    const [longitude, latitude] = pieces[index].from;
    parts.push(
      `<circle class="diagram-signal" cx="${toX(longitude).toFixed(1)}" cy="${toY(latitude).toFixed(1)}" r="2"/>`
    );
  }
  const cellPixels = cellWidth * cosLatitude * scale;
  if (pieces.length <= 9 && cellPixels >= 22) {
    pieces.forEach((piece, index) => {
      const x = toX((piece.from[0] + piece.to[0]) / 2);
      const y = toY((piece.from[1] + piece.to[1]) / 2);
      const offset = index % 2 === 0 ? -6 : 14;
      parts.push(
        `<text class="diagram-ink" x="${x.toFixed(1)}" y="${(y + offset).toFixed(1)}" text-anchor="middle">${piece.kilometers.toFixed(1)}</text>`
      );
    });
  }
  const total = pieces.reduce((sum, piece) => sum + piece.kilometers, 0);
  const baseline = originY + blockHeight * scale + 16;
  parts.push(
    `<text class="diagram-ink" x="${DIAGRAM_WIDTH / 2}" y="${baseline.toFixed(1)}" text-anchor="middle">${pieces.length} pieces, ${total.toFixed(1)} km in total</text>`,
    `<text class="diagram-muted" x="${DIAGRAM_WIDTH / 2}" y="${(baseline + 15).toFixed(1)}" text-anchor="middle">${escapeText(segment.typeLabel)}, ${Math.round(segment.seconds / 60)} min between fixes</text>`
  );
  return {
    kind: 'diagram',
    width: DIAGRAM_WIDTH,
    height: Math.ceil(baseline + 24),
    description: `One ${segment.typeLabel.toLowerCase()} segment of ${segment.kilometers.toFixed(1)} kilometres drawn over the grid. It crosses ${pieces.length} cells and is cut into pieces of ${pieces.map(piece => piece.kilometers.toFixed(1)).join(', ')} kilometres.`,
    svg: parts.join('')
  };
}

/** Offset in hours (whole or half) of an IANA zone from UTC on 9 January 2023. */
export function getZoneOffsetHours(timeZone: string): number {
  const instant = new Date(Date.UTC(2023, 0, 9, 12));
  const parts = new Intl.DateTimeFormat('en-US', {
    hour: 'numeric',
    minute: 'numeric',
    hourCycle: 'h23',
    timeZone
  }).formatToParts(instant);
  const hour = Number(parts.find(part => part.type === 'hour')?.value ?? 12);
  const minute = Number(parts.find(part => part.type === 'minute')?.value ?? 0);
  let offset = hour + minute / 60 - 12;
  if (offset > 12) offset -= 24;
  if (offset < -12) offset += 24;
  return offset;
}

/** Short zone name on 9 January 2023 (`CST`, `EST`, `PST`). */
export function getZoneAbbreviation(timeZone: string): string {
  return (
    new Intl.DateTimeFormat('en-US', {timeZoneName: 'short', timeZone})
      .formatToParts(new Date(Date.UTC(2023, 0, 9, 12)))
      .find(part => part.type === 'timeZoneName')?.value ?? timeZone
  );
}

/** Inputs of {@link getHourlyCrossingsDiagram}. */
export type HourlyCrossingsOptions = {
  /** Crossings per UTC hour in the first and in the second direction. */
  first: ArrayLike<number>;
  second: ArrayLike<number>;
  firstLabel: string;
  secondLabel: string;
  /** Fill colours (CSS hex) of the two directions, the same as the rings on the map. */
  firstColor: string;
  secondColor: string;
  /** Playhead in hours of the UTC day, or `null`. */
  playheadHours: number | null;
  /** IANA zone of the local-time axis, or `null` for UTC only. */
  timeZone: string | null;
};

/** The smallest of 1, 2, 5 times a power of ten that is at least `value`. */
function getNiceMaximum(value: number): number {
  if (value <= 1) return 1;
  const power = 10 ** Math.floor(Math.log10(value));
  for (const step of [1, 2, 5, 10]) {
    if (step * power >= value) return step * power;
  }
  return 10 * power;
}

/**
 * Crossings per hour as stacked bars, the first direction at the bottom, over a UTC axis with the
 * local clock of the gate's zone under it. The vertical rule is the playhead.
 */
export function getHourlyCrossingsDiagram(options: HourlyCrossingsOptions): DiagramChartData {
  const left = 26;
  const right = 314;
  const top = 26;
  const bottom = 108;
  const barWidth = (right - left) / 24;
  const totals = Array.from({length: 24}, (_, hour) => options.first[hour] + options.second[hour]);
  const maximum = getNiceMaximum(Math.max(...totals));
  const toY = (value: number) => bottom - (value / maximum) * (bottom - top);
  const parts: string[] = [
    `<rect fill="${options.firstColor}" x="${left}" y="6" width="9" height="9" rx="1.5"/>`,
    `<text class="diagram-ink" x="${left + 13}" y="14.5">${escapeText(options.firstLabel)}</text>`,
    `<rect fill="${options.secondColor}" x="${left + 140}" y="6" width="9" height="9" rx="1.5"/>`,
    `<text class="diagram-ink" x="${left + 153}" y="14.5">${escapeText(options.secondLabel)}</text>`,
    `<line class="diagram-muted" x1="${left}" y1="${toY(maximum / 2).toFixed(1)}" x2="${right}" y2="${toY(maximum / 2).toFixed(1)}" stroke-dasharray="2 3"/>`
  ];
  for (let hour = 0; hour < 24; hour++) {
    const x = left + hour * barWidth + 1;
    const firstHeight = ((bottom - top) * options.first[hour]) / maximum;
    const secondHeight = ((bottom - top) * options.second[hour]) / maximum;
    if (firstHeight > 0) {
      parts.push(
        `<rect fill="${options.firstColor}" x="${x.toFixed(1)}" y="${(bottom - firstHeight).toFixed(1)}" width="${(barWidth - 2).toFixed(1)}" height="${firstHeight.toFixed(1)}"/>`
      );
    }
    if (secondHeight > 0) {
      parts.push(
        `<rect fill="${options.secondColor}" x="${x.toFixed(1)}" y="${(bottom - firstHeight - secondHeight).toFixed(1)}" width="${(barWidth - 2).toFixed(1)}" height="${secondHeight.toFixed(1)}"/>`
      );
    }
  }
  parts.push(
    `<line class="diagram-muted" x1="${left}" y1="${bottom}" x2="${right}" y2="${bottom}"/>`,
    `<text class="diagram-muted" x="${left - 4}" y="${bottom + 3}" text-anchor="end">0</text>`,
    `<text class="diagram-muted" x="${left - 4}" y="${top + 4}" text-anchor="end">${maximum}</text>`
  );
  if (options.playheadHours !== null) {
    const x = left + (Math.min(24, Math.max(0, options.playheadHours)) / 24) * (right - left);
    parts.push(
      `<line class="diagram-signal" x1="${x.toFixed(1)}" y1="${top - 4}" x2="${x.toFixed(1)}" y2="${bottom}"/>`
    );
  }
  const offset = options.timeZone ? getZoneOffsetHours(options.timeZone) : 0;
  for (const hour of [0, 6, 12, 18, 24]) {
    const x = left + (hour / 24) * (right - left);
    parts.push(
      `<text class="diagram-muted" x="${x.toFixed(1)}" y="${bottom + 13}" text-anchor="middle">${String(hour % 24).padStart(2, '0')}</text>`
    );
    if (options.timeZone) {
      const local = (((hour + offset) % 24) + 24) % 24;
      parts.push(
        `<text class="diagram-ink" x="${x.toFixed(1)}" y="${bottom + 26}" text-anchor="middle">${String(Math.floor(local)).padStart(2, '0')}</text>`
      );
    }
  }
  parts.push(
    `<text class="diagram-muted" x="${left - 4}" y="${bottom + 13}" text-anchor="end">UTC</text>`
  );
  if (options.timeZone) {
    parts.push(
      `<text class="diagram-ink" x="${left - 4}" y="${bottom + 26}" text-anchor="end">${escapeText(getZoneAbbreviation(options.timeZone))}</text>`
    );
  }
  const total = totals.reduce((sum, value) => sum + value, 0);
  return {
    kind: 'diagram',
    width: DIAGRAM_WIDTH,
    height: options.timeZone ? 144 : 132,
    description: `Crossings per hour of the UTC day, ${total} in all, stacked by direction: ${options.firstLabel} and ${options.secondLabel}.`,
    svg: parts.join('')
  };
}
