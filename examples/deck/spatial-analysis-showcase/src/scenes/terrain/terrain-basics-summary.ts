// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Pure CPU summaries of terrain-basics: what is counted from the rasters the GPU wrote (class
 * counts and histograms of slope, the aspect rose, the decode check) and the structured hover
 * tooltip. Nothing here touches the GPU; the compute module reads the buffers back and calls in.
 */

import {getClassIndexOf, getClassLabel} from '../../cartography/class-table';
import type {ClassColor, ClassTable} from '../../cartography/types';
import type {RoseChartData, BarChartData} from '../chart-types';
import type {TooltipContent, TooltipRow} from '../scene';
import type {DemProbe} from './cpu-dem';
import {ASPECT_SLOPE_ALPHA, getAspectColor, SLOPE_BREAKS} from './terrain-palettes';
import type {BasicsOptions, BasicsTables} from './terrain-basics.style';

/** Bins of the slope histogram: one per degree from 0 to 90. */
export const SLOPE_HISTOGRAM_BINS = 90;

/** Sectors of the aspect rose. */
export const ASPECT_SECTORS = 16;

const COMPASS_WORDS = [
  'north',
  'north-east',
  'east',
  'south-east',
  'south',
  'south-west',
  'west',
  'north-west'
];

/** Compass word of an azimuth in degrees clockwise from north (eight points). */
export function getCompassWord(degrees: number): string {
  return COMPASS_WORDS[Math.round((((degrees % 360) + 360) % 360) / 45) % 8];
}

// ---------------------------------------------------------------------------------------------
// Counting
// ---------------------------------------------------------------------------------------------

/** What is counted from one slope raster. */
export type SlopeSummary = {
  /** Cells with a finite slope. */
  valid: number;
  /** Cells per class of the slope table (`SLOPE_BREAKS.length + 1` classes). */
  classCounts: number[];
  /** Cells per degree, `SLOPE_HISTOGRAM_BINS` bins over 0 to 90. */
  histogram: number[];
  /** The steepest finite value, degrees. */
  maximum: number;
  /** Raster index of the steepest cell, -1 without valid cells. */
  maximumIndex: number;
};

/** Counts the class and histogram of a float32 slope raster (degrees; NaN cells are skipped). */
export function summarizeSlope(
  values: Float32Array,
  breaks: readonly number[] = SLOPE_BREAKS
): SlopeSummary {
  const classCounts = new Array<number>(breaks.length + 1).fill(0);
  const histogram = new Array<number>(SLOPE_HISTOGRAM_BINS).fill(0);
  let valid = 0;
  let maximum = Number.NEGATIVE_INFINITY;
  let maximumIndex = -1;
  for (let index = 0; index < values.length; index++) {
    const value = values[index];
    if (!Number.isFinite(value)) continue;
    valid++;
    if (value > maximum) {
      maximum = value;
      maximumIndex = index;
    }
    let classIndex = 0;
    while (classIndex < breaks.length && value >= breaks[classIndex]) classIndex++;
    classCounts[classIndex]++;
    histogram[Math.min(SLOPE_HISTOGRAM_BINS - 1, Math.max(0, Math.floor(value)))]++;
  }
  return {valid, classCounts, histogram, maximum: valid > 0 ? maximum : Number.NaN, maximumIndex};
}

/** Share of the valid cells in the classes from `firstClass` up (a threshold at a class break). */
export function getShareFromClass(summary: SlopeSummary, firstClass: number): number {
  if (summary.valid === 0) return Number.NaN;
  let count = 0;
  for (let index = firstClass; index < summary.classCounts.length; index++) {
    count += summary.classCounts[index];
  }
  return count / summary.valid;
}

/** What is counted from the aspect and slope rasters. */
export type AspectSummary = {
  /** Cells steeper than the full-colour slope, the population of the rose. */
  steepCount: number;
  /** Steep cells per 22.5 degree sector; sector 0 is centred on north, clockwise. */
  sectorCounts: number[];
  /** Steep cells facing north-west, north or north-east (aspect from 292.5 to 67.5 degrees). */
  northCount: number;
};

/** Counts steep cells by the direction they face. Flat cells (aspect -1) and no-data are skipped. */
export function summarizeAspect(aspect: Float32Array, slope: Float32Array): AspectSummary {
  const sectorCounts = new Array<number>(ASPECT_SECTORS).fill(0);
  let steepCount = 0;
  let northCount = 0;
  const sectorWidth = 360 / ASPECT_SECTORS;
  for (let index = 0; index < aspect.length; index++) {
    const direction = aspect[index];
    const steepness = slope[index];
    if (!(direction >= 0) || !(steepness >= ASPECT_SLOPE_ALPHA.fullDegrees)) continue;
    steepCount++;
    sectorCounts[Math.floor((direction + sectorWidth / 2) / sectorWidth) % ASPECT_SECTORS]++;
    if (direction >= 292.5 || direction < 67.5) northCount++;
  }
  return {steepCount, sectorCounts, northCount};
}

/** What the decode check finds in the GPU heights. */
export type ElevationSummary = {
  valid: number;
  minimum: number;
  maximum: number;
  /** Largest absolute difference to the CPU decode, or null when the input was modified. */
  maximumDifference: number | null;
};

/** Valid count, range and the GPU-versus-CPU difference of the decoded heights. */
export function summarizeElevation(
  heights: Float32Array,
  reference: Float32Array | null
): ElevationSummary {
  let valid = 0;
  let minimum = Number.POSITIVE_INFINITY;
  let maximum = Number.NEGATIVE_INFINITY;
  let maximumDifference = 0;
  for (let index = 0; index < heights.length; index++) {
    const value = heights[index];
    if (!Number.isFinite(value)) continue;
    valid++;
    if (value < minimum) minimum = value;
    if (value > maximum) maximum = value;
    if (reference) {
      maximumDifference = Math.max(maximumDifference, Math.abs(value - reference[index]));
    }
  }
  return {
    valid,
    minimum,
    maximum,
    maximumDifference: reference ? maximumDifference : null
  };
}

// ---------------------------------------------------------------------------------------------
// Charts
// ---------------------------------------------------------------------------------------------

/** The slope histogram chart: one bar per degree, coloured by class, the decision threshold marked. */
export function getSlopeHistogramChart(
  summary: SlopeSummary,
  table: ClassTable,
  thresholdDegrees: number
): BarChartData {
  // The first class is transparent on the map; as a bar it is a muted neutral.
  const classColors: ClassColor[] = table.colors.map((color, index) =>
    index === 0 ? [150, 156, 164, 255] : [color[0], color[1], color[2], 255]
  );
  return {
    kind: 'histogram',
    values: summary.histogram,
    xDomain: [0, SLOPE_HISTOGRAM_BINS],
    breaks: table.breaks,
    classColors,
    markers: [{x: thresholdDegrees, label: `${thresholdDegrees}°`}],
    xLabel: 'Slope (degrees)',
    yLabel: 'Cells',
    formatX: value => `${value}°`,
    height: 120,
    title: 'Cells by slope, one bar per degree',
    description:
      'Histogram of slope in one-degree bins, coloured by the map classes, with the decision threshold marked.'
  };
}

/** The aspect rose: share of steep cells per sector, each sector in its aspect colour. */
export function getAspectRoseChart(summary: AspectSummary): RoseChartData {
  const sectorWidth = 360 / ASPECT_SECTORS;
  return {
    kind: 'rose',
    values: summary.sectorCounts.map(count =>
      summary.steepCount ? count / summary.steepCount : 0
    ),
    labels: ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
    colors: summary.sectorCounts.map((_, sector) => {
      const [red, green, blue] = getAspectColor(sector * sectorWidth);
      return [red, green, blue, 255] as const;
    }),
    height: 190,
    title: 'Share of steep cells by the direction they face',
    description:
      'Rose chart of steep cells in sixteen compass sectors, each coloured as on the map.'
  };
}

// ---------------------------------------------------------------------------------------------
// Tooltip
// ---------------------------------------------------------------------------------------------

/** The GPU probe of one hovered cell (the displayed product's value and the decoded height). */
export type HoverCell = {value: number; elevation: number; column: number; row: number};

/** Inputs of {@link getBasicsTooltip}. */
export type TooltipInput = {
  state: BasicsOptions;
  tables: BasicsTables;
  cell: HoverCell;
  probe: DemProbe;
  /** Ground and Mercator cell sizes in metres, for the other-model slope row. */
  groundCellMeters: number;
  mercatorCellMeters: number;
  /** Whether the cell lies inside an OpenStreetMap glacier outline. */
  onGlacier: boolean;
  /** `[west, south, east, north]` of the 3 x 3 window of the cell, for the highlight box. */
  getWindowBounds: (column: number, row: number) => [number, number, number, number];
};

const opaque = (color: ClassColor): ClassColor => [color[0], color[1], color[2], 255];
const MUTED_SWATCH: ClassColor = [150, 156, 164, 255];

function formatHeight(value: number): string {
  return value.toLocaleString('en-US', {maximumFractionDigits: 1, minimumFractionDigits: 1});
}

function getClassSwatch(table: ClassTable, value: number): ClassColor {
  const index = getClassIndexOf(table, value);
  if (index < 0) return MUTED_SWATCH;
  const color = table.colors[index];
  return (color[3] ?? 255) === 0 ? MUTED_SWATCH : opaque(color);
}

/** The structured tooltip of a hovered cell: the mapped value with its class swatch first. */
export function getBasicsTooltip(input: TooltipInput): TooltipContent | null {
  const {state, tables, cell, probe} = input;
  const {value, elevation, column, row} = cell;
  if (!Number.isFinite(elevation)) {
    return {
      title: 'No data',
      subtitle: 'Decoded height',
      rows: [{label: 'Height', value: 'no data', swatch: tables.decoded.noData?.color}],
      note: 'The decode marked this cell nodata: alpha 0, outside the valid range, or the wrong decoder.'
    };
  }
  const elevationRow: TooltipRow = {
    label: 'Elevation',
    value: formatHeight(elevation),
    unit: 'm'
  };
  if (state.view !== 'analysis') {
    const table = state.view === 'decoded' ? tables.decoded : tables.elevation;
    const classIndex = getClassIndexOf(table, elevation);
    return {
      title: 'Elevation',
      subtitle: getClassLabel(table, classIndex)
        ? `${getClassLabel(table, classIndex)} m class`
        : '',
      rows: [
        {...elevationRow, swatch: getClassSwatch(table, elevation), emphasis: true},
        ...(input.onGlacier && state.view === 'relief'
          ? [{label: 'Surface', value: 'glacier (OpenStreetMap outline)'}]
          : [])
      ]
    };
  }
  const horn = probe.hornAt(column, row);
  const window = horn.neighbours.map(height => Math.round(height).toLocaleString('en-US'));
  const windowRows: TooltipRow[] = [
    {label: 'Window, north row', value: window.slice(0, 3).join('   '), unit: 'm'},
    {label: 'Window, middle row', value: window.slice(3, 6).join('   '), unit: 'm'},
    {label: 'Window, south row', value: window.slice(6, 9).join('   '), unit: 'm'}
  ];
  const highlight = {kind: 'box' as const, bounds: input.getWindowBounds(column, row)};
  const slopeHere: TooltipRow = {
    label: 'Slope here',
    value: horn.slopeDeg.toFixed(1),
    unit: '°'
  };
  switch (state.product) {
    case 'slope': {
      if (!Number.isFinite(value)) {
        return {title: 'No slope', rows: [elevationRow], note: 'Edge cell or no data.'};
      }
      const classIndex = getClassIndexOf(tables.slope, value);
      const ratio = input.groundCellMeters / input.mercatorCellMeters;
      // The same face on the other cell model: tan(slope) scales with ground over Mercator size.
      const tangent = Math.tan((value * Math.PI) / 180);
      const otherSlope =
        (Math.atan(state.cellModel === 'ground' ? tangent / ratio : tangent * ratio) * 180) /
        Math.PI;
      return {
        title: `${value.toFixed(1)}° slope`,
        subtitle: `Class ${getClassLabel(tables.slope, classIndex)}`,
        rows: [
          {
            label: state.cellModel === 'ground' ? 'Slope, ground cells' : 'Slope, Mercator pixels',
            value: value.toFixed(1),
            unit: '°',
            swatch: getClassSwatch(tables.slope, value),
            emphasis: true
          },
          {
            label: state.cellModel === 'ground' ? 'On Mercator pixels' : 'On ground cells',
            value: otherSlope.toFixed(1),
            unit: '°'
          },
          {
            label: 'Faces',
            value: Number.isFinite(horn.aspectDeg)
              ? `${getCompassWord(horn.aspectDeg)} (${horn.aspectDeg.toFixed(0)}°)`
              : 'flat: no direction'
          },
          elevationRow,
          ...windowRows,
          {
            label: 'dz/dx, dz/dy',
            value: `${horn.dzdx.toFixed(2)}, ${horn.dzdy.toFixed(2)}`,
            unit: 'm per m'
          }
        ],
        highlight
      };
    }
    case 'aspect': {
      const faces = Number.isFinite(value) && value >= 0;
      const [red, green, blue] = faces ? getAspectColor(value) : MUTED_SWATCH;
      return {
        title: faces ? `Faces ${getCompassWord(value)}` : 'Flat: no direction',
        rows: [
          {
            label: 'Aspect',
            value: faces ? value.toFixed(0) : 'none',
            unit: faces ? '°' : undefined,
            swatch: [red, green, blue, 255],
            emphasis: true
          },
          slopeHere,
          elevationRow
        ],
        note:
          horn.slopeDeg < ASPECT_SLOPE_ALPHA.flatDegrees
            ? 'Flat ground is not coloured: its aspect is noise.'
            : undefined,
        highlight
      };
    }
    case 'tpi': {
      if (!Number.isFinite(value)) return {title: 'No TPI', rows: [elevationRow]};
      const reading =
        value > tables.tpi.breaks[2]
          ? 'above its 8 neighbours (ridge or knoll)'
          : value < tables.tpi.breaks[1]
            ? 'below its 8 neighbours (hollow)'
            : 'about level with its neighbours';
      return {
        title: `TPI ${value >= 0 ? '+' : ''}${value.toFixed(1)} m`,
        subtitle: reading,
        rows: [
          {
            label: 'TPI',
            value: `${value >= 0 ? '+' : ''}${value.toFixed(1)}`,
            unit: 'm',
            swatch: getClassSwatch(tables.tpi, value),
            emphasis: true
          },
          slopeHere,
          elevationRow
        ],
        highlight
      };
    }
    case 'tri':
    case 'vrm': {
      if (!Number.isFinite(value)) return {title: 'No ruggedness', rows: [elevationRow]};
      const isTri = state.product === 'tri';
      return {
        title: isTri ? `TRI ${value.toFixed(1)} m` : `VRM ${value.toFixed(3)}`,
        subtitle: `Class ${getClassLabel(tables.rugged, getClassIndexOf(tables.rugged, value))}`,
        rows: [
          {
            label: isTri ? 'TRI' : 'VRM',
            value: isTri ? value.toFixed(1) : value.toFixed(3),
            unit: isTri ? 'm' : 'index 0-1',
            swatch: getClassSwatch(tables.rugged, value),
            emphasis: true
          },
          slopeHere,
          elevationRow
        ],
        highlight
      };
    }
  }
}
