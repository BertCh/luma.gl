// Copied from examples/deck/spatial-analysis-showcase/src/cartography/types.ts; the showcase still owns its copy until it switches to this module.
// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** RGBA colour with 0-255 channels (alpha optional, default 255). */
export type ClassColor = readonly [number, number, number, number?];

/**
 * One classification, read by the layer (`classBreaks` + `classColors`), the `classes` legend
 * (`table`), tooltips and histogram ticks, so they cannot drift apart. Build it with
 * `makeClassTable` (`cartography/class-table.ts`).
 */
export type ClassTable = {
  /** Interior class breaks, ascending, at most 15 (16 classes). */
  breaks: readonly number[];
  /** One colour per class, low class first; alpha 0 makes a class transparent. */
  colors: readonly ClassColor[];
  /** One label per class (generated from the breaks when omitted). */
  labels?: readonly string[];
  /** Value unit, for the legend title and tooltips. */
  unit?: string;
  /** Data `[min, max]`, for the outer legend labels. */
  extent?: readonly [number, number];
  /** Classification method shown in the legend ("Natural breaks (Jenks), GVF 0.91"). */
  method?: string;
  /** Classes drawn with a hatch (suppressed, low n, not significant), by index. */
  hatched?: readonly number[];
  /** The no-data swatch of the legend: colour, label and (optional) count. */
  noData?: {color?: ClassColor; label?: string; count?: number; hatched?: boolean};
  /** Value formatter for generated labels. */
  format?: (value: number) => string;
};
