// Copied from examples/deck/spatial-analysis-showcase/src/scenes/scene.ts; the showcase still owns its copy until it switches to this module.
// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ClassColor as LegendColor, ClassTable} from './cartography-types';

/** Local class legend contract used by the copied class-table adapter. */
export type LegendSpec = {
  id?: string;
  basis?: string;
  /**
   * Classed (stepped) colour: `breaks.length + 1` classes, as drawn by a layer with the same
   * `classBreaks` and `colormap`. Swatch colours come from `getClassColors(ramp, n, reverse)`
   * unless `colors` is given (one per class).
   */
  kind: 'classes';
  title: string;
  unit?: string;
  /** One colour per class, low class first. Overrides `ramp`. */
  colors?: readonly LegendColor[];
  /**
   * Interior class breaks, ascending: the same array the layer gets as `classBreaks`.
   * Required unless `table` is given.
   */
  breaks?: readonly number[];
  /** Data `[min, max]` for the outer labels. Without it the ends read `< b0` and `>= bN`. */
  extent?: readonly [number, number];
  /** One label per class, replacing the generated ranges (for example `'Hot 99%'`). */
  labels?: readonly string[];
  /** Features per class, shown next to each class (and as bar lengths in `'list'` layout). */
  counts?: readonly number[];
  /**
   * `'bar'` (default without `counts`): equal blocks with the break values between them.
   * `'list'` (default with `counts`): one row per class with range, swatch and count.
   */
  layout?: 'bar' | 'list';
  /** Number formatter for the break labels. */
  format?: (value: number) => string;
  /** Optional distribution under the classes: bin counts spanning `extent` evenly. */
  histogram?: readonly number[];
  /** Short note under the classes, for example the classification method. */
  note?: string;
  /**
   * The class table the layer draws (`makeClassTable`). When given, `breaks`, `colors`,
   * `labels`, `extent`, `unit`, `hatched` and `noData` default to the table's.
   */
  table?: ClassTable;
  /** Classes drawn hatched, by index. */
  hatched?: readonly number[];
  /** A "No data" swatch after the classes (always add one when rows can be missing). */
  noData?: {color?: LegendColor; label?: string; count?: number; hatched?: boolean};
  /**
   * Class swatches act as filters: hover isolates a class, click locks it, Esc shows all.
   * The scene receives `onLegendFilter(id, classIndices | null)` and dims the others.
   */
  interactive?: boolean;
};
