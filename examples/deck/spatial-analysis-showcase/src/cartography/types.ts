// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Declarative cartography types shared by scenes (`scene.ts` re-exports them), the Deck host and
 * the shell. Pure types: no DOM, no luma.gl. See `src/scenes/CARTOGRAPHY-GUIDE.md`.
 */

/** `[longitude, latitude]` in degrees. */
export type LngLat = readonly [number, number];

/** Page theme. */
export type CartographyTheme = 'light' | 'dark';

// ---------------------------------------------------------------------------------------------
// Basemap and figure-ground
// ---------------------------------------------------------------------------------------------

/**
 * A basemap style. The CARTO GL styles are free to use with attribution.
 *
 * - `positron`: light grey, quiet; the default ground of the light theme.
 * - `dark-matter`: near-black; the default ground of the dark theme.
 * - `voyager`: light with coloured roads, water and parks; for orientation-heavy steps.
 * - `none`: no tiles, a flat paper (light) or ink (dark) ground from the `--map-paper` token.
 */
export type BasemapStyleName = 'positron' | 'dark-matter' | 'voyager' | 'none';

/**
 * Where the basemap's place labels go.
 *
 * - `basemap`: drawn by the basemap, under the data (the default, today's behaviour).
 * - `above`: drawn by a second, label-only map stacked over the data with stronger halos, so
 *   place names stay legible over opaque analysis layers (choropleths, rasters).
 * - `none`: no basemap labels; the scene names places with annotations instead.
 */
export type BasemapLabels = 'basemap' | 'above' | 'none';

/**
 * Basemap treatment of a scene or a story step. Every field is optional; scene values are the
 * base and each step's `basemap` merges over the previous steps' (like cameras). A scene that
 * declares nothing gets exactly the old behaviour: `positron` in light, `dark-matter` in dark,
 * labels in the basemap, no dimming.
 */
export type BasemapSpec = {
  /**
   * `'auto'` (default) pairs the style with the page theme (positron / dark-matter). A style
   * name forces that style in both themes, for example `'dark-matter'` for an additive density
   * map that must glow even when the page is light. `{light, dark}` picks one per theme.
   */
  style?: 'auto' | BasemapStyleName | {light: BasemapStyleName; dark: BasemapStyleName};
  /** Label placement. Defaults to `'basemap'`. */
  labels?: BasemapLabels;
  /**
   * Fades the basemap toward the paper colour so the data reads as figure, `0` (off, default) to
   * `1` (no basemap visible). `0.3`-`0.5` is a good "recede" value under choropleths.
   */
  dim?: number;
  /** Removes colour from the basemap, `0` (default) to `1` (greyscale). Useful under voyager. */
  desaturate?: number;
  /**
   * Recolours the basemap ground (land, water, parks, roads, boundaries; the sheet colour under
   * `style: 'none'`). One palette for both themes, or `{light, dark}`. The `GROUNDS` presets in
   * `cartography/grounds.ts` fill this with the showcase's exact colours.
   */
  palette?: GroundPalette | {light?: GroundPalette; dark?: GroundPalette};
  /**
   * Which basemap label layers are kept (on the basemap or on the labels-above map). Defaults to
   * `'all'` (every symbol layer minus POI and house numbers, the round-1 behaviour).
   */
  labelPreset?: LabelPreset;
  /** Multiplies basemap label sizes (0.6-1.4). Below 1 also caps city names at 15 px above z11. */
  labelScale?: number;
  /**
   * Basemap place names to hide because the scene annotates the same place itself (exact `name`
   * match), so "Chicago" never appears twice.
   */
  suppressNames?: readonly string[];
  /**
   * Natural Earth land and coastline drawn by the basemap when `style` is `'none'` (oceans,
   * orbits, flows at world scale): `'land'` fills land and strokes the coast with the palette's
   * `land` and `boundary` colours. Ignored when tiles are drawn.
   */
  world?: 'land' | 'coast' | false;
  /** Graticule drawn under the data (on the basemap canvas). `true` = 10° lines. */
  graticule?: boolean | GraticuleSpec;
  /** Reference latitudes drawn under the data with the graticule style. */
  referenceLines?: readonly ReferenceLatitude[];
  /**
   * Cross-fade length in milliseconds when a step switches to this ground (default 600; the
   * shell uses 0 under `prefers-reduced-motion`).
   */
  transitionMs?: number;
};

/** Named ground presets (`cartography/grounds.ts`); see CARTOGRAPHY-GUIDE.md section 1. */
export type GroundName = 'paperCity' | 'paperSheet' | 'night' | 'abyss' | 'space' | 'relief';

/**
 * Basemap ground colours as CSS colours. Every field is optional: a missing field keeps the
 * style's own colour. `roadOpacity` 0 hides roads; `boundaryOpacity` 0 hides admin boundaries.
 */
export type GroundPalette = {
  /** Sheet colour: the map background under `style: 'none'` and the tile background otherwise. */
  background?: string;
  land?: string;
  water?: string;
  park?: string;
  road?: string;
  roadOpacity?: number;
  boundary?: string;
  boundaryOpacity?: number;
  /** Graticule and reference-line colour (CSS colour with alpha). */
  graticule?: string;
};

/**
 * Basemap label whitelist:
 *
 * - `orientation`: cities, towns, suburbs, water names, states (no roads, no POI).
 * - `places-only`: settlement names only.
 * - `water-only`: water names only.
 * - `streets`: `orientation` plus road names from z13 (routing and network stories).
 * - `all`: every symbol layer minus POI and house numbers (default).
 * - `none`: no basemap labels (same as `labels: 'none'`).
 */
export type LabelPreset = 'orientation' | 'places-only' | 'water-only' | 'streets' | 'all' | 'none';

/** Graticule options. */
export type GraticuleSpec = {
  /** Spacing in degrees. Default 10. */
  stepDegrees?: number;
  /** Line opacity 0-1 (multiplies the palette's `graticule` colour). */
  opacity?: number;
  /** Line width in CSS pixels. Default 0.5. */
  widthPixels?: number;
};

/** A named reference latitude: the equator, the tropics (±23.44°), the polar circles (±66.56°). */
export type ReferenceLatitude = 'equator' | 'tropics' | 'polar-circles';

/** A {@link BasemapSpec} with every field resolved for one theme. */
export type ResolvedBasemap = {
  style: BasemapStyleName;
  labels: BasemapLabels;
  dim: number;
  desaturate: number;
  /** The palette for this theme (empty when none is set). */
  palette: GroundPalette;
  labelPreset: LabelPreset;
  labelScale: number;
  suppressNames: readonly string[];
  world: 'land' | 'coast' | false;
  graticule: Required<GraticuleSpec> | null;
  referenceLines: readonly ReferenceLatitude[];
  transitionMs: number;
};

// ---------------------------------------------------------------------------------------------
// Map furniture
// ---------------------------------------------------------------------------------------------

/** Units of the scale bar. `'nautical'` shows nautical miles (ships, aircraft). */
export type ScaleBarUnits = 'metric' | 'imperial' | 'both' | 'nautical';

/** Scale bar options. */
export type ScaleBarSpec = {
  units?: ScaleBarUnits;
  /**
   * Measure at this latitude instead of the map centre, and say so ("Scale at 39° N"). Use it on
   * national Web Mercator maps where the scale varies by 1.6x between the south and the north.
   */
  latitude?: number;
  /** Hide the bar below this zoom (national and global frames). */
  minZoom?: number;
  /**
   * Distances in metres marked with a labelled tick on the bar, for example the search radius
   * or the bandwidth of the step, so the parameter and the bar read together.
   */
  ticks?: readonly number[];
};

/** Title cartouche options. */
export type CartoucheSpec = {
  /** Line 1: the claim or the question, nine words or fewer. Defaults to the scene title. */
  title?: string;
  /** Line 2: variable, unit, method, N, vintage. Defaults to nothing. */
  subtitle?: string;
  /**
   * The standing sample line of the dataset ("42 birds, 101 animal-years, 4 tagging sites").
   * Set it once on the scene; it never changes between steps. Hidden on phones.
   */
  sample?: string;
  /** Short honesty chips: `'Simulated'`, `'Modelled'`, `'Sampled'`, `'Scheduled, not observed'`. */
  chips?: readonly string[];
};

/** A clock drawn on the map for playback stories (bound to a time option). */
export type MapClockSpec = {
  /** Option id holding the playhead. */
  option: string;
  /**
   * How the option value maps to an instant: `'epoch-seconds'` / `'epoch-ms'` (absolute),
   * or `{origin: ISO date string, unit: 'seconds' | 'minutes' | 'hours' | 'days'}` (relative).
   */
  time:
    | 'epoch-seconds'
    | 'epoch-ms'
    | {origin: string; unit: 'seconds' | 'minutes' | 'hours' | 'days'};
  /**
   * IANA zones shown, first one large (default `['UTC']`). The UTC line is always shown with its
   * zone name; local zones say their abbreviation.
   */
  zones?: readonly string[];
  /** `'datetime'` (default), `'date'`, `'time'`. */
  show?: 'datetime' | 'date' | 'time';
  /** Thin progress strip under the clock over this option range. */
  progress?: readonly [number, number];
};

/**
 * Declarative map furniture of a scene or a step (steps merge over the scene and earlier steps).
 * A scene that declares nothing shows no furniture, as before.
 */
export type FurnitureSpec = {
  /** Scale bar: `true` (metric) or options. Use it whenever the story talks about distances. */
  scaleBar?: boolean | ScaleBarSpec;
  /**
   * North arrow. `'auto'` (default once `furniture` is declared) shows it only when the camera
   * has a bearing or pitch; `'always'` keeps it; `'never'` hides it.
   */
  northArrow?: 'auto' | 'always' | 'never';
  /**
   * Title cartouche in the map corner. `true` shows the scene title with the step title as
   * subtitle; an object overrides either line; `false` hides it.
   */
  title?: boolean | CartoucheSpec;
  /**
   * Source credit line under the map. `true` joins the `attribution` of the scene's datasets;
   * a string replaces it. Basemap attribution stays in the corner control.
   */
  credit?: boolean | string;
  /**
   * A one-line honesty note under the credit, for example the Web Mercator caveat of a national
   * map (`mercatorCaveat()` in `cartography/projection-notes.ts`) or "Counts measure where people
   * look". Keep it under ~90 characters.
   */
  caveat?: string;
  /** A clock for playback stories. */
  clock?: MapClockSpec | false;
};

// ---------------------------------------------------------------------------------------------
// Annotations
// ---------------------------------------------------------------------------------------------

/**
 * Ink of an annotation: `ink` (default, text colour), `accent` (the story's finding colour),
 * `muted` (context places, 0.55), `water` blue, or `signal` (`--map-signal`: the thing the reader
 * controls, such as a parameter ring, a probe or an origin).
 */
export type AnnotationTone = 'ink' | 'accent' | 'muted' | 'water' | 'signal';

/** Candidate label positions around an anchor (Imhof order NE, SE, NW, SW is tried by `'auto'`). */
export type LabelAnchor = 'auto' | 'ne' | 'se' | 'nw' | 'sw' | 'n' | 's' | 'e' | 'w';

/** Fields every annotation accepts. */
export type AnnotationBase = {
  /** Higher priority wins collisions. Defaults to 0; callouts and rings are never culled. */
  priority?: number;
  /** Shown only from this zoom (inclusive). */
  minZoom?: number;
  /** Shown only below this zoom (exclusive). */
  maxZoom?: number;
  /** Ink. */
  tone?: AnnotationTone;
  /**
   * Stable identity. Annotations with an `id` keep their DOM nodes when the list is replaced, so
   * a label that follows a moving entity (a satellite, a storm) can be updated every frame.
   */
  id?: string;
  /**
   * Shown only while the annotation time set by `ctx.setAnnotationTime(t)` is inside
   * `[start, end)` (same units as the scene's time option). Without a time set, always shown.
   */
  timeRange?: readonly [number, number];
};

/**
 * A map annotation, drawn above the data in screen space with a halo, re-positioned on every
 * camera change, theme-aware, and thinned by greedy collision avoidance (higher `priority`
 * first). Coordinates are `[longitude, latitude]`.
 */
export type MapAnnotation = AnnotationBase &
  (
    | {
        /**
         * A named point: a small marker and a label. With `offset` the label sits that many CSS
         * pixels away (`[dx, dy]`, y down) and a thin leader line joins it to the marker.
         */
        kind: 'point';
        coordinate: LngLat;
        text: string;
        /** Optional second line in a smaller size, for example a value. */
        detail?: string;
        offset?: readonly [number, number];
        /** Marker style. Defaults to `'dot'`. */
        marker?: 'dot' | 'ring' | 'none';
        /**
         * Label position when there is no `offset`: `'auto'` (default) tries NE, SE, NW, SW at a
         * 6 px gap and keeps the first that does not collide.
         */
        anchor?: LabelAnchor;
        /** Settlement rank: `'subject'` (12.5 px, 600) or `'context'` (11 px, 500, muted). */
        rank?: 'subject' | 'context';
      }
    | {
        /** A region or neighbourhood name: small caps, letter-spaced, no marker. */
        kind: 'area';
        coordinate: LngLat;
        text: string;
        /** Relative size, `'small'` / `'medium'` (default) / `'large'`. */
        size?: 'small' | 'medium' | 'large';
      }
    | {
        /** A water or natural feature: italic, water-blue by default, no marker. */
        kind: 'water';
        coordinate: LngLat;
        text: string;
        size?: 'small' | 'medium' | 'large';
      }
    | {
        /**
         * A landform (peak, ridge, glacier, canyon): Serif italic in ink, with an optional
         * elevation line in Sans (`elevationMeters` renders as "4,478 m"). Marker: a small
         * triangle for peaks (`marker: 'peak'`), none otherwise.
         */
        kind: 'landform';
        coordinate: LngLat;
        text: string;
        elevationMeters?: number;
        marker?: 'peak' | 'none';
        size?: 'small' | 'medium' | 'large';
        anchor?: LabelAnchor;
      }
    | {
        /**
         * A finding note: a paper plate (headline + optional body, max 200 px wide) joined to
         * the feature by a leader that ends in a 3 px dot. At most three per step. Headlines are
         * sentences with a number and a unit read from a readout.
         */
        kind: 'note';
        coordinate: LngLat;
        /** Bold headline, for example "41 % of trips". */
        title: string;
        /** Optional body line. */
        text?: string;
        /** Where the plate sits relative to the feature. Defaults to `'auto'`. */
        anchor?: LabelAnchor;
        /** Leader length in CSS pixels. Default 36. */
        distance?: number;
      }
    | {
        /**
         * A numbered marker (18 px ink circle, white numeral). Use when more than three features
         * are discussed or on phones; the card or legend lists what each number is.
         */
        kind: 'marker';
        coordinate: LngLat;
        number: number | string;
        /** Accessible name and hover title. */
        text?: string;
      }
    | {
        /**
         * A district outline: a 2 px accent stroke with a 4 px halo glow, no fill, around one or
         * more rings (outer rings of a polygon in `[lng, lat]`). Optional label at `labelAt` or the
         * ring's centroid.
         */
        kind: 'outline';
        rings: readonly (readonly LngLat[])[];
        text?: string;
        labelAt?: LngLat;
        dashed?: boolean;
      }
    | {
        /** A dimension line: two end ticks and a centred label ("2.4 km"). */
        kind: 'dimension';
        from: LngLat;
        to: LngLat;
        /** Defaults to the geodesic length, formatted in m or km. */
        text?: string;
      }
    | {
        /** A bracket along a stretch (a rupture, a reach), with a label on the outer side. */
        kind: 'bracket';
        from: LngLat;
        to: LngLat;
        text?: string;
        /** Which side of the from-to direction the bracket opens to. Default `'left'`. */
        side?: 'left' | 'right';
      }
    | {
        /** A star symbol (a mainshock, a record), with an optional label. */
        kind: 'star';
        coordinate: LngLat;
        text?: string;
        /** Outer radius in CSS pixels. Default 7. */
        radiusPixels?: number;
      }
    | {
        /**
         * A polyline in screen space above the data: a probe line, a great-circle arc, a
         * latitude limit, a straight-line comparison. Optional label at the midpoint.
         */
        kind: 'line';
        coordinates: readonly LngLat[];
        text?: string;
        dashed?: boolean;
        /** Stroke width in CSS pixels. Default 1.5. */
        widthPixels?: number;
      }
    | {
        /**
         * A dashed frame where clipping creates an apparent edge ("Data ends here"): a rectangle
         * `[west, south, east, north]` or a ring.
         */
        kind: 'frame';
        bounds?: readonly [number, number, number, number];
        ring?: readonly LngLat[];
        text?: string;
      }
    | {
        /** A circle of a true ground radius around a coordinate (a search radius, a buffer). */
        kind: 'ring';
        coordinate: LngLat;
        radiusMeters: number;
        /** Label at the top of the ring, with the unit ("epsilon 150 m"). */
        text?: string;
        /** Dashed when the ring is a parameter, solid when it is a result. */
        dashed?: boolean;
        /**
         * Draw a geodesic circle (true great-circle distance; use above ~50 km, near the poles,
         * or across the antimeridian). Default: a local circle, exact for city radii.
         */
        geodesic?: boolean;
      }
    | {
        /** An arrow from one coordinate to another (a flow, "this way"), optional label at `from`. */
        kind: 'arrow';
        from: LngLat;
        to: LngLat;
        text?: string;
      }
    | {
        /** The accent bubble of a step `callout`. Never culled. */
        kind: 'callout';
        coordinate: LngLat;
        text: string;
      }
  );

// ---------------------------------------------------------------------------------------------
// Highlight geometry (tooltips, linked highlighting)
// ---------------------------------------------------------------------------------------------

/**
 * Geometry outlined on the map above the data: the hovered feature of a tooltip, or features a
 * chart, legend or list points at (`ctx.setHighlight`). Drawn as an achromatic 2 px ink core with
 * a 4 px ground-colour halo (the selection convention), or in `--map-signal` with `tone: 'signal'`.
 */
export type MapHighlight = (
  | {kind: 'point'; coordinate: LngLat; radiusPixels?: number}
  | {kind: 'circle'; coordinate: LngLat; radiusMeters: number}
  | {kind: 'polygon'; rings: readonly (readonly LngLat[])[]}
  | {kind: 'line'; coordinates: readonly LngLat[]}
  | {kind: 'box'; bounds: readonly [number, number, number, number]}
) & {
  tone?: 'ink' | 'signal';
  /** A one-shot expanding ring that draws the eye (programmatic selection). */
  pulse?: boolean;
};

// ---------------------------------------------------------------------------------------------
// Class tables
// ---------------------------------------------------------------------------------------------

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
