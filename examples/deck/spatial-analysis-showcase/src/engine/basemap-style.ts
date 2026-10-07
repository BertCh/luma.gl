// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * MapLibre style transforms of the showcase basemap: pure functions from a
 * {@link ResolvedBasemap} to `transformStyle` callbacks (or a whole style), so the Deck host only
 * decides which to apply and when.
 *
 * - Ground transform: recolours the CARTO styles (background, land, water, parks, roads,
 *   boundaries) from a {@link GroundPalette}, draws the graticule and reference latitudes.
 * - Label transform: keeps a whitelist of the CARTO symbol layers ({@link LabelPreset}), re-inks
 *   them for the ground, adds halos, scales, and hides names the scene annotates itself.
 * - Flat style: the `none` ground, a background plus optional Natural Earth land and coast.
 *
 * Layer ids are those of the CARTO `positron`, `dark-matter` and `voyager` GL styles (identical
 * in all three); filters and sizes there use the legacy function syntax, which these transforms
 * read and write without converting.
 */

import type {GeoJSONSourceSpecification, LayerSpecification, StyleSpecification} from 'maplibre-gl';
import {formatCssColor, parseCssColor} from '../cartography/grounds';
import type {
  BasemapStyleName,
  CartographyTheme,
  GroundPalette,
  ReferenceLatitude,
  ResolvedBasemap
} from '../cartography/types';
import {getDataFileUrl} from '../data/loaders';

/** CARTO GL style URLs (free with attribution, which the attribution control shows). */
export const BASEMAP_STYLE_URLS: Record<Exclude<BasemapStyleName, 'none'>, string> = {
  positron: 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
  'dark-matter': 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
  voyager: 'https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json'
};

/** A style with no layers (the map before its first style, and the `none` ground without a palette). */
export const EMPTY_STYLE: StyleSpecification = {version: 8, sources: {}, layers: []};

/** The callback MapLibre's `setStyle` calls with the previous and the freshly loaded style. */
export type StyleTransform = (
  previous: StyleSpecification | undefined,
  next: StyleSpecification
) => StyleSpecification;

/** Natural Earth files drawn by `world` (dataset `natural-earth`, see `data/`). */
const WORLD_LAND_FILE = 'ne_110m_land.geojson';
const WORLD_COAST_FILE = 'ne_110m_coastline.geojson';

/** Label layers dropped from a labels-above map: point-of-interest and house-number clutter. */
const LABEL_CLUTTER = /housenumber|poi/i;

/** Halo colour on a ground with no palette, per ground luminance (alpha 0.92). */
const DEFAULT_HALOS: Record<CartographyTheme, string> = {
  light: 'rgba(255, 255, 255, 0.92)',
  dark: 'rgba(9, 12, 16, 0.92)'
};

/** Label ink per ground luminance (SYNTHESIS 1.5, cross report 04 section 4.3). */
const LABEL_INK: Record<
  CartographyTheme,
  Record<'city' | 'secondary' | 'water' | 'road', string>
> = {
  light: {city: '#2B3440', secondary: '#4A5663', water: '#35729F', road: '#6B7684'},
  dark: {city: '#E8EDF2', secondary: '#AAB6C3', water: '#7FB3D9', road: '#8F9BA9'}
};

/** Default graticule colour per ground luminance. */
const DEFAULT_GRATICULE_COLORS: Record<CartographyTheme, string> = {
  light: 'rgba(31, 41, 51, 0.08)',
  dark: 'rgba(255, 255, 255, 0.06)'
};

// Layer id patterns of the CARTO styles ------------------------------------------------------

/** Background fills tinted by land use: take the palette's `land`. */
const LAND_LAYERS = /^(landuse|landuse_residential)$/;
/** Landcover and park polygons: take the palette's `park`. */
const PARK_LAYERS = /^(landcover|park_national_park|park_nature_reserve)$/;
/** Water polygons and lines: take the palette's `water`. */
const WATER_LAYERS = /^(water|water_shadow|waterway)$/;
/** Admin boundary lines. `boundary_country_outline` is the 8 px glow under them. */
const BOUNDARY_LAYERS = /^boundary_(county|state|country_inner)$/;
const BOUNDARY_GLOW = 'boundary_country_outline';
/** Road, tunnel, bridge, rail and aeroway lines (the whole transport network of the ground). */
const ROAD_LAYERS = /^(road|tunnel|bridge)_|^(rail|rail_dash|aeroway-runway|aeroway-taxiway)$/;
/** Road casings (the outline under the fill), removed when roads are recoloured to a hairline. */
const ROAD_CASINGS = /_case/;
const BUILDING_LAYER = 'building';
const BUILDING_TOP_LAYER = 'building-top';

/** City and capital names (and towns): the strongest ink. */
const CITY_LABELS = /^(place_(city|capital)|place_town$)/;
/** Smaller settlements: the secondary ink. */
const MINOR_PLACE_LABELS = /^place_(suburbs|villages|hamlet)$/;
const SUBURB_AND_STATE_LABELS = /^place_(suburbs|state)$/;
const REGION_LABELS = /^place_(state|country|continent)/;
const WATER_LABELS = /^(watername_|waterway_label)/;
const ROAD_LABELS = /^roadname_/;

/** Zoom below which a recoloured road network is hidden, and where it reaches full strength. */
const ROAD_HIDDEN_BELOW_ZOOM = 11.5;
const ROAD_FULL_AT_ZOOM = 12.5;
/** Width factor of recoloured roads (a hairline next to the style's own widths). */
const ROAD_HAIRLINE_SCALE = 0.6;

/** Halo widths in pixels: place labels on the labels canvas, and the thin italic water names. */
const HALO_WIDTH = 1.6;
const WATER_HALO_WIDTH = 1.2;
const HALO_BLUR = 0.3;
/** Letter-spacing (em) of suburb and state names. */
const WIDE_LETTER_SPACING = 0.12;
/** City names shrink to this size above zoom 11 when the labels are scaled below 1. */
const CITY_LABEL_CAP_PIXELS = 15;
const CITY_LABEL_CAP_ZOOM = 11;
/** First zoom of road names in the `streets` preset. */
const STREET_LABEL_MIN_ZOOM = 13;

type PaintLayer = LayerSpecification & {
  paint?: Record<string, unknown>;
  layout?: Record<string, unknown>;
  filter?: unknown;
};

// ---------------------------------------------------------------------------------------------
// Keys
// ---------------------------------------------------------------------------------------------

/**
 * Cache key of everything that changes the base map's style (not its CSS dim and desaturate), so
 * the host skips redundant reloads. `includeLabels` is whether the labels stay in this map.
 */
export function getGroundStyleKey(
  basemap: ResolvedBasemap,
  tone: CartographyTheme,
  includeLabels: boolean
): string {
  return JSON.stringify([
    basemap.style,
    tone,
    basemap.palette,
    basemap.world,
    basemap.graticule,
    basemap.referenceLines,
    includeLabels ? getLabelSettings(basemap) : null
  ]);
}

/** Cache key of everything that changes the labels-above map's style. */
export function getLabelStyleKey(
  basemap: ResolvedBasemap,
  tone: CartographyTheme,
  styleName: Exclude<BasemapStyleName, 'none'>
): string {
  return JSON.stringify([styleName, tone, basemap.palette, getLabelSettings(basemap)]);
}

function getLabelSettings(basemap: ResolvedBasemap) {
  return [basemap.labels, basemap.labelPreset, basemap.labelScale, basemap.suppressNames];
}

/** Whether the `none` ground needs a generated style (otherwise the map stays hidden and empty). */
export function needsFlatStyle(basemap: ResolvedBasemap): boolean {
  const {palette} = basemap;
  return Boolean(
    palette.background ||
      palette.land ||
      basemap.world ||
      basemap.graticule ||
      basemap.referenceLines.length
  );
}

// ---------------------------------------------------------------------------------------------
// Colour helpers
// ---------------------------------------------------------------------------------------------

/** The label halo colour: the ground (palette land or background) at 0.92, else white or ink. */
export function getLabelHaloColor(palette: GroundPalette, tone: CartographyTheme): string {
  const color = parseCssColor(palette.land ?? palette.background);
  return color ? formatCssColor(color, 0.92) : DEFAULT_HALOS[tone];
}

function hasCustomGround(palette: GroundPalette): boolean {
  return Boolean(palette.background || palette.land);
}

// ---------------------------------------------------------------------------------------------
// Ground transform
// ---------------------------------------------------------------------------------------------

/**
 * Builds the transform of a base map that shows CARTO tiles: recolours the ground from the
 * palette, adds the graticule and reference lines under the labels, and either drops the symbol
 * layers (`includeLabels` false: they are on the labels-above map) or keeps them through the
 * label whitelist. With no palette, graticule or label settings the result is the style itself
 * minus (optionally) its labels, today's look.
 */
export function createGroundTransform(
  basemap: ResolvedBasemap,
  tone: CartographyTheme,
  includeLabels: boolean
): StyleTransform {
  return (_previous, next) => {
    const {palette} = basemap;
    const decorations = getDecorations(basemap, tone);
    const labelOptions = {...getLabelOptions(basemap, tone), reink: hasCustomGround(palette)};
    const layers: LayerSpecification[] = [];
    let decorated = false;
    const decorate = () => {
      if (decorated) return;
      decorated = true;
      layers.push(...decorations.layers);
    };
    for (const layer of next.layers) {
      if (layer.type === 'symbol') {
        if (!includeLabels) continue;
        const label = processLabel(layer as PaintLayer, {
          ...labelOptions,
          // Under the data the style is the reader's default unless the scene asked otherwise.
          dropClutter: basemap.labelPreset !== 'all'
        });
        if (label) {
          decorate();
          layers.push(label);
        }
        continue;
      }
      const ground = recolourLayer(layer as PaintLayer, palette);
      if (ground) layers.push(ground);
    }
    decorate();
    return {...next, sources: {...next.sources, ...decorations.sources}, layers};
  };
}

/** Paint property prefix each layer type accepts; anything else would invalidate the style. */
const PAINT_PREFIX: Record<string, string> = {
  background: 'background-',
  fill: 'fill-',
  line: 'line-'
};

/** Merges paint properties, dropping those the layer's type does not have (MapLibre rejects them). */
function setPaint(layer: PaintLayer, paint: Record<string, unknown>): PaintLayer {
  const prefix = PAINT_PREFIX[layer.type];
  const allowed = Object.fromEntries(
    Object.entries(paint).filter(([name]) => prefix !== undefined && name.startsWith(prefix))
  );
  return {...layer, paint: {...layer.paint, ...allowed}} as PaintLayer;
}

/** A fade from hidden to `opacity` between two zooms (roads wake up at neighbourhood scale). */
function getRoadOpacity(opacity: number): unknown {
  return [
    'interpolate',
    ['linear'],
    ['zoom'],
    ROAD_HIDDEN_BELOW_ZOOM,
    0,
    ROAD_FULL_AT_ZOOM,
    opacity
  ];
}

/**
 * Applies the palette to one non-symbol layer; returns `null` to remove it. A palette field that
 * is missing keeps the style's own paint.
 */
function recolourLayer(layer: PaintLayer, palette: GroundPalette): PaintLayer | null {
  const {id} = layer;
  if (layer.type === 'background') {
    const color = palette.background ?? palette.land;
    return color ? setPaint(layer, {'background-color': color}) : layer;
  }
  if (id === BOUNDARY_GLOW) {
    return palette.boundary !== undefined || palette.boundaryOpacity === 0 ? null : layer;
  }
  if (BOUNDARY_LAYERS.test(id)) {
    if (palette.boundaryOpacity === 0) return null;
    return setPaint(layer, {
      ...(palette.boundary ? {'line-color': palette.boundary} : {}),
      ...(palette.boundaryOpacity !== undefined ? {'line-opacity': palette.boundaryOpacity} : {})
    });
  }
  if (WATER_LAYERS.test(id)) {
    if (!palette.water) return layer;
    return setPaint(
      layer,
      layer.type === 'line' ? {'line-color': palette.water} : {'fill-color': palette.water}
    );
  }
  if (PARK_LAYERS.test(id)) {
    return palette.park ? setPaint(layer, {'fill-color': palette.park}) : layer;
  }
  if (LAND_LAYERS.test(id)) {
    return palette.land ? setPaint(layer, {'fill-color': palette.land}) : layer;
  }
  if (id === BUILDING_LAYER || id === BUILDING_TOP_LAYER) {
    if (!palette.road) return layer;
    return id === BUILDING_LAYER
      ? setPaint(layer, {'fill-color': palette.road})
      : setPaint(layer, {
          'fill-color': palette.land ?? palette.background ?? palette.road,
          'fill-outline-color': palette.road
        });
  }
  if (ROAD_LAYERS.test(id)) return recolourRoad(layer, palette);
  return layer;
}

/**
 * Roads: removed at `roadOpacity` 0; with a road colour or opacity they become a quiet network
 * (casings removed, hairline widths, hidden below zoom 12 and fading in to `roadOpacity`).
 */
function recolourRoad(layer: PaintLayer, palette: GroundPalette): PaintLayer | null {
  if (palette.roadOpacity === 0) return null;
  if (layer.type !== 'line') return layer;
  if (palette.road === undefined && palette.roadOpacity === undefined) return layer;
  if (palette.road !== undefined && ROAD_CASINGS.test(layer.id)) return null;
  const paint: Record<string, unknown> = {
    'line-opacity': getRoadOpacity(palette.roadOpacity ?? 1),
    'line-width': scaleZoomValue(layer.paint?.['line-width'], ROAD_HAIRLINE_SCALE)
  };
  if (palette.road !== undefined) {
    paint['line-color'] =
      layer.id === 'rail_dash'
        ? (palette.background ?? palette.land ?? palette.road)
        : palette.road;
  }
  return setPaint(layer, paint);
}

// ---------------------------------------------------------------------------------------------
// Label transform
// ---------------------------------------------------------------------------------------------

type LabelOptions = {
  preset: ResolvedBasemap['labelPreset'];
  scale: number;
  suppressNames: readonly string[];
  tone: CartographyTheme;
  halo: string;
  reink: boolean;
  dropClutter: boolean;
};

function getLabelOptions(basemap: ResolvedBasemap, tone: CartographyTheme): LabelOptions {
  return {
    preset: basemap.labelPreset,
    scale: basemap.labelScale,
    suppressNames: basemap.suppressNames,
    tone,
    halo: getLabelHaloColor(basemap.palette, tone),
    reink: true,
    dropClutter: true
  };
}

/**
 * Builds the transform of the labels-above map: keeps only the symbol layers of the label preset
 * and drops everything else (the ground is on the base map). Text is re-inked for the ground
 * (`tone`), haloed with the ground colour at 0.92 (1.6 px, 1.2 px on water), scaled by
 * `labelScale` and stripped of `suppressNames`.
 */
export function createLabelTransform(
  basemap: ResolvedBasemap,
  tone: CartographyTheme
): StyleTransform {
  return (_previous, next) => {
    const options = getLabelOptions(basemap, tone);
    const layers: LayerSpecification[] = [];
    for (const layer of next.layers) {
      if (layer.type !== 'symbol') continue;
      const label = processLabel(layer as PaintLayer, options);
      if (label) layers.push(label);
    }
    return {...next, layers};
  };
}

/** Whether a CARTO symbol layer belongs to a label preset. */
function isInPreset(id: string, preset: ResolvedBasemap['labelPreset']): boolean {
  const isOrientation =
    CITY_LABELS.test(id) ||
    /^place_(suburbs|villages)$/.test(id) ||
    (REGION_LABELS.test(id) && id !== 'place_continent') ||
    WATER_LABELS.test(id);
  switch (preset) {
    case 'none':
      return false;
    case 'orientation':
      return isOrientation;
    case 'places-only':
      return CITY_LABELS.test(id) || MINOR_PLACE_LABELS.test(id);
    case 'water-only':
      return WATER_LABELS.test(id);
    case 'streets':
      return isOrientation || ROAD_LABELS.test(id);
    default:
      return true;
  }
}

/** Processes one symbol layer for a label preset; `null` when the preset drops it. */
function processLabel(layer: PaintLayer, options: LabelOptions): PaintLayer | null {
  const {id} = layer;
  if (!isInPreset(id, options.preset)) return null;
  if (options.preset === 'all' && options.dropClutter && LABEL_CLUTTER.test(id)) return null;
  const untouched =
    options.scale === 1 &&
    !options.reink &&
    !options.suppressNames.length &&
    options.preset !== 'streets';
  if (untouched) return layer;

  const layout: Record<string, unknown> = {...layer.layout};
  const paint: Record<string, unknown> = {...layer.paint};
  const isCity = CITY_LABELS.test(id) && id !== 'place_town';
  if (options.scale !== 1) {
    layout['text-size'] = scaleTextSize(
      layout['text-size'],
      options.scale,
      isCity && options.scale < 1 ? CITY_LABEL_CAP_PIXELS : undefined
    );
  }
  if (options.reink) {
    const water = WATER_LABELS.test(id);
    const ink = LABEL_INK[options.tone];
    paint['text-color'] = water
      ? ink.water
      : CITY_LABELS.test(id)
        ? ink.city
        : ROAD_LABELS.test(id)
          ? ink.road
          : ink.secondary;
    paint['text-halo-color'] = options.halo;
    paint['text-halo-width'] = water ? WATER_HALO_WIDTH : HALO_WIDTH;
    paint['text-halo-blur'] = HALO_BLUR;
    if (SUBURB_AND_STATE_LABELS.test(id)) layout['text-letter-spacing'] = WIDE_LETTER_SPACING;
  }
  const result: PaintLayer = {...layer, layout, paint} as PaintLayer;
  if (options.preset === 'streets' && ROAD_LABELS.test(id)) {
    result.minzoom = Math.max(layer.minzoom ?? 0, STREET_LABEL_MIN_ZOOM);
  }
  if (options.suppressNames.length && !ROAD_LABELS.test(id)) {
    result.filter = excludeNames(layer.filter, options.suppressNames);
  }
  return result;
}

/**
 * Multiplies a MapLibre numeric property by `factor`. Constants and legacy `{stops}` functions
 * are rewritten in place, and `interpolate` / `step` expressions have their output values
 * rewritten (a zoom curve must stay the top-level expression); other expressions are wrapped in
 * `['*', value, factor]`. `capPixels` caps outputs from zoom 11 up (city names shrunk below 1x).
 */
function scaleTextSize(size: unknown, factor: number, capPixels?: number): unknown {
  return scaleZoomValue(size, factor, capPixels);
}

type StopFunction = {base?: number; stops: [number, unknown][]};

function isStopFunction(value: unknown): value is StopFunction {
  return (
    typeof value === 'object' &&
    value !== null &&
    !Array.isArray(value) &&
    Array.isArray((value as StopFunction).stops)
  );
}

function scaleOutput(value: unknown, zoom: number | null, factor: number, capPixels?: number) {
  if (typeof value === 'number') {
    const scaled = value * factor;
    return capPixels !== undefined && zoom !== null && zoom >= CITY_LABEL_CAP_ZOOM
      ? Math.min(scaled, capPixels)
      : scaled;
  }
  return Array.isArray(value) ? scaleZoomValue(value, factor, capPixels) : value;
}

/** Scales a numeric paint or layout value (see {@link scaleTextSize}). */
function scaleZoomValue(value: unknown, factor: number, capPixels?: number): unknown {
  if (typeof value === 'number') return value * factor;
  if (Array.isArray(value)) {
    const operator = value[0];
    if (typeof operator === 'string' && operator.startsWith('interpolate')) {
      // ['interpolate', type, input, zoom1, out1, zoom2, out2, ...]
      return value.map((item, index) =>
        index >= 4 && index % 2 === 0
          ? scaleOutput(item, Number(value[index - 1]), factor, capPixels)
          : item
      );
    }
    if (operator === 'step') {
      // ['step', input, out0, zoom1, out1, ...]
      return value.map((item, index) =>
        index >= 2 && index % 2 === 0
          ? scaleOutput(item, index === 2 ? null : Number(value[index - 1]), factor, capPixels)
          : item
      );
    }
    return ['*', value, factor];
  }
  if (isStopFunction(value)) {
    return {
      ...value,
      stops: value.stops.map(([zoom, stop]) => [zoom, scaleOutput(stop, zoom, factor, capPixels)])
    };
  }
  return value;
}

// Name suppression ----------------------------------------------------------------------------

const LEGACY_OPERATORS = new Set(['==', '!=', '>', '>=', '<', '<=', 'in', '!in', 'has', '!has']);
const COMPOUND_OPERATORS = new Set(['all', 'any', 'none']);

/**
 * True when a filter can only be an expression filter (`['==', ['get', 'class'], 'x']`, `case`,
 * `match`, `get` ...). The CARTO styles use the legacy syntax (`['==', 'class', 'x']`), which is
 * the default here.
 */
function isExpressionFilter(filter: unknown): boolean {
  if (!Array.isArray(filter) || typeof filter[0] !== 'string') return false;
  const operator = filter[0];
  if (COMPOUND_OPERATORS.has(operator)) return filter.slice(1).some(isExpressionFilter);
  if (!LEGACY_OPERATORS.has(operator)) return true;
  // A legacy comparison names its property with a string; an expression nests an array there.
  return Array.isArray(filter[1]);
}

/**
 * Combines a layer's filter with "name is not one of `names`" (exact match on `name` or
 * `name_en`, which CARTO uses below zoom 13), in the filter syntax the layer already uses.
 */
function excludeNames(filter: unknown, names: readonly string[]): unknown {
  if (filter && isExpressionFilter(filter)) {
    const nameOf = (property: string) => ['to-string', ['coalesce', ['get', property], '']];
    return [
      'all',
      filter,
      ['!', ['in', nameOf('name'), ['literal', names]]],
      ['!', ['in', nameOf('name_en'), ['literal', names]]]
    ];
  }
  const exclusions = [
    ['!in', 'name', ...names],
    ['!in', 'name_en', ...names]
  ];
  return filter ? ['all', filter, ...exclusions] : ['all', ...exclusions];
}

// ---------------------------------------------------------------------------------------------
// The flat (no tiles) ground
// ---------------------------------------------------------------------------------------------

/**
 * The style of the `none` ground: a `background` layer in the palette colour (`background`, else
 * `land`), Natural Earth land (`world: 'land'`) and coast (`'land'` or `'coast'`) from the
 * `natural-earth` dataset (`ne_110m_land.geojson`, `ne_110m_coastline.geojson`), then the
 * graticule and reference lines. If the dataset files are missing MapLibre logs an error and the
 * ground shows without land.
 */
export function createFlatStyle(
  basemap: ResolvedBasemap,
  tone: CartographyTheme
): StyleSpecification {
  const {palette} = basemap;
  const decorations = getDecorations(basemap, tone);
  const sources: StyleSpecification['sources'] = {};
  const layers: LayerSpecification[] = [
    {
      id: 'ground-background',
      type: 'background',
      paint: {'background-color': palette.background ?? palette.land ?? '#000000'}
    }
  ];
  if (basemap.world) {
    sources['world-land'] = geoJsonSource(getAbsoluteDataUrl(WORLD_LAND_FILE));
    sources['world-coast'] = geoJsonSource(getAbsoluteDataUrl(WORLD_COAST_FILE));
    if (basemap.world === 'land') {
      layers.push({
        id: 'world-land',
        type: 'fill',
        source: 'world-land',
        paint: {'fill-color': palette.land ?? '#808080', 'fill-antialias': true}
      });
    }
    if (palette.boundaryOpacity !== 0) {
      layers.push({
        id: 'world-coast',
        type: 'line',
        source: 'world-coast',
        paint: {
          'line-color': palette.boundary ?? '#808080',
          'line-opacity': palette.boundaryOpacity ?? 1,
          'line-width': 0.6
        }
      });
    }
  }
  return {
    version: 8,
    sources: {...sources, ...decorations.sources},
    layers: [...layers, ...decorations.layers]
  };
}

function geoJsonSource(url: string): GeoJSONSourceSpecification {
  return {type: 'geojson', data: url};
}

/** MapLibre fetches GeoJSON in a worker, where a relative URL has no base: make it absolute. */
function getAbsoluteDataUrl(file: string): string {
  return new URL(getDataFileUrl('natural-earth', file), document.baseURI).href;
}

// ---------------------------------------------------------------------------------------------
// Graticule and reference lines
// ---------------------------------------------------------------------------------------------

/** Spacing in degrees between the vertices of a graticule line, so it curves in any projection. */
const GRATICULE_DENSIFY_DEGREES = 2;
/** The Web Mercator limit of the graticule. */
const GRATICULE_MAX_LATITUDE = 85;
/** Obliquity of the ecliptic: the tropics sit at +-23.44 degrees, the polar circles at +-66.56. */
const TROPIC_LATITUDE = 23.44;
const POLAR_CIRCLE_LATITUDE = 66.56;

type LineFeatureCollection = GeoJSON.FeatureCollection<GeoJSON.LineString, {kind: string}>;

function lineFeature(kind: string, coordinates: [number, number][]) {
  return {
    type: 'Feature' as const,
    properties: {kind},
    geometry: {type: 'LineString' as const, coordinates}
  };
}

function parallel(latitude: number): [number, number][] {
  const points: [number, number][] = [];
  for (let longitude = -180; longitude <= 180; longitude += GRATICULE_DENSIFY_DEGREES) {
    points.push([longitude, latitude]);
  }
  return points;
}

function meridian(longitude: number): [number, number][] {
  const points: [number, number][] = [];
  for (
    let latitude = -GRATICULE_MAX_LATITUDE;
    latitude <= GRATICULE_MAX_LATITUDE;
    latitude += GRATICULE_DENSIFY_DEGREES
  ) {
    points.push([longitude, latitude]);
  }
  points.push([longitude, GRATICULE_MAX_LATITUDE]);
  return points;
}

/**
 * Graticule lines every `stepDegrees` (meridians -180..180 and parallels within +-85 degrees),
 * densified every 2 degrees so they curve correctly in any projection. The equator is left out
 * when `skipEquator` is set (a reference line draws it).
 */
export function buildGraticuleGeoJson(
  stepDegrees: number,
  skipEquator = false
): LineFeatureCollection {
  const step = Math.max(stepDegrees, 1);
  const features = [];
  for (let longitude = -180; longitude <= 180; longitude += step) {
    features.push(lineFeature('meridian', meridian(longitude)));
  }
  for (let latitude = step; latitude < GRATICULE_MAX_LATITUDE; latitude += step) {
    features.push(lineFeature('parallel', parallel(latitude)));
    features.push(lineFeature('parallel', parallel(-latitude)));
  }
  if (!skipEquator) features.push(lineFeature('parallel', parallel(0)));
  return {type: 'FeatureCollection', features};
}

/** The named reference latitudes as lines (`kind`: `equator`, `tropics` or `polar-circles`). */
export function buildReferenceLinesGeoJson(
  lines: readonly ReferenceLatitude[]
): LineFeatureCollection {
  const features = [];
  for (const kind of lines) {
    if (kind === 'equator') features.push(lineFeature(kind, parallel(0)));
    else {
      const latitude = kind === 'tropics' ? TROPIC_LATITUDE : POLAR_CIRCLE_LATITUDE;
      features.push(lineFeature(kind, parallel(latitude)), lineFeature(kind, parallel(-latitude)));
    }
  }
  return {type: 'FeatureCollection', features};
}

/** Reference line strength relative to the graticule: the equator is bolder, the circles dashed. */
const EQUATOR_OPACITY_FACTOR = 2;
const CIRCLE_OPACITY_FACTOR = 4 / 3;
const EQUATOR_WIDTH_FACTOR = 1.6;
const CIRCLE_DASH = [6, 4];

/** Sources and layers of the graticule and the reference latitudes of a basemap. */
function getDecorations(basemap: ResolvedBasemap, tone: CartographyTheme) {
  const sources: StyleSpecification['sources'] = {};
  const layers: LayerSpecification[] = [];
  const {graticule, referenceLines} = basemap;
  if (!graticule && !referenceLines.length) return {sources, layers};
  const color =
    parseCssColor(basemap.palette.graticule) ?? parseCssColor(DEFAULT_GRATICULE_COLORS[tone]);
  if (!color) return {sources, layers};
  const rgb = formatCssColor({...color, a: 1});
  const opacity = (graticule?.opacity ?? 1) * color.a;
  const widthPixels = graticule?.widthPixels ?? 0.5;
  const hasEquator = referenceLines.includes('equator');
  if (graticule) {
    sources['ground-graticule'] = {
      type: 'geojson',
      data: buildGraticuleGeoJson(graticule.stepDegrees, hasEquator)
    };
    layers.push({
      id: 'ground-graticule',
      type: 'line',
      source: 'ground-graticule',
      paint: {'line-color': rgb, 'line-opacity': opacity, 'line-width': widthPixels}
    });
  }
  if (referenceLines.length) {
    sources['ground-reference'] = {
      type: 'geojson',
      data: buildReferenceLinesGeoJson(referenceLines)
    };
    if (hasEquator) {
      layers.push({
        id: 'ground-reference-equator',
        type: 'line',
        source: 'ground-reference',
        filter: ['==', ['get', 'kind'], 'equator'],
        paint: {
          'line-color': rgb,
          'line-opacity': Math.min(opacity * EQUATOR_OPACITY_FACTOR, 1),
          'line-width': widthPixels * EQUATOR_WIDTH_FACTOR
        }
      });
    }
    if (referenceLines.some(kind => kind !== 'equator')) {
      layers.push({
        id: 'ground-reference-circles',
        type: 'line',
        source: 'ground-reference',
        filter: ['!=', ['get', 'kind'], 'equator'],
        paint: {
          'line-color': rgb,
          'line-opacity': Math.min(opacity * CIRCLE_OPACITY_FACTOR, 1),
          'line-width': widthPixels,
          'line-dasharray': CIRCLE_DASH
        }
      });
    }
  }
  return {sources, layers};
}
