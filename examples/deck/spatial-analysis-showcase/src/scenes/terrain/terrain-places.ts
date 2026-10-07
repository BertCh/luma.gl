// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Place labels of the terrain chapter. Names, coordinates and elevations come from the verified
 * `ALPS` gazetteer first and from the `alps-context` dataset (OpenStreetMap, ODbL) for what the
 * gazetteer lacks (Obergabelhorn, Wellenkuppe, Riffelhorn, glacier and lake polygons, stations);
 * no scene types a coordinate.
 *
 * A scene lists `{id: 'alps-context', role: 'glaciers, peaks, places'}` in its `datasets`, loads the
 * context once with {@link loadAlpsContext} and builds annotations with the helpers below. Peak
 * labels are `landform` annotations with the triangle marker and the published elevation
 * (Matterhorn 4,478 m, Dufourspitze 4,634 m); settlements and stations are `point` annotations
 * with the elevation as detail (Gornergrat station 3,089 m, Zermatt 1,608 m).
 */

import {findPlace, placeToAnnotation, ALPS} from '../../cartography/gazetteer';
import {getPolygonLabelPoint, haversineMeters} from '../../cartography/anchors';
import type {LngLatBounds, PolygonRings} from '../../cartography/picking';
import type {MapAnnotation} from '../../cartography/types';
import type {LoadedDataset} from '../../data/catalog';
import {fetchGeoJson, type GeoJsonCollection} from '../../data/loaders';
import type {TerrainDem} from './cpu-dem';

/** A named summit of the context dataset. */
export type ContextPeak = {
  /** OSM name, or null for an unnamed summit. */
  name: string | null;
  lngLat: readonly [number, number];
  /** Published elevation in metres, or null. */
  elevationMeters: number | null;
  /** OSM-derived prominence in metres, or null. */
  prominenceMeters: number | null;
  osmId: number;
};

/** A saddle of the context dataset. */
export type ContextSaddle = {
  name: string | null;
  lngLat: readonly [number, number];
  elevationMeters: number | null;
  osmId: number;
};

/** What a context place is. */
export type ContextPlaceKind = 'settlement' | 'rail-station' | 'lift-station' | 'hut';

/** A settlement, station or hut of the context dataset. */
export type ContextPlace = {
  name: string | null;
  kind: ContextPlaceKind;
  lngLat: readonly [number, number];
  elevationMeters: number | null;
  /** OSM `place` value of settlements (`town`, `village`, `hamlet`, ...). */
  placeType?: string;
  osmId: number;
};

/** A named polygon (glacier or lake) of the context dataset, as the parts of its geometry. */
export type ContextPolygon = {
  name: string | null;
  /** The polygons of the feature (outer ring first, then holes), `[longitude, latitude]`. */
  polygons: PolygonRings[];
  /** Glacier area in km², when given. */
  areaKm2?: number;
  /** Lake area in hectares, when given. */
  areaHa?: number;
  osmId: number;
};

/** A line of the context dataset (the Gornergrat railway). */
export type ContextLine = {
  name: string | null;
  coordinates: readonly (readonly [number, number])[];
  osmId: number;
};

/** The `alps-context` dataset, typed. */
export type AlpsContext = {
  peaks: ContextPeak[];
  saddles: ContextSaddle[];
  places: ContextPlace[];
  glaciers: ContextPolygon[];
  lakes: ContextPolygon[];
  railway: ContextLine[];
  /** The glacier GeoJSON as fetched, for `rasterizeGlacierMask` (one fetch serves both). */
  glacierGeoJson: GeoJsonCollection;
};

type Properties = Record<string, unknown>;

const asNumber = (value: unknown): number | null =>
  typeof value === 'number' && Number.isFinite(value) ? value : null;
const asName = (value: unknown): string | null =>
  typeof value === 'string' && value.trim() ? value : null;

function getPoints(collection: GeoJsonCollection) {
  const points: {properties: Properties; lngLat: readonly [number, number]}[] = [];
  for (const feature of collection.features) {
    if (feature.geometry?.type !== 'Point') continue;
    const [longitude, latitude] = feature.geometry.coordinates as [number, number];
    points.push({properties: feature.properties ?? {}, lngLat: [longitude, latitude]});
  }
  return points;
}

function getPolygons(collection: GeoJsonCollection): ContextPolygon[] {
  return collection.features.map(feature => {
    const properties = feature.properties ?? {};
    const geometry = feature.geometry;
    const polygons: PolygonRings[] =
      geometry?.type === 'Polygon'
        ? [geometry.coordinates as PolygonRings]
        : geometry?.type === 'MultiPolygon'
          ? (geometry.coordinates as PolygonRings[])
          : [];
    const polygon: ContextPolygon = {
      name: asName(properties.name),
      polygons,
      osmId: asNumber(properties.osmId) ?? 0
    };
    const areaKm2 = asNumber(properties.areaKm2);
    const areaHa = asNumber(properties.areaHa);
    if (areaKm2 !== null) polygon.areaKm2 = areaKm2;
    if (areaHa !== null) polygon.areaHa = areaHa;
    return polygon;
  });
}

/**
 * Fetches and types the six side files of the `alps-context` dataset (peaks, saddles, places,
 * glaciers, lakes, railway; about 480 KB together).
 */
export async function loadAlpsContext(
  dataset: LoadedDataset,
  signal?: AbortSignal
): Promise<AlpsContext> {
  const [peaks, saddles, places, glaciers, lakes, railway] = await Promise.all(
    ['peaks', 'saddles', 'places', 'glaciers', 'lakes', 'railway'].map(name =>
      fetchGeoJson(dataset.fileUrl(`${name}.geojson`), signal)
    )
  );
  return {
    peaks: getPoints(peaks).map(({properties, lngLat}) => ({
      name: asName(properties.name),
      lngLat,
      elevationMeters: asNumber(properties.elevationM),
      prominenceMeters: asNumber(properties.prominenceM),
      osmId: asNumber(properties.osmId) ?? 0
    })),
    saddles: getPoints(saddles).map(({properties, lngLat}) => ({
      name: asName(properties.name),
      lngLat,
      elevationMeters: asNumber(properties.elevationM),
      osmId: asNumber(properties.osmId) ?? 0
    })),
    places: getPoints(places).map(({properties, lngLat}) => ({
      name: asName(properties.name),
      kind: properties.kind as ContextPlaceKind,
      lngLat,
      elevationMeters: asNumber(properties.elevationM),
      ...(typeof properties.place === 'string' ? {placeType: properties.place} : {}),
      osmId: asNumber(properties.osmId) ?? 0
    })),
    glaciers: getPolygons(glaciers),
    lakes: getPolygons(lakes),
    railway: railway.features
      .filter(feature => feature.geometry?.type === 'LineString')
      .map(feature => ({
        name: asName(feature.properties?.name),
        coordinates: feature.geometry?.coordinates as [number, number][],
        osmId: asNumber(feature.properties?.osmId) ?? 0
      })),
    glacierGeoJson: glaciers
  };
}

/** `4,478 m`: an elevation as the labels spell it. */
export function formatElevationMeters(meters: number): string {
  return `${Math.round(meters).toLocaleString('en-US')} m`;
}

// ---------------------------------------------------------------------------------------------
// Gazetteer labels
// ---------------------------------------------------------------------------------------------

/**
 * The annotation of an `ALPS` gazetteer place. Peaks become `landform` labels with the triangle
 * marker and `elevationMeters`; stations and towns become `point` labels whose `detail` is the
 * elevation ("Gornergrat station" 3,089 m, "Zermatt" 1,608 m); glaciers and lakes become italic
 * `landform` and `water` labels. `overrides` win.
 *
 * @throws In development, when the id is not in the `ALPS` gazetteer.
 */
export function terrainLabel(id: string, overrides: Partial<MapAnnotation> = {}): MapAnnotation {
  const place = ALPS.places[id];
  if (!place) throw new Error(`Unknown ALPS place "${id}"`);
  const annotation = placeToAnnotation(place);
  if (annotation.kind === 'point' && place.elevationM !== undefined) {
    return {
      ...annotation,
      detail: formatElevationMeters(place.elevationM),
      ...overrides
    } as MapAnnotation;
  }
  return {...annotation, ...overrides} as MapAnnotation;
}

/** {@link terrainLabel} for several ids, in order, with optional overrides per id. */
export function terrainLabels(
  ids: readonly string[],
  overrides: Readonly<Record<string, Partial<MapAnnotation>>> = {}
): MapAnnotation[] {
  return ids.map(id => terrainLabel(id, overrides[id]));
}

// ---------------------------------------------------------------------------------------------
// Peaks
// ---------------------------------------------------------------------------------------------

/** Lower case, no diacritics, only letters and digits (so "Ober Gabelhorn" matches "Obergabelhorn"). */
function normalizeName(name: string): string {
  return name
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '');
}

function isInside(lngLat: readonly [number, number], window: LngLatBounds | undefined): boolean {
  if (!window) return true;
  return (
    lngLat[0] >= window[0] &&
    lngLat[0] <= window[2] &&
    lngLat[1] >= window[1] &&
    lngLat[1] <= window[3]
  );
}

/** Selection of {@link peakLabels}. */
export type PeakLabelOptions = {
  /**
   * Peaks to label, by the name as it should be spelled ("Obergabelhorn") or an `ALPS` place id.
   * A name found in the gazetteer uses its verified coordinate and elevation; otherwise the
   * context peak whose name matches (ignoring spaces and diacritics, or starting with it) is used.
   * Names that match nothing are skipped. Without `names`, the highest-ranked named summits.
   */
  names?: readonly string[];
  /** Cap on the returned labels. Defaults to all `names`, or 6 for a ranked selection. */
  max?: number;
  /** Drop peaks below this elevation (metres). */
  minElevation?: number;
  /** Keep peaks inside `[west, south, east, north]` only. */
  window?: LngLatBounds;
  /** Ranking without `names`: `'prominence'` (default) or `'elevation'`. */
  rankBy?: 'prominence' | 'elevation';
  /** Zoom from which the labels show. */
  minZoom?: number;
  /** Ink of the labels. */
  tone?: Extract<MapAnnotation, {kind: 'landform'}>['tone'];
};

function peakAnnotation(
  text: string,
  lngLat: readonly [number, number],
  elevation: number | null,
  id: string,
  priority: number,
  options: PeakLabelOptions
): MapAnnotation {
  return {
    kind: 'landform',
    id,
    coordinate: lngLat,
    text,
    marker: 'peak',
    ...(elevation !== null ? {elevationMeters: elevation} : {}),
    priority,
    ...(options.minZoom !== undefined ? {minZoom: options.minZoom} : {}),
    ...(options.tone ? {tone: options.tone} : {})
  };
}

/**
 * Peak labels (`landform`, `marker: 'peak'`, `elevationMeters`) from the gazetteer and the context.
 * Priority follows the order of `names`, or the rank of an automatic selection. Snap them to the
 * DEM summit cells with {@link snapPeaksToDem} once the DEM is loaded.
 *
 * @example
 * peakLabels(context, {names: ['Matterhorn', 'Dufourspitze', 'Obergabelhorn'], max: 6});
 * peakLabels(context, {max: 4, minElevation: 4000, window: demBounds}); // by prominence
 */
export function peakLabels(context: AlpsContext, options: PeakLabelOptions = {}): MapAnnotation[] {
  const labels: MapAnnotation[] = [];
  if (options.names) {
    const normalized = context.peaks
      .filter(peak => peak.name)
      .map(peak => ({peak, key: normalizeName(peak.name as string)}));
    options.names.forEach((requested, index) => {
      const priority = options.names!.length - index;
      const place = ALPS.places[requested] ?? findPlace(ALPS, requested);
      const text = place && ALPS.places[requested] ? place.name : requested;
      let lngLat: readonly [number, number] | null = null;
      let elevation: number | null = null;
      let id = `peak:${normalizeName(text)}`;
      if (place && (place.kind === 'peak' || place.kind === 'pass')) {
        lngLat = place.lngLat;
        elevation = place.elevationM ?? null;
        id = `place:${place.id}`;
      } else {
        const key = normalizeName(requested);
        const matches = normalized.filter(
          ({key: candidate, peak}) =>
            candidate === key ||
            candidate.startsWith(key) ||
            normalizeName((peak.name as string).split('/')[0]) === key
        );
        matches.sort(
          (a, b) =>
            (a.key === key ? 0 : 1) - (b.key === key ? 0 : 1) ||
            (b.peak.elevationMeters ?? 0) - (a.peak.elevationMeters ?? 0)
        );
        const best = matches[0]?.peak;
        if (best) {
          lngLat = best.lngLat;
          elevation = best.elevationMeters;
        }
      }
      if (!lngLat || !isInside(lngLat, options.window)) return;
      if (options.minElevation !== undefined && (elevation ?? 0) < options.minElevation) return;
      labels.push(peakAnnotation(text, lngLat, elevation, id, priority, options));
    });
    return labels.slice(0, options.max ?? labels.length);
  }
  const rankBy = options.rankBy ?? 'prominence';
  const ranked = context.peaks
    .filter(
      peak =>
        peak.name &&
        isInside(peak.lngLat, options.window) &&
        (options.minElevation === undefined || (peak.elevationMeters ?? 0) >= options.minElevation)
    )
    .sort((a, b) =>
      rankBy === 'elevation'
        ? (b.elevationMeters ?? 0) - (a.elevationMeters ?? 0)
        : (b.prominenceMeters ?? -1) - (a.prominenceMeters ?? -1) ||
          (b.elevationMeters ?? 0) - (a.elevationMeters ?? 0)
    )
    .slice(0, options.max ?? 6);
  ranked.forEach((peak, index) => {
    labels.push(
      peakAnnotation(
        peak.name as string,
        peak.lngLat,
        peak.elevationMeters,
        `peak:${peak.osmId}`,
        ranked.length - index,
        options
      )
    );
  });
  return labels;
}

/** The highest DEM cell near a position, see {@link snapLngLatToHighestCell}. */
export type SnappedSummit = {
  /** Centre of the cell, `[longitude, latitude]`. */
  lngLat: [number, number];
  column: number;
  row: number;
  /** DEM elevation of the cell, metres. */
  elevation: number;
  /** Ground distance moved, metres. */
  moveMeters: number;
};

/**
 * Moves a position onto the highest finite DEM cell within `radiusMeters` (cell centres compared,
 * ties to the nearest). Like `snapToHighestCell` of `cartography/anchors.ts`, but it maps cells
 * with the DEM's own Web Mercator pixel function: that helper assumes rows linear in latitude,
 * which places the summit cell up to about one wide-DEM cell (17 m) off within this window.
 * Returns `null` when the circle misses the DEM.
 */
export function snapLngLatToHighestCell(
  dem: TerrainDem,
  lngLat: readonly [number, number],
  radiusMeters = 160
): SnappedSummit | null {
  const [x, y] = dem.getPixelCoordinates(lngLat[0], lngLat[1]);
  const cell = dem.getGroundCellSize(Math.min(dem.height - 1, Math.max(0, Math.floor(y))));
  const reach = Math.ceil(radiusMeters / cell) + 1;
  const centerColumn = Math.floor(x);
  const centerRow = Math.floor(y);
  let best: SnappedSummit | null = null;
  for (
    let row = Math.max(0, centerRow - reach);
    row <= Math.min(dem.height - 1, centerRow + reach);
    row++
  ) {
    for (
      let column = Math.max(0, centerColumn - reach);
      column <= Math.min(dem.width - 1, centerColumn + reach);
      column++
    ) {
      const elevation = dem.values[row * dem.width + column];
      if (!Number.isFinite(elevation)) continue;
      const center = dem.getLongitudeLatitude(column, row);
      const distance = haversineMeters([lngLat[0], lngLat[1]], center);
      if (distance > radiusMeters) continue;
      if (
        !best ||
        elevation > best.elevation ||
        (elevation === best.elevation && distance < best.moveMeters)
      ) {
        best = {lngLat: center, column, row, elevation, moveMeters: distance};
      }
    }
  }
  return best;
}

/**
 * Snaps every peak annotation (`landform` with `marker: 'peak'`) to the highest DEM cell within
 * `radiusMeters` (default 160 m), so the triangle sits on the summit cell of this DEM. Other
 * annotations and peaks that miss the DEM pass through unchanged. The published
 * `elevationMeters` is kept: say once that the DEM cell reads differently (Matterhorn 4,478 m
 * published, 4,476 m in the 6.6 m DEM).
 */
export function snapPeaksToDem(
  annotations: readonly MapAnnotation[],
  dem: TerrainDem,
  radiusMeters = 160
): MapAnnotation[] {
  return annotations.map(annotation => {
    if (annotation.kind !== 'landform' || annotation.marker !== 'peak') return annotation;
    const snapped = snapLngLatToHighestCell(dem, annotation.coordinate, radiusMeters);
    return snapped ? {...annotation, coordinate: snapped.lngLat} : annotation;
  });
}

// ---------------------------------------------------------------------------------------------
// Glaciers and lakes
// ---------------------------------------------------------------------------------------------

type Point = readonly [number, number];

/** Sutherland-Hodgman clip of one ring to a rectangle. */
function clipRing(ring: readonly Point[], window: LngLatBounds): Point[] {
  const [west, south, east, north] = window;
  const edges: {inside: (p: Point) => boolean; cut: (a: Point, b: Point) => Point}[] = [
    {
      inside: p => p[0] >= west,
      cut: (a, b) => [west, a[1] + ((b[1] - a[1]) * (west - a[0])) / (b[0] - a[0])]
    },
    {
      inside: p => p[0] <= east,
      cut: (a, b) => [east, a[1] + ((b[1] - a[1]) * (east - a[0])) / (b[0] - a[0])]
    },
    {
      inside: p => p[1] >= south,
      cut: (a, b) => [a[0] + ((b[0] - a[0]) * (south - a[1])) / (b[1] - a[1]), south]
    },
    {
      inside: p => p[1] <= north,
      cut: (a, b) => [a[0] + ((b[0] - a[0]) * (north - a[1])) / (b[1] - a[1]), north]
    }
  ];
  let output: Point[] = ring.map(p => [p[0], p[1]] as Point);
  for (const edge of edges) {
    const input = output;
    output = [];
    for (let i = 0; i < input.length; i++) {
      const current = input[i];
      const previous = input[(i + input.length - 1) % input.length];
      if (edge.inside(current)) {
        if (!edge.inside(previous)) output.push(edge.cut(previous, current));
        output.push(current);
      } else if (edge.inside(previous)) {
        output.push(edge.cut(previous, current));
      }
    }
    if (!output.length) break;
  }
  return output;
}

/** Shoelace area of a ring in degrees squared, scaled to a common metric by cos(latitude). */
function getRingArea(ring: readonly Point[]): number {
  let sum = 0;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    sum += (ring[j][0] + ring[i][0]) * (ring[j][1] - ring[i][1]);
  }
  return Math.abs(sum) / 2;
}

/**
 * The label point of a polygon feature inside a window: each polygon part is clipped to the
 * window (outer ring and holes) and the part with the largest clipped area gives
 * `getPolygonLabelPoint` of its clipped rings. Returns null when nothing remains in the window.
 */
export function getClippedLabelPoint(
  polygon: ContextPolygon,
  window: LngLatBounds
): {lngLat: [number, number]; clippedArea: number} | null {
  let best: {lngLat: [number, number]; clippedArea: number} | null = null;
  for (const rings of polygon.polygons) {
    const clipped = rings
      .map(ring => clipRing(ring as readonly Point[], window))
      .filter(ring => ring.length >= 3);
    if (!clipped.length || (clipped[0] !== undefined && clipped[0].length < 3)) continue;
    const outerArea = getRingArea(clipped[0]);
    const holeArea = clipped.slice(1).reduce((sum, ring) => sum + getRingArea(ring), 0);
    const area = outerArea - holeArea;
    if (area <= 0) continue;
    const lngLat = getPolygonLabelPoint(clipped as unknown as (readonly Point[])[]);
    if (lngLat && (!best || area > best.clippedArea)) best = {lngLat, clippedArea: area};
  }
  return best;
}

/** Selection of {@link glacierLabels} and {@link lakeLabels}. */
export type PolygonLabelOptions = {
  /** `[west, south, east, north]` of the story's DEM window (the polygons are clipped to it). */
  window: LngLatBounds;
  /** Names to label, as the OSM name or the spelling to show (matched ignoring case and spaces). */
  names?: readonly string[];
  /** Cap on the returned labels. Defaults to 4. */
  max?: number;
  /** Smallest clipped area (relative to the window area, 0-1) to label. Defaults to 0.002. */
  minWindowShare?: number;
  minZoom?: number;
  size?: 'small' | 'medium' | 'large';
};

function polygonLabels(
  polygons: readonly ContextPolygon[],
  options: PolygonLabelOptions,
  idPrefix: string
): MapAnnotation[] {
  const [west, south, east, north] = options.window;
  const windowArea = (east - west) * (north - south);
  const wanted = options.names?.map(normalizeName);
  const byName = new Map<
    string,
    {polygon: ContextPolygon; lngLat: [number, number]; share: number}
  >();
  for (const polygon of polygons) {
    if (!polygon.name) continue;
    const key = normalizeName(polygon.name);
    if (wanted && !wanted.includes(key)) continue;
    const label = getClippedLabelPoint(polygon, options.window);
    if (!label) continue;
    const share = label.clippedArea / windowArea;
    if (share < (options.minWindowShare ?? 0.002) && !wanted) continue;
    // One label per name (Feegletscher is two features): the larger clipped part wins.
    const existing = byName.get(key);
    if (!existing || share > existing.share)
      byName.set(key, {polygon, lngLat: label.lngLat, share});
  }
  const chosen = [...byName.values()].sort((a, b) => b.share - a.share).slice(0, options.max ?? 4);
  return chosen.map(({polygon, lngLat}, index) => ({
    kind: 'water',
    id: `${idPrefix}:${polygon.osmId}`,
    coordinate: lngLat,
    text: polygon.name as string,
    priority: chosen.length - index,
    ...(options.minZoom !== undefined ? {minZoom: options.minZoom} : {}),
    size: options.size ?? 'medium'
  })) as MapAnnotation[];
}

/**
 * Glacier names (`water` labels, serif italic in water blue, as the chapter sheet asks) at the
 * label point of each glacier polygon clipped to the story's window, largest first, one label per
 * name.
 */
export function glacierLabels(context: AlpsContext, options: PolygonLabelOptions): MapAnnotation[] {
  return polygonLabels(context.glaciers, options, 'glacier');
}

/** Lake names (`water` labels) at the label point of each lake polygon clipped to the window. */
export function lakeLabels(context: AlpsContext, options: PolygonLabelOptions): MapAnnotation[] {
  return polygonLabels(context.lakes, options, 'lake');
}

// ---------------------------------------------------------------------------------------------
// Settlements and stations
// ---------------------------------------------------------------------------------------------

/** Selection of {@link placeLabels}. */
export type PlaceLabelOptions = {
  /** Which context kinds to label. Defaults to settlements. */
  kinds?: readonly ContextPlaceKind[];
  /** Names to label (matched ignoring case, diacritics and spaces); others are dropped. */
  names?: readonly string[];
  /** Keep places inside `[west, south, east, north]` only. */
  window?: LngLatBounds;
  /** Settlements: smallest OSM `place` rank kept (`'village'` keeps towns and villages). */
  minimumPlaceType?: 'town' | 'village' | 'hamlet';
  /** Cap on the returned labels. Defaults to 6. */
  max?: number;
  minZoom?: number;
  /** `'subject'` (12.5 px) or `'context'` (default, 11 px muted). */
  rank?: 'subject' | 'context';
};

const PLACE_TYPE_RANK: Record<string, number> = {
  city: 4,
  town: 3,
  village: 2,
  hamlet: 1,
  isolated_dwelling: 0
};

/**
 * Settlement, station and hut labels (`point`, elevation as `detail`) from the context. Without
 * `names`, the largest settlements first (town, village, hamlet), then by elevation.
 */
export function placeLabels(
  context: AlpsContext,
  options: PlaceLabelOptions = {}
): MapAnnotation[] {
  const kinds = options.kinds ?? ['settlement'];
  const wanted = options.names?.map(normalizeName);
  const minimumRank = options.minimumPlaceType ? PLACE_TYPE_RANK[options.minimumPlaceType] : -1;
  const candidates = context.places.filter(place => {
    if (!place.name || !kinds.includes(place.kind) || !isInside(place.lngLat, options.window)) {
      return false;
    }
    if (wanted && !wanted.includes(normalizeName(place.name))) return false;
    return (
      place.kind !== 'settlement' || (PLACE_TYPE_RANK[place.placeType ?? ''] ?? 0) >= minimumRank
    );
  });
  candidates.sort(
    (a, b) =>
      (PLACE_TYPE_RANK[b.placeType ?? ''] ?? 0) - (PLACE_TYPE_RANK[a.placeType ?? ''] ?? 0) ||
      (b.elevationMeters ?? 0) - (a.elevationMeters ?? 0)
  );
  const unique = new Map<string, ContextPlace>();
  for (const place of candidates) {
    const key = `${place.kind}:${normalizeName(place.name as string)}`;
    if (!unique.has(key)) unique.set(key, place);
  }
  return [...unique.values()].slice(0, options.max ?? 6).map(
    (place, index, list) =>
      ({
        kind: 'point',
        id: `place:${place.osmId}`,
        coordinate: place.lngLat,
        text: place.name as string,
        ...(place.elevationMeters !== null
          ? {detail: formatElevationMeters(place.elevationMeters)}
          : {}),
        marker: 'dot',
        rank: options.rank ?? 'context',
        priority: list.length - index,
        ...(options.minZoom !== undefined ? {minZoom: options.minZoom} : {})
      }) as MapAnnotation
  );
}

/** A context place by name and kind, for stations and huts used as observers. */
export function findContextPlace(
  context: AlpsContext,
  name: string,
  kinds: readonly ContextPlaceKind[] = ['rail-station', 'lift-station', 'hut', 'settlement']
): ContextPlace | null {
  const key = normalizeName(name);
  return (
    context.places.find(
      place => place.name && kinds.includes(place.kind) && normalizeName(place.name) === key
    ) ?? null
  );
}
