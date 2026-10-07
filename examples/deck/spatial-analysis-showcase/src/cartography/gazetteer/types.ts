// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * What a place is. The kind decides how `placeToAnnotation` draws it: water, rivers, oceans and
 * currents become italic `water` labels; districts, neighborhoods and regions become `area` names;
 * peaks, landforms, glaciers and passes become `landform` labels; everything else is a `point`.
 */
export type PlaceKind =
  | 'city'
  | 'district'
  | 'neighborhood'
  | 'park'
  | 'water'
  | 'river'
  | 'ocean'
  | 'current'
  | 'landform'
  | 'peak'
  | 'glacier'
  | 'pass'
  | 'airport'
  | 'port'
  | 'station'
  | 'landmark'
  | 'region'
  | 'site';

/**
 * A named place with one verified coordinate. The source of each coordinate is recorded in a
 * comment next to it in the gazetteer file.
 */
export type Place = {
  /** Kebab-case, unique within its gazetteer. */
  id: string;
  /** Spelled once, as it should appear on the map. */
  name: string;
  kind: PlaceKind;
  /** `[longitude, latitude]` in degrees. Snap it to data with `anchors.ts` where data exists. */
  lngLat: readonly [number, number];
  /** Smallest zoom at which the label makes sense. */
  minZoom?: number;
  /** 0-3 (3 = always worth naming). */
  priority?: number;
  /** Peaks, passes and stations, in metres above sea level. */
  elevationM?: number;
  /** Free text for finding notes, for example `"summit; DEM cell reads 4,476 m"`. */
  note?: string;
  /** Other spellings and codes that `findPlace` matches (IATA codes, local names). */
  aliases?: readonly string[];
  /** Label size for area and water kinds. Defaults to `'medium'`. */
  size?: 'small' | 'medium' | 'large';
};

/** A named set of places for one geography, keyed by `Place.id`. */
export type Gazetteer = {
  id: string;
  name: string;
  places: Readonly<Record<string, Place>>;
};
