// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Gazetteer} from './types';

/**
 * The Randstad and its neighbours: cities, central stations, lakes, rivers and the coast.
 * Every coordinate is `[longitude, latitude]` and carries its source in the comment above it.
 */
export const RANDSTAD: Gazetteer = {
  id: 'randstad',
  name: 'Randstad',
  places: {
    // wiki
    amsterdam: {
      id: 'amsterdam',
      name: 'Amsterdam',
      kind: 'city',
      lngLat: [4.89361, 52.37278],
      minZoom: 8,
      priority: 3
    },
    // wiki
    rotterdam: {
      id: 'rotterdam',
      name: 'Rotterdam',
      kind: 'city',
      lngLat: [4.48, 51.92],
      minZoom: 8,
      priority: 3
    },
    // wiki
    'den-haag': {
      id: 'den-haag',
      name: 'The Hague',
      kind: 'city',
      lngLat: [4.31, 52.08],
      minZoom: 8,
      priority: 3,
      aliases: ['Den Haag', "'s-Gravenhage"]
    },
    // wiki
    utrecht: {
      id: 'utrecht',
      name: 'Utrecht',
      kind: 'city',
      lngLat: [5.12167, 52.09083],
      minZoom: 8,
      priority: 3
    },
    // wiki
    leiden: {id: 'leiden', name: 'Leiden', kind: 'city', lngLat: [4.49, 52.16], minZoom: 9},
    // wiki
    delft: {id: 'delft', name: 'Delft', kind: 'city', lngLat: [4.35917, 52.01167], minZoom: 10},
    // wiki
    haarlem: {
      id: 'haarlem',
      name: 'Haarlem',
      kind: 'city',
      lngLat: [4.63556, 52.38139],
      minZoom: 9
    },
    // wiki
    almere: {id: 'almere', name: 'Almere', kind: 'city', lngLat: [5.21667, 52.36667], minZoom: 9},
    // wiki
    amersfoort: {
      id: 'amersfoort',
      name: 'Amersfoort',
      kind: 'city',
      lngLat: [5.38333, 52.15],
      minZoom: 9
    },
    // wiki
    gouda: {id: 'gouda', name: 'Gouda', kind: 'city', lngLat: [4.71111, 52.01111], minZoom: 10},
    // wiki
    dordrecht: {
      id: 'dordrecht',
      name: 'Dordrecht',
      kind: 'city',
      lngLat: [4.67833, 51.79583],
      minZoom: 9.5
    },
    // wiki
    groningen: {
      id: 'groningen',
      name: 'Groningen',
      kind: 'city',
      lngLat: [6.5675, 53.21889],
      minZoom: 7,
      priority: 2
    },
    // wiki
    schiphol: {
      id: 'schiphol',
      name: 'Schiphol',
      kind: 'airport',
      lngLat: [4.765, 52.3],
      minZoom: 9,
      priority: 2,
      aliases: ['AMS']
    },
    // wiki
    'amsterdam-centraal': {
      id: 'amsterdam-centraal',
      name: 'Amsterdam Centraal',
      kind: 'station',
      lngLat: [4.9, 52.37833],
      minZoom: 11.5
    },
    // wiki
    'rotterdam-centraal': {
      id: 'rotterdam-centraal',
      name: 'Rotterdam Centraal',
      kind: 'station',
      lngLat: [4.46944, 51.92444],
      minZoom: 11.5
    },
    // wiki
    'den-haag-centraal': {
      id: 'den-haag-centraal',
      name: 'Den Haag Centraal',
      kind: 'station',
      lngLat: [4.32917, 52.08167],
      minZoom: 11.5
    },
    // wiki
    'utrecht-centraal': {
      id: 'utrecht-centraal',
      name: 'Utrecht Centraal',
      kind: 'station',
      lngLat: [5.10972, 52.08917],
      minZoom: 11.5
    },
    // wiki
    ijsselmeer: {
      id: 'ijsselmeer',
      name: 'IJsselmeer',
      kind: 'water',
      lngLat: [5.25, 52.81667],
      minZoom: 8
    },
    // wiki
    markermeer: {
      id: 'markermeer',
      name: 'Markermeer',
      kind: 'water',
      lngLat: [5.21667, 52.56667],
      minZoom: 8.5
    },
    // wiki
    ijmuiden: {
      id: 'ijmuiden',
      name: 'IJmuiden',
      kind: 'port',
      lngLat: [4.61944, 52.45861],
      minZoom: 10
    },
    // wiki
    'hoek-van-holland': {
      id: 'hoek-van-holland',
      name: 'Hook of Holland',
      kind: 'port',
      lngLat: [4.12861, 51.98111],
      minZoom: 10,
      aliases: ['Hoek van Holland']
    },
    // osm way 206380608 vertex
    'nieuwe-waterweg': {
      id: 'nieuwe-waterweg',
      name: 'Nieuwe Waterweg',
      kind: 'river',
      lngLat: [4.16632, 51.95319],
      minZoom: 10
    },
    // osm way 195743674 vertex
    lek: {id: 'lek', name: 'Lek', kind: 'river', lngLat: [4.95376, 51.96225], minZoom: 10},
    // osm way 41948489 vertex
    'nieuwe-maas': {
      id: 'nieuwe-maas',
      name: 'Nieuwe Maas',
      kind: 'river',
      lngLat: [4.45127, 51.89981],
      minZoom: 10.5
    },
    // label placement inside Natural Earth marine polygon North Sea, off Noordwijk
    'north-sea': {
      id: 'north-sea',
      name: 'North Sea',
      kind: 'ocean',
      lngLat: [3.9, 52.6],
      minZoom: 6
    }
  }
};
