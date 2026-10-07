// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Gazetteer} from './types';

/**
 * Grand Canyon National Park, central corridor: rims, trails, temples and creeks.
 * Every coordinate is `[longitude, latitude]` and carries its source in the comment above it.
 */
export const GRAND_CANYON: Gazetteer = {
  id: 'grand-canyon',
  name: 'Grand Canyon',
  places: {
    // wiki
    'grand-canyon-village': {
      id: 'grand-canyon-village',
      name: 'Grand Canyon Village',
      kind: 'site',
      lngLat: [-112.15667, 36.04917],
      minZoom: 10,
      priority: 2
    },
    // wiki
    'bright-angel-point': {
      id: 'bright-angel-point',
      name: 'Bright Angel Point',
      kind: 'landform',
      lngLat: [-112.05191, 36.19703],
      minZoom: 11,
      priority: 2
    },
    // wiki
    'phantom-ranch': {
      id: 'phantom-ranch',
      name: 'Phantom Ranch',
      kind: 'site',
      lngLat: [-112.09469, 36.10605],
      minZoom: 11,
      priority: 3
    },
    // osm relation 15976432
    'indian-garden': {
      id: 'indian-garden',
      name: 'Indian Garden',
      kind: 'site',
      lngLat: [-112.1293, 36.07695],
      minZoom: 11.5,
      aliases: ['Havasupai Gardens']
    },
    // wiki
    'plateau-point': {
      id: 'plateau-point',
      name: 'Plateau Point',
      kind: 'landform',
      lngLat: [-112.11609, 36.09326],
      minZoom: 12
    },
    // osm node 359253711
    'yavapai-point': {
      id: 'yavapai-point',
      name: 'Yavapai Point',
      kind: 'landform',
      lngLat: [-112.11822, 36.06665],
      minZoom: 12
    },
    // wiki
    'mather-point': {
      id: 'mather-point',
      name: 'Mather Point',
      kind: 'landform',
      lngLat: [-112.10795, 36.06165],
      minZoom: 12
    },
    // osm node 359253704
    'yaki-point': {
      id: 'yaki-point',
      name: 'Yaki Point',
      kind: 'landform',
      lngLat: [-112.08378, 36.05859],
      minZoom: 12
    },
    // wiki
    'zoroaster-temple': {
      id: 'zoroaster-temple',
      name: 'Zoroaster Temple',
      kind: 'peak',
      lngLat: [-112.04524, 36.11881],
      minZoom: 11,
      elevationM: 2171
    },
    // wiki
    'brahma-temple': {
      id: 'brahma-temple',
      name: 'Brahma Temple',
      kind: 'peak',
      lngLat: [-112.0387, 36.13046],
      minZoom: 11,
      elevationM: 2302
    },
    // wiki
    'vishnu-temple': {
      id: 'vishnu-temple',
      name: 'Vishnu Temple',
      kind: 'peak',
      lngLat: [-111.93571, 36.08876],
      minZoom: 11,
      elevationM: 2296
    },
    // wiki
    'wotans-throne': {
      id: 'wotans-throne',
      name: 'Wotans Throne',
      kind: 'peak',
      lngLat: [-111.96028, 36.10483],
      minZoom: 11.5,
      elevationM: 2353
    },
    // usgs nwis 09402500
    'colorado-river': {
      id: 'colorado-river',
      name: 'Colorado River',
      kind: 'river',
      lngLat: [-112.08628, 36.10137],
      minZoom: 10
    },
    // wiki
    'bright-angel-creek': {
      id: 'bright-angel-creek',
      name: 'Bright Angel Creek',
      kind: 'river',
      lngLat: [-112.09578, 36.1017],
      minZoom: 11.5
    },
    // osm way 229834036 vertex
    'garden-creek': {
      id: 'garden-creek',
      name: 'Garden Creek',
      kind: 'river',
      lngLat: [-112.136, 36.06757],
      minZoom: 12
    },
    // wiki
    'grand-canyon-lodge': {
      id: 'grand-canyon-lodge',
      name: 'Grand Canyon Lodge',
      kind: 'site',
      lngLat: [-112.05333, 36.19722],
      minZoom: 11,
      aliases: ['North Rim']
    },
    // wiki
    'desert-view': {
      id: 'desert-view',
      name: 'Desert View',
      kind: 'landmark',
      lngLat: [-111.82583, 36.04389],
      minZoom: 11
    },
    // wiki
    'hermits-rest': {
      id: 'hermits-rest',
      name: 'Hermits Rest',
      kind: 'landform',
      lngLat: [-112.21111, 36.06222],
      minZoom: 11
    }
  }
};
