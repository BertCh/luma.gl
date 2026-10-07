// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Gazetteer} from './types';

/**
 * Dixie Fire 2021 area, Plumas, Butte and Lassen counties, California.
 * Every coordinate is `[longitude, latitude]` and carries its source in the comment above it.
 */
export const DIXIE: Gazetteer = {
  id: 'dixie',
  name: 'Dixie Fire',
  places: {
    // wiki
    greenville: {
      id: 'greenville',
      name: 'Greenville',
      kind: 'city',
      lngLat: [-120.95111, 40.13972],
      minZoom: 9,
      priority: 3
    },
    // wiki
    taylorsville: {
      id: 'taylorsville',
      name: 'Taylorsville',
      kind: 'city',
      lngLat: [-120.83806, 40.07361],
      minZoom: 10
    },
    // wiki
    chester: {
      id: 'chester',
      name: 'Chester',
      kind: 'city',
      lngLat: [-121.23472, 40.3025],
      minZoom: 9,
      priority: 2
    },
    // wiki
    'canyon-dam': {
      id: 'canyon-dam',
      name: 'Canyon Dam',
      kind: 'city',
      lngLat: [-121.07556, 40.17],
      minZoom: 10.5
    },
    // wiki
    'lake-almanor': {
      id: 'lake-almanor',
      name: 'Lake Almanor',
      kind: 'water',
      lngLat: [-121.16028, 40.25278],
      minZoom: 8.5,
      priority: 2
    },
    // osm relation 1356542
    'butt-valley': {
      id: 'butt-valley',
      name: 'Butt Valley Reservoir',
      kind: 'water',
      lngLat: [-121.17605, 40.14527],
      minZoom: 10.5
    },
    // osm node 369170071
    'indian-valley': {
      id: 'indian-valley',
      name: 'Indian Valley',
      kind: 'landform',
      lngLat: [-120.93107, 40.13295],
      minZoom: 10
    },
    // osm relation 1355374
    'round-valley': {
      id: 'round-valley',
      name: 'Round Valley Reservoir',
      kind: 'water',
      lngLat: [-120.95902, 40.10716],
      minZoom: 11
    },
    // osm way 92797725
    'cresta-dam': {
      id: 'cresta-dam',
      name: 'Cresta Dam',
      kind: 'landmark',
      lngLat: [-121.37374, 39.87613],
      minZoom: 11,
      note: 'Dixie Fire ignition: west of the dam, 13 Jul 2021 (Cal Fire)',
      aliases: ['Dixie Fire ignition']
    },
    // wiki
    'lassen-peak': {
      id: 'lassen-peak',
      name: 'Lassen Peak',
      kind: 'peak',
      lngLat: [-121.505, 40.48806],
      minZoom: 8.5,
      priority: 2,
      elevationM: 3187
    },
    // wiki
    susanville: {
      id: 'susanville',
      name: 'Susanville',
      kind: 'city',
      lngLat: [-120.65306, 40.41639],
      minZoom: 8.5
    },
    // wiki
    quincy: {
      id: 'quincy',
      name: 'Quincy',
      kind: 'city',
      lngLat: [-120.94806, 39.93639],
      minZoom: 9
    },
    // wiki
    paradise: {
      id: 'paradise',
      name: 'Paradise',
      kind: 'city',
      lngLat: [-121.62194, 39.75972],
      minZoom: 8.5
    },
    // wiki
    oroville: {
      id: 'oroville',
      name: 'Oroville',
      kind: 'city',
      lngLat: [-121.55667, 39.51361],
      minZoom: 8.5
    },
    // osm way 92800645 vertex
    'north-fork-feather': {
      id: 'north-fork-feather',
      name: 'North Fork Feather River',
      kind: 'river',
      lngLat: [-121.27683, 39.97977],
      minZoom: 10
    },
    // osm way 93191807 vertex
    'middle-fork-feather': {
      id: 'middle-fork-feather',
      name: 'Middle Fork Feather River',
      kind: 'river',
      lngLat: [-121.07851, 39.79305],
      minZoom: 10
    },
    // wiki
    'plumas-county': {
      id: 'plumas-county',
      name: 'Plumas County',
      kind: 'region',
      lngLat: [-120.83, 40.01],
      minZoom: 8
    },
    // wiki
    'butte-county': {
      id: 'butte-county',
      name: 'Butte County',
      kind: 'region',
      lngLat: [-121.6, 39.66],
      minZoom: 8
    },
    // wiki
    'lassen-county': {
      id: 'lassen-county',
      name: 'Lassen County',
      kind: 'region',
      lngLat: [-120.62122, 40.71529],
      minZoom: 8
    }
  }
};
