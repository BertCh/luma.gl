// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Gazetteer} from './types';

/**
 * The Zermatt valley, Matterhorn and Monte Rosa massif: peaks, glaciers, lakes and the Gornergrat railway.
 * Every coordinate is `[longitude, latitude]` and carries its source in the comment above it.
 */
export const ALPS: Gazetteer = {
  id: 'alps',
  name: 'Alps (Zermatt)',
  places: {
    // wiki
    matterhorn: {
      id: 'matterhorn',
      name: 'Matterhorn',
      kind: 'peak',
      lngLat: [7.65861, 45.97639],
      minZoom: 9,
      priority: 3,
      elevationM: 4478,
      note: 'summit; DEM cell reads 4,476 m'
    },
    // swisstopo
    zermatt: {
      id: 'zermatt',
      name: 'Zermatt',
      kind: 'city',
      lngLat: [7.74657, 46.01754],
      minZoom: 9,
      priority: 3,
      elevationM: 1608
    },
    // swisstopo
    gornergrat: {
      id: 'gornergrat',
      name: 'Gornergrat',
      kind: 'peak',
      lngLat: [7.78463, 45.98332],
      minZoom: 11,
      priority: 2,
      elevationM: 3135,
      note: 'summit; station 3,089 m; DEM cell 3,124 m'
    },
    // wiki
    'gornergrat-station': {
      id: 'gornergrat-station',
      name: 'Gornergrat station',
      kind: 'station',
      lngLat: [7.7822, 45.9833],
      minZoom: 12,
      elevationM: 3089
    },
    // swisstopo
    dufourspitze: {
      id: 'dufourspitze',
      name: 'Dufourspitze',
      kind: 'peak',
      lngLat: [7.86676, 45.93692],
      minZoom: 9.5,
      priority: 2,
      elevationM: 4634,
      aliases: ['Monte Rosa']
    },
    // wiki
    liskamm: {
      id: 'liskamm',
      name: 'Liskamm',
      kind: 'peak',
      lngLat: [7.83556, 45.9225],
      minZoom: 10,
      elevationM: 4532
    },
    // swisstopo
    castor: {
      id: 'castor',
      name: 'Castor',
      kind: 'peak',
      lngLat: [7.79321, 45.92084],
      minZoom: 10.5,
      elevationM: 4225
    },
    // swisstopo
    pollux: {
      id: 'pollux',
      name: 'Pollux',
      kind: 'peak',
      lngLat: [7.7853, 45.92778],
      minZoom: 10.5,
      elevationM: 4089
    },
    // swisstopo
    breithorn: {
      id: 'breithorn',
      name: 'Breithorn',
      kind: 'peak',
      lngLat: [7.74773, 45.94101],
      minZoom: 10,
      elevationM: 4164
    },
    // swisstopo
    'klein-matterhorn': {
      id: 'klein-matterhorn',
      name: 'Klein Matterhorn',
      kind: 'peak',
      lngLat: [7.72993, 45.93842],
      minZoom: 11,
      elevationM: 3883
    },
    // swisstopo
    theodulpass: {
      id: 'theodulpass',
      name: 'Theodulpass',
      kind: 'pass',
      lngLat: [7.70883, 45.94331],
      minZoom: 11,
      elevationM: 3295,
      aliases: ['Theodul Pass']
    },
    // swisstopo
    gornergletscher: {
      id: 'gornergletscher',
      name: 'Gornergletscher',
      kind: 'glacier',
      lngLat: [7.82969, 45.96965],
      minZoom: 10,
      priority: 2
    },
    // swisstopo
    findelgletscher: {
      id: 'findelgletscher',
      name: 'Findelgletscher',
      kind: 'glacier',
      lngLat: [7.87522, 45.99347],
      minZoom: 10.5
    },
    // swisstopo
    grenzgletscher: {
      id: 'grenzgletscher',
      name: 'Grenzgletscher',
      kind: 'glacier',
      lngLat: [7.83417, 45.93822],
      minZoom: 11
    },
    // swisstopo
    riffelsee: {
      id: 'riffelsee',
      name: 'Riffelsee',
      kind: 'water',
      lngLat: [7.76209, 45.98338],
      minZoom: 12.5
    },
    // swisstopo
    stellisee: {
      id: 'stellisee',
      name: 'Stellisee',
      kind: 'water',
      lngLat: [7.8003, 46.01342],
      minZoom: 12.5
    },
    // wiki
    riffelberg: {
      id: 'riffelberg',
      name: 'Riffelberg',
      kind: 'station',
      lngLat: [7.755, 45.993],
      minZoom: 12.5,
      elevationM: 2582
    },
    // swisstopo
    dom: {
      id: 'dom',
      name: 'Dom',
      kind: 'peak',
      lngLat: [7.85887, 46.09391],
      minZoom: 9.5,
      elevationM: 4546
    },
    // swisstopo
    weisshorn: {
      id: 'weisshorn',
      name: 'Weisshorn',
      kind: 'peak',
      lngLat: [7.71607, 46.10125],
      minZoom: 9.5,
      elevationM: 4505
    },
    // swisstopo
    'dent-blanche': {
      id: 'dent-blanche',
      name: 'Dent Blanche',
      kind: 'peak',
      lngLat: [7.61191, 46.03418],
      minZoom: 9.5,
      elevationM: 4357
    },
    // swisstopo
    taschhorn: {
      id: 'taschhorn',
      name: 'Täschhorn',
      kind: 'peak',
      lngLat: [7.85718, 46.08342],
      minZoom: 10,
      elevationM: 4491
    },
    // osm node 253338916
    cervinia: {
      id: 'cervinia',
      name: 'Breuil-Cervinia',
      kind: 'city',
      lngLat: [7.63044, 45.9356],
      minZoom: 10.5
    },
    // swisstopo
    'hornli-hut': {
      id: 'hornli-hut',
      name: 'Hörnli Hut',
      kind: 'site',
      lngLat: [7.67695, 45.98214],
      minZoom: 12,
      elevationM: 3260
    },
    // swisstopo
    schwarzsee: {
      id: 'schwarzsee',
      name: 'Schwarzsee',
      kind: 'water',
      lngLat: [7.70683, 45.9908],
      minZoom: 12.5
    }
  }
};
