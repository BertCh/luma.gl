// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Gazetteer} from './types';

/**
 * Montreal island: boroughs, Mount Royal, parks, the canal and river.
 * Every coordinate is `[longitude, latitude]` and carries its source in the comment above it.
 */
export const MONTREAL: Gazetteer = {
  id: 'montreal',
  name: 'Montreal',
  places: {
    // osm relation 12713328
    downtown: {
      id: 'downtown',
      name: 'Downtown',
      kind: 'district',
      lngLat: [-73.57045, 45.49555],
      minZoom: 11,
      priority: 2
    },
    // osm relation 9937244
    'old-montreal': {
      id: 'old-montreal',
      name: 'Old Montreal',
      kind: 'district',
      lngLat: [-73.55541, 45.50221],
      minZoom: 12
    },
    // montreal borough boundary pole
    plateau: {
      id: 'plateau',
      name: 'Plateau-Mont-Royal',
      kind: 'district',
      lngLat: [-73.58062, 45.52591],
      minZoom: 11,
      priority: 1,
      aliases: ['Plateau']
    },
    // osm relation 9940746
    'mile-end': {
      id: 'mile-end',
      name: 'Mile End',
      kind: 'neighborhood',
      lngLat: [-73.59512, 45.52314],
      minZoom: 12
    },
    // montreal borough boundary pole
    outremont: {
      id: 'outremont',
      name: 'Outremont',
      kind: 'district',
      lngLat: [-73.61046, 45.51687],
      minZoom: 11.5
    },
    // montreal borough boundary pole
    verdun: {
      id: 'verdun',
      name: 'Verdun',
      kind: 'district',
      lngLat: [-73.55139, 45.44955],
      minZoom: 11.5
    },
    // montreal borough boundary pole
    rosemont: {
      id: 'rosemont',
      name: 'Rosemont-La Petite-Patrie',
      kind: 'district',
      lngLat: [-73.5737, 45.55488],
      minZoom: 11.5,
      aliases: ['Rosemont']
    },
    // montreal borough boundary pole
    villeray: {
      id: 'villeray',
      name: 'Villeray',
      kind: 'district',
      lngLat: [-73.61706, 45.56678],
      minZoom: 11.5
    },
    // montreal borough boundary pole
    ahuntsic: {
      id: 'ahuntsic',
      name: 'Ahuntsic',
      kind: 'district',
      lngLat: [-73.65533, 45.55232],
      minZoom: 11
    },
    // osm relation 15976157
    hochelaga: {
      id: 'hochelaga',
      name: 'Hochelaga-Maisonneuve',
      kind: 'district',
      lngLat: [-73.54004, 45.55147],
      minZoom: 11.5,
      aliases: ['Hochelaga']
    },
    // wiki
    'mount-royal': {
      id: 'mount-royal',
      name: 'Mount Royal',
      kind: 'peak',
      lngLat: [-73.58889, 45.50639],
      minZoom: 11,
      priority: 1,
      elevationM: 233
    },
    // wiki
    'parc-jean-drapeau': {
      id: 'parc-jean-drapeau',
      name: 'Parc Jean-Drapeau',
      kind: 'park',
      lngLat: [-73.5274, 45.5095],
      minZoom: 11.5
    },
    // wiki
    'parc-la-fontaine': {
      id: 'parc-la-fontaine',
      name: 'Parc La Fontaine',
      kind: 'park',
      lngLat: [-73.5689, 45.5267],
      minZoom: 12.5
    },
    // wiki
    'parc-jeanne-mance': {
      id: 'parc-jeanne-mance',
      name: 'Parc Jeanne-Mance',
      kind: 'park',
      lngLat: [-73.5842, 45.5158],
      minZoom: 13
    },
    // osm way 121548871 vertex
    'lachine-canal': {
      id: 'lachine-canal',
      name: 'Lachine Canal',
      kind: 'water',
      lngLat: [-73.59848, 45.46331],
      minZoom: 12
    },
    // wiki
    trudeau: {
      id: 'trudeau',
      name: 'Trudeau Airport',
      kind: 'airport',
      lngLat: [-73.74083, 45.47056],
      minZoom: 9.5,
      aliases: ['YUL']
    },
    // wiki
    'berri-uqam': {
      id: 'berri-uqam',
      name: 'Berri-UQAM',
      kind: 'station',
      lngLat: [-73.56111, 45.51528],
      minZoom: 12.5
    },
    // wiki
    longueuil: {
      id: 'longueuil',
      name: 'Longueuil',
      kind: 'city',
      lngLat: [-73.51667, 45.53333],
      minZoom: 10.5
    },
    // osm relation 1747225 vertex
    'riviere-des-prairies': {
      id: 'riviere-des-prairies',
      name: 'Rivière des Prairies',
      kind: 'river',
      lngLat: [-73.65532, 45.58189],
      minZoom: 10.5
    },
    // osm relation 6122656 vertex
    'saint-lawrence': {
      id: 'saint-lawrence',
      name: 'Saint Lawrence River',
      kind: 'river',
      lngLat: [-73.55841, 45.4611],
      minZoom: 10
    }
  }
};
