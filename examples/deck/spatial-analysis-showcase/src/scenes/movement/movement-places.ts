// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Gazetteer} from '../../cartography/gazetteer';

/**
 * Europe and West Africa along the East Atlantic flyway: the places the bird stories (season,
 * timing, flyways, stopovers, gulls) name. Chapter-local until the shared gazetteer adds a flyway
 * set (requested in the movement handoff). Seas, the Wadden Sea, the Bay of Biscay and the Strait
 * of Gibraltar are in `WORLD`; take them from there.
 *
 * Regions are label placements (no marker); wetlands and towns carry the coordinate of their
 * Wikipedia infobox, rounded. At the zooms these stories use (3-7) a 0.05 degree error is under
 * two pixels; snap finding notes to the data (a stop cluster, the bird's fixes) all the same.
 */
export const FLYWAY: Gazetteer = {
  id: 'flyway',
  name: 'East Atlantic flyway',
  places: {
    // label placement: the Belgian lowland between the coast and Antwerp
    flanders: {
      id: 'flanders',
      name: 'Flanders',
      kind: 'region',
      lngLat: [3.75, 51.05],
      minZoom: 4.5,
      priority: 2,
      size: 'small'
    },
    // label placement over the Low Countries (Belgium and the Netherlands)
    'low-countries': {
      id: 'low-countries',
      name: 'Low Countries',
      kind: 'region',
      lngLat: [5.2, 52.0],
      minZoom: 2.5,
      priority: 2,
      size: 'small'
    },
    // label placement: the Iberian peninsula
    iberia: {
      id: 'iberia',
      name: 'Iberia',
      kind: 'region',
      lngLat: [-4.0, 40.2],
      minZoom: 2.5,
      priority: 2,
      size: 'medium'
    },
    // label placement: central Sahara
    sahara: {
      id: 'sahara',
      name: 'Sahara',
      kind: 'region',
      lngLat: [1.0, 23.5],
      minZoom: 2.5,
      priority: 2,
      size: 'large'
    },
    // label placement: the Sahel belt in Mali and Mauritania
    sahel: {
      id: 'sahel',
      name: 'Sahel',
      kind: 'region',
      lngLat: [-6.0, 15.4],
      minZoom: 2.5,
      priority: 3,
      size: 'medium'
    },
    // Wikipedia infobox, Inner Niger Delta: 15.2 N 4.1 W
    'inner-niger-delta': {
      id: 'inner-niger-delta',
      name: 'Inner Niger Delta',
      kind: 'site',
      lngLat: [-4.1, 15.2],
      minZoom: 4,
      priority: 2
    },
    // Wikipedia infobox, Banc d'Arguin National Park: 20.233 N 16.100 W
    'banc-d-arguin': {
      id: 'banc-d-arguin',
      name: "Banc d'Arguin",
      kind: 'site',
      lngLat: [-16.1, 20.23],
      minZoom: 3.5,
      priority: 2
    },
    // Wikipedia infobox, Djoudj National Bird Sanctuary (Senegal delta): 16.358 N 16.274 W
    'senegal-delta': {
      id: 'senegal-delta',
      name: 'Senegal delta',
      kind: 'site',
      lngLat: [-16.27, 16.36],
      minZoom: 4,
      priority: 1
    },
    // Wikipedia infobox, Doñana National Park: 36.99 N 6.44 W
    donana: {
      id: 'donana',
      name: 'Doñana',
      kind: 'site',
      lngLat: [-6.44, 36.99],
      minZoom: 4.5,
      priority: 2,
      aliases: ['Donana', 'Guadalquivir marshes']
    },
    // Wikipedia infobox, Tagus Estuary Natural Reserve: 38.87 N 8.97 W
    'tagus-estuary': {
      id: 'tagus-estuary',
      name: 'Tagus estuary',
      kind: 'water',
      lngLat: [-8.97, 38.82],
      minZoom: 5,
      priority: 1,
      size: 'small'
    },
    // label placement: the Gulf of Cadiz off the Guadalquivir mouth
    'gulf-of-cadiz': {
      id: 'gulf-of-cadiz',
      name: 'Gulf of Cádiz',
      kind: 'water',
      lngLat: [-7.2, 36.6],
      minZoom: 5,
      priority: 1,
      size: 'small'
    },
    // label placement: the Atlantic plains of Morocco (Chaouia and Doukkala)
    'atlantic-morocco': {
      id: 'atlantic-morocco',
      name: 'Atlantic plains of Morocco',
      kind: 'region',
      lngLat: [-7.6, 32.6],
      minZoom: 4.5,
      priority: 1,
      size: 'small'
    },
    // label placement: the Vendée and Charente marshes on the French Atlantic coast
    vendee: {
      id: 'vendee',
      name: 'Vendée coast',
      kind: 'region',
      lngLat: [-1.6, 46.6],
      minZoom: 5,
      priority: 1,
      size: 'small'
    },
    // label placement: the English Midlands (inland gull roosts and landfills)
    'english-midlands': {
      id: 'english-midlands',
      name: 'English Midlands',
      kind: 'region',
      lngLat: [-1.4, 52.5],
      minZoom: 5,
      priority: 1,
      size: 'small'
    },
    // Wikipedia, Zeebrugge (port of Bruges): 51.33 N 3.20 E
    zeebrugge: {
      id: 'zeebrugge',
      name: 'Zeebrugge',
      kind: 'port',
      lngLat: [3.2, 51.33],
      minZoom: 6,
      priority: 2
    },
    // Wikipedia, Ostend: 51.23 N 2.91 E
    ostend: {
      id: 'ostend',
      name: 'Ostend',
      kind: 'port',
      lngLat: [2.91, 51.23],
      minZoom: 6.5,
      priority: 1
    },
    // Wikipedia, Vlissingen: 51.44 N 3.57 E
    vlissingen: {
      id: 'vlissingen',
      name: 'Vlissingen',
      kind: 'port',
      lngLat: [3.57, 51.44],
      minZoom: 6.5,
      priority: 1
    }
  }
};
