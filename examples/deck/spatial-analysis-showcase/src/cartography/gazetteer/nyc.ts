// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Gazetteer} from './types';

/**
 * New York City and the harbour: boroughs, bays, rivers, airports, ferry terminals and landmarks.
 * Every coordinate is `[longitude, latitude]` and carries its source in the comment above it.
 */
export const NYC: Gazetteer = {
  id: 'nyc',
  name: 'New York City',
  places: {
    // nyc borough boundary pole, excluding the Central Park band
    manhattan: {
      id: 'manhattan',
      name: 'Manhattan',
      kind: 'district',
      lngLat: [-73.992, 40.725],
      minZoom: 9,
      priority: 2
    },
    // nyc borough boundary pole
    brooklyn: {
      id: 'brooklyn',
      name: 'Brooklyn',
      kind: 'district',
      lngLat: [-73.97125, 40.62988],
      minZoom: 9,
      priority: 2
    },
    // nyc borough boundary pole
    queens: {
      id: 'queens',
      name: 'Queens',
      kind: 'district',
      lngLat: [-73.80257, 40.71771],
      minZoom: 9,
      priority: 2
    },
    // nyc borough boundary pole
    bronx: {
      id: 'bronx',
      name: 'The Bronx',
      kind: 'district',
      lngLat: [-73.86429, 40.8673],
      minZoom: 9,
      priority: 2,
      aliases: ['Bronx']
    },
    // nyc borough boundary pole
    'staten-island': {
      id: 'staten-island',
      name: 'Staten Island',
      kind: 'district',
      lngLat: [-74.13872, 40.59342],
      minZoom: 9,
      priority: 2
    },
    // wiki
    midtown: {
      id: 'midtown',
      name: 'Midtown',
      kind: 'district',
      lngLat: [-73.9842, 40.7547],
      minZoom: 11
    },
    // wiki
    'lower-manhattan': {
      id: 'lower-manhattan',
      name: 'Lower Manhattan',
      kind: 'district',
      lngLat: [-74.0119, 40.7075],
      minZoom: 11,
      aliases: ['Downtown']
    },
    // wiki
    'central-park': {
      id: 'central-park',
      name: 'Central Park',
      kind: 'park',
      lngLat: [-73.96528, 40.78222],
      minZoom: 10.5
    },
    // wiki
    'prospect-park': {
      id: 'prospect-park',
      name: 'Prospect Park',
      kind: 'park',
      lngLat: [-73.97083, 40.66167],
      minZoom: 11.5
    },
    // wiki
    'jersey-city': {
      id: 'jersey-city',
      name: 'Jersey City',
      kind: 'city',
      lngLat: [-74.06, 40.71],
      minZoom: 10
    },
    // wiki
    newark: {id: 'newark', name: 'Newark', kind: 'city', lngLat: [-74.1722, 40.7356], minZoom: 9.5},
    // wiki
    hoboken: {
      id: 'hoboken',
      name: 'Hoboken',
      kind: 'city',
      lngLat: [-74.0325, 40.745],
      minZoom: 11.5
    },
    // wiki
    jfk: {
      id: 'jfk',
      name: 'JFK',
      kind: 'airport',
      lngLat: [-73.77889, 40.63972],
      minZoom: 9,
      priority: 2,
      aliases: ['John F. Kennedy International Airport']
    },
    // wiki
    lga: {
      id: 'lga',
      name: 'LaGuardia',
      kind: 'airport',
      lngLat: [-73.875, 40.775],
      minZoom: 9.5,
      priority: 2,
      aliases: ['LGA']
    },
    // wiki
    ewr: {
      id: 'ewr',
      name: 'Newark Airport',
      kind: 'airport',
      lngLat: [-74.16861, 40.6925],
      minZoom: 9,
      priority: 2,
      aliases: ['EWR']
    },
    // wiki
    'penn-station': {
      id: 'penn-station',
      name: 'Penn Station',
      kind: 'station',
      lngLat: [-73.9939, 40.75064],
      minZoom: 12
    },
    // wiki
    'grand-central': {
      id: 'grand-central',
      name: 'Grand Central',
      kind: 'station',
      lngLat: [-73.9772, 40.7528],
      minZoom: 12
    },
    // wiki
    'statue-of-liberty': {
      id: 'statue-of-liberty',
      name: 'Statue of Liberty',
      kind: 'landmark',
      lngLat: [-74.04444, 40.68917],
      minZoom: 11.5
    },
    // wiki
    'governors-island': {
      id: 'governors-island',
      name: 'Governors Island',
      kind: 'landmark',
      lngLat: [-74.01611, 40.69139],
      minZoom: 12
    },
    // wiki
    'ellis-island': {
      id: 'ellis-island',
      name: 'Ellis Island',
      kind: 'landmark',
      lngLat: [-74.03972, 40.69944],
      minZoom: 12.5
    },
    // wiki
    'st-george-terminal': {
      id: 'st-george-terminal',
      name: 'St. George Ferry Terminal',
      kind: 'port',
      lngLat: [-74.07417, 40.64333],
      minZoom: 12.5
    },
    // wiki
    'whitehall-terminal': {
      id: 'whitehall-terminal',
      name: 'Whitehall Terminal',
      kind: 'port',
      lngLat: [-74.01313, 40.70141],
      minZoom: 12.5
    },
    // wiki
    'port-newark': {
      id: 'port-newark',
      name: 'Port Newark-Elizabeth',
      kind: 'port',
      lngLat: [-74.1505, 40.68155],
      minZoom: 10.5,
      priority: 1
    },
    // wiki
    'red-hook-terminal': {
      id: 'red-hook-terminal',
      name: 'Red Hook Container Terminal',
      kind: 'port',
      lngLat: [-74.00333, 40.6875],
      minZoom: 12
    },
    // wiki
    verrazzano: {
      id: 'verrazzano',
      name: 'Verrazzano-Narrows Bridge',
      kind: 'landmark',
      lngLat: [-74.04556, 40.60639],
      minZoom: 11.5
    },
    // wiki
    'the-narrows': {
      id: 'the-narrows',
      name: 'The Narrows',
      kind: 'water',
      lngLat: [-74.04806, 40.61333],
      minZoom: 11.5
    },
    // wiki
    'upper-bay': {
      id: 'upper-bay',
      name: 'Upper Bay',
      kind: 'water',
      lngLat: [-74.04556, 40.66833],
      minZoom: 10,
      priority: 2
    },
    // wiki
    'lower-bay': {
      id: 'lower-bay',
      name: 'Lower Bay',
      kind: 'water',
      lngLat: [-74.04972, 40.51667],
      minZoom: 9.5,
      priority: 2
    },
    // osm way 459202666 vertex
    'hudson-river': {
      id: 'hudson-river',
      name: 'Hudson River',
      kind: 'river',
      lngLat: [-74.0172, 40.74919],
      minZoom: 10
    },
    // osm relation 5912630 vertex
    'east-river': {
      id: 'east-river',
      name: 'East River',
      kind: 'river',
      lngLat: [-73.94855, 40.76],
      minZoom: 10.5
    },
    // wiki
    'newark-bay': {
      id: 'newark-bay',
      name: 'Newark Bay',
      kind: 'water',
      lngLat: [-74.13145, 40.6796],
      minZoom: 11
    },
    // wiki
    'jamaica-bay': {
      id: 'jamaica-bay',
      name: 'Jamaica Bay',
      kind: 'water',
      lngLat: [-73.8425, 40.61778],
      minZoom: 10
    },
    // wiki
    'raritan-bay': {
      id: 'raritan-bay',
      name: 'Raritan Bay',
      kind: 'water',
      lngLat: [-74.18306, 40.48333],
      minZoom: 9.5
    },
    // wiki
    'ambrose-channel': {
      id: 'ambrose-channel',
      name: 'Ambrose Channel',
      kind: 'water',
      lngLat: [-73.99042, 40.51844],
      minZoom: 10.5
    },
    // wiki
    'kill-van-kull': {
      id: 'kill-van-kull',
      name: 'Kill Van Kull',
      kind: 'water',
      lngLat: [-74.12, 40.644],
      minZoom: 12
    }
  }
};
