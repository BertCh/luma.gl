// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Gazetteer} from './types';

/**
 * Chicago and Cook County: community areas, lakefront parks, the river, airports and landmarks. Serves about 40 stories.
 * Every coordinate is `[longitude, latitude]` and carries its source in the comment above it.
 */
export const CHICAGO: Gazetteer = {
  id: 'chicago',
  name: 'Chicago',
  places: {
    // label placement, 9 km east of the Loop shore, outside every community area
    'lake-michigan': {
      id: 'lake-michigan',
      name: 'Lake Michigan',
      kind: 'water',
      lngLat: [-87.5, 41.9],
      minZoom: 8.5,
      priority: 3,
      size: 'large'
    },
    // chicago community area pole
    loop: {
      id: 'loop',
      name: 'The Loop',
      kind: 'district',
      lngLat: [-87.62719, 41.87945],
      minZoom: 10,
      priority: 3
    },
    // chicago community area pole
    'near-north-side': {
      id: 'near-north-side',
      name: 'Near North Side',
      kind: 'district',
      lngLat: [-87.63151, 41.89682],
      minZoom: 10.5,
      priority: 1
    },
    // chicago community area pole
    'near-west-side': {
      id: 'near-west-side',
      name: 'Near West Side',
      kind: 'district',
      lngLat: [-87.65846, 41.87433],
      minZoom: 10.5,
      priority: 1
    },
    // chicago community area pole
    'near-south-side': {
      id: 'near-south-side',
      name: 'Near South Side',
      kind: 'district',
      lngLat: [-87.62189, 41.86037],
      minZoom: 10.5,
      priority: 1
    },
    // chicago community area pole
    'rogers-park': {
      id: 'rogers-park',
      name: 'Rogers Park',
      kind: 'district',
      lngLat: [-87.66907, 42.00626],
      minZoom: 10,
      priority: 1
    },
    // chicago community area pole
    uptown: {
      id: 'uptown',
      name: 'Uptown',
      kind: 'district',
      lngLat: [-87.65409, 41.96399],
      minZoom: 10.5
    },
    // chicago community area pole
    lakeview: {
      id: 'lakeview',
      name: 'Lakeview',
      kind: 'district',
      lngLat: [-87.65714, 41.94352],
      minZoom: 10.5
    },
    // osm node 153509185
    'logan-square': {
      id: 'logan-square',
      name: 'Logan Square',
      kind: 'neighborhood',
      lngLat: [-87.70676, 41.92837],
      minZoom: 11
    },
    // osm node 353790529
    'wicker-park': {
      id: 'wicker-park',
      name: 'Wicker Park',
      kind: 'neighborhood',
      lngLat: [-87.67747, 41.9115],
      minZoom: 11
    },
    // chicago community area pole
    austin: {
      id: 'austin',
      name: 'Austin',
      kind: 'district',
      lngLat: [-87.75829, 41.9025],
      minZoom: 10.5,
      priority: 1
    },
    // osm node 153715851
    pilsen: {
      id: 'pilsen',
      name: 'Pilsen',
      kind: 'neighborhood',
      lngLat: [-87.66188, 41.85704],
      minZoom: 11
    },
    // osm node 10700333291
    'little-village': {
      id: 'little-village',
      name: 'Little Village',
      kind: 'neighborhood',
      lngLat: [-87.7148, 41.84439],
      minZoom: 11
    },
    // osm node 9584056387
    bronzeville: {
      id: 'bronzeville',
      name: 'Bronzeville',
      kind: 'neighborhood',
      lngLat: [-87.61715, 41.83134],
      minZoom: 11
    },
    // chicago community area pole
    'hyde-park': {
      id: 'hyde-park',
      name: 'Hyde Park',
      kind: 'district',
      lngLat: [-87.59122, 41.79426],
      minZoom: 10.5,
      priority: 1
    },
    // chicago community area pole
    englewood: {
      id: 'englewood',
      name: 'Englewood',
      kind: 'district',
      lngLat: [-87.64171, 41.78474],
      minZoom: 10.5,
      priority: 1
    },
    // chicago community area pole
    'south-shore': {
      id: 'south-shore',
      name: 'South Shore',
      kind: 'district',
      lngLat: [-87.57698, 41.76252],
      minZoom: 10.5,
      priority: 1
    },
    // chicago community area pole
    'albany-park': {
      id: 'albany-park',
      name: 'Albany Park',
      kind: 'district',
      lngLat: [-87.72206, 41.96739],
      minZoom: 11
    },
    // chicago community area pole
    'edison-park': {
      id: 'edison-park',
      name: 'Edison Park',
      kind: 'district',
      lngLat: [-87.81405, 42.0025],
      minZoom: 11
    },
    // chicago community area pole
    'norwood-park': {
      id: 'norwood-park',
      name: 'Norwood Park',
      kind: 'district',
      lngLat: [-87.80225, 41.98493],
      minZoom: 11
    },
    // chicago community area pole
    'fuller-park': {
      id: 'fuller-park',
      name: 'Fuller Park',
      kind: 'district',
      lngLat: [-87.63258, 41.812],
      minZoom: 11.5
    },
    // chicago community area pole
    bridgeport: {
      id: 'bridgeport',
      name: 'Bridgeport',
      kind: 'neighborhood',
      lngLat: [-87.64906, 41.83873],
      minZoom: 11
    },
    // wiki
    evanston: {
      id: 'evanston',
      name: 'Evanston',
      kind: 'city',
      lngLat: [-87.685, 42.04722],
      minZoom: 9.5
    },
    // wiki
    'oak-park': {
      id: 'oak-park',
      name: 'Oak Park',
      kind: 'city',
      lngLat: [-87.78944, 41.88833],
      minZoom: 10
    },
    // wiki
    'forest-park': {
      id: 'forest-park',
      name: 'Forest Park',
      kind: 'city',
      lngLat: [-87.81111, 41.87306],
      minZoom: 10.5
    },
    // wiki
    cicero: {
      id: 'cicero',
      name: 'Cicero',
      kind: 'city',
      lngLat: [-87.75917, 41.84444],
      minZoom: 10.5
    },
    // cpd park boundary, widest point near the park midlatitude
    'lincoln-park': {
      id: 'lincoln-park',
      name: 'Lincoln Park',
      kind: 'park',
      lngLat: [-87.637, 41.94311],
      minZoom: 10,
      priority: 2
    },
    // cpd park boundary pole
    'jackson-park': {
      id: 'jackson-park',
      name: 'Jackson Park',
      kind: 'park',
      lngLat: [-87.58095, 41.77978],
      minZoom: 10.5,
      priority: 2
    },
    // cpd park boundary pole
    'washington-park': {
      id: 'washington-park',
      name: 'Washington Park',
      kind: 'park',
      lngLat: [-87.61101, 41.78998],
      minZoom: 11
    },
    // cpd park boundary pole
    'grant-park': {
      id: 'grant-park',
      name: 'Grant Park',
      kind: 'park',
      lngLat: [-87.61997, 41.87908],
      minZoom: 11
    },
    // cpd park boundary pole
    'humboldt-park': {
      id: 'humboldt-park',
      name: 'Humboldt Park',
      kind: 'park',
      lngLat: [-87.70202, 41.9065],
      minZoom: 10.5,
      priority: 2
    },
    // cpd park boundary pole
    'garfield-park': {
      id: 'garfield-park',
      name: 'Garfield Park',
      kind: 'park',
      lngLat: [-87.71803, 41.88292],
      minZoom: 11
    },
    // wiki
    'northerly-island': {
      id: 'northerly-island',
      name: 'Northerly Island',
      kind: 'park',
      lngLat: [-87.6083, 41.8604],
      minZoom: 11.5
    },
    // osm way 23946659
    'montrose-point': {
      id: 'montrose-point',
      name: 'Montrose Point',
      kind: 'park',
      lngLat: [-87.63317, 41.96299],
      minZoom: 10.5,
      priority: 2,
      note: 'bird sanctuary on the Lincoln Park lakefront'
    },
    // osm relation 17766874
    'montrose-harbor': {
      id: 'montrose-harbor',
      name: 'Montrose Harbor',
      kind: 'water',
      lngLat: [-87.63923, 41.96106],
      minZoom: 12
    },
    // osm relation 18995982 pole
    'wooded-island': {
      id: 'wooded-island',
      name: 'Wooded Island',
      kind: 'park',
      lngLat: [-87.58267, 41.78606],
      minZoom: 13
    },
    // cpd park boundary pole
    'big-marsh': {
      id: 'big-marsh',
      name: 'Big Marsh',
      kind: 'park',
      lngLat: [-87.56949, 41.68998],
      minZoom: 11.5
    },
    // osm way 34088847
    'wolf-lake': {
      id: 'wolf-lake',
      name: 'Wolf Lake',
      kind: 'water',
      lngLat: [-87.53155, 41.6715],
      minZoom: 11
    },
    // wiki
    'lake-calumet': {
      id: 'lake-calumet',
      name: 'Lake Calumet',
      kind: 'water',
      lngLat: [-87.59, 41.68],
      minZoom: 11
    },
    // osm node 4591260894
    'labagh-woods': {
      id: 'labagh-woods',
      name: 'Labagh Woods',
      kind: 'park',
      lngLat: [-87.7479, 41.97753],
      minZoom: 11.5
    },
    // osm relation 20951526
    'dan-ryan-woods': {
      id: 'dan-ryan-woods',
      name: 'Dan Ryan Woods',
      kind: 'park',
      lngLat: [-87.67988, 41.73153],
      minZoom: 11.5
    },
    // osm way 407267816 vertex
    'chicago-river': {
      id: 'chicago-river',
      name: 'Chicago River',
      kind: 'river',
      lngLat: [-87.63755, 41.88654],
      minZoom: 11.5
    },
    // osm way 25003663 vertex
    'north-branch': {
      id: 'north-branch',
      name: 'North Branch Chicago River',
      kind: 'river',
      lngLat: [-87.69472, 41.96036],
      minZoom: 11.5
    },
    // osm way 24984723 vertex
    'south-branch': {
      id: 'south-branch',
      name: 'South Branch Chicago River',
      kind: 'river',
      lngLat: [-87.6467, 41.84927],
      minZoom: 11.5
    },
    // osm way 164124999 vertex
    'calumet-river': {
      id: 'calumet-river',
      name: 'Calumet River',
      kind: 'river',
      lngLat: [-87.52961, 41.73294],
      minZoom: 11.5
    },
    // wiki
    ohare: {
      id: 'ohare',
      name: "O'Hare Airport",
      kind: 'airport',
      lngLat: [-87.90472, 41.97861],
      minZoom: 9,
      priority: 2,
      aliases: ['ORD', "O'Hare"]
    },
    // wiki
    midway: {
      id: 'midway',
      name: 'Midway Airport',
      kind: 'airport',
      lngLat: [-87.7525, 41.78611],
      minZoom: 9.5,
      priority: 2,
      aliases: ['MDW', 'Midway']
    },
    // wiki
    'union-station': {
      id: 'union-station',
      name: 'Union Station',
      kind: 'station',
      lngLat: [-87.6402, 41.8786],
      minZoom: 12
    },
    // wiki
    ogilvie: {
      id: 'ogilvie',
      name: 'Ogilvie Transportation Center',
      kind: 'station',
      lngLat: [-87.64029, 41.88269],
      minZoom: 12
    },
    // wiki
    howard: {
      id: 'howard',
      name: 'Howard',
      kind: 'station',
      lngLat: [-87.67309, 42.01916],
      minZoom: 11.5,
      aliases: ['Howard station']
    },
    // wiki
    'ninety-fifth': {
      id: 'ninety-fifth',
      name: '95th/Dan Ryan',
      kind: 'station',
      lngLat: [-87.62439, 41.7226],
      minZoom: 11.5
    },
    // wiki
    kimball: {
      id: 'kimball',
      name: 'Kimball',
      kind: 'station',
      lngLat: [-87.71293, 41.96763],
      minZoom: 11.5
    },
    // osm node 13166659192
    'forest-park-terminal': {
      id: 'forest-park-terminal',
      name: 'Forest Park (Blue Line)',
      kind: 'station',
      lngLat: [-87.81676, 41.87427],
      minZoom: 11.5
    },
    // wiki
    'ohare-station': {
      id: 'ohare-station',
      name: "O'Hare (Blue Line)",
      kind: 'station',
      lngLat: [-87.90088, 41.98113],
      minZoom: 11.5
    },
    // wiki
    'midway-station': {
      id: 'midway-station',
      name: 'Midway (Orange Line)',
      kind: 'station',
      lngLat: [-87.73788, 41.78661],
      minZoom: 11.5
    },
    // wiki
    'willis-tower': {
      id: 'willis-tower',
      name: 'Willis Tower',
      kind: 'landmark',
      lngLat: [-87.6358, 41.8789],
      minZoom: 12.5
    },
    // wiki
    'wrigley-field': {
      id: 'wrigley-field',
      name: 'Wrigley Field',
      kind: 'landmark',
      lngLat: [-87.65556, 41.94806],
      minZoom: 12.5
    },
    // wiki
    'soldier-field': {
      id: 'soldier-field',
      name: 'Soldier Field',
      kind: 'landmark',
      lngLat: [-87.6167, 41.8623],
      minZoom: 12.5
    },
    // wiki
    'united-center': {
      id: 'united-center',
      name: 'United Center',
      kind: 'landmark',
      lngLat: [-87.67417, 41.88056],
      minZoom: 12.5
    },
    // wiki
    'rate-field': {
      id: 'rate-field',
      name: 'Rate Field',
      kind: 'landmark',
      lngLat: [-87.63389, 41.83],
      minZoom: 12.5,
      aliases: ['Guaranteed Rate Field']
    },
    // osm way 24826230
    msi: {
      id: 'msi',
      name: 'Museum of Science and Industry',
      kind: 'landmark',
      lngLat: [-87.58293, 41.79075],
      minZoom: 12.5,
      aliases: ['MSI']
    },
    // wiki
    'navy-pier': {
      id: 'navy-pier',
      name: 'Navy Pier',
      kind: 'landmark',
      lngLat: [-87.59972, 41.89139],
      minZoom: 12.5
    }
  }
};
