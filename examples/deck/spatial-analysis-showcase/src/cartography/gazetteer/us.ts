// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Gazetteer} from './types';

/**
 * The contiguous United States: states, regions, belts, ports and airport hubs.
 * Every coordinate is `[longitude, latitude]` and carries its source in the comment above it.
 */
export const US: Gazetteer = {
  id: 'us',
  name: 'United States',
  places: {
    // wiki
    atl: {
      id: 'atl',
      name: 'ATL',
      kind: 'airport',
      lngLat: [-84.42806, 33.63667],
      minZoom: 4,
      priority: 2,
      aliases: ['Atlanta', 'Hartsfield–Jackson Atlanta International Airport']
    },
    // wiki
    ord: {
      id: 'ord',
      name: 'ORD',
      kind: 'airport',
      lngLat: [-87.90472, 41.97861],
      minZoom: 4,
      priority: 2,
      aliases: ['Chicago', "O'Hare International Airport"]
    },
    // wiki
    dfw: {
      id: 'dfw',
      name: 'DFW',
      kind: 'airport',
      lngLat: [-97.03806, 32.89694],
      minZoom: 4,
      priority: 2,
      aliases: ['Dallas-Fort Worth', 'Dallas Fort Worth International Airport']
    },
    // osm way 851424893
    den: {
      id: 'den',
      name: 'DEN',
      kind: 'airport',
      lngLat: [-104.68537, 39.86067],
      minZoom: 4,
      priority: 2,
      aliases: ['Denver', 'Denver International Airport']
    },
    // wiki
    lax: {
      id: 'lax',
      name: 'LAX',
      kind: 'airport',
      lngLat: [-118.40806, 33.9425],
      minZoom: 4,
      priority: 2,
      aliases: ['Los Angeles', 'Los Angeles International Airport']
    },
    // wiki
    jfk: {
      id: 'jfk',
      name: 'JFK',
      kind: 'airport',
      lngLat: [-73.77889, 40.63972],
      minZoom: 4,
      priority: 2,
      aliases: ['New York', 'John F. Kennedy International Airport']
    },
    // wiki
    ewr: {
      id: 'ewr',
      name: 'EWR',
      kind: 'airport',
      lngLat: [-74.16861, 40.6925],
      minZoom: 4,
      priority: 1,
      aliases: ['Newark', 'Newark Liberty International Airport']
    },
    // wiki
    sfo: {
      id: 'sfo',
      name: 'SFO',
      kind: 'airport',
      lngLat: [-122.375, 37.61889],
      minZoom: 4,
      priority: 1,
      aliases: ['San Francisco', 'San Francisco International Airport']
    },
    // wiki
    sea: {
      id: 'sea',
      name: 'SEA',
      kind: 'airport',
      lngLat: [-122.30944, 47.44889],
      minZoom: 4,
      priority: 1,
      aliases: ['Seattle', 'Seattle–Tacoma International Airport']
    },
    // wiki
    mia: {
      id: 'mia',
      name: 'MIA',
      kind: 'airport',
      lngLat: [-80.29056, 25.79333],
      minZoom: 4,
      priority: 1,
      aliases: ['Miami', 'Miami International Airport']
    },
    // wiki
    las: {
      id: 'las',
      name: 'LAS',
      kind: 'airport',
      lngLat: [-115.15222, 36.08],
      minZoom: 4,
      priority: 1,
      aliases: ['Las Vegas', 'Harry Reid International Airport']
    },
    // wiki
    clt: {
      id: 'clt',
      name: 'CLT',
      kind: 'airport',
      lngLat: [-80.94306, 35.21389],
      minZoom: 4,
      priority: 1,
      aliases: ['Charlotte', 'Charlotte Douglas International Airport']
    },
    // wiki
    phx: {
      id: 'phx',
      name: 'PHX',
      kind: 'airport',
      lngLat: [-112.01167, 33.43417],
      minZoom: 4,
      priority: 1,
      aliases: ['Phoenix', 'Phoenix Sky Harbor International Airport']
    },
    // wiki
    iah: {
      id: 'iah',
      name: 'IAH',
      kind: 'airport',
      lngLat: [-95.34139, 29.98444],
      minZoom: 4,
      priority: 1,
      aliases: ['Houston', 'George Bush Intercontinental Airport']
    },
    // wiki
    bos: {
      id: 'bos',
      name: 'BOS',
      kind: 'airport',
      lngLat: [-71.00639, 42.36306],
      minZoom: 4,
      priority: 1,
      aliases: ['Boston', 'Logan International Airport']
    },
    // wiki
    msp: {
      id: 'msp',
      name: 'MSP',
      kind: 'airport',
      lngLat: [-93.22167, 44.88194],
      minZoom: 4,
      priority: 1,
      aliases: ['Minneapolis', 'Minneapolis–Saint Paul International Airport']
    },
    // wiki
    dtw: {
      id: 'dtw',
      name: 'DTW',
      kind: 'airport',
      lngLat: [-83.35333, 42.2125],
      minZoom: 4,
      priority: 1,
      aliases: ['Detroit', 'Detroit Metropolitan Wayne County Airport']
    },
    // wiki
    'port-houston': {
      id: 'port-houston',
      name: 'Houston',
      kind: 'port',
      lngLat: [-95.25, 29.71667],
      minZoom: 5,
      priority: 2
    },
    // wiki
    'port-galveston': {
      id: 'port-galveston',
      name: 'Galveston',
      kind: 'port',
      lngLat: [-94.81, 29.305],
      minZoom: 6,
      priority: 1
    },
    // wiki
    'port-la': {
      id: 'port-la',
      name: 'Los Angeles',
      kind: 'port',
      lngLat: [-118.2625, 33.73],
      minZoom: 5,
      priority: 2
    },
    // wiki
    'port-long-beach': {
      id: 'port-long-beach',
      name: 'Long Beach',
      kind: 'port',
      lngLat: [-118.215, 33.755],
      minZoom: 6,
      priority: 1
    },
    // wiki
    'port-ny-nj': {
      id: 'port-ny-nj',
      name: 'New York-New Jersey',
      kind: 'port',
      lngLat: [-74.04556, 40.66833],
      minZoom: 5,
      priority: 2
    },
    // wiki
    'port-savannah': {
      id: 'port-savannah',
      name: 'Savannah',
      kind: 'port',
      lngLat: [-81.15191, 32.1287],
      minZoom: 5,
      priority: 1
    },
    // wiki
    'port-charleston': {
      id: 'port-charleston',
      name: 'Charleston',
      kind: 'port',
      lngLat: [-79.924, 32.7846],
      minZoom: 6,
      priority: 1
    },
    // wiki
    'port-seattle': {
      id: 'port-seattle',
      name: 'Seattle',
      kind: 'port',
      lngLat: [-122.35417, 47.61389],
      minZoom: 5,
      priority: 1
    },
    // wiki
    'port-tacoma': {
      id: 'port-tacoma',
      name: 'Tacoma',
      kind: 'port',
      lngLat: [-122.40833, 47.26028],
      minZoom: 6,
      priority: 1
    },
    // wiki
    'port-oakland': {
      id: 'port-oakland',
      name: 'Oakland',
      kind: 'port',
      lngLat: [-122.2846, 37.79553],
      minZoom: 5,
      priority: 1
    },
    // wiki
    'port-hampton-roads': {
      id: 'port-hampton-roads',
      name: 'Hampton Roads',
      kind: 'port',
      lngLat: [-76.4, 36.95],
      minZoom: 5,
      priority: 1
    },
    // wiki
    'port-baltimore': {
      id: 'port-baltimore',
      name: 'Baltimore',
      kind: 'port',
      lngLat: [-76.585, 39.275],
      minZoom: 5,
      priority: 1
    },
    // wiki
    'port-new-orleans': {
      id: 'port-new-orleans',
      name: 'New Orleans',
      kind: 'port',
      lngLat: [-90.06194, 29.93694],
      minZoom: 5,
      priority: 1
    },
    // wiki
    'port-south-louisiana': {
      id: 'port-south-louisiana',
      name: 'South Louisiana',
      kind: 'port',
      lngLat: [-90.5, 30.051],
      minZoom: 6,
      priority: 0
    },
    // wiki
    'port-miami': {
      id: 'port-miami',
      name: 'Miami',
      kind: 'port',
      lngLat: [-80.16621, 25.77238],
      minZoom: 5,
      priority: 1
    },
    // wiki
    'port-jacksonville': {
      id: 'port-jacksonville',
      name: 'Jacksonville',
      kind: 'port',
      lngLat: [-81.56444, 30.38083],
      minZoom: 6,
      priority: 0
    },
    // wiki
    'port-mobile': {
      id: 'port-mobile',
      name: 'Mobile',
      kind: 'port',
      lngLat: [-88.04331, 30.71217],
      minZoom: 6,
      priority: 0
    },
    // wiki
    'port-duluth': {
      id: 'port-duluth',
      name: 'Duluth-Superior',
      kind: 'port',
      lngLat: [-92.09139, 46.77944],
      minZoom: 6,
      priority: 0
    },
    // osm way 21160211, North Port Avenue inside the port
    'port-corpus-christi': {
      id: 'port-corpus-christi',
      name: 'Corpus Christi',
      kind: 'port',
      lngLat: [-97.41055, 27.80392],
      minZoom: 6,
      priority: 0
    },
    // wiki
    'bolivar-roads': {
      id: 'bolivar-roads',
      name: 'Bolivar Roads',
      kind: 'water',
      lngLat: [-94.75269, 29.35051],
      minZoom: 8,
      note: 'Galveston Bay entrance, Houston Ship Channel gate; 33 CFR 161.35 gives 29 20.9 N 94 47.0 W',
      aliases: ['Galveston gate']
    },
    // mean of ne state label points
    'great-plains': {
      id: 'great-plains',
      name: 'Great Plains',
      kind: 'region',
      lngLat: [-99.14086, 41.47816],
      minZoom: 3,
      priority: 1,
      note: 'label placement: mean of Natural Earth state label points NE KS SD ND OK'
    },
    // mean of ne state label points
    'corn-belt': {
      id: 'corn-belt',
      name: 'Corn Belt',
      kind: 'region',
      lngLat: [-89.57593, 40.62523],
      minZoom: 3,
      priority: 1,
      note: 'label placement: mean of Natural Earth state label points IA IL IN'
    },
    // mean of ne state label points
    'rust-belt': {
      id: 'rust-belt',
      name: 'Rust Belt',
      kind: 'region',
      lngLat: [-81.74307, 41.46227],
      minZoom: 3,
      priority: 1,
      note: 'label placement: mean of Natural Earth state label points OH PA MI'
    },
    // mean of ne state label points
    'sun-belt': {
      id: 'sun-belt',
      name: 'Sun Belt',
      kind: 'region',
      lngLat: [-92.02324, 31.81361],
      minZoom: 3,
      priority: 1,
      note: 'label placement: mean of Natural Earth state label points AZ TX LA MS AL GA FL'
    },
    // mean of ne state label points
    appalachia: {
      id: 'appalachia',
      name: 'Appalachia',
      kind: 'region',
      lngLat: [-82.71757, 37.38332],
      minZoom: 3.5,
      priority: 1,
      note: 'label placement: mean of Natural Earth state label points WV KY VA TN'
    },
    // wiki
    'black-belt': {
      id: 'black-belt',
      name: 'Black Belt',
      kind: 'region',
      lngLat: [-87.0247, 32.4164],
      minZoom: 4,
      priority: 1,
      note: 'label at Selma, the Alabama Black Belt hub'
    },
    // wiki
    'mississippi-delta': {
      id: 'mississippi-delta',
      name: 'Mississippi Delta',
      kind: 'region',
      lngLat: [-90.4, 33.8],
      minZoom: 4.5,
      priority: 1
    },
    // wiki
    'four-corners': {
      id: 'four-corners',
      name: 'Four Corners',
      kind: 'site',
      lngLat: [-109.04517, 36.99898],
      minZoom: 5,
      priority: 1
    },
    // wiki
    'rio-grande-valley': {
      id: 'rio-grande-valley',
      name: 'Rio Grande Valley',
      kind: 'region',
      lngLat: [-98.12, 26.22],
      minZoom: 5
    },
    // wiki
    'front-range': {
      id: 'front-range',
      name: 'Front Range',
      kind: 'region',
      lngLat: [-105.81694, 39.63389],
      minZoom: 5
    },
    // wiki
    'navajo-nation': {
      id: 'navajo-nation',
      name: 'Navajo Nation',
      kind: 'region',
      lngLat: [-109.06389, 35.67056],
      minZoom: 5,
      note: 'label at Window Rock, the capital'
    },
    // wiki
    nantucket: {
      id: 'nantucket',
      name: 'Nantucket',
      kind: 'city',
      lngLat: [-70.09944, 41.28278],
      minZoom: 6
    },
    // wiki
    'loving-county': {
      id: 'loving-county',
      name: 'Loving County, TX',
      kind: 'region',
      lngLat: [-103.57, 31.84],
      minZoom: 6
    },
    // wiki
    'robeson-county': {
      id: 'robeson-county',
      name: 'Robeson County, NC',
      kind: 'region',
      lngLat: [-79.1, 34.64],
      minZoom: 6
    },
    // wiki
    'navajo-county': {
      id: 'navajo-county',
      name: 'Navajo County, AZ',
      kind: 'region',
      lngLat: [-110.28972, 35.49778],
      minZoom: 6
    },
    // wiki
    'cook-county': {
      id: 'cook-county',
      name: 'Cook County, IL',
      kind: 'region',
      lngLat: [-87.88889, 41.80861],
      minZoom: 6
    },
    // natural earth 110m admin_1 label point
    'state-mn': {
      id: 'state-mn',
      name: 'Minnesota',
      kind: 'region',
      lngLat: [-93.364, 46.0592],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['MN']
    },
    // natural earth 110m admin_1 label point
    'state-mt': {
      id: 'state-mt',
      name: 'Montana',
      kind: 'region',
      lngLat: [-110.044, 46.9965],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['MT']
    },
    // natural earth 110m admin_1 label point
    'state-nd': {
      id: 'state-nd',
      name: 'North Dakota',
      kind: 'region',
      lngLat: [-100.302, 47.4675],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['ND']
    },
    // natural earth 110m admin_1 label point
    'state-id': {
      id: 'state-id',
      name: 'Idaho',
      kind: 'region',
      lngLat: [-114.133, 43.7825],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['ID']
    },
    // natural earth 110m admin_1 label point
    'state-wa': {
      id: 'state-wa',
      name: 'Washington',
      kind: 'region',
      lngLat: [-120.361, 47.4865],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['WA']
    },
    // natural earth 110m admin_1 label point
    'state-az': {
      id: 'state-az',
      name: 'Arizona',
      kind: 'region',
      lngLat: [-111.935, 34.3046],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['AZ']
    },
    // natural earth 110m admin_1 label point
    'state-ca': {
      id: 'state-ca',
      name: 'California',
      kind: 'region',
      lngLat: [-119.591, 36.7496],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['CA']
    },
    // natural earth 110m admin_1 label point
    'state-co': {
      id: 'state-co',
      name: 'Colorado',
      kind: 'region',
      lngLat: [-105.543, 38.9998],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['CO']
    },
    // natural earth 110m admin_1 label point
    'state-nv': {
      id: 'state-nv',
      name: 'Nevada',
      kind: 'region',
      lngLat: [-117.02, 39.4299],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['NV']
    },
    // natural earth 110m admin_1 label point
    'state-nm': {
      id: 'state-nm',
      name: 'New Mexico',
      kind: 'region',
      lngLat: [-106.024, 34.5002],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['NM']
    },
    // natural earth 110m admin_1 label point
    'state-or': {
      id: 'state-or',
      name: 'Oregon',
      kind: 'region',
      lngLat: [-120.386, 43.8333],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['OR']
    },
    // natural earth 110m admin_1 label point
    'state-ut': {
      id: 'state-ut',
      name: 'Utah',
      kind: 'region',
      lngLat: [-111.544, 39.5007],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['UT']
    },
    // natural earth 110m admin_1 label point
    'state-wy': {
      id: 'state-wy',
      name: 'Wyoming',
      kind: 'region',
      lngLat: [-107.552, 42.9999],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['WY']
    },
    // natural earth 110m admin_1 label point
    'state-ar': {
      id: 'state-ar',
      name: 'Arkansas',
      kind: 'region',
      lngLat: [-92.1428, 34.7563],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['AR']
    },
    // natural earth 110m admin_1 label point
    'state-ia': {
      id: 'state-ia',
      name: 'Iowa',
      kind: 'region',
      lngLat: [-93.3891, 42.0423],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['IA']
    },
    // natural earth 110m admin_1 label point
    'state-ks': {
      id: 'state-ks',
      name: 'Kansas',
      kind: 'region',
      lngLat: [-98.3309, 38.5],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['KS']
    },
    // natural earth 110m admin_1 label point
    'state-mo': {
      id: 'state-mo',
      name: 'Missouri',
      kind: 'region',
      lngLat: [-92.446, 38.5487],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['MO']
    },
    // natural earth 110m admin_1 label point
    'state-ne': {
      id: 'state-ne',
      name: 'Nebraska',
      kind: 'region',
      lngLat: [-99.6855, 41.5002],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['NE']
    },
    // natural earth 110m admin_1 label point
    'state-ok': {
      id: 'state-ok',
      name: 'Oklahoma',
      kind: 'region',
      lngLat: [-97.1309, 35.452],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['OK']
    },
    // natural earth 110m admin_1 label point
    'state-sd': {
      id: 'state-sd',
      name: 'South Dakota',
      kind: 'region',
      lngLat: [-100.255, 44.4711],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['SD']
    },
    // natural earth 110m admin_1 label point
    'state-la': {
      id: 'state-la',
      name: 'Louisiana',
      kind: 'region',
      lngLat: [-91.9991, 30.5274],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['LA']
    },
    // natural earth 110m admin_1 label point
    'state-tx': {
      id: 'state-tx',
      name: 'Texas',
      kind: 'region',
      lngLat: [-98.7607, 31.131],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['TX']
    },
    // natural earth 110m admin_1 label point
    'state-ct': {
      id: 'state-ct',
      name: 'Connecticut',
      kind: 'region',
      lngLat: [-72.7594, 41.6486],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['CT']
    },
    // natural earth 110m admin_1 label point
    'state-ma': {
      id: 'state-ma',
      name: 'Massachusetts',
      kind: 'region',
      lngLat: [-71.9993, 42.3739],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['MA']
    },
    // natural earth 110m admin_1 label point
    'state-nh': {
      id: 'state-nh',
      name: 'New Hampshire',
      kind: 'region',
      lngLat: [-71.6301, 43.5993],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['NH']
    },
    // natural earth 110m admin_1 label point
    'state-ri': {
      id: 'state-ri',
      name: 'Rhode Island',
      kind: 'region',
      lngLat: [-71.5082, 41.6242],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['RI']
    },
    // natural earth 110m admin_1 label point
    'state-vt': {
      id: 'state-vt',
      name: 'Vermont',
      kind: 'region',
      lngLat: [-72.7317, 44.0886],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['VT']
    },
    // natural earth 110m admin_1 label point
    'state-al': {
      id: 'state-al',
      name: 'Alabama',
      kind: 'region',
      lngLat: [-86.7184, 32.8551],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['AL']
    },
    // natural earth 110m admin_1 label point
    'state-fl': {
      id: 'state-fl',
      name: 'Florida',
      kind: 'region',
      lngLat: [-81.6228, 28.1568],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['FL']
    },
    // natural earth 110m admin_1 label point
    'state-ga': {
      id: 'state-ga',
      name: 'Georgia',
      kind: 'region',
      lngLat: [-83.4078, 32.8547],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['GA']
    },
    // natural earth 110m admin_1 label point
    'state-ms': {
      id: 'state-ms',
      name: 'Mississippi',
      kind: 'region',
      lngLat: [-89.7189, 32.8657],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['MS']
    },
    // natural earth 110m admin_1 label point
    'state-sc': {
      id: 'state-sc',
      name: 'South Carolina',
      kind: 'region',
      lngLat: [-80.6471, 33.8578],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['SC']
    },
    // natural earth 110m admin_1 label point
    'state-il': {
      id: 'state-il',
      name: 'Illinois',
      kind: 'region',
      lngLat: [-89.1991, 39.946],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['IL']
    },
    // natural earth 110m admin_1 label point
    'state-in': {
      id: 'state-in',
      name: 'Indiana',
      kind: 'region',
      lngLat: [-86.1396, 39.8874],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['IN']
    },
    // natural earth 110m admin_1 label point
    'state-ky': {
      id: 'state-ky',
      name: 'Kentucky',
      kind: 'region',
      lngLat: [-85.5729, 37.3994],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['KY']
    },
    // natural earth 110m admin_1 label point
    'state-nc': {
      id: 'state-nc',
      name: 'North Carolina',
      kind: 'region',
      lngLat: [-78.866, 35.6152],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['NC']
    },
    // natural earth 110m admin_1 label point
    'state-oh': {
      id: 'state-oh',
      name: 'Ohio',
      kind: 'region',
      lngLat: [-82.6719, 40.0924],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['OH']
    },
    // natural earth 110m admin_1 label point
    'state-tn': {
      id: 'state-tn',
      name: 'Tennessee',
      kind: 'region',
      lngLat: [-86.3415, 35.7514],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['TN']
    },
    // natural earth 110m admin_1 label point
    'state-va': {
      id: 'state-va',
      name: 'Virginia',
      kind: 'region',
      lngLat: [-78.2431, 37.7403],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['VA']
    },
    // natural earth 110m admin_1 label point
    'state-wi': {
      id: 'state-wi',
      name: 'Wisconsin',
      kind: 'region',
      lngLat: [-89.5831, 44.3709],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['WI']
    },
    // natural earth 110m admin_1 label point
    'state-wv': {
      id: 'state-wv',
      name: 'West Virginia',
      kind: 'region',
      lngLat: [-80.7128, 38.6422],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['WV']
    },
    // natural earth 110m admin_1 label point
    'state-de': {
      id: 'state-de',
      name: 'Delaware',
      kind: 'region',
      lngLat: [-75.4112, 38.8657],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['DE']
    },
    // natural earth 110m admin_1 label point
    'state-dc': {
      id: 'state-dc',
      name: 'District of Columbia',
      kind: 'region',
      lngLat: [-77.0113, 38.8922],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['DC']
    },
    // natural earth 110m admin_1 label point
    'state-md': {
      id: 'state-md',
      name: 'Maryland',
      kind: 'region',
      lngLat: [-77.0454, 39.3874],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['MD']
    },
    // natural earth 110m admin_1 label point
    'state-nj': {
      id: 'state-nj',
      name: 'New Jersey',
      kind: 'region',
      lngLat: [-74.4653, 40.0449],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['NJ']
    },
    // natural earth 110m admin_1 label point
    'state-ny': {
      id: 'state-ny',
      name: 'New York',
      kind: 'region',
      lngLat: [-75.3242, 43.1988],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['NY']
    },
    // natural earth 110m admin_1 label point
    'state-pa': {
      id: 'state-pa',
      name: 'Pennsylvania',
      kind: 'region',
      lngLat: [-77.6094, 40.8601],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['PA']
    },
    // natural earth 110m admin_1 label point
    'state-me': {
      id: 'state-me',
      name: 'Maine',
      kind: 'region',
      lngLat: [-69.1973, 45.148],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['ME']
    },
    // natural earth 110m admin_1 label point
    'state-mi': {
      id: 'state-mi',
      name: 'Michigan',
      kind: 'region',
      lngLat: [-84.9479, 43.4343],
      minZoom: 3,
      priority: 0,
      size: 'medium',
      aliases: ['MI']
    }
  }
};
