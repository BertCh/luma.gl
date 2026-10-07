// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Gazetteer} from './types';

/**
 * Oceans, seas, currents, continents, circles of latitude and major hubs.
 * Every coordinate is `[longitude, latitude]` and carries its source in the comment above it.
 */
export const WORLD: Gazetteer = {
  id: 'world',
  name: 'World',
  places: {
    // label placement
    'atlantic-ocean': {
      id: 'atlantic-ocean',
      name: 'Atlantic Ocean',
      kind: 'ocean',
      lngLat: [-32.0, 10.0],
      minZoom: 0,
      priority: 3,
      size: 'large'
    },
    // label placement
    'pacific-ocean': {
      id: 'pacific-ocean',
      name: 'Pacific Ocean',
      kind: 'ocean',
      lngLat: [-150.0, 10.0],
      minZoom: 0,
      priority: 3,
      size: 'large'
    },
    // wiki
    'indian-ocean': {
      id: 'indian-ocean',
      name: 'Indian Ocean',
      kind: 'ocean',
      lngLat: [80.0, -20.0],
      minZoom: 0,
      priority: 3,
      size: 'large'
    },
    // label placement
    'arctic-ocean': {
      id: 'arctic-ocean',
      name: 'Arctic Ocean',
      kind: 'ocean',
      lngLat: [-150.0, 80.0],
      minZoom: 0,
      priority: 2,
      size: 'large'
    },
    // label placement
    'southern-ocean': {
      id: 'southern-ocean',
      name: 'Southern Ocean',
      kind: 'ocean',
      lngLat: [80.0, -63.0],
      minZoom: 0,
      priority: 3,
      size: 'large'
    },
    // wiki
    'gulf-of-mexico': {
      id: 'gulf-of-mexico',
      name: 'Gulf of Mexico',
      kind: 'water',
      lngLat: [-90.0, 25.0],
      minZoom: 2,
      priority: 1,
      size: 'large'
    },
    // wiki
    'caribbean-sea': {
      id: 'caribbean-sea',
      name: 'Caribbean Sea',
      kind: 'water',
      lngLat: [-75.0, 15.0],
      minZoom: 2,
      priority: 1,
      size: 'large'
    },
    // wiki
    'north-sea': {
      id: 'north-sea',
      name: 'North Sea',
      kind: 'water',
      lngLat: [3.0, 56.0],
      minZoom: 2,
      priority: 1,
      size: 'large'
    },
    // wiki
    'mediterranean-sea': {
      id: 'mediterranean-sea',
      name: 'Mediterranean Sea',
      kind: 'water',
      lngLat: [18.0, 35.0],
      minZoom: 2,
      priority: 1,
      size: 'large'
    },
    // wiki
    'sargasso-sea': {
      id: 'sargasso-sea',
      name: 'Sargasso Sea',
      kind: 'water',
      lngLat: [-66.0, 28.0],
      minZoom: 2,
      priority: 1,
      size: 'large'
    },
    // wiki
    'banda-sea': {
      id: 'banda-sea',
      name: 'Banda Sea',
      kind: 'water',
      lngLat: [127.0, -6.0],
      minZoom: 2,
      priority: 1,
      size: 'large'
    },
    // label placement, pole of the Natural Earth marine polygon
    'wadden-sea': {
      id: 'wadden-sea',
      name: 'Wadden Sea',
      kind: 'water',
      lngLat: [5.11, 53.15],
      minZoom: 2,
      priority: 1,
      size: 'large'
    },
    // label placement
    'labrador-sea': {
      id: 'labrador-sea',
      name: 'Labrador Sea',
      kind: 'water',
      lngLat: [-53.0, 58.0],
      minZoom: 2,
      priority: 1,
      size: 'large'
    },
    // label placement
    'norwegian-sea': {
      id: 'norwegian-sea',
      name: 'Norwegian Sea',
      kind: 'water',
      lngLat: [3.0, 66.0],
      minZoom: 2,
      priority: 1,
      size: 'large'
    },
    // label placement, pole of the Natural Earth marine polygon
    'baltic-sea': {
      id: 'baltic-sea',
      name: 'Baltic Sea',
      kind: 'water',
      lngLat: [18.98, 55.89],
      minZoom: 2,
      priority: 1,
      size: 'large'
    },
    // label placement
    'arabian-sea': {
      id: 'arabian-sea',
      name: 'Arabian Sea',
      kind: 'water',
      lngLat: [65.0, 16.0],
      minZoom: 2,
      priority: 1,
      size: 'large'
    },
    // label placement
    'bay-of-bengal': {
      id: 'bay-of-bengal',
      name: 'Bay of Bengal',
      kind: 'water',
      lngLat: [88.0, 15.0],
      minZoom: 2,
      priority: 1,
      size: 'large'
    },
    // label placement
    'south-china-sea': {
      id: 'south-china-sea',
      name: 'South China Sea',
      kind: 'water',
      lngLat: [114.0, 12.0],
      minZoom: 2,
      priority: 1,
      size: 'large'
    },
    // label placement
    'sea-of-japan': {
      id: 'sea-of-japan',
      name: 'Sea of Japan',
      kind: 'water',
      lngLat: [134.0, 40.0],
      minZoom: 2,
      priority: 1,
      size: 'large'
    },
    // label placement
    'philippine-sea': {
      id: 'philippine-sea',
      name: 'Philippine Sea',
      kind: 'water',
      lngLat: [132.0, 20.0],
      minZoom: 2,
      priority: 1,
      size: 'large'
    },
    // label placement
    'tasman-sea': {
      id: 'tasman-sea',
      name: 'Tasman Sea',
      kind: 'water',
      lngLat: [160.0, -38.0],
      minZoom: 2,
      priority: 1,
      size: 'large'
    },
    // label placement
    'bering-sea': {
      id: 'bering-sea',
      name: 'Bering Sea',
      kind: 'water',
      lngLat: [-178.0, 59.0],
      minZoom: 2,
      priority: 1,
      size: 'large'
    },
    // label placement, pole of the Natural Earth marine polygon
    'gulf-of-alaska': {
      id: 'gulf-of-alaska',
      name: 'Gulf of Alaska',
      kind: 'water',
      lngLat: [-148.47, 58.11],
      minZoom: 2,
      priority: 1,
      size: 'large'
    },
    // label placement
    'bay-of-biscay': {
      id: 'bay-of-biscay',
      name: 'Bay of Biscay',
      kind: 'water',
      lngLat: [-4.0, 45.0],
      minZoom: 2,
      priority: 1,
      size: 'large'
    },
    // label placement
    'red-sea': {
      id: 'red-sea',
      name: 'Red Sea',
      kind: 'water',
      lngLat: [38.0, 20.0],
      minZoom: 2,
      priority: 1,
      size: 'large'
    },
    // label placement
    'black-sea': {
      id: 'black-sea',
      name: 'Black Sea',
      kind: 'water',
      lngLat: [34.0, 43.5],
      minZoom: 2,
      priority: 1,
      size: 'large'
    },
    // label placement
    'hudson-bay': {
      id: 'hudson-bay',
      name: 'Hudson Bay',
      kind: 'water',
      lngLat: [-85.0, 60.0],
      minZoom: 2,
      priority: 1,
      size: 'large'
    },
    // label placement on the mean path, approximate
    'gulf-stream': {
      id: 'gulf-stream',
      name: 'Gulf Stream',
      kind: 'current',
      lngLat: [-72.0, 37.5],
      minZoom: 2,
      priority: 1,
      note: 'label placed on the mean path off Cape Hatteras (35.2 N, 75.5 W, Wikipedia) to the northeast (about 100 km)'
    },
    // label placement on the mean path, approximate
    kuroshio: {
      id: 'kuroshio',
      name: 'Kuroshio',
      kind: 'current',
      lngLat: [136.0, 32.7],
      minZoom: 2,
      priority: 1,
      note: 'label placed on the mean path south of Honshu (about 100 km)'
    },
    // label placement on the mean path, approximate
    'north-atlantic-current': {
      id: 'north-atlantic-current',
      name: 'North Atlantic Current',
      kind: 'current',
      lngLat: [-40.0, 47.0],
      minZoom: 2,
      priority: 1,
      note: 'label placed on the mean path mid-Atlantic, continuation of the Gulf Stream (about 100 km)'
    },
    // label placement on the mean path, approximate
    'labrador-current': {
      id: 'labrador-current',
      name: 'Labrador Current',
      kind: 'current',
      lngLat: [-53.5, 52.5],
      minZoom: 2,
      priority: 1,
      note: 'label placed on the mean path off the Labrador coast (about 100 km)'
    },
    // label placement on the mean path, approximate
    'canary-current': {
      id: 'canary-current',
      name: 'Canary Current',
      kind: 'current',
      lngLat: [-18.5, 27.0],
      minZoom: 2,
      priority: 1,
      note: 'label placed on the mean path off northwest Africa (about 100 km)'
    },
    // label placement on the mean path, approximate
    'california-current': {
      id: 'california-current',
      name: 'California Current',
      kind: 'current',
      lngLat: [-126.0, 36.0],
      minZoom: 2,
      priority: 1,
      note: 'label placed on the mean path off California (about 100 km)'
    },
    // label placement on the mean path, approximate
    'humboldt-current': {
      id: 'humboldt-current',
      name: 'Humboldt Current',
      kind: 'current',
      lngLat: [-77.0, -25.0],
      minZoom: 2,
      priority: 1,
      note: 'label placed on the mean path off Chile (about 100 km)'
    },
    // label placement on the mean path, approximate
    'benguela-current': {
      id: 'benguela-current',
      name: 'Benguela Current',
      kind: 'current',
      lngLat: [11.0, -25.0],
      minZoom: 2,
      priority: 1,
      note: 'label placed on the mean path off Namibia (about 100 km)'
    },
    // label placement on the mean path, approximate
    'agulhas-current': {
      id: 'agulhas-current',
      name: 'Agulhas Current',
      kind: 'current',
      lngLat: [31.5, -31.5],
      minZoom: 2,
      priority: 1,
      note: 'label placed on the mean path off southeast Africa (about 100 km)'
    },
    // label placement on the mean path, approximate
    'brazil-current': {
      id: 'brazil-current',
      name: 'Brazil Current',
      kind: 'current',
      lngLat: [-40.0, -24.0],
      minZoom: 2,
      priority: 1,
      note: 'label placed on the mean path off southeast Brazil (about 100 km)'
    },
    // label placement on the mean path, approximate
    'antarctic-circumpolar-current': {
      id: 'antarctic-circumpolar-current',
      name: 'Antarctic Circumpolar Current',
      kind: 'current',
      lngLat: [-65.0, -58.0],
      minZoom: 2,
      priority: 1,
      note: 'label placed on the mean path Drake Passage (about 100 km)'
    },
    // label placement
    'north-america': {
      id: 'north-america',
      name: 'North America',
      kind: 'region',
      lngLat: [-100.0, 46.0],
      minZoom: 0,
      priority: 2,
      size: 'large'
    },
    // label placement
    'south-america': {
      id: 'south-america',
      name: 'South America',
      kind: 'region',
      lngLat: [-60.0, -12.0],
      minZoom: 0,
      priority: 2,
      size: 'large'
    },
    // label placement
    europe: {
      id: 'europe',
      name: 'Europe',
      kind: 'region',
      lngLat: [15.0, 50.0],
      minZoom: 0,
      priority: 2,
      size: 'large'
    },
    // label placement
    africa: {
      id: 'africa',
      name: 'Africa',
      kind: 'region',
      lngLat: [20.0, 6.0],
      minZoom: 0,
      priority: 2,
      size: 'large'
    },
    // label placement
    asia: {
      id: 'asia',
      name: 'Asia',
      kind: 'region',
      lngLat: [90.0, 46.0],
      minZoom: 0,
      priority: 2,
      size: 'large'
    },
    // label placement
    australia: {
      id: 'australia',
      name: 'Australia',
      kind: 'region',
      lngLat: [134.0, -25.0],
      minZoom: 0,
      priority: 2,
      size: 'large'
    },
    // label placement
    antarctica: {
      id: 'antarctica',
      name: 'Antarctica',
      kind: 'region',
      lngLat: [60.0, -80.0],
      minZoom: 0,
      priority: 2,
      size: 'large'
    },
    // definition
    equator: {
      id: 'equator',
      name: 'Equator',
      kind: 'region',
      lngLat: [-140.0, 0.0],
      minZoom: 0,
      priority: 0,
      size: 'small'
    },
    // wikipedia 23 26 08 N, 2026
    'tropic-of-cancer': {
      id: 'tropic-of-cancer',
      name: 'Tropic of Cancer',
      kind: 'region',
      lngLat: [-150.0, 23.4356],
      minZoom: 0,
      priority: 0,
      size: 'small'
    },
    // wikipedia 23 26 08 S
    'tropic-of-capricorn': {
      id: 'tropic-of-capricorn',
      name: 'Tropic of Capricorn',
      kind: 'region',
      lngLat: [-150.0, -23.4356],
      minZoom: 0,
      priority: 0,
      size: 'small'
    },
    // wikipedia 66 33 51 N
    'arctic-circle': {
      id: 'arctic-circle',
      name: 'Arctic Circle',
      kind: 'region',
      lngLat: [-170.0, 66.5642],
      minZoom: 0,
      priority: 0,
      size: 'small'
    },
    // wikipedia 66 33 51 S
    'antarctic-circle': {
      id: 'antarctic-circle',
      name: 'Antarctic Circle',
      kind: 'region',
      lngLat: [-150.0, -66.5642],
      minZoom: 0,
      priority: 0,
      size: 'small'
    },
    // wiki
    lhr: {
      id: 'lhr',
      name: 'LHR',
      kind: 'airport',
      lngLat: [-0.46139, 51.4775],
      minZoom: 2,
      priority: 1,
      aliases: ['London', 'Heathrow Airport']
    },
    // wiki
    cdg: {
      id: 'cdg',
      name: 'CDG',
      kind: 'airport',
      lngLat: [2.54778, 49.00972],
      minZoom: 2,
      priority: 1,
      aliases: ['Paris', 'Paris Charles de Gaulle Airport']
    },
    // wiki
    ams: {
      id: 'ams',
      name: 'AMS',
      kind: 'airport',
      lngLat: [4.765, 52.3],
      minZoom: 2,
      priority: 1,
      aliases: ['Amsterdam', 'Amsterdam Airport Schiphol']
    },
    // wiki
    fra: {
      id: 'fra',
      name: 'FRA',
      kind: 'airport',
      lngLat: [8.57056, 50.03333],
      minZoom: 2,
      priority: 1,
      aliases: ['Frankfurt', 'Frankfurt Airport']
    },
    // wiki
    ist: {
      id: 'ist',
      name: 'IST',
      kind: 'airport',
      lngLat: [28.72778, 41.26222],
      minZoom: 2,
      priority: 1,
      aliases: ['Istanbul', 'Istanbul Airport']
    },
    // wiki
    dxb: {
      id: 'dxb',
      name: 'DXB',
      kind: 'airport',
      lngLat: [55.36444, 25.25278],
      minZoom: 2,
      priority: 1,
      aliases: ['Dubai', 'Dubai International Airport']
    },
    // wiki
    doh: {
      id: 'doh',
      name: 'DOH',
      kind: 'airport',
      lngLat: [51.60806, 25.27306],
      minZoom: 2,
      priority: 1,
      aliases: ['Doha', 'Hamad International Airport']
    },
    // wiki
    hnd: {
      id: 'hnd',
      name: 'HND',
      kind: 'airport',
      lngLat: [139.78111, 35.55333],
      minZoom: 2,
      priority: 1,
      aliases: ['Tokyo', 'Haneda Airport']
    },
    // wiki
    pek: {
      id: 'pek',
      name: 'PEK',
      kind: 'airport',
      lngLat: [116.5975, 40.0725],
      minZoom: 2,
      priority: 1,
      aliases: ['Beijing', 'Beijing Capital International Airport']
    },
    // wiki
    pvg: {
      id: 'pvg',
      name: 'PVG',
      kind: 'airport',
      lngLat: [121.80528, 31.14333],
      minZoom: 2,
      priority: 1,
      aliases: ['Shanghai', 'Shanghai Pudong International Airport']
    },
    // wiki
    sin: {
      id: 'sin',
      name: 'SIN',
      kind: 'airport',
      lngLat: [103.98944, 1.35917],
      minZoom: 2,
      priority: 1,
      aliases: ['Singapore', 'Changi Airport']
    },
    // wiki
    hkg: {
      id: 'hkg',
      name: 'HKG',
      kind: 'airport',
      lngLat: [113.91444, 22.30889],
      minZoom: 2,
      priority: 1,
      aliases: ['Hong Kong', 'Hong Kong International Airport']
    },
    // wiki
    icn: {
      id: 'icn',
      name: 'ICN',
      kind: 'airport',
      lngLat: [126.44, 37.46333],
      minZoom: 2,
      priority: 1,
      aliases: ['Seoul', 'Incheon International Airport']
    },
    // wiki
    bkk: {
      id: 'bkk',
      name: 'BKK',
      kind: 'airport',
      lngLat: [100.75, 13.6925],
      minZoom: 2,
      priority: 1,
      aliases: ['Bangkok', 'Suvarnabhumi Airport']
    },
    // wiki
    del: {
      id: 'del',
      name: 'DEL',
      kind: 'airport',
      lngLat: [77.11222, 28.56861],
      minZoom: 2,
      priority: 1,
      aliases: ['Delhi', 'Indira Gandhi International Airport']
    },
    // wiki
    syd: {
      id: 'syd',
      name: 'SYD',
      kind: 'airport',
      lngLat: [151.17722, -33.94611],
      minZoom: 2,
      priority: 1,
      aliases: ['Sydney', 'Sydney Airport']
    },
    // wiki
    akl: {
      id: 'akl',
      name: 'AKL',
      kind: 'airport',
      lngLat: [174.79167, -37.00806],
      minZoom: 2,
      priority: 1,
      aliases: ['Auckland', 'Auckland Airport']
    },
    // wiki
    jnb: {
      id: 'jnb',
      name: 'JNB',
      kind: 'airport',
      lngLat: [28.25, -26.13333],
      minZoom: 2,
      priority: 1,
      aliases: ['Johannesburg', 'O. R. Tambo International Airport']
    },
    // wiki
    cai: {
      id: 'cai',
      name: 'CAI',
      kind: 'airport',
      lngLat: [31.40556, 30.12194],
      minZoom: 2,
      priority: 1,
      aliases: ['Cairo', 'Cairo International Airport']
    },
    // wiki
    nbo: {
      id: 'nbo',
      name: 'NBO',
      kind: 'airport',
      lngLat: [36.92583, -1.31861],
      minZoom: 2,
      priority: 1,
      aliases: ['Nairobi', 'Jomo Kenyatta International Airport']
    },
    // wiki
    gru: {
      id: 'gru',
      name: 'GRU',
      kind: 'airport',
      lngLat: [-46.47306, -23.43556],
      minZoom: 2,
      priority: 1,
      aliases: ['São Paulo', 'São Paulo/Guarulhos International Airport']
    },
    // wiki
    mex: {
      id: 'mex',
      name: 'MEX',
      kind: 'airport',
      lngLat: [-99.07194, 19.43611],
      minZoom: 2,
      priority: 1,
      aliases: ['Mexico City', 'Mexico City International Airport']
    },
    // wiki
    yyz: {
      id: 'yyz',
      name: 'YYZ',
      kind: 'airport',
      lngLat: [-79.63056, 43.67611],
      minZoom: 2,
      priority: 1,
      aliases: ['Toronto', 'Toronto Pearson International Airport']
    },
    // wiki
    atl: {
      id: 'atl',
      name: 'ATL',
      kind: 'airport',
      lngLat: [-84.42806, 33.63667],
      minZoom: 2,
      priority: 1,
      aliases: ['Atlanta', 'Hartsfield–Jackson Atlanta International Airport']
    },
    // wiki
    ord: {
      id: 'ord',
      name: 'ORD',
      kind: 'airport',
      lngLat: [-87.90472, 41.97861],
      minZoom: 2,
      priority: 1,
      aliases: ['Chicago', "O'Hare International Airport"]
    },
    // wiki
    jfk: {
      id: 'jfk',
      name: 'JFK',
      kind: 'airport',
      lngLat: [-73.77889, 40.63972],
      minZoom: 2,
      priority: 1,
      aliases: ['New York', 'John F. Kennedy International Airport']
    },
    // wiki
    lax: {
      id: 'lax',
      name: 'LAX',
      kind: 'airport',
      lngLat: [-118.40806, 33.9425],
      minZoom: 2,
      priority: 1,
      aliases: ['Los Angeles', 'Los Angeles International Airport']
    },
    // wiki
    dfw: {
      id: 'dfw',
      name: 'DFW',
      kind: 'airport',
      lngLat: [-97.03806, 32.89694],
      minZoom: 2,
      priority: 1,
      aliases: ['Dallas-Fort Worth', 'Dallas Fort Worth International Airport']
    },
    // wiki
    'strait-of-gibraltar': {
      id: 'strait-of-gibraltar',
      name: 'Strait of Gibraltar',
      kind: 'water',
      lngLat: [-5.5, 35.95],
      minZoom: 4,
      priority: 0
    },
    // wiki
    'panama-canal': {
      id: 'panama-canal',
      name: 'Panama Canal',
      kind: 'water',
      lngLat: [-79.75, 9.12],
      minZoom: 4,
      priority: 0
    },
    // wiki
    'suez-canal': {
      id: 'suez-canal',
      name: 'Suez Canal',
      kind: 'water',
      lngLat: [32.34417, 30.705],
      minZoom: 4,
      priority: 0
    }
  }
};
