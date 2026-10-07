import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'alps-context',
  title: 'Zermatt glaciers, peaks and places (OpenStreetMap)',
  description:
    'Glaciers (polygons), peaks, saddles, villages, stations, huts, lakes and the Gornergrat railway around Zermatt, from OpenStreetMap, as GeoArrow glaciers plus GeoJSON side files.',
  license: 'ODbL 1.0 (Open Database License)',
  attribution: '© OpenStreetMap contributors (ODbL)',
  sourceUrl: 'https://www.openstreetmap.org/copyright',
  approxBytes: 483092,
  bbox: [7.58, 45.86, 7.98, 46.12]
} satisfies DatasetInfo;
