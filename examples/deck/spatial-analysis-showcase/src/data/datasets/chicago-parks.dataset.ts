import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'chicago-parks',
  title: 'Chicago parks, forest preserves and woodland (OpenStreetMap)',
  description:
    '1,139 parks, nature reserves, forest preserves / protected areas and woodland in and around Chicago (160 km2 of green space) with name, kind and area, plus one dissolved green-space mask.',
  license: 'ODbL 1.0 (Open Database License)',
  attribution: '© OpenStreetMap contributors (ODbL)',
  sourceUrl: 'https://www.openstreetmap.org/copyright',
  approxBytes: 973504,
  bbox: [-87.94, 41.64, -87.52, 42.02]
} satisfies DatasetInfo;
