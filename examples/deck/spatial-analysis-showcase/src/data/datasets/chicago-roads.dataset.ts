import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'chicago-roads',
  title: 'Chicago street network (OSM, drivable)',
  description:
    'Simplified OpenStreetMap drive network of the City of Chicago: directed edges with class, speed, length, travel time and geometry; largest strongly connected component.',
  license: 'ODbL 1.0',
  attribution: '© OpenStreetMap contributors',
  sourceUrl: 'https://www.openstreetmap.org/copyright',
  approxBytes: 3_854_226,
  bbox: [-87.8539, 41.64461, -87.52458, 42.02264]
} satisfies DatasetInfo;
