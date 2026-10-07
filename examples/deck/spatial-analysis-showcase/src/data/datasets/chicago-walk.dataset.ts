import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'chicago-walk',
  title: 'Chicago pedestrian network, Loop (OSM)',
  description:
    'OpenStreetMap sidewalks, crossings and paths of the Loop and near neighbourhoods (about 5 x 5 km), both directions, for walk isochrones and network statistics.',
  license: 'ODbL 1.0',
  attribution: '© OpenStreetMap contributors',
  sourceUrl: 'https://www.openstreetmap.org/copyright',
  approxBytes: 2_844_291,
  bbox: [-87.6669, 41.85619, -87.59963, 41.90362]
} satisfies DatasetInfo;
