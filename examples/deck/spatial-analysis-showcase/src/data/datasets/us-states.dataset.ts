import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'us-states',
  title: 'US state outlines',
  description:
    'Contiguous state outlines dissolved from the county geometry, for context overlays.',
  license: 'US Government work (US Census Bureau): public domain',
  attribution: 'US Census Bureau cartographic boundaries 2022',
  sourceUrl:
    'https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.html',
  approxBytes: 230000,
  bbox: [-124.7258, 24.4981, -66.9499, 49.3844]
} satisfies DatasetInfo;
