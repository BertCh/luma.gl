import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'poopdeck-wildfires',
  title: 'Western US wildfire perimeters, 2020-2023 (poopdeck.gl archive)',
  description:
    '90 large western US fires (421 polygon parts, 341k vertices) from the poopdeck.gl `wildfires` archive of NIFC interagency perimeters: acres, name, year, perimeter date and an acreage class. Not the complete NIFC record: some well-known fires (August Complex, Caldor) are absent.',
  license: 'Public domain (US Government work, NIFC)',
  attribution:
    'Fire perimeters: National Interagency Fire Center (NIFC) Open Data, public domain; packaged by poopdeck.gl',
  sourceUrl: 'https://data-nifc.opendata.arcgis.com/',
  approxBytes: 2_732_000,
  bbox: [-123.6835, 31.5995, -107.5267, 47.9522]
} satisfies DatasetInfo;
