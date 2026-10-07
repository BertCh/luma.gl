import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'dixie-perimeter',
  title: 'Dixie Fire perimeter, 2021',
  description:
    'The final 2021 Dixie Fire perimeter (California, 963,405 acres), extracted from the NIFC perimeters of the wildfires dataset and simplified to 20 m.',
  license: 'Public domain (US Government work, NIFC)',
  attribution:
    'Fire perimeters: National Interagency Fire Center (NIFC) Open Data, public domain; packaged by poopdeck.gl',
  sourceUrl: 'https://data-nifc.opendata.arcgis.com/',
  approxBytes: 334166,
  bbox: [-121.5423, 39.8602, -120.1867, 40.7844]
} satisfies DatasetInfo;
