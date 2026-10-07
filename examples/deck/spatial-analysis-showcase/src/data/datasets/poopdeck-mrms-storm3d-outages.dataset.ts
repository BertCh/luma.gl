import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'poopdeck-mrms-storm3d-outages',
  title: 'County power-outage snapshots, 21-22 May 2024',
  description:
    '40,291 county snapshots (15-minute steps, 2,111 counties) of customers without power, from 12:00 UTC on 21 May to 06:00 UTC on 22 May 2024. A table keyed by county FIPS; county geometry comes from the us-counties dataset.',
  license: 'CC BY 4.0',
  attribution:
    'DOE / Oak Ridge National Laboratory EAGLE-I power outage data (CC BY 4.0), via poopdeck.gl',
  sourceUrl: 'https://figshare.com/articles/dataset/24237376',
  approxBytes: 490_000,
  bbox: [-124.7258, 24.4981, -66.9499, 49.3844]
} satisfies DatasetInfo;
