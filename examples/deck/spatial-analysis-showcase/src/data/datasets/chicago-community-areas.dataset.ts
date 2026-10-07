import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'chicago-community-areas',
  title: 'Chicago community areas',
  description: 'The 77 official Chicago community areas with names.',
  license: 'City of Chicago Data Portal Terms of Use',
  attribution: 'City of Chicago',
  sourceUrl:
    'https://data.cityofchicago.org/Facilities-Geographic-Boundaries/Boundaries-Community-Areas/igwz-8jzy',
  approxBytes: 419528,
  bbox: [-87.94, 41.645, -87.524, 42.023]
} satisfies DatasetInfo;
