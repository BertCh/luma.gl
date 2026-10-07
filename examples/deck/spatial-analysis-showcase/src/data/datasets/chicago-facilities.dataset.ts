import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'chicago-facilities',
  title: 'Chicago public facilities',
  description:
    'Hospitals, public libraries, CPS schools and fire stations for location-allocation and coverage analysis.',
  license:
    'City of Chicago Data Portal Terms of Use; hospitals from Overture Maps (CDLA-Permissive-2.0)',
  attribution:
    'City of Chicago, Chicago Public Library, Chicago Public Schools, Chicago Fire Department; Overture Maps Foundation',
  sourceUrl: 'https://data.cityofchicago.org/',
  approxBytes: 37385,
  bbox: [-87.841, 41.653, -87.528, 42.021]
} satisfies DatasetInfo;
