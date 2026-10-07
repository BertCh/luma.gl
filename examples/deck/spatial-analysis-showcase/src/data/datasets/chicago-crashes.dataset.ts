import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'chicago-crashes',
  title: 'Chicago traffic crashes, 2023, snapped to streets',
  description:
    'About 110,000 reported traffic crashes in 2023 with time, injury severity and a precomputed snap to the chicago-roads network (edge index and offset).',
  license: 'City of Chicago Data Portal Terms of Use',
  attribution: 'City of Chicago / Chicago Police Department',
  sourceUrl: 'https://data.cityofchicago.org/resource/85ca-t3if',
  approxBytes: 2_636_095,
  bbox: [-87.93399, 41.64467, -87.52459, 42.02275]
} satisfies DatasetInfo;
