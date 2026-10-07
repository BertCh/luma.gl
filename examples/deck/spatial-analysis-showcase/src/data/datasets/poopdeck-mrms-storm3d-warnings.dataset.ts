import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'poopdeck-mrms-storm3d-warnings',
  title: 'NWS tornado, severe thunderstorm and flash flood warning polygons, 21-22 May 2024',
  description:
    '750 storm-based warning polygon versions (284 warnings) with the time each version became valid, when it was replaced or expired, and when the warning was first issued. Phenomena: tornado, severe thunderstorm, flash flood.',
  license: 'Public domain (US Government work)',
  attribution:
    'NOAA National Weather Service warnings via the Iowa Environmental Mesonet, via poopdeck.gl',
  sourceUrl: 'https://mesonet.agron.iastate.edu/',
  approxBytes: 80_000,
  bbox: [-99.57, 25.44, -71.17, 45.9]
} satisfies DatasetInfo;
