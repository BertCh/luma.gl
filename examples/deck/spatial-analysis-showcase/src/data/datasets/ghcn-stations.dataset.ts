import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'ghcn-stations',
  title: 'GHCN-Daily rain and temperature, Hurricane Helene (SE US)',
  description:
    '6,135 weather stations with 2024-09-27 precipitation (and previous day, TMAX where reported) plus elevation for the southeastern US.',
  license: 'NOAA NCEI GHCN-Daily, public domain',
  attribution: 'NOAA National Centers for Environmental Information, GHCN-Daily',
  sourceUrl: 'https://registry.opendata.aws/noaa-ghcn/',
  approxBytes: 370044,
  bbox: [-92.0, 24.0, -74.0, 40.0]
} satisfies DatasetInfo;
