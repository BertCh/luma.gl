import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'gfs-wind',
  title: 'GFS wind during Hurricane Helene landfall (10 m and 250 hPa)',
  description:
    'Hourly 10 m and 3-hourly 250 hPa u/v wind from the 2024-09-26 12Z NOAA GFS run on a 0.25 degree grid over the Gulf, Southeast US and Atlantic, quantised into RG PNGs.',
  license: 'US Government work (NOAA), public domain',
  attribution: 'NOAA / NCEP Global Forecast System',
  sourceUrl: 'https://registry.opendata.aws/noaa-gfs-bdp-pds/',
  approxBytes: 877108,
  bbox: [-105.0, 12.0, -60.0, 50.0]
} satisfies DatasetInfo;
