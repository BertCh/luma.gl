import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'ndvi-timeseries',
  title: 'Greenville NDVI time series 2017-2024 (Sentinel-2)',
  description:
    '16 cloud-screened summer NDVI snapshots (uint8, 100 m, 256 x 256) across the 2021 Dixie Fire burn scar and its recovery.',
  license: 'Copernicus Sentinel data: free, full and open',
  attribution: 'Contains modified Copernicus Sentinel data 2017-2024',
  sourceUrl: 'https://registry.opendata.aws/sentinel-2-l2a-cogs/',
  approxBytes: 1053282,
  bbox: [-121.1534, 40.0568, -120.8462, 40.2822]
} satisfies DatasetInfo;
