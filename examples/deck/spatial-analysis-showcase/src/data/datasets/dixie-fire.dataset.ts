import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'dixie-fire',
  title: 'Dixie Fire before/after, Greenville CA (Sentinel-2, WorldCover, DEM)',
  description:
    '15 km window at 20 m on one UTM 10N grid: Sentinel-2 L2A red/NIR/SWIR22 reflectance and scene classification for 2021-07-13 and 2021-09-21, ESA WorldCover 2021 classes and a Terrarium DEM.',
  license:
    'Copernicus Sentinel data: free, full and open; ESA WorldCover: CC BY 4.0; terrain: AWS Terrain Tiles',
  attribution:
    'Contains modified Copernicus Sentinel data 2021; (c) ESA WorldCover 2021 (CC BY 4.0); terrain from AWS Terrain Tiles / USGS 3DEP',
  sourceUrl: 'https://registry.opendata.aws/sentinel-2-l2a-cogs/',
  approxBytes: 7256791,
  bbox: [-121.0899, 40.1039, -120.9099, 40.2359]
} satisfies DatasetInfo;
