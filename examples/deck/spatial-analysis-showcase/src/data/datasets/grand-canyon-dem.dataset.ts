import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'grand-canyon-dem',
  title: 'Grand Canyon terrain (Bright Angel / Phantom Ranch)',
  description:
    'Terrarium DEM of the central Grand Canyon at about 15.5 m ground resolution (2048 x 2048, Web Mercator): rims, side canyons and the Colorado River gorge.',
  license:
    'AWS Open Data Terrain Tiles (public bucket); underlying USGS 3DEP/NED is public domain, other sources require attribution per Tilezen list',
  attribution:
    'Terrain Tiles on AWS (Mapzen/Tilezen): USGS 3DEP/NED, SRTM and other sources; see https://github.com/tilezen/joerd/blob/master/docs/attribution.md',
  sourceUrl: 'https://registry.opendata.aws/terrain-tiles/',
  approxBytes: 3252119,
  bbox: [-112.2758, 35.9579, -111.9242, 36.2419]
} satisfies DatasetInfo;
