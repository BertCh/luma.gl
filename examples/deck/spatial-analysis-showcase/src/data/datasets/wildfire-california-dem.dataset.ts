import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'wildfire-california-dem',
  title: 'Northern and central California DEM, 239 m (Terrarium z9)',
  description:
    'A 1166 x 2148 Web Mercator elevation grid (about 239 m on the ground) from the key-free AWS Terrain Tiles, covering the Coast Ranges, Sierra foothills and the 2020 and 2021 complex fires. Ocean and water at or below 0 m are noData; 37 isolated spikes are median-repaired.',
  license: 'Terrain Tiles on AWS: public domain and open source data, attribution required',
  attribution:
    'Terrain Tiles on AWS (Mapzen, Tilezen): USGS NED, SRTM, GMTED2010, ETOPO1 and others; see https://github.com/tilezen/joerd/blob/master/docs/attribution.md',
  sourceUrl: 'https://registry.opendata.aws/terrain-tiles/',
  approxBytes: 2_380_000,
  bbox: [-123.302, 36.399, -120.1, 41.001]
} satisfies DatasetInfo;
