import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'poopdeck-goes-glm-lightning',
  title: 'GOES-16 lightning flashes, 21-22 May 2024',
  description:
    'A seeded 43% sample (150,000 of 346,450) of the lightning flashes the GOES-16 Geostationary Lightning Mapper saw over the US between 12:00 UTC on 21 May and 06:00 UTC on 22 May 2024, with flash time, optical energy and footprint area.',
  license: 'Public domain (US Government work)',
  attribution: 'NOAA GOES-R GLM Level 2 (LCFA), via poopdeck.gl',
  sourceUrl: 'https://registry.opendata.aws/noaa-goes/',
  approxBytes: 3_000_000,
  bbox: [-112.7012, 24.0002, -66.0016, 49.9973]
} satisfies DatasetInfo;
