import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'alps-dem',
  title: 'Matterhorn and Zermatt terrain (Swiss Alps)',
  description:
    'Terrarium-encoded DEM of the Matterhorn, Gornergrat and Gorner Glacier at 6.6 m ground resolution (2048 x 2048, Web Mercator, 0.25 m quantisation).',
  license:
    'Mapterhorn terrain tiles built from swisstopo swissALTI3D (OGD, free use with attribution) and Copernicus GLO-30',
  attribution: '(c) swisstopo (swissALTI3D), (c) Mapterhorn; contains modified Copernicus data',
  sourceUrl: 'https://mapterhorn.com/',
  approxBytes: 3377909,
  bbox: [7.6541, 45.9239, 7.8299, 46.046]
} satisfies DatasetInfo;
