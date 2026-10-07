import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'swissimage-zermatt',
  title: 'Zermatt orthophotos, 2018 and 2024 (SWISSIMAGE)',
  description:
    'A 424 m natural-colour aerial view of central Zermatt from SWISSIMAGE Journey through time, sampled to a 512 × 512 browser-ready grid from Web Mercator zoom 18 tiles.',
  license: 'swisstopo Open Government Data (OGD), free use with attribution',
  attribution: '© swisstopo',
  sourceUrl: 'https://www.swisstopo.admin.ch/en/orthoimage-swissimage-10',
  approxBytes: 1_200_000,
  bbox: [7.743988037109375, 46.018899738911934, 7.749481201171875, 46.02271417608516]
} satisfies DatasetInfo;
