import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'sentinel-cloud-repair',
  title: 'Sentinel-2 cloud-repair pairs: Pilbara and Colorado Plateau',
  description:
    'Two natural-colour Sentinel-2 Level-2A crops (1536 × 1536; 25.6 km Pilbara, 20.5 km Colorado Plateau) pairing a cumulus-covered target with a clear observation two to five days older, plus a cloud and cloud-shadow mask from Sen2Cor cloud probability, projected dark-NIR shadows and multitemporal checks against the prior.',
  license: 'Copernicus Sentinel data legal notice; free, full and open access with attribution',
  attribution: 'Contains modified Copernicus Sentinel data (2026)',
  sourceUrl: 'https://element84.com/earth-search/',
  approxBytes: 17_900_000,
  bbox: [-110.48678363159458, -22.18301573358643, 118.48015030713725, 36.653154275573904]
} satisfies DatasetInfo;
