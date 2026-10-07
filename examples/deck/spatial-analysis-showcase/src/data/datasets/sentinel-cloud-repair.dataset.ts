import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'sentinel-cloud-repair',
  title: 'Sentinel-2 cloud-repair pairs: Oʻahu and Venice',
  description:
    'Two 5.12 km natural-colour Sentinel-2 Level-2A crops pairing a cloudy target with an aligned older clear observation and the target Scene Classification Layer. Oʻahu spans eight months; Venice spans one day.',
  license: 'Copernicus Sentinel data legal notice; free, full and open access with attribution',
  attribution: 'Contains modified Copernicus Sentinel data (2026)',
  sourceUrl: 'https://element84.com/earth-search/',
  approxBytes: 2_050_000,
  bbox: [-157.73984625487188, 21.311994516997498, 12.385830942445828, 45.46620536499362]
} satisfies DatasetInfo;
