import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'ibtracs-north-atlantic',
  title: 'Atlantic hurricane best tracks, 1980-2025 (IBTrACS)',
  description:
    '739 North Atlantic tropical cyclones, 6-hourly fixes with wind, pressure, Saffir-Simpson category, name, season and storm id. Satellite era only: earlier storms far from land were undersampled.',
  license: 'Public domain (NOAA NCEI)',
  attribution:
    'Knapp, K. R., Kruk, M. C., Levinson, D. H., Diamond, H. J., Neumann, C. J. (2010). IBTrACS, Bulletin of the American Meteorological Society 91, 363-376. NOAA NCEI, v04r01',
  sourceUrl: 'https://www.ncei.noaa.gov/products/international-best-track-archive',
  approxBytes: 452_000,
  bbox: [-105, 7, 13.5, 70.7]
} satisfies DatasetInfo;
