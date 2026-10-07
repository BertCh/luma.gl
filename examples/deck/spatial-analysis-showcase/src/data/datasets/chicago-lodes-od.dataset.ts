import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'chicago-lodes-od',
  title: 'Chicago commuting flows (LODES 2021)',
  description:
    'Tract-to-tract home-to-work flows within Chicago (top 40,000) split by earnings band, plus jobs per workplace tract.',
  license: 'Public domain (US Census Bureau LEHD)',
  attribution: 'US Census Bureau, LEHD Origin-Destination Employment Statistics (LODES8, 2021)',
  sourceUrl: 'https://lehd.ces.census.gov/data/',
  approxBytes: 978771,
  bbox: [-87.94, 41.644, -87.524, 42.023]
} satisfies DatasetInfo;
