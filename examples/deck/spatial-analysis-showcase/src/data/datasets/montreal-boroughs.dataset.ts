import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'montreal-boroughs',
  title: 'Montreal boroughs and linked cities',
  description:
    'The 19 boroughs and 15 linked cities of the Montreal agglomeration, simplified to about 25 m. Each feature carries its name as written in the source and the matching borough string of the bixi-flows dataset.',
  license: 'Creative Commons Attribution 4.0 International',
  attribution: 'Ville de Montréal, données ouvertes (CC BY 4.0)',
  sourceUrl: 'https://donnees.montreal.ca/dataset/limites-administratives-agglomeration',
  approxBytes: 84_691,
  bbox: [-74.0, 45.38, -73.47, 45.71]
} satisfies DatasetInfo;
