import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'alps-dem-wide',
  title: 'Gornergrat view terrain (Swiss Alps, wide)',
  description:
    'Terrarium-encoded DEM of the whole Gornergrat view from Zermatt to Monte Rosa and the Weisshorn at 13 m ground resolution (2331 x 2181, Web Mercator zoom 12, 0.25 m quantisation).',
  license:
    'Mapterhorn terrain tiles built from swisstopo swissALTI3D (OGD), Copernicus GLO-30 and Italian regional models (CC BY 4.0)',
  attribution:
    "(c) swisstopo (swissALTI3D), (c) Mapterhorn; contains modified Copernicus data (provided under COPERNICUS by the European Union and ESA); Regione Autonoma Valle d'Aosta and Regione Piemonte DTM (CC BY 4.0)",
  sourceUrl: 'https://mapterhorn.com/attribution/',
  approxBytes: 4200219,
  bbox: [7.5799, 45.8599, 7.98, 46.12]
} satisfies DatasetInfo;
