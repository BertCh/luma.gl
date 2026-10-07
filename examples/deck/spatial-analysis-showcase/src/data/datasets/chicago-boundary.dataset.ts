import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'chicago-boundary',
  title: 'Chicago city limit, Lake Michigan and land mask',
  description:
    'The Chicago city limit (dissolved community areas), Lake Michigan clipped to the Chicago region (Natural Earth, generalised) and the matching land mask for fading the lake.',
  license: 'City of Chicago Data Portal Terms of Use; public domain (Natural Earth)',
  attribution: 'City of Chicago; Made with Natural Earth',
  sourceUrl:
    'https://data.cityofchicago.org/Facilities-Geographic-Boundaries/Boundaries-Community-Areas/igwz-8jzy',
  approxBytes: 64905,
  bbox: [-88, 41.55, -87.2, 42.15]
} satisfies DatasetInfo;
