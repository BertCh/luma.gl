import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'chicago-nature',
  title: 'Chicago nature observations, 2023',
  description:
    'iNaturalist observations of wild plants, birds, insects, fungi and other life inside Chicago in 2023 (43.6k sightings of 4.8k taxa) with taxon group, research-grade and introduced-species flags, community area and tract.',
  license: 'CC0, CC BY 4.0 or CC BY-NC 4.0 per observation (non-commercial use)',
  attribution: 'iNaturalist contributors',
  sourceUrl: 'https://www.inaturalist.org/observations?place_id=49906&year=2023',
  approxBytes: 961720,
  bbox: [-87.916, 41.645, -87.525, 42.023]
} satisfies DatasetInfo;
