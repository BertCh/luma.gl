import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'naturalearth-atlantic-coast',
  title: 'Atlantic basin coastline (Natural Earth 1:50m)',
  description:
    'Generalised coastline of the Americas, Caribbean, western Europe and West Africa, used as seeds for distance-to-coast.',
  license: 'Public domain (Natural Earth)',
  attribution: 'Made with Natural Earth',
  sourceUrl: 'https://www.naturalearthdata.com/downloads/50m-physical-vectors/50m-coastline/',
  approxBytes: 92_000,
  bbox: [-105, 5, 15, 62]
} satisfies DatasetInfo;
