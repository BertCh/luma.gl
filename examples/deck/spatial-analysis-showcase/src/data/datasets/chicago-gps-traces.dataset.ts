import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'chicago-gps-traces',
  title: 'Chicago simulated GPS traces (for map matching)',
  description:
    'SIMULATED noisy GPS traces from 200 random shortest paths on chicago-roads (5-15 s sampling, 10-25 m noise, dropouts) with ground-truth edge sequences. Not real vehicles.',
  license: 'Synthetic; derived from OpenStreetMap, ODbL 1.0',
  attribution: '© OpenStreetMap contributors',
  sourceUrl: 'https://www.openstreetmap.org/copyright',
  approxBytes: 842_433,
  bbox: [-87.83374, 41.65208, -87.52523, 42.01797]
} satisfies DatasetInfo;
