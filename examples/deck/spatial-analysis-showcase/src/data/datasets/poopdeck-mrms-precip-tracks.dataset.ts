import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'poopdeck-mrms-precip-tracks',
  title: 'Storm-cell tracks, 21-22 May 2024',
  description:
    '289 radar storm-cell tracks (cells of 51 dBZ and above followed by the NOAA MRMS composite) from 12:00 UTC on 21 May to 06:00 UTC on 22 May 2024 across the central and eastern US. The archive stores hourly pieces; they were chained back into tracks. Every vertex has a time and the cell reflectivity.',
  license: 'Public domain (US Government work)',
  attribution: 'NOAA MRMS (Multi-Radar Multi-Sensor), via poopdeck.gl',
  sourceUrl: 'https://tiles.poopdeck.gl/data/mrms-precip-tracks/manifest.json',
  approxBytes: 70_000,
  bbox: [-104.0945, 24.0143, -70.9678, 48.6489]
} satisfies DatasetInfo;
