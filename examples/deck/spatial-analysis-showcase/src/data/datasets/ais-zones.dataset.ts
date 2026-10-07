import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'ais-zones',
  title: 'NY/NJ Harbor zones',
  description:
    'Official NOAA ENC general anchorages and maintained channels plus clearly labelled approximate terminal, bridge-span, ferry-lane and tour-area polygons, for zone-event analysis of ais-vessels.',
  license: 'NOAA Coast Survey ENC data (public domain); approximate polygons hand-drawn',
  attribution: 'NOAA Coast Survey ENCDirect; approximate zones by the showcase authors',
  sourceUrl: 'https://encdirect.noaa.gov/',
  approxBytes: 39023,
  bbox: [-74.312, 40.396, -73.82, 40.833]
} satisfies DatasetInfo;
