import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'natural-earth',
  title: 'Natural Earth world reference geometry',
  description:
    '1:110m world land, coastline, countries, boundary lines and lakes, and 1:10m land, coastline, boundary lines, lakes and rivers for the Netherlands region.',
  license: 'Public domain (Natural Earth)',
  attribution: 'Made with Natural Earth',
  sourceUrl: 'https://www.naturalearthdata.com/',
  approxBytes: 716817,
  bbox: [-180, -90, 180, 83.6451]
} satisfies DatasetInfo;
