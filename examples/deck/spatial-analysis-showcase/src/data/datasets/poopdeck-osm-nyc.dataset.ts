import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'poopdeck-osm-nyc',
  title: 'OpenStreetMap New York City: node creations 2007-2026 (400k sample)',
  description:
    'A seeded uniform sample of 400,000 of the 901,827 tagged nodes ever created in OpenStreetMap New York City, with creation time, a coarse kind and an anonymous contributor rank, plus exact full-history monthly counts and top-contributor shares. No user names or ids are included.',
  license: 'ODbL 1.0 (Open Database License)',
  attribution: '© OpenStreetMap contributors',
  sourceUrl: 'https://tiles.poopdeck.gl/data/osm-nyc-nodes/manifest.json',
  approxBytes: 6_000_000,
  bbox: [-74.2692, 40.49, -73.68, 40.92]
} satisfies DatasetInfo;
