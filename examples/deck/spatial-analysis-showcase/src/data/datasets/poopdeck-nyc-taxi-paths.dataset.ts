import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'poopdeck-nyc-taxi-paths',
  title: 'NYC taxi routes, Friday 2 January 2015, 08:00 to 08:30 (9,000 trips)',
  description:
    '9,000 yellow-taxi trips active during the Friday morning rush, as OSRM-routed paths with a timestamp on every vertex plus fare and distance. The routes are derived by a routing engine between the recorded pickup and dropoff points; they are not GPS traces.',
  license:
    'NYC Open Data terms of use (NYC TLC trip records, no restrictions); OSRM routes on OpenStreetMap data (ODbL)',
  attribution:
    'NYC Taxi & Limousine Commission trip records via NYC Open Data; poopdeck.gl nyc-taxi-paths archive; © OpenStreetMap contributors',
  sourceUrl: 'https://tiles.poopdeck.gl/data/nyc-taxi-paths/manifest.json',
  approxBytes: 5_114_140,
  bbox: [-74.03, 40.68, -73.9, 40.82]
} satisfies DatasetInfo;
