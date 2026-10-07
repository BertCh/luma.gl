import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'poopdeck-nyc-taxi',
  title: 'NYC yellow-taxi trips, 1 to 2 January 2015 (440,000 trips)',
  description:
    '440,000 yellow-taxi trips as origin and destination pairs with pickup time, route duration, distance, fare and passenger count, covering New Year’s Day and the morning of Friday 2 January 2015. Pickup and dropoff points are the two ends of the routed paths in the poopdeck.gl archive; the duration is derived from the OSRM route, not metered.',
  license:
    'NYC Open Data terms of use (NYC TLC trip records, no restrictions); route endpoints snapped by OSRM on OpenStreetMap data (ODbL)',
  attribution:
    'NYC Taxi & Limousine Commission trip records via NYC Open Data; poopdeck.gl nyc-taxi-paths archive; © OpenStreetMap contributors',
  sourceUrl: 'https://tiles.poopdeck.gl/data/nyc-taxi-paths/manifest.json',
  approxBytes: 6_160_000,
  bbox: [-74.05, 40.6, -73.7, 40.9]
} satisfies DatasetInfo;
