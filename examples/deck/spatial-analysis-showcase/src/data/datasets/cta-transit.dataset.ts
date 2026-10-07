import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'cta-transit',
  title: 'CTA bus stops, rail stations, routes and hops',
  description:
    'Derived from the CTA GTFS feed: stops and stations with route counts and weekday frequency, route shapes, and a weekday stop-to-stop hop graph with median scheduled times.',
  license:
    'CTA Developer License Agreement and Terms of Use (derived data for transit-accessibility use; credit optional; no stand-alone resale)',
  attribution:
    'Data provided by Chicago Transit Authority (not affiliated with or endorsed by CTA)',
  sourceUrl: 'https://www.transitchicago.com/developers/gtfs/',
  approxBytes: 949_168,
  bbox: [-87.90422, 41.64417, -87.52571, 42.03288]
} satisfies DatasetInfo;
