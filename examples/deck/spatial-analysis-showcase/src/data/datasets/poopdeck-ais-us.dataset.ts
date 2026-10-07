import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'poopdeck-ais-us',
  title: 'US coastal AIS vessel tracks, 9 January 2023',
  description:
    '15,555 AIS trajectories of 13,436 vessels in US waters (lower 48, Gulf and Great Lakes) over one UTC day, rebuilt from one position report per vessel per about 12 minutes, with vessel type, length and reported speed. Terrestrial receivers only: the open ocean and many small-craft (Class B) reports are missing.',
  license: 'Public domain (US Government work)',
  attribution:
    'NOAA Office for Coastal Management / BOEM / U.S. Coast Guard, Marine Cadastre AIS data (via the poopdeck.gl ais-all-us archive)',
  sourceUrl: 'https://hub.marinecadastre.gov/pages/vesseltraffic',
  approxBytes: 6_080_000,
  bbox: [-127.33, 24, -66.08, 49.49]
} satisfies DatasetInfo;
