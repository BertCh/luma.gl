import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'poopdeck-adsb-paths',
  title: 'US jet flights, 6 January 2020 (OpenSky ADS-B)',
  description:
    '38,135 flights of one UTC day (Monday 6 January 2020) over the contiguous United States, rebuilt from the OpenSky state vectors in the poopdeck.gl `flights` archive: ground pings dropped, flights split at coverage gaps, stale repeated positions removed, then simplified in space and time to 455k vertices with per-vertex time, altitude and reported ground speed. Only flights that reach 7,500 m are kept, so light aircraft and helicopters are excluded.',
  license:
    'OpenSky Network data terms: research and non-commercial use with attribution (the licence text could not be fetched when this entry was written, so treat the data as NON-COMMERCIAL until checked). Not CC0 or CC BY: do not redistribute commercially.',
  attribution:
    'Data from the OpenSky Network, https://opensky-network.org. Matthias Schafer, Martin Strohmeier, Vincent Lenders, Ivan Martinovic and Matthias Wilhelm, "Bringing Up OpenSky: A Large-scale ADS-B Sensor Network for Research", IPSN 2014. Prepared via the poopdeck.gl archives.',
  sourceUrl: 'https://opensky-network.org/data/impala',
  approxBytes: 6_974_544,
  bbox: [-124.99, 25, -65, 50]
} satisfies DatasetInfo;
