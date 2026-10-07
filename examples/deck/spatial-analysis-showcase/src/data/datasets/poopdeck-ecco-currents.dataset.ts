import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'poopdeck-ecco-currents',
  title: 'ECCO V4r4 modelled surface currents, 2016-12 to 2017-12',
  description:
    'A 30,000-piece sample of the modelled surface particles of the ECCO ocean state estimate (3.5-day displacements with speed), plus a 0.5 degree annual-mean current field (u, v in degrees per day) derived here from all 1.44 million pieces.',
  license: 'NASA Earth science data, open and free to use',
  attribution:
    'ECCO Consortium, NASA/JPL ECCO V4r4 ocean state estimate (PO.DAAC), particle archive from poopdeck.gl; mean field derived by this showcase',
  sourceUrl: 'https://ecco-group.org/',
  approxBytes: 4_398_375,
  bbox: [-180, -79.1, 180, 81]
} satisfies DatasetInfo;
