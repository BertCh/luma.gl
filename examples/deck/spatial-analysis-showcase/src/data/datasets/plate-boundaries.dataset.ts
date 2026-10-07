import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'plate-boundaries',
  title: 'Tectonic plate boundaries (PB2002)',
  description:
    "The 5,824 boundary steps of Bird's PB2002 plate model as short lines, each with its class (subduction zone, spreading ridge, transform fault and so on) and the relative plate motion.",
  license: 'ODC-BY 1.0',
  attribution:
    'Bird (2003), PB2002; GeoJSON conversion by Hugo Ahlenius, Nordpil (github.com/fraxen/tectonicplates)',
  sourceUrl: 'https://github.com/fraxen/tectonicplates',
  approxBytes: 170_000,
  bbox: [-180, -78, 180, 82]
} satisfies DatasetInfo;
