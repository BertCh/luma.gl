import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'openflights',
  title: 'OpenFlights global airline network',
  description:
    '3,257 airports and 18,930 undirected airport pairs with route counts and airline counts; community data frozen around 2014.',
  license: 'Open Database License (ODbL) 1.0',
  attribution: 'OpenFlights.org (airports.dat, routes.dat), ODbL',
  sourceUrl: 'https://openflights.org/data.php',
  approxBytes: 601131,
  bbox: [-179.877, -54.843, 179.341, 78.246]
} satisfies DatasetInfo;
