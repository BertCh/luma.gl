// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Gazetteer} from '../../cartography/gazetteer';

/**
 * A scene-local gazetteer for the water bodies the `us-shipping-day` story names but the shared
 * `US` and `WORLD` gazetteers do not hold (the inland waterways and the offshore Atlantic).
 * Labels only: each point is a label position on the water, not a surveyed feature.
 *
 * Toolkit request: promote these to `cartography/gazetteer/us.ts` with a source per coordinate.
 */
export const US_SHIPPING_PLACES: Gazetteer = {
  id: 'us-shipping-waters',
  name: 'US shipping waters',
  places: {
    // label position on the river reach at Vicksburg, Mississippi (wiki: Vicksburg 32.35 N, 90.88 W)
    'mississippi-river': {
      id: 'mississippi-river',
      name: 'Mississippi River',
      kind: 'river',
      lngLat: [-90.93, 32.3],
      minZoom: 3,
      priority: 2,
      size: 'medium'
    },
    // label position in the middle of Lake Michigan (wiki: lake centre about 44 N, 87 W)
    'lake-michigan': {
      id: 'lake-michigan',
      name: 'Lake Michigan',
      kind: 'water',
      lngLat: [-87.0, 44.0],
      minZoom: 3,
      priority: 1,
      size: 'small'
    },
    // label position off the Mid-Atlantic coast, in the receivers' blind zone (label placement)
    'atlantic-offshore': {
      id: 'atlantic-offshore',
      name: 'Atlantic Ocean',
      kind: 'ocean',
      lngLat: [-70.5, 33.5],
      minZoom: 3,
      priority: 1,
      size: 'medium'
    }
  }
};
