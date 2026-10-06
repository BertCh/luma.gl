// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Reference fixture generated with Shapely 2.1.2: hand-built multipolygons (explicitly closed
 * rings) with `MultiPolygon.is_valid` and `explain_validity` recorded as `shapelyValid` and `reason`.
 */
export const SHAPELY_VALIDITY_FIXTURE: readonly {
  name: string;
  polygons: [number, number][][][];
  shapelyValid: boolean;
  reason: string;
}[] = [
  {
    name: 'valid_square',
    polygons: [
      [
        [
          [0, 0],
          [4, 0],
          [4, 4],
          [0, 4],
          [0, 0]
        ]
      ]
    ],
    shapelyValid: true,
    reason: 'Valid Geometry'
  },
  {
    name: 'valid_hole',
    polygons: [
      [
        [
          [0, 0],
          [4, 0],
          [4, 4],
          [0, 4],
          [0, 0]
        ],
        [
          [1, 1],
          [1, 3],
          [3, 3],
          [3, 1],
          [1, 1]
        ]
      ]
    ],
    shapelyValid: true,
    reason: 'Valid Geometry'
  },
  {
    name: 'cw_square',
    polygons: [
      [
        [
          [0, 0],
          [0, 4],
          [4, 4],
          [4, 0],
          [0, 0]
        ]
      ]
    ],
    shapelyValid: true,
    reason: 'Valid Geometry'
  },
  {
    name: 'bowtie',
    polygons: [
      [
        [
          [0, 0],
          [4, 4],
          [4, 0],
          [0, 4],
          [0, 0]
        ]
      ]
    ],
    shapelyValid: false,
    reason: 'Self-intersection[2 2]'
  },
  {
    name: 'hole_outside',
    polygons: [
      [
        [
          [0, 0],
          [4, 0],
          [4, 4],
          [0, 4],
          [0, 0]
        ],
        [
          [10, 10],
          [10, 12],
          [12, 12],
          [12, 10],
          [10, 10]
        ]
      ]
    ],
    shapelyValid: false,
    reason: 'Hole lies outside shell[10 10]'
  },
  {
    name: 'hole_crossing_shell',
    polygons: [
      [
        [
          [0, 0],
          [4, 0],
          [4, 4],
          [0, 4],
          [0, 0]
        ],
        [
          [2, 2],
          [2, 6],
          [6, 6],
          [6, 2],
          [2, 2]
        ]
      ]
    ],
    shapelyValid: false,
    reason: 'Self-intersection[2 4]'
  },
  {
    name: 'holes_overlap',
    polygons: [
      [
        [
          [0, 0],
          [10, 0],
          [10, 10],
          [0, 10],
          [0, 0]
        ],
        [
          [1, 1],
          [1, 5],
          [5, 5],
          [5, 1],
          [1, 1]
        ],
        [
          [3, 3],
          [3, 7],
          [7, 7],
          [7, 3],
          [3, 3]
        ]
      ]
    ],
    shapelyValid: false,
    reason: 'Self-intersection[5 3]'
  },
  {
    name: 'hole_touches_shell',
    polygons: [
      [
        [
          [0, 0],
          [4, 0],
          [4, 4],
          [0, 4],
          [0, 0]
        ],
        [
          [0, 2],
          [2, 3],
          [2, 1],
          [0, 2]
        ]
      ]
    ],
    shapelyValid: true,
    reason: 'Valid Geometry'
  },
  {
    name: 'self_touch_ring',
    polygons: [
      [
        [
          [0, 0],
          [4, 0],
          [2, 2],
          [4, 4],
          [0, 4],
          [2, 2],
          [0, 0]
        ]
      ]
    ],
    shapelyValid: false,
    reason: 'Ring Self-intersection[2 2]'
  },
  {
    name: 'repeated_vertex',
    polygons: [
      [
        [
          [0, 0],
          [4, 0],
          [4, 0],
          [4, 4],
          [0, 4],
          [0, 0]
        ]
      ]
    ],
    shapelyValid: true,
    reason: 'Valid Geometry'
  },
  {
    name: 'spike_ring',
    polygons: [
      [
        [
          [0, 0],
          [4, 0],
          [4, 4],
          [2, 4],
          [2, 6],
          [2, 4],
          [0, 4],
          [0, 0]
        ]
      ]
    ],
    shapelyValid: false,
    reason: 'Self-intersection[2 6]'
  },
  {
    name: 'multi_disjoint',
    polygons: [
      [
        [
          [0, 0],
          [4, 0],
          [4, 4],
          [0, 4],
          [0, 0]
        ]
      ],
      [
        [
          [10, 0],
          [14, 0],
          [14, 4],
          [10, 4],
          [10, 0]
        ]
      ]
    ],
    shapelyValid: true,
    reason: 'Valid Geometry'
  },
  {
    name: 'multi_overlap',
    polygons: [
      [
        [
          [0, 0],
          [4, 0],
          [4, 4],
          [0, 4],
          [0, 0]
        ]
      ],
      [
        [
          [2, 2],
          [6, 2],
          [6, 6],
          [2, 6],
          [2, 2]
        ]
      ]
    ],
    shapelyValid: false,
    reason: 'Self-intersection[2 4]'
  },
  {
    name: 'multi_shared_edge',
    polygons: [
      [
        [
          [0, 0],
          [4, 0],
          [4, 4],
          [0, 4],
          [0, 0]
        ]
      ],
      [
        [
          [4, 0],
          [8, 0],
          [8, 4],
          [4, 4],
          [4, 0]
        ]
      ]
    ],
    shapelyValid: false,
    reason: 'Self-intersection[4 4]'
  },
  {
    name: 'multi_touch_point',
    polygons: [
      [
        [
          [0, 0],
          [4, 0],
          [4, 4],
          [0, 4],
          [0, 0]
        ]
      ],
      [
        [
          [4, 4],
          [8, 4],
          [8, 8],
          [4, 8],
          [4, 4]
        ]
      ]
    ],
    shapelyValid: true,
    reason: 'Valid Geometry'
  }
];
