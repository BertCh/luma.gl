// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Great-circle nearest-feature oracle values on the sphere R = 6371008.8 m.
 *
 * Script: scratchpad/fix/work/E/E6-geodesic-nearest/geodesic_nearest_fixtures.py (numpy 2.5.4
 * haversine, cross-checked with pyproj 3.8.0 `Geod(a=R, f=0)`). Each feature is one two-vertex
 * line; an edge is straight in longitude/latitude (longitude difference wrapped to +-180), and the
 * distance to it is the minimum haversine over 200001 samples of the edge, whose argmin is the
 * foot point. `ranked[query]` lists every feature by `(distance, row)`.
 */
export type GeodesicNearestScene = {
  features: [number, number][][];
  queries: [number, number][];
  ranked: {row: number; distance: number; foot: [number, number]}[][];
};

export const GEODESIC_NEAREST_SCENES: Record<string, GeodesicNearestScene> = {
  equator: {
    features: [
      [
        [0.01, -0.01],
        [0.01, 0.01]
      ],
      [
        [0.03, 0],
        [0.05, 0]
      ],
      [
        [-0.02, 0.02],
        [-0.02, 0.04]
      ],
      [
        [1, 1],
        [1.01, 1]
      ]
    ],
    queries: [
      [0, 0],
      [0.04, 0.005]
    ],
    ranked: [
      [
        {row: 0, distance: 1111.9508023353292, foot: [0.01, 0.0]},
        {row: 2, distance: 3145.0717787737444, foot: [-0.02, 0.02]},
        {row: 1, distance: 3335.852407005987, foot: [0.03, 0.0]},
        {row: 3, distance: 157249.59847404022, foot: [1.0, 1.0]}
      ],
      [
        {row: 1, distance: 555.9754011676646, foot: [0.04, 0.0]},
        {row: 0, distance: 3335.8523943040013, foot: [0.01, 0.005000000000000003]},
        {row: 2, distance: 6877.035740259994, foot: [-0.02, 0.02]},
        {row: 3, distance: 153736.23442320194, foot: [1.0, 1.0]}
      ]
    ]
  },
  latitude60: {
    features: [
      [
        [10, 60.001],
        [10.02, 60.001]
      ],
      [
        [10, 59.99],
        [10.02, 59.99]
      ],
      [
        [10.05, 60],
        [10.06, 60]
      ]
    ],
    queries: [
      [10.01, 60],
      [10.03, 60.0]
    ],
    ranked: [
      [
        {row: 0, distance: 111.19508023324306, foot: [10.01, 60.001]},
        {row: 1, distance: 1111.9508023338453, foot: [10.01, 59.99]},
        {row: 2, distance: 2223.901570798751, foot: [10.05, 60.0]}
      ],
      [
        {row: 0, distance: 566.9776429460171, foot: [10.02, 60.001]},
        {row: 2, distance: 1111.9507981014094, foot: [10.05, 60.0]},
        {row: 1, distance: 1243.2363706464987, foot: [10.02, 59.99]}
      ]
    ]
  },
  antimeridian: {
    features: [
      [
        [179.99, 0.01],
        [-179.99, 0.01]
      ],
      [
        [-179.97, 0],
        [-179.96, 0]
      ],
      [
        [179.95, -0.02],
        [179.96, -0.02]
      ]
    ],
    queries: [
      [180, 0],
      [-179.98, 0.005],
      [179.97, 0]
    ],
    ranked: [
      [
        {row: 0, distance: 1111.9508023353292, foot: [180.0, 0.01]},
        {row: 1, distance: 3335.852407002953, foot: [-179.97, 0.0]},
        {row: 2, distance: 4972.795082524325, foot: [179.96, -0.02]}
      ],
      [
        {row: 0, distance: 1243.1987819913757, foot: [-179.99, 0.01]},
        {row: 1, distance: 1243.198789565376, foot: [-179.97, 0.0]},
        {row: 2, distance: 7227.680113563987, foot: [179.96, -0.02]}
      ],
      [
        {row: 2, distance: 2486.3975715581087, foot: [179.96, -0.02]},
        {row: 0, distance: 2486.3975715595784, foot: [179.99, 0.01]},
        {row: 1, distance: 6671.704814012227, foot: [-179.97, 0.0]}
      ]
    ]
  },
  pole: {
    features: [
      [
        [0, 89.99],
        [20, 89.99]
      ],
      [
        [180, 89.995],
        [160, 89.995]
      ],
      [
        [30, 89.98],
        [31, 89.98]
      ]
    ],
    queries: [
      [10, 89.995],
      [170, 89.99]
    ],
    ranked: [
      [
        {row: 0, distance: 555.9754011676299, foot: [10.0, 89.99]},
        {row: 1, distance: 1074.0619974479123, foot: [160.0, 89.995]},
        {row: 2, distance: 1712.048503361953, foot: [30.0, 89.98]}
      ],
      [
        {row: 1, distance: 555.9754011676299, foot: [170.0, 89.995]},
        {row: 0, distance: 2148.1239943471514, foot: [20.0, 89.99]},
        {row: 2, distance: 3148.7726218971848, foot: [31.0, 89.98]}
      ]
    ]
  }
};
