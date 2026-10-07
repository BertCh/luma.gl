// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Reference fixture generated with Shapely 2.1.2 (GEOS): `shared_paths(left[i], right[j])` for
 * every pair of 8 left and 8 right self-avoiding lattice walks (some with collinear vertices kept,
 * some compressed), plus two handcrafted pairs (a backward run and a forward run across
 * vertices). `shared` lists the pairs with a nonzero result: total length and merged pieces (each
 * stored with its lexicographically smaller end first) of the forward and backward parts.
 */
export const SHAPELY_SHARED_PATHS_FIXTURE = {
  left: [
    [
      [4, 3],
      [3, 3],
      [3, 2],
      [3, 1],
      [3, 0],
      [4, 0],
      [5, 1],
      [6, 2],
      [6, 3]
    ],
    [
      [6, 1],
      [5, 2],
      [4, 2],
      [5, 3],
      [5, 4],
      [6, 5],
      [6, 6],
      [5, 5],
      [4, 6],
      [3, 5]
    ],
    [
      [4, 1],
      [3, 1],
      [2, 0],
      [1, 1],
      [1, 0],
      [0, 1],
      [0, 2]
    ],
    [
      [2, 0],
      [2, 1],
      [1, 1],
      [1, 0],
      [0, 1],
      [0, 2],
      [1, 3],
      [1, 2]
    ],
    [
      [0, 2],
      [0, 1],
      [1, 1],
      [1, 2],
      [2, 2],
      [2, 3],
      [3, 4],
      [4, 5]
    ],
    [
      [5, 3],
      [5, 2],
      [4, 3],
      [3, 3],
      [2, 4],
      [3, 5],
      [4, 4],
      [3, 4],
      [2, 3],
      [2, 2]
    ],
    [
      [0, 0],
      [1, 0],
      [2, 0],
      [3, 0],
      [4, 0]
    ],
    [
      [0, 3],
      [2, 3],
      [4, 3]
    ]
  ],
  right: [
    [
      [1, 4],
      [2, 3],
      [1, 3],
      [0, 3],
      [0, 4],
      [1, 5],
      [2, 6]
    ],
    [
      [2, 6],
      [2, 5],
      [1, 6],
      [0, 5],
      [0, 4],
      [4, 4],
      [3, 3],
      [4, 2]
    ],
    [
      [6, 4],
      [5, 4],
      [4, 5],
      [5, 5],
      [4, 6],
      [5, 6],
      [6, 5]
    ],
    [
      [1, 1],
      [0, 2],
      [1, 2],
      [2, 2],
      [3, 3],
      [4, 3]
    ],
    [
      [2, 2],
      [1, 2],
      [2, 3],
      [3, 2],
      [3, 1],
      [4, 2],
      [4, 1],
      [5, 2]
    ],
    [
      [5, 0],
      [6, 1],
      [6, 2],
      [5, 1],
      [4, 1],
      [3, 2],
      [3, 3],
      [4, 2]
    ],
    [
      [4, 0],
      [3, 0],
      [2, 0]
    ],
    [
      [1, 3],
      [2, 3],
      [3, 3],
      [4, 3]
    ]
  ],
  shared: [
    {
      left: 0,
      right: 3,
      forward: {length: 0, pieces: []},
      backward: {
        length: 1.0,
        pieces: [
          [
            [3.0, 3.0],
            [4.0, 3.0]
          ]
        ]
      }
    },
    {
      left: 0,
      right: 4,
      forward: {
        length: 1.0,
        pieces: [
          [
            [3.0, 1.0],
            [3.0, 2.0]
          ]
        ]
      },
      backward: {length: 0, pieces: []}
    },
    {
      left: 0,
      right: 5,
      forward: {length: 0, pieces: []},
      backward: {
        length: 2.414213562373095,
        pieces: [
          [
            [3.0, 2.0],
            [3.0, 3.0]
          ],
          [
            [5.0, 1.0],
            [6.0, 2.0]
          ]
        ]
      }
    },
    {
      left: 0,
      right: 6,
      forward: {length: 0, pieces: []},
      backward: {
        length: 1.0,
        pieces: [
          [
            [3.0, 0.0],
            [4.0, 0.0]
          ]
        ]
      }
    },
    {
      left: 0,
      right: 7,
      forward: {length: 0, pieces: []},
      backward: {
        length: 1.0,
        pieces: [
          [
            [3.0, 3.0],
            [4.0, 3.0]
          ]
        ]
      }
    },
    {
      left: 1,
      right: 2,
      forward: {
        length: 1.4142135623730951,
        pieces: [
          [
            [4.0, 6.0],
            [5.0, 5.0]
          ]
        ]
      },
      backward: {length: 0, pieces: []}
    },
    {
      left: 4,
      right: 3,
      forward: {
        length: 1.0,
        pieces: [
          [
            [1.0, 2.0],
            [2.0, 2.0]
          ]
        ]
      },
      backward: {length: 0, pieces: []}
    },
    {
      left: 4,
      right: 4,
      forward: {length: 0, pieces: []},
      backward: {
        length: 1.0,
        pieces: [
          [
            [1.0, 2.0],
            [2.0, 2.0]
          ]
        ]
      }
    },
    {
      left: 5,
      right: 1,
      forward: {length: 0, pieces: []},
      backward: {
        length: 1.0,
        pieces: [
          [
            [3.0, 4.0],
            [4.0, 4.0]
          ]
        ]
      }
    },
    {
      left: 5,
      right: 3,
      forward: {length: 0, pieces: []},
      backward: {
        length: 1.0,
        pieces: [
          [
            [3.0, 3.0],
            [4.0, 3.0]
          ]
        ]
      }
    },
    {
      left: 5,
      right: 7,
      forward: {length: 0, pieces: []},
      backward: {
        length: 1.0,
        pieces: [
          [
            [3.0, 3.0],
            [4.0, 3.0]
          ]
        ]
      }
    },
    {
      left: 6,
      right: 6,
      forward: {length: 0, pieces: []},
      backward: {
        length: 2.0,
        pieces: [
          [
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0]
          ]
        ]
      }
    },
    {
      left: 7,
      right: 0,
      forward: {length: 0, pieces: []},
      backward: {
        length: 2.0,
        pieces: [
          [
            [0.0, 3.0],
            [1.0, 3.0],
            [2.0, 3.0]
          ]
        ]
      }
    },
    {
      left: 7,
      right: 3,
      forward: {
        length: 1.0,
        pieces: [
          [
            [3.0, 3.0],
            [4.0, 3.0]
          ]
        ]
      },
      backward: {length: 0, pieces: []}
    },
    {
      left: 7,
      right: 7,
      forward: {
        length: 3.0,
        pieces: [
          [
            [1.0, 3.0],
            [2.0, 3.0],
            [3.0, 3.0],
            [4.0, 3.0]
          ]
        ]
      },
      backward: {length: 0, pieces: []}
    }
  ]
} as const;
