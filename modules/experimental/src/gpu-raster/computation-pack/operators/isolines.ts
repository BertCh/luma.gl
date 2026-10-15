// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RasterOperatorTable} from '../types';

const CELL = {kind: 'cell', label: 'grid cell', indexSpace: '2d'} as const;

const ISOLINE_SOURCES = [
  'modules/experimental/src/gpu-raster/isolines/gpu-isolines.ts',
  'modules/experimental/src/gpu-raster/isolines/isolines-utils.ts'
] as const;

const ISOBAND_SOURCES = [
  'modules/experimental/src/gpu-raster/isolines/gpu-isobands.ts',
  'modules/experimental/src/gpu-raster/isolines/isobands-wgsl.ts'
] as const;

/** Raster pack entries of the `isolines` family, keyed by exported class name. */
export const RASTER_ISOLINES_OPERATORS: RasterOperatorTable = {
  GPUIsolines: {
    family: 'isolines',
    descriptor: {scope: 'focal', reduction: 'scan-sort'},
    variables: {
      isolinesElevation: 'elevation',
      isolinesVertices: 'measure',
      isolinesPolylineOffsets: 'measure',
      isolinesPolylineLevels: 'rank',
      isolinesPolylineClosed: 'category',
      isolinesPolylineCount: 'measure',
      isolinesCount: 'measure',
      isolinesOverflow: 'category',
      isolinesPolylineOverflow: 'category'
    },
    stages: ['representation', 'neighbourhood', 'operator'],
    datumKinds: [{kind: 'polyline', label: 'contour polyline', indexSpace: '1d'}, CELL],
    neighbourhood: 'window',
    guards: [],
    choices: [
      {stage: 'operator', name: 'levelCount', sweepable: true},
      {stage: 'support', name: 'localOrigin', sweepable: false}
    ],
    formula: {
      id: 'gpu-raster.isolines.rule',
      derivation: 'curated',
      tex: String.raw`\term{vertex}{p}=p_a+\term{t}{t}\,(p_b-p_a),\qquad \term{t}{t}=\frac{\term{level}{L}-\term{startValue}{v_a}}{\term{endValue}{v_b}-\term{startValue}{v_a}}`,
      ascii:
        'Samples sit at cell centres; a corner is high when value >= level. On each cell edge whose ends differ, with a and b the canonical start and end samples (+x or +y), the vertex is p_a + t (p_b - p_a), t = (level - v_a) / (v_b - v_a). A segment joins each high-to-low exit edge to its paired entry edge (saddles are joined when the mean of the four corners >= level); a cell with a nodata corner emits nothing; vertices are relative to localOrigin; segments sharing an edge are stitched into polylines.',
      symbols: [
        {id: 'vertex', role: 'output', label: 'vertex', binding: 'wgsl:segments'},
        {id: 't', role: 'intermediate', label: 't', binding: 'wgsl:t'},
        {id: 'level', role: 'parameter', label: 'level', binding: 'wgsl:level'},
        {id: 'startValue', role: 'input', label: 'start sample', binding: 'wgsl:a'},
        {id: 'endValue', role: 'input', label: 'end sample', binding: 'wgsl:b'}
      ]
    },
    sources: ISOLINE_SOURCES
  },
  GPUIsobands: {
    family: 'isolines',
    descriptor: {scope: 'focal', reduction: 'scan-sort'},
    variables: {
      isobandsElevation: 'elevation',
      isobandsTriangles: 'measure',
      isobandsTriangleOffsets: 'measure',
      isobandsTriangleBands: 'rank',
      isobandsTriangleCount: 'measure',
      isobandsOverflow: 'category'
    },
    stages: ['representation', 'neighbourhood', 'operator'],
    datumKinds: [{kind: 'triangle', label: 'band triangle', indexSpace: '1d'}, CELL],
    neighbourhood: 'window',
    guards: [],
    choices: [
      {stage: 'operator', name: 'breakCount', sweepable: true},
      {stage: 'operator', name: 'firstBand', sweepable: false},
      {stage: 'operator', name: 'lastBand', sweepable: false},
      {stage: 'support', name: 'localOrigin', sweepable: false}
    ],
    formula: {
      formula: 'none',
      reason:
        "Each cell's band fragment is built combinatorially from the cell boundary walk and fan-triangulated; the triangle list has no closed form per cell."
    },
    sources: ISOBAND_SOURCES
  },
  GPUIsobandRings: {
    family: 'isolines',
    descriptor: {scope: 'focal', reduction: 'scan-sort'},
    variables: {
      isobandRingsElevation: 'elevation',
      isobandRingPositions: 'measure',
      isobandRingOffsets: 'measure',
      isobandRingBands: 'rank',
      isobandRingCount: 'measure',
      isobandRingOverflow: 'category',
      isobandRingEdgeCount: 'measure',
      isobandRingEdgeOverflow: 'category'
    },
    stages: ['representation', 'neighbourhood', 'operator'],
    datumKinds: [{kind: 'ring', label: 'band ring', indexSpace: '1d'}, CELL],
    neighbourhood: 'window',
    guards: [],
    choices: [
      {stage: 'operator', name: 'breakCount', sweepable: true},
      {stage: 'support', name: 'edgeCapacity', sweepable: false},
      {stage: 'support', name: 'localOrigin', sweepable: false}
    ],
    formula: {
      formula: 'none',
      reason:
        'Band boundary edges are hashed, matched and chained into closed rings by pointer jumping; the rings have no closed form.'
    },
    sources: [
      'modules/experimental/src/gpu-raster/isolines/gpu-isoband-rings.ts',
      'modules/experimental/src/gpu-raster/isolines/gpu-isobands.ts'
    ]
  }
};
