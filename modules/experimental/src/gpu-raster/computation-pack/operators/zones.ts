// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RasterOperatorTable} from '../types';

const CELL = {kind: 'cell', label: 'grid cell', indexSpace: '2d'} as const;
const ZONE = {kind: 'zone', label: 'zone', indexSpace: '1d'} as const;
const POINT = {kind: 'point', label: 'joined point', indexSpace: '1d'} as const;

/** Raster pack entries of the `zones` family, keyed by exported class name. */
export const RASTER_ZONES_OPERATORS: RasterOperatorTable = {
  GPURasterZonalStatistics: {
    family: 'zones',
    descriptor: {scope: 'zonal', reduction: 'sum'},
    variables: {
      zonalElevation: 'elevation',
      zonalZones: 'category',
      zoneCellCounts: 'measure',
      zoneMeans: 'elevation',
      zoneMinimums: 'elevation',
      zoneMaximums: 'elevation',
      zoneOverflow: 'category'
    },
    stages: ['representation', 'support', 'operator'],
    datumKinds: [ZONE, CELL],
    neighbourhood: 'zone',
    guards: [],
    choices: [
      {stage: 'support', name: 'zoneCapacity', sweepable: false},
      {stage: 'operator', name: 'sumOrder', alternatives: ['sorted', 'atomic'], sweepable: false}
    ],
    formula: {
      id: 'gpu-raster.raster-zonal-statistics.rule',
      derivation: 'curated',
      tex: String.raw`\term{mean}{\bar v_z}=\frac{\term{sum}{\sum_{c:\,\mathrm{zone}(c)=z,\ \mathrm{valid}} v_c}}{\term{count}{n_z}},\quad \term{minimum}{v_z^{\min}}=\min_{c\in z,\ \mathrm{valid}} v_c,\quad \term{maximum}{v_z^{\max}}=\max_{c\in z,\ \mathrm{valid}} v_c`,
      ascii:
        'Cell c is in zone z when its zone ID is below zoneCapacity and not ignoredZone; it is valid when its calibrated value is finite and validity is nonzero. sums = valid values added in a fixed sorted tree (sumOrder sorted), means = sum / valueCount, minimums/maximums over valid cells; a zone with no valid cell gives sum 0 and NaN mean, minimum and maximum; cellCounts counts every in-zone cell. Uniform cell weights (projected grid).',
      symbols: [
        {id: 'mean', role: 'output', label: 'mean', binding: 'wgsl:means'},
        {id: 'sum', role: 'intermediate', label: 'sum', binding: 'wgsl:sums'},
        {id: 'count', role: 'intermediate', label: 'valid count', binding: 'wgsl:valueCounts'},
        {id: 'minimum', role: 'output', label: 'minimum', binding: 'wgsl:minimums'},
        {id: 'maximum', role: 'output', label: 'maximum', binding: 'wgsl:maximums'}
      ]
    },
    sources: [
      'modules/experimental/src/gpu-raster/raster-zonal-statistics/gpu-raster-zonal-statistics.ts'
    ]
  },
  GPUPolygonRasterization: {
    family: 'zones',
    descriptor: {scope: 'zonal', reduction: 'select'},
    variables: {
      perimeterPositions: 'measure',
      perimeterRingOffsets: 'measure',
      polygonZones: 'category',
      polygonBoundary: 'category',
      polygonOverflow: 'category',
      polygonCrossingCount: 'measure'
    },
    stages: ['representation', 'support', 'operator'],
    datumKinds: [CELL],
    neighbourhood: 'none',
    guards: [],
    choices: [
      {stage: 'support', name: 'crossingCapacity', sweepable: false},
      {stage: 'operator', name: 'spatialContext', sweepable: false}
    ],
    formula: {
      formula: 'none',
      reason:
        'Edge crossings per raster row are sorted and paired into even-odd column spans that cells fill with an atomic minimum; the result has no per-cell closed form.'
    },
    sources: [
      'modules/experimental/src/gpu-raster/polygon-rasterization/gpu-polygon-rasterization.ts',
      'modules/experimental/src/gpu-raster/polygon-rasterization/polygon-rasterization-parameters.ts'
    ]
  },
  GPURasterJoin: {
    family: 'zones',
    descriptor: {scope: 'zonal', reduction: 'count'},
    variables: {
      joinPoints: 'measure',
      joinedPoints: 'measure',
      joinPointZones: 'category',
      joinZones: 'category',
      joinZoneCounts: 'measure',
      joinOutsideCount: 'measure'
    },
    stages: ['representation', 'support', 'operator'],
    datumKinds: [POINT, CELL],
    neighbourhood: 'none',
    guards: [],
    choices: [
      {stage: 'support', name: 'zoneCount', sweepable: false},
      {stage: 'operator', name: 'spatialContext', sweepable: false}
    ],
    formula: {
      id: 'gpu-raster.raster-join.rule',
      derivation: 'curated',
      tex: String.raw`\term{zone}{z(p)}=\term{zones}{Z}\!\left[\left\lfloor\frac{\term{position}{p}-\term{origin}{o}}{\term{cellSize}{s}}\right\rfloor\right]`,
      ascii:
        'cell = floor((p - origin) / cellSize) per axis, with origin the south-west corner of the raster and the row counted from the smallest y (mirrored for north-up grids); the point takes Z[cell], or no zone when the cell is outside the raster or Z[cell] >= zoneCount; counts[z] += 1 for each joined point; points outside the raster are tallied in outsideCount.',
      symbols: [
        {id: 'zone', role: 'output', label: 'zone', binding: 'wgsl:pointZones'},
        {id: 'zones', role: 'input', label: 'zone raster', binding: 'wgsl:zones'},
        {id: 'position', role: 'input', label: 'point', binding: 'wgsl:position'},
        {id: 'origin', role: 'parameter', label: 'origin', binding: 'wgsl:origin'},
        {id: 'cellSize', role: 'parameter', label: 'cell size', binding: 'wgsl:cellSize'}
      ]
    },
    sources: [
      'modules/experimental/src/gpu-raster/polygon-rasterization/gpu-raster-join.ts',
      'modules/experimental/src/gpu-raster/polygon-rasterization/polygon-rasterization-parameters.ts'
    ]
  }
};
