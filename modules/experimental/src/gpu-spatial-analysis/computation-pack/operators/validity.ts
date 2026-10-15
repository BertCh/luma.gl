// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {SpatialFamilyOperators} from '../types';

/**
 * Spatial computation-pack entries for the `validity` family, keyed by exported name. Every entry
 * must use `family: 'validity'`.
 */
export const VALIDITY_OPERATORS: SpatialFamilyOperators = {
  GPUSegmentIntersection: {
    family: 'validity',
    descriptor: {scope: 'pairwise', reduction: 'select'},
    variables: {kinds: 'category', points: 'measure', leftIds: 'category', rightIds: 'category'},
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'segmentPair', label: 'intersecting segment pair', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {stage: 'operator', name: 'mode', alternatives: ['self', 'two-sided'], sweepable: false},
      {stage: 'operator', name: 'sameFeatureOnly', alternatives: [true, false], sweepable: false},
      {stage: 'support', name: 'pairCapacity', sweepable: false}
    ],
    formula: {
      formula: 'none',
      reason:
        'A BVH probe finds candidate segment pairs and exact orientation tests classify each one; there is no closed form.'
    },
    metadata: {coordinateModel: 'projected', metricUnit: 'm'}
  },
  GPUGeometryValidity: {
    family: 'validity',
    descriptor: {scope: 'local', reduction: 'none'},
    variables: {mask: 'category'},
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'feature', label: 'polygon feature', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {
        stage: 'operator',
        name: 'orientation',
        alternatives: ['counter-clockwise-shell', 'clockwise-shell', 'ignore'],
        sweepable: false
      },
      {
        stage: 'operator',
        name: 'ringClosure',
        alternatives: ['implicit', 'explicit'],
        sweepable: false
      },
      {stage: 'support', name: 'intersectionCapacity', sweepable: false}
    ],
    formula: {
      formula: 'none',
      reason:
        'Validity is a set of per-ring rules and a same-feature segment intersection, reported as a bitmask rather than computed by a formula.'
    },
    metadata: {coordinateModel: 'projected', metricUnit: 'm'}
  },
  GPUGeometryPredicates: {
    family: 'validity',
    descriptor: {scope: 'local', reduction: 'none'},
    variables: {isSimple: 'category', numCoordinates: 'count', numInteriorRings: 'count'},
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'feature', label: 'polygon feature', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {stage: 'operator', name: 'columns', sweepable: false},
      {stage: 'support', name: 'intersectionCapacity', sweepable: false}
    ],
    formula: {
      formula: 'none',
      reason:
        'Each column is a structural predicate or count read from the offsets, with simplicity from a segment intersection; none is a formula.'
    },
    metadata: {coordinateModel: 'projected', metricUnit: 'm'}
  },
  GPUMinimumClearance: {
    family: 'validity',
    descriptor: {scope: 'zonal', reduction: 'argmin'},
    variables: {clearances: 'measure', vertexIds: 'category'},
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'feature', label: 'polygon feature', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [{stage: 'operator', name: 'outputs', sweepable: false}],
    // gpu-minimum-clearance.ts `probe` kernel (segmentCandidate, rowCandidate) and the feature minimum.
    formula: {
      id: 'gpu-spatial-analysis.minimum-clearance',
      tex: String.raw`\term{clearance}{C_f}=\min_{v\in f}\ \min_{s\in f,\ v\notin s}\ \term{distance}{d(v,s)}`,
      ascii:
        'C_f = min over vertices v of feature f of the distance from v to every segment s of f that does not have v as an ' +
        'endpoint, or to its other endpoint when v is an endpoint of s. Infinity when nothing can be measured.',
      symbols: [
        {id: 'clearance', role: 'output', label: 'C_f', binding: 'wgsl:clearances'},
        {id: 'distance', role: 'intermediate', label: 'd(v, s)', binding: 'wgsl:segmentCandidate'}
      ],
      derivation: 'curated'
    },
    metadata: {coordinateModel: 'projected', metricUnit: 'm'}
  },
  GPUCoverageValidity: {
    family: 'validity',
    descriptor: {scope: 'pairwise', reduction: 'count'},
    variables: {segmentFlags: 'category', featureInvalidCounts: 'count'},
    stages: ['representation', 'support', 'operator'],
    datumKinds: [{kind: 'feature', label: 'tract polygon', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [{stage: 'support', name: 'gapWidth', label: 'gap width', unit: 'm', sweepable: true}],
    formula: {
      formula: 'none',
      reason:
        'Each ring edge is compared with nearby edges of other polygons by a rule set (match, cross, gap, overlap) and counted per feature.'
    },
    metadata: {coordinateModel: 'projected', metricUnit: 'm'}
  }
};
