// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Option state of the noding-and-coverage scene. Type only, so scene files load no GPU code. */
export type NodingAndCoverageOptions = {
  view: 'crossings' | 'noding' | 'dissolve' | 'generalise';
  // Crossings.
  crossMode: 'streets-rail' | 'streets-self';
  crossKinds: 'proper' | 'touches' | 'overlaps' | 'all';
  sameFeatureOnly: boolean;
  crossSpatialSort: boolean;
  // Noding.
  nodingInput: 'rail' | 'rail-bus';
  nodingOutput: 'pieces' | 'merged' | 'network';
  nodingTolerance: number;
  nodingCapacity: string;
  nodingSpatialSort: boolean;
  // Dissolve with ring assembly.
  dissolveBy: 'state' | 'rucc' | 'none';
  dissolveColor: 'group' | 'hole';
  cancelOpposing: boolean;
  splitTouching: boolean;
  vertexTolerance: string;
  interiorSide: 'left' | 'right';
  normalizeWinding: boolean;
  showFill: boolean;
  // Generalise with coverage simplification.
  coverageTolerance: number;
  topologyRounds: string;
  coverageSnap: string;
  coverageRounds: string;
  coverageOutline: 'coverage' | 'independent' | 'both' | 'original';
};
