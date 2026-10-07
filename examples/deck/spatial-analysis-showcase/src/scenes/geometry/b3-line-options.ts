// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Option state of the line-operations scene. Type only: scene files import it without GPU code. */
export type LineOperationsOptions = {
  view: 'reshape' | 'snap' | 'tracks';
  // Reshape the L lines.
  lineTool: 'densify' | 'chunk' | 'substring' | 'locate';
  lineSystem: 'planar' | 'spherical';
  densifyLength: number;
  chunkLength: number;
  substringStart: number;
  substringEnd: number;
  locateMode: 'distance' | 'fraction';
  locateSpacing: number;
  locateLateral: number;
  locateAnimate: boolean;
  // Snap eligible community places to streets.
  snapRadius: number;
  snapColor: 'side' | 'measure' | 'distance';
  placeCategory:
    | 'all'
    | 'grocery'
    | 'school_education'
    | 'park_recreation'
    | 'arts_culture'
    | 'worship_community';
  // Simplify and smooth ship tracks.
  trackTool: 'simplify' | 'smooth';
  simplifyMetric: 'segment' | 'time-ratio';
  simplifyRounds: string;
  simplifyTolerance: number;
  simplifyAuto: boolean;
  simplifyPixels: number;
  smoothIterations: number;
  smoothRatio: number;
  showOriginal: boolean;
};

/** Option state of the great-circles scene. */
export type GreatCirclesOptions = {
  worldHub: string;
  arcColor: 'distance' | 'airlines' | 'routes';
  arcMinimumSegments: number;
  arcMaximumLength: number;
  geodesicModel: 'sphere' | 'wgs84';
  ringDistance: number;
  airportColor: 'distance' | 'bearing';
  showArcs: boolean;
  showRing: boolean;
  /** The rhumb comparison is a deliberate story state, never background decoration. */
  showRhumbComparison: boolean;
};
