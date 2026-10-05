// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {SpatialAnalysisModeDefinition} from '../spatial-analysis-mode';
import {densityMode} from './density-mode';
import {timeMode} from './time-mode';
import {lassoMode} from './lasso-mode';
import {polygonJoinMode} from './polygon-join-mode';
import {nearestJoinMode} from './nearest-join-mode';
import {zonalMode} from './zonal-mode';
import {bufferMode} from './buffer-mode';
import {clustersMode} from './clusters-mode';
import {flowsMode} from './flows-mode';
import {trajectoriesMode} from './trajectories-mode';
import {reachabilityMode} from './reachability-mode';
import {networkAnalysisMode} from './network-analysis-mode';
import {terrainMode} from './terrain-mode';
import {hydrologyMode} from './hydrology-mode';
import {costDistanceMode} from './cost-distance-mode';
import {rasterZonalMode} from './raster-zonal-mode';
import {tileLodMode} from './tile-lod-mode';
import {hotSpotsMode} from './hot-spots-mode';
import {distanceFieldMode} from './distance-field-mode';
import {rasterJoinMode} from './raster-join-mode';
import {cellPyramidMode} from './cell-pyramid-mode';
import {playheadMode} from './playhead-mode';
import {accessibilityMode} from './accessibility-mode';
import {simplificationMode} from './simplification-mode';
import {interpolationMode} from './interpolation-mode';
import {classificationMode} from './classification-mode';
import {groupStatisticsMode} from './group-statistics-mode';
import {cellsMode} from './cells-mode';
import {dotDensityMode} from './dot-density-mode';
import {spatialWeightsMode} from './spatial-weights-mode';
import {pointPatternMode} from './point-pattern-mode';
import {geometryMode} from './geometry-mode';
import {regressionMode} from './regression-mode';
import {contoursMode} from './contours-mode';
import {suitabilityMode} from './suitability-mode';
import {reliefMode} from './relief-mode';
import {flowFieldMode} from './flow-field-mode';
import {spaceTimeMode} from './space-time-mode';
import {visibilityMode} from './visibility-mode';
import {geomorphometryMode} from './geomorphometry-mode';
import {reliefVisualizationMode} from './relief-visualization-mode';
import {drainageMode} from './drainage-mode';
import {terrainFeaturesMode} from './terrain-features-mode';

/**
 * Registered explorer modes, in tab order.
 *
 * To add a mode: create `modes/<name>-mode.ts` exporting a `SpatialAnalysisModeDefinition` and append it
 * here. The shell, panel tabs, `?mode=` URL parameter, and headless capture script pick it up.
 */
export const SPATIAL_ANALYSIS_MODES: readonly SpatialAnalysisModeDefinition[] = [
  timeMode,
  densityMode,
  lassoMode,
  polygonJoinMode,
  nearestJoinMode,
  zonalMode,
  bufferMode,
  clustersMode,
  flowsMode,
  trajectoriesMode,
  reachabilityMode,
  networkAnalysisMode,
  terrainMode,
  hydrologyMode,
  costDistanceMode,
  rasterZonalMode,
  tileLodMode,
  hotSpotsMode,
  distanceFieldMode,
  rasterJoinMode,
  cellPyramidMode,
  playheadMode,
  accessibilityMode,
  simplificationMode,
  interpolationMode,
  classificationMode,
  groupStatisticsMode,
  cellsMode,
  dotDensityMode,
  spatialWeightsMode,
  pointPatternMode,
  geometryMode,
  regressionMode,
  contoursMode,
  suitabilityMode,
  reliefMode,
  flowFieldMode,
  spaceTimeMode,
  visibilityMode,
  geomorphometryMode,
  reliefVisualizationMode,
  drainageMode,
  terrainFeaturesMode
];
