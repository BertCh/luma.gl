// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RecipeSceneBuilder} from './recipes-kit';
import {
  buildHotSpotScene,
  buildPeriodComparisonScene,
  buildSpaceTimeScene
} from './recipes-scenes-cells';
import {createFleetDwellScene} from './recipes-scenes-fleet';
import {
  buildClusterAndOutlineScene,
  buildDriveTimeScene,
  buildSpatialRegressionScene,
  buildStraightLineScene
} from './recipes-scenes-spatial';
import {
  buildChangeOfSupportScene,
  createChoroplethScene,
  createRateClusterScene
} from './recipes-scenes-zones';

/** One entry of the recipe select. */
export type RecipeEntry = {
  id: string;
  label: string;
  /** One sentence: what the recipe computes. */
  summary: string;
  build: RecipeSceneBuilder;
};

/** Every recipe scene, in select order. */
export const RECIPE_ENTRIES: readonly RecipeEntry[] = [
  {
    id: 'hot-spot-analysis',
    label: 'addHotSpotAnalysisRecipe',
    summary:
      'Taxi-trip vertices to Quadbin cells to Getis-Ord Gi* hot and cold spots, confirmed by conditional permutation.',
    build: buildHotSpotScene
  },
  {
    id: 'rate-cluster-map',
    label: 'addRateClusterMapRecipe (standardized rate)',
    summary:
      'Event counts and populations per district to empirical-Bayes rates to significant local Moran clusters.',
    build: createRateClusterScene('standardized')
  },
  {
    id: 'rate-cluster-map-smoothed',
    label: 'addRateClusterMapRecipe (smoothed rate)',
    summary:
      'The same chain analysing the empirical-Bayes smoothed rate instead of the standardized one.',
    build: createRateClusterScene('smoothed')
  },
  {
    id: 'points-in-polygons-choropleth',
    label: 'addPointsInPolygonsChoroplethRecipe (zonal)',
    summary:
      'Points of interest counted per district by GPUZonalStatistics, classified and colored on the GPU.',
    build: createChoroplethScene('zonal')
  },
  {
    id: 'points-in-polygons-choropleth-group',
    label: 'addPointsInPolygonsChoroplethRecipe (join + group)',
    summary:
      'Trip vertices joined to districts, mean trip time per district by GPUGroupStatistics, classified and colored.',
    build: createChoroplethScene('group')
  },
  {
    id: 'space-time-hot-spots',
    label: 'addSpaceTimeHotSpotsRecipe',
    summary:
      'Timestamped taxi-trip vertices to a space-time cube to emerging hot and cold spot categories.',
    build: buildSpaceTimeScene
  },
  {
    id: 'period-comparison',
    label: 'addPeriodComparisonRecipe',
    summary:
      'Two time windows of the same trips aggregated into cells, joined, and colored by change.',
    build: buildPeriodComparisonScene
  },
  {
    id: 'cluster-and-outline',
    label: 'addClusterAndOutlineRecipe',
    summary:
      'DBSCAN clusters of taxi pickups and drop-offs, each outlined by its convex hull and measured.',
    build: buildClusterAndOutlineScene
  },
  {
    id: 'spatial-regression',
    label: 'addSpatialRegressionRecipe',
    summary:
      'OLS of points of interest on taxi-trip activity and distance, spatial diagnostics, residual local Moran and geographically weighted fits.',
    build: buildSpatialRegressionScene
  },
  {
    id: 'drive-time-catchment',
    label: 'addDriveTimeCatchmentRecipe',
    summary:
      'Facilities and demand snapped to the street graph, drive times relaxed on the GPU, demand banded by drive time.',
    build: buildDriveTimeScene
  },
  {
    id: 'straight-line-catchments',
    label: 'addStraightLineCatchmentsRecipe',
    summary:
      'Nearest-facility zones of points of interest on a raster, with the trip density each zone holds.',
    build: buildStraightLineScene
  },
  {
    id: 'change-of-support',
    label: 'addChangeOfSupportRecipe',
    summary:
      'Points of interest per irregular source district transferred to a regular target grid by overlap area.',
    build: buildChangeOfSupportScene
  },
  {
    id: 'fleet-dwell',
    label: 'addFleetDwellRecipe (stops)',
    summary: 'Taxi stops detected per trip, joined to zones, with dwell time summarized per zone.',
    build: createFleetDwellScene('stops')
  },
  {
    id: 'fleet-dwell-zone-events',
    label: 'addFleetDwellZoneEventsRecipe',
    summary:
      'Zone enter and exit events per taxi track, dwell per (track, zone), rolled up per zone.',
    build: createFleetDwellScene('zone-events')
  }
];
