// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** A group of scenes in the gallery. Scene builders pick one by `id` and never edit this file. */
export type Chapter = {
  id: string;
  title: string;
  /** One sentence shown under the chapter heading. */
  summary: string;
};

/** Chapters in display order. */
export const CHAPTERS: readonly Chapter[] = [
  {
    id: 'points',
    title: 'Points & density',
    summary: 'Binning, smoothing and nearest-neighbour questions about where things are.'
  },
  {
    id: 'joins',
    title: 'Joins & overlay',
    summary: 'Point-in-polygon, nearest and raster joins that attach one layer to another.'
  },
  {
    id: 'geometry',
    title: 'Geometry & topology',
    summary: 'Validity, intersections, simplification and the utilities that clean vector data.'
  },
  {
    id: 'weights',
    title: 'Spatial weights & autocorrelation',
    summary: 'Who counts as a neighbour, and whether similar values cluster in space.'
  },
  {
    id: 'statistics',
    title: 'Rates, classes & segregation',
    summary: 'Smoothed rates, classification schemes and indices of inequality and segregation.'
  },
  {
    id: 'regression',
    title: 'Regression & regionalization',
    summary: 'Spatial regression, geographically weighted models and constrained clustering.'
  },
  {
    id: 'interpolation',
    title: 'Interpolation & change of support',
    summary: 'Turning samples into surfaces and moving data between incompatible zones.'
  },
  {
    id: 'cells',
    title: 'Cells & DGGS',
    summary: 'Discrete global grids such as H3 and A5 as a common index for analysis.'
  },
  {
    id: 'networks',
    title: 'Networks & accessibility',
    summary: 'Shortest paths, service areas and who can reach what within a travel budget.'
  },
  {
    id: 'flows',
    title: 'Flows & graphs',
    summary: 'Origin-destination flows, graph metrics, layouts and bundling.'
  },
  {
    id: 'movement',
    title: 'Trajectories & movement',
    summary:
      'Ships, birds and aircraft over time: playback, corridors, stops, dwell and encounters.'
  },
  {
    id: 'earth',
    title: 'Oceans, storms & orbits',
    summary:
      'Drifters against a current model, hurricane tracks, severe storms and satellite ground tracks.'
  },
  {
    id: 'time',
    title: 'Space-time',
    summary: 'Statistics and tests that treat time as a dimension of the map.'
  },
  {
    id: 'terrain',
    title: 'Terrain',
    summary: 'Slope, aspect, viewsheds, relief shading and landform analysis on elevation rasters.'
  },
  {
    id: 'hydrology',
    title: 'Hydrology & cost surfaces',
    summary: 'Flow direction, catchments, flooding and least-cost movement across a surface.'
  },
  {
    id: 'raster',
    title: 'Raster analysis',
    summary: 'Zonal statistics, reclassification, overlays, isolines and change detection.'
  },
  {
    id: 'dataframe',
    title: 'Tables & time series',
    summary: 'Column profiles, group statistics, key joins and temporal reductions on the GPU.'
  }
];

/** Returns a chapter by id. */
export function getChapter(id: string): Chapter | undefined {
  return CHAPTERS.find(chapter => chapter.id === id);
}
