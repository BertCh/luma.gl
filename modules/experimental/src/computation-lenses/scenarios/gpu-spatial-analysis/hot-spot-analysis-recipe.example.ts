// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  addHotSpotAnalysisRecipe,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  GPUParameterBuffer
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Inputs the lens scenario uploads from the Chicago North Side observations and areas. */
export type HotSpotAnalysisRecipeExampleInput = {
  /** iNaturalist observations as longitude/latitude degrees. */
  observations: GraphDataView<'float32x2'>;
  /** Study-area polygons (community areas) as longitude/latitude four-level views. */
  studyArea: {
    positions: GraphDataView<'float32x2'>;
    featureOffsets: GraphDataView<'uint32'>;
    polygonOffsets: GraphDataView<'uint32'>;
    ringOffsets: GraphDataView<'uint32'>;
  };
  /** Rows of the study table: an upper bound on the Quadbin cells inside the study area. */
  cellCapacity: number;
  /** Candidate cells the cover tests: an upper bound from the polygon bounding boxes. */
  candidateCapacity: number;
};

/** Getis-Ord Gi* hot and cold spots of observation density on Quadbin cells inside the study area. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {observations, studyArea, cellCapacity, candidateCapacity}: HotSpotAnalysisRecipeExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const output = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    id: string,
    format: Format,
    length: number
  ) => createTransientView(graph, id, format, length, usage);
  const parameters = new GPUParameterBuffer(graph.device, {
    id: 'autocorrelation-parameters',
    format: 'float32',
    length: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
  });
  const zScores = output('zScores', 'float32', cellCapacity);
  const pValues = output('pValues', 'float32', cellCapacity);
  const bins = createTransientView(graph, 'bins', 'sint32', cellCapacity, usage);
  const neighborCounts = output('neighborCounts', 'uint32', cellCapacity);
  const pointsOutsideStudyArea = output('pointsOutsideStudyArea', 'uint32', 1);
  const centers = output('centers', 'float32x2', cellCapacity);
  const values = output('values', 'float32', cellCapacity);
  // #region example
  const hotSpots = addHotSpotAnalysisRecipe(graph, {
    source: {
      kind: 'points',
      positions: observations,
      family: 'quadbin',
      resolution: 16,
      studyArea: {
        kind: 'polygons',
        ...studyArea,
        candidateCapacity,
        coverCapacity: 2 * cellCapacity
      },
      cellCapacity
    },
    parameters,
    settings: {significanceLevel: 0.05},
    outputs: {zScores, pValues, bins, neighborCounts, status: {pointsOutsideStudyArea}},
    scratch: {centers, values}
  });
  // zScores at centers: bins +-1, 2, 3 are 90, 95, 99% confidence hot or cold spots
  // #endregion example
  return {
    hotSpots,
    zScores,
    pValues,
    bins,
    neighborCounts,
    pointsOutsideStudyArea,
    centers,
    values,
    parameters
  };
}
