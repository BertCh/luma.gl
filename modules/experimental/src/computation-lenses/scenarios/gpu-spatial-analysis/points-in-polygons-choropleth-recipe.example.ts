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
  getGPUClassBreaksParameterLength,
  getGPUClassBreaksParameterValues,
  getGPUColorScaleParameterValues,
  GPU_COLOR_SCALE_PARAMETER_LENGTH,
  packGPUColor
} from '@luma.gl/experimental/gpu-dataframe';
import {
  addPointsInPolygonsChoroplethRecipe,
  GPUParameterBuffer,
  type GPUChoroplethPolygons,
  type GPUSpatialContext
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Inputs the lens scenario uploads from the Chicago North Side observations and tracts. */
export type PointsInPolygonsChoroplethRecipeExampleInput = {
  /** iNaturalist observations, projected to local-frame metres. */
  observations: GraphDataView<'float32x2'>;
  /** Census tracts as four-level polygon views in the same frame. */
  tracts: Omit<GPUChoroplethPolygons, 'candidateCapacity'>;
  /** Planar context of the local frame. */
  spatialContext: GPUSpatialContext;
};

/** Observations per tract in five quantile classes. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {observations, tracts, spatialContext}: PointsInPolygonsChoroplethRecipeExampleInput
) {
  const classCount = 5;
  const featureCount = tracts.featureOffsets.length - 1;
  const palette = [
    packGPUColor(237, 248, 233),
    packGPUColor(186, 228, 179),
    packGPUColor(116, 196, 118),
    packGPUColor(49, 163, 84),
    packGPUColor(0, 109, 44)
  ];
  const parameters = [
    new GPUParameterBuffer(graph.device, {
      id: 'class-breaks-parameters',
      format: 'float32',
      length: getGPUClassBreaksParameterLength(classCount),
      values: getGPUClassBreaksParameterValues({method: 'quantile', classCount}, classCount)
    }),
    new GPUParameterBuffer(graph.device, {
      id: 'color-scale-parameters',
      format: 'float32',
      length: GPU_COLOR_SCALE_PARAMETER_LENGTH,
      values: getGPUColorScaleParameterValues({
        scale: 'quantile',
        domainCount: classCount + 1,
        paletteCount: classCount
      })
    }),
    new GPUParameterBuffer(graph.device, {
      id: 'palette',
      format: 'uint32',
      length: classCount,
      values: Uint32Array.from(palette)
    })
  ] as const;
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const output = <Format extends 'float32' | 'uint32'>(
    id: string,
    format: Format,
    length: number
  ) => createTransientView(graph, id, format, length, usage);
  const counts = output('counts', 'uint32', featureCount);
  const classIndices = output('classIndices', 'uint32', featureCount);
  const colors = output('colors', 'uint32', featureCount);
  const breaks = output('breaks', 'float32', classCount + 1);
  const overflow = output('overflow', 'uint32', 1);
  const uncertainCount = output('uncertainCount', 'uint32', 1);
  // #region example
  const choropleth = addPointsInPolygonsChoroplethRecipe(graph, {
    spatialContext,
    points: {kind: 'points', positions: observations},
    polygons: {...tracts, candidateCapacity: 32768},
    statistic: 'count',
    color: {
      classBreaksParameters: parameters[0].importToGraph(graph),
      colorScaleParameters: parameters[1].importToGraph(graph),
      palette: parameters[2].importToGraph(graph),
      methods: ['quantile'],
      maximumClassCount: classCount,
      maximumPaletteCount: classCount
    },
    outputs: {
      counts,
      color: {breaks, colors, classIndices, classCount: output('classCount', 'uint32', 1)},
      status: {overflow, uncertainCount}
    }
  });
  // counts colour each tract; classIndices holds its quantile class
  // #endregion example
  return {choropleth, counts, classIndices, colors, breaks, overflow, uncertainCount, parameters};
}
