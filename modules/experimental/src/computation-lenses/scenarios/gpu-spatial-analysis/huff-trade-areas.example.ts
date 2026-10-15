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
  getGPUHuffTradeAreasParameterValues,
  getGPUNeighborSearchParameterValues,
  GPUHuffTradeAreas,
  GPUNeighborSearch,
  GPUParameterBuffer,
  type GPUSpatialContext
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Inputs the lens scenario uploads from the park and tract data files. */
export type HuffTradeAreasExampleInput = {
  /** Park centroids, projected to local-frame metres. */
  parks: GraphDataView<'float32x2'>;
  /** Park area in square kilometres, the attractiveness `A_j`. */
  attractiveness: GraphDataView<'float32'>;
  /** Census-tract centroids, projected to local-frame metres. */
  tractCentroids: GraphDataView<'float32x2'>;
  /** Residents per tract. */
  population: GraphDataView<'float32'>;
  /** Planar context of the local frame. */
  spatialContext: GPUSpatialContext;
  /** Search extent `[minX, minY, maxX, maxY]` in frame metres, covering every point. */
  bounds: readonly [number, number, number, number];
};

/** Huff trade areas of parks: inverse-square distance decay within 2.5 km, park area as attraction. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {
    parks,
    attractiveness,
    tractCentroids,
    population,
    spatialContext,
    bounds
  }: HuffTradeAreasExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const slotCapacity = 8192;
  const createRows = (name: string, rowCount: number) => ({
    offsets: createTransientView(graph, `${name}Offsets`, 'uint32', rowCount + 1, usage),
    neighbors: createTransientView(graph, `${name}Neighbors`, 'uint32', slotCapacity, usage),
    weights: createTransientView(graph, `${name}Weights`, 'float32', slotCapacity, usage),
    status: {overflow: createTransientView(graph, `${name}Overflow`, 'uint32', 1, usage)}
  });
  const tractRows = createRows('tract', tractCentroids.length);
  const parkRows = createRows('park', parks.length);
  const tradeArea = createTransientView(graph, 'tradeArea', 'uint32', tractCentroids.length, usage);
  const tradeAreaProbability = createTransientView(
    graph,
    'tradeAreaProbability',
    'float32',
    tractCentroids.length,
    usage
  );
  const probabilities = createTransientView(graph, 'probabilities', 'float32', slotCapacity, usage);
  const expectedDemand = createTransientView(
    graph,
    'expectedDemand',
    'float32',
    parks.length,
    usage
  );
  const searchParameters = new GPUParameterBuffer(graph.device, {
    id: 'huff-search',
    format: 'float32',
    length: 12,
    values: getGPUNeighborSearchParameterValues({
      bounds,
      radius: 2500,
      weightKind: 'inverseDistance',
      power: 2,
      distanceFloor: 100
    })
  });
  const huffParameters = new GPUParameterBuffer(graph.device, {
    id: 'huff-exponent',
    format: 'float32',
    length: 1,
    values: getGPUHuffTradeAreasParameterValues({exponent: 1})
  });
  const search = {
    mode: 'radius',
    gridSize: [16, 16],
    spatialContext,
    weightTransform: 'raw',
    rowStandardize: false,
    parameters: searchParameters.importToGraph(graph)
  } as const;
  const parksOfTract = new GPUNeighborSearch({
    ...search,
    id: 'parks-of-tract',
    positions: parks,
    queryPositions: tractCentroids,
    output: tractRows
  });
  const tractsOfPark = new GPUNeighborSearch({
    ...search,
    id: 'tracts-of-park',
    positions: tractCentroids,
    queryPositions: parks,
    output: parkRows
  });
  graph.add(parksOfTract);
  graph.add(tractsOfPark);
  // #region example
  // weights: radius searches, max(d, 100 m)^-2 within 2.5 km; attractiveness exponent alpha = 1
  graph.add(
    new GPUHuffTradeAreas({
      attractiveness,
      demand: population,
      parameters: huffParameters.importToGraph(graph),
      demandWeights: parksOfTract.outputWeights,
      facilityWeights: tractsOfPark.outputWeights,
      output: {tradeArea, tradeAreaProbability, probabilities, expectedDemand}
    })
  );
  // tradeAreaProbability colours tracts; expectedDemand is the residents each park captures
  // #endregion example
  return {
    tradeArea,
    tradeAreaProbability,
    probabilities,
    expectedDemand,
    tractOffsets: tractRows.offsets,
    tractNeighbors: tractRows.neighbors,
    tractOverflow: tractRows.status.overflow,
    parkOverflow: parkRows.status.overflow,
    searchParameters,
    huffParameters
  };
}
