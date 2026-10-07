// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, CommandEncoder, Device} from '@luma.gl/core';
import {
  getGPUNeighborSearchParameterValues,
  GPUContiguityWeights,
  GPUNeighborSearch,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  GPUSpatialWeightsTransform,
  type GPUNeighborSearchKernel,
  type GPUNeighborSearchWeightKind,
  type GPUSpatialWeightsKernel
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import type {SpatialAnalysisResources} from '../../engine/resources';
import type {Geography} from './b4-geography';

/** How neighbours of a polygon are defined. */
export type WeightsSource = 'queen' | 'rook' | 'knn' | 'band';

/** Transform applied in place to the produced weights. */
export type WeightsTransform =
  | 'none'
  | 'row'
  | 'binary'
  | 'double'
  | 'variance'
  | 'symmetrize'
  | 'kernel';

/** Everything the producer and the transform need. Buffer-write fields never recompile. */
export type WeightsConfig = {
  source: WeightsSource;
  /** Neighbours per row of `knn`. Compile-time. */
  k: number;
  /** Vertex snap tolerance of contiguity in metres, 0 for exact. Compile-time. */
  snapTolerance: number;
  /** Distance band in multiples of the median nearest-centroid distance. */
  bandFactor: number;
  /** Maximum kNN distance in the same unit; 0 for unbounded. */
  knnCapFactor: number;
  weightKind: GPUNeighborSearchWeightKind;
  kernel: GPUNeighborSearchKernel;
  power: number;
  distanceFloor: number;
  /** Row standardisation inside the search (a parameter). */
  rowStandardize: boolean;
  transform: WeightsTransform;
  /** Kernel of the `kernel` transform. Compile-time. */
  transformKernel: GPUSpatialWeightsKernel;
  /** Bandwidth of the `kernel` transform in median spacings, 0 for adaptive. Compile-time. */
  bandwidthFactor: number;
  /** `double` standardisation total. Compile-time. */
  doubleSum: 'one' | 'rows';
};

/** A weights CSR with optional distances. */
export type WeightsCsr = {
  offsets: Buffer;
  neighbors: Buffer;
  weights: Buffer;
  distances: Buffer;
};

/** Producer and transform graphs of one geography. */
export type WeightsCore = {
  csr: WeightsCsr;
  positions: Buffer;
  /** Slot capacity of `csr`. */
  slots: number;
  overflow: Buffer;
  total: Buffer;
  /** Compiles any graph the configuration needs (no-op when cached). Returns true when it compiled. */
  prepare: (config: WeightsConfig) => boolean;
  /** Writes the parameter buffers of the search. */
  writeParameters: (config: WeightsConfig) => void;
  /** Encodes the producer and then the transform. */
  encode: (commandEncoder: CommandEncoder, config: WeightsConfig) => void;
  /** Every compiled graph so far. */
  getGraphs: () => CompiledGPUCommandGraph<never>[];
  /** Distance band in metres for a factor. */
  getBandMeters: (config: WeightsConfig) => number;
  /** True when the config's source writes distances (so the kernel transform applies). */
  hasDistances: (config: WeightsConfig) => boolean;
  /** Creates an independent kNN neighbour list (binary weights) for set algebra. */
  createPartner: (name: string) => {
    csr: WeightsCsr;
    slots: number;
    prepare: (k: number) => boolean;
    encode: (commandEncoder: CommandEncoder, k: number) => void;
  };
};

/** Neighbour slots per row the CSR buffers hold. */
export const SLOTS_PER_ROW = 24;
/** Lattice of the neighbour search. Results never depend on it. */
const SEARCH_GRID: readonly [number, number] = [256, 256];

/**
 * Creates the weights producers of one geography. Contiguity (`GPUContiguityWeights`) works on the
 * polygon vertices; kNN and distance band (`GPUNeighborSearch`) work on centroids. Every variant
 * writes the same CSR, so the analysis never cares where it came from. Variants are compiled the
 * first time a configuration needs them and cached, which keeps start-up fast and makes the
 * "rebuilds" badge of a compile-time option truthful.
 */
export function createWeightsCore(props: {
  device: Device;
  resources: SpatialAnalysisResources;
  id: string;
  geography: Geography;
}): WeightsCore {
  const {device, resources, id, geography} = props;
  const rowCount = geography.count;
  const slots = rowCount * SLOTS_PER_ROW;
  const create = (name: string, data: number | Float32Array | Uint32Array) =>
    resources.createBuffer(`${id}-${name}`, data);

  const positions = create('centroids', geography.centroids);
  const verticesBuffer = create('contiguity-vertices', geography.contiguityVertices);
  const ringOffsetsBuffer = create('contiguity-ring-offsets', geography.contiguityRingOffsets);
  const polygonOffsetsBuffer = create('feature-ring-offsets', geography.featureRingOffsets);
  const csr: WeightsCsr = {
    offsets: create('offsets', (rowCount + 1) * 4),
    neighbors: create('neighbors', slots * 4),
    weights: create('weights', slots * 4),
    distances: create('distances', slots * 4)
  };
  const overflow = create('overflow', 4);
  const total = create('total', 4);
  const scratch = create('transform-scratch', slots * 4);
  const searchParameters = resources.createParameterBuffer(
    'search-parameters',
    'float32',
    GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH
  );
  const margin = Math.max(1000, geography.medianSpacing);
  const searchBounds: [number, number, number, number] = [
    geography.bounds[0] - margin,
    geography.bounds[1] - margin,
    geography.bounds[2] + margin,
    geography.bounds[3] + margin
  ];

  const compiled = new Map<string, CompiledGPUCommandGraph<void>>();
  const getOrCompile = (key: string, build: () => GPUCommandGraph<void>): boolean => {
    if (compiled.has(key)) return false;
    compiled.set(key, resources.track(build().compile()));
    return true;
  };

  const importCsr = (
    graph: GPUCommandGraph<void>,
    target: WeightsCsr,
    withDistances: boolean,
    capacity = slots
  ) => ({
    offsets: importGraphBuffer(graph, 'offsets', target.offsets, 'uint32', rowCount + 1),
    neighbors: importGraphBuffer(graph, 'neighbors', target.neighbors, 'uint32', capacity),
    weights: importGraphBuffer(graph, 'weights', target.weights, 'float32', capacity),
    ...(withDistances
      ? {distances: importGraphBuffer(graph, 'distances', target.distances, 'float32', capacity)}
      : {})
  });

  const buildContiguity = (criterion: 'queen' | 'rook', snapTolerance: number) => {
    const graph = new GPUCommandGraph<void>(device, {
      id: `${id}-${criterion}-snap${snapTolerance}`
    });
    graph.add(
      new GPUContiguityWeights({
        id: 'contiguity',
        criterion,
        snapTolerance: snapTolerance > 0 ? snapTolerance : undefined,
        positions: importGraphBuffer(
          graph,
          'vertices',
          verticesBuffer,
          'float32x2',
          geography.contiguityVertices.length / 2
        ),
        ringOffsets: importGraphBuffer(
          graph,
          'ring-offsets',
          ringOffsetsBuffer,
          'uint32',
          geography.contiguityRingOffsets.length
        ),
        polygonOffsets: importGraphBuffer(
          graph,
          'polygon-offsets',
          polygonOffsetsBuffer,
          'uint32',
          rowCount + 1
        ),
        weights: importCsr(graph, csr, false),
        overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1),
        totalNeighbors: importGraphBuffer(graph, 'total', total, 'uint32', 1)
      })
    );
    return graph;
  };

  const buildSearch = (mode: 'knn' | 'radius', k: number) => {
    const graph = new GPUCommandGraph<void>(device, {id: `${id}-search-${mode}-${k}`});
    graph.add(
      new GPUNeighborSearch({
        id: 'neighbor-search',
        mode,
        k: mode === 'knn' ? k : undefined,
        gridSize: SEARCH_GRID,
        positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', rowCount),
        parameters: searchParameters.importToGraph(graph),
        weights: importCsr(graph, csr, true),
        overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1),
        totalNeighbors: importGraphBuffer(graph, 'total', total, 'uint32', 1)
      })
    );
    return graph;
  };

  const getProducerKey = (config: WeightsConfig): string =>
    config.source === 'knn'
      ? `search-knn-${config.k}`
      : config.source === 'band'
        ? 'search-radius'
        : `${config.source}-${config.snapTolerance}`;

  const getTransformKey = (config: WeightsConfig): string | null => {
    if (config.transform === 'none') return null;
    if (config.transform === 'kernel') {
      return `kernel-${config.transformKernel}-${config.bandwidthFactor}`;
    }
    if (config.transform === 'double') return `double-${config.doubleSum}`;
    return config.transform;
  };

  const hasDistances = (config: WeightsConfig) =>
    config.source === 'knn' || config.source === 'band';

  const getBandMeters = (config: WeightsConfig) => config.bandFactor * geography.medianSpacing;

  const buildTransform = (config: WeightsConfig) => {
    const operation = config.transform as Exclude<WeightsTransform, 'none'>;
    const key = getTransformKey(config)!;
    const graph = new GPUCommandGraph<void>(device, {id: `${id}-transform-${key}`});
    const withDistances = operation === 'kernel';
    graph.add(
      new GPUSpatialWeightsTransform({
        id: `transform-${operation}`,
        operation,
        weights: importCsr(graph, csr, withDistances),
        output:
          operation === 'symmetrize'
            ? importGraphBuffer(graph, 'scratch', scratch, 'float32', slots)
            : undefined,
        kernel: operation === 'kernel' ? config.transformKernel : undefined,
        bandwidth:
          operation === 'kernel'
            ? config.bandwidthFactor > 0
              ? config.bandwidthFactor * geography.medianSpacing
              : 'adaptive'
            : undefined,
        doubleSum: operation === 'double' ? config.doubleSum : undefined
      })
    );
    return graph;
  };

  const effectiveTransform = (config: WeightsConfig): WeightsConfig =>
    config.transform === 'kernel' && !hasDistances(config)
      ? {...config, transform: 'none'}
      : config;

  return {
    csr,
    positions,
    slots,
    overflow,
    total,
    hasDistances,
    getBandMeters,

    prepare(rawConfig) {
      const config = effectiveTransform(rawConfig);
      let didCompile = false;
      const producerKey = getProducerKey(config);
      didCompile =
        getOrCompile(producerKey, () =>
          config.source === 'knn'
            ? buildSearch('knn', config.k)
            : config.source === 'band'
              ? buildSearch('radius', 1)
              : buildContiguity(config.source, config.snapTolerance)
        ) || didCompile;
      const transformKey = getTransformKey(config);
      if (transformKey) {
        didCompile = getOrCompile(transformKey, () => buildTransform(config)) || didCompile;
      }
      return didCompile;
    },

    writeParameters(config) {
      const spacing = geography.medianSpacing;
      searchParameters.write(
        getGPUNeighborSearchParameterValues({
          bounds: searchBounds,
          radius:
            config.source === 'band'
              ? config.bandFactor * spacing
              : config.knnCapFactor > 0
                ? config.knnCapFactor * spacing
                : undefined,
          weightKind: config.weightKind,
          kernel: config.kernel,
          power: config.power,
          distanceFloor: config.distanceFloor,
          rowStandardize: config.rowStandardize
        })
      );
    },

    encode(commandEncoder, rawConfig) {
      const config = effectiveTransform(rawConfig);
      compiled.get(getProducerKey(config))!.encode(commandEncoder, {parameters: undefined});
      const transformKey = getTransformKey(config);
      if (transformKey) {
        compiled.get(transformKey)!.encode(commandEncoder, {parameters: undefined});
        if (config.transform === 'symmetrize') {
          commandEncoder.copyBufferToBuffer({
            sourceBuffer: scratch,
            destinationBuffer: csr.weights,
            size: slots * 4
          });
        }
      }
    },

    getGraphs: () => [...compiled.values()] as CompiledGPUCommandGraph<never>[],

    createPartner(name) {
      const partnerCsr: WeightsCsr = {
        offsets: create(`${name}-offsets`, (rowCount + 1) * 4),
        neighbors: create(`${name}-neighbors`, slots * 4),
        weights: create(`${name}-weights`, slots * 4),
        distances: create(`${name}-distances`, slots * 4)
      };
      const partnerOverflow = create(`${name}-overflow`, 4);
      const partnerParameters = resources.createParameterBuffer(
        `${name}-parameters`,
        'float32',
        GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
        getGPUNeighborSearchParameterValues({bounds: searchBounds, weightKind: 'binary'})
      );
      const build = (k: number) => {
        const graph = new GPUCommandGraph<void>(device, {id: `${id}-${name}-knn-${k}`});
        graph.add(
          new GPUNeighborSearch({
            id: `${name}-search`,
            mode: 'knn',
            k,
            gridSize: SEARCH_GRID,
            positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', rowCount),
            parameters: partnerParameters.importToGraph(graph),
            weights: importCsr(graph, partnerCsr, false),
            overflow: importGraphBuffer(graph, 'overflow', partnerOverflow, 'uint32', 1)
          })
        );
        return graph;
      };
      return {
        csr: partnerCsr,
        slots,
        prepare: (k: number) => getOrCompile(`${name}-knn-${k}`, () => build(k)),
        encode: (commandEncoder: CommandEncoder, k: number) =>
          compiled.get(`${name}-knn-${k}`)!.encode(commandEncoder, {parameters: undefined})
      };
    }
  };
}
