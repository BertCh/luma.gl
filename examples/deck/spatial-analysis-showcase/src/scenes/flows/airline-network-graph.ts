// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUData, GPUVector} from '@luma.gl/gpgpu/gpu-data';
import {
  GPUGraph,
  GPUGraphCoreNumber,
  GPUGraphDegree,
  GPUGraphLabelPropagation,
  GPUGraphModularity,
  GPUGraphModularityOptimization,
  GPUGraphPageRank,
  GPUGraphTopology,
  type GPUGraphAdjacency
} from '@luma.gl/gpgpu/gpu-graph';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';

const SCALAR_BYTES = 4;

type ScalarFormat = 'uint32' | 'float32';

/**
 * Creates the caller-owned `GPUVector`s that `@luma.gl/gpgpu/gpu-graph` reads and writes, and
 * destroys them (vectors first, then their buffers). The contributors never allocate: every
 * output of every graph algorithm lives in a vector made here.
 */
export class GraphVectorFactory {
  private readonly buffers: Buffer[] = [];
  private readonly vectors: GPUVector[] = [];

  constructor(
    private readonly device: Device,
    private readonly prefix: string
  ) {}

  /** A packed scalar column, zero filled or initialised from `values`. */
  scalar<Format extends ScalarFormat>(
    name: string,
    format: Format,
    length: number,
    values?: Uint32Array | Float32Array
  ): GPUVector<Format> {
    const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
    const buffer = this.device.createBuffer({
      id: `${this.prefix}-${name}`,
      usage,
      ...(values && values.length > 0
        ? {data: values}
        : {byteLength: Math.max(length, 1) * SCALAR_BYTES})
    });
    this.buffers.push(buffer);
    const vector = new GPUVector<Format>({
      type: 'buffer',
      name,
      format,
      buffer,
      length,
      ownsBuffer: false
    });
    this.vectors.push(vector);
    return vector;
  }

  /** Two-component simulation state; `additionalUsage` makes the same allocation a vertex buffer. */
  coordinates(name: string, values: Float32Array, additionalUsage = 0): GPUVector<'float32x2'> {
    const buffer = this.device.createBuffer({
      id: `${this.prefix}-${name}`,
      data: values,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST | additionalUsage
    });
    this.buffers.push(buffer);
    const vector = new GPUVector<'float32x2'>({
      type: 'buffer',
      name,
      format: 'float32x2',
      buffer,
      length: values.length / 2,
      ownsBuffer: false
    });
    this.vectors.push(vector);
    return vector;
  }

  /** A single-chunk edge column; `GPUGraph` borrows the chunk and never copies it. */
  edgeColumn(name: string, values: Uint32Array): GPUVector<'uint32'> {
    const buffer = this.device.createBuffer({
      id: `${this.prefix}-${name}`,
      data: values,
      usage: Buffer.STORAGE | Buffer.COPY_DST
    });
    this.buffers.push(buffer);
    const data = new GPUData<'uint32'>({
      buffer,
      format: 'uint32',
      length: values.length,
      ownsBuffer: false
    });
    const vector = new GPUVector<'uint32'>({
      type: 'data',
      name,
      format: 'uint32',
      data: [data],
      ownsData: false
    });
    this.vectors.push(vector);
    return vector;
  }

  /** Offsets, neighbors, edge ids and the count and overflow rows of one CSR direction. */
  adjacency(name: string, vertexCount: number, capacity: number): GPUGraphAdjacency {
    return {
      offsets: this.scalar(`${name}-offsets`, 'uint32', vertexCount + 1),
      neighbors: this.scalar(`${name}-neighbors`, 'uint32', capacity),
      edgeIds: this.scalar(`${name}-edge-ids`, 'uint32', capacity),
      count: this.scalar(`${name}-count`, 'uint32', 1),
      overflow: this.scalar(`${name}-overflow`, 'uint32', 1)
    };
  }

  /** The physical buffer behind a single-chunk vector. */
  getBuffer(vector: GPUVector): Buffer {
    return vector.data[0].buffer as Buffer;
  }

  destroy(): void {
    for (const vector of this.vectors) vector.destroy();
    for (const buffer of this.buffers) buffer.destroy();
    this.vectors.length = 0;
    this.buffers.length = 0;
  }
}

/** The graph, its CSR and every persistent analysis output of the airline network. */
export type NetworkStack = {
  factory: GraphVectorFactory;
  vertexCount: number;
  edgeCount: number;
  graph: GPUGraph;
  topology: GPUGraphTopology;
  forwardOverflow: GPUVector<'uint32'>;
  degree: GPUVector<'uint32'>;
  pageRank: GPUVector<'float32'>;
  pageRankResidual: GPUVector<'float32'>;
  coreNumber: GPUVector<'uint32'>;
  coreConverged: GPUVector<'uint32'>;
  degeneracy: GPUVector<'uint32'>;
  propagation: GPUVector<'uint32'>;
  propagationConverged: GPUVector<'uint32'>;
  /** Continent index per airport, the partition the communities are compared with. */
  continents: GPUVector<'uint32'>;
  propagationModularity: GPUVector<'float32'>;
  continentModularity: GPUVector<'float32'>;
  propagationValid: GPUVector<'uint32'>;
  continentValid: GPUVector<'uint32'>;
};

/**
 * Describes the OpenFlights pairs as an undirected `GPUGraph`, allocates symmetric forward CSR
 * (every pair appears twice, so the neighbor capacity is twice the pair count) and every output
 * vector. Nothing is computed yet: {@link buildAnalysisGraph} declares the work.
 */
export function createNetworkStack(
  factory: GraphVectorFactory,
  vertexCount: number,
  source: Uint32Array,
  target: Uint32Array,
  continent: Uint8Array
): NetworkStack {
  const edgeCount = source.length;
  const graph = new GPUGraph({
    vertexCount,
    sourceVertices: factory.edgeColumn('sources', source),
    targetVertices: factory.edgeColumn('targets', target),
    directed: false
  });
  const forward = factory.adjacency('forward', vertexCount, edgeCount * 2);
  const topology = new GPUGraphTopology({
    id: 'airline-topology',
    graph,
    forward,
    invalidEdgeCount: factory.scalar('invalid-edges', 'uint32', 1)
  });
  return {
    factory,
    vertexCount,
    edgeCount,
    graph,
    topology,
    forwardOverflow: forward.overflow,
    degree: factory.scalar('degree', 'uint32', vertexCount),
    pageRank: factory.scalar('page-rank', 'float32', vertexCount),
    pageRankResidual: factory.scalar('page-rank-residual', 'float32', 1),
    coreNumber: factory.scalar('core-number', 'uint32', vertexCount),
    coreConverged: factory.scalar('core-converged', 'uint32', 1),
    degeneracy: factory.scalar('degeneracy', 'uint32', 1),
    propagation: factory.scalar('propagation', 'uint32', vertexCount),
    propagationConverged: factory.scalar('propagation-converged', 'uint32', 1),
    continents: factory.scalar('continents', 'uint32', vertexCount, Uint32Array.from(continent)),
    propagationModularity: factory.scalar('propagation-modularity', 'float32', 1),
    continentModularity: factory.scalar('continent-modularity', 'float32', 1),
    propagationValid: factory.scalar('propagation-valid', 'uint32', 1),
    continentValid: factory.scalar('continent-valid', 'uint32', 1)
  };
}

/** Compile-time settings of the analysis graph. */
export type AnalysisSettings = {
  pageRankDamping: number;
  pageRankIterations: number;
  propagationRounds: number;
};

/**
 * One command graph: CSR rebuild, degree, PageRank, core numbers, label propagation, and the
 * modularity of the propagation and of the continent partitions. Damping, iteration counts and
 * round counts are baked into the shaders, so changing them builds a new graph.
 */
export function buildAnalysisGraph(
  device: Device,
  stack: NetworkStack,
  settings: AnalysisSettings,
  id: string
): CompiledGPUCommandGraph<void> {
  const {topology} = stack;
  const graph = new GPUCommandGraph<void>(device, {id});
  topology.addToGraph(graph);
  new GPUGraphDegree({id: 'airline-degree', topology, output: stack.degree}).addToGraph(graph);
  new GPUGraphPageRank({
    id: 'airline-page-rank',
    topology,
    output: stack.pageRank,
    damping: settings.pageRankDamping,
    iterations: settings.pageRankIterations,
    residual: stack.pageRankResidual
  }).addToGraph(graph);
  new GPUGraphCoreNumber({
    id: 'airline-core',
    topology,
    output: stack.coreNumber,
    iterations: 32,
    converged: stack.coreConverged,
    degeneracy: stack.degeneracy
  }).addToGraph(graph);
  new GPUGraphLabelPropagation({
    id: 'airline-propagation',
    topology,
    output: stack.propagation,
    iterations: settings.propagationRounds,
    converged: stack.propagationConverged
  }).addToGraph(graph);
  new GPUGraphModularity({
    id: 'airline-modularity-propagation',
    graph: stack.graph,
    communities: stack.propagation,
    output: stack.propagationModularity,
    resolution: 1,
    valid: stack.propagationValid
  }).addToGraph(graph);
  new GPUGraphModularity({
    id: 'airline-modularity-continents',
    graph: stack.graph,
    communities: stack.continents,
    output: stack.continentModularity,
    resolution: 1,
    valid: stack.continentValid
  }).addToGraph(graph);
  return graph.compile();
}

/** Byte layout of the analysis summary: one block of per-airport columns, then the scalars. */
export function getAnalysisSources(stack: NetworkStack): {buffer: Buffer; size: number}[] {
  const {factory, vertexCount} = stack;
  const column = (vector: GPUVector) => ({
    buffer: factory.getBuffer(vector),
    size: vertexCount * SCALAR_BYTES
  });
  const scalar = (vector: GPUVector) => ({buffer: factory.getBuffer(vector), size: SCALAR_BYTES});
  return [
    column(stack.pageRank),
    column(stack.degree),
    column(stack.coreNumber),
    column(stack.propagation),
    scalar(stack.pageRankResidual),
    scalar(stack.degeneracy),
    scalar(stack.coreConverged),
    scalar(stack.propagationConverged),
    scalar(stack.propagationModularity),
    scalar(stack.continentModularity),
    scalar(stack.propagationValid),
    scalar(stack.continentValid),
    scalar(stack.forwardOverflow)
  ];
}

/** Decoded analysis summary. */
export type AnalysisSummary = {
  pageRank: Float32Array;
  degree: Uint32Array;
  coreNumber: Uint32Array;
  propagation: Uint32Array;
  residual: number;
  degeneracy: number;
  coreConverged: boolean;
  propagationConverged: boolean;
  propagationModularity: number;
  continentModularity: number;
  valid: boolean;
  adjacencyOverflow: boolean;
};

/** Decodes the bytes read with {@link getAnalysisSources}. */
export function decodeAnalysis(bytes: ArrayBuffer, vertexCount: number): AnalysisSummary {
  const columnBytes = vertexCount * SCALAR_BYTES;
  const pageRank = new Float32Array(bytes, 0, vertexCount);
  const degree = new Uint32Array(bytes, columnBytes, vertexCount);
  const coreNumber = new Uint32Array(bytes, columnBytes * 2, vertexCount);
  const propagation = new Uint32Array(bytes, columnBytes * 3, vertexCount);
  const floats = new Float32Array(bytes, columnBytes * 4);
  const words = new Uint32Array(bytes, columnBytes * 4);
  return {
    pageRank,
    degree,
    coreNumber,
    propagation,
    residual: floats[0],
    degeneracy: words[1],
    coreConverged: words[2] === 1,
    propagationConverged: words[3] === 1,
    propagationModularity: floats[4],
    continentModularity: floats[5],
    valid: words[6] === 1 && words[7] === 1,
    adjacencyOverflow: words[8] !== 0
  };
}

/** Settings of one modularity-optimization run. */
export type OptimizationSettings = {
  resolution: number;
  rounds: number;
  minimumGain: number;
};

/** One compiled optimization run with its own outputs and readback. */
export type OptimizationRun = {
  resources: SpatialAnalysisResources;
  compiled: CompiledGPUCommandGraph<void>;
  reader: SummaryReader;
  settings: OptimizationSettings;
  /** True until the run has been encoded once. */
  pending: boolean;
};

/** What an optimization run reports. */
export type OptimizationResult = {
  labels: Uint32Array;
  modularity: number;
  converged: boolean;
  valid: boolean;
  /** Modularity of the label-propagation partition at the same resolution. */
  propagationModularity: number;
  /** Modularity of the continent partition at the same resolution. */
  continentModularity: number;
};

/**
 * Starts from the label-propagation partition and improves it with `GPUGraphModularityOptimization`
 * (single-level local moving, one move per round), then scores the propagation and the continent
 * partitions at the same resolution so all three numbers are comparable. The resolution is a
 * shader constant, so each value is its own compiled graph.
 */
export function buildOptimizationRun(
  device: Device,
  stack: NetworkStack,
  settings: OptimizationSettings,
  id: string,
  onResult: (result: OptimizationResult) => void
): OptimizationRun {
  const resources = new SpatialAnalysisResources(device, id);
  const factory = resources.track(new GraphVectorFactory(device, id));
  const {vertexCount} = stack;
  const labels = factory.scalar('labels', 'uint32', vertexCount);
  const modularity = factory.scalar('modularity', 'float32', 1);
  const converged = factory.scalar('converged', 'uint32', 1);
  const valid = factory.scalar('valid', 'uint32', 1);
  const propagationScore = factory.scalar('propagation-score', 'float32', 1);
  const continentScore = factory.scalar('continent-score', 'float32', 1);
  const graph = new GPUCommandGraph<void>(device, {id});
  new GPUGraphModularityOptimization({
    id: 'airline-optimization',
    topology: stack.topology,
    output: labels,
    modularity,
    initialCommunities: stack.propagation,
    resolution: settings.resolution,
    iterations: settings.rounds,
    minimumGain: settings.minimumGain,
    converged,
    valid
  }).addToGraph(graph);
  new GPUGraphModularity({
    id: 'airline-score-propagation',
    graph: stack.graph,
    communities: stack.propagation,
    output: propagationScore,
    resolution: settings.resolution
  }).addToGraph(graph);
  new GPUGraphModularity({
    id: 'airline-score-continents',
    graph: stack.graph,
    communities: stack.continents,
    output: continentScore,
    resolution: settings.resolution
  }).addToGraph(graph);
  const compiled = resources.track(graph.compile());
  const columnBytes = vertexCount * SCALAR_BYTES;
  const reader = new SummaryReader(
    resources,
    id,
    [
      {buffer: factory.getBuffer(labels), size: columnBytes},
      {buffer: factory.getBuffer(modularity), size: SCALAR_BYTES},
      {buffer: factory.getBuffer(converged), size: SCALAR_BYTES},
      {buffer: factory.getBuffer(valid), size: SCALAR_BYTES},
      {buffer: factory.getBuffer(propagationScore), size: SCALAR_BYTES},
      {buffer: factory.getBuffer(continentScore), size: SCALAR_BYTES}
    ],
    bytes => {
      const floats = new Float32Array(bytes, columnBytes);
      const words = new Uint32Array(bytes, columnBytes);
      onResult({
        labels: new Uint32Array(bytes.slice(0, columnBytes)),
        modularity: floats[0],
        converged: words[1] === 1,
        valid: words[2] === 1,
        propagationModularity: floats[3],
        continentModularity: floats[4]
      });
    }
  );
  return {resources, compiled, reader, settings, pending: true};
}
