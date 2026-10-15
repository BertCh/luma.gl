// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, Buffer, type Device} from '@luma.gl/core';
import {
  createGPUComputationRecord,
  getGPUComputationAdapter,
  validateGPUComputationRecord,
  validateGPUComputationSidecars,
  type CreateGPUComputationRecordProps,
  type GPUComputationCapture,
  type GPUComputationKeySpace,
  type GPUComputationRelation,
  type GPUComputationRecordWithSidecars
} from '@luma.gl/gpgpu';
import {
  GPUCommandGraph,
  GraphDataView,
  type GPUCommandNode,
  type GraphBufferDescriptor,
  type GraphBufferHandle
} from '@luma.gl/gpgpu/gpu-core';
import roads from '../../data/n-chicago-loop-roads.json';

/** Shared, versioned provenance of the deterministic Chicago Loop extract. */
export const NETWORK_DATASET = {
  id: 'n-chicago-loop-roads',
  licence: 'ODbL 1.0',
  source: 'OpenStreetMap via BBBike Chicago extract'
};

/** Prepared network data, with facility snapping performed by the extraction script. */
export function loadChicagoLoopRoads() {
  return roads;
}

/** Shared key spaces; scenario-specific sizes can replace these defaults. */
export const NETWORK_KEY_SPACES: readonly GPUComputationKeySpace[] = [
  {id: 'node', label: 'network node', indexSpace: '1d', size: roads.nodeCount},
  {id: 'edge', label: 'directed edge', indexSpace: 'csr', size: roads.edgeCount},
  {id: 'facility', label: 'facility', indexSpace: '1d', size: roads.facilities.length},
  {id: 'frontier-slot', label: 'frontier slot', indexSpace: '1d', size: 2 * roads.nodeCount}
];

/** Constructs an explicit observation graph with retained transients. */
export function createNetworkLensGraph(device: Device, id: string, retain: readonly string[]) {
  return new NetworkLensGraph(device, {id, recording: {retain: [...retain]}});
}

/** Uploads a packed, caller-owned input with copy access for honest geometry readback. */
export function uploadNetworkView<Format extends 'float32' | 'float32x2' | 'uint32'>(
  device: Device,
  graph: GPUCommandGraph,
  id: string,
  values: Float32Array | Uint32Array,
  format: Format
) {
  const buffer = device.createBuffer({
    id,
    data: values,
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  const handle = graph.importBuffer(
    {id, byteLength: buffer.byteLength, usage: buffer.usage},
    buffer
  );
  return {
    buffer,
    view: graph.createDataView(handle, {
      format,
      length: values.length / (format === 'float32x2' ? 2 : 1)
    })
  };
}

/** Uploads CSR, chosen impedance and native positions; caller destroys ownedBuffers. */
export function uploadNetworkTopology(
  device: Device,
  graph: GPUCommandGraph,
  data: ReturnType<typeof loadChicagoLoopRoads>,
  options: {weights: 'travelTimeSeconds' | 'lengthMeters'}
) {
  const offsets = uploadNetworkView(
    device,
    graph,
    `${graph.id}-offsets`,
    Uint32Array.from(data.offsets),
    'uint32'
  );
  const neighbors = uploadNetworkView(
    device,
    graph,
    `${graph.id}-neighbors`,
    Uint32Array.from(data.neighbors),
    'uint32'
  );
  const weights = uploadNetworkView(
    device,
    graph,
    `${graph.id}-weights`,
    Float32Array.from(data[options.weights]),
    'float32'
  );
  const nodePositions = uploadNetworkView(
    device,
    graph,
    `${graph.id}-node-positions`,
    Float32Array.from(data.nodes),
    'float32x2'
  );
  return {
    offsets: offsets.view,
    neighbors: neighbors.view,
    weights: weights.view,
    nodePositions: nodePositions.view,
    ownedBuffers: [offsets.buffer, neighbors.buffer, weights.buffer, nodePositions.buffer]
  };
}

/** Declares static CSR and an intentionally uncaptured, time-dependent queue relation. */
export function createNetworkRelations(
  topology: ReturnType<typeof uploadNetworkTopology>,
  queuesResource?: string
): GPUComputationRelation[] {
  return [
    {
      id: 'node-out-edges',
      from: 'node',
      to: 'node',
      cardinality: '1:N',
      kind: 'buffer',
      timeDependent: false,
      ref: {
        layout: 'csr',
        resource: topology.neighbors.buffer.id,
        offsetsResource: topology.offsets.buffer.id,
        capture: 'neighbors'
      }
    },
    ...(queuesResource
      ? [
          {
            id: 'frontier-queue',
            from: 'frontier-slot',
            to: 'node',
            cardinality: 'N:1' as const,
            kind: 'buffer' as const,
            timeDependent: true,
            ref: {layout: 'index' as const, resource: queuesResource}
          }
        ]
      : [])
  ];
}

/** Finds a declared scratch resource through public command-node uses, without private graph access. */
export function findNetworkResource(
  nodes: readonly GPUCommandNode[],
  id: string
): GraphBufferHandle {
  for (const node of nodes)
    for (const resource of node.resources ?? []) {
      if (!('buffer' in resource)) continue;
      const handle =
        resource.buffer instanceof GraphDataView ? resource.buffer.buffer : resource.buffer;
      if (handle.id === id) return handle;
    }
  throw new Error(`Missing observed network resource: ${id}`);
}

/** Takes the last readback of a named buffer from a step or end capture. */
export function getNetworkFinalCapture(
  captures: readonly GPUComputationCapture[],
  name: string
): GPUComputationCapture {
  const capture = captures.filter(candidate => candidate.name === name).at(-1);
  // A requested capture slot must have been copied by the observation run.
  assert(capture);
  return {...capture, id: name};
}

/** Context roads and start points assembled from real uploaded-buffer readbacks. */
export function captureInputGeometry(
  captures: readonly GPUComputationCapture[],
  starts: readonly number[]
): GPUComputationCapture[] {
  const positions = getNetworkFinalCapture(captures, 'nodePositions');
  const offsets = getNetworkFinalCapture(captures, 'offsets');
  const neighbors = getNetworkFinalCapture(captures, 'neighbors');
  return [
    positions,
    offsets,
    neighbors,
    {
      ...positions,
      id: 'road-context',
      name: 'road-context',
      role: 'context',
      keySpace: 'node',
      geometry: {
        kind: 'graph',
        offsetsCapture: offsets.id,
        targetsCapture: neighbors.id,
        layoutCapture: positions.id,
        coordinateModel: 'geographic'
      }
    },
    {
      id: 'start-intersections',
      name: 'start-intersections',
      elementType: 'f32',
      components: 2,
      values: Float32Array.from(
        starts.flatMap(node => [positions.values[node * 2], positions.values[node * 2 + 1]])
      ),
      provenance: 'gpu-readback',
      role: 'input',
      meaning: 'input: start intersections (positions gathered from the nodePositions readback)',
      geometry: {kind: 'points', coordinateModel: 'geographic'}
    }
  ];
}

/** Assembles and validates structure and optional sidecars for any network contributor. */
export function assembleNetworkLensRecord(
  device: Device,
  props: Omit<CreateGPUComputationRecordProps, 'adapter' | 'domains'> & {variant?: string},
  sidecars: Pick<GPUComputationRecordWithSidecars, 'captures' | 'metrics'> = {}
): GPUComputationRecordWithSidecars {
  // Composed primitives (for example GPUHistogram) may name formulas owned by another pack. Their
  // refs are not in this record, so keep the id as an attribute instead of a dangling reference.
  const knownFormulas = new Set((props.formulas ?? []).map(formula => formula.id));
  const graphs = props.graphs.map(input => {
    const graph = 'graph' in input ? input.graph : input;
    const nodes = graph.nodes.map(node =>
      node.formula === undefined || knownFormulas.has(node.formula)
        ? node
        : {...node, formula: undefined, attrs: {...node.attrs, foreignFormula: node.formula}}
    );
    return 'graph' in input ? {...input, graph: {...graph, nodes}} : {...graph, nodes};
  });
  const record = {
    ...createGPUComputationRecord({
      ...props,
      graphs,
      adapter: getGPUComputationAdapter(device, props.variant),
      domains: ['gpu-network'],
      datasetVersion: props.datasetVersion ?? 'n-chicago-loop-roads-v1'
    }),
    ...sidecars
  };
  const issues = [
    ...validateGPUComputationRecord(record).issues,
    ...validateGPUComputationSidecars(record, record).issues
  ].filter(issue => issue.level === 'error');
  if (issues.length) throw new Error(JSON.stringify(issues));
  return record;
}

// COPY_SRC is an observation-only allocation choice. Production frontier declarations stay unchanged.
class NetworkLensGraph extends GPUCommandGraph {
  override createTransientBuffer(descriptor: GraphBufferDescriptor): GraphBufferHandle {
    return super.createTransientBuffer(
      this.recording?.retain?.includes(descriptor.id)
        ? {...descriptor, usage: descriptor.usage | Buffer.COPY_SRC}
        : descriptor
    );
  }
}
