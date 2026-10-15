// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, assert, type Device} from '@luma.gl/core';
import {Kernel} from '@luma.gl/engine';
import {
  captureGPUCommandGraph,
  createGPUComputationCaptures,
  createGPUComputationMetrics,
  createGPUComputationRecord,
  getGPUComputationAdapter,
  validateGPUComputationRecord,
  validateGPUComputationSidecars,
  type GPUAnalysisChoice,
  type GPUComputationCapture,
  type GPUComputationGraph,
  type GPUComputationGraphInput,
  type GPUComputationKeySpace,
  type GPUComputationDispatch,
  type GPUComputationRecordWithSidecars,
  type GPUComputationSeries,
  type GPUComputationFormulaRef
} from '@luma.gl/gpgpu';
import {
  GPUCommandGraph,
  describeGPUProgramCompilation,
  type CompiledGPUCommandGraph,
  type GPUCommandGraphRecordingOptions,
  type GPUProgramCompilation,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUData, type GPUVectorFormat} from '@luma.gl/gpgpu/gpu-data';
import {getDefaultViewPacks} from '../../../computation-views/registry/default-registry';

/** One captured array window and its explicitly declared result grammar. */
export type CoreCaptureSpec = {
  view: GraphDataView;
  role?: GPUComputationCapture['role'];
  keySpace?: string;
  meaning?: string;
  /** Spatial drawing contract, such as the `points` input of a grid capture. */
  geometry?: GPUComputationCapture['geometry'];
  /** Read back for `series` or `keyFrames` only; never emitted as a record capture. */
  observeOnly?: boolean;
  highlight?:
    | GPUComputationCapture['highlight']
    | ((values: ArrayLike<number>) => NonNullable<GPUComputationCapture['highlight']>);
};

/** Imported buffers owned by the harness and destroyed after recording or structural inspection. */
export type CoreUploads = {
  /** Imports packed rows; also used with zero-filled arrays for caller-owned destinations. */
  view<Format extends GPUVectorFormat>(
    name: string,
    values: Uint32Array | Float32Array | Int32Array,
    format: Format,
    length?: number
  ): GraphDataView<Format>;
};

/** Imported device buffers for the external vectors of a compiled GPUProgram. */
export type CoreProgramUploads = {
  /** Creates a harness-owned buffer holding `values`, destroyed after recording. */
  data(
    name: string,
    values: Uint32Array | Float32Array | Int32Array,
    format: 'float32' | 'uint32' | 'sint32'
  ): GPUData;
};

/** A recorded GPUProgram compilation and the views to capture from its lowered graph. */
export type CoreProgramBuild = {
  compilation: GPUProgramCompilation<undefined>;
  specs: Record<string, CoreCaptureSpec>;
  /** Formula id recorded on one intent node, such as the loop operation of a solver. */
  formula?: {id: string; intentNodeId: string};
};

/** What a run observed, for series and key-frame selection after the single execution. */
export type CoreObservation = {
  /** The described compiled graph, including loop coordinates and `derivedFrom` links. */
  graph: GPUComputationGraph;
  /** Every raw snapshot frame, in readback order. */
  snapshots: readonly GPUComputationCapture[];
};

/** Build and capture contract shared by all GPU Core scenarios. */
export type CoreLensRunProps = {
  scenarioId: string;
  element: string;
  label: string;
  keySpaces: readonly GPUComputationKeySpace[];
  choices?: readonly GPUAnalysisChoice[];
  retain?: readonly string[];
  at?: 'end' | 'step';
  build?(
    graph: GPUCommandGraph<undefined>,
    uploads: CoreUploads
  ): Promise<Record<string, CoreCaptureSpec>>;
  /**
   * Alternative to `build` for GPUProgram scenarios: compiles the program with the given
   * recording options and describes both its intent and compiled graph forms.
   */
  buildProgram?(
    device: Device,
    recording: GPUCommandGraphRecordingOptions,
    uploads: CoreProgramUploads
  ): Promise<CoreProgramBuild>;
  series?(
    compiled: CompiledGPUCommandGraph<undefined>,
    observation: CoreObservation
  ): GPUComputationSeries[];
  /**
   * With `at: 'step'`, keeps only the listed raw snapshot frames (index to iteration) instead of
   * one frame per step. The last frame of each captured name must be listed; it carries the roles.
   */
  keyFrames?(observation: CoreObservation): ReadonlyMap<number, number>;
  /** Size of the `iteration` axis when it differs from the number of scheduled nodes. */
  iterationCount?: number;
  /**
   * Replays the geometry of the first recorded node with this `op` as one extra dispatch that
   * writes `global_invocation_id.x` per lane, for DispatchView. Idle lanes keep `0xffffffff`.
   */
  idProbe?: {op: string};
};

// Record-local formulas keep these independently loadable examples independent of view packs.
const FORMULAS: Readonly<Record<string, {tex: string; ascii: string}>> = {
  'gpu-core.scan': {
    tex: String.raw`p_i = \sum_{j=s(i)}^{i-\delta} x_j \pmod{2^{32}},\quad \delta\in\{0,1\}`,
    ascii:
      'prefix[i] = sum(input[j], segmentStart(i) <= j <= i - delta) mod 2^32; delta = 0 inclusive, 1 exclusive; segmentStart = 0 without flags'
  },
  'gpu-core.flag-offsets': {
    tex: String.raw`o_i = \sum_{j<i} f_j,\quad c = o_{n-1}+f_{n-1}`,
    ascii:
      'offsets[i] = sum(flags[j], j < i); count = offsets[n-1] + flags[n-1], or 0 for empty input; flags are binary'
  },
  'gpu-core.compaction': {
    tex: String.raw`o_i = \sum_{j<i} f_j,\quad y_{o_i}=x_i\ \text{when }f_i=1`,
    ascii:
      'offset[i] = sum(flags[j], j < i); output[offset[i]] = input[i] when flags[i] = 1; count = sum(flags); flags are binary'
  },
  'gpu-core.mask': {
    tex: String.raw`m_i = [w_i\ne 0]\land[h_i\ne 0]`,
    ascii: 'output[i] = (wet[i] != 0) AND (high[i] != 0), represented as 0 or 1'
  },
  'gpu-core.reduction': {
    tex: String.raw`r = \sum_{i:m_i\ne 0} p_i`,
    ascii:
      'totalRain = sum(rainfall[i] where mask[i] != 0), accumulated as float32 in a hierarchical reduction'
  }
};

/** Runs and captures the graph exactly once; only real GPU readbacks become result captures. */
export async function runCoreLens(
  device: Device,
  props: CoreLensRunProps
): Promise<GPUComputationRecordWithSidecars> {
  return buildCoreLens(device, props, true);
}

/** Builds and validates the same graph without executing it; node tests stub NullDevice pipelines. */
export async function describeCoreLens(
  device: Device,
  props: CoreLensRunProps
): Promise<GPUComputationRecordWithSidecars> {
  return buildCoreLens(device, props, false);
}

async function buildCoreLens(
  device: Device,
  props: CoreLensRunProps,
  execute: boolean
): Promise<GPUComputationRecordWithSidecars> {
  const recording = {sourceText: true, retain: [...(props.retain ?? [])]};
  const graph = new GPUCommandGraph<undefined>(device, {
    id: props.scenarioId.replace(/\./g, '-'),
    recording
  });
  const owned: Buffer[] = [];
  const uploads: CoreUploads = {
    view(name, values, format, length) {
      const buffer = device.createBuffer({
        id: name,
        data: values,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      owned.push(buffer);
      const handle = graph.importBuffer(
        {id: name, byteLength: buffer.byteLength, usage: buffer.usage},
        buffer
      );
      const components = Number(/x([234])$/.exec(format)?.[1] ?? 1);
      return graph.createDataView(handle, {format, length: length ?? values.length / components});
    }
  };
  const programUploads: CoreProgramUploads = {
    data(name, values, format) {
      const buffer = device.createBuffer({
        id: name,
        data: values,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      owned.push(buffer);
      return new GPUData({buffer, format, length: values.length});
    }
  };
  try {
    // Exactly one of build and buildProgram is supplied: programs bring their own lowered graph.
    assert(Boolean(props.build) !== Boolean(props.buildProgram));
    const program = props.buildProgram
      ? await props.buildProgram(device, recording, programUploads)
      : undefined;
    const specs = program ? program.specs : await props.build!(graph, uploads);
    const compiled = (program ? program.compilation.graph : graph).compile();
    try {
      const programDescription = program
        ? describeGPUProgramCompilation(program.compilation, {compiled, domain: 'gpu-core'})
        : undefined;
      const description = programDescription
        ? {graph: programDescription.graphs[1], sources: programDescription.sources}
        : compiled.describe({label: props.label, domain: 'gpu-core'});
      const recordGraphs: GPUComputationGraphInput[] = programDescription
        ? programDescription.graphs.map(recordGraph =>
            program!.formula && recordGraph.form === 'intent'
              ? {
                  ...recordGraph,
                  nodes: recordGraph.nodes.map(node =>
                    node.id === program!.formula!.intentNodeId
                      ? {...node, formula: program!.formula!.id}
                      : node
                  )
                }
              : recordGraph
          )
        : [description];
      const formulaIds = [
        ...new Set(
          recordGraphs.flatMap(input =>
            ('graph' in input ? input.graph : input).nodes.flatMap(node =>
              node.formula ? [node.formula] : []
            )
          )
        )
      ];
      const registeredFormulas = getDefaultViewPacks().flatMap(pack =>
        Object.values(pack.formulas ?? {}).filter(
          (formula): formula is GPUComputationFormulaRef => 'tex' in formula
        )
      );
      const formulas = formulaIds.map(id => {
        const registered = registeredFormulas.find(formula => formula.id === id);
        if (registered) return registered;
        const formula = FORMULAS[id];
        // Standalone scenarios support the K-1 formulas; other operators supply a registered pack.
        assert(formula);
        return {id, ...formula, symbols: [], derivation: 'curated' as const};
      });
      const record = createGPUComputationRecord({
        id: props.scenarioId,
        adapter: getGPUComputationAdapter(device),
        domains: ['gpu-core'],
        graphs: programDescription ? recordGraphs : [description],
        ...(programDescription && {
          sources: programDescription.sources,
          // An unrolled loop repeats each lowering decision per iteration; keep the first of each.
          decisions: programDescription.decisions.filter(decision => decision.id.endsWith('#0'))
        }),
        formulas,
        datasetVersion: 'k-core-2024-09-27',
        choices: props.choices,
        axes: {
          keySpaces: props.keySpaces,
          relations: [],
          axes: [
            ...props.keySpaces.map(keySpace => ({
              id: keySpace.id,
              label: keySpace.label,
              kind: 'data' as const,
              keySpace: keySpace.id,
              size: keySpace.size
            })),
            {
              id: 'node',
              label: 'Scheduled node',
              kind: 'structure',
              size: compiled.stats.nodeOrder.length
            },
            ...(props.at === 'step'
              ? [
                  {
                    id: 'iteration',
                    label: props.iterationCount === undefined ? 'Execution step' : 'Iteration',
                    kind: 'execution' as const,
                    size: props.iterationCount ?? compiled.stats.nodeOrder.length
                  }
                ]
              : [])
          ]
        }
      });
      const validation = validateGPUComputationRecord(record);
      if (!validation.valid) throw new Error(JSON.stringify(validation.issues));
      if (!execute) return record;
      const capture = await captureGPUCommandGraph(compiled, {
        encodeOptions: {parameters: undefined},
        views: Object.fromEntries(Object.entries(specs).map(([name, spec]) => [name, spec.view])),
        at: props.at ?? 'end',
        runId: record.runId,
        graphId: description.graph.id
      });
      const observation: CoreObservation = {graph: description.graph, snapshots: capture.captures};
      const keyFrames = props.keyFrames?.(observation);
      const lastIndices = new Map(
        capture.captures.map((snapshot, index) => [snapshot.name, index])
      );
      const snapshots = capture.captures.flatMap((snapshot, index) => {
        const spec = specs[snapshot.name];
        const {view, highlight, observeOnly, ...tags} = spec;
        if (observeOnly || (keyFrames && !keyFrames.has(index))) return [];
        const iteration = keyFrames?.get(index) ?? Math.floor(index / Object.keys(specs).length);
        if (lastIndices.get(snapshot.name) !== index) {
          // Window origins remain explicit on partial frames; only final frames carry result roles.
          return {
            ...snapshot,
            iteration,
            keySpace: spec.keySpace,
            ...(spec.meaning?.match(/first:\d+/) && {meaning: spec.meaning.match(/first:\d+/)![0]})
          };
        }
        return {
          ...snapshot,
          ...(props.at === 'step' && {iteration}),
          ...tags,
          ...(highlight && {
            highlight: typeof highlight === 'function' ? highlight(snapshot.values) : highlight
          })
        };
      });
      const replays: GPUComputationCapture[] = [];
      if (props.idProbe) {
        const node = description.graph.nodes.find(candidate => candidate.op === props.idProbe!.op);
        // The probe scenario is built around a recorded node of this op with a dispatch.
        assert(node?.dispatch && node.workgroupSize);
        replays.push({
          ...(await replayIdProbe(device, node.id, node.workgroupSize, node.dispatch)),
          graph: capture.captures[0]?.graph ?? description.graph.id
        });
      }
      const captures = createGPUComputationCaptures({runId: record.runId, snapshots, replays});
      const metrics = createGPUComputationMetrics({
        runId: record.runId,
        graph: description.graph.id,
        timing: {nodes: []},
        series: props.series?.(compiled, observation),
        encoding: {
          coalesced: props.at !== 'step',
          timestamped: false,
          taps: 0,
          lensTwin: true,
          retained: props.retain ?? []
        }
      });
      const sidecars = validateGPUComputationSidecars(record, {captures, metrics: [metrics]});
      if (!sidecars.valid) throw new Error(JSON.stringify(sidecars.issues));
      return {...record, captures, metrics: [metrics]};
    } finally {
      compiled.destroy();
    }
  } finally {
    for (const buffer of owned) buffer.destroy();
  }
}

/** Dispatches the recorded geometry once more and reads back each lane's global invocation id. */
async function replayIdProbe(
  device: Device,
  afterNode: string,
  workgroupSize: readonly [number, number, number],
  dispatch: GPUComputationDispatch
): Promise<GPUComputationCapture> {
  const [x, y, z] = dispatch.workgroups;
  // Elementwise rows fit one dimension of workgroups, so the lane id is global_invocation_id.x.
  assert(y === 1 && z === 1 && workgroupSize[1] === 1 && workgroupSize[2] === 1);
  const laneCount = x * workgroupSize[0];
  const invocationCount = dispatch.invocationCount ?? laneCount;
  const buffer = device.createBuffer({
    id: 'id-probe',
    data: new Uint32Array(laneCount).fill(0xffffffff),
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  const kernel = new Kernel(device, {
    id: 'id-probe',
    source: `const INVOCATION_COUNT: u32 = ${invocationCount}u;
@group(0) @binding(0) var<storage, read_write> idProbe: array<u32>;
@compute @workgroup_size(${workgroupSize[0]}) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x < INVOCATION_COUNT) { idProbe[id.x] = id.x; }
}`,
    shaderLayout: {bindings: [{name: 'idProbe', type: 'storage', group: 0, location: 0}]}
  });
  try {
    const commandEncoder = device.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass({id: 'id-probe'});
    kernel.dispatch(computePass, {bindings: {idProbe: buffer}, x, y, z});
    computePass.end();
    device.submit(commandEncoder.finish());
    commandEncoder.destroy();
    const bytes = await buffer.readAsync();
    return {
      id: 'id-probe',
      name: 'id-probe',
      afterNode,
      elementType: 'u32',
      values: new Uint32Array(bytes.buffer, bytes.byteOffset, laneCount).slice(),
      provenance: 'gpu-replay',
      meaning: 'global_invocation_id per lane (replay dispatch)'
    };
  } finally {
    kernel.destroy();
    buffer.destroy();
  }
}
