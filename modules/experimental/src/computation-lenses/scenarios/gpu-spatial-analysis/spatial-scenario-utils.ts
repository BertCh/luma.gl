// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, Buffer, type Device} from '@luma.gl/core';
import {
  captureGPUCommandGraph,
  createGPUComputationCaptures,
  createGPUComputationRecord,
  getGPUComputationAdapter,
  validateGPUComputationRecord,
  validateGPUComputationSidecars,
  type GPUComputationAttributeValue,
  type GPUComputationCapture,
  type GPUComputationAxes,
  type GPUComputationRecordWithSidecars
} from '@luma.gl/gpgpu';
import {
  GPUCommandGraph,
  type GPUCommandGraphEncodeOptions,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createSpatialChoice} from '../../../gpu-spatial-analysis/computation-pack/choices';
import {mergeGPUCoreFormulaRefsForGraphs} from '../../../computation-views/packs/gpu-core/formulas';
import type {SpatialOperatorEntry} from '../../../gpu-spatial-analysis/computation-pack/types';

/**
 * Creates an explicit observation graph that retains the transients named in `retain`. The list is
 * kept by reference and read at compile time, so ids pushed after the example has built the graph
 * (but before compilation) are retained too.
 */
export function createSpatialRecordingGraph(
  device: Device,
  id: string,
  retain: readonly string[]
): GPUCommandGraph {
  return new GPUCommandGraph(device, {id, recording: {retain}});
}

/**
 * Ids of the graph-owned (transient) buffers behind `views`, for a recording graph's `retain` list.
 * Imported uploads are caller-owned and read back directly, so retaining them is an error.
 */
export function getRetainedViewIds(views: Readonly<Record<string, GraphDataView>>): string[] {
  return [
    ...new Set(
      Object.values(views)
        .filter(view => view.buffer.transient)
        .map(view => view.buffer.id)
    )
  ];
}

/** Uploads a packed input for both computation and provenance-preserving readback. Caller destroys it. */
export function uploadSpatialView<Format extends 'float32' | 'float32x2' | 'uint32'>(
  graph: GPUCommandGraph,
  name: string,
  data: Float32Array | Uint32Array,
  format: Format
): {view: GraphDataView<Format>; destroy(): void} {
  const buffer = graph.device.createBuffer({
    id: name,
    data,
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  const handle = graph.importBuffer(
    {id: name, byteLength: buffer.byteLength, usage: buffer.usage},
    buffer
  );
  const view = graph.createDataView(handle, {
    format,
    length: data.length / (format === 'float32x2' ? 2 : 1)
  });
  return {view, destroy: () => buffer.destroy()};
}

/** Role and support of one scalar readback; geometry aliases use the captured coordinate buffer. */
export type SpatialCaptureTag = {
  role: NonNullable<GPUComputationCapture['role']>;
  keySpace: string;
  polygons?: {verticesCapture: string; offsetsCapture: string};
  meaning?: string;
};

/**
 * Tags scalar captures and adds coordinate views for ResultView's single-ring polygon grammar.
 * Scalar names remain the kernel output names. Geometry aliases share actual GPU-readback arrays;
 * no CPU reconstruction, derived values, geometry flattening or additional GPU execution occurs.
 * Coordinate views keep the vertex key space so datum receipts cannot mistake a vertex for a tract.
 */
export function tagSpatialCaptures(
  captures: readonly GPUComputationCapture[],
  tags: Readonly<Record<string, SpatialCaptureTag>>
): GPUComputationCapture[] {
  const result: GPUComputationCapture[] = [];
  for (const capture of captures) {
    const tag = tags[capture.name];
    if (!tag) {
      result.push({...capture, role: 'intermediate'});
      continue;
    }
    result.push({
      ...capture,
      role: tag.polygons ? 'intermediate' : tag.role,
      keySpace: tag.keySpace,
      ...(tag.meaning && {meaning: tag.meaning})
    });
    if (!tag.polygons) continue;
    const vertices = captures.find(candidate => candidate.name === tag.polygons!.verticesCapture);
    const offsets = captures.find(candidate => candidate.name === tag.polygons!.offsetsCapture);
    // Geometry-bearing captures store coordinates; scalar values are referenced separately.
    assert(vertices && offsets && vertices.components === 2);
    result.push({
      ...vertices,
      id: `${capture.id}-geometry`,
      name: `${capture.name}.geometry`,
      role: tag.role,
      keySpace: tags[vertices.name]?.keySpace,
      meaning: tag.meaning,
      geometry: {
        kind: 'polygons',
        coordinateModel: 'geographic',
        offsetsCapture: offsets.id,
        valuesCapture: capture.id
      }
    });
  }
  return result;
}

/** Inputs to one recorded execution; no GPU-dependent code runs until this helper is called. */
export type SpatialScenarioRun = {
  id: string;
  graph: GPUCommandGraph;
  encodeOptions: GPUCommandGraphEncodeOptions<void>;
  views: Readonly<Record<string, GraphDataView>>;
  entry: SpatialOperatorEntry;
  choices: Readonly<Record<string, GPUComputationAttributeValue>>;
  datasetVersion: string;
  seed: number;
  axes: Partial<GPUComputationAxes>;
  tags: Readonly<Record<string, SpatialCaptureTag>>;
  /** Additional factual metadata needed by the pack's guards. */
  metadata?: Readonly<Record<string, GPUComputationAttributeValue>>;
  /**
   * Names of `views` that a GPU replay wrote (for example one pinned row's null distribution). They are
   * stored in the sidecar's `replays` with `gpu-replay` provenance instead of as run snapshots.
   */
  replays?: Readonly<Record<string, {keySpace: string; meaning: string}>>;
  /**
   * CSR slot captures keyed by the capture name of their row offsets. Each slot capture is read
   * back only up to the used slot count (the last row offset): slots past it are unused capacity
   * that no view reads (their contents are whatever the allocation held), so they stay out of the
   * record. The capture's meaning states the captured and allocated slot counts.
   */
  csrSlotCaptures?: Readonly<Record<string, readonly string[]>>;
};

/**
 * Compiles once, captures at the end and validates both record and readback sidecar.
 * Imported buffers remain caller-owned; compiled transients are always released here.
 * The capture API exposes no timings, so timing metrics are left absent rather than estimated.
 */
export async function runSpatialScenario(
  props: SpatialScenarioRun
): Promise<GPUComputationRecordWithSidecars> {
  const compiled = props.graph.compile();
  try {
    const description = compiled.describe();
    const record = createGPUComputationRecord({
      id: props.id,
      adapter: getGPUComputationAdapter(props.graph.device),
      domains: ['gpu-spatial-analysis'],
      graphs: [description],
      axes: props.axes,
      datasetVersion: props.datasetVersion,
      seed: props.seed,
      formulas: mergeGPUCoreFormulaRefsForGraphs(
        [description],
        'formula' in props.entry.formula ? [] : [props.entry.formula]
      ),
      choices: [
        ...props.entry.choices.map(template => {
          // Every displayed choice must describe the actual run.
          assert(Object.hasOwn(props.choices, template.name));
          return createSpatialChoice(template, props.choices[template.name]);
        }),
        ...Object.entries(props.metadata ?? {}).map(([name, value]) =>
          createSpatialChoice({name, stage: 'presentation', sweepable: false}, value)
        )
      ]
    });
    const validation = validateGPUComputationRecord(record);
    if (!validation.valid) throw new Error(JSON.stringify(validation.issues));
    const captured = await captureGPUCommandGraph(compiled, {
      encodeOptions: props.encodeOptions,
      views: props.views,
      at: 'end',
      maximumSnapshots: 1,
      runId: record.runId
    });
    // Capacity failure must not produce a plausible-looking figure from truncated weights.
    const overflow = captured.captures.find(capture => capture.name === 'weightsOverflow');
    if (overflow && overflow.values[0] !== 0)
      throw new Error('Spatial lens weights capacity exceeded');
    trimCSRSlotCaptures(captured.captures, props.csrSlotCaptures ?? {});
    const replayTags = props.replays ?? {};
    const isReplay = (capture: GPUComputationCapture) => Object.hasOwn(replayTags, capture.name);
    const captures = createGPUComputationCaptures({
      runId: record.runId,
      snapshots: tagSpatialCaptures(
        captured.captures.filter(capture => !isReplay(capture)),
        props.tags
      ),
      replays: captured.captures.filter(isReplay).map(capture => ({
        ...capture,
        ...replayTags[capture.name],
        role: 'intermediate' as const,
        provenance: 'gpu-replay' as const
      }))
    });
    const sidecars = validateGPUComputationSidecars(record, {captures});
    if (!sidecars.valid) throw new Error(JSON.stringify(sidecars.issues));
    return {...record, captures};
  } finally {
    compiled.destroy();
  }
}

import type {ChicagoTracts} from './spatial-lens-data';
import {
  getGPUNeighborSearchParameterValues,
  getGPUMapColoringParameterValues,
  type GPUContiguityWeightsProps,
  type GPUSpatialWeights
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploads one-ring Chicago features, preserving their source order and caller-owned buffers. */
export function uploadChicagoTractPolygons(
  graph: GPUCommandGraph,
  tracts: ChicagoTracts,
  count = tracts.count
) {
  // The compact Chicago extract has one polygon and one ring per feature.
  assert(count > 0 && count <= tracts.count);
  const vertexCount = tracts.ringOffsets[count];
  const vertices = uploadSpatialView(
    graph,
    'vertices',
    tracts.vertices.slice(0, vertexCount * 2),
    'float32x2'
  );
  const ringOffsets = uploadSpatialView(
    graph,
    'ringOffsets',
    tracts.ringOffsets.slice(0, count + 1),
    'uint32'
  );
  const featureOffsets = uploadSpatialView(
    graph,
    'featureOffsets',
    Uint32Array.from({length: count + 1}, (_, index) => index),
    'uint32'
  );
  const polygonOffsets = uploadSpatialView(
    graph,
    'polygonOffsets',
    tracts.polygonRingOffsets.slice(0, count + 1),
    'uint32'
  );
  return {
    polygons: {
      kind: 'polygons' as const,
      positions: vertices.view,
      ringOffsets: ringOffsets.view,
      featureOffsets: featureOffsets.view,
      polygonOffsets: polygonOffsets.view
    },
    views: {vertices: vertices.view, ringOffsets: ringOffsets.view},
    destroy() {
      for (const uploaded of [vertices, ringOffsets, featureOffsets, polygonOffsets])
        uploaded.destroy();
    }
  };
}

/** Creates tract and CSR-slot axes; geometry keeps a separate vertex key space. */
export function createTractAxes(
  count: number,
  weightsCapacity: number,
  vertexCount = 0
): Partial<GPUComputationAxes> {
  return {
    axes: [{id: 'entity', label: 'Chicago tract', kind: 'data', keySpace: 'tract', size: count}],
    keySpaces: [
      {id: 'tract', label: 'Chicago tract', size: count, indexSpace: 'csr'},
      {id: 'weights', label: 'CSR slot capacity', size: weightsCapacity, indexSpace: '1d'},
      {id: 'vertex', label: 'Ring vertex', size: vertexCount, indexSpace: '1d'},
      {id: 'summary', label: 'Weights moments S0, S1, S2', size: 3, indexSpace: '1d'},
      {
        id: 'counts',
        label: 'Slots, asymmetry, isolates, minimum, maximum, empty',
        size: 6,
        indexSpace: '1d'
      },
      {id: 'status', label: 'Status scalar', size: 1, indexSpace: '1d'}
    ]
  };
}

/** Uploaded inputs shared by the independent tract examples; no derived CPU geometry is uploaded. */
export type TractWeightsExampleInput = {
  graph: GPUCommandGraph;
  tracts: GPUContiguityWeightsProps['polygons'] & {positions: GraphDataView<'float32x2'>};
  values: GraphDataView<'float32'>;
  searchParameters: GraphDataView<'float32'>;
  colorParameters: GraphDataView<'uint32'>;
};

/** Graph and caller-named transient outputs returned by an executable weights example. */
export type SpatialWeightsExampleExecution = {
  graph: GPUCommandGraph;
  weights: GPUSpatialWeights;
  views: Readonly<Record<string, GraphDataView>>;
};

/** Adds native geometry and factual island metadata to captured GPU outputs, without recalculating them. */
export function finishSpatialWeightsRecord(
  record: GPUComputationRecordWithSidecars,
  geometries: Readonly<
    Record<
      string,
      {
        role: NonNullable<GPUComputationCapture['role']>;
        geometry: NonNullable<GPUComputationCapture['geometry']>;
      }
    >
  >
): GPUComputationRecordWithSidecars {
  const snapshots = record.captures!.snapshots.map(capture => ({
    ...capture,
    ...geometries[capture.name]
  }));
  const counts = snapshots.find(capture => capture.name === 'counts');
  const islandCount = counts?.values[2];
  // Fail explicitly instead of drawing truncated adjacency or an invalid nominal map.
  for (const capture of snapshots) {
    if (capture.name.endsWith('Overflow')) assert(capture.values[0] === 0);
    if (capture.name === 'colorCount') assert(capture.values[0] <= 5);
    if (capture.name === 'conflictCount') assert(capture.values[0] === 0);
    if (capture.name === 'converged') assert(capture.values[0] === 1);
  }
  const captures = createGPUComputationCaptures({runId: record.runId, snapshots});
  const result = {
    ...record,
    captures,
    choices: [
      ...(record.choices ?? []),
      ...(islandCount !== undefined && Number.isFinite(islandCount)
        ? [
            createSpatialChoice(
              {name: 'islandCount', stage: 'neighbourhood', sweepable: false},
              islandCount
            )
          ]
        : [])
    ]
  };
  // Choice metadata is appended from actual readback; the recorded run identity remains unchanged.
  assert(validateGPUComputationRecord(result).valid);
  assert(validateGPUComputationSidecars(result, {captures}).valid);
  return result;
}

/** Cuts CSR slot captures to the used slot count; see {@link SpatialScenarioRun.csrSlotCaptures}. */
function trimCSRSlotCaptures(
  captures: GPUComputationCapture[],
  csrSlotCaptures: Readonly<Record<string, readonly string[]>>
): void {
  for (const [offsetsName, slotNames] of Object.entries(csrSlotCaptures)) {
    const offsets = captures.find(capture => capture.name === offsetsName);
    if (!offsets?.values.length) continue;
    const usedSlots = offsets.values[offsets.values.length - 1];
    for (const [index, capture] of captures.entries()) {
      if (!slotNames.includes(capture.name)) continue;
      const components = capture.components ?? 1;
      const allocatedSlots = capture.values.length / components;
      if (usedSlots >= allocatedSlots) continue;
      const note = `slots 0-${usedSlots - 1} of ${allocatedSlots} allocated (unused capacity not captured)`;
      captures[index] = {
        ...capture,
        // Readback captures are typed arrays; slicing keeps their element type.
        values: (capture.values as Float32Array | Uint32Array | Int32Array).slice(
          0,
          usedSlots * components
        ),
        meaning: capture.meaning ? `${capture.meaning}; ${note}` : note
      };
    }
  }
}

/** Runs a tract weights card, retaining its transient captures before compilation. */
export async function runTractWeightsScenario(props: {
  device: Device;
  id: string;
  tracts: ChicagoTracts;
  entry: SpatialOperatorEntry;
  choices: SpatialScenarioRun['choices'];
  metadata?: SpatialScenarioRun['metadata'];
  presentation: 'graph' | 'lag' | 'coloring' | 'algebra' | 'summary' | 'transpose' | 'validation';
  count?: number;
  runExample(
    device: Device,
    input: TractWeightsExampleInput
  ): Promise<SpatialWeightsExampleExecution>;
}): Promise<GPUComputationRecordWithSidecars> {
  const retained: string[] = [];
  const graph = new GPUCommandGraph(props.device, {id: props.id, recording: {retain: retained}});
  const count = props.count ?? props.tracts.count;
  const geometry = uploadChicagoTractPolygons(graph, props.tracts, count);
  const values = uploadSpatialView(
    graph,
    'values',
    props.tracts.columns['poverty150Pct'].slice(0, count),
    'float32'
  );
  const searchParameters = uploadSpatialView(
    graph,
    'searchParameters',
    getGPUNeighborSearchParameterValues({
      bounds: [-88, 41, -87, 43],
      weightKind: 'binary',
      rowStandardize: false
    }),
    'float32'
  );
  const colorParameters = uploadSpatialView(
    graph,
    'colorParameters',
    getGPUMapColoringParameterValues({seed: 1}),
    'uint32'
  );
  try {
    const execution = await props.runExample(props.device, {
      graph,
      tracts: geometry.polygons,
      values: values.view,
      searchParameters: searchParameters.view,
      colorParameters: colorParameters.view
    });
    retained.push(...getRetainedViewIds(execution.views));
    const polygons = {verticesCapture: 'vertices', offsetsCapture: 'ringOffsets'};
    const tags: Record<string, SpatialCaptureTag> = {
      vertices: {role: 'intermediate', keySpace: 'vertex'},
      weightsRowOffsets: {role: 'intermediate', keySpace: 'tract'},
      weightsNeighbors: {role: 'intermediate', keySpace: 'weights'},
      weightsValues: {role: 'intermediate', keySpace: 'weights'},
      sourceWeightsRowOffsets: {role: 'intermediate', keySpace: 'tract'},
      sourceWeightsValues: {role: 'intermediate', keySpace: 'weights'},
      neighborCounts: {role: 'intermediate', keySpace: 'tract'},
      centroids: {role: 'intermediate', keySpace: 'tract'},
      statistics: {role: 'intermediate', keySpace: 'summary', meaning: 'S0, S1, S2'},
      counts: {
        role: 'intermediate',
        keySpace: 'counts',
        meaning: 'Slots, asymmetric slots, isolates, minimum degree, maximum degree, empty rows'
      }
    };
    if (props.presentation === 'lag') {
      tags['values'] = {
        role: 'input',
        keySpace: 'tract',
        polygons,
        meaning: 'Population below 150% poverty line (%)'
      };
      tags['lag'] = {
        role: 'output',
        keySpace: 'tract',
        polygons,
        meaning: 'Row-weighted neighboring poverty percentage'
      };
    }
    if (props.presentation === 'coloring') {
      tags['colors'] = {
        role: 'output',
        keySpace: 'tract',
        polygons,
        meaning: 'Nominal greedy map color (not a score)'
      };
    }
    if (props.presentation === 'summary') {
      tags['weightsValues'] = {role: 'input', keySpace: 'weights'};
      tags['statistics'] = {...tags['statistics'], role: 'output'};
      tags['counts'] = {...tags['counts'], role: 'output'};
    }
    if (props.presentation === 'algebra' || props.presentation === 'transpose') {
      tags['sourceWeightsValues'] = {
        role: 'input',
        keySpace: 'weights',
        meaning: 'Source CSR weights by slot'
      };
      tags['weightsValues'] = {
        role: 'output',
        keySpace: 'weights',
        meaning: 'Result CSR weights by slot; row offsets define the pattern'
      };
      tags['weightsRowOffsets'] = {role: 'output', keySpace: 'tract'};
      tags['weightsNeighbors'] = {role: 'output', keySpace: 'weights'};
    }
    if (props.presentation === 'validation') {
      tags['weightsRowOffsets'] = {role: 'input', keySpace: 'tract'};
      tags['flags'] = {
        role: 'output',
        keySpace: 'tract',
        meaning: 'Violation bitmask: offsets 1, range 2, order 4, self 8, weight 16'
      };
      tags['invalidCount'] = {role: 'output', keySpace: 'status'};
    }
    const record = await runSpatialScenario({
      id: props.id,
      graph,
      encodeOptions: {parameters: undefined},
      views: {
        // Tract outlines are captured only where a figure draws them (graph, coloring and lag
        // maps). The other presentations draw no map, so their 0.1 MB of vertices is not read back.
        ...(['graph', 'coloring', 'lag'].includes(props.presentation) && geometry.views),
        ...(props.presentation === 'lag' && {values: values.view}),
        ...execution.views
      },
      entry: props.entry,
      choices: props.choices,
      metadata: {coordinateModel: 'geographic', ...props.metadata},
      csrSlotCaptures: {
        weightsRowOffsets: ['weightsNeighbors', 'weightsValues', 'distances'],
        sourceWeightsRowOffsets: ['sourceWeightsValues']
      },
      datasetVersion: props.tracts.dataset.id,
      seed: 1,
      axes: {
        ...createTractAxes(
          count,
          execution.weights.neighbors.length,
          geometry.views.vertices.length
        ),
        relations: [
          {
            id: 'weights',
            from: 'tract',
            to: 'tract',
            cardinality: 'N:M',
            kind: 'buffer',
            timeDependent: false,
            ref: {
              layout: 'csr',
              resource: execution.weights.neighbors.buffer.id,
              offsetsResource: execution.weights.offsets.buffer.id
            }
          }
        ]
      },
      tags
    });
    return finishSpatialWeightsRecord(record, {
      ...(props.presentation === 'graph' && {
        vertices: {
          role: 'input',
          geometry: {kind: 'polygons', coordinateModel: 'geographic', offsetsCapture: 'ringOffsets'}
        },
        neighborCounts: {
          role: 'output',
          geometry: {
            kind: 'graph',
            coordinateModel: 'geographic',
            offsetsCapture: 'weightsRowOffsets',
            targetsCapture: 'weightsNeighbors',
            layoutCapture: 'centroids'
          }
        }
      }),
      ...(props.presentation === 'coloring' && {
        vertices: {
          role: 'input',
          geometry: {kind: 'polygons', coordinateModel: 'geographic', offsetsCapture: 'ringOffsets'}
        }
      })
    });
  } finally {
    geometry.destroy();
    values.destroy();
    searchParameters.destroy();
    colorParameters.destroy();
  }
}

/** Tag of a tract-keyed capture; `polygons` draws its values on the tract outlines. */
export type TractCaptureTag = Omit<SpatialCaptureTag, 'keySpace' | 'polygons'> & {
  keySpace?: string;
  polygons?: boolean;
};

/** What a tract statistics scenario hands back after running its example and adding its tags. */
export type TractStatisticsExecution = {
  encodeOptions?: SpatialScenarioRun['encodeOptions'];
  weights?: GPUSpatialWeights;
  /** Every view to read back, by capture name; transient ones are retained automatically. */
  views: Readonly<Record<string, GraphDataView>>;
  tags: Readonly<Record<string, TractCaptureTag>>;
  /** Extra key spaces of non-tract captures (summary fields, permutations). */
  keySpaces?: NonNullable<GPUComputationAxes['keySpaces']>;
  replays?: SpatialScenarioRun['replays'];
};

/**
 * Runs a Chicago tract statistics card: uploads the tract outlines and the named float columns, lets the
 * scenario run its example, retains what it returns and records it with tract, vertex and CSR key spaces.
 */
export async function runTractStatisticsScenario(props: {
  device: Device;
  id: string;
  tracts: ChicagoTracts;
  entry: SpatialOperatorEntry;
  choices: SpatialScenarioRun['choices'];
  metadata?: SpatialScenarioRun['metadata'];
  /** Tract columns to upload as float32 views, by view name. */
  columns: Readonly<Record<string, string>>;
  execute(context: {
    graph: GPUCommandGraph;
    polygons: ReturnType<typeof uploadChicagoTractPolygons>['polygons'];
    columns: Record<string, GraphDataView<'float32'>>;
    upload: <Format extends 'float32' | 'uint32'>(
      name: string,
      data: Format extends 'float32' ? Float32Array : Uint32Array,
      format: Format
    ) => GraphDataView<Format>;
  }): Promise<TractStatisticsExecution>;
}): Promise<GPUComputationRecordWithSidecars> {
  const retained: string[] = [];
  const graph = new GPUCommandGraph(props.device, {id: props.id, recording: {retain: retained}});
  const geometry = uploadChicagoTractPolygons(graph, props.tracts);
  const uploads: {destroy(): void}[] = [];
  const upload: Parameters<typeof props.execute>[0]['upload'] = (name, data, format) => {
    const uploaded = uploadSpatialView(graph, name, data, format);
    uploads.push(uploaded);
    return uploaded.view;
  };
  try {
    const columns = Object.fromEntries(
      Object.entries(props.columns).map(([name, column]) => [
        name,
        upload(name, props.tracts.columns[column], 'float32')
      ])
    );
    const execution = await props.execute({graph, polygons: geometry.polygons, columns, upload});
    retained.push(...getRetainedViewIds(execution.views));
    const count = props.tracts.count;
    const vertexCount = geometry.views.vertices.length;
    const slotCount = execution.weights?.neighbors.length;
    const geometryReference = {verticesCapture: 'vertices', offsetsCapture: 'ringOffsets'};
    const tags: Record<string, SpatialCaptureTag> = {
      vertices: {role: 'intermediate', keySpace: 'vertex'},
      weightsRowOffsets: {role: 'intermediate', keySpace: 'tract'},
      weightsNeighbors: {role: 'intermediate', keySpace: 'weights'},
      weightsValues: {role: 'intermediate', keySpace: 'weights'}
    };
    for (const [name, view] of Object.entries(execution.views)) {
      // Untagged tract-length outputs are intermediates of the same support.
      if (view.length === count) tags[name] = {role: 'intermediate', keySpace: 'tract'};
    }
    for (const [name, {polygons, keySpace, ...tag}] of Object.entries(execution.tags)) {
      tags[name] = {
        ...tag,
        keySpace: keySpace ?? 'tract',
        ...(polygons && {polygons: geometryReference})
      };
    }
    return await runSpatialScenario({
      id: props.id,
      graph,
      encodeOptions: execution.encodeOptions ?? {parameters: undefined},
      // Tract outlines are read back only when a figure draws them (a polygon-tagged capture).
      views: {
        ...(Object.values(execution.tags).some(tag => tag.polygons) ? geometry.views : {}),
        ...execution.views
      },
      entry: props.entry,
      choices: props.choices,
      metadata: {coordinateModel: 'geographic', ...props.metadata},
      datasetVersion: props.tracts.dataset.id,
      seed: 1,
      axes: {
        axes: [
          {id: 'entity', label: 'Chicago tract', kind: 'data', keySpace: 'tract', size: count}
        ],
        keySpaces: [
          {id: 'tract', label: 'Chicago tract', size: count, indexSpace: 'csr'},
          {id: 'vertex', label: 'Ring vertex', size: vertexCount, indexSpace: '1d'},
          ...(slotCount === undefined
            ? []
            : [
                {
                  id: 'weights',
                  label: 'CSR slot capacity',
                  size: slotCount,
                  indexSpace: '1d' as const
                }
              ]),
          ...(execution.keySpaces ?? [])
        ],
        ...(execution.weights && {
          relations: [
            {
              id: 'weights',
              from: 'tract',
              to: 'tract',
              cardinality: 'N:M' as const,
              kind: 'buffer' as const,
              timeDependent: false,
              ref: {
                layout: 'csr' as const,
                resource: execution.weights.neighbors.buffer.id,
                offsetsResource: execution.weights.offsets.buffer.id
              }
            }
          ]
        })
      },
      tags,
      replays: execution.replays
    });
  } finally {
    geometry.destroy();
    uploads.forEach(uploaded => uploaded.destroy());
  }
}
