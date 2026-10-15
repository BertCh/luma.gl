// Pure data model for the GPU Core compiler anatomy. Everything shown is read from a recorded
// scenario file (`docs/lens-records/gpu-core.scan.anatomy.json`, decoded with
// `fromRecordedLensFile`); nothing is hand-modelled and unknown fields stay unknown.

/** Id of the recorded scenario the anatomy is rebuilt from. */
export const GPU_CORE_ANATOMY_SCENARIO_ID = 'gpu-core.scan.anatomy';

export type GPUCoreCompilerStageId =
  | 'declared'
  | 'hazards'
  | 'schedule'
  | 'lifetimes'
  | 'allocations'
  | 'encoded'
  | 'slices';

export type GPUCoreCompilerStage = {
  id: GPUCoreCompilerStageId;
  label: string;
  explanation: string;
};

export const GPU_CORE_COMPILER_STAGES: readonly GPUCoreCompilerStage[] = [
  {
    id: 'declared',
    label: 'Declared graph',
    explanation:
      'Contributors have expanded into nodes and logical resources. The application has not recorded any GPU commands yet.'
  },
  {
    id: 'hazards',
    label: 'Derived hazards',
    explanation:
      'Declared reads and writes imply ordering edges. A writer must finish before a later reader or writer touches the same resource range.'
  },
  {
    id: 'schedule',
    label: 'Execution order',
    explanation:
      'Compilation chooses a stable topological order that satisfies explicit dependencies and inferred resource hazards.'
  },
  {
    id: 'lifetimes',
    label: 'Resource lifetimes',
    explanation:
      'Every transient is live from its first scheduled use through its last. Imported resources remain caller-owned.'
  },
  {
    id: 'allocations',
    label: 'Aliasing plan',
    explanation:
      'Compatible transients with non-overlapping lifetimes may share physical storage. Logical identities remain distinct.'
  },
  {
    id: 'encoded',
    label: 'Encoded work',
    explanation:
      'The compiled plan records compute and render work into a caller-owned command encoder and reports actual dispatches and draws.'
  },
  {
    id: 'slices',
    label: 'Frame slices',
    explanation:
      'A budgeted execution advances only legal node boundaries. Oversized indivisible nodes run alone so execution can still make progress.'
  }
];

/** One compiled node with the facts the inspector shows. Unknown fields are `undefined`. */
export type GPUCoreAnatomyNode = {
  id: string;
  label: string;
  kind: string;
  /** Contributor label, such as `GPUCompaction`. */
  contributor: string;
  /** `GPUCompaction › GPUScan` when the node's operator is nested in another contributor. */
  contributorPath: string;
  namespace?: string;
  operator?: string;
  reads: readonly string[];
  writes: readonly string[];
  workgroups?: readonly number[];
  invocationCount?: number;
  indirect?: boolean;
  /** Producing shader module, such as `GPUScan/level-0`, when the record names it. */
  moduleId?: string;
  sourceId?: string;
};

export type GPUCoreAnatomyEdge = {
  from: string;
  to: string;
  kind: string;
  resource?: string;
};

export type GPUCoreAnatomyResource = {
  id: string;
  ownership: 'imported' | 'transient';
  /** Scheduled index of the first node using the resource, when recorded. */
  firstNode?: number;
  lastNode?: number;
  /** Physical allocation id for transients. */
  allocation?: string;
  byteLength?: number;
  allocationByteLength?: number;
};

export type GPUCoreAnatomyContributor = {
  label: string;
  nodeIds: readonly string[];
  /** Nodes whose operator differs from the contributor (nested contributors). */
  nested: readonly {nodeId: string; path: string}[];
};

export type GPUCoreAnatomyEncodedPass = {
  id: string;
  label: string;
  detail: string;
};

export type GPUCoreAnatomySlice = {
  /** Frame index. */
  index: number;
  nodeIds: readonly string[];
  /** Sum of the nodes' recorded invocation counts, when every node recorded one. */
  invocationCount?: number;
  /** A single node whose invocations exceed the budget, so it runs alone. */
  oversized: boolean;
};

export type GPUCoreAnatomySliceSeries = {
  /** Invocation budget parsed from the recorded series name `slices@<budget>`. */
  budget: number;
  slices: readonly GPUCoreAnatomySlice[];
};

export type GPUCoreAnatomyModel = {
  status: 'ready';
  /** Provenance line fields from the recorded envelope and run. */
  provenance: {recordedAt: string; adapterLabel: string; runId: string};
  nodes: readonly GPUCoreAnatomyNode[];
  edges: readonly GPUCoreAnatomyEdge[];
  contributors: readonly GPUCoreAnatomyContributor[];
  resources: readonly GPUCoreAnatomyResource[];
  /** Transients grouped by physical allocation, in first-seen order. */
  allocations: readonly {allocation: string; resourceIds: readonly string[]; byteLength?: number}[];
  allocationSentence: string;
  /** Present only when the record carries an `encoded` graph form. */
  encodedPasses?: readonly GPUCoreAnatomyEncodedPass[];
  encodedNote: string;
  sliceSeries: readonly GPUCoreAnatomySliceSeries[];
};

export type GPUCoreAnatomyUnavailable = {status: 'pending' | 'invalid'; reason: string};

type Loose = Record<string, unknown>;

function isRecordObject(value: unknown): value is Loose {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function asString(value: unknown): string | undefined {
  return typeof value === 'string' ? value : undefined;
}

function asFiniteNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function asStringList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [];
}

function toKebabCase(name: string): string {
  return name
    .replace(/([a-z0-9])([A-Z])/g, '$1-$2')
    .replace(/([A-Z]+)([A-Z][a-z])/g, '$1-$2')
    .toLowerCase();
}

/**
 * Builds the anatomy model from a decoded recorded lens file (`fromRecordedLensFile` output).
 * `undefined` or `null` means the record is not available yet; a value of the wrong shape is invalid.
 */
export function buildGPUCoreAnatomyModel(
  file: unknown
): GPUCoreAnatomyModel | GPUCoreAnatomyUnavailable {
  if (file === undefined || file === null) {
    return {status: 'pending', reason: 'The recorded run has not been loaded.'};
  }
  if (!isRecordObject(file) || file['schema'] !== 'luma.lens-record/1') {
    return {status: 'invalid', reason: 'Not a luma.lens-record/1 file.'};
  }
  const record = file['record'];
  if (!isRecordObject(record) || !Array.isArray(record['graphs'])) {
    return {status: 'invalid', reason: 'The file has no computation record with graphs.'};
  }
  const graphs = record['graphs'].filter(isRecordObject);
  const compiled = graphs.find(graph => graph['form'] === 'compiled');
  if (!compiled || !Array.isArray(compiled['nodes'])) {
    return {status: 'invalid', reason: 'The record has no compiled graph form.'};
  }
  const rawNodes = compiled['nodes'].filter(isRecordObject);
  if (!rawNodes.length || rawNodes.some(node => typeof node['id'] !== 'string')) {
    return {status: 'invalid', reason: 'The compiled graph has no identifiable nodes.'};
  }

  const sourceModules = new Map<string, string>();
  const refs = record['refs'];
  if (isRecordObject(refs) && Array.isArray(refs['sources'])) {
    for (const source of refs['sources']) {
      if (isRecordObject(source) && typeof source['id'] === 'string' && typeof source['moduleId'] === 'string') {
        sourceModules.set(source['id'], source['moduleId']);
      }
    }
  }

  // A namespace such as `gpu-compaction` names the contributor whose operator kebab-cases to it.
  const contributorByNamespace = new Map<string, string>();
  for (const node of rawNodes) {
    const operator = asString(node['op']);
    const topNamespace = asString(node['namespace'])?.split('/')[0];
    if (operator && topNamespace && toKebabCase(operator) === topNamespace) {
      contributorByNamespace.set(topNamespace, operator);
    }
  }

  const nodes: GPUCoreAnatomyNode[] = rawNodes.map(node => {
    const operator = asString(node['op']);
    const namespace = asString(node['namespace']);
    const topNamespace = namespace?.split('/')[0] || undefined;
    const contributor =
      (topNamespace && contributorByNamespace.get(topNamespace)) ||
      topNamespace ||
      operator ||
      'unknown contributor';
    const contributorPath = operator && operator !== contributor ? `${contributor} › ${operator}` : contributor;
    const dispatch = isRecordObject(node['dispatch']) ? node['dispatch'] : undefined;
    const sourceId = asString(node['source']);
    return {
      id: node['id'] as string,
      label: asString(node['label']) ?? (node['id'] as string),
      kind: asString(node['kind']) ?? 'unknown',
      contributor,
      contributorPath,
      namespace,
      operator,
      reads: asStringList(node['reads']),
      writes: asStringList(node['writes']),
      workgroups: Array.isArray(dispatch?.['workgroups'])
        ? (dispatch['workgroups'] as unknown[]).filter((value): value is number => typeof value === 'number')
        : undefined,
      invocationCount: asFiniteNumber(dispatch?.['invocationCount']),
      indirect: typeof dispatch?.['indirect'] === 'boolean' ? dispatch['indirect'] : undefined,
      moduleId: sourceId ? sourceModules.get(sourceId) : undefined,
      sourceId
    };
  });
  const nodeIndexById = new Map(nodes.map((node, index) => [node.id, index]));

  const edges: GPUCoreAnatomyEdge[] = (Array.isArray(compiled['edges']) ? compiled['edges'] : [])
    .filter(isRecordObject)
    .filter(edge => typeof edge['from'] === 'string' && typeof edge['to'] === 'string')
    .map(edge => ({
      from: edge['from'] as string,
      to: edge['to'] as string,
      kind: asString(edge['kind']) ?? 'unknown',
      resource: asString(edge['resource'])
    }));

  const contributors: GPUCoreAnatomyContributor[] = [];
  for (const node of nodes) {
    let contributor = contributors.find(candidate => candidate.label === node.contributor);
    if (!contributor) {
      contributor = {label: node.contributor, nodeIds: [], nested: []};
      contributors.push(contributor);
    }
    (contributor.nodeIds as string[]).push(node.id);
    if (node.contributorPath !== node.contributor) {
      (contributor.nested as {nodeId: string; path: string}[]).push({
        nodeId: node.id,
        path: node.contributorPath
      });
    }
  }

  const resources: GPUCoreAnatomyResource[] = (
    Array.isArray(compiled['resources']) ? compiled['resources'] : []
  )
    .filter(isRecordObject)
    .filter(resource => typeof resource['id'] === 'string')
    .map(resource => {
      const firstNodeId = asString(resource['firstNode']);
      const lastNodeId = asString(resource['lastNode']);
      return {
        id: resource['id'] as string,
        ownership: resource['ownership'] === 'transient' ? 'transient' : 'imported',
        firstNode: firstNodeId !== undefined ? nodeIndexById.get(firstNodeId) : undefined,
        lastNode: lastNodeId !== undefined ? nodeIndexById.get(lastNodeId) : undefined,
        allocation: asString(resource['allocation']),
        byteLength: asFiniteNumber(resource['bytes']),
        allocationByteLength: asFiniteNumber(resource['allocationBytes'])
      };
    });

  const allocations: {allocation: string; resourceIds: string[]; byteLength?: number}[] = [];
  for (const resource of resources) {
    if (resource.ownership !== 'transient' || !resource.allocation) continue;
    let group = allocations.find(candidate => candidate.allocation === resource.allocation);
    if (!group) {
      group = {allocation: resource.allocation, resourceIds: [], byteLength: resource.allocationByteLength};
      allocations.push(group);
    }
    group.resourceIds.push(resource.id);
  }
  const sharedAllocations = allocations.filter(group => group.resourceIds.length > 1);
  const allocationSentence = !resources.some(resource => resource.ownership === 'transient' && resource.allocation)
    ? 'The record does not say how transients map to physical allocations.'
    : sharedAllocations.length === 0
      ? 'In this run no logical transients share a physical allocation.'
      : sharedAllocations
          .map(group => `${formatList(group.resourceIds)} share physical allocation ${group.allocation}`)
          .join('; ') + '.';

  const encodedGraph = graphs.find(graph => graph['form'] === 'encoded');
  let encodedPasses: GPUCoreAnatomyEncodedPass[] | undefined;
  let encodedNote: string;
  if (encodedGraph && Array.isArray(encodedGraph['nodes']) && encodedGraph['nodes'].length) {
    encodedPasses = encodedGraph['nodes'].filter(isRecordObject).map(pass => {
      const id = asString(pass['id']) ?? 'pass';
      const dispatch = isRecordObject(pass['dispatch']) ? pass['dispatch'] : undefined;
      const workgroups = Array.isArray(dispatch?.['workgroups']) ? (dispatch['workgroups'] as unknown[]).join(' × ') : undefined;
      return {
        id,
        label: asString(pass['label']) ?? id,
        detail: workgroups ? `${workgroups} workgroups` : 'dispatch shape not recorded'
      };
    });
    encodedNote = 'Passes of the recorded encoded graph form.';
  } else {
    encodedNote =
      'Encoded form not recorded; showing each compiled node’s recorded dispatch shape instead.';
  }

  const metricsList = Array.isArray(record['metrics']) ? record['metrics'].filter(isRecordObject) : [];
  const sliceSeries: GPUCoreAnatomySliceSeries[] = [];
  for (const metrics of metricsList) {
    for (const series of Array.isArray(metrics['series']) ? metrics['series'].filter(isRecordObject) : []) {
      const match = /^slices@(\d+)$/.exec(asString(series['name']) ?? '');
      const values = series['values'];
      if (!match || !Array.isArray(values) || values.length !== nodes.length) continue;
      const budget = Number(match[1]);
      const frameIndices = values.map(value => asFiniteNumber(value));
      if (frameIndices.some(index => index === undefined)) continue;
      const frames = new Map<number, string[]>();
      for (const [nodeIndex, frame] of frameIndices.entries()) {
        const list = frames.get(frame as number) ?? [];
        list.push(nodes[nodeIndex].id);
        frames.set(frame as number, list);
      }
      const slices = [...frames.entries()]
        .sort((first, second) => first[0] - second[0])
        .map(([index, nodeIds]) => {
          const counts = nodeIds.map(id => nodes[nodeIndexById.get(id)!].invocationCount);
          const known = counts.every(count => count !== undefined);
          const invocationCount = known ? counts.reduce<number>((sum, count) => sum + (count as number), 0) : undefined;
          return {
            index,
            nodeIds,
            invocationCount,
            oversized: nodeIds.length === 1 && (counts[0] ?? 0) > budget
          };
        });
      sliceSeries.push({budget, slices});
    }
  }
  sliceSeries.sort((first, second) => first.budget - second.budget);

  return {
    status: 'ready',
    provenance: {
      recordedAt: asString(file['recordedAt']) ?? 'unknown',
      adapterLabel: asString(file['adapterLabel']) ?? 'unknown',
      runId: asString(record['runId']) ?? 'unknown'
    },
    nodes,
    edges,
    contributors,
    resources,
    allocations,
    allocationSentence,
    encodedPasses,
    encodedNote,
    sliceSeries
  };
}

function formatList(items: readonly string[]): string {
  if (items.length <= 2) return items.join(' and ');
  return `${items.slice(0, -1).join(', ')}, and ${items[items.length - 1]}`;
}
