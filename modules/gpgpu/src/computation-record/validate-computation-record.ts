// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_COMPUTATION_RECORD_SCHEMA,
  GPU_COMPUTATION_METRICS_SCHEMA,
  GPU_COMPUTATION_CAPTURES_SCHEMA
} from './computation-record-types';
import {parseGPUComputationNodeRef} from './computation-node-ref';
import type {
  GPUComputationRecord,
  GPUComputationMetrics,
  GPUComputationCaptures
} from './computation-record-types';

/** One diagnostic at a JSON-style record or sidecar path. */
export type GPUComputationValidationIssue = {
  level: 'error' | 'warning';
  path: string;
  message: string;
};
/** Validation succeeds when no error-level diagnostics were found. */
export type GPUComputationValidationReport = {
  valid: boolean;
  issues: readonly GPUComputationValidationIssue[];
};

type DataObject = Record<string, unknown>;

function isObject(value: unknown): value is DataObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function createValidator() {
  const issues: GPUComputationValidationIssue[] = [];
  const issue = (path: string, message: string, level: 'error' | 'warning' = 'error'): void => {
    issues.push({level, path, message});
  };
  const object = (value: unknown, path: string): DataObject => {
    if (isObject(value)) return value;
    issue(path, 'Expected an object');
    return {};
  };
  const array = (value: unknown, path: string): unknown[] => {
    if (Array.isArray(value)) return value;
    issue(path, 'Expected an array');
    return [];
  };
  const objects = (value: unknown, path: string): DataObject[] =>
    array(value, path).map((entry, index) => object(entry, `${path}[${index}]`));
  const strings = (value: DataObject, names: string[], path: string): void => {
    for (const name of names) {
      if (typeof value[name] !== 'string' || (name !== 'namespace' && value[name] === '')) {
        issue(
          path ? `${path}.${name}` : name,
          'Expected a string' + (name === 'namespace' ? '' : ' with content')
        );
      }
    }
  };
  const unique = (values: DataObject[], path: string): Set<unknown> => {
    const ids = new Set<unknown>();
    values.forEach((value, index) => {
      strings(value, ['id'], `${path}[${index}]`);
      if (ids.has(value['id'])) issue(`${path}[${index}].id`, 'Duplicate id');
      ids.add(value['id']);
    });
    return ids;
  };
  const reference = (value: unknown, ids: Set<unknown>, path: string): void => {
    if (typeof value !== 'string' || !ids.has(value)) issue(path, 'Unknown reference');
  };
  return {
    issues,
    issue,
    object,
    array,
    objects,
    strings,
    unique,
    reference,
    report: (): GPUComputationValidationReport => ({
      valid: !issues.some(entry => entry.level === 'error'),
      issues
    })
  };
}

/** Validates JSON record structure and references without throwing on malformed input. */
export function validateGPUComputationRecord(record: unknown): GPUComputationValidationReport {
  const validator = createValidator();
  const {issue, object, array, objects, strings, unique, reference} = validator;
  // JSON input cannot have throwing accessors, but unknown callers may supply them.
  try {
    const root = object(record, '');
    if (root['schema'] !== GPU_COMPUTATION_RECORD_SCHEMA)
      issue('schema', 'Unsupported record schema');
    strings(root, ['id', 'runId', 'createdAt'], '');
    strings(object(root['adapter'], 'adapter'), ['backend'], 'adapter');
    array(root['domains'], 'domains').forEach((domain, index) => {
      if (typeof domain !== 'string' || !domain)
        issue(`domains[${index}]`, 'Expected a domain string');
    });
    const graphs = objects(root['graphs'], 'graphs');
    const graphIds = unique(graphs, 'graphs');
    const nodesByGraph = graphs.map((graph, index) =>
      objects(graph['nodes'], `graphs[${index}].nodes`)
    );
    const nodeIdsByGraph = nodesByGraph.map((nodes, index) =>
      unique(nodes, `graphs[${index}].nodes`)
    );
    const axes = object(root['axes'], 'axes');
    const axisEntries = objects(axes['axes'], 'axes.axes');
    const keySpaces = objects(axes['keySpaces'], 'axes.keySpaces');
    const relations = objects(axes['relations'], 'axes.relations');
    const axisIds = unique(axisEntries, 'axes.axes');
    const keySpaceIds = unique(keySpaces, 'axes.keySpaces');
    const relationIds = unique(relations, 'axes.relations');
    axisEntries.forEach((axis, index) => {
      const path = `axes.axes[${index}]`;
      strings(axis, ['label', 'kind'], path);
      if (axis['parent'] !== undefined) reference(axis['parent'], axisIds, `${path}.parent`);
      if (axis['keySpace'] !== undefined)
        reference(axis['keySpace'], keySpaceIds, `${path}.keySpace`);
    });
    keySpaces.forEach((space, index) => strings(space, ['label'], `axes.keySpaces[${index}]`));
    relations.forEach((relation, index) => {
      const path = `axes.relations[${index}]`;
      strings(relation, ['kind', 'cardinality'], path);
      reference(relation['from'], keySpaceIds, `${path}.from`);
      reference(relation['to'], keySpaceIds, `${path}.to`);
      if (relation['ref'] !== undefined && relation['kind'] !== 'unknown') {
        const mapping = object(relation['ref'], `${path}.ref`);
        strings(
          mapping,
          relation['kind'] === 'function' ? ['function'] : ['layout', 'resource'],
          `${path}.ref`
        );
      }
      if (relation['kind'] === 'unknown' && relation['ref'] !== undefined)
        issue(`${path}.ref`, 'Unknown relations should not specify a mapping', 'warning');
    });
    const refs = object(root['refs'], 'refs');
    const formulas = objects(refs['formulas'], 'refs.formulas');
    const sources = objects(refs['sources'], 'refs.sources');
    const formulaIds = unique(formulas, 'refs.formulas');
    const sourceIds = unique(sources, 'refs.sources');
    sources.forEach((source, index) => {
      strings(source, ['language', 'moduleId', 'hash'], `refs.sources[${index}]`);
      if (formulaIds.has(source['id'])) issue(`refs.sources[${index}].id`, 'Duplicate ref id');
    });
    formulas.forEach((formula, index) => {
      const path = `refs.formulas[${index}]`;
      strings(formula, ['tex', 'ascii', 'derivation'], path);
      objects(formula['symbols'], `${path}.symbols`).forEach((symbol, symbolIndex) => {
        const symbolPath = `${path}.symbols[${symbolIndex}]`;
        strings(symbol, ['id', 'role'], symbolPath);
        if (symbol['binding'] === undefined)
          issue(`${symbolPath}.binding`, 'Formula symbol has no binding', 'warning');
      });
    });
    graphs.forEach((graph, graphIndex) => {
      const path = `graphs[${graphIndex}]`;
      strings(graph, ['domain'], path);
      if (typeof graph['id'] === 'string' && graph['id'].includes(':'))
        issue(`${path}.id`, 'Graph ids cannot contain a colon');
      if (!['intent', 'lowered', 'compiled', 'encoded'].includes(String(graph['form'])))
        issue(`${path}.form`, 'Invalid graph form');
      const nodes = nodesByGraph[graphIndex];
      const nodeIds = nodeIdsByGraph[graphIndex];
      const resources =
        graph['resources'] === undefined
          ? undefined
          : objects(graph['resources'], `${path}.resources`);
      const resourceIds = resources && unique(resources, `${path}.resources`);
      if (graph['form'] === 'compiled' && resources === undefined)
        issue(`${path}.resources`, 'Compiled graph has no resources', 'warning');
      nodes.forEach((node, nodeIndex) => {
        const nodePath = `${path}.nodes[${nodeIndex}]`;
        strings(node, ['label', 'namespace', 'kind'], nodePath);
        object(node['attrs'], `${nodePath}.attrs`);
        if (node['condition'] !== undefined) {
          strings(
            object(node['condition'], `${nodePath}.condition`),
            ['id', 'source', 'mode'],
            `${nodePath}.condition`
          );
        }
        if (node['publication'] !== undefined) {
          strings(
            object(node['publication'], `${nodePath}.publication`),
            ['id', 'completeness'],
            `${nodePath}.publication`
          );
        }
        for (const access of ['reads', 'writes']) {
          array(node[access], `${nodePath}.${access}`).forEach((resource, index) => {
            if (resourceIds) reference(resource, resourceIds, `${nodePath}.${access}[${index}]`);
            else if (typeof resource !== 'string')
              issue(`${nodePath}.${access}[${index}]`, 'Expected a resource id');
          });
        }
        if (node['bindings'] !== undefined)
          objects(node['bindings'], `${nodePath}.bindings`).forEach((binding, index) => {
            strings(binding, ['name', 'resource', 'access'], `${nodePath}.bindings[${index}]`);
            if (resourceIds)
              reference(
                binding['resource'],
                resourceIds,
                `${nodePath}.bindings[${index}].resource`
              );
          });
        for (const [name, ids] of [
          ['source', sourceIds],
          ['formula', formulaIds]
        ] as const) {
          if (node[name] !== undefined) reference(node[name], ids, `${nodePath}.${name}`);
        }
        if (node['derivedFrom'] !== undefined)
          array(node['derivedFrom'], `${nodePath}.derivedFrom`).forEach((ref, index) => {
            const refPath = `${nodePath}.derivedFrom[${index}]`;
            try {
              if (typeof ref !== 'string') throw new Error();
              const parsed = parseGPUComputationNodeRef(ref);
              const originIndex = graphs.findIndex(origin => origin['id'] === parsed.graph);
              if (
                originIndex < 0 ||
                originIndex >= graphIndex ||
                !nodeIdsByGraph[originIndex].has(parsed.node)
              ) {
                issue(refPath, 'Expected a node in an earlier graph');
              }
            } catch {
              issue(refPath, 'Expected <graphId>:<nodeId>');
            }
          });
        if (node['loop'] !== undefined) {
          const loop = object(node['loop'], `${nodePath}.loop`);
          strings(loop, ['id'], `${nodePath}.loop`);
          if (!Number.isInteger(loop['iteration']) || Number(loop['iteration']) < 0)
            issue(`${nodePath}.loop.iteration`, 'Expected a non-negative integer');
          if (loop['maximumIterations'] !== undefined) {
            if (
              !Number.isInteger(loop['maximumIterations']) ||
              Number(loop['maximumIterations']) < 0
            )
              issue(`${nodePath}.loop.maximumIterations`, 'Expected a non-negative integer');
            else if (Number(loop['iteration']) >= Number(loop['maximumIterations']))
              issue(`${nodePath}.loop.iteration`, 'Iteration exceeds the loop bound');
          }
        }
        if (node['datum'] !== undefined) {
          const datum = object(node['datum'], `${nodePath}.datum`);
          reference(datum['keySpace'], keySpaceIds, `${nodePath}.datum.keySpace`);
          if (datum['relation'] !== undefined)
            reference(datum['relation'], relationIds, `${nodePath}.datum.relation`);
        }
      });
      objects(graph['edges'], `${path}.edges`).forEach((edge, index) => {
        const edgePath = `${path}.edges[${index}]`;
        reference(edge['from'], nodeIds, `${edgePath}.from`);
        reference(edge['to'], nodeIds, `${edgePath}.to`);
        if (
          !['dataflow', 'explicit', 'raw', 'war', 'waw', 'indirect'].includes(String(edge['kind']))
        )
          issue(`${edgePath}.kind`, 'Invalid edge kind');
      });
      resources?.forEach((resource, index) => {
        const resourcePath = `${path}.resources[${index}]`;
        strings(resource, ['kind', 'ownership'], resourcePath);
        for (const endpoint of ['firstNode', 'lastNode']) {
          if (resource[endpoint] !== undefined)
            reference(resource[endpoint], nodeIds, `${resourcePath}.${endpoint}`);
        }
        const first = nodes.findIndex(node => node['id'] === resource['firstNode']);
        const last = nodes.findIndex(node => node['id'] === resource['lastNode']);
        if (first >= 0 && last >= 0 && first > last)
          issue(`${resourcePath}.lastNode`, 'Lifetime ends before it starts');
        if (resource['allocation'] === undefined || resource['retained'] || first < 0 || last < 0)
          return;
        resources.slice(0, index).forEach(previous => {
          if (previous['allocation'] !== resource['allocation'] || previous['retained']) return;
          const previousFirst = nodes.findIndex(node => node['id'] === previous['firstNode']);
          const previousLast = nodes.findIndex(node => node['id'] === previous['lastNode']);
          if (
            previousFirst >= 0 &&
            previousLast >= 0 &&
            first <= previousLast &&
            previousFirst <= last
          ) {
            issue(`${resourcePath}.allocation`, 'Aliased resource lifetimes overlap');
          }
        });
      });
    });
    objects(root['decisions'], 'decisions').forEach((decision, index) => {
      const path = `decisions[${index}]`;
      strings(decision, ['id', 'subject', 'kind', 'chosen'], path);
      const candidates = objects(decision['candidates'], `${path}.candidates`);
      candidates.forEach((candidate, candidateIndex) =>
        strings(candidate, ['id', 'reason'], `${path}.candidates[${candidateIndex}]`)
      );
      reference(
        decision['chosen'],
        new Set(candidates.map(candidate => candidate['id'])),
        `${path}.chosen`
      );
    });
    objects(root['choices'], 'choices').forEach((choice, index) =>
      strings(choice, ['stage', 'name'], `choices[${index}]`)
    );
    if (root['workflow'] !== undefined) {
      const workflow = object(root['workflow'], 'workflow');
      strings(workflow, ['id'], 'workflow');
      objects(workflow['invocations'], 'workflow.invocations').forEach((invocation, index) =>
        reference(invocation['graph'], graphIds, `workflow.invocations[${index}].graph`)
      );
    }
  } catch {
    issue('', 'Unable to read record data');
  }
  return validator.report();
}

/** Checks sidecar schemas, run identity, graph references, timestamps and capture dimensions. */
export function validateGPUComputationSidecars(
  record: GPUComputationRecord,
  sidecars: {
    metrics?: readonly GPUComputationMetrics[];
    captures?: GPUComputationCaptures;
  }
): GPUComputationValidationReport {
  const validator = createValidator();
  const {issue} = validator;
  for (const [index, metrics] of (sidecars.metrics ?? []).entries()) {
    const path = `metrics[${index}]`;
    if (metrics.schema !== GPU_COMPUTATION_METRICS_SCHEMA)
      issue(`${path}.schema`, 'Unsupported metrics schema');
    if (metrics.runId !== record.runId) issue(`${path}.runId`, 'Run id does not match');
    const graph = record.graphs.find(candidate => candidate.id === metrics.graph);
    if (!graph) issue(`${path}.graph`, 'Unknown graph');
    const nodeIds = new Set(graph?.nodes.map(node => node.id));
    for (const [id, node] of Object.entries(metrics.nodes)) {
      const nodePath = `${path}.nodes[${JSON.stringify(id)}]`;
      if (!nodeIds.has(id)) issue(nodePath, 'Unknown node');
      if (node.gpuBegin !== undefined && node.gpuEnd !== undefined && node.gpuEnd < node.gpuBegin)
        issue(`${nodePath}.gpuEnd`, 'GPU end precedes begin');
    }
  }
  const captures = sidecars.captures;
  if (captures) {
    if (captures.schema !== GPU_COMPUTATION_CAPTURES_SCHEMA)
      issue('captures.schema', 'Unsupported captures schema');
    if (captures.runId !== record.runId) issue('captures.runId', 'Run id does not match');
    for (const category of ['snapshots', 'stamps', 'replays'] as const) {
      captures[category].forEach((capture, index) => {
        const path = `captures.${category}[${index}]`;
        const graphs =
          capture.graph === undefined
            ? record.graphs
            : record.graphs.filter(graph => graph.id === capture.graph);
        if (!graphs.length && capture.graph !== undefined) issue(`${path}.graph`, 'Unknown graph');
        if (
          capture.resource !== undefined &&
          !graphs.some(graph => graph.resources?.some(resource => resource.id === capture.resource))
        )
          issue(`${path}.resource`, 'Unknown resource');
        if (
          capture.afterNode !== undefined &&
          !graphs.some(graph => graph.nodes.some(node => node.id === capture.afterNode))
        )
          issue(`${path}.afterNode`, 'Unknown node');
        if (
          capture.keySpace !== undefined &&
          !record.axes.keySpaces.some(space => space.id === capture.keySpace)
        )
          issue(`${path}.keySpace`, 'Unknown key space');
        const components = capture.components ?? 1;
        if (!Number.isInteger(components) || components <= 0)
          issue(`${path}.components`, 'Expected a positive integer');
        else if (capture.values.length % components !== 0)
          issue(`${path}.values`, 'Values must contain whole components');
      });
    }
  }
  return validator.report();
}
