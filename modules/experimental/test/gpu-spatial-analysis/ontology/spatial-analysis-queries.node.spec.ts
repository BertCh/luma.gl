// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  GPU_SPATIAL_ANALYSIS_AUDIT,
  GPU_SPATIAL_ANALYSIS_CAPABILITIES,
  GPU_SPATIAL_ANALYSIS_COORDINATE_MODELS,
  GPU_SPATIAL_ANALYSIS_DATA_KINDS,
  GPU_SPATIAL_ANALYSIS_DATA_ONTOLOGY,
  GPU_SPATIAL_ANALYSIS_DOMAINS,
  GPU_SPATIAL_ANALYSIS_OPERATORS,
  GPU_SPATIAL_ANALYSIS_STAGES,
  formatGPUSpatialAnalysisAuditMarkdown,
  getGPUSpatialAnalysisCapability,
  getGPUSpatialAnalysisCapabilityConnections,
  getGPUSpatialAnalysisCapabilityForOperator,
  getGPUSpatialAnalysisOperator,
  isGPUSpatialAnalysisDataKind,
  queryGPUSpatialAnalysisCapabilities,
  queryGPUSpatialAnalysisOperators
} from '../../../src/gpu-spatial-analysis/ontology/index';

it('data ontology defines every data kind exactly once with prose', () => {
  const ids = GPU_SPATIAL_ANALYSIS_DATA_ONTOLOGY.map(concept => concept.id);
  expect([...ids].sort()).toEqual([...GPU_SPATIAL_ANALYSIS_DATA_KINDS].sort());
  for (const concept of GPU_SPATIAL_ANALYSIS_DATA_ONTOLOGY) {
    expect(concept.title.trim(), concept.id).not.toBe('');
    expect(concept.description.trim(), concept.id).not.toBe('');
  }
});

it('capabilities only use declared stages, domains, coordinate models and data kinds', () => {
  const kinds = new Set<string>(GPU_SPATIAL_ANALYSIS_DATA_KINDS);
  for (const capability of GPU_SPATIAL_ANALYSIS_CAPABILITIES) {
    const label = capability.id;
    expect(capability.id, label).toMatch(/^[a-z0-9]+(-[a-z0-9]+)*$/);
    expect(GPU_SPATIAL_ANALYSIS_STAGES, label).toContain(capability.stage);
    expect(GPU_SPATIAL_ANALYSIS_DOMAINS, label).toContain(capability.domain);
    for (const model of capability.coordinateModels) {
      expect(GPU_SPATIAL_ANALYSIS_COORDINATE_MODELS, `${label} ${model}`).toContain(model);
    }
    for (const kind of [...capability.inputs, ...capability.outputs]) {
      expect(kinds.has(kind), `${label} ${kind}`).toBe(true);
    }
    // Available families name their exports and carry evidence; opportunities have neither.
    if (capability.status === 'available') {
      expect(capability.operators.length, label).toBeGreaterThan(0);
      expect(capability.evidence, label).toBeDefined();
    } else {
      expect(capability.operators, label).toEqual([]);
      expect(capability.evidence, label).toBeUndefined();
    }
  }
});

it('operator descriptors use declared data kinds and resolve their tags', () => {
  const kinds = new Set<string>(GPU_SPATIAL_ANALYSIS_DATA_KINDS);
  for (const operator of GPU_SPATIAL_ANALYSIS_OPERATORS) {
    for (const kind of [...operator.inputs, ...operator.outputs]) {
      expect(kinds.has(kind), `${operator.exportName} ${kind}`).toBe(true);
    }
    for (const tag of operator.tags) {
      expect(
        getGPUSpatialAnalysisCapability(tag),
        `${operator.exportName} tag ${tag}`
      ).toBeDefined();
    }
    expect(operator.package).toBe('@luma.gl/experimental/gpu-spatial-analysis');
  }
});

it('lookups return undefined for unknown identifiers', () => {
  expect(getGPUSpatialAnalysisCapability('no-such-capability')).toBeUndefined();
  expect(getGPUSpatialAnalysisOperator('GPUNoSuchOperator')).toBeUndefined();
  expect(getGPUSpatialAnalysisCapabilityForOperator('GPUNoSuchOperator')).toBeUndefined();
  expect(getGPUSpatialAnalysisCapabilityConnections('no-such-capability')).toBeUndefined();
  expect(queryGPUSpatialAnalysisCapabilities({search: 'zzz-no-match-zzz'})).toEqual([]);
  expect(queryGPUSpatialAnalysisOperators({search: 'zzz-no-match-zzz'})).toEqual([]);
});

it('empty and whitespace queries return the whole catalogue in order', () => {
  expect(queryGPUSpatialAnalysisCapabilities()).toEqual(GPU_SPATIAL_ANALYSIS_CAPABILITIES);
  expect(queryGPUSpatialAnalysisCapabilities({search: '   '})).toEqual(
    GPU_SPATIAL_ANALYSIS_CAPABILITIES
  );
  expect(queryGPUSpatialAnalysisOperators()).toEqual(GPU_SPATIAL_ANALYSIS_OPERATORS);
  expect(queryGPUSpatialAnalysisOperators({search: '  '})).toEqual(GPU_SPATIAL_ANALYSIS_OPERATORS);
});

it('capability filters are conjunctive and each one narrows by its own field', () => {
  for (const capability of GPU_SPATIAL_ANALYSIS_CAPABILITIES) {
    const byStatus = queryGPUSpatialAnalysisCapabilities({status: capability.status});
    expect(byStatus.every(entry => entry.status === capability.status)).toBe(true);
    expect(queryGPUSpatialAnalysisCapabilities({domain: capability.domain})).toContain(capability);
    expect(queryGPUSpatialAnalysisCapabilities({stage: capability.stage})).toContain(capability);
    for (const input of capability.inputs) {
      expect(queryGPUSpatialAnalysisCapabilities({input})).toContain(capability);
    }
    for (const model of capability.coordinateModels) {
      expect(queryGPUSpatialAnalysisCapabilities({coordinateModel: model})).toContain(capability);
    }
  }
  const both = queryGPUSpatialAnalysisCapabilities({status: 'available', stage: 'relate'});
  expect(both.length).toBeGreaterThan(0);
  expect(both.every(entry => entry.status === 'available' && entry.stage === 'relate')).toBe(true);
  expect(
    queryGPUSpatialAnalysisCapabilities({status: 'opportunity'}).length +
      queryGPUSpatialAnalysisCapabilities({status: 'available'}).length
  ).toBe(GPU_SPATIAL_ANALYSIS_CAPABILITIES.length);
});

it('search is case-insensitive, trims, and matches export names', () => {
  const lower = queryGPUSpatialAnalysisCapabilities({search: 'gpulocalmoran'});
  expect(lower.map(({id}) => id)).toEqual(['spatial-autocorrelation-and-inference']);
  expect(queryGPUSpatialAnalysisCapabilities({search: '  GPULocalMoran  '})).toEqual(lower);
  expect(
    queryGPUSpatialAnalysisOperators({search: 'GPULOCALMORAN'}).map(o => o.exportName)
  ).toContain('GPULocalMoran');
});

it('operator filters narrow by capability, tag, input, output, semantics and evidence', () => {
  const sample = getGPUSpatialAnalysisOperator('GPULocationAllocation')!;
  expect(queryGPUSpatialAnalysisOperators({primaryCapability: sample.primaryCapability})).toContain(
    sample
  );
  expect(queryGPUSpatialAnalysisOperators({primaryCapability: 'no-such-capability'})).toEqual([]);
  expect(queryGPUSpatialAnalysisOperators({input: 'network', semantics: 'model-based'})).toContain(
    sample
  );
  expect(queryGPUSpatialAnalysisOperators({semantics: 'exact', input: 'network'})).not.toContain(
    sample
  );
  expect(queryGPUSpatialAnalysisOperators({evidence: 'proven'})).toContain(sample);
  for (const operator of queryGPUSpatialAnalysisOperators({evidence: 'partial'})) {
    expect(operator.evidence.status).toBe('partial');
  }
  const tagged = GPU_SPATIAL_ANALYSIS_OPERATORS.find(operator => operator.tags.length > 0);
  if (tagged) {
    const result = queryGPUSpatialAnalysisOperators({tag: tagged.tags[0]});
    expect(result).toContain(tagged);
    expect(result.every(operator => operator.tags.includes(tagged.tags[0]))).toBe(true);
  }
  for (const model of GPU_SPATIAL_ANALYSIS_COORDINATE_MODELS) {
    for (const operator of queryGPUSpatialAnalysisOperators({coordinateModel: model})) {
      expect(operator.coordinateModels).toContain(model);
    }
  }
});

it('connections resolve related capabilities in curated order and skip dangling IDs', () => {
  for (const capability of GPU_SPATIAL_ANALYSIS_CAPABILITIES) {
    const connections = getGPUSpatialAnalysisCapabilityConnections(capability.id)!;
    expect(connections.capability).toBe(capability);
    expect(connections.related.map(({id}) => id)).toEqual(capability.relatedCapabilities ?? []);
  }
});

it('operator-to-capability lookup agrees with the operator descriptors', () => {
  for (const operator of GPU_SPATIAL_ANALYSIS_OPERATORS) {
    expect(getGPUSpatialAnalysisCapabilityForOperator(operator.exportName)?.id).toBe(
      operator.primaryCapability
    );
  }
});

it('audit counts are consistent with the catalogues and the markdown table', () => {
  const audit = GPU_SPATIAL_ANALYSIS_AUDIT;
  expect(audit.capabilityFamilyCount).toBe(GPU_SPATIAL_ANALYSIS_CAPABILITIES.length);
  expect(audit.availableCapabilityFamilyCount + audit.opportunityCount).toBe(
    audit.capabilityFamilyCount
  );
  expect(audit.operatorCount).toBe(GPU_SPATIAL_ANALYSIS_OPERATORS.length);
  expect(audit.provenOperatorCount + audit.partialOperatorCount).toBeLessThanOrEqual(
    audit.operatorCount
  );
  expect(Object.values(audit.operatorsByCapability).reduce((sum, count) => sum + count, 0)).toBe(
    audit.operatorCount
  );
  const lines = formatGPUSpatialAnalysisAuditMarkdown().split('\n');
  expect(lines.slice(0, 2)).toEqual([
    '| Capability | Catalogue status | Operators | Proven |',
    '| --- | --- | ---: | ---: |'
  ]);
  expect(lines).toHaveLength(2 + GPU_SPATIAL_ANALYSIS_CAPABILITIES.length);
  GPU_SPATIAL_ANALYSIS_CAPABILITIES.forEach((capability, index) => {
    const operators = GPU_SPATIAL_ANALYSIS_OPERATORS.filter(
      operator => operator.primaryCapability === capability.id
    );
    const proven = operators.filter(operator => operator.evidence.status === 'proven').length;
    expect(lines[2 + index]).toBe(
      `| ${capability.id} | ${capability.status} | ${operators.length} | ${proven} |`
    );
  });
});

it('isGPUSpatialAnalysisDataKind accepts catalogue kinds and rejects other strings', () => {
  for (const capability of GPU_SPATIAL_ANALYSIS_CAPABILITIES) {
    for (const kind of [...capability.inputs, ...capability.outputs]) {
      expect(isGPUSpatialAnalysisDataKind(kind)).toBe(true);
    }
  }
  expect(isGPUSpatialAnalysisDataKind('not-a-data-kind')).toBe(false);
  expect(isGPUSpatialAnalysisDataKind('')).toBe(false);
});
