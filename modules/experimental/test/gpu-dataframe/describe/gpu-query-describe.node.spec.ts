// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {hashGPUComputationText} from '@luma.gl/gpgpu';
import {GPUCommandGraph, createTransientView} from '@luma.gl/gpgpu/gpu-core';
import {
  and,
  parameter,
  type GPUDataFrameQueryParameters
} from '@luma.gl/experimental/gpu-dataframe';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {addGPUQueryPredicatePass} from '../../../src/gpu-dataframe/gpu-query-compiler';
import {
  describeGPUDataFrameQueryIntent,
  describeGPUDataFrameQueryShaderPlan,
  createGPUDataFrameExpressionFormulas,
  type GPUDataFrameQueryDescriptionInput
} from '../../../src/gpu-dataframe/gpu-query-describe';
import {listTexTerms} from '../../../src/computation-views/views/formula-tex';
import {GPURecordBatch, GPUTable} from '@luma.gl/experimental/gpu-tables';
import {Buffer} from '@luma.gl/core';
import {GPUData} from '@luma.gl/gpgpu/gpu-data';
import {GPUDataFrame, column, literal} from '@luma.gl/experimental/gpu-dataframe';
import {makeGPUQueryExpressionShaderPlan} from '../../../src/gpu-dataframe/gpu-expression-shader';

function createFixture() {
  const table = new GPUTable<{fare: 'float32'}>({
    schema: {fields: [{name: 'fare', format: 'float32', nullable: true}], metadata: new Map()},
    bufferLayout: [{name: 'fare', format: 'float32'}]
  });
  const source = new GPUDataFrame({table});
  const query = source
    .withColumn('adjustedFare', column('fare').multiply(literal(1.2)))
    .withColumn('serviceCharge', column('adjustedFare').subtract(column('fare')))
    .filter(column('serviceCharge').greaterThan(literal(1)))
    .select(['adjustedFare', 'serviceCharge']);
  const plan = makeGPUQueryExpressionShaderPlan(
    source,
    query.predicates,
    query.derivedColumns,
    query.columnNames
  );
  return {source, table, query, plan};
}

test('retains the pre-instrumentation WGSL statements and condition byte for byte', () => {
  const {source, table, plan} = createFixture();
  expect([...plan.statements, plan.condition].join('\n')).toMatchInlineSnapshot(`
    "let expression0Value: f32 = input0[INPUT_0_OFFSET + index];
    let expression0Valid: bool = validity0[VALIDITY_0_OFFSET + index] != 0u;
    let expression1Value: f32 = bitcast<f32>(queryControls[CONTROL_OFFSET + 0u]);
    let expression1Valid: bool = queryControls[CONTROL_OFFSET + 1u] != 0u;
    let expression2Value: f32 = (expression0Value * expression1Value);
    let expression2Valid: bool = expression0Valid && expression1Valid;
    let expression3Value: f32 = input0[INPUT_0_OFFSET + index];
    let expression3Valid: bool = validity0[VALIDITY_0_OFFSET + index] != 0u;
    let expression4Value: f32 = (expression2Value - expression3Value);
    let expression4Valid: bool = expression2Valid && expression3Valid;
    let expression5Value: f32 = bitcast<f32>(queryControls[CONTROL_OFFSET + 2u]);
    let expression5Valid: bool = queryControls[CONTROL_OFFSET + 3u] != 0u;
    let expression6Value: bool = (expression4Value > expression5Value);
    let expression6Valid: bool = expression4Valid && expression5Valid;
    (expression6Valid && expression6Value)"
  `);
  source.destroy();
  table.destroy();
});

function descriptionInput(
  fixture: ReturnType<typeof createFixture>
): GPUDataFrameQueryDescriptionInput {
  return {
    queryId: 'proof',
    plan: fixture.plan,
    predicates: fixture.query.predicates.map(predicate => predicate.node),
    derivedColumns: fixture.query.derivedColumns.map(column => ({
      name: column.name,
      node: column.expression.node
    })),
    selectedColumns: fixture.query.columnNames,
    batchRows: [3, 2],
    graphRecording: true
  };
}

test('maps AST paths, memoised derived references, operands and materialized roots exactly', () => {
  const fixture = createFixture();
  expect(
    fixture.plan.expressions.map(({index, operands, astNodeIds, kind}) => ({
      index,
      operands,
      astNodeIds,
      kind
    }))
  ).toEqual([
    {index: 0, operands: [], astNodeIds: ['derived-adjustedFare/left'], kind: 'column'},
    {index: 1, operands: [], astNodeIds: ['derived-adjustedFare/right'], kind: 'control'},
    {
      index: 2,
      operands: [0, 1],
      astNodeIds: ['derived-adjustedFare', 'derived-serviceCharge/left'],
      kind: 'binary'
    },
    {index: 3, operands: [], astNodeIds: ['derived-serviceCharge/right'], kind: 'column'},
    {
      index: 4,
      operands: [2, 3],
      astNodeIds: ['derived-serviceCharge', 'predicate-0/left'],
      kind: 'binary'
    },
    {index: 5, operands: [], astNodeIds: ['predicate-0/right'], kind: 'control'},
    {index: 6, operands: [4, 5], astNodeIds: ['predicate-0'], kind: 'binary'}
  ]);
  expect(fixture.plan.roots).toEqual([
    {
      kind: 'derived',
      name: 'adjustedFare',
      astNodeId: 'derived-adjustedFare',
      expression: 2,
      output: 0
    },
    {
      kind: 'derived',
      name: 'serviceCharge',
      astNodeId: 'derived-serviceCharge',
      expression: 4,
      output: 1
    },
    {kind: 'predicate', index: 0, astNodeId: 'predicate-0', expression: 6}
  ]);
  fixture.source.destroy();
  fixture.table.destroy();
});

test('prints golden formulas from the AST, with a bijection between term ids and symbols', () => {
  const fixture = createFixture();
  const input = descriptionInput(fixture);
  const formulas = createGPUDataFrameExpressionFormulas(input);
  expect(formulas.map(({tex, ascii}) => ({tex, ascii}))).toMatchInlineSnapshot(`
    [
      {
        "ascii": "adjustedFare[i] = fare[i] * 1.2; null if any operand is null",
        "tex": "\\term{adjustedFare}{\\mathit{adjustedFare}}_i = \\term{expression2}{\\term{fare}{\\mathit{fare}}_i \\times 1.2}",
      },
      {
        "ascii": "serviceCharge[i] = adjustedFare[i] - fare[i]; null if any operand is null",
        "tex": "\\term{serviceCharge}{\\mathit{serviceCharge}}_i = \\term{expression4}{\\term{adjustedFare}{\\mathit{adjustedFare}}_i - \\term{fare}{\\mathit{fare}}_i}",
      },
      {
        "ascii": "keep[i] = 1[valid(P[i]) and P[i]]; P[i] = serviceCharge[i] > 1; three-valued logic keeps only valid true predicates (false and null are dropped)",
        "tex": "\\term{keep}{\\mathrm{keep}}_i = \\mathbb{1}\\left[\\operatorname{valid}(P_i) \\land P_i\\right],\\; P_i = \\term{expression6}{\\term{serviceCharge}{\\mathit{serviceCharge}}_i > 1}",
      },
    ]
  `);
  for (const formula of formulas) {
    const terms = new Set(listTexTerms(formula.tex).map(term => term.symbolId));
    expect(terms).toEqual(new Set(formula.symbols.map(symbol => symbol.id)));
    expect(formula.symbols.length).toBe(terms.size);
    expect(formula.derivation).toBe('derived');
  }
  const intent = describeGPUDataFrameQueryIntent(input);
  const lowered = describeGPUDataFrameQueryShaderPlan(input);
  const intentIds = new Set(intent.nodes.map(node => `${intent.id}:${node.id}`));
  expect(lowered.nodes.length).toBe(fixture.plan.expressions.length + 3);
  for (const node of lowered.nodes) {
    expect(node.derivedFrom?.length).toBeGreaterThan(0);
    for (const reference of node.derivedFrom ?? []) expect(intentIds.has(reference)).toBe(true);
  }
  for (const graph of [intent, lowered]) {
    const ids = new Set(graph.nodes.map(node => node.id));
    expect(ids.size).toBe(graph.nodes.length);
    for (const edge of graph.edges) {
      expect(ids.has(edge.from)).toBe(true);
      expect(ids.has(edge.to)).toBe(true);
    }
  }
  fixture.source.destroy();
  fixture.table.destroy();
});

function recordPredicate(recording: boolean) {
  const fixture = createFixture();
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph<GPUDataFrameQueryParameters>(device, {
    id: 'proof',
    ...(recording ? {recording: {sourceText: true}} : {})
  });
  const vector = <Format extends 'float32' | 'sint32' | 'uint32'>(id: string, format: Format) =>
    createVectorView(id, format, [createTransientView(graph, id, format, 3)]);
  const sourceViews = new Map([
    ['fare', {values: vector('fare', 'float32'), validity: vector('fare-valid', 'uint32')}]
  ]);
  const derivedViews = new Map(
    fixture.plan.outputs.map(output => [
      output.name,
      {
        values: vector(output.name, output.format),
        validity: vector(`${output.name}-valid`, 'uint32')
      }
    ])
  );
  const output = createTransientView(graph, 'keep', 'uint32', 3);
  const controls = createTransientView(
    graph,
    'controls',
    'uint32',
    fixture.plan.controls.length * 2
  );
  const addPass = vi.spyOn(graph, 'addComputePass');
  addGPUQueryPredicatePass(graph, {
    id: 'predicate',
    queryId: 'proof',
    batchIndex: 0,
    columns: fixture.plan.columns,
    sourceViews,
    derivedViews,
    output,
    controls,
    plan: fixture.plan
  });
  const node = addPass.mock.calls[0][0];
  addPass.mockRestore();
  const sources = getGPUCommandNodeKernelSources([node]);
  return {fixture, device, graph, node, source: sources[0].source};
}

test('recording annotates exact WGSL, bindings and lineage without changing production nodes or kernels', () => {
  const recorded = recordPredicate(true);
  const production = recordPredicate(false);
  expect('annotation' in production.node).toBe(false);
  expect(
    Object.keys(recorded.node)
      .filter(key => key !== 'annotation')
      .sort()
  ).toEqual(Object.keys(production.node).sort());
  expect(recorded.source).toBe(production.source);
  const resourceUses = (node: typeof recorded.node) =>
    node.resources?.map(resource => ({
      id:
        'buffer' in resource
          ? 'buffer' in resource.buffer
            ? resource.buffer.buffer.id
            : resource.buffer.id
          : '',
      usage: resource.usage
    }));
  expect(resourceUses(recorded.node)).toEqual(resourceUses(production.node));
  const annotation = recorded.node.annotation!;
  expect(annotation.op).toBe('GPUDataFrameQuery');
  // Computation's shader assembler prepends blank module separators to the supplied source.
  expect(recorded.source).toBe(`\n\n\n${annotation.source?.text}`);
  expect(annotation.source?.hash).toBe(hashGPUComputationText(annotation.source!.text!));
  expect(annotation.workgroupSize).toEqual([256, 1, 1]);
  expect(annotation.dispatch).toEqual({workgroups: [1, 1, 1], invocationCount: 3});
  expect(annotation.datum).toEqual({keySpace: 'row'});
  expect(
    annotation.bindings?.map(binding => [
      binding.name,
      binding.elementType,
      binding.access,
      binding.elementOffset,
      binding.length
    ])
  ).toEqual([
    ['input0', 'f32', 'read', 0, 3],
    ['validity0', 'u32', 'read', 0, 3],
    ['queryControls', 'u32', 'read', 0, 4],
    ['derived0', 'f32', 'read_write', 0, 3],
    ['derivedValidity0', 'u32', 'read_write', 0, 3],
    ['derived1', 'f32', 'read_write', 0, 3],
    ['derivedValidity1', 'u32', 'read_write', 0, 3],
    ['outputMask', 'u32', 'read_write', 0, 3]
  ]);
  const input = descriptionInput(recorded.fixture);
  const lowered = describeGPUDataFrameQueryShaderPlan(input);
  expect(annotation.derivedFrom).toEqual(lowered.nodes.map(node => `${lowered.id}:${node.id}`));
  for (const formula of createGPUDataFrameExpressionFormulas(input)) {
    for (const symbol of formula.symbols) {
      expect(symbol.binding).toMatch(/^wgsl:/);
      expect(recorded.source).toMatch(new RegExp(`\\b${symbol.binding!.slice(5)}\\b`));
    }
  }
  for (const result of [recorded, production]) {
    result.fixture.source.destroy();
    result.fixture.table.destroy();
    result.device.destroy();
  }
});

test('format inference gives early derived expressions their own root path', () => {
  const fixture = createFixture();
  const plan = makeGPUQueryExpressionShaderPlan(
    fixture.source,
    fixture.query.predicates,
    fixture.query.derivedColumns,
    ['serviceCharge']
  );
  expect(plan.expressions[2].astNodeIds).toEqual([
    'derived-adjustedFare',
    'derived-serviceCharge/left'
  ]);
  expect(plan.roots[0]).toEqual({
    kind: 'derived',
    name: 'adjustedFare',
    astNodeId: 'derived-adjustedFare',
    expression: 2
  });
  expect(
    createGPUDataFrameExpressionFormulas({...descriptionInput(fixture), plan})[0].symbols[0].binding
  ).toBe('wgsl:expression2Value');
  fixture.source.destroy();
  fixture.table.destroy();
});

test('parenthesizes right operands and nested comparisons; unary validity removes null propagation', () => {
  const fixture = createFixture();
  const expression = column('fare')
    .subtract(column('fare').subtract(literal(2)))
    .divide(column('fare').add(literal(1)));
  const predicates = [
    and(expression.greaterThan(parameter('minimumFare', 10)), column('fare').isValid())
  ];
  const derivedColumns = [{name: 'result', expression}];
  const plan = makeGPUQueryExpressionShaderPlan(fixture.source, predicates, derivedColumns, [
    'result'
  ]);
  const formulas = createGPUDataFrameExpressionFormulas({
    ...descriptionInput(fixture),
    plan,
    predicates: predicates.map(predicate => predicate.node),
    derivedColumns: [{name: 'result', node: expression.node}]
  });
  expect(formulas[0].ascii).toBe(
    'result[i] = (fare[i] - (fare[i] - 2)) / (fare[i] + 1); null if any operand is null'
  );
  expect(formulas[0].tex).toContain('\\frac{');
  expect(formulas[1].tex).toContain('\\term{minimumFare}{\\mathrm{minimumFare}}');
  expect(formulas[1].symbols.find(symbol => symbol.id === 'minimumFare')?.binding).toBe(
    'wgsl:queryControls'
  );
  expect(plan.expressions.find(statement => statement.operator === 'is-valid')?.nullable).toBe(
    false
  );
  fixture.source.destroy();
  fixture.table.destroy();
});

test('compiled describe exposes graph forms and options without submitting GPU work', () => {
  const fixture = createFixture();
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph<GPUDataFrameQueryParameters>(device, {id: 'empty-proof'});
  const submit = vi.spyOn(device, 'submit');
  const compiled = fixture.query.compile(graph);
  try {
    const description = compiled.describe({graphId: 'custom-command-graph', label: 'Proof'});
    expect(description.intent.id).toBe('empty-proof-gpu-dataframe-query-intent');
    expect(description.lowered.id).toBe('empty-proof-gpu-dataframe-query-shader-plan');
    expect(description.compiled.graph.id).toBe('custom-command-graph');
    expect(description.compiled.graph.label).toBe('Proof');
    expect(description.compiled.sources).toEqual([]);
    expect(description.keySpaces).toEqual([
      {id: 'row', label: 'Source row', size: 0, indexSpace: '1d'},
      {id: 'invocation', label: 'Invocation', size: 0, indexSpace: '1d'}
    ]);
    expect(description.relations[0].ref).toEqual({function: 'identity', params: {batchCount: 0}});
    expect(description.formulas).toHaveLength(3);
    expect(submit).not.toHaveBeenCalled();
  } finally {
    compiled.destroy();
    fixture.source.destroy();
    fixture.table.destroy();
    device.destroy();
  }
});

test('unused derived expressions remain intent-only instead of inventing WGSL bindings', () => {
  const fixture = createFixture();
  const plan = makeGPUQueryExpressionShaderPlan(fixture.source, [], fixture.query.derivedColumns, [
    'fare'
  ]);
  const input = {...descriptionInput(fixture), plan, predicates: [], selectedColumns: ['fare']};
  expect(plan.statements).toEqual([]);
  expect(plan.expressions).toEqual([]);
  expect(plan.condition).toBe('true');
  expect(describeGPUDataFrameQueryIntent(input).nodes.some(node => node.id === 'filter')).toBe(
    false
  );
  for (const formula of createGPUDataFrameExpressionFormulas(input)) {
    expect(formula.ascii).toContain('not lowered (unused derived column)');
    expect(formula.symbols.every(symbol => symbol.binding === undefined)).toBe(true);
  }
  fixture.source.destroy();
  fixture.table.destroy();
});

test.each([
  false,
  true
])('visibility lineage is scoped only on recording graphs (%s)', recording => {
  const device = createNullWebGPUDevice();
  const buffer = device.createBuffer({byteLength: 8, usage: Buffer.STORAGE});
  const table = new GPUTable<{fare: 'float32'}>({
    batches: [
      new GPURecordBatch({
        gpuData: {
          fare: new GPUData({buffer, format: 'float32', length: 2, ownsBuffer: true})
        }
      })
    ]
  });
  const source = new GPUDataFrame({table});
  const graph = new GPUCommandGraph<GPUDataFrameQueryParameters>(device, {
    id: 'scope-proof',
    ...(recording ? {recording: {}} : {})
  });
  const scope = vi.spyOn(graph, 'withRecordScope');
  // Stop at the compilation boundary: this test checks declaration without creating GPU pipelines.
  const boundary = new Error('node-only compilation boundary');
  vi.spyOn(graph, 'compile').mockImplementation(() => {
    throw boundary;
  });
  try {
    expect(() => source.filter(column('fare').greaterThan(literal(1))).compile(graph)).toThrow(
      boundary
    );
    const visibilityScopes = scope.mock.calls
      .map(([scope]) => scope)
      .filter(scope => scope.namespace === 'visibility');
    expect(visibilityScopes).toEqual(
      recording
        ? [
            {
              namespace: 'visibility',
              derivedFrom: ['scope-proof-gpu-dataframe-query-intent:visibility']
            }
          ]
        : []
    );
  } finally {
    vi.restoreAllMocks();
    source.destroy();
    table.destroy();
    device.destroy();
  }
});
