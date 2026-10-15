// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileComment: Independently implemented for WebGPU; inspired by NVIDIA RAPIDS cuDF.

import type {
  GPUComputationGraph,
  GPUComputationNode,
  GPUComputationEdge,
  GPUComputationFormulaRef,
  GPUComputationFormulaSymbol,
  GPUComputationSourceRef,
  GPUComputationKeySpace,
  GPUComputationRelation
} from '@luma.gl/gpgpu';
import type {GPUExpressionNode} from './gpu-expression';
import type {
  GPUQueryExpressionShaderPlan,
  GPUQueryExpressionStatement
} from './gpu-expression-shader';

/** A generated WGSL statement pair and the expression-tree paths that it realizes. */
export type GPUDataFrameExpressionStatement = GPUQueryExpressionStatement;

/** Linked expression, shader and command graphs with AST-derived formulas. */
export type GPUDataFrameQueryDescription = {
  intent: GPUComputationGraph;
  lowered: GPUComputationGraph;
  compiled: {graph: GPUComputationGraph; sources: GPUComputationSourceRef[]};
  formulas: GPUComputationFormulaRef[];
  expressions: readonly GPUDataFrameExpressionStatement[];
  keySpaces: GPUComputationKeySpace[];
  relations: GPUComputationRelation[];
};

/** Immutable construction inputs retained by a compiled query; no GPU resources. @internal */
export type GPUDataFrameQueryDescriptionInput = {
  queryId: string;
  plan: GPUQueryExpressionShaderPlan;
  predicates: readonly GPUExpressionNode[];
  derivedColumns: readonly {name: string; node: GPUExpressionNode}[];
  selectedColumns: readonly string[];
  batchRows: readonly number[];
  graphRecording: boolean;
};

const OPERATORS: Record<string, {ascii: string; tex: string; precedence: number}> = {
  or: {ascii: 'or', tex: '\\lor', precedence: 1},
  and: {ascii: 'and', tex: '\\land', precedence: 2},
  equal: {ascii: '=', tex: '=', precedence: 3},
  'not-equal': {ascii: '!=', tex: '\\ne', precedence: 3},
  'greater-than': {ascii: '>', tex: '>', precedence: 3},
  'greater-than-or-equal': {ascii: '>=', tex: '\\ge', precedence: 3},
  'less-than': {ascii: '<', tex: '<', precedence: 3},
  'less-than-or-equal': {ascii: '<=', tex: '\\le', precedence: 3},
  add: {ascii: '+', tex: '+', precedence: 4},
  subtract: {ascii: '-', tex: '-', precedence: 4},
  multiply: {ascii: '*', tex: '\\times', precedence: 5},
  divide: {ascii: '/', tex: '/', precedence: 5}
};

function getChildren(node: GPUExpressionNode): [string, GPUExpressionNode][] {
  return node.kind === 'binary'
    ? [
        ['left', node.left],
        ['right', node.right]
      ]
    : node.kind === 'unary'
      ? [['operand', node.operand]]
      : [];
}

function getRoots(input: GPUDataFrameQueryDescriptionInput): [string, GPUExpressionNode][] {
  return [
    ...input.derivedColumns.map(({name, node}): [string, GPUExpressionNode] => [
      `derived-${name}`,
      node
    ]),
    ...input.predicates.map((node, index): [string, GPUExpressionNode] => [
      `predicate-${index}`,
      node
    ])
  ];
}

function makeNode(id: string, op: string, label: string): GPUComputationNode {
  return {id, op, label, namespace: '', kind: 'operation', reads: [], writes: [], attrs: {}};
}

function makeEdge(source: string, target: string): GPUComputationEdge {
  return {from: source, to: target, kind: 'dataflow'};
}

/** Builds the expression AST and query operations without touching GPU resources. */
export function describeGPUDataFrameQueryIntent(
  input: GPUDataFrameQueryDescriptionInput
): GPUComputationGraph {
  const nodes: GPUComputationNode[] = [];
  const edges: GPUComputationEdge[] = [];
  const visit = (node: GPUExpressionNode, id: string): void => {
    const label =
      node.kind === 'column'
        ? node.name
        : node.kind === 'literal'
          ? String(node.value)
          : node.kind === 'parameter'
            ? `${node.name} (${node.value})`
            : node.kind === 'binary'
              ? node.operator === 'multiply'
                ? '×'
                : OPERATORS[node.operator].ascii
              : {'is-valid': 'isValid', 'is-null': 'isNull', not: 'not', negate: '-'}[
                  node.operator
                ];
    nodes.push({
      ...makeNode(id, 'operator' in node ? node.operator : node.kind, label),
      attrs: {text: printExpression(node, id, input, false)}
    });
    for (const [path, child] of getChildren(node)) {
      visit(child, `${id}/${path}`);
      edges.push(makeEdge(`${id}/${path}`, id));
    }
    if (node.kind === 'column' && input.derivedColumns.some(column => column.name === node.name)) {
      edges.push(makeEdge(`derive-${node.name}`, id));
    }
  };
  for (const [id, node] of getRoots(input)) visit(node, id);
  for (const {name} of input.derivedColumns) {
    nodes.push(makeNode(`derive-${name}`, 'withColumn', name));
    edges.push(makeEdge(`derived-${name}`, `derive-${name}`), makeEdge(`derive-${name}`, 'select'));
  }
  if (input.predicates.length) {
    nodes.push(makeNode('filter', 'filter', 'filter'));
    input.predicates.forEach((_, index) => edges.push(makeEdge(`predicate-${index}`, 'filter')));
    edges.push(makeEdge('filter', 'select'));
  }
  nodes.push(
    {
      ...makeNode('select', 'select', 'select'),
      attrs: {columns: JSON.stringify(input.selectedColumns)}
    },
    makeNode('visibility', 'compact', 'compact row ids')
  );
  edges.push(makeEdge('select', 'visibility'));
  return {id: `${input.queryId}-intent`, form: 'intent', domain: 'gpu-dataframe', nodes, edges};
}

/** Builds one lowered node per emitted let pair, with exact source statements and lineage. */
export function describeGPUDataFrameQueryShaderPlan(
  input: GPUDataFrameQueryDescriptionInput
): GPUComputationGraph {
  const intent = describeGPUDataFrameQueryIntent(input);
  const nodes: GPUComputationNode[] = input.plan.expressions.map(expression => ({
    ...makeNode(
      `expression${expression.index}`,
      'wgsl-let',
      `expression${expression.index} = ${intent.nodes.find(node => node.id === expression.astNodeIds[0])?.attrs?.['text'] ?? expression.value}`
    ),
    derivedFrom: expression.astNodeIds.map(id => `${intent.id}:${id}`),
    attrs: {
      statement: expression.statements[0],
      validStatement: expression.statements[1],
      format: expression.format,
      nullable: expression.nullable
    }
  }));
  const edges = input.plan.expressions.flatMap(expression =>
    expression.operands.map(operand =>
      makeEdge(`expression${operand}`, `expression${expression.index}`)
    )
  );
  for (const output of input.plan.outputs) {
    const id = `write-derived${output.index}`;
    nodes.push({
      ...makeNode(id, 'wgsl-write', output.name),
      derivedFrom: [`${intent.id}:derive-${output.name}`],
      attrs: {
        statement: `derived${output.index}[DERIVED_${output.index}_OFFSET + index] = ${output.value};`,
        ...(output.nullable
          ? {
              validStatement: `derivedValidity${output.index}[DERIVED_VALIDITY_${output.index}_OFFSET + index] = select(0u, 1u, ${output.valid});`
            }
          : {})
      }
    });
    edges.push(makeEdge(output.value.replace('Value', ''), id));
  }
  nodes.push({
    ...makeNode('write-outputMask', 'wgsl-write', 'keep'),
    derivedFrom: [`${intent.id}:${input.predicates.length ? 'filter' : 'select'}`],
    attrs: {
      statement: `outputMask[OUTPUT_OFFSET + index] = select(0u, 1u, ${input.plan.condition});`
    }
  });
  for (const root of input.plan.roots) {
    if (root.kind === 'predicate')
      edges.push(makeEdge(`expression${root.expression}`, 'write-outputMask'));
  }
  return {
    id: `${input.queryId}-shader-plan`,
    form: 'lowered',
    domain: 'gpu-dataframe',
    nodes,
    edges
  };
}

function escapeTex(name: string): string {
  return name.replace(/[\\{}_$%&#^~]/g, character => `\\${character}`);
}

function printExpression(
  node: GPUExpressionNode,
  path: string,
  input: GPUDataFrameQueryDescriptionInput,
  tex: boolean,
  symbols?: Map<string, GPUComputationFormulaSymbol>,
  parentPrecedence = 0
): string {
  const statement = input.plan.expressions.find(expression => expression.astNodeIds.includes(path));
  const term = (
    id: string,
    body: string,
    role: GPUComputationFormulaSymbol['role'],
    binding?: string
  ): string => {
    symbols?.set(id, {
      id,
      role,
      ...(binding ? {binding: `wgsl:${binding}`} : {}),
      ...(role === 'intermediate' ? {label: `e_${statement?.index}`} : {})
    });
    return `\\term{${id}}{${body}}`;
  };
  if (node.kind === 'column') {
    const output = input.plan.outputs.find(output => output.name === node.name);
    const binding =
      statement?.columnIndex !== undefined
        ? `input${statement.columnIndex}`
        : output
          ? `derived${output.index}`
          : statement?.control !== undefined
            ? 'queryControls'
            : statement?.value;
    return tex
      ? `${term(node.name, `\\mathit{${escapeTex(node.name)}}`, 'input', binding)}_i`
      : `${node.name}[i]`;
  }
  if (node.kind === 'literal')
    return node.value === null && tex ? '\\mathrm{null}' : String(node.value);
  if (node.kind === 'parameter')
    return tex
      ? term(
          node.name,
          `\\mathrm{${escapeTex(node.name)}}`,
          'parameter',
          statement ? 'queryControls' : undefined
        )
      : node.name;
  const precedence = node.kind === 'binary' ? OPERATORS[node.operator].precedence : 6;
  const child = (operand: GPUExpressionNode, suffix: string, minimum: number): string =>
    printExpression(operand, `${path}/${suffix}`, input, tex, symbols, minimum);
  let body: string;
  if (node.kind === 'binary') {
    body =
      tex && node.operator === 'divide'
        ? `\\frac{${child(node.left, 'left', 0)}}{${child(node.right, 'right', 0)}}`
        : `${child(node.left, 'left', precedence)} ${tex ? OPERATORS[node.operator].tex : OPERATORS[node.operator].ascii} ${child(node.right, 'right', precedence + 1)}`;
    if (tex && statement)
      body = term(`expression${statement.index}`, body, 'intermediate', statement.value);
  } else {
    const operand = child(
      node.operand,
      'operand',
      node.operator === 'is-valid' || node.operator === 'is-null' ? 0 : precedence
    );
    body =
      node.operator === 'is-valid' || node.operator === 'is-null'
        ? tex
          ? `\\operatorname{${node.operator === 'is-valid' ? 'valid' : 'null'}}\\left(${operand}\\right)`
          : `${node.operator === 'is-valid' ? 'isValid' : 'isNull'}(${operand})`
        : `${node.operator === 'negate' ? '-' : tex ? '\\lnot ' : 'not '}${operand}`;
  }
  return precedence < parentPrecedence ? (tex ? `\\left(${body}\\right)` : `(${body})`) : body;
}

/** Prints formulas from the retained AST and binds terms using the actual statement map. */
export function createGPUDataFrameExpressionFormulas(
  input: GPUDataFrameQueryDescriptionInput
): GPUComputationFormulaRef[] {
  const formulas: GPUComputationFormulaRef[] = input.derivedColumns.map(({name, node}) => {
    const root = input.plan.roots.find(root => root.kind === 'derived' && root.name === name);
    const output = input.plan.outputs.find(output => output.name === name);
    const statement = root && input.plan.expressions[root.expression];
    const symbols = new Map<string, GPUComputationFormulaSymbol>();
    symbols.set(name, {
      id: name,
      role: 'output',
      ...(output
        ? {binding: `wgsl:derived${output.index}`}
        : statement
          ? {binding: `wgsl:${statement.value}`}
          : {})
    });
    const expression = printExpression(node, `derived-${name}`, input, true, symbols);
    return {
      id: `gpu-dataframe:derived:${name}`,
      derivation: 'derived',
      tex: `\\term{${name}}{\\mathit{${escapeTex(name)}}}_i = ${expression}`,
      ascii: `${name}[i] = ${printExpression(node, `derived-${name}`, input, false)}${statement?.nullable ? '; null if any operand is null' : ''}${!statement ? '; not lowered (unused derived column)' : ''}`,
      symbols: [...symbols.values()]
    };
  });
  if (input.predicates.length) {
    const symbols = new Map<string, GPUComputationFormulaSymbol>([
      ['keep', {id: 'keep', role: 'output', binding: 'wgsl:outputMask'}]
    ]);
    const predicate = input.predicates
      .map((node, index) => printExpression(node, `predicate-${index}`, input, true, symbols, 3))
      .join(' \\land ');
    formulas.push({
      id: 'gpu-dataframe:filter',
      derivation: 'derived',
      tex: `\\term{keep}{\\mathrm{keep}}_i = \\mathbb{1}\\left[\\operatorname{valid}(P_i) \\land P_i\\right],\\; P_i = ${predicate}`,
      ascii: `keep[i] = 1[valid(P[i]) and P[i]]; P[i] = ${input.predicates.map((node, index) => printExpression(node, `predicate-${index}`, input, false, undefined, 3)).join(' and ')}; three-valued logic keeps only valid true predicates (false and null are dropped)`,
      symbols: [...symbols.values()]
    });
  }
  return formulas;
}
