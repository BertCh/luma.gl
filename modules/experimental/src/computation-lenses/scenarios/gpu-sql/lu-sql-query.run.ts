// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUComputationGraph} from '@luma.gl/gpgpu';
import type {LuSQLStatement} from '../../../gpu-sql/lu-sql';
import type {GPUDataFrameQueryParameters} from '@luma.gl/experimental/gpu-dataframe';
import {runExample} from './lu-sql-query.example';
import {
  loadDataFrameLensData,
  createDataFrameLensGraph,
  createDataFrameChoice,
  readGPUVectorValues,
  createTableCapture,
  createDataFrameQueryLensRecord,
  submitDataFrameQuery
} from '../gpu-dataframe/dataframe-lens-support';

const MINIMUM_POPULATION = 1000;
const SCENARIO_ID = 'gpu-sql.lu-sql-query';

/**
 * Links the dataframe plan back to the SQL clauses it realises. Expression nodes keep their path
 * below the clause (`predicate-0/left` realises `where/expr/left`); `filter`, `withColumn` and
 * `select` realise the WHERE and SELECT item clauses. Nodes without a SQL clause stay unlinked.
 */
function linkIntentToStatement(
  intent: GPUComputationGraph,
  statementGraph: GPUComputationGraph,
  statement: LuSQLStatement
): GPUComputationGraph {
  const statementNodeIds = new Set(statementGraph.nodes.map(node => node.id));
  const itemIndexOf = (alias: string) => statement.items.findIndex(item => item.alias === alias);
  const getClause = (nodeId: string): string[] => {
    const predicate = /^predicate-0(\/.*)?$/.exec(nodeId);
    if (predicate) return [`where/expr${predicate[1] ?? ''}`];
    const derived = /^derived-([^/]+)(\/.*)?$/.exec(nodeId);
    if (derived) return [`select-${itemIndexOf(derived[1])}/expr${derived[2] ?? ''}`];
    const derive = /^derive-(.+)$/.exec(nodeId);
    if (derive) return [`select-${itemIndexOf(derive[1])}`];
    if (nodeId === 'filter') return ['where'];
    if (nodeId === 'select') return statement.items.map((_, index) => `select-${index}`);
    return [];
  };
  return {
    ...intent,
    nodes: intent.nodes.map(node => {
      const clauses = getClause(node.id).filter(clause => statementNodeIds.has(clause));
      return clauses.length
        ? {...node, derivedFrom: clauses.map(clause => `${statementGraph.id}:${clause}`)}
        : node;
    })
  };
}

/** Executes the planned SQL once and reads the Census columns and the query outputs from GPU buffers. */
export async function run(device: Device, options?: {signal?: AbortSignal}) {
  options?.signal?.throwIfAborted();
  const columns = (await loadDataFrameLensData('f-chicago-tracts')).columns;
  const graph = createDataFrameLensGraph<GPUDataFrameQueryParameters>(device, SCENARIO_ID);
  const {query, compiled, tracts} = await runExample(device, {
    graph,
    tracts: {
      population: Float32Array.from(columns.population),
      households: Float32Array.from(columns.households),
      noVehicle: Float32Array.from(columns.noVehicle)
    }
  });
  try {
    options?.signal?.throwIfAborted();
    submitDataFrameQuery(device, compiled, {minimumPopulation: MINIMUM_POPULATION});
    const captures = await Promise.all(
      ['population', 'households', 'noVehicle'].map(async name =>
        createTableCapture({
          id: name,
          name,
          keySpace: 'row',
          role: 'input',
          values: await readGPUVectorValues(tracts.table.gpuVectors[name], 'f32'),
          elementType: 'f32'
        })
      )
    );
    for (const [name, vector, elementType, keySpace] of [
      ['noVehicleRate', compiled.table.gpuVectors['noVehicleRate'], 'f32', 'row'],
      ['keep', compiled.selectionMask, 'u32', 'row'],
      ['rowIndices', compiled.rowIndices, 'u32', 'compactSlot'],
      ['selectedCounts', compiled.selectedCounts, 'u32', 'batch']
    ] as const) {
      captures.push(
        createTableCapture({
          id: name,
          name,
          keySpace,
          role: 'output',
          values: await readGPUVectorValues(vector, elementType),
          elementType,
          ...(name === 'rowIndices'
            ? {
                meaning:
                  'Only the first selectedCounts[batch] slots are compact row ids; remaining slots are unused.'
              }
            : {})
        })
      );
    }
    const statementGraph = query.describeStatement();
    const description = compiled.describe({graphId: `${SCENARIO_ID}-compiled`});
    return createDataFrameQueryLensRecord({
      scenarioId: SCENARIO_ID,
      element: 'LuSQLQuery',
      device,
      domains: ['gpu-dataframe', 'gpu-sql'],
      graphs: [
        statementGraph,
        linkIntentToStatement(description.intent, statementGraph, query.statement),
        description.lowered,
        description.compiled
      ],
      derivedFormulas: description.formulas,
      keySpaces: [
        ...description.keySpaces,
        {
          id: 'compactSlot',
          label: 'Compacted row slot',
          size: compiled.rowIndices.length,
          indexSpace: '1d'
        },
        {
          id: 'batch',
          label: 'Source batch',
          size: compiled.selectedCounts.length,
          indexSpace: 'batch'
        }
      ],
      datumKeySpace: 'row',
      relations: description.relations,
      choices: [
        createDataFrameChoice('filter', 'minimumPopulation', MINIMUM_POPULATION, {sweepable: true})
      ],
      captures
    });
  } finally {
    compiled.destroy();
    tracts.destroy();
  }
}
