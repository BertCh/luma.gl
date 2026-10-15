// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import path from 'node:path';
import {describe, expect, test} from 'vitest';

import {
  evaluateGPUCoreTutorial,
  GPU_CORE_TUTORIAL_DEFAULT_FLAGS,
  GPU_CORE_TUTORIAL_VALUES
} from '../../website/src/components/docs/gpu-core-pipeline-model';
import {
  buildGPUCoreAnatomyModel,
  GPU_CORE_ANATOMY_SCENARIO_ID,
  GPU_CORE_COMPILER_STAGES
} from '../../website/src/components/docs/gpu-core-compiler-model';

import {fromRecordedLensFile} from '../../modules/experimental/src/computation-lenses/registry';

const ANATOMY_RECORD = fromRecordedLensFile(
  JSON.parse(
    readFileSync(
      path.join(process.cwd(), `docs/lens-records/${GPU_CORE_ANATOMY_SCENARIO_ID}.json`),
      'utf8'
    )
  )
);

// The anatomy and execution lab now display recorded GPUScan work, replacing the hand-modelled
// classify/scan/style/render pipeline. Test the recorded allocation and slicing evidence.
describe('GPU Core pipeline tutorial', () => {
  test('explains the default stable compaction result', () => {
    expect(
      evaluateGPUCoreTutorial(GPU_CORE_TUTORIAL_VALUES, GPU_CORE_TUTORIAL_DEFAULT_FLAGS)
    ).toEqual({
      flags: [1, 0, 1, 0, 1, 0, 0, 1],
      offsets: [0, 1, 1, 2, 2, 3, 3, 3],
      compactedValues: [8, 5, 9, 4],
      compactedSourceIndices: [0, 2, 4, 7],
      instanceCount: 4
    });
  });

  test('normalizes nonzero flags and handles empty output', () => {
    expect(evaluateGPUCoreTutorial([4, 7, 2], [9, -1, 0])).toEqual({
      flags: [1, 1, 0],
      offsets: [0, 1, 2],
      compactedValues: [4, 7],
      compactedSourceIndices: [0, 1],
      instanceCount: 2
    });
    expect(evaluateGPUCoreTutorial([4, 7, 2], [0, 0, 0])).toEqual({
      flags: [0, 0, 0],
      offsets: [0, 0, 0],
      compactedValues: [],
      compactedSourceIndices: [],
      instanceCount: 0
    });
  });

  test('rejects mismatched source and flag lengths', () => {
    expect(() => evaluateGPUCoreTutorial([1, 2], [1])).toThrow(/same length/);
  });

  test('places the compact lab on the overview and the full comparison on the tutorial route', () => {
    const overviewSource = readFileSync(
      path.join(process.cwd(), 'docs/api-reference/experimental/gpu-core/README.md'),
      'utf8'
    );
    const tutorialSource = readFileSync(
      path.join(process.cwd(), 'docs/api-reference/experimental/gpu-core/tutorial.md'),
      'utf8'
    );
    const componentSource = readFileSync(
      path.join(process.cwd(), 'website/src/components/docs/gpu-core-pipeline-tutorial.tsx'),
      'utf8'
    );

    expect(overviewSource).toContain('<GPUCorePipelineTutorial compact />');
    expect(overviewSource).toContain('The small WebGPU leap behind GPU Core');
    expect(overviewSource).toContain('Compute shaders and storage buffers');
    expect(overviewSource).toContain('GPU-writable indirect draw and dispatch arguments');
    expect(overviewSource).toContain('new GPUCommandGraph(device');
    expect(overviewSource).toContain('graph.add(');
    expect(tutorialSource).toContain('<GPUCorePipelineTutorial />');
    expect(tutorialSource).toContain('Terminology in one minute');
    expect(tutorialSource).toMatch(
      /\*\*Read\/write hazard \(resource conflict\)\*\*[\s\S]*ordering rule inferred from data access/
    );
    expect(tutorialSource).toMatch(/\*\*Resource\*\*[\s\S]*buffer or texture/);
    expect(tutorialSource).toContain('Translate familiar WebGPU concepts');
    expect(tutorialSource).toMatch(/\*\*GPU Core\*\*[\s\S]*\*\*GPU Graph\*\*/);
    expect(tutorialSource).toMatch(/\*\*Graph compilation\*\*[\s\S]*\*\*shader compilation\*\*/);
    expect(tutorialSource).toMatch(/\*\*batch or chunk\*\*[\s\S]*\*\*execution slice\*\*/);
    expect(componentSource).toMatch(/Manual WebGPU and GPU Core/);
    expect(componentSource).toMatch(/mask, exclusive scan,[\s\S]*stable scatter/);
    expect(componentSource).toContain("['Composed'");
    expect(componentSource).toMatch(/\['Declared',[\s\S]*getCommandNodes\(\) expands compaction/);
    expect(componentSource).toContain('drawIndirect');
  });

  test('exposes every compiler view and the recorded non-overlapping alias example', () => {
    expect(GPU_CORE_COMPILER_STAGES.map(stage => stage.id)).toEqual([
      'declared',
      'hazards',
      'schedule',
      'lifetimes',
      'allocations',
      'encoded',
      'slices'
    ]);
    const model = buildGPUCoreAnatomyModel(ANATOMY_RECORD);
    expect(model.status).toBe('ready');
    if (model.status !== 'ready') throw new Error(model.reason);
    expect(model.provenance.runId).toBe(ANATOMY_RECORD.record.runId);
    expect(model.nodes.some(node => node.operator === 'GPUScan')).toBe(true);

    const sharedAllocations = model.allocations.filter(
      allocation => allocation.resourceIds.length > 1
    );
    expect(sharedAllocations.length).toBeGreaterThan(0);
    for (const allocation of sharedAllocations) {
      const resources = model.resources
        .filter(resource => allocation.resourceIds.includes(resource.id))
        .sort((first, second) => first.firstNode! - second.firstNode!);
      for (let index = 1; index < resources.length; index++) {
        expect(resources[index - 1].lastNode).toBeLessThan(resources[index].firstNode!);
      }
    }
  });

  test('displays recorded frame slices at node boundaries and surfaces oversized steps', () => {
    const model = buildGPUCoreAnatomyModel(ANATOMY_RECORD);
    expect(model.status).toBe('ready');
    if (model.status !== 'ready') throw new Error(model.reason);
    expect(model.sliceSeries.map(series => series.budget)).toEqual([512, 2048, 8192]);
    for (const series of model.sliceSeries) {
      expect(series.slices.flatMap(slice => slice.nodeIds)).toEqual(
        model.nodes.map(node => node.id)
      );
      const recordedSeries = ANATOMY_RECORD.record.metrics
        ?.flatMap(metrics => metrics.series ?? [])
        .find(candidate => candidate.name === `slices@${series.budget}`);
      expect(recordedSeries).toBeDefined();
      expect(
        model.nodes.map(node => series.slices.find(slice => slice.nodeIds.includes(node.id))?.index)
      ).toEqual(Array.from(recordedSeries!.values));
      for (const slice of series.slices) {
        expect(slice.invocationCount).toBeDefined();
        if (slice.oversized) {
          expect(slice.nodeIds).toHaveLength(1);
          expect(slice.invocationCount).toBeGreaterThan(series.budget);
        }
      }
    }
    expect(model.sliceSeries[0].slices.some(slice => slice.oversized)).toBe(true);
    expect(model.sliceSeries[2].slices).toHaveLength(1);

    const withoutSlices = buildGPUCoreAnatomyModel({
      ...ANATOMY_RECORD,
      record: {...ANATOMY_RECORD.record, metrics: []}
    });
    expect(withoutSlices.status).toBe('ready');
    if (withoutSlices.status !== 'ready') throw new Error(withoutSlices.reason);
    expect(withoutSlices.sliceSeries).toEqual([]);
  });

  test('connects the compiler anatomy, execution lab, glossary, and trace walkthrough', () => {
    const concepts = readFileSync(
      path.join(process.cwd(), 'docs/api-reference/experimental/gpu-core/concepts.md'),
      'utf8'
    );
    const recipes = readFileSync(
      path.join(process.cwd(), 'docs/api-reference/experimental/gpu-core/recipes.md'),
      'utf8'
    );
    const tutorial = readFileSync(
      path.join(process.cwd(), 'docs/api-reference/experimental/gpu-core/tutorial.md'),
      'utf8'
    );
    const trace = readFileSync(
      path.join(process.cwd(), 'docs/api-reference/experimental/gpu-trace.md'),
      'utf8'
    );

    expect(concepts).toContain('<GPUCoreCompilerAnatomy />');
    expect(recipes).toContain('<GPUCoreExecutionLab />');
    expect(tutorial).toContain('<GPUCoreTerm term="hazard" />');
    expect(trace).toContain('<GPUTracePipelineWalkthrough />');
  });
});
