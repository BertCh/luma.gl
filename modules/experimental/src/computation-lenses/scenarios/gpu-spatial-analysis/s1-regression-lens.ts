// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {LensRungId} from '../../../computation-views/types';
import {registerLensScenario} from '../../registry';
// Types only: the GPU-facing model module loads when a scenario is activated.
import type {
  RegressionExampleInput,
  RegressionExecution,
  RegressionParameterUploads
} from './s1-regression-model';
import type {SpatialScenarioRun} from './spatial-scenario-utils';

/** The GPU-facing helpers of the shared model, loaded when a scenario is activated. */
export type RegressionModelTools = typeof import('./s1-regression-model');

/** Everything that differs between the ten regression cards. */
export type RegressionLensProps<ExampleExecution> = {
  /** Element kebab id, also the example file name: `gpu-spatial-analysis.<slug>`. */
  slug: string;
  /** Exported class name; the pack entry and formula use the same key. */
  element: string;
  title: string;
  rungs: readonly LensRungId[];
  /** Whether the card draws the tract map (reads the outlines back). */
  withMap: boolean;
  /** Values the run uses for every choice template of the pack entry. */
  choices: SpatialScenarioRun['choices'];
  /** Guard metadata beyond the choices. */
  metadata?: SpatialScenarioRun['metadata'];
  /** Per-frame parameter arrays to upload beside the design (GWR ladder, permutations). */
  loadUploads?(): Promise<RegressionParameterUploads>;
  /** Loads the sibling example module lazily. */
  loadExample(): Promise<{
    runExample(device: Device, input: RegressionExampleInput): Promise<ExampleExecution>;
  }>;
  /**
   * Called after the example built its nodes and before compilation: appends presentation kernels,
   * names the views to read back and tags their roles.
   */
  prepare(
    execution: ExampleExecution,
    input: RegressionExampleInput,
    model: RegressionModelTools
  ): RegressionExecution;
};

/** Registers one regression scenario; no GPU code loads until the scenario is activated. */
export function registerRegressionLens<ExampleExecution>(
  props: RegressionLensProps<ExampleExecution>
): void {
  const id = `gpu-spatial-analysis.${props.slug}`;
  registerLensScenario({
    id,
    entry: 'gpu-spatial-analysis',
    element: props.element,
    title: props.title,
    audience: 'analyst',
    rungs: props.rungs,
    defaultRung: 'result',
    dataset: {
      id: 's1-chicago-tracts/1',
      source: 'Census cb_2022_17_tract_500k; CDC/ATSDR SVI 2022 (ACS 2018-2022)',
      licence:
        'US Census Bureau cartographic boundary file cb_2022_17_tract_500k, public domain; CDC/ATSDR SVI 2022, public domain'
    },
    async load() {
      const [example, model, {REGRESSION_OPERATORS}] = await Promise.all([
        props.loadExample(),
        import('./s1-regression-model'),
        import('../../../gpu-spatial-analysis/computation-pack/operators/regression')
      ]);
      const regressionModel = await model.loadRegressionModel();
      const uploads = await props.loadUploads?.();
      return {
        run: device =>
          model.runRegressionScenario({
            device,
            id,
            model: regressionModel,
            entry: REGRESSION_OPERATORS[props.element],
            choices: props.choices,
            metadata: props.metadata,
            withMap: props.withMap,
            uploads,
            async execute(input) {
              return props.prepare(await example.runExample(device, input), input, model);
            }
          })
      };
    }
  });
}
