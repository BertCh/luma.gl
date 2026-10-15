// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUComputationRecord, GPUComputationRecordWithSidecars} from '@luma.gl/gpgpu';
import type {LensRungId} from '../computation-views/types';

/** Scenario identifier: `<entry>.<element>[.<variant>]`, with kebab-case segments. */
export type LensScenarioId = string;
/** One real execution with optional measurement and capture sidecars. */
export type LensScenarioResult = GPUComputationRecordWithSidecars;

/** Lazily loaded GPU scenario. Register descriptors with registerLensScenario at module scope. */
export type LensScenarioDescriptor = {
  /** `<entry>.<element-kebab>[.<variant>]`, e.g. `gpu-spatial-analysis.local-moran`. */
  id: LensScenarioId;
  /** Documented entry; must equal the first id segment, e.g. `gpu-spatial-analysis`. */
  entry: string;
  /**
   * Documented export this scenario covers: the exported class name (`GPULocalMoran`), or the
   * function name for recipes. Pack `operators` and `formulas` use the same key.
   */
  element: string;
  title: string;
  audience: 'analyst' | 'engineer';
  rungs: readonly LensRungId[];
  defaultRung: LensRungId;
  /** Defaults to getLensExamplePath(id); required by the coverage spec once scenarios exist. */
  example?: {path: string};
  dataset?: {id: string; licence: string; source: string};
  /** Imports GPU code only when explicitly activated; never during a docs build. */
  load(): Promise<{
    run(device: Device, options?: {signal?: AbortSignal}): Promise<LensScenarioResult>;
    describeStructure?(device: Device): Promise<GPUComputationRecord>;
  }>;
};

/**
 * Recorded scenario envelope. Use fromRecordedLensFile for parsed JSON: on disk, typed capture
 * values are base64 text (other value arrays encode non-finite numbers as NaN/Infinity/-Infinity
 * strings) and the record is losslessly compacted (repeated captures, word runs, source lines and
 * graph strings; see `record-encoding.ts`, `encoding` table); in memory values are numeric.
 */
export type RecordedLensFile = {
  schema: 'luma.lens-record/1';
  scenarioId: LensScenarioId;
  recordedAt: string;
  adapterLabel: string;
  variant?: string;
  record: LensScenarioResult;
};
