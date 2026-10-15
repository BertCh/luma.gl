// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUEmergingHotSpots} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Slices of the cube: one per month of 2023. */
export const EMERGING_HOT_SPOTS_SLICE_COUNT = 12;

/** Observation counts per lattice cell and month, and the lattice they index. */
export type EmergingHotSpotsExampleInput = {
  /** `cell * 12 + month` counts, cell = row * gridWidth + column (row 0 southmost). */
  cube: GraphDataView<'uint32'>;
  /** Lattice columns. */
  gridWidth: number;
  /** Lattice rows. */
  gridHeight: number;
  /** `getGPUEmergingHotSpotsParameterValues({radius: 1.5, temporalWindow: 1})`. */
  parameters: GraphDataView<'float32'>;
};

/** Emerging hot spots of monthly observation counts on an 800 m lattice. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {cube, gridWidth, gridHeight, parameters}: EmergingHotSpotsExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const cellCount = gridWidth * gridHeight;
  const binCount = cellCount * EMERGING_HOT_SPOTS_SLICE_COUNT;
  const output = {
    giZScores: createTransientView(graph, 'giZScores', 'float32', binCount, usage),
    adjustedPValues: createTransientView(graph, 'adjustedPValues', 'float32', binCount, usage),
    trendZ: createTransientView(graph, 'trendZ', 'float32', cellCount, usage),
    trendP: createTransientView(graph, 'trendP', 'float32', cellCount, usage),
    trendS: createTransientView(graph, 'trendS', 'sint32', cellCount, usage),
    category: createTransientView(graph, 'category', 'uint32', cellCount, usage),
    hotSliceCount: createTransientView(graph, 'hotSliceCount', 'uint32', cellCount, usage),
    coldSliceCount: createTransientView(graph, 'coldSliceCount', 'uint32', cellCount, usage)
  };
  // #region example
  // Gi* over cells within 1.5 cells and the current plus one previous month, Benjamini-Hochberg
  // adjusted, then a Mann-Kendall trend and one of the 17 ArcGIS categories per cell.
  graph.add(
    new GPUEmergingHotSpots({
      values: cube,
      gridWidth,
      gridHeight,
      sliceCount: 12,
      maximumRadius: 2,
      parameters,
      output
    })
  );
  // category: 0 no pattern, 1 to 8 hot, 9 to 16 cold, 17 undetermined
  // #endregion example
  return output;
}
