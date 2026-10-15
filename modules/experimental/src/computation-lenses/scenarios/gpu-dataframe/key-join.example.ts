// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUKeyJoin} from '@luma.gl/experimental/gpu-dataframe';

/** Columns the lens scenario reads from `f-chicago-nature.json` and `f-chicago-tracts.json`. */
export type KeyJoinExampleInput = {
  graph: GPUCommandGraph<undefined>;
  observations: {
    /** Tract id of each iNaturalist observation; the left key. */
    tract: Uint32Array;
  };
  tracts: {
    /** Tract id of each tract; unique right keys. */
    tract: Uint32Array;
    /** Share of residents below 150% of the poverty line; suppressed estimates are NaN. */
    poverty150Pct: Float32Array;
    /** 1 when the tract has a poverty estimate; tracts without one are masked out of the join. */
    hasEstimate: Uint32Array;
  };
};

/** Left join of tract poverty onto each observation; tracts without an estimate never match. */
export async function runExample(
  device: Device,
  {graph, observations, tracts}: KeyJoinExampleInput
) {
  const buffers: Buffer[] = [];
  const importColumn = <Format extends 'float32' | 'uint32'>(
    id: string,
    data: Float32Array | Uint32Array,
    format: Format
  ) => {
    const buffer = device.createBuffer({
      id,
      data,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    const handle = graph.importBuffer(
      {id, byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    );
    return graph.createDataView(handle, {format, length: data.length});
  };
  const view = <Format extends 'float32' | 'uint32'>(id: string, format: Format, length: number) =>
    createTransientView(graph, id, format, length, Buffer.STORAGE | Buffer.COPY_SRC);
  const rowCount = observations.tract.length;
  // #region example
  const leftKeys = importColumn('leftKeys', observations.tract, 'uint32');
  const rightKeys = importColumn('tractKeys', tracts.tract, 'uint32');
  const rightMask = importColumn('hasEstimate', tracts.hasEstimate, 'uint32');
  const rightPoverty = importColumn('poverty150Pct', tracts.poverty150Pct, 'float32');
  const joinedPoverty = view('joinedPoverty', 'float32', rowCount);
  const output = {
    rightRows: view('rightRows', 'uint32', rowCount), // 0xffffffff when unmatched
    matchCounts: view('matchCounts', 'uint32', rowCount),
    matched: view('matched', 'uint32', rowCount)
  };
  graph.add(
    new GPUKeyJoin({
      leftKeys,
      rightKeys,
      rightMask,
      kind: 'left', // 'inner' also compacts the matched row ids
      gather: [{column: rightPoverty, output: joinedPoverty}],
      output
    })
  );
  const {rightRows, matchCounts, matched} = output;
  // #endregion example
  return {
    leftKeys,
    rightKeys,
    rightMask,
    rightPoverty,
    joinedPoverty,
    rightRows,
    matchCounts,
    matched,
    buffers
  };
}
