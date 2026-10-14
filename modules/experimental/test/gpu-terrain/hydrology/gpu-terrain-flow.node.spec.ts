// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPU_TERRAIN_FLOW_CELL_CLASS,
  GPU_TERRAIN_FLOW_PARAMETER_LENGTH,
  GPU_TERRAIN_FLOW_STREAM_UNRESOLVED,
  GPU_TERRAIN_FLOW_UNRESOLVED,
  GPUTerrainFlow,
  getGPUTerrainFlowParameterValues,
  type GPUTerrainFlowOutput,
  type GPUTerrainFlowProps
} from '../../../src/gpu-terrain/hydrology';
import {makeGPURasterBand, makeGPURasterMetadata} from '../../../src/gpu-raster';
import {getRasterGridGeoreference} from '../../../src/analysis-core/raster-grid';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {getRasterPipelines} from '../../gpu-raster/raster-pipeline-count';

const GRID = makeGPURasterMetadata({width: 6, height: 5});

/** Props overrides whose `output` members are merged over the default `flowDirections` output. */
type PropsOverrides = Partial<Omit<GPUTerrainFlowProps, 'output'>> & {
  output?: Partial<GPUTerrainFlowOutput>;
};

function createFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = ({output, ...overrides}: PropsOverrides = {}) => {
    instance++;
    return new GPUTerrainFlow({
      grid: GRID,
      elevation: makeGPURasterBand(
        createTransientView(graph, `elevation-${instance}`, 'float32', 30),
        GRID,
        {id: `elevation-${instance}`}
      ),
      parameters: createTransientView(graph, `settings-${instance}`, 'float32', 8),
      ...overrides,
      output: {
        flowDirections: createTransientView(graph, `directions-${instance}`, 'uint32', 30),
        ...output
      }
    });
  };
  return {device, graph, create};
}

it('getGPUTerrainFlowParameterValues packs settings', () => {
  expect(GPU_TERRAIN_FLOW_PARAMETER_LENGTH).toBe(8);
  // Cell size and row edges come from the grid.
  const grid = makeGPURasterMetadata({width: 6, height: 5, cellSize: [2, 3], origin: [1, 40]});
  const {cellSize, northEdge, southEdge} = getRasterGridGeoreference(grid);
  expect([cellSize, northEdge, southEdge]).toEqual([[2, 3], 40, 25]);
  expect(
    Array.from(getGPUTerrainFlowParameterValues({fillEpsilon: 0.125, streamThreshold: 10}, grid))
  ).toEqual([2, 3, 40, 25, 0.125, 10, 0, 0]);
  const defaults = getGPUTerrainFlowParameterValues({}, GRID);
  expect(defaults[4]).toBe(0);
  expect(defaults[5]).toBe(Infinity);
  expect(() => getGPUTerrainFlowParameterValues({}, GRID, new Float32Array(7))).toThrow(/8 values/);
});

it('GPUTerrainFlow prefixes node ids and grows with iteration limits', () => {
  const {device, graph, create} = createFixture();
  const small = create({
    maximumFillIterations: 2,
    maximumAccumulationIterations: 2,
    fillDepressions: true,
    output: {
      accumulation: createTransientView(graph, 'acc-small', 'float32', 30),
      status: {
        fillConverged: createTransientView(graph, 'fc-small', 'uint32', 1),
        accumulationConverged: createTransientView(graph, 'ac-small', 'uint32', 1)
      }
    }
  });
  const smallNodes = small.getCommandNodes(graph);
  expect(smallNodes.every(node => node.id.startsWith('terrain-flow-'))).toBe(true);
  const ids = smallNodes.map(node => node.id);
  expect(ids).toContain('terrain-flow-flow-direction');
  // The spill-graph fill publishes the selected path's status (relaxation when fillEpsilon > 0).
  expect(ids).toContain('terrain-flow-spill-publish');
  expect(ids).toContain('terrain-flow-accumulate-finalize');
  expect(new Set(ids).size).toBe(ids.length);

  const {graph: largeGraph, create: createLarge} = createFixture();
  const large = createLarge({
    id: 'big',
    maximumFillIterations: 5,
    maximumAccumulationIterations: 7,
    fillDepressions: true,
    output: {streams: createTransientView(largeGraph, 'streams', 'uint32', 30)}
  });
  const largeNodes = large.getCommandNodes(largeGraph);
  expect(largeNodes.every(node => node.id.startsWith('big-'))).toBe(true);
  // The relaxation fill (fillEpsilon > 0) still grows by one node per iteration plus one gate per
  // four iterations; the spill-graph nodes do not depend on maximumFillIterations.
  const moreFill = createLarge({
    id: 'big9',
    maximumFillIterations: 9,
    maximumAccumulationIterations: 7,
    fillDepressions: true,
    output: {streams: createTransientView(largeGraph, 'streams-fill9', 'uint32', 30)}
  }).getCommandNodes(largeGraph);
  expect(moreFill.length - largeNodes.length).toBe(4 + 1);
  const withoutFill = createLarge({
    id: 'nofill',
    maximumAccumulationIterations: 7,
    output: {streams: createTransientView(largeGraph, 'streams2', 'uint32', 30)}
  }).getCommandNodes(largeGraph);
  // Fill always adds the relaxation path, the spill-graph path, and the unresolved-output arm and
  // gated cell pass (review C-09).
  expect(largeNodes.length - withoutFill.length).toBeGreaterThan(3 + 1 + 5 + 2 + 2);
  const moreRounds = createLarge({
    id: 'rounds',
    maximumAccumulationIterations: 9,
    output: {streams: createTransientView(largeGraph, 'streams3', 'uint32', 30)}
  }).getCommandNodes(largeGraph);
  // D8 'cells' accumulation runs on the Euler tour, whose round count follows the grid, not the limit.
  expect(moreRounds.length).toBe(withoutFill.length);
  // Weighted accumulation keeps the pull rounds: two more rounds add one gate group of 4 -> 2 nodes + 1 gate.
  const runoff = createTransientView(largeGraph, 'runoff-weights', 'float32', 30);
  const pullBase = createLarge({
    id: 'pull7',
    maximumAccumulationIterations: 7,
    runoff,
    output: {streams: createTransientView(largeGraph, 'streams4', 'uint32', 30)}
  }).getCommandNodes(largeGraph);
  const pullMore = createLarge({
    id: 'pull9',
    maximumAccumulationIterations: 9,
    runoff,
    output: {streams: createTransientView(largeGraph, 'streams5', 'uint32', 30)}
  }).getCommandNodes(largeGraph);
  expect(pullMore.length - pullBase.length).toBe(2 + 1);
  // Direction-only requests skip accumulation entirely.
  expect(
    create({id: 'dir'})
      .getCommandNodes(graph)
      .map(node => node.id)
  ).toEqual(['dir-elevation', 'dir-flow-direction']);
  device.destroy();
});

it('GPUTerrainFlow validates its props', () => {
  const {device, graph, create} = createFixture();
  const view = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  expect(() => create({grid: {...GRID, width: 0}})).toThrow(/dimensions/);
  expect(() => create({output: {flowDirections: undefined}})).toThrow(/at least one output/);
  expect(() => create({parameters: view('s7', 'float32', 7)})).toThrow(/parameters/);
  expect(() => create({output: {flowDirections: view('short', 'uint32', 29)}})).toThrow(
    /one value per cell/
  );
  expect(() =>
    create({output: {accumulation: view('wrong-format', 'uint32', 30) as never}})
  ).toThrow(/float32/);
  expect(() => create({output: {filledElevation: view('filled', 'float32', 30)}})).toThrow(
    /fillDepressions/
  );
  expect(() => create({output: {status: {fillConverged: view('fc', 'uint32', 1)}}})).toThrow(
    /fillDepressions/
  );
  expect(() =>
    create({output: {status: {accumulationConverged: view('ac', 'uint32', 1)}}})
  ).toThrow(/accumulation or output.streams/);
  expect(() =>
    create({
      output: {
        streams: view('st', 'uint32', 30),
        status: {accumulationConverged: view('ac2', 'uint32', 2)}
      }
    })
  ).toThrow(/one row/);
  expect(() => create({maximumFillIterations: 0})).toThrow(/maximumFillIterations/);
  expect(() => create({maximumFlatIterations: 0})).toThrow(/maximumFlatIterations/);
  expect(() => create({maximumAccumulationIterations: 2000})).toThrow(
    /maximumAccumulationIterations/
  );
  expect(() => create({cellSizeMode: 'polar' as never})).toThrow(/cellSizeMode/);
  expect(() => create({accumulationUnits: 'acres' as never})).toThrow(/accumulationUnits/);
  expect(() => create({runoff: view('runoff', 'float32', 29)})).toThrow(/one value per cell/);
  expect(() =>
    create({cellSizeMode: 'geographic', grid: {...GRID, affine: [1, 0.5, 0, 0, -1, 0]}})
  ).toThrow(/axis-aligned/);
  const shared = view('shared', 'uint32', 30);
  expect(() => create({output: {flowDirections: shared, cellClasses: shared}})).toThrow(
    /share buffers/
  );
  const otherGraph = new GPUCommandGraph(device);
  expect(() =>
    new GPUTerrainFlow({
      grid: makeGPURasterMetadata({width: 2, height: 2}),
      elevation: makeGPURasterBand(
        createTransientView(otherGraph, 'foreign', 'float32', 4),
        makeGPURasterMetadata({width: 2, height: 2}),
        {id: 'foreign'}
      ),
      parameters: createTransientView(otherGraph, 'foreign-settings', 'float32', 8),
      output: {flowDirections: createTransientView(otherGraph, 'foreign-out', 'uint32', 4)}
    }).getCommandNodes(graph)
  ).toThrow(/belong to the target graph/);
  device.destroy();
});

it('GPUTerrainFlow lets outputs share a buffer only over disjoint byte ranges', () => {
  const {device, graph, create} = createFixture();
  const summary = graph.createTransientBuffer({id: 'summary', byteLength: 1024, usage: 128});
  const flagsAt = (byteOffset: number) =>
    graph.createDataView(summary, {format: 'uint32', length: 1, byteOffset});
  const streams = createTransientView(graph, 'st', 'uint32', 30);
  const common = {fillDepressions: true};
  const withStatus = (status: NonNullable<GPUTerrainFlowOutput['status']>) => ({
    ...common,
    output: {streams, status}
  });

  // Two 1-row flags in different words of one buffer are accepted and schedule normally.
  const contributor = create(
    withStatus({fillConverged: flagsAt(0), accumulationConverged: flagsAt(4)})
  );
  expect(() => contributor.getCommandNodes(graph)).not.toThrow();

  // The same bytes, or overlapping ranges, are refused.
  expect(() =>
    create(withStatus({fillConverged: flagsAt(0), accumulationConverged: flagsAt(0)}))
  ).toThrow(/overlapping byte ranges/);
  expect(() =>
    create({
      output: {
        flowDirections: graph.createDataView(summary, {format: 'uint32', length: 30}),
        cellClasses: graph.createDataView(summary, {format: 'uint32', length: 30, byteOffset: 64})
      }
    })
  ).toThrow(/overlapping byte ranges/);

  // Disjoint ranges are not enough for outputs written by one node: they would share a
  // storage binding window.
  const columns = graph.createTransientBuffer({id: 'columns', byteLength: 4096, usage: 128});
  const column = (byteOffset: number) =>
    graph.createDataView(columns, {format: 'uint32', length: 30, byteOffset});
  expect(() => create({output: {flowDirections: column(0), cellClasses: column(128)}})).toThrow(
    /256 bytes apart/
  );
  expect(() =>
    create({output: {flowDirections: column(0), cellClasses: column(256)}})
  ).not.toThrow();

  // An output may never share a buffer with an input, even over disjoint ranges.
  const inputs = graph.createTransientBuffer({id: 'inputs', byteLength: 4096, usage: 128});
  const parameters = graph.createDataView(inputs, {format: 'float32', length: 8});
  expect(() =>
    create({
      parameters,
      fillDepressions: true,
      output: {
        status: {
          fillConverged: graph.createDataView(inputs, {
            format: 'uint32',
            length: 1,
            byteOffset: 1024
          })
        }
      }
    })
  ).toThrow(/share buffers with inputs/);
  device.destroy();
});

it('review C-09: a truncated fill or flat resolution marks every valid output unresolved', () => {
  const {device, graph, create} = createFixture();
  const nodes = create({
    id: 'trunc',
    fillDepressions: true,
    resolveFlats: true,
    maximumFillIterations: 1,
    maximumFlatIterations: 1,
    output: {
      filledElevation: createTransientView(graph, 'trunc-filled', 'float32', 30),
      cellClasses: createTransientView(graph, 'trunc-classes', 'uint32', 30),
      accumulation: createTransientView(graph, 'trunc-accumulation', 'float32', 30),
      streams: createTransientView(graph, 'trunc-streams', 'uint32', 30)
    }
  }).getCommandNodes(graph);
  const ids = nodes.map(node => node.id);
  expect(ids.slice(-2)).toEqual(['trunc-unresolved-arm', 'trunc-unresolved-cells']);
  const cells = nodes.at(-1)!;
  expect(cells.condition).toMatchObject({source: 'gpu', mode: 'indirect'});
  expect((cells.resources ?? []).length).toBeLessThanOrEqual(8);
  const sources = new Map(
    getRasterPipelines(nodes).map(pipeline => [pipeline.id, pipeline.source])
  );
  const cellSource = sources.get('trunc-unresolved-cells')!;
  expect(cellSource).toContain(
    `directions[directionsOffset + index] = ${GPU_TERRAIN_FLOW_UNRESOLVED}u;`
  );
  expect(cellSource).toContain(
    `classes[classesOffset + index] = ${GPU_TERRAIN_FLOW_CELL_CLASS.unresolved}u;`
  );
  expect(cellSource).toContain('accumulation[accumulationOffset + index] = 0x7fc00000u;');
  expect(cellSource).toContain(
    `streams[streamsOffset + index] = ${GPU_TERRAIN_FLOW_STREAM_UNRESOLVED}u;`
  );
  // The arm requires the fill and all three flat relaxations to converge.
  expect(sources.get('trunc-unresolved-arm')!.match(/Offset \+ 3u\] != 0u/g)).toHaveLength(4);
  // Without fill or flats there is nothing to mark.
  expect(
    create({id: 'plain'})
      .getCommandNodes(graph)
      .map(node => node.id)
  ).not.toContain('plain-unresolved-arm');
  device.destroy();
});

it('GPUTerrainFlow rejects routing scratch above the binding limit at getCommandNodes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  // 4500 x 4500 float32 bands are 81 MB and pass the band checks, but the D-infinity routing
  // records are 8 B per cell (162 MB), above the 128 MiB default storage binding limit.
  const grid = makeGPURasterMetadata({width: 4500, height: 4500});
  const cellCount = 4500 * 4500;
  const flow = new GPUTerrainFlow({
    grid,
    elevation: makeGPURasterBand(
      createTransientView(graph, 'large-elevation', 'float32', cellCount),
      grid,
      {id: 'large-elevation'}
    ),
    parameters: createTransientView(graph, 'large-settings', 'float32', 8),
    flowRouting: 'd-infinity',
    output: {accumulation: createTransientView(graph, 'large-accumulation', 'float32', cellCount)}
  });
  expect(() => flow.getCommandNodes(graph)).toThrow(
    /routing-prepare binding routing needs 162000000 bytes/
  );
  device.destroy();
});
