// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it, vi} from 'vitest';
import {log} from '@luma.gl/core';
import {WgslReflect} from 'wgsl_reflect';
import {
  getGPUGreatCircleArcsParameterValues,
  getGPULineChunkParameterValues,
  getGPULineSegmentizeParameterValues,
  getGPULineSmoothParameterValues,
  GPUGreatCircleArcs,
  GPULineChunk,
  GPULineSegmentize,
  GPULineSmooth,
  type GPULineGeometryOutput,
  type GPULineSegmentizeProps
} from '../../../src/gpu-spatial-analysis/line-segmentize';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {countDistinctPipelines} from '../../utils/gpu-contributor-test-utils';

const GREAT_CIRCLE = {
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle'
} as const;

let viewSerial = 0;

/** Returns a graph-unique resource ID. */
function getViewId(name: string): string {
  return `${name}-${viewSerial++}`;
}
import {
  createFlatLines,
  segmentizeLines,
  tessellateGreatCircleArcs
} from './line-segmentize-oracle';

function createOutput(
  graph: GPUCommandGraph,
  lineCount: number,
  capacity = 64
): GPULineGeometryOutput {
  return {
    positions: createTransientView(graph, getViewId('o-positions'), 'float32x2', capacity),
    lineOffsets: createTransientView(graph, getViewId('o-offsets'), 'uint32', lineCount + 1),
    status: {
      count: createTransientView(graph, getViewId('o-count'), 'uint32', 1),
      overflow: createTransientView(graph, getViewId('o-overflow'), 'uint32', 1)
    },
    measures: createTransientView(graph, getViewId('o-measures'), 'float32', capacity)
  };
}

function createSegmentizeProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPULineSegmentizeProps> = {}
): GPULineSegmentizeProps {
  return {
    positions: createTransientView(graph, getViewId('positions'), 'float32x2', 10),
    lineOffsets: createTransientView(graph, getViewId('offsets'), 'uint32', 3),
    parameters: createTransientView(graph, getViewId('parameters'), 'float32', 4),
    output: createOutput(graph, 2),
    ...overrides
  };
}

it('GPULineSegmentize validates inputs and emits deterministic node IDs', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'segmentize'});
  const contributor = new GPULineSegmentize({
    ...createSegmentizeProps(graph),
    spatialContext: GREAT_CIRCLE
  });
  expect(contributor.coordinateSystem).toBe('spherical');
  expect(contributor.radius).toBe(6371008.8);
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
  expect(ids[0]).toBe('line-segmentize-prefix');
  expect(ids).toContain('line-segmentize-count');
  expect(ids).toContain('line-segmentize-emit');
  expect(ids.at(-1)).toBe('line-segmentize-publish');

  // Planar is the default; an unsupported context or a bad radius throws at construction.
  expect(new GPULineSegmentize(createSegmentizeProps(graph)).coordinateSystem).toBe('planar');
  expect(
    () =>
      new GPULineSegmentize(
        createSegmentizeProps(graph, {
          spatialContext: {coordinateSpace: 'discrete-grid', metric: 'none', gridFamily: 'h3'}
        })
      )
  ).toThrow(/does not support discrete-grid/);
  expect(
    () =>
      new GPULineSegmentize(
        createSegmentizeProps(graph, {
          spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'}
        })
      )
  ).toThrow(/great-circle/);
  expect(
    () =>
      new GPULineSegmentize(
        createSegmentizeProps(graph, {spatialContext: {...GREAT_CIRCLE, sphereRadius: -1}})
      )
  ).toThrow(/sphereRadius/);
  expect(
    new GPULineSegmentize(
      createSegmentizeProps(graph, {spatialContext: {...GREAT_CIRCLE, units: 'kilometers'}})
    ).radius
  ).toBeCloseTo(6371.0088, 6);
  expect(
    () =>
      new GPULineSegmentize(
        createSegmentizeProps(graph, {
          parameters: createTransientView(graph, getViewId('short-parameters'), 'float32', 2)
        })
      )
  ).toThrow(/parameters/);
  expect(
    () => new GPULineSegmentize(createSegmentizeProps(graph, {output: createOutput(graph, 5)}))
  ).toThrow(/lineOffsets/);
  expect(
    () => new GPULineSegmentize(createSegmentizeProps(graph, {maximumPieceCount: 2 ** 30}))
  ).toThrow(/maximumPieceCount/);
  const props = createSegmentizeProps(graph);
  expect(
    () =>
      new GPULineSegmentize({
        ...props,
        output: {...props.output, measures: undefined, positions: props.positions}
      })
  ).toThrow(/share buffers/);
});

it('GPUGreatCircleArcs validates pair inputs', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'arcs'});
  const sources = createTransientView(graph, getViewId('sources'), 'float32x2', 3);
  const parameters = createTransientView(graph, getViewId('parameters'), 'float32', 4);
  const contributor = new GPUGreatCircleArcs({
    sources,
    targets: createTransientView(graph, getViewId('targets'), 'float32x2', 3),
    parameters,
    output: createOutput(graph, 3)
  });
  expect(contributor.getCommandNodes(graph).map(node => node.id)).toContain(
    'great-circle-arcs-emit'
  );
  expect(
    () =>
      new GPUGreatCircleArcs({
        sources,
        targets: createTransientView(graph, getViewId('short-targets'), 'float32x2', 2),
        parameters,
        output: createOutput(graph, 3)
      })
  ).toThrow(/targets length/);
});

it('line segmentize parameter packing validates values', () => {
  expect(Array.from(getGPULineSegmentizeParameterValues({maximumSegmentLength: 2.5}))).toEqual([
    2.5, 0, 0, 0
  ]);
  expect(Array.from(getGPULineSegmentizeParameterValues({maximumSegmentLength: Infinity}))).toEqual(
    [0, 0, 0, 0]
  );
  expect(() => getGPULineSegmentizeParameterValues({maximumSegmentLength: -1})).toThrow();
  expect(
    Array.from(
      getGPUGreatCircleArcsParameterValues({maximumSegmentLength: 1000, minimumSegments: 8})
    )
  ).toEqual([1000, 8, 0, 0]);
  expect(() =>
    getGPUGreatCircleArcsParameterValues({maximumSegmentLength: 0, minimumSegments: 0.5})
  ).toThrow(/minimumSegments/);
  expect(() =>
    getGPULineSegmentizeParameterValues({maximumSegmentLength: 1}, new Float32Array(2))
  ).toThrow(/target/);
});

it('line segmentize oracles keep vertices, measures and endpoints', () => {
  const lines = createFlatLines([
    [
      [0, 0],
      [3, 4],
      [3, 4],
      [3, 5]
    ],
    [],
    [[1, 1]]
  ]);
  const result = segmentizeLines(lines, {maximumSegmentLength: 2, maximumPieces: 100});
  expect(result.lineOffsets).toEqual([0, 6, 6, 7]);
  expect(result.positions[1]).toEqual([1, 4 / 3]);
  expect(result.measures[3]).toBe(5);
  expect(result.measures[5]).toBe(6);
  const arcs = tessellateGreatCircleArcs([[170, 0]], [[-170, 0]], {
    maximumSegmentLength: 0,
    minimumSegments: 4,
    maximumSegmentCount: 10,
    radius: 1
  });
  expect(arcs.positions.at(-1)).toEqual([190, 0]);
  expect(arcs.positions[2][0]).toBeCloseTo(180, 10);
});

it('GPULineSmooth and GPULineChunk validate their layouts', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'smooth-chunk'});
  const positions = createTransientView(graph, getViewId('positions'), 'float32x2', 10);
  const lineOffsets = createTransientView(graph, getViewId('offsets'), 'uint32', 3);
  const parameters = createTransientView(graph, getViewId('parameters'), 'float32', 4);
  const smooth = new GPULineSmooth({
    positions,
    lineOffsets,
    iterationCount: 3,
    parameters,
    output: {...createOutput(graph, 2), measures: undefined}
  });
  const smoothIds = smooth.getCommandNodes(graph).map(node => node.id);
  expect(smoothIds.filter(id => /^line-smooth-level-\d$/.test(id))).toEqual([
    'line-smooth-level-0',
    'line-smooth-level-1',
    'line-smooth-level-2'
  ]);
  expect(
    () =>
      new GPULineSmooth({
        positions,
        lineOffsets,
        iterationCount: 3,
        parameters,
        output: createOutput(graph, 2)
      })
  ).toThrow(/not supported/);
  expect(
    () =>
      new GPULineSmooth({
        positions,
        lineOffsets,
        iterationCount: 11,
        parameters,
        output: {...createOutput(graph, 2), measures: undefined}
      })
  ).toThrow(/iterationCount/);
  expect(Array.from(getGPULineSmoothParameterValues())).toEqual([0.25, 0, 0, 0]);
  expect(() => getGPULineSmoothParameterValues({ratio: 0.75})).toThrow(/ratio/);

  // Chunk mode: line capacity comes from the output; substring mode needs one line per input.
  const chunk = new GPULineChunk({
    positions,
    lineOffsets,
    mode: 'chunk',
    parameters,
    output: createOutput(graph, 7)
  });
  expect(chunk.lineCapacity).toBe(7);
  expect(chunk.getCommandNodes(graph).map(node => node.id)).toContain('line-chunk-piece-range');
  expect(
    () =>
      new GPULineChunk({
        positions,
        lineOffsets,
        mode: 'substring',
        parameters,
        output: createOutput(graph, 7)
      })
  ).toThrow(/lineOffsets/);
  expect(Array.from(getGPULineChunkParameterValues({chunkLength: 2}))).toEqual([
    2,
    0,
    Math.fround(3.4e38),
    0
  ]);
  expect(() => getGPULineChunkParameterValues({chunkLength: -1})).toThrow();
});

it('line operations declare their spatial context', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'line-contexts'});
  const positions = createTransientView(graph, getViewId('positions'), 'float32x2', 10);
  const lineOffsets = createTransientView(graph, getViewId('offsets'), 'uint32', 3);
  const parameters = createTransientView(graph, getViewId('parameters'), 'float32', 4);
  // Planar-only: great-circle longitude/latitude throws with a project-first hint.
  expect(
    () =>
      new GPULineSmooth({
        positions,
        lineOffsets,
        iterationCount: 1,
        parameters,
        spatialContext: GREAT_CIRCLE,
        output: {...createOutput(graph, 2), measures: undefined}
      })
  ).toThrow(/project positions to a planar frame first/);
  // Chunk accepts both.
  const chunkProps = {
    positions,
    lineOffsets,
    mode: 'chunk' as const,
    parameters,
    output: {...createOutput(graph, 2), measures: undefined}
  };
  expect(new GPULineChunk(chunkProps).coordinateSystem).toBe('planar');
  expect(new GPULineChunk({...chunkProps, spatialContext: GREAT_CIRCLE}).coordinateSystem).toBe(
    'spherical'
  );
  // Great-circle arcs are inherently geographic: default is longitude/latitude, planar throws.
  const arcProps = {
    sources: createTransientView(graph, getViewId('sources'), 'float32x2', 3),
    targets: createTransientView(graph, getViewId('targets'), 'float32x2', 3),
    parameters,
    output: createOutput(graph, 3)
  };
  expect(new GPUGreatCircleArcs(arcProps).spatialContext.metric).toBe('great-circle');
  expect(
    () =>
      new GPUGreatCircleArcs({
        ...arcProps,
        spatialContext: {coordinateSpace: 'planar', metric: 'native'}
      })
  ).toThrow(/does not support planar/);
});

it('GPULineSmooth builds the same pipelines for any number of passes and parses', () => {
  for (const closed of [false, true]) {
    const buildNodes = (iterationCount: number) => {
      const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: `smooth-${iterationCount}`});
      const positions = createTransientView(graph, getViewId('positions'), 'float32x2', 10);
      const lineOffsets = createTransientView(graph, getViewId('offsets'), 'uint32', 3);
      const parameters = createTransientView(graph, getViewId('parameters'), 'float32', 4);
      return new GPULineSmooth({
        positions,
        lineOffsets,
        iterationCount,
        closed,
        parameters,
        output: {...createOutput(graph, 2), measures: undefined}
      }).getCommandNodes(graph);
    };
    // The level number lives in a GPU word, so levels share pipelines (rule 12).
    expect(countDistinctPipelines(buildNodes(4))).toBe(countDistinctPipelines(buildNodes(3)));
    expect(countDistinctPipelines(buildNodes(5))).toBe(countDistinctPipelines(buildNodes(4)));
    for (const {source} of getGPUCommandNodeKernelSources(buildNodes(3))) {
      const reflect = new WgslReflect(source);
      expect(reflect.storage.length).toBeLessThanOrEqual(8);
    }
  }
});

it('GPULineSegmentize validates closedLines and builds closed-ring kernels with every output', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'closed-lines'});
  expect(
    () =>
      new GPULineSegmentize(
        createSegmentizeProps(graph, {
          closedLines: 1 as unknown as boolean
        })
      )
  ).toThrow('line-segmentize closedLines must be a boolean');
  for (const closedLines of [false, true]) {
    const props = createSegmentizeProps(graph, {
      id: `closed-${closedLines}`,
      closedLines,
      spatialContext: GREAT_CIRCLE
    });
    props.output.sourceRows = createTransientView(graph, getViewId('source-rows'), 'uint32', 64);
    const nodes = new GPULineSegmentize(props).getCommandNodes(graph);
    for (const {source} of getGPUCommandNodeKernelSources(nodes)) {
      expect(new WgslReflect(source).storage.length).toBeLessThanOrEqual(8);
    }
  }
});

it('line length writers warn for short lon/lat lengths without changing packed values', () => {
  const writers = [
    {
      operation: 'GPULineSegmentize',
      write: (
        spatialContext?: typeof GREAT_CIRCLE | {coordinateSpace: 'planar'; metric: 'native'}
      ) => getGPULineSegmentizeParameterValues({maximumSegmentLength: 5, spatialContext})
    },
    {
      operation: 'GPUGreatCircleArcs',
      write: (
        spatialContext?: typeof GREAT_CIRCLE | {coordinateSpace: 'planar'; metric: 'native'}
      ) => getGPUGreatCircleArcsParameterValues({maximumSegmentLength: 5, spatialContext})
    },
    {
      operation: 'GPULineChunk',
      write: (
        spatialContext?: typeof GREAT_CIRCLE | {coordinateSpace: 'planar'; metric: 'native'}
      ) => getGPULineChunkParameterValues({chunkLength: 5, spatialContext})
    }
  ];
  const warning = vi.spyOn(log, 'warn');
  try {
    for (const {operation, write} of writers) {
      warning.mockClear();
      const expected = write();
      expect(write({coordinateSpace: 'planar', metric: 'native'})).toEqual(expected);
      expect(warning).not.toHaveBeenCalled();
      expect(write(GREAT_CIRCLE)).toEqual(expected);
      expect(warning).toHaveBeenCalledExactlyOnceWith(
        `${operation} distance is too short for float32 lon/lat; use createGPULocalFrame`
      );
    }
  } finally {
    warning.mockRestore();
  }
});

it('segmentize oracle closes rings, removes only a repeated closing row and retains empty lines', () => {
  const square = [
    [0, 0],
    [2, 0],
    [2, 2],
    [0, 2]
  ];
  const lines = createFlatLines([
    square,
    [...square, square[0]],
    [],
    [[7, 8]],
    [
      [4, 5],
      [4, 5]
    ]
  ]);
  const result = segmentizeLines(lines, {
    closedLines: true,
    maximumSegmentLength: 1,
    maximumPieces: 32
  });
  expect(result.lineOffsets).toEqual([0, 8, 16, 16, 17, 18]);
  expect(result.positions.slice(0, 8)).toEqual([
    [0, 0],
    [1, 0],
    [2, 0],
    [2, 1],
    [2, 2],
    [1, 2],
    [0, 2],
    [0, 1]
  ]);
  expect(result.positions.slice(8, 16)).toEqual(result.positions.slice(0, 8));
  expect(result.measures[7]).toBe(7);
});

it('segmentize degree-step oracle counts angular extent, preserves turns and reports clamping', () => {
  const lines = createFlatLines([
    [
      [150, 45],
      [-150, 45]
    ]
  ]);
  const options = {
    maximumSegmentLength: 0,
    maximumSegmentDegrees: 0.1,
    maximumPieces: 1024,
    spherical: true,
    pieceCountModel: 'degree-step' as const,
    wrapOutputLongitudes: true
  };
  const result = segmentizeLines(lines, options);
  expect(result.positions).toHaveLength(601);
  expect(result.positions[0]).toEqual([150, 45]);
  expect(result.positions[600]).toEqual([-150, 45]);
  expect(result.turns[600]).toBe(1);
  expect(result.positions[300][1]).toBeGreaterThan(45);
  const linear = segmentizeLines(lines, {...options, interpolation: 'linear-degrees'});
  expect(linear.positions[300]).toEqual([180, 45]);
  expect(linear.approximation).toBe(0);
  expect(segmentizeLines(lines, {...options, maximumPieces: 8}).approximation).toBe(1);
  expect(segmentizeLines(lines, {...options, maximumSegmentDegrees: 0}).positions).toHaveLength(2);
  const latitude = segmentizeLines(
    createFlatLines([
      [
        [1, -20],
        [2, 20]
      ]
    ]),
    {
      ...options,
      maximumSegmentDegrees: 10,
      interpolation: 'linear-degrees'
    }
  );
  expect(latitude.positions).toHaveLength(5);
});
