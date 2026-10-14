// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GPUCommandNode} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUPolygonTriangulation,
  type GPUPolygonTriangulationProps,
  getPolygonTriangulationIndexCount
} from '../../../src/gpu-spatial-analysis/polygon-triangulation/index';
import {
  getGPUPolygonTriangulationMonotoneMaximumVertexCount,
  getMonotoneJumpPairCount,
  getMonotoneTriangulationNodes,
  type MonotoneTriangulationProps
} from '../../../src/gpu-spatial-analysis/polygon-triangulation/polygon-triangulation-monotone';
import {
  CORE_PRIVATE_MEMORY_LIMIT_BYTES,
  CORE_WORKGROUP_STORAGE_LIMIT_BYTES,
  createInputBuffer,
  createNullWebGPUDevice,
  createOutputBuffer,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getPrivateMemoryBytes,
  getStorageBindingCount,
  getWorkgroupMemoryBytes,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

function getEarcutSource(): string {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'triangulation'});
  const view = <Format extends 'float32x2' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  const contributor = new GPUPolygonTriangulation({
    positions: view('positions', 'float32x2', 8),
    featureOffsets: view('feature-offsets', 'uint32', 2),
    polygonOffsets: view('polygon-offsets', 'uint32', 2),
    ringOffsets: view('ring-offsets', 'uint32', 3),
    output: {
      indices: view('indices', 'uint32', getPolygonTriangulationIndexCount(8, 2, 1)),
      valid: view('valid', 'uint32', 1)
    }
  });
  const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
  const earcut = sources.find(entry => entry.id.endsWith('-earcut'));
  expect(earcut).toBeDefined();
  return earcut!.source;
}

function getFunctionBody(source: string, name: string): string {
  const start = source.indexOf(`fn ${name}(`);
  expect(start).toBeGreaterThanOrEqual(0);
  const next = source.indexOf('\nfn ', start + 1);
  return source.slice(start, next < 0 ? undefined : next);
}

it('review Rule 1: middleInside decides the diagonal parity with the exact SoS rule', () => {
  const source = getEarcutSource();
  expect(() => new WgslReflect(source)).not.toThrow();
  const middleInside = getFunctionBody(source, 'middleInside');
  // Exact parity: orientOf (the kernel's existing orientSign wrapper) plus the helper's SoS side.
  expect(middleInside).toContain('sosPointSide(orientOf(');
  expect(middleInside).toContain('>= middle.y');
  // The old f32 division crossing is gone.
  expect(middleInside).not.toMatch(/\) \/ \(edgeEnd\.y - edgeStart\.y\)/);
  // No new orientSign call site: the existing wrappers and ringWinding call it, and the helper's
  // exactOrientSigns (never called by this kernel) keeps its own.
  const callers = source
    .split('\nfn ')
    .filter(chunk => /\borientSign\(/.test(chunk.split('\n').slice(1).join('\n')))
    .map(chunk => chunk.split('(')[0]);
  expect(callers.sort()).toEqual(['areaSign', 'exactOrientSigns', 'orientOf', 'ringWinding']);
  expect(middleInside).not.toMatch(/\borientSign\(/);
  // exactOrientSigns is reachable only from the helper's probe and segment functions, which this
  // kernel never calls (a declaration plus the definition of each, no other call site).
  expect(source.match(/\bexactProbeCrossing\(/g)?.length).toBe(1);
  expect(source.match(/\bexactSegmentsCross\(/g)?.length).toBe(1);
  expect(source.match(/\bexactCompareSign\(/g)?.length).toBeGreaterThan(1);
});

/** Builds the nodes of a polygon layout with `vertexCount` vertices in two polygons. */
function buildNodes(
  props: Partial<GPUPolygonTriangulationProps> = {},
  vertexCount: number = 5000
): {nodes: readonly GPUCommandNode<unknown>[]; graph: GPUCommandGraph<unknown>} {
  const graph = new GPUCommandGraph<unknown>(createNullWebGPUDevice(), {id: 'triangulation'});
  const view = <Format extends 'float32x2' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  const contributor = new GPUPolygonTriangulation({
    positions: view('positions', 'float32x2', vertexCount),
    featureOffsets: view('feature-offsets', 'uint32', 3),
    polygonOffsets: view('polygon-offsets', 'uint32', 3),
    ringOffsets: view('ring-offsets', 'uint32', 4),
    output: {
      indices: view('indices', 'uint32', getPolygonTriangulationIndexCount(vertexCount, 3, 2)),
      valid: view('valid', 'uint32', 2),
      triangleCount: view('triangle-count', 'uint32', 2)
    },
    ...props
  });
  return {nodes: contributor.getCommandNodes(graph), graph};
}

it('largePolygonMethod none prepares the layout then builds Earcut without routing', () => {
  const {nodes} = buildNodes({largePolygonMethod: 'none'});
  expect(nodes.at(-1)?.id).toBe('polygon-triangulation-earcut');
  expect(nodes.some(node => node.id.includes('-monotone-'))).toBe(false);
  const earcut = getGPUCommandNodeKernelSources(nodes).find(kernel =>
    kernel.id.endsWith('-earcut')
  )!;
  expect(earcut.source).not.toContain('routing');
  expect(getStorageBindingCount(earcut.source)).toBe(7);
});

it('largePolygonMethod monotone routes inside Earcut and triangulates the large polygons after it', () => {
  const {nodes} = buildNodes();
  const ids = nodes.map(node => node.id);
  const earcutIndex = ids.indexOf('polygon-triangulation-earcut');
  // Layout preparation precedes Earcut; size routing remains inside Earcut.
  expect(earcutIndex).toBeGreaterThan(0);
  // The monotone nodes follow, then a second Earcut pass for the polygons they flagged invalid.
  expect(ids[ids.length - 1]).toBe('polygon-triangulation-earcut-fallback');
  expect(
    ids.slice(earcutIndex + 1, -1).every(id => id.startsWith('polygon-triangulation-monotone-'))
  ).toBe(true);
  const sources = getGPUCommandNodeKernelSources(nodes);
  const earcut = sources.find(entry => entry.id.endsWith('-earcut'))!;
  // Size-routed polygons are skipped, failed ones are marked for the retry; nothing else changed.
  expect(earcut.source).toContain('routing[routingOffset + polygon] = sizeRouteOf(polygon);');
  expect(earcut.source).toContain('if (routing[routingOffset + polygon] == 1u) { return; }');
  expect(earcut.source).toContain('routing[routingOffset + polygon] = 2u;');
  expect(getStorageBindingCount(earcut.source)).toBe(8);
  const fallback = sources.find(entry => entry.id.endsWith('-earcut-fallback'))!;
  expect(fallback.source).toContain('if (routing[routingOffset + polygon] != 4u) { return; }');
  expect(fallback.source).not.toContain('routing[routingOffset + polygon] = 2u');
  const names = ids.slice(0, -1).map(id => id.replace('polygon-triangulation-monotone-', ''));
  // The kernels of the design, in order.
  const order = [
    'route-counts',
    'route-overflow',
    'index',
    'ring',
    'counts',
    'compact',
    'keys',
    'rank',
    'flip',
    'link',
    'left-edge',
    'members',
    'diagonals',
    'next',
    'jump-init',
    'jump-0-state',
    'rep-count',
    'pieces',
    'piece-count',
    'measure',
    'polygon-plan',
    'piece-plan',
    'triangulate',
    'final',
    'pad'
  ];
  let cursor = -1;
  for (const name of order) {
    const found = names.indexOf(name);
    expect(found, `${name} is declared`).toBeGreaterThan(cursor);
    cursor = found;
  }
});

it('every monotone kernel parses, binds at most 8 storage buffers and stays within core limits', () => {
  const {nodes} = buildNodes();
  const sources = getGPUCommandNodeKernelSources(nodes);
  expect(sources.length).toBeGreaterThan(40);
  for (const kernel of sources) {
    expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
    expect(getStorageBindingCount(kernel.source), `${kernel.id} bindings`).toBeLessThanOrEqual(8);
    expect(getPrivateMemoryBytes(kernel.source), `${kernel.id} private`).toBeLessThanOrEqual(
      CORE_PRIVATE_MEMORY_LIMIT_BYTES
    );
    expect(getWorkgroupMemoryBytes(kernel.source), `${kernel.id} workgroup`).toBeLessThanOrEqual(
      CORE_WORKGROUP_STORAGE_LIMIT_BYTES
    );
    if (
      kernel.id.includes('-monotone-') ||
      /-(classify-rings|polygon-capacities|publish-layout)$/.test(kernel.id)
    ) {
      // Reserved words, NaN self-compares and calls to undeclared functions, the compile traps
      // that wgsl_reflect accepts and Dawn rejects.
      expect(auditWGSLModule(kernel.source), kernel.id).toEqual([]);
    }
  }
});

it('the monotone path orients through one wrapper and tests NaN-free (rules 6 and 12 helpers)', () => {
  const sources = getGPUCommandNodeKernelSources(buildNodes().nodes).filter(entry =>
    entry.id.includes('-monotone-')
  );
  for (const kernel of sources) {
    const callers = kernel.source
      .split('\nfn ')
      .filter(chunk => /\borientSign\(/.test(chunk.split('\n').slice(1).join('\n')))
      .map(chunk => chunk.split('(')[0]);
    // orientSign is called by the wrapper only, and the exact helper functions are not duplicated.
    expect(callers.length, kernel.id).toBeLessThanOrEqual(1);
    if (callers.length === 1) expect(callers[0], kernel.id).toBe('orientOf');
    expect(kernel.source, kernel.id).not.toMatch(/(\w+) == \1\b/);
  }
});

it('the pointer-jumping rounds do not reach the WGSL (rule 12) and are gated on a GPU word (rule 13)', () => {
  const graph = new GPUCommandGraph<unknown>(createNullWebGPUDevice(), {id: 'rounds'});
  const view = <Format extends 'float32x2' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  const vertexCount = 600;
  const base = (name: string): Omit<MonotoneTriangulationProps, 'jumpPairCount'> => ({
    id: name,
    operation: 'GPUPolygonTriangulation',
    positions: view(`${name}-positions`, 'float32x2', vertexCount),
    polygonOffsets: view(`${name}-polygon-offsets`, 'uint32', 3),
    ringOffsets: view(`${name}-ring-offsets`, 'uint32', 4),
    routing: view(`${name}-routing`, 'uint32', 2),
    output: {
      indices: view(
        `${name}-indices`,
        'uint32',
        getPolygonTriangulationIndexCount(vertexCount, 3, 2)
      ),
      valid: view(`${name}-valid`, 'uint32', 2)
    },
    capacity: vertexCount,
    minimumVertices: 100
  });
  // Same views would share transient IDs; the kernels do not depend on the IDs (the offsets are 0).
  const four = getMonotoneTriangulationNodes(graph, {...base('four'), jumpPairCount: 4});
  const five = getMonotoneTriangulationNodes(graph, {...base('five'), jumpPairCount: 5});
  expect(getGPUCommandNodePipelineKeys(five).size).toBe(getGPUCommandNodePipelineKeys(four).size);
  // Per pair: the state step, the dispatch update, and the two gated rounds.
  expect(five.length - four.length).toBe(4);
  const jumps = five.filter(node => /-jump-\d+-(forward|backward)$/.test(node.id));
  expect(jumps).toHaveLength(10);
  for (const node of jumps) {
    expect(node.condition, node.id).toMatchObject({source: 'gpu', mode: 'indirect'});
  }
  // The state step and the dispatch update that write the gate stay ungated.
  for (const node of five.filter(entry => /-jump-\d+-state$|-jump-\d+-dispatch$/.test(entry.id))) {
    expect(node.condition, node.id).toBeUndefined();
  }
  // Rounds enough for polygons of `capacity` vertices: 2^rounds reaches 3 n + 3.
  expect(2 ** (2 * getMonotoneJumpPairCount(vertexCount))).toBeGreaterThanOrEqual(
    3 * vertexCount + 3
  );
  expect(2 ** (2 * getMonotoneJumpPairCount(vertexCount) - 2)).toBeLessThan(
    4 * (3 * vertexCount + 3)
  );
});

it('the work after the routing is sized by GPU-resident counts (indirect dispatch)', () => {
  const {nodes} = buildNodes();
  const gated = nodes.filter(node => node.condition);
  const names = gated.map(node => node.id.replace('polygon-triangulation-monotone-', ''));
  for (const name of [
    'index',
    'ring',
    'compact',
    'rank',
    'flip',
    'link',
    'left-edge',
    'diagonals',
    'next',
    'jump-init',
    'rep-count',
    'pieces',
    'measure',
    'piece-plan',
    'triangulate',
    'pad'
  ]) {
    expect(names, name).toContain(name);
  }
  // The one-thread and static-size kernels are not gated: routing, state words, sort keys.
  for (const name of ['route-counts', 'route-overflow', 'counts', 'keys', 'members', 'final']) {
    expect(names, name).not.toContain(name);
  }
});

it('validates the large polygon options and the capacity', () => {
  expect(() => buildNodes({largePolygonMethod: 'delaunay' as 'none'})).toThrow(
    /largePolygonMethod/
  );
  expect(() => buildNodes({monotoneMinimumVertices: 0})).toThrow(/monotoneMinimumVertices/);
  expect(() => buildNodes({monotoneMaximumVertices: 0})).toThrow(/monotoneMaximumVertices/);
  const device = createNullWebGPUDevice();
  const maximum = getGPUPolygonTriangulationMonotoneMaximumVertexCount(device);
  expect(maximum).toBe(Math.floor(134217728 / 40));
  expect(() => buildNodes({monotoneMaximumVertices: maximum + 1})).toThrow(
    /monotoneMaximumVertices must be at most/
  );
  // A capacity above the input is clamped, below it sizes the scratch.
  const small = getGPUCommandNodeKernelSources(buildNodes({monotoneMaximumVertices: 100}).nodes);
  expect(small.find(entry => entry.id.endsWith('-monotone-keys'))!.source).toContain(
    'CAPACITY: u32 = 100u'
  );
  const whole = getGPUCommandNodeKernelSources(
    buildNodes({monotoneMaximumVertices: 3_000_000}).nodes
  );
  expect(whole.find(entry => entry.id.endsWith('-monotone-keys'))!.source).toContain(
    'CAPACITY: u32 = 5000u'
  );
});

it('the default never throws for inputs above the monotone limit: they behave like none', () => {
  const device = createNullWebGPUDevice();
  const rows = getGPUPolygonTriangulationMonotoneMaximumVertexCount(device) + 1;
  expect(rows).toBe(3_355_444);
  const {nodes} = buildNodes({useZOrderHash: false}, rows);
  expect(nodes.at(-1)?.id).toBe('polygon-triangulation-earcut');
  expect(nodes.some(node => node.id.includes('-monotone-'))).toBe(false);
  expect(getGPUCommandNodeKernelSources(nodes).at(-1)!.source).not.toContain('routing');
});

it('accepts empty geometry and legacy capacities while validating triangle offset output', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const view = <Format extends 'uint32' | 'float32x2'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  const props: GPUPolygonTriangulationProps = {
    positions: view('empty-positions', 'float32x2', 0),
    featureOffsets: view('empty-features', 'uint32', 2),
    polygonOffsets: view('empty-polygons', 'uint32', 1),
    ringOffsets: view('empty-rings', 'uint32', 1),
    output: {
      indices: view('empty-indices', 'uint32', 0),
      valid: view('empty-valid', 'uint32', 0),
      triangleCount: view('empty-counts', 'uint32', 0),
      triangleOffsets: view('empty-triangle-offsets', 'uint32', 1)
    }
  };
  expect(() => new GPUPolygonTriangulation(props).getCommandNodes(graph)).not.toThrow();
  expect(
    () =>
      new GPUPolygonTriangulation({
        ...props,
        output: {...props.output, triangleOffsets: view('bad-offsets', 'uint32', 2)}
      })
  ).toThrow(/triangleOffsets/);
  expect(getPolygonTriangulationIndexCount(4, 1, 5)).toBe(6);
  expect(getPolygonTriangulationIndexCount(0, 0, 0)).toBe(0);
  // Two ordinary squares still accept the former exact capacity, 12 index words.
  expect(() =>
    new GPUPolygonTriangulation({
      id: 'legacy-capacity',
      positions: view('positions', 'float32x2', 8),
      featureOffsets: view('features', 'uint32', 2),
      polygonOffsets: view('polygons', 'uint32', 3),
      ringOffsets: view('rings', 'uint32', 3),
      output: {indices: view('indices', 'uint32', 12), valid: view('valid', 'uint32', 2)}
    }).getCommandNodes(graph)
  ).not.toThrow();
});

it('ordinary polygons retain the original Earcut index order and slice offsets', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device, {id: 'ordinary-polygon-triangulation'});
  const square = [0, 0, 4, 0, 4, 4, 0, 4];
  const buffers = {
    positions: createInputBuffer(device, new Float32Array([...square, ...square])),
    featureOffsets: createInputBuffer(device, new Uint32Array([0, 1, 2])),
    polygonOffsets: createInputBuffer(device, new Uint32Array([0, 1, 2])),
    ringOffsets: createInputBuffer(device, new Uint32Array([0, 4, 8])),
    // The original exact capacity for two squares remains supported.
    indices: createOutputBuffer(device, 12),
    valid: createOutputBuffer(device, 2),
    triangleCount: createOutputBuffer(device, 2),
    triangleOffsets: createOutputBuffer(device, 3)
  };
  try {
    graph.add(
      new GPUPolygonTriangulation({
        positions: importGraphBuffer(graph, 'positions', buffers.positions, 'float32x2', 8),
        featureOffsets: importGraphBuffer(graph, 'features', buffers.featureOffsets, 'uint32', 3),
        polygonOffsets: importGraphBuffer(graph, 'polygons', buffers.polygonOffsets, 'uint32', 3),
        ringOffsets: importGraphBuffer(graph, 'rings', buffers.ringOffsets, 'uint32', 3),
        output: {
          indices: importGraphBuffer(graph, 'indices', buffers.indices, 'uint32', 12),
          valid: importGraphBuffer(graph, 'valid', buffers.valid, 'uint32', 2),
          triangleCount: importGraphBuffer(graph, 'counts', buffers.triangleCount, 'uint32', 2),
          triangleOffsets: importGraphBuffer(graph, 'offsets', buffers.triangleOffsets, 'uint32', 3)
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      // Pinned order of the existing Earcut walk: start at the last ring vertex, then advance.
      expect(await readUint32(buffers.indices, 12)).toEqual([2, 3, 0, 2, 0, 1, 6, 7, 4, 6, 4, 5]);
      expect(await readUint32(buffers.valid, 2)).toEqual([1, 1]);
      expect(await readUint32(buffers.triangleCount, 2)).toEqual([2, 2]);
      const triangleOffsets = await readUint32(buffers.triangleOffsets, 3);
      expect(triangleOffsets.map(offset => 3 * offset)).toEqual([0, 6, 12]);
    } finally {
      compiled.destroy();
    }
  } finally {
    for (const buffer of Object.values(buffers)) {
      buffer.destroy();
    }
  }
});
