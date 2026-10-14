// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared GPU harness of the snap-rounding overlay spec and the A/B spec: packs polygon operands in
// the GeoArrow layout, runs GPUPolygonOverlay (or the frozen legacy copy), reads the result back and
// converts it to lattice integers for exact comparison with the shapely fixtures.

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUPolygonOverlay,
  type GPUPolygonOverlayFillRule,
  type GPUPolygonOverlayOperation
} from '../../../src/gpu-spatial-analysis/polygon-overlay/index';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {LegacyGPUPolygonOverlay} from './ab/legacy-polygon-overlay';

/** `[feature][polygon][ring][vertex]` with open rings (no repeated closing vertex); first ring is the shell. */
export type OverlayOperand = [number, number][][][][];

/** Lattice frame of a fixture case. */
export type OverlayLattice = {origin: [number, number]; cell: number};

/** Capacities of one overlay run. */
export type OverlayRunCapacity = {
  intersections: number;
  nodedSegments: number;
  vertices: number;
  rings: number;
  boundary: number;
};

/** Generous capacities for the small fixtures. */
export const FIXTURE_CAPACITY: OverlayRunCapacity = {
  intersections: 4096,
  nodedSegments: 16384,
  vertices: 8192,
  rings: 512,
  boundary: 8192
};

/** One overlay run. */
export type OverlayRunProps = {
  implementation: 'current' | 'legacy';
  operation: GPUPolygonOverlayOperation;
  fillRule?: GPUPolygonOverlayFillRule;
  left: OverlayOperand;
  right?: OverlayOperand;
  vertexTolerance: number;
  capacity: OverlayRunCapacity;
};

/** Packs an operand into GeoArrow-layout arrays. */
export function packOperand(operand: OverlayOperand) {
  const positions: number[] = [];
  const featureOffsets = [0];
  const polygonOffsets = [0];
  const ringOffsets = [0];
  for (const feature of operand) {
    for (const polygon of feature) {
      for (const ring of polygon) {
        for (const [x, y] of ring) {
          positions.push(x, y);
        }
        ringOffsets.push(positions.length / 2);
      }
      polygonOffsets.push(ringOffsets.length - 1);
    }
    featureOffsets.push(polygonOffsets.length - 1);
  }
  return {
    positions: Float32Array.from(positions),
    featureOffsets: Uint32Array.from(featureOffsets),
    polygonOffsets: Uint32Array.from(polygonOffsets),
    ringOffsets: Uint32Array.from(ringOffsets)
  };
}

/** Compiled overlay that can be submitted repeatedly. */
export type PreparedOverlay = {
  compileMilliseconds: number;
  /** Submits once and waits for the published feature count. */
  submitOnce(): Promise<{milliseconds: number; count: number}>;
  /** Reads rings and status after a submit. */
  read(): Promise<OverlayReadout>;
  destroy(): void;
};

/** Result of one run read back to the CPU. */
export type OverlayReadout = {
  /** Output rings in world coordinates, with their polygon index and whether they are the shell. */
  rings: {polygon: number; isShell: boolean; vertices: [number, number][]}[];
  featureCount: number;
  requiredRingCount: number;
  overflow: number;
  candidateOverflow: number;
  boundaryCount: number;
  boundaryRequiredCount: number;
  boundaryOverflow: number;
  nodedSegmentRequiredCount: number;
};

/** Sizes a repeat run from complete GPU counts; uncertain/candidate-truncated plans are refused. */
export function planOverlayCapacity(
  capacity: OverlayRunCapacity,
  readout: OverlayReadout
): OverlayRunCapacity {
  if (readout.candidateOverflow || readout.boundaryOverflow) {
    const {rings, ...status} = readout;
    throw new Error(`Incomplete overlay plan: ${JSON.stringify(status)}`);
  }
  return {
    ...capacity,
    // The noder derives its incidence scratch from both this capacity and the crossing budget.
    nodedSegments: Math.max(1, readout.nodedSegmentRequiredCount || capacity.nodedSegments),
    boundary: Math.max(1, readout.boundaryRequiredCount),
    rings: Math.max(1, readout.requiredRingCount),
    // Ring assembly repeats the closing vertex: one per boundary edge plus one per ring.
    vertices: Math.max(1, readout.boundaryRequiredCount + readout.requiredRingCount)
  };
}

/** Builds and compiles the graph; compile time is returned separately from submit time. */
export function prepareOverlay(
  device: Device,
  props: OverlayRunProps,
  name: string
): PreparedOverlay {
  const graph = new GPUCommandGraph(device, {id: `overlay-${name}`});
  const buffers: Buffer[] = [];
  const input = <Format extends 'float32x2' | 'uint32'>(
    suffix: string,
    values: Float32Array | Uint32Array,
    format: Format
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      `${name}-${suffix}`,
      buffer,
      format,
      format === 'uint32' ? values.length : values.length / 2
    );
  };
  const output = <Format extends 'float32x2' | 'float32x4' | 'uint32'>(
    suffix: string,
    format: Format,
    length: number
  ) => {
    const components = format === 'float32x4' ? 4 : format === 'float32x2' ? 2 : 1;
    const buffer = createOutputBuffer(device, length * components);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, `${name}-${suffix}`, buffer, format, length)};
  };
  const polygons = (prefix: string, operand: OverlayOperand) => {
    const packed = packOperand(operand);
    return {
      kind: 'polygons' as const,
      positions: input(`${prefix}-positions`, packed.positions, 'float32x2'),
      featureOffsets: input(`${prefix}-feature-offsets`, packed.featureOffsets, 'uint32'),
      polygonOffsets: input(`${prefix}-polygon-offsets`, packed.polygonOffsets, 'uint32'),
      ringOffsets: input(`${prefix}-ring-offsets`, packed.ringOffsets, 'uint32')
    };
  };
  const {capacity} = props;
  const positions = output('positions', 'float32x2', capacity.vertices);
  const featureOffsets = output('feature-offsets', 'uint32', capacity.rings + 1);
  const polygonOffsets = output('polygon-offsets', 'uint32', capacity.rings + 1);
  const ringOffsets = output('ring-offsets', 'uint32', capacity.rings + 1);
  const sourceIds = output('source-ids', 'uint32', capacity.rings);
  const scalar = (suffix: string) => output(suffix, 'uint32', 1);
  const count = scalar('count');
  const requiredCount = scalar('required-count');
  const overflow = scalar('overflow');
  const candidateOverflow = scalar('candidate-overflow');
  const nodedSegmentRequiredCount = scalar('noded-segment-required-count');
  const boundaryCount = scalar('boundary-count');
  const boundaryRequired = scalar('boundary-required');
  const boundaryOverflow = scalar('boundary-overflow');
  const boundaryCandidateOverflow = scalar('boundary-candidate-overflow');
  const overlayProps = {
    left: polygons('left', props.left),
    right: props.right && polygons('right', props.right),
    operation: props.operation,
    fillRule: props.fillRule,
    capacity: {intersections: capacity.intersections, nodedSegments: capacity.nodedSegments},
    vertexTolerance: props.vertexTolerance,
    output: {
      geometry: {
        kind: 'polygons' as const,
        positions: positions.view,
        featureOffsets: featureOffsets.view,
        polygonOffsets: polygonOffsets.view,
        ringOffsets: ringOffsets.view,
        sourceIds: sourceIds.view
      },
      status: {
        count: count.view,
        requiredCount: requiredCount.view,
        overflow: overflow.view,
        candidateOverflow: candidateOverflow.view,
        ...(props.implementation === 'current'
          ? {nodedSegmentRequiredCount: nodedSegmentRequiredCount.view}
          : {})
      },
      boundary: {
        endpoints: output('boundary', 'float32x4', capacity.boundary).view,
        operandIds: output('boundary-operands', 'uint32', capacity.boundary).view,
        sourceFeatureIds: output('boundary-features', 'uint32', capacity.boundary).view,
        status: {
          count: boundaryCount.view,
          requiredCount: boundaryRequired.view,
          overflow: boundaryOverflow.view,
          candidateOverflow: boundaryCandidateOverflow.view
        }
      }
    }
  };
  graph.add(
    props.implementation === 'legacy'
      ? new LegacyGPUPolygonOverlay(overlayProps)
      : new GPUPolygonOverlay(overlayProps)
  );
  const compileStart = performance.now();
  const compiled = graph.compile();
  const compileMilliseconds = performance.now() - compileStart;
  return {
    compileMilliseconds,
    async submitOnce() {
      const start = performance.now();
      submitGraph(device, compiled, undefined);
      const featureCount = (await readUint32(count.buffer, 1))[0];
      return {milliseconds: performance.now() - start, count: featureCount};
    },
    async read() {
      const featureCount = (await readUint32(count.buffer, 1))[0];
      const scalarValue = async (target: {buffer: Buffer}) =>
        (await readUint32(target.buffer, 1))[0];
      const readout: OverlayReadout = {
        rings: [],
        featureCount,
        requiredRingCount: await scalarValue(requiredCount),
        overflow: await scalarValue(overflow),
        candidateOverflow: await scalarValue(candidateOverflow),
        boundaryCount: await scalarValue(boundaryCount),
        boundaryRequiredCount: await scalarValue(boundaryRequired),
        boundaryOverflow: await scalarValue(boundaryOverflow),
        nodedSegmentRequiredCount: await scalarValue(nodedSegmentRequiredCount)
      };
      if (featureCount === 0) {
        return readout;
      }
      const features = await readUint32(featureOffsets.buffer, featureCount + 1);
      const polygonCount = features[featureCount];
      const polygonRows = await readUint32(polygonOffsets.buffer, polygonCount + 1);
      const ringCount = polygonRows[polygonCount];
      const ringRows = await readUint32(ringOffsets.buffer, ringCount + 1);
      const coordinates = await readFloat32(positions.buffer, ringRows[ringCount] * 2);
      for (let polygon = 0; polygon < polygonCount; polygon++) {
        for (let ring = polygonRows[polygon]; ring < polygonRows[polygon + 1]; ring++) {
          const vertices: [number, number][] = [];
          for (let row = ringRows[ring]; row < ringRows[ring + 1]; row++) {
            vertices.push([coordinates[row * 2], coordinates[row * 2 + 1]]);
          }
          readout.rings.push({polygon, isShell: ring === polygonRows[polygon], vertices});
        }
      }
      return readout;
    },
    destroy() {
      compiled.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

/** Runs once and returns the readout. */
export async function runOverlay(
  device: Device,
  props: OverlayRunProps,
  name: string
): Promise<OverlayReadout> {
  const prepared = prepareOverlay(device, props, name);
  await prepared.submitOnce();
  const readout = await prepared.read();
  prepared.destroy();
  return readout;
}

// ---------------------------------------------------------------- lattice conversion

const signedArea2 = (ring: [number, number][]) =>
  ring.reduce((sum, [x, y], i) => {
    const [nextX, nextY] = ring[(i + 1) % ring.length];
    return sum + (x * nextY - nextX * y);
  }, 0);

/** Converts a world coordinate to a lattice integer; throws when it is not within 1e-3 of one. */
export function toLatticeInteger(value: number, origin: number, cell: number): number {
  const scaled = (value - origin) / cell;
  const rounded = Math.round(scaled);
  if (Math.abs(scaled - rounded) > 1e-3) {
    throw new Error(`coordinate ${value} is off the lattice (${scaled})`);
  }
  return rounded;
}

/** Removes vertices strictly inside a straight run (a doubling back vertex stays). */
export function removeCollinearVertices(ring: [number, number][]): [number, number][] {
  let result = ring;
  for (let changed = true; changed && result.length > 3; ) {
    changed = false;
    for (let i = 0; i < result.length; i++) {
      const a = result[(i + result.length - 1) % result.length];
      const b = result[i];
      const c = result[(i + 1) % result.length];
      const cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
      const dot = (b[0] - a[0]) * (c[0] - b[0]) + (b[1] - a[1]) * (c[1] - b[1]);
      if (cross === 0 && dot > 0) {
        result = [...result.slice(0, i), ...result.slice(i + 1)];
        changed = true;
        break;
      }
    }
  }
  return result;
}

const compareVertices = (a: [number, number], b: [number, number]) => a[0] - b[0] || a[1] - b[1];

/** Normalizes one ring like the fixture generator: shell CCW, hole CW, start at the smallest vertex. */
function normalizeRing(vertices: [number, number][], isShell: boolean): [number, number][] {
  const oriented = signedArea2(vertices) > 0 === isShell ? vertices : [...vertices].reverse();
  let start = 0;
  oriented.forEach((vertex, i) => {
    if (compareVertices(vertex, oriented[start]) < 0) {
      start = i;
    }
  });
  return [...oriented.slice(start), ...oriented.slice(0, start)];
}

/** Lattice summary of a readout, in the fixture's `expected` shape (rings, corners, area2, ringCount). */
export function summarizeReadout(readout: OverlayReadout, lattice: OverlayLattice) {
  const rings = readout.rings.map(ring =>
    normalizeRing(
      ring.vertices.map(
        ([x, y]) =>
          [
            toLatticeInteger(x, lattice.origin[0], lattice.cell),
            toLatticeInteger(y, lattice.origin[1], lattice.cell)
          ] as [number, number]
      ),
      ring.isShell
    )
  );
  // Canonical order independent of how touching rings tie: by the serialized ring.
  const order = (a: [number, number][], b: [number, number][]) => {
    const keyA = JSON.stringify(a);
    const keyB = JSON.stringify(b);
    return keyA < keyB ? -1 : keyA > keyB ? 1 : 0;
  };
  const corners = rings
    .map(ring => normalizeRing(removeCollinearVertices(ring), signedArea2(ring) > 0))
    .sort(order);
  return {
    rings: [...rings].sort(order),
    corners,
    // Shells are CCW and holes CW after normalization, so the plain sum is the signed area.
    area2: rings.reduce((sum, ring) => sum + signedArea2(ring), 0),
    ringCount: rings.length
  };
}
