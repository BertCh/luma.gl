// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {
  captureGPUCommandGraph,
  createGPUComputationCaptures,
  createGPUComputationRecord,
  getGPUComputationAdapter,
  type GPUAnalysisChoice,
  type GPUComputationAttributeValue,
  type GPUComputationAxis,
  type GPUComputationCapture,
  type GPUComputationCaptureGeometry,
  type GPUComputationKeySpace,
  type GPUComputationRelation
} from '@luma.gl/gpgpu';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/gpgpu/gpu-data';
import {
  createGPULocalFrame,
  type GPULocalFrame
} from '../../../analysis-core/contracts/local-frame';
import {
  createSpatialChoice,
  getSpatialOperatorEntry,
  SPATIAL_COMPUTATION_PACK_ID
} from '../../../gpu-spatial-analysis/computation-pack/index';
import type {GPUSpatialWeights} from '../../../gpu-spatial-analysis/spatial-weights/spatial-weights';
import {mergeGPUCoreFormulaRefsForGraphs} from '../../../computation-views/packs/gpu-core/formulas';
import type {LensScenarioResult} from '../../types';

/** One captured view of an S2 lens run and how the result view should draw it. */
export type S2CaptureSpec = {
  view: GraphDataView;
  role?: 'input' | 'output' | 'intermediate' | 'context';
  geometry?: GPUComputationCaptureGeometry;
  /** Key space the captured rows index, declared in {@link S2LensRunProps.keySpaces}. */
  keySpace?: string;
  /**
   * Free-form meaning of the capture. Faint basemaps use `role: 'context'` and the one answer uses
   * the capture's `highlight` (CARD-DESIGN §2.5).
   */
  meaning?: string;
  elementType?: GPUComputationCapture['elementType'];
};

/** Inputs of one recorded S2 lens run. */
export type S2LensRunProps = {
  /** Scenario id, also the record id. */
  scenarioId: string;
  /** Exported element name; its pack entry supplies the formula and choice templates. */
  element: string;
  /** Data file ids the run read, joined into the record's dataset version. */
  datasets: readonly string[];
  /** Deterministic seed, when the run samples. */
  seed?: number;
  /** Transient ids to retain for capture (they must be created with `Buffer.COPY_SRC`). */
  retain?: readonly string[];
  /** Key spaces the captures index; the first one is the datum axis. */
  keySpaces?: readonly GPUComputationKeySpace[];
  /** Relations between key spaces (CSR neighbours, point → polygon). */
  relations?: readonly GPUComputationRelation[];
  /** Values the run used for the element's pack choice templates, by template name. */
  choices?: Readonly<Record<string, GPUComputationAttributeValue>>;
  /**
   * Adds the uploaded inputs and the element (through the scenario's `runExample`) to the graph
   * and returns the views to capture after the run, by capture name.
   */
  build(
    graph: GPUCommandGraph<undefined>,
    uploads: S2Uploads
  ): Promise<Record<string, S2CaptureSpec>> | Record<string, S2CaptureSpec>;
};

/** Device buffers uploaded for one run; every buffer is readable so it can be captured. */
export type S2Uploads = {
  /** Uploads typed rows as an imported, capturable graph view. */
  view<Format extends GPUVectorFormat>(
    id: string,
    data: Float32Array | Uint32Array | Int32Array,
    format: Format,
    length?: number
  ): GraphDataView<Format>;
  /** Destroys `resource` (parameter buffers, helper outputs) after the run. */
  own<Resource extends {destroy(): void}>(resource: Resource): Resource;
};

/**
 * Runs one S2 lens scenario: builds a recording graph, compiles it, records `describe()`, executes
 * it once with an end-of-run snapshot of the requested views, and returns the record with its
 * captures sidecar. The capture snapshot adds one extra submission after the run (observer effect).
 */
export async function runS2Lens(
  device: Device,
  props: S2LensRunProps
): Promise<LensScenarioResult> {
  const graph = new GPUCommandGraph<undefined>(device, {
    id: props.scenarioId.replace(/\./g, '-'),
    recording: {retain: [...(props.retain ?? [])]}
  });
  const owned: {destroy(): void}[] = [];
  const uploads: S2Uploads = {
    own(resource) {
      owned.push(resource);
      return resource;
    },
    view(id, data, format, length) {
      const buffer = device.createBuffer({
        id,
        data,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      owned.push(buffer);
      const handle = graph.importBuffer(
        {id, byteLength: buffer.byteLength, usage: buffer.usage},
        buffer
      );
      const rowLength = format.endsWith('x2')
        ? 2
        : format.endsWith('x3')
          ? 3
          : format.endsWith('x4')
            ? 4
            : 1;
      return graph.createDataView(handle, {format, length: length ?? data.length / rowLength});
    }
  };
  try {
    const captureSpecs = await props.build(graph, uploads);
    const compiled = graph.compile();
    try {
      const entry = getSpatialOperatorEntry(props.element);
      const formula = entry && 'tex' in entry.formula ? entry.formula : undefined;
      const choices: GPUAnalysisChoice[] = (entry?.choices ?? []).flatMap(template =>
        props.choices && template.name in props.choices
          ? [createSpatialChoice(template, props.choices[template.name])]
          : []
      );
      const keySpaces = props.keySpaces ?? [];
      const axes: GPUComputationAxis[] = keySpaces.map((keySpace, index) => ({
        id: keySpace.id,
        label: keySpace.label,
        kind: index === 0 ? 'data' : 'structure',
        keySpace: keySpace.id,
        ...(keySpace.size !== undefined && {size: keySpace.size})
      }));
      const description = compiled.describe({domain: SPATIAL_COMPUTATION_PACK_ID});
      const record = createGPUComputationRecord({
        id: props.scenarioId,
        adapter: getGPUComputationAdapter(device),
        domains: [SPATIAL_COMPUTATION_PACK_ID],
        graphs: [description],
        axes: {axes, keySpaces, relations: props.relations ?? []},
        formulas: mergeGPUCoreFormulaRefsForGraphs([description], formula ? [formula] : []),
        choices,
        datasetVersion: props.datasets.join('+'),
        ...(props.seed !== undefined && {seed: props.seed})
      });
      const names = Object.keys(captureSpecs);
      const capture = await captureGPUCommandGraph(compiled, {
        encodeOptions: {parameters: undefined},
        views: Object.fromEntries(names.map(name => [name, captureSpecs[name].view])),
        elementTypes: Object.fromEntries(
          names.flatMap(name =>
            captureSpecs[name].elementType ? [[name, captureSpecs[name].elementType!]] : []
          )
        ),
        at: 'end',
        runId: record.runId
      });
      const snapshots = capture.captures.map(captured => {
        const spec = captureSpecs[captured.name];
        return {
          ...captured,
          ...(spec.role && {role: spec.role}),
          ...(spec.geometry && {geometry: spec.geometry}),
          ...(spec.keySpace && {keySpace: spec.keySpace}),
          ...(spec.meaning && {meaning: spec.meaning})
        };
      });
      return {
        ...record,
        captures: createGPUComputationCaptures({runId: record.runId, snapshots})
      };
    } finally {
      compiled.destroy();
    }
  } finally {
    for (const resource of owned) {
      resource.destroy();
    }
  }
}

/**
 * The local planar frame of the S2 Chicago North Side datasets: azimuthal equidistant (distance
 * operators) or Lambert equal-area (area operators) around the study-area centre, in metres.
 */
export function createS2ChicagoFrame(
  kind: 'azimuthal-equidistant' | 'lambert-equal-area' = 'azimuthal-equidistant'
): GPULocalFrame {
  return createGPULocalFrame({origin: [-87.69, 41.96], kind, datum: 'EPSG:4326'});
}

/** Projects interleaved longitude/latitude to frame metres and returns the padded extent. */
export function projectS2Positions(
  frame: GPULocalFrame,
  lngLats: ArrayLike<number>,
  padding = 0
): {positions: Float32Array; extent: [number, number, number, number]} {
  const positions = frame.projectArray(lngLats);
  const extent: [number, number, number, number] = [Infinity, Infinity, -Infinity, -Infinity];
  for (let index = 0; index < positions.length; index += 2) {
    extent[0] = Math.min(extent[0], positions[index]);
    extent[1] = Math.min(extent[1], positions[index + 1]);
    extent[2] = Math.max(extent[2], positions[index]);
    extent[3] = Math.max(extent[3], positions[index + 1]);
  }
  return {
    positions,
    extent: [extent[0] - padding, extent[1] - padding, extent[2] + padding, extent[3] + padding]
  };
}

/** Rounds an extent outwards to whole cells of `cellSize`. */
export function snapS2Extent(
  extent: readonly [number, number, number, number],
  cellSize: number
): [number, number, number, number] {
  return [
    Math.floor(extent[0] / cellSize) * cellSize,
    Math.floor(extent[1] / cellSize) * cellSize,
    Math.ceil(extent[2] / cellSize) * cellSize,
    Math.ceil(extent[3] / cellSize) * cellSize
  ];
}

/** Closes the open rings in S2 polygon datasets for both uploaded geometry and result drawing. */
export function closeS2Rings(
  positions: Float32Array,
  ringOffsets: readonly number[]
): {
  positions: Float32Array;
  ringOffsets: Uint32Array;
} {
  const closedPositions: number[] = [];
  const closedOffsets = [0];
  for (let ring = 0; ring + 1 < ringOffsets.length; ring++) {
    const first = ringOffsets[ring];
    const end = ringOffsets[ring + 1];
    for (let vertex = first; vertex < end; vertex++) {
      closedPositions.push(positions[2 * vertex], positions[2 * vertex + 1]);
    }
    if (
      end > first &&
      (positions[2 * first] !== positions[2 * (end - 1)] ||
        positions[2 * first + 1] !== positions[2 * (end - 1) + 1])
    ) {
      closedPositions.push(positions[2 * first], positions[2 * first + 1]);
    }
    closedOffsets.push(closedPositions.length / 2);
  }
  return {
    positions: Float32Array.from(closedPositions),
    ringOffsets: Uint32Array.from(closedOffsets)
  };
}

/** Upload full polygon topology plus closed exterior rings for S2 join figures (holes remain in computation). */
export function uploadS2JoinPolygons(
  uploads: S2Uploads,
  frame: GPULocalFrame,
  name: string,
  data: {
    vertices: number[];
    featureOffsets: number[];
    ringOffsets: number[];
  }
) {
  const projected = projectS2Positions(frame, data.vertices).positions;
  const positions = uploads.view(`${name}-positions`, projected, 'float32x2');
  const featureOffsets = uploads.view(
    `${name}-feature-offsets`,
    Uint32Array.from(data.featureOffsets, (_, index) => index),
    'uint32'
  );
  const polygonOffsets = uploads.view(
    `${name}-polygon-offsets`,
    new Uint32Array(data.featureOffsets),
    'uint32'
  );
  const ringOffsets = uploads.view(
    `${name}-ring-offsets`,
    new Uint32Array(data.ringOffsets),
    'uint32'
  );
  const exteriorPositions: number[] = [];
  const exteriorOffsets = [0];
  for (let feature = 0; feature < data.featureOffsets.length - 1; feature++) {
    const ring = data.featureOffsets[feature];
    const start = data.ringOffsets[ring];
    const end = data.ringOffsets[ring + 1];
    exteriorPositions.push(
      ...projected.slice(start * 2, end * 2),
      projected[start * 2],
      projected[start * 2 + 1]
    );
    exteriorOffsets.push(exteriorPositions.length / 2);
  }
  const display = uploads.view(
    `${name}-exteriors`,
    new Float32Array(exteriorPositions),
    'float32x2'
  );
  const displayOffsets = uploads.view(
    `${name}-exterior-offsets`,
    new Uint32Array(exteriorOffsets),
    'uint32'
  );
  const keySpaces = [
    {id: name, label: name, size: data.featureOffsets.length - 1, indexSpace: '1d' as const},
    {
      id: `${name}-vertex`,
      label: 'computation vertex',
      size: positions.length,
      indexSpace: '1d' as const
    },
    ...[
      ['feature', featureOffsets.length],
      ['polygon', polygonOffsets.length],
      ['ring', ringOffsets.length],
      ['display', displayOffsets.length]
    ].map(([level, size]) => ({
      id: `${name}-${level}-offset`,
      label: `${level} offset`,
      size: Number(size),
      indexSpace: '1d' as const
    }))
  ];
  const captures: Record<string, S2CaptureSpec> = {
    [`${name}Vertices`]: {view: positions, role: 'input', keySpace: `${name}-vertex`},
    [`${name}FeatureOffsets`]: {
      view: featureOffsets,
      role: 'input',
      keySpace: `${name}-feature-offset`
    },
    [`${name}PolygonOffsets`]: {
      view: polygonOffsets,
      role: 'input',
      keySpace: `${name}-polygon-offset`
    },
    [`${name}RingOffsets`]: {view: ringOffsets, role: 'input', keySpace: `${name}-ring-offset`},
    [`${name}DisplayOffsets`]: {
      view: displayOffsets,
      role: 'input',
      keySpace: `${name}-display-offset`
    },
    [name]: {
      view: display,
      role: 'input',
      keySpace: name,
      geometry: {
        kind: 'polygons',
        offsetsCapture: `${name}DisplayOffsets`,
        coordinateModel: 'projected',
        unit: 'm'
      }
    }
  };
  return {
    geometry: {kind: 'polygons' as const, positions, featureOffsets, polygonOffsets, ringOffsets},
    display,
    keySpaces,
    captures
  };
}

/** Midwest county table fields shared by the dynamics scenarios (`s2-midwest-counties.json`). */
export type S2CountyTable = {
  state: readonly string[];
  unemploymentRate: {years: readonly number[]; values: readonly (readonly number[])[]};
  queen: {offsets: readonly number[]; neighbors: readonly number[]};
};

/** Flattens a county-by-period table into the period-major column (`t * counties + i`) of the dynamics contributors. */
export function getS2PeriodMajorColumn(values: readonly (readonly number[])[]): Float32Array {
  const counties = values.length;
  const periods = values[0].length;
  const column = new Float32Array(counties * periods);
  for (let period = 0; period < periods; period++) {
    for (let county = 0; county < counties; county++) {
      column[period * counties + county] = values[county][period];
    }
  }
  return column;
}

/**
 * Uploads the queen contiguity of the county table as row-standardised weights (`w_ij = 1 / deg(i)`).
 * The queen graph was computed from the full source geometry; the weights are declared `'row'`.
 */
export function uploadS2QueenWeights(
  uploads: S2Uploads,
  queen: S2CountyTable['queen']
): GPUSpatialWeights {
  const rows = queen.offsets.length - 1;
  const weights = new Float32Array(queen.neighbors.length);
  for (let row = 0; row < rows; row++) {
    const degree = queen.offsets[row + 1] - queen.offsets[row];
    weights.fill(1 / degree, queen.offsets[row], queen.offsets[row + 1]);
  }
  return {
    offsets: uploads.view('queenOffsets', Uint32Array.from(queen.offsets), 'uint32'),
    neighbors: uploads.view('queenNeighbors', Uint32Array.from(queen.neighbors), 'uint32'),
    weights: uploads.view('queenWeights', weights, 'float32'),
    information: {
      rowCount: rows,
      transform: 'row',
      symmetric: true,
      provenance: {source: 'upload', criterion: 'queen'}
    }
  };
}

/**
 * Uploads the county display polygons (lon/lat, rings closed for drawing) and returns the context
 * captures, their key spaces and `paint`, which makes a coloured output capture of the same rings.
 */
export function uploadS2CountyShapes(
  uploads: S2Uploads,
  polygons: {vertices: readonly number[]; ringOffsets: readonly number[]}
) {
  const closed = closeS2Rings(Float32Array.from(polygons.vertices), polygons.ringOffsets);
  const vertices = uploads.view('countyVertices', closed.positions, 'float32x2');
  const offsets = uploads.view('countyOffsets', closed.ringOffsets, 'uint32');
  const keySpaces: GPUComputationKeySpace[] = [
    {id: 'county-vertex', label: 'ring vertex', size: vertices.length, indexSpace: '1d'},
    {id: 'county-offset', label: 'ring offset', size: offsets.length, indexSpace: '1d'}
  ];
  const geometry = (valuesCapture?: string): GPUComputationCaptureGeometry => ({
    kind: 'polygons',
    offsetsCapture: 'countyOffsets',
    coordinateModel: 'geographic',
    ...(valuesCapture && {valuesCapture})
  });
  const captures: Record<string, S2CaptureSpec> = {
    countyVertices: {view: vertices, role: 'input', keySpace: 'county-vertex'},
    countyOffsets: {view: offsets, role: 'input', keySpace: 'county-offset'},
    counties: {
      view: vertices,
      role: 'context',
      keySpace: 'county',
      geometry: geometry()
    }
  };
  return {
    vertices,
    keySpaces,
    captures,
    /** An output capture of the county rings coloured by the one-value-per-county capture `valuesCapture`. */
    paint: (valuesCapture: string): S2CaptureSpec => ({
      view: vertices,
      role: 'output',
      keySpace: 'county',
      geometry: geometry(valuesCapture)
    })
  };
}

/**
 * Area-weighted centroids of single-ring polygons in frame metres (open or closed rings; the
 * vertex mean for a ring with no area). Returns interleaved `x, y` per ring.
 */
export function computeS2RingCentroids(
  positions: Float32Array,
  ringOffsets: ArrayLike<number>
): Float32Array {
  const centroids = new Float32Array((ringOffsets.length - 1) * 2);
  for (let ring = 0; ring + 1 < ringOffsets.length; ring++) {
    const first = ringOffsets[ring];
    const end = ringOffsets[ring + 1];
    let signedArea = 0;
    let centerX = 0;
    let centerY = 0;
    for (let vertex = first; vertex < end; vertex++) {
      const next = vertex + 1 < end ? vertex + 1 : first;
      const x0 = positions[2 * vertex];
      const y0 = positions[2 * vertex + 1];
      const x1 = positions[2 * next];
      const y1 = positions[2 * next + 1];
      const cross = x0 * y1 - x1 * y0;
      signedArea += cross;
      centerX += (x0 + x1) * cross;
      centerY += (y0 + y1) * cross;
    }
    if (Math.abs(signedArea) > 1e-9) {
      centroids[2 * ring] = centerX / (3 * signedArea);
      centroids[2 * ring + 1] = centerY / (3 * signedArea);
    } else {
      let sumX = 0;
      let sumY = 0;
      for (let vertex = first; vertex < end; vertex++) {
        sumX += positions[2 * vertex];
        sumY += positions[2 * vertex + 1];
      }
      centroids[2 * ring] = sumX / Math.max(end - first, 1);
      centroids[2 * ring + 1] = sumY / Math.max(end - first, 1);
    }
  }
  return centroids;
}

/**
 * Chooses the coarsest 10 m multiple of cell size that fits `extent` into at most `maxCells`
 * cells per side, and snaps the extent outwards to whole cells.
 */
export function createS2RasterGrid(
  extent: readonly [number, number, number, number],
  maxCells = 128
): {
  cellSize: number;
  width: number;
  height: number;
  extent: [number, number, number, number];
} {
  let cellSize =
    Math.ceil(Math.max(extent[2] - extent[0], extent[3] - extent[1]) / maxCells / 10) * 10;
  for (;;) {
    const snapped = snapS2Extent(extent, cellSize);
    const width = Math.round((snapped[2] - snapped[0]) / cellSize);
    const height = Math.round((snapped[3] - snapped[1]) / cellSize);
    if (width <= maxCells && height <= maxCells) {
      return {cellSize, width, height, extent: snapped};
    }
    cellSize += 10;
  }
}

/**
 * Upper bound on the (edge, raster row) crossings `GPUPolygonRasterization` records for rings of
 * `positions`: every edge crosses at most `|dy| / cellSize + 1` row centre lines.
 */
export function estimateS2RasterCrossings(
  positions: Float32Array,
  ringOffsets: ArrayLike<number>,
  cellSize: number
): number {
  let crossings = 0;
  for (let ring = 0; ring + 1 < ringOffsets.length; ring++) {
    const first = ringOffsets[ring];
    const end = ringOffsets[ring + 1];
    for (let vertex = first; vertex < end; vertex++) {
      const next = vertex + 1 < end ? vertex + 1 : first;
      crossings += Math.abs(positions[2 * vertex + 1] - positions[2 * next + 1]) / cellSize + 1;
    }
  }
  return Math.ceil(crossings) + 16;
}
