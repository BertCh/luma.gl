// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUGridCellCount,
  getGPUGridGeneratorParameterValues,
  getGPUGridVerticesPerCell,
  getGPULineDensityParameterValues,
  GPUGridGenerator,
  GPULineDensity,
  GPULineLengthPerPolygon,
  GPU_GRID_GENERATOR_PARAMETER_LENGTH,
  GPU_LINE_DENSITY_PARAMETER_LENGTH,
  type GPUGridType
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {importGraphBuffer} from '../graph-buffers';
import {LocalMetricProjection} from '../spatial-analysis-data';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer
} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {
  formatCount,
  getViewportMetricBounds,
  SpatialAnalysisResources
} from '../spatial-analysis-resources';
import {
  createDistrictTiling,
  createMovedZipSet,
  getFeatureRowAt,
  rasterizeFeatureRows,
  type FeatureRowGrid,
  type LineDensityPolygonSet
} from './line-density-layers';
import {addKernelPass} from './mode-kernels';
import {SummaryReader} from './summary-reader';

/** Compile-time density lattice; the cell size per frame decides how much map it covers. */
const COLUMNS = 192;
const ROWS = 120;
const CELL_COUNT = COLUMNS * ROWS;
/** Compile-time lattice of the grid generators. */
const OVERLAY_COLUMNS = 22;
const OVERLAY_ROWS = 13;
const GRID_TYPES: readonly GPUGridType[] = ['square', 'hex', 'triangle', 'point'];

/** Compile-time choropleth raster width; the set's bounding box fixes the rows. */
const CHOROPLETH_COLUMNS = 360;
const NO_DATA = Number.NaN;

type OverlayChoice = GPUGridType | 'none';
type ViewChoice = 'raster' | 'polygons';
type MetricChoice = 'length' | 'per-area';
type PolygonSetChoice = 'districts' | 'zips';

/** Everything one polygon set owns: buffers, compiled graph, readback and the last summary. */
type PolygonSetState = {
  set: LineDensityPolygonSet;
  grid: FeatureRowGrid;
  featureCount: number;
  outline: ReturnType<SpatialAnalysisResources['createBuffer']>;
  cellRows: ReturnType<SpatialAnalysisResources['createBuffer']>;
  lengths: ReturnType<SpatialAnalysisResources['createBuffer']>;
  perArea: ReturnType<SpatialAnalysisResources['createBuffer']>;
  compiled: ReturnType<GPUCommandGraph<void>['compile']>;
  reader: SummaryReader;
  /** True when the set partitions the road extent, so lengths must sum to the network length. */
  partition: boolean;
  dirty: boolean;
  lengthValues: Float32Array;
  perAreaValues: Float32Array;
  countValues: Uint32Array;
  lengthMaximum: number;
  perAreaMaximum: number;
  inside: number;
  overflow: boolean;
};

/**
 * Road density of New York on a grid that follows the camera. `GPULineDensity` clips every street
 * segment to the cells it crosses and sums the length per cell; the cell size is a slider and the
 * grid origin snaps to the cell size, so panning and the slider are buffer writes. A second lattice
 * from `GPUGridGenerator` (square, hexagon, triangle or point, all compiled once) overlays the view.
 */
export const lineDensityMode: SpatialAnalysisModeDefinition = {
  id: 'line-density',
  title: 'Line density',
  contributors: ['GPULineDensity', 'GPULineLengthPerPolygon', 'GPUGridGenerator'],
  description:
    'Street length per cell on a grid that follows the map, or street length clipped into each ' +
    'polygon as a choropleth (per polygon or per area). Switch the view, polygon set and metric, ' +
    'change the cell size, pan and zoom, hover a polygon; nothing recompiles.',
  initialViewState: {longitude: -73.985, latitude: 40.735, zoom: 12.4},

  async create(context) {
    const [roads, zips] = await Promise.all([
      context.data.getNewYorkRoads(),
      context.data.getSanFranciscoZipCodes()
    ]);
    context.signal.throwIfAborted();
    const {device} = context;
    const projection = new LocalMetricProjection(roads.origin);
    const resources = new SpatialAnalysisResources(device, 'line-density');
    const segmentCount = roads.segments.length / 4;
    const vertexCount = segmentCount * 2;
    const coordinateOrigin: [number, number, number] = [roads.origin[0], roads.origin[1], 0];

    // Every segment is its own two-vertex path.
    const pathOffsets = new Uint32Array(segmentCount + 1);
    let networkLength = 0;
    let roadMinX = Infinity;
    let roadMinY = Infinity;
    let roadMaxX = -Infinity;
    let roadMaxY = -Infinity;
    for (let segment = 0; segment < segmentCount; segment++) {
      pathOffsets[segment] = segment * 2;
      for (const offset of [0, 2]) {
        roadMinX = Math.min(roadMinX, roads.segments[segment * 4 + offset]);
        roadMaxX = Math.max(roadMaxX, roads.segments[segment * 4 + offset]);
        roadMinY = Math.min(roadMinY, roads.segments[segment * 4 + offset + 1]);
        roadMaxY = Math.max(roadMaxY, roads.segments[segment * 4 + offset + 1]);
      }
      networkLength += Math.hypot(
        roads.segments[segment * 4 + 2] - roads.segments[segment * 4],
        roads.segments[segment * 4 + 3] - roads.segments[segment * 4 + 1]
      );
    }
    pathOffsets[segmentCount] = vertexCount;
    const maximumRecords = Math.max(1024, 4 * vertexCount);

    let cellSize = 100;
    let overlay: OverlayChoice = 'hex';
    let showRoads = true;
    let viewChoice: ViewChoice = 'raster';
    let metric: MetricChoice = 'length';
    let polygonSetChoice: PolygonSetChoice = 'districts';
    let densityMaximum = 0.01;
    let lastSignature = '';
    let dirty = true;
    let viewWidth = 1;

    const segmentsBuffer = resources.createBuffer('segments', roads.segments);
    const pathOffsetsBuffer = resources.createBuffer('path-offsets', pathOffsets);
    const densityParameters = resources.createParameterBuffer(
      'density-parameters',
      'float32',
      GPU_LINE_DENSITY_PARAMETER_LENGTH
    );
    const gridBounds = resources.createParameterBuffer('grid-bounds', 'float32', 4);
    const overlayParameters = resources.createParameterBuffer(
      'overlay-parameters',
      'float32',
      GPU_GRID_GENERATOR_PARAMETER_LENGTH
    );
    const lengths = resources.createBuffer('lengths', CELL_COUNT * 4);
    const densities = resources.createBuffer('densities', CELL_COUNT * 4);
    const overflow = resources.createBuffer('overflow', 4);
    const totalRecords = resources.createBuffer('total-records', 4);

    // Density graph.
    const densityGraph = new GPUCommandGraph<void>(device, {id: 'line-density'});
    densityGraph.add(
      new GPULineDensity({
        id: 'line-density',
        positions: importGraphBuffer(
          densityGraph,
          'positions',
          segmentsBuffer,
          'float32x2',
          vertexCount
        ),
        pathOffsets: importGraphBuffer(
          densityGraph,
          'path-offsets',
          pathOffsetsBuffer,
          'uint32',
          segmentCount + 1
        ),
        columns: COLUMNS,
        rows: ROWS,
        maximumRecords,
        parameters: densityParameters.importToGraph(densityGraph),
        output: {
          lengths: importGraphBuffer(densityGraph, 'lengths', lengths, 'float32', CELL_COUNT),
          densities: importGraphBuffer(densityGraph, 'densities', densities, 'float32', CELL_COUNT),
          overflow: importGraphBuffer(densityGraph, 'overflow', overflow, 'uint32', 1),
          totalRecords: importGraphBuffer(densityGraph, 'total-records', totalRecords, 'uint32', 1)
        }
      })
    );
    const compiledDensity = resources.track(densityGraph.compile());

    // Overlay graph: one generator per lattice type, plus edge passes that turn cell vertices
    // into drawable segments.
    const overlayGraph = new GPUCommandGraph<void>(device, {id: 'grid-generators'});
    const overlayBuffers = new Map<
      GPUGridType,
      {
        vertices: ReturnType<typeof resources.createBuffer>;
        edges: ReturnType<typeof resources.createBuffer>;
        edgeCount: number;
      }
    >();
    {
      const parameters = overlayParameters.importToGraph(overlayGraph);
      for (const gridType of GRID_TYPES) {
        const cells = getGPUGridCellCount(gridType, OVERLAY_COLUMNS, OVERLAY_ROWS);
        const verticesPerCell = getGPUGridVerticesPerCell(gridType);
        const rowCount = gridType === 'point' ? cells : cells * verticesPerCell;
        const vertices = resources.createBuffer(`${gridType}-vertices`, rowCount * 8);
        const view: GraphDataView<'float32x2'> = importGraphBuffer(
          overlayGraph,
          `${gridType}-vertices`,
          vertices,
          'float32x2',
          rowCount
        );
        overlayGraph.add(
          new GPUGridGenerator({
            id: `grid-${gridType}`,
            gridType,
            columns: OVERLAY_COLUMNS,
            rows: OVERLAY_ROWS,
            parameters,
            output: gridType === 'point' ? {centers: view} : {positions: view}
          })
        );
        const edgeCount = gridType === 'point' ? 0 : cells * verticesPerCell;
        const edges = resources.createBuffer(`${gridType}-edges`, Math.max(1, edgeCount) * 16);
        if (edgeCount > 0) {
          addKernelPass(overlayGraph, {
            id: `${gridType}-edges`,
            invocationCount: edgeCount,
            bindings: [
              {name: 'vertices', view, type: 'f32', access: 'read'},
              {
                name: 'edges',
                view: importGraphBuffer(
                  overlayGraph,
                  `${gridType}-edges`,
                  edges,
                  'float32',
                  edgeCount * 4
                ),
                type: 'f32',
                access: 'read_write'
              }
            ],
            declarations: `const PER_CELL: u32 = ${verticesPerCell}u;`,
            body: /* wgsl */ `
  let cell = index / PER_CELL;
  let corner = index % PER_CELL;
  let next = cell * PER_CELL + (corner + 1u) % PER_CELL;
  edges[edgesOffset + index * 4u] = vertices[verticesOffset + index * 2u];
  edges[edgesOffset + index * 4u + 1u] = vertices[verticesOffset + index * 2u + 1u];
  edges[edgesOffset + index * 4u + 2u] = vertices[verticesOffset + next * 2u];
  edges[edgesOffset + index * 4u + 3u] = vertices[verticesOffset + next * 2u + 1u];`
          });
        }
        overlayBuffers.set(gridType, {vertices, edges, edgeCount});
      }
    }
    const compiledOverlay = resources.track(overlayGraph.compile());

    // Polygon graphs: one compiled `GPULineLengthPerPolygon` per polygon set, both compiled now.
    // Switching the set or the metric only changes which buffers the layers read.
    const padding = 25;
    const polygonSets: Record<PolygonSetChoice, PolygonSetState> = {
      districts: createPolygonSet(
        createDistrictTiling(
          [roadMinX - padding, roadMinY - padding, roadMaxX + padding, roadMaxY + padding],
          14,
          11
        ),
        true
      ),
      zips: createPolygonSet(
        createMovedZipSet(
          zips,
          [(roadMinX + roadMaxX) / 2, (roadMinY + roadMaxY) / 2],
          zips.source === 'remote'
            ? 'ZIP code shapes (San Francisco, moved over the roads; gaps)'
            : 'Synthetic ZIP-like shapes (moved over the roads; gaps)'
        ),
        false
      )
    };
    function createPolygonSet(set: LineDensityPolygonSet, partition: boolean): PolygonSetState {
      const key = partition ? 'districts' : 'zips';
      const featureCount = set.featureOffsets.length - 1;
      const grid = rasterizeFeatureRows(set, CHOROPLETH_COLUMNS);
      // One extra NaN row: choropleth cells outside every polygon index it.
      const noData = new Float32Array(featureCount + 1).fill(NO_DATA);
      const lengths = resources.createBuffer(`${key}-lengths`, noData);
      const perArea = resources.createBuffer(`${key}-per-area`, noData);
      const areas = resources.createBuffer(`${key}-areas`, set.areas);
      const counts = resources.createBuffer(`${key}-counts`, featureCount * 4);
      const polygonOverflow = resources.createBuffer(`${key}-overflow`, 4);
      const graph = new GPUCommandGraph<void>(device, {id: `line-length-${key}`});
      const lengthView = importGraphBuffer(graph, 'lengths', lengths, 'float32', featureCount);
      const featureOffsetsBuffer = resources.createBuffer(
        `${key}-feature-offsets`,
        set.featureOffsets
      );
      const polygonOffsetsBuffer = resources.createBuffer(
        `${key}-polygon-offsets`,
        set.polygonOffsets
      );
      const ringOffsetsBuffer = resources.createBuffer(`${key}-ring-offsets`, set.ringOffsets);
      const polygonPositionsBuffer = resources.createBuffer(`${key}-positions`, set.positions);
      graph.add(
        new GPULineLengthPerPolygon({
          id: `line-length-${key}`,
          positions: importGraphBuffer(
            graph,
            'positions',
            segmentsBuffer,
            'float32x2',
            vertexCount
          ),
          pathOffsets: importGraphBuffer(
            graph,
            'path-offsets',
            pathOffsetsBuffer,
            'uint32',
            segmentCount + 1
          ),
          polygons: {
            kind: 'polygons',
            positions: importGraphBuffer(
              graph,
              'polygon-positions',
              polygonPositionsBuffer,
              'float32x2',
              set.positions.length / 2
            ),
            featureOffsets: importGraphBuffer(
              graph,
              'feature-offsets',
              featureOffsetsBuffer,
              'uint32',
              set.featureOffsets.length
            ),
            polygonOffsets: importGraphBuffer(
              graph,
              'polygon-offsets',
              polygonOffsetsBuffer,
              'uint32',
              set.polygonOffsets.length
            ),
            ringOffsets: importGraphBuffer(
              graph,
              'ring-offsets',
              ringOffsetsBuffer,
              'uint32',
              set.ringOffsets.length
            )
          },
          maximumCandidatePairs: Math.max(1024, 3 * vertexCount),
          output: {
            lengths: lengthView,
            segmentCounts: importGraphBuffer(graph, 'counts', counts, 'uint32', featureCount),
            overflow: importGraphBuffer(graph, 'overflow', polygonOverflow, 'uint32', 1)
          }
        })
      );
      // Length per unit area in km per km2 (m / m2 * 1000).
      addKernelPass(graph, {
        id: `${key}-per-area`,
        invocationCount: featureCount,
        bindings: [
          {name: 'lengths', view: lengthView, type: 'f32', access: 'read'},
          {
            name: 'areas',
            view: importGraphBuffer(graph, 'areas', areas, 'float32', featureCount),
            type: 'f32',
            access: 'read'
          },
          {
            name: 'perArea',
            view: importGraphBuffer(graph, 'per-area', perArea, 'float32', featureCount),
            type: 'f32',
            access: 'read_write'
          }
        ],
        body: `perArea[perAreaOffset + index] = lengths[lengthsOffset + index] / max(areas[areasOffset + index], 1.0) * 1000.0;`
      });
      const state: PolygonSetState = {
        set,
        grid,
        featureCount,
        outline: resources.createBuffer(`${key}-outline`, set.outlineSegments),
        cellRows: resources.createBuffer(`${key}-cell-rows`, grid.featureRows),
        lengths,
        perArea,
        compiled: resources.track(graph.compile()),
        partition,
        dirty: true,
        lengthValues: new Float32Array(featureCount),
        perAreaValues: new Float32Array(featureCount),
        countValues: new Uint32Array(featureCount),
        lengthMaximum: 1,
        perAreaMaximum: 1,
        inside: 0,
        overflow: false,
        reader: new SummaryReader(
          resources,
          key,
          [
            {buffer: lengths, size: featureCount * 4},
            {buffer: perArea, size: featureCount * 4},
            {buffer: counts, size: featureCount * 4},
            {buffer: polygonOverflow, size: 4}
          ],
          bytes => {
            state.lengthValues = new Float32Array(bytes.slice(0, featureCount * 4));
            state.perAreaValues = new Float32Array(bytes.slice(featureCount * 4, featureCount * 8));
            state.countValues = new Uint32Array(bytes.slice(featureCount * 8, featureCount * 12));
            state.overflow = new Uint32Array(bytes, featureCount * 12, 1)[0] !== 0;
            state.inside = 0;
            state.lengthMaximum = 1e-6;
            state.perAreaMaximum = 1e-6;
            for (let row = 0; row < featureCount; row++) {
              state.inside += state.lengthValues[row];
              state.lengthMaximum = Math.max(state.lengthMaximum, state.lengthValues[row]);
              state.perAreaMaximum = Math.max(state.perAreaMaximum, state.perAreaValues[row]);
            }
            updatePolygonReadouts();
            context.updateLayers();
          }
        )
      };
      return state;
    }

    // ---- Controls ----
    context.controls.addSelect<ViewChoice>({
      label: 'View (both graphs compiled once)',
      options: [
        {value: 'raster', label: 'Density raster (GPULineDensity)'},
        {value: 'polygons', label: 'Length per polygon (GPULineLengthPerPolygon)'}
      ],
      value: viewChoice,
      onChange: value => {
        viewChoice = value;
        updatePolygonReadouts();
        context.updateLayers();
      }
    });
    context.controls.addSelect<PolygonSetChoice>({
      label: 'Polygon set (polygon view)',
      options: [
        {value: 'districts', label: polygonSets.districts.set.label},
        {value: 'zips', label: polygonSets.zips.set.label}
      ],
      value: polygonSetChoice,
      onChange: value => {
        polygonSetChoice = value;
        updatePolygonReadouts();
        context.updateLayers();
      }
    });
    context.controls.addSelect<MetricChoice>({
      label: 'Choropleth value (polygon view)',
      options: [
        {value: 'length', label: 'Street length per polygon'},
        {value: 'per-area', label: 'Street length per unit area'}
      ],
      value: metric,
      onChange: value => {
        metric = value;
        context.updateLayers();
      }
    });
    context.controls.addSlider({
      label: 'Cell size (raster view, per-frame parameter)',
      min: 40,
      max: 400,
      step: 10,
      value: cellSize,
      format: value => `${value} m`,
      onChange: value => {
        cellSize = value;
        lastSignature = '';
      }
    });
    context.controls.addSelect<OverlayChoice>({
      label: 'Generated lattice overlay (GPUGridGenerator, all compiled once)',
      options: [
        {value: 'none', label: 'None'},
        {value: 'square', label: 'Square grid'},
        {value: 'hex', label: 'Hexagon grid'},
        {value: 'triangle', label: 'Triangle grid'},
        {value: 'point', label: 'Point grid'}
      ],
      value: overlay,
      onChange: value => {
        overlay = value;
        overlayReadout.setValue(value === 'none' ? 'none' : formatCount(overlayCellCount(value)));
        context.updateLayers();
      }
    });
    context.controls.addToggle({
      label: 'Show street segments',
      value: showRoads,
      onChange: value => {
        showRoads = value;
        context.updateLayers();
      }
    });
    const legend = {
      title: 'Raster view: street length per area (sqrt scale, to the 99th percentile)',
      gradient: {
        colors: [
          [0, 0, 4],
          [120, 28, 109],
          [237, 105, 37],
          [252, 255, 164]
        ] as const,
        minimumLabel: '0',
        maximumLabel: 'dense'
      }
    };
    context.controls.addLegend(legend);
    context.controls.addLegend({
      title: 'Polygon view: street length per polygon or per area (linear to the maximum)',
      gradient: {
        colors: [
          [68, 1, 84],
          [49, 104, 142],
          [53, 183, 121],
          [253, 231, 37]
        ] as const,
        minimumLabel: '0',
        maximumLabel: 'max'
      }
    });
    context.controls.addReadout('Street segments', formatCount(segmentCount));
    const gridReadout = context.controls.addReadout(
      'Density grid',
      `${COLUMNS} x ${ROWS} = ${formatCount(CELL_COUNT)} cells`
    );
    const coverageReadout = context.controls.addReadout('Grid covers the view width');
    const lengthReadout = context.controls.addReadout('Length in the grid');
    const maximumReadout = context.controls.addReadout('Densest cell (99th percentile)');
    const recordsReadout = context.controls.addReadout('Clipped records');
    const overflowReadout = context.controls.addReadout('Record overflow');
    const polygonCountReadout = context.controls.addReadout('Polygons');
    const insideReadout = context.controls.addReadout('Length inside polygons');
    const conservationReadout = context.controls.addReadout('Conservation check');
    const polygonMaximumReadout = context.controls.addReadout('Longest / densest polygon');
    const polygonOverflowReadout = context.controls.addReadout('Polygon overflow');
    const hoverReadout = context.controls.addReadout('Hovered polygon', 'hover the map');
    const overlayReadout = context.controls.addReadout(
      'Overlay cells',
      formatCount(getGPUGridCellCount('hex', OVERLAY_COLUMNS, OVERLAY_ROWS))
    );
    context.controls.addReadout('Data', roads.attribution);

    // ---- Readback ----
    const reader = new SummaryReader(
      resources,
      'line-density',
      [
        {buffer: lengths, size: CELL_COUNT * 4},
        {buffer: densities, size: CELL_COUNT * 4},
        {buffer: overflow, size: 4},
        {buffer: totalRecords, size: 4}
      ],
      bytes => {
        const floats = new Float32Array(bytes, 0, CELL_COUNT * 2);
        const words = new Uint32Array(bytes, CELL_COUNT * 8, 2);
        let total = 0;
        const positive: number[] = [];
        for (let cell = 0; cell < CELL_COUNT; cell++) {
          total += floats[cell];
          if (floats[CELL_COUNT + cell] > 0) positive.push(floats[CELL_COUNT + cell]);
        }
        positive.sort((left, right) => left - right);
        const percentile = positive.length
          ? positive[Math.min(positive.length - 1, Math.floor(positive.length * 0.99))]
          : 0.001;
        lengthReadout.setValue(
          `${(total / 1000).toFixed(0)} km of ${(networkLength / 1000).toFixed(0)} km (${(
            (100 * total) / networkLength
          ).toFixed(0)}%)`
        );
        maximumReadout.setValue(`${(percentile * 1000).toFixed(1)} m per 1000 m2`);
        recordsReadout.setValue(`${formatCount(words[1])} of ${formatCount(maximumRecords)}`);
        overflowReadout.setValue(words[0] ? 'yes' : 'no');
        if (Math.abs(percentile - densityMaximum) > densityMaximum * 0.1) {
          densityMaximum = percentile;
          context.updateLayers();
        }
      }
    );
    const readySets = new Set<PolygonSetChoice>();
    function updatePolygonReadouts() {
      const state = polygonSets[polygonSetChoice];
      polygonCountReadout.setValue(
        `${formatCount(state.featureCount)} (${state.partition ? 'partition of the road extent' : 'with gaps'})`
      );
      if (!readySets.has(polygonSetChoice)) {
        for (const readout of [insideReadout, conservationReadout, polygonMaximumReadout]) {
          readout.setValue('switch to the polygon view');
        }
        polygonOverflowReadout.setValue('-');
        return;
      }
      const percent = (100 * state.inside) / networkLength;
      insideReadout.setValue(
        `${(state.inside / 1000).toFixed(1)} km of ${(networkLength / 1000).toFixed(1)} km (${percent.toFixed(2)}%)`
      );
      conservationReadout.setValue(
        state.partition
          ? `${Math.abs(percent - 100) < 0.5 ? 'conserved' : 'MISMATCH'}: inside - total = ${(
              (state.inside - networkLength) / 1000
            ).toFixed(3)} km`
          : `${(networkLength / 1000 - state.inside / 1000).toFixed(1)} km of streets fall in the gaps`
      );
      polygonMaximumReadout.setValue(
        `${(state.lengthMaximum / 1000).toFixed(1)} km / ${state.perAreaMaximum.toFixed(1)} km per km2`
      );
      polygonOverflowReadout.setValue(state.overflow ? 'yes (lengths may be low)' : 'no');
    }
    updatePolygonReadouts();
    const overlayCellCount = (type: GPUGridType) =>
      getGPUGridCellCount(type, OVERLAY_COLUMNS, OVERLAY_ROWS);

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [
        compiledDensity,
        compiledOverlay,
        polygonSets.districts.compiled,
        polygonSets.zips.compiled
      ],
      encode(commandEncoder, frame) {
        const bounds = getViewportMetricBounds(frame.viewport, projection);
        viewWidth = bounds[2] - bounds[0];
        const minimumX = Math.floor(bounds[0] / cellSize) * cellSize;
        const minimumY = Math.floor(bounds[1] / cellSize) * cellSize;
        const overlayCellWidth = viewWidth / OVERLAY_COLUMNS;
        const overlayCellHeight = (bounds[3] - bounds[1]) / OVERLAY_ROWS;
        const signature = [minimumX, minimumY, cellSize, ...bounds.map(Math.round)].join(',');
        if (signature !== lastSignature || frame.frameIndex < 2) {
          lastSignature = signature;
          densityParameters.write(
            getGPULineDensityParameterValues({
              minX: minimumX,
              minY: minimumY,
              cellWidth: cellSize,
              cellHeight: cellSize
            })
          );
          gridBounds.write(
            Float32Array.of(
              minimumX,
              minimumY,
              minimumX + COLUMNS * cellSize,
              minimumY + ROWS * cellSize
            )
          );
          overlayParameters.write(
            getGPUGridGeneratorParameterValues({
              minX: bounds[0],
              minY: bounds[1],
              cellWidth: overlayCellWidth,
              cellHeight: overlayCellHeight
            })
          );
          dirty = true;
        }
        if (dirty) {
          compiledDensity.encode(commandEncoder, {parameters: undefined});
          compiledOverlay.encode(commandEncoder, {parameters: undefined});
          reader.request(commandEncoder);
          dirty = false;
          gridReadout.setValue(
            `${COLUMNS} x ${ROWS} cells of ${cellSize} m (${((COLUMNS * cellSize) / 1000).toFixed(1)} x ${((ROWS * cellSize) / 1000).toFixed(1)} km)`
          );
          coverageReadout.setValue(`${Math.round((100 * COLUMNS * cellSize) / viewWidth)}%`);
        }
        reader.flush(commandEncoder);
        for (const state of Object.values(polygonSets)) state.reader.flush(commandEncoder);
        const active = polygonSets[polygonSetChoice];
        if (viewChoice === 'polygons' && active.dirty) {
          active.compiled.encode(commandEncoder, {parameters: undefined});
          active.reader.request(commandEncoder);
          active.dirty = false;
          readySets.add(polygonSetChoice);
        }
      },
      getLayers() {
        if (viewChoice === 'polygons') {
          const state = polygonSets[polygonSetChoice];
          const perArea = metric === 'per-area';
          return [
            new SpatialAnalysisRasterLayer({
              id: `line-length-choropleth-${polygonSetChoice}-${metric}`,
              coordinateOrigin,
              gridSize: [state.grid.columns, state.grid.rows],
              bounds: state.grid.bounds,
              rowOrigin: 'south',
              values: perArea ? state.perArea : state.lengths,
              valueFormat: 'float32',
              valueIndices: state.cellRows,
              colormap: 'viridis',
              valueRange: [0, perArea ? state.perAreaMaximum : state.lengthMaximum],
              noDataColor: [0, 0, 0, 0],
              color: [255, 255, 255, 215]
            }),
            ...(showRoads
              ? [
                  new SpatialAnalysisSegmentLayer({
                    id: 'line-density-roads',
                    coordinateOrigin,
                    segments: segmentsBuffer,
                    instanceCount: segmentCount,
                    widthPixels: 0.5,
                    color: [255, 255, 255, 45]
                  })
                ]
              : []),
            new SpatialAnalysisSegmentLayer({
              id: `line-length-outline-${polygonSetChoice}`,
              coordinateOrigin,
              segments: state.outline,
              instanceCount: state.set.outlineSegments.length / 4,
              widthPixels: 1.6,
              color: [255, 220, 140, 230]
            })
          ];
        }
        const layers: Layer[] = [
          new SpatialAnalysisRasterLayer({
            id: 'line-density-cells',
            coordinateOrigin,
            gridSize: [COLUMNS, ROWS],
            bounds: gridBounds.buffer,
            values: densities,
            valueFormat: 'float32',
            colormap: 'inferno',
            valueRange: [0, densityMaximum],
            sqrtScale: true,
            discardAtOrBelow: 0,
            color: [255, 255, 255, 215]
          })
        ];
        if (showRoads) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: 'line-density-roads',
              coordinateOrigin,
              segments: segmentsBuffer,
              instanceCount: segmentCount,
              widthPixels: 0.7,
              color: [140, 200, 255, 70]
            })
          );
        }
        if (overlay !== 'none') {
          const buffers = overlayBuffers.get(overlay)!;
          if (overlay === 'point') {
            layers.push(
              new SpatialAnalysisPointLayer({
                id: 'line-density-overlay-points',
                coordinateOrigin,
                positions: buffers.vertices,
                instanceCount: overlayCellCount('point'),
                radiusPixels: 2.4,
                color: [90, 235, 255, 255]
              })
            );
          } else {
            layers.push(
              new SpatialAnalysisSegmentLayer({
                id: `line-density-overlay-${overlay}`,
                coordinateOrigin,
                segments: buffers.edges,
                instanceCount: buffers.edgeCount,
                widthPixels: 1.3,
                color: [90, 235, 255, 200]
              })
            );
          }
        }
        return layers;
      },
      getTooltip(event) {
        if (viewChoice !== 'polygons' || !readySets.has(polygonSetChoice)) return null;
        const state = polygonSets[polygonSetChoice];
        if (!event.coordinate) return null;
        const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
        const row = getFeatureRowAt(state.grid, state.featureCount, x, y);
        if (row < 0) {
          hoverReadout.setValue('outside the polygons');
          return null;
        }
        const text = `${state.set.names[row]}: ${(state.lengthValues[row] / 1000).toFixed(2)} km in ${formatCount(
          state.countValues[row]
        )} segments, ${state.perAreaValues[row].toFixed(1)} km per km2 (${(state.set.areas[row] / 1e6).toFixed(2)} km2)`;
        hoverReadout.setValue(text);
        return text;
      },
      destroy() {
        for (const state of Object.values(polygonSets)) state.reader.stop();
        reader.stop();
        resources.destroy();
      }
    };
    return instance;
  }
};
