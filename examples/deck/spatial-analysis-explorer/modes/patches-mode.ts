// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPURasterSieveParameterValues,
  GPURasterConnectedComponents,
  GPURasterDenseComponents,
  GPURasterPatchMetrics,
  GPURasterSieve,
  GPURasterThreshold,
  GPU_RASTER_SIEVE_PARAMETER_LENGTH,
  type GPURasterBufferBand,
  type GPURasterSieveMode
} from '@luma.gl/experimental/gpu-raster';
import {importGraphBuffer} from '../graph-buffers';
import {LocalMetricProjection} from '../spatial-analysis-data';
import {SpatialAnalysisRasterLayer, SpatialAnalysisSegmentLayer} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {SummaryReader} from './summary-reader';

/** Largest patch label the sieve and the metrics keep (compile-time). */
const PATCH_CAPACITY = 4096;
const SCALAR_WORDS = 5;
const PALETTE = [
  [78, 201, 255, 255],
  [255, 148, 72, 255],
  [189, 122, 255, 255],
  [87, 235, 168, 255],
  [255, 105, 168, 255],
  [245, 220, 87, 255],
  [107, 158, 255, 255],
  [255, 92, 92, 255]
] as const;

type Display = 'raw' | 'remove' | 'merge';

/**
 * Raster patches of the San Francisco elevation raster. A per-frame elevation threshold becomes a
 * mask, `GPURasterConnectedComponents` and `GPURasterDenseComponents` label the patches,
 * `GPURasterPatchMetrics` measures each (area, perimeter, bounding box) and `GPURasterSieve`
 * removes or merges the small ones below a per-frame minimum size. Both sieve modes are compiled
 * once, so switching them is only a display choice.
 */
export const patchesMode: SpatialAnalysisModeDefinition = {
  id: 'patches',
  title: 'Patches',
  contributors: [
    'GPURasterThreshold',
    'GPURasterConnectedComponents',
    'GPURasterDenseComponents',
    'GPURasterPatchMetrics',
    'GPURasterSieve'
  ],
  description:
    'Hills above an elevation threshold, labeled as patches and measured on the GPU. Raise the ' +
    'minimum patch size to sieve small patches away (or merge them into a neighbour); click a ' +
    'patch for its area, perimeter and bounding box.',
  initialViewState: {longitude: -122.44, latitude: 37.735, zoom: 11.9},

  async create(context) {
    const terrain = await context.data.getSanFranciscoTerrain();
    context.signal.throwIfAborted();
    const {device} = context;
    const {width, height, bounds, cellSize} = terrain;
    const cellCount = width * height;
    const projection = new LocalMetricProjection(terrain.origin);
    const origin: [number, number, number] = [terrain.origin[0], terrain.origin[1], 0];
    const resources = new SpatialAnalysisResources(device, 'patches');
    const cellArea = cellSize[0] * cellSize[1];

    // Sea (elevation 0) is invalid.
    const validityValues = new Uint32Array(cellCount);
    for (let index = 0; index < cellCount; index++) {
      validityValues[index] = terrain.elevation[index] > 0.5 ? 1 : 0;
    }
    const elevationBuffer = resources.createBuffer('elevation', terrain.elevation);
    const validityBuffer = resources.createBuffer('validity', validityValues);
    const thresholdParameter = resources.createParameterBuffer('threshold', 'float32', 1);
    const sieveParameters = resources.createParameterBuffer(
      'sieve-parameters',
      'uint32',
      GPU_RASTER_SIEVE_PARAMETER_LENGTH
    );

    const makeBuffer = (name: string, length: number) =>
      resources.createBuffer(name, Math.max(1, length) * 4);
    const buffers = {
      mask: makeBuffer('mask', cellCount),
      sparse: makeBuffer('sparse-labels', cellCount),
      sparseValidity: makeBuffer('sparse-validity', cellCount),
      sparseConverged: makeBuffer('sparse-converged', 1),
      iterations: makeBuffer('iterations', 1),
      labels: makeBuffer('labels', cellCount),
      labelValidity: makeBuffer('label-validity', cellCount),
      componentCount: makeBuffer('component-count', 1),
      overflow: makeBuffer('overflow', 1),
      pixelCounts: makeBuffer('pixel-counts', PATCH_CAPACITY),
      areas: makeBuffer('areas', PATCH_CAPACITY),
      perimeters: makeBuffer('perimeters', PATCH_CAPACITY),
      minColumns: makeBuffer('min-columns', PATCH_CAPACITY),
      minRows: makeBuffer('min-rows', PATCH_CAPACITY),
      maxColumns: makeBuffer('max-columns', PATCH_CAPACITY),
      maxRows: makeBuffer('max-rows', PATCH_CAPACITY)
    };
    const sieves = {
      remove: {
        labels: makeBuffer('remove-labels', cellCount),
        targets: makeBuffer('remove-targets', PATCH_CAPACITY),
        count: makeBuffer('remove-count', 1)
      },
      merge: {
        labels: makeBuffer('merge-labels', cellCount),
        targets: makeBuffer('merge-targets', PATCH_CAPACITY),
        count: makeBuffer('merge-count', 1)
      }
    };
    const highlightSegments = resources.createBuffer('highlight', 4 * 16);

    const graph = new GPUCommandGraph<void>(device, {id: 'patches'});
    const view = <Format extends 'uint32' | 'float32'>(
      name: string,
      buffer: Buffer,
      format: Format,
      length: number
    ) => importGraphBuffer(graph, name, buffer, format, length) as GraphDataView<Format>;
    {
      const elevation: GPURasterBufferBand<'float32'> = {
        id: 'elevation',
        format: 'float32',
        storage: {kind: 'buffer', values: view('elevation', elevationBuffer, 'float32', cellCount)},
        validity: view('validity', validityBuffer, 'uint32', cellCount)
      };
      const mask = view('mask', buffers.mask, 'uint32', cellCount);
      new GPURasterThreshold({
        id: 'above-threshold',
        width,
        height,
        input: elevation,
        output: mask,
        threshold: thresholdParameter.importToGraph(graph),
        operation: 'above'
      }).addToGraph(graph);
      const sparse = view('sparse', buffers.sparse, 'uint32', cellCount);
      const sparseValidity = view('sparse-validity', buffers.sparseValidity, 'uint32', cellCount);
      const sparseConverged = view('sparse-converged', buffers.sparseConverged, 'uint32', 1);
      new GPURasterConnectedComponents({
        id: 'components',
        width,
        height,
        input: {id: 'mask', format: 'uint32', storage: {kind: 'buffer', values: mask}},
        output: sparse,
        outputValidity: sparseValidity,
        converged: sparseConverged,
        iterationCount: view('iterations', buffers.iterations, 'uint32', 1),
        connectivity: 4
      }).addToGraph(graph);
      const labels = view('labels', buffers.labels, 'uint32', cellCount);
      const labelValidity = view('label-validity', buffers.labelValidity, 'uint32', cellCount);
      const componentCount = view('component-count', buffers.componentCount, 'uint32', 1);
      const overflow = view('overflow', buffers.overflow, 'uint32', 1);
      new GPURasterDenseComponents({
        id: 'dense',
        width,
        height,
        input: sparse,
        inputValidity: sparseValidity,
        converged: sparseConverged,
        output: labels,
        outputValidity: labelValidity,
        componentCount,
        overflow,
        capacity: PATCH_CAPACITY
      }).addToGraph(graph);
      const patchInput = {
        width,
        height,
        labels,
        labelValidity,
        converged: sparseConverged,
        componentCount,
        overflow
      };
      graph.add(
        new GPURasterPatchMetrics({
          ...patchInput,
          id: 'metrics',
          affine: [cellSize[0], 0, bounds[0], 0, -cellSize[1], bounds[3]],
          output: {
            pixelCounts: view('pixel-counts', buffers.pixelCounts, 'uint32', PATCH_CAPACITY),
            areas: view('areas', buffers.areas, 'float32', PATCH_CAPACITY),
            perimeters: view('perimeters', buffers.perimeters, 'float32', PATCH_CAPACITY),
            minColumns: view('min-columns', buffers.minColumns, 'uint32', PATCH_CAPACITY),
            minRows: view('min-rows', buffers.minRows, 'uint32', PATCH_CAPACITY),
            maxColumns: view('max-columns', buffers.maxColumns, 'uint32', PATCH_CAPACITY),
            maxRows: view('max-rows', buffers.maxRows, 'uint32', PATCH_CAPACITY)
          }
        })
      );
      const sieveView = sieveParameters.importToGraph(graph);
      for (const mode of ['remove', 'merge'] as GPURasterSieveMode[]) {
        const output = sieves[mode];
        graph.add(
          new GPURasterSieve({
            ...patchInput,
            id: `sieve-${mode}`,
            patchCapacity: PATCH_CAPACITY,
            parameters: sieveView,
            mode,
            connectivity: 4,
            output: {
              labels: view(`${mode}-labels`, output.labels, 'uint32', cellCount),
              patchTargets: view(`${mode}-targets`, output.targets, 'uint32', PATCH_CAPACITY),
              sievedCount: view(`${mode}-count`, output.count, 'uint32', 1)
            }
          })
        );
      }
    }
    const compiled = resources.track(graph.compile());

    // State.
    let display: Display = 'remove';
    let thresholdMeters = 120;
    let minimumPixels = 60;
    let dirty = true;
    let selectedPatch = 0;
    let snapshot: {
      words: Uint32Array;
      floats: Float32Array;
      labels: Uint32Array;
      column: (index: number) => number;
    } | null = null;

    const writeThreshold = () => {
      thresholdParameter.write(Float32Array.of(thresholdMeters));
      dirty = true;
    };
    const writeSieve = () => {
      sieveParameters.write(getGPURasterSieveParameterValues({minimumPixels}));
      dirty = true;
    };
    writeThreshold();
    writeSieve();

    // Controls.
    context.controls.addSlider({
      label: 'Elevation threshold (per-frame)',
      min: 10,
      max: 200,
      step: 5,
      value: thresholdMeters,
      format: value => `${value} m`,
      onChange: value => {
        thresholdMeters = value;
        writeThreshold();
      }
    });
    context.controls.addSlider({
      label: 'Minimum patch size (per-frame sieve parameter)',
      min: 1,
      max: 400,
      step: 1,
      value: minimumPixels,
      format: value => `${value} cells (${((value * cellArea) / 10000).toFixed(1)} ha)`,
      onChange: value => {
        minimumPixels = value;
        writeSieve();
      }
    });
    context.controls.addSelect<Display>({
      label: 'Show patches',
      options: [
        {value: 'raw', label: 'Labeled components (before the sieve)'},
        {value: 'remove', label: 'Sieve: remove small patches'},
        {value: 'merge', label: 'Sieve: merge small patches into a neighbour'}
      ],
      value: display,
      onChange: value => {
        display = value;
        context.updateLayers();
        updateSelection();
      }
    });
    context.controls.addLegend({
      title: 'Patch (colour repeats every 8 labels); grey = sieved away',
      entries: PALETTE.slice(0, 4).map((color, index) => ({color, label: `#${index + 1}`}))
    });
    context.controls.addNote(
      'Remove clears a small patch; merge hands it to the largest large patch it touches ' +
        '(clumps of one mask never touch, so merge behaves like remove here).'
    );
    context.controls.addReadout('Raster', `${width} x ${height} cells`);
    const countReadout = context.controls.addReadout('Patches (converged, overflow)');
    const sievedReadout = context.controls.addReadout('Sieved away');
    const sizeReadout = context.controls.addReadout('Largest patch');
    const meanReadout = context.controls.addReadout('Mean patch area');
    const selectedReadout = context.controls.addReadout('Selected patch (click)', 'none');
    const areaReadout = context.controls.addReadout('Area / perimeter');
    const boxReadout = context.controls.addReadout('Bounding box');
    const fateReadout = context.controls.addReadout('After the sieve');
    context.controls.addReadout('Data', terrain.attribution);

    // Readback: scalars, per-patch columns and the label raster.
    const columnNames = [
      'pixelCounts',
      'areas',
      'perimeters',
      'minColumns',
      'minRows',
      'maxColumns',
      'maxRows'
    ] as const;
    const reader = new SummaryReader(
      resources,
      'patches',
      [
        {buffer: buffers.componentCount, size: 4},
        {buffer: buffers.sparseConverged, size: 4},
        {buffer: buffers.overflow, size: 4},
        {buffer: sieves.remove.count, size: 4},
        {buffer: sieves.merge.count, size: 4},
        ...columnNames.map(name => ({buffer: buffers[name], size: PATCH_CAPACITY * 4})),
        {buffer: sieves.remove.targets, size: PATCH_CAPACITY * 4},
        {buffer: sieves.merge.targets, size: PATCH_CAPACITY * 4},
        {buffer: buffers.labels, size: cellCount * 4}
      ],
      bytes => {
        const words = new Uint32Array(bytes);
        const floats = new Float32Array(bytes);
        const columnOffset = (name: (typeof columnNames)[number]) =>
          SCALAR_WORDS + columnNames.indexOf(name) * PATCH_CAPACITY;
        const targetOffset = SCALAR_WORDS + columnNames.length * PATCH_CAPACITY;
        snapshot = {
          words,
          floats,
          labels: words.subarray(targetOffset + 2 * PATCH_CAPACITY),
          column: index => index
        };
        const count = Math.min(words[0], PATCH_CAPACITY);
        let largest = 0;
        let totalArea = 0;
        for (let patch = 1; patch <= count; patch++) {
          const area = floats[columnOffset('areas') + patch - 1];
          totalArea += area;
          if (area > (floats[columnOffset('areas') + largest - 1] ?? 0)) largest = patch;
        }
        countReadout.setValue(
          `${formatCount(words[0])} (${words[1] ? 'yes' : 'no'}, ${words[2] ? 'overflow' : 'none'})`
        );
        sievedReadout.setValue(
          `${formatCount(words[3])} removed, ${formatCount(words[4])} merged-mode`
        );
        sizeReadout.setValue(
          largest
            ? `#${largest}: ${(floats[columnOffset('areas') + largest - 1] / 10000).toFixed(1)} ha`
            : 'none'
        );
        meanReadout.setValue(count ? `${(totalArea / count / 10000).toFixed(2)} ha` : 'none');
        updateSelection();
      }
    );

    function updateSelection() {
      if (!snapshot || selectedPatch < 1 || selectedPatch > PATCH_CAPACITY) {
        selectedReadout.setValue('none');
        areaReadout.setValue('-');
        boxReadout.setValue('-');
        fateReadout.setValue('-');
        highlightSegments.write(new Float32Array(16).fill(Number.NaN));
        return;
      }
      const {words, floats} = snapshot;
      const patchIndex = selectedPatch - 1;
      const at = (name: (typeof columnNames)[number]) =>
        SCALAR_WORDS + columnNames.indexOf(name) * PATCH_CAPACITY + patchIndex;
      const area = floats[at('areas')];
      const perimeter = floats[at('perimeters')];
      const columns = [words[at('minColumns')], words[at('maxColumns')]];
      const rows = [words[at('minRows')], words[at('maxRows')]];
      selectedReadout.setValue(
        `#${selectedPatch} (${formatCount(words[at('pixelCounts')])} cells)`
      );
      areaReadout.setValue(
        `${(area / 10000).toFixed(2)} ha / ${perimeter.toFixed(0)} m (compactness ${(
          (4 * Math.PI * area) / (perimeter * perimeter)
        ).toFixed(2)})`
      );
      boxReadout.setValue(
        `${((columns[1] - columns[0] + 1) * cellSize[0]).toFixed(0)} x ${(
          (rows[1] - rows[0] + 1) * cellSize[1]
        ).toFixed(0)} m`
      );
      const targetBase = SCALAR_WORDS + columnNames.length * PATCH_CAPACITY;
      const removeTarget = words[targetBase + patchIndex];
      const mergeTarget = words[targetBase + PATCH_CAPACITY + patchIndex];
      const describe = (target: number) =>
        target === selectedPatch ? 'kept' : target === 0 ? 'removed' : `merged into #${target}`;
      fateReadout.setValue(`remove: ${describe(removeTarget)}; merge: ${describe(mergeTarget)}`);
      const left = bounds[0] + columns[0] * cellSize[0];
      const right = bounds[0] + (columns[1] + 1) * cellSize[0];
      const top = bounds[3] - rows[0] * cellSize[1];
      const bottom = bounds[3] - (rows[1] + 1) * cellSize[1];
      highlightSegments.write(
        Float32Array.of(
          left,
          bottom,
          right,
          bottom,
          right,
          bottom,
          right,
          top,
          right,
          top,
          left,
          top,
          left,
          top,
          left,
          bottom
        )
      );
    }
    highlightSegments.write(new Float32Array(16).fill(Number.NaN));

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [compiled],
      encode(commandEncoder, frame) {
        if (dirty || frame.frameIndex < 2) {
          compiled.encode(commandEncoder, {parameters: undefined});
          reader.request(commandEncoder);
          dirty = false;
        }
        reader.flush(commandEncoder);
      },
      getLayers() {
        const rasterProps = {
          coordinateOrigin: origin,
          gridSize: [width, height] as const,
          bounds,
          rowOrigin: 'north' as const
        };
        const patchProps = {
          ...rasterProps,
          valueFormat: 'uint32' as const,
          colormap: 'category' as const,
          palette: PALETTE,
          noDataValue: 0,
          noDataColor: [0, 0, 0, 0] as const
        };
        const layers: Layer[] = [
          new SpatialAnalysisRasterLayer({
            ...rasterProps,
            id: 'patches-elevation',
            values: elevationBuffer,
            valueFormat: 'float32',
            colormap: 'grayscale',
            valueRange: [0, 280],
            discardAtOrBelow: 0.5,
            color: [255, 255, 255, 120]
          })
        ];
        if (display !== 'raw') {
          layers.push(
            new SpatialAnalysisRasterLayer({
              ...rasterProps,
              id: 'patches-sieved-away',
              values: buffers.labels,
              valueFormat: 'uint32',
              colormap: 'mask',
              color: [150, 150, 150, 170],
              noDataColor: [0, 0, 0, 0]
            })
          );
        }
        layers.push(
          new SpatialAnalysisRasterLayer({
            ...patchProps,
            id: `patches-${display}`,
            values: display === 'raw' ? buffers.labels : sieves[display].labels,
            color: [255, 255, 255, 235]
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'patches-highlight',
            coordinateOrigin: origin,
            segments: highlightSegments,
            instanceCount: 4,
            widthPixels: 2.5,
            color: [255, 255, 255, 255]
          })
        );
        return layers;
      },
      onClick(event) {
        if (!event.coordinate || !snapshot) return false;
        const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
        const column = Math.floor((x - bounds[0]) / cellSize[0]);
        const row = Math.floor((bounds[3] - y) / cellSize[1]);
        if (column < 0 || column >= width || row < 0 || row >= height) return false;
        selectedPatch = snapshot.labels[row * width + column] ?? 0;
        updateSelection();
        return true;
      },
      destroy() {
        reader.stop();
        resources.destroy();
      }
    };
    return instance;
  }
};
