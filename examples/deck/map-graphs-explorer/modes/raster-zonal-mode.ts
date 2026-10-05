// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Raster zonal statistics: the San Francisco ZIP-code polygons are rasterized once on the CPU
 * (scanline, even-odd fill) into a zone raster, and two `GPURasterZonalStatistics` recipes in one
 * graph tabulate the elevation band and the GPU slope band per zone (count, mean, minimum,
 * maximum). The choropleth draws each cell through `valueIndices` (cell to zone) into the selected
 * recipe's `means` column. The graph is encoded once (its inputs never change); one small
 * readback of every statistic column feeds the ranked readouts, the color range and the tooltip.
 */

import type {Layer} from '@deck.gl/core';
import type {Buffer, CommandEncoder} from '@luma.gl/core';
import {
  GPUCommandGraph,
  GPUReadbackRing,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPURasterBand} from '@luma.gl/experimental/gpu-raster';
import {getGPUTerrainDerivativesParameterValues, GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH, GPUTerrainDerivatives} from '@luma.gl/experimental/gpu-terrain';
import {GPURasterZonalStatistics} from '@luma.gl/experimental/gpu-raster';
import {importGraphBuffer} from '@luma.gl/experimental/UNRESOLVED';
import {LocalMetricProjection} from '../map-graphs-data';
import {MapGraphsRasterLayer, MapGraphsSegmentLayer} from '../map-graphs-layers';
import type {
  MapGraphsModeDefinition,
  MapGraphsModeInstance,
  MapGraphsPointerEvent
} from '../map-graphs-mode';
import {formatCount, MapGraphsResources} from '../map-graphs-resources';
import {addHideNaNPass, rasterizePolygonZones} from './raster-zonal-layers';

type ZoneValue = 'elevation' | 'slope';

const SUN_AZIMUTH_DEGREES = 315;
const SUN_ALTITUDE_DEGREES = 40;
/** Columns read back: cell counts, then value counts, means, minimums, maximums per value band. */
const COLUMN_COUNT = 9;

type ZoneStatistics = {
  valueCounts: Uint32Array;
  means: Float32Array;
  minimums: Float32Array;
  maximums: Float32Array;
};

/**
 * Raster zonal-statistics demo: per-ZIP elevation and slope statistics from `GPURasterZonalStatistics`
 * over a CPU-rasterized zone raster.
 */
export const rasterZonalMode: MapGraphsModeDefinition = {
  id: 'raster-zonal',
  title: 'Raster zonal',
  recipes: ['GPURasterZonalStatistics', 'GPUTerrainDerivatives'],
  description:
    'Per-ZIP-code mean elevation or slope of a raster. The ZIP polygons are rasterized once on ' +
    'the CPU; the statistics are computed on the GPU and drawn through a cell-to-zone lookup. ' +
    'Switching the value never recompiles.',
  initialViewState: {longitude: -122.44, latitude: 37.755, zoom: 12},

  async create(context) {
    const [terrain, zips] = await Promise.all([
      context.data.getSanFranciscoTerrain(),
      context.data.getSanFranciscoZipCodes()
    ]);
    context.signal.throwIfAborted();
    const {device} = context;
    const {width, height, bounds, cellSize} = terrain;
    const pixelCount = width * height;
    const projection = new LocalMetricProjection(terrain.origin);
    const origin: [number, number, number] = [terrain.origin[0], terrain.origin[1], 0];
    const resources = new MapGraphsResources(device, 'raster-zonal');

    // Sea (elevation 0) is invalid, so it is excluded from every zone statistic.
    const validityValues = new Uint32Array(pixelCount);
    for (let index = 0; index < pixelCount; index++) {
      validityValues[index] = terrain.elevation[index] > 0.5 ? 1 : 0;
    }
    // Zone ID = feature row + 1; 0 (outside every ZIP) is the ignored zone.
    const zoneValues = rasterizePolygonZones(zips, {width, height, bounds, cellSize});
    const featureCount = zips.featureOffsets.length - 1;
    const zoneCapacity = featureCount + 1;

    const elevationBuffer = resources.createBuffer('elevation', terrain.elevation);
    const validityBuffer = resources.createBuffer('validity', validityValues);
    const slopeBuffer = resources.createBuffer('slope', pixelCount * 4);
    const hillshadeBuffer = resources.createBuffer('hillshade', pixelCount * 4);
    const zonesBuffer = resources.createBuffer('zones', zoneValues);
    const outlineSegmentsBuffer = resources.createBuffer('outline-segments', zips.outlineSegments);
    const createColumn = (name: string) => resources.createBuffer(name, zoneCapacity * 4);
    const cellCountsBuffer = createColumn('cell-counts');
    const columns = {
      elevation: {
        valueCounts: createColumn('elevation-value-counts'),
        means: createColumn('elevation-means'),
        minimums: createColumn('elevation-minimums'),
        maximums: createColumn('elevation-maximums')
      },
      slope: {
        valueCounts: createColumn('slope-value-counts'),
        means: createColumn('slope-means'),
        minimums: createColumn('slope-minimums'),
        maximums: createColumn('slope-maximums')
      }
    };
    const displayMeans = {
      elevation: createColumn('elevation-display-means'),
      slope: createColumn('slope-display-means')
    };
    const derivativesSettings = resources.createParameterBuffer(
      'derivatives-settings',
      'float32',
      GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH
    );

    const createBand = (
      graph: GPUCommandGraph<void>,
      name: string,
      buffer: Buffer,
      validity: GraphDataView<'uint32'>
    ): GPURasterBand => ({
      id: name,
      format: 'float32',
      storage: {
        kind: 'buffer',
        values: importGraphBuffer(graph, name, buffer, 'float32', pixelCount)
      },
      validity
    });
    const importValidity = (graph: GPUCommandGraph<void>) =>
      importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', pixelCount);

    // Base: slope (and hillshade backdrop). The zonal graph reads slope, so it is encoded after.
    const baseGraph = new GPUCommandGraph<void>(device, {id: 'raster-zonal-base'});
    baseGraph.add(
      new GPUTerrainDerivatives({
        id: 'derivatives',
        width,
        height,
        elevation: createBand(baseGraph, 'elevation', elevationBuffer, importValidity(baseGraph)),
        settings: derivativesSettings.importToGraph(baseGraph),
        slope: importGraphBuffer(baseGraph, 'slope', slopeBuffer, 'float32', pixelCount),
        hillshade: importGraphBuffer(
          baseGraph,
          'hillshade',
          hillshadeBuffer,
          'float32',
          pixelCount
        ),
        cellSizeMode: 'uniform',
        rowDirection: 'south'
      })
    );
    const baseCompiled: CompiledGPUCommandGraph<void> = resources.track(baseGraph.compile());

    const zonalGraph = new GPUCommandGraph<void>(device, {id: 'raster-zonal'});
    const zonesView = importGraphBuffer(zonalGraph, 'zones', zonesBuffer, 'uint32', pixelCount);
    const uint32Column = (name: string, buffer: Buffer) =>
      importGraphBuffer(zonalGraph, name, buffer, 'uint32', zoneCapacity);
    const float32Column = (name: string, buffer: Buffer) =>
      importGraphBuffer(zonalGraph, name, buffer, 'float32', zoneCapacity);
    const zonalValidity = importValidity(zonalGraph);
    const meansViews = {
      elevation: float32Column('elevation-means', columns.elevation.means),
      slope: float32Column('slope-means', columns.slope.means)
    };
    for (const [name, sourceBuffer] of [
      ['elevation', elevationBuffer],
      ['slope', slopeBuffer]
    ] as const) {
      zonalGraph.add(
        new GPURasterZonalStatistics({
          id: `zonal-${name}`,
          width,
          height,
          zones: zonesView,
          values: createBand(zonalGraph, name, sourceBuffer, zonalValidity),
          zoneCapacity,
          ignoredZone: 0,
          output: {
            ...(name === 'elevation'
              ? {cellCounts: uint32Column('cell-counts', cellCountsBuffer)}
              : {}),
            valueCounts: uint32Column(`${name}-value-counts`, columns[name].valueCounts),
            means: meansViews[name],
            minimums: float32Column(`${name}-minimums`, columns[name].minimums),
            maximums: float32Column(`${name}-maximums`, columns[name].maximums)
          }
        })
      );
    }
    for (const name of ['elevation', 'slope'] as const) {
      addHideNaNPass(zonalGraph, {
        id: `hide-nan-${name}`,
        length: zoneCapacity,
        input: meansViews[name],
        output: float32Column(`${name}-display-means`, displayMeans[name])
      });
    }
    const zonalCompiled: CompiledGPUCommandGraph<void> = resources.track(zonalGraph.compile());
    const readbackRing = resources.track(
      new GPUReadbackRing(device, {
        id: 'raster-zonal-statistics',
        byteLength: COLUMN_COUNT * zoneCapacity * 4
      })
    );

    // --- State ---------------------------------------------------------------------------------
    let zoneValue: ZoneValue = 'elevation';
    let dirty = true;
    let readbackRequested = false;
    let readbackPending = false;
    let readbackDone = false;
    let destroyed = false;
    let statistics: Record<ZoneValue, ZoneStatistics> | null = null;
    let cellCounts: Uint32Array | null = null;
    // Color ranges until the readback arrives: elevation range, typical slope.
    const valueRange: Record<ZoneValue, [number, number]> = {
      elevation: [terrain.elevationRange[0], terrain.elevationRange[1]],
      slope: [0, 30]
    };
    derivativesSettings.write(
      getGPUTerrainDerivativesParameterValues({
        cellSize,
        azimuthDegrees: SUN_AZIMUTH_DEGREES,
        altitudeDegrees: SUN_ALTITUDE_DEGREES
      })
    );

    const getZoneName = (zone: number) => {
      const id = String(zips.featureIds[zone - 1]);
      const name = zips.featureNames[zone - 1];
      return name && name !== id ? `${id} ${name}` : id;
    };
    const getZone = (event: MapGraphsPointerEvent): number => {
      if (!event.coordinate) return 0;
      const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
      const column = Math.floor((x - bounds[0]) / cellSize[0]);
      const row = Math.floor((bounds[3] - y) / cellSize[1]);
      if (column < 0 || row < 0 || column >= width || row >= height) return 0;
      return zoneValues[row * width + column];
    };

    // --- Controls ------------------------------------------------------------------------------
    context.controls.addSelect<ZoneValue>({
      label: 'Zone value',
      options: [
        {value: 'elevation', label: 'Elevation (m)'},
        {value: 'slope', label: 'Slope (degrees)'}
      ],
      value: zoneValue,
      onChange: value => {
        zoneValue = value;
        refreshReadouts();
        context.updateLayers();
      }
    });
    context.controls.addLegend({
      title: 'Zone mean (range from the GPU statistics)',
      gradient: {
        colors: [
          [68, 1, 84],
          [33, 145, 140],
          [253, 231, 37]
        ],
        minimumLabel: 'lowest zone',
        maximumLabel: 'highest zone'
      }
    });
    const topReadouts = Array.from({length: 5}, (_, rank) =>
      context.controls.addReadout(`Top zone ${rank + 1}`, '...')
    );
    context.controls.addReadout(
      'Raster',
      `${width} × ${height} cells, ${cellSize[0].toFixed(1)} m`
    );
    context.controls.addReadout('Zones', `${featureCount} ZIP codes, rasterized once on the CPU`);
    context.controls.addReadout('Data', `${terrain.attribution}; ${zips.attribution}`);

    function refreshReadouts(): void {
      if (!statistics) return;
      const {means, minimums, maximums, valueCounts} = statistics[zoneValue];
      const ranked = Array.from({length: featureCount}, (_, row) => row + 1)
        .filter(zone => Number.isFinite(means[zone]))
        .sort((a, b) => means[b] - means[a]);
      topReadouts.forEach((readout, rank) => {
        const zone = ranked[rank];
        readout.setValue(
          zone === undefined
            ? '-'
            : `${getZoneName(zone)}: mean ${means[zone].toFixed(1)} ` +
                `(${minimums[zone].toFixed(0)}–${maximums[zone].toFixed(0)}), ` +
                `${formatCount(valueCounts[zone])} cells`
        );
      });
    }

    const readStatistics = (commandEncoder: CommandEncoder): boolean => {
      const ticket = readbackRing.tryAcquire();
      if (!ticket) return false;
      const sources: Buffer[] = [
        cellCountsBuffer,
        columns.elevation.valueCounts,
        columns.elevation.means,
        columns.elevation.minimums,
        columns.elevation.maximums,
        columns.slope.valueCounts,
        columns.slope.means,
        columns.slope.minimums,
        columns.slope.maximums
      ];
      sources.forEach((buffer, column) => {
        commandEncoder.copyBufferToBuffer({
          sourceBuffer: buffer,
          destinationBuffer: ticket.buffer,
          destinationOffset: column * zoneCapacity * 4,
          size: zoneCapacity * 4
        });
      });
      const byteLength = COLUMN_COUNT * zoneCapacity * 4;
      ticket.markEncoded({byteOffset: 0, byteLength});
      readbackPending = true;
      void (async () => {
        try {
          const bytes = await ticket.read();
          if (destroyed) return;
          // Copy: the staging view is only valid until the ticket is released.
          const words = new Uint32Array(
            bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + byteLength)
          );
          const floats = new Float32Array(words.buffer);
          const range = (column: number) =>
            [column * zoneCapacity, (column + 1) * zoneCapacity] as const;
          const uints = (column: number) => words.slice(...range(column));
          const reals = (column: number) => floats.slice(...range(column));
          cellCounts = uints(0);
          statistics = {
            elevation: {
              valueCounts: uints(1),
              means: reals(2),
              minimums: reals(3),
              maximums: reals(4)
            },
            slope: {
              valueCounts: uints(5),
              means: reals(6),
              minimums: reals(7),
              maximums: reals(8)
            }
          };
          for (const name of ['elevation', 'slope'] as const) {
            const means = Array.from(statistics[name].means.slice(1)).filter(Number.isFinite);
            if (means.length > 0) valueRange[name] = [Math.min(...means), Math.max(...means)];
          }
          readbackDone = true;
          refreshReadouts();
          context.updateLayers();
        } catch {
          // The ring or device was destroyed while the read was in flight.
        } finally {
          readbackPending = false;
        }
      })();
      return true;
    };

    const instance: MapGraphsModeInstance = {
      getCompiledGraphs: () => [baseCompiled, zonalCompiled],
      encode(commandEncoder) {
        // Encode once: base first (slope), then the zonal graph that reads it.
        if (dirty) {
          baseCompiled.encode(commandEncoder, {parameters: undefined});
          zonalCompiled.encode(commandEncoder, {parameters: undefined});
          dirty = false;
          readbackRequested = true;
        }
        // A ticket may be unavailable while earlier reads are in flight: retry next frame.
        if (
          readbackRequested &&
          !readbackPending &&
          !readbackDone &&
          readStatistics(commandEncoder)
        ) {
          readbackRequested = false;
        }
      },
      getLayers() {
        const layers: Layer[] = [
          new MapGraphsRasterLayer({
            coordinateOrigin: origin,
            gridSize: [width, height],
            bounds,
            rowOrigin: 'north',
            id: 'raster-zonal-hillshade',
            values: hillshadeBuffer,
            valueFormat: 'float32',
            colormap: 'grayscale',
            valueRange: [0, 1],
            color: [255, 255, 255, 150]
          }),
          new MapGraphsRasterLayer({
            coordinateOrigin: origin,
            gridSize: [width, height],
            bounds,
            rowOrigin: 'north',
            id: 'raster-zonal-zones',
            // Cell -> zone -> that zone's mean. Zone 0 and empty zones are -1: hidden.
            values: displayMeans[zoneValue],
            valueFormat: 'float32',
            valueIndices: zonesBuffer,
            colormap: 'viridis',
            valueRange: valueRange[zoneValue],
            // Hides the -1 sentinel of zone 0 and empty zones.
            discardAtOrBelow: 0,
            color: [255, 255, 255, 190],
            noDataColor: [0, 0, 0, 0]
          }),
          new MapGraphsSegmentLayer({
            id: 'raster-zonal-outlines',
            coordinateOrigin: origin,
            segments: outlineSegmentsBuffer,
            instanceCount: zips.outlineSegments.length / 4,
            widthPixels: 1.4,
            color: [255, 255, 255, 230]
          })
        ];
        return layers;
      },
      getTooltip(event) {
        const zone = getZone(event);
        if (zone === 0 || !statistics) return null;
        const mean = statistics[zoneValue].means[zone];
        const unit = zoneValue === 'elevation' ? 'm' : '°';
        return (
          `${getZoneName(zone)}\nmean ${zoneValue}: ` +
          `${Number.isFinite(mean) ? `${mean.toFixed(1)} ${unit}` : 'no data'}` +
          (cellCounts ? `\n${formatCount(cellCounts[zone])} cells` : '')
        );
      },
      destroy() {
        destroyed = true;
        resources.destroy();
      }
    };
    return instance;
  }
};
