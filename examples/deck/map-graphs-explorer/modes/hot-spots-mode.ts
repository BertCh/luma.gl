// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Local spatial autocorrelation of San Francisco bike-parking capacity. Each point carries a value
 * column (parking spaces). `GPUHotSpotAnalysis` (Getis-Ord Gi*) and `GPULocalMoran` (LISA) are both
 * compiled once, with and without false discovery rate correction, so the four graphs exist before
 * the first frame. The radius, the Moran significance level, the statistic and the FDR choice are
 * buffer writes or a choice between graphs that are already compiled: the rebuild counter stays 0.
 * The graphs run only when an input changed (the data is static), and `GPUHistogram` counts the
 * classes on the GPU so the only readback is a handful of integers.
 */

import type {Layer} from '@deck.gl/core';
import {
  GPUCommandGraph,
  GPUHistogram,
  GPUReadbackRing,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {getGPUSpatialAutocorrelationParameterValues, GPUHotSpotAnalysis, GPULocalMoran, GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH, GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH} from '@luma.gl/experimental/geospatial';
import {importGraphBuffer} from '@luma.gl/experimental/UNRESOLVED';
import {MapGraphsPointLayer, type MapGraphsColor} from '../map-graphs-layers';
import type {MapGraphsModeDefinition, MapGraphsModeInstance} from '../map-graphs-mode';
import {formatCount, MapGraphsResources} from '../map-graphs-resources';

/** Maximum neighbor-search lattice. Compile-time; results do not depend on it. */
const GRID_SIZE: readonly [number, number] = [256, 256];
const GI_BIN_COUNT = 7;
const MORAN_CLASS_COUNT = 5;
const SUMMARY_WORDS = GI_BIN_COUNT + GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH;
/** A class value no category can take, so negative Gi* bins (large uint32) are never "no data". */
const NO_DATA_VALUE = 0x7fffffff;
const HIDDEN: MapGraphsColor = [0, 0, 0, 0];
const NEUTRAL: MapGraphsColor = [150, 160, 175, 150];

/**
 * Gi* bins are `sint32` in `-3..3`. Reinterpreted as uint32 and reduced modulo the palette size 8
 * (which divides 2^32) they select entries `0` (not significant), `1..3` (hot 90/95/99%) and
 * `7, 6, 5` (cold 90/95/99%).
 */
const GI_COLORS: Record<number, MapGraphsColor> = {
  1: [253, 174, 97, 235],
  2: [244, 109, 67, 245],
  3: [165, 0, 38, 255],
  7: [171, 217, 233, 235],
  6: [116, 173, 209, 245],
  5: [49, 54, 149, 255]
};
/** Local Moran quadrant codes 1 HH, 2 LH, 3 LL, 4 HL. */
const MORAN_COLORS: Record<number, MapGraphsColor> = {
  1: [165, 0, 38, 255],
  2: [171, 217, 233, 245],
  3: [49, 54, 149, 255],
  4: [253, 174, 97, 245]
};

type Statistic = 'gi-star' | 'local-moran';

type Variant = {
  compiled: CompiledGPUCommandGraph<void>;
};

export const hotSpotsMode: MapGraphsModeDefinition = {
  id: 'hot-spots',
  title: 'Hot spots',
  recipes: ['GPUHotSpotAnalysis', 'GPULocalMoran', 'GPUHistogram'],
  description:
    'Where is bike-parking capacity unusually high or low compared with its neighbors? Getis-Ord ' +
    'Gi* hot and cold spots or local Moran quadrants over San Francisco, from a per-frame radius.',
  initialViewState: {longitude: -122.435, latitude: 37.765, zoom: 12.2},

  async create(context) {
    const parking = await context.data.getSanFranciscoBikeParking();
    context.signal.throwIfAborted();
    const {device} = context;
    const resources = new MapGraphsResources(device, 'hot-spots');
    const pointCount = parking.spaces.length;

    let minimumX = Infinity;
    let minimumY = Infinity;
    let maximumX = -Infinity;
    let maximumY = -Infinity;
    for (let index = 0; index < pointCount; index++) {
      minimumX = Math.min(minimumX, parking.positions[index * 2]);
      maximumX = Math.max(maximumX, parking.positions[index * 2]);
      minimumY = Math.min(minimumY, parking.positions[index * 2 + 1]);
      maximumY = Math.max(maximumY, parking.positions[index * 2 + 1]);
    }
    const bounds = [minimumX - 10, minimumY - 10, maximumX + 10, maximumY + 10] as const;

    let statistic: Statistic = 'gi-star';
    let falseDiscoveryRate = false;
    let radiusMeters = 400;
    let significanceLevel = 0.05;
    let showNotSignificant = true;
    let dirty = true;
    let needsReadback = true;
    let readbackPending = false;
    let destroyed = false;

    const positionsBuffer = resources.createBuffer('positions', parking.positions);
    const valuesBuffer = resources.createBuffer('values', parking.spaces);
    const parameters = resources.createParameterBuffer(
      'parameters',
      'float32',
      GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
    );
    const outputs = {
      'gi-star': {
        zScores: resources.createBuffer('gi-z-scores', pointCount * 4),
        classes: resources.createBuffer('gi-bins', pointCount * 4),
        counts: resources.createBuffer('gi-counts', GI_BIN_COUNT * 4),
        statistics: resources.createBuffer(
          'gi-statistics',
          GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH * 4
        )
      },
      'local-moran': {
        zScores: resources.createBuffer('moran-z-scores', pointCount * 4),
        classes: resources.createBuffer('moran-quadrants', pointCount * 4),
        counts: resources.createBuffer('moran-counts', MORAN_CLASS_COUNT * 4),
        statistics: resources.createBuffer(
          'moran-statistics',
          GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH * 4
        )
      }
    } as const;
    const readbackRing = resources.track(
      new GPUReadbackRing(device, {id: 'hot-spots-summary', byteLength: SUMMARY_WORDS * 4})
    );

    // FDR is a compile-time option of both recipes, so both settings are compiled up front and the
    // toggle selects between graphs that share one set of output buffers.
    function compileVariant(kind: Statistic, fdr: boolean): Variant {
      const graph = new GPUCommandGraph<void>(device, {
        id: `hot-spots-${kind}${fdr ? '-fdr' : ''}`
      });
      const positions = importGraphBuffer(
        graph,
        'positions',
        positionsBuffer,
        'float32x2',
        pointCount
      );
      const values = importGraphBuffer(graph, 'values', valuesBuffer, 'float32', pointCount);
      const parameterView = parameters.importToGraph(graph);
      const output = outputs[kind];
      const zScores = importGraphBuffer(graph, 'z-scores', output.zScores, 'float32', pointCount);
      const statistics = importGraphBuffer(
        graph,
        'statistics',
        output.statistics,
        'float32',
        GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH
      );
      if (kind === 'gi-star') {
        const bins = importGraphBuffer(graph, 'bins', output.classes, 'sint32', pointCount);
        graph.add(
          new GPUHotSpotAnalysis({
            id: 'gi-star',
            positions,
            values,
            parameters: parameterView,
            gridSize: GRID_SIZE,
            zScores,
            bins,
            globalStatistics: statistics,
            falseDiscoveryRate: fdr
          })
        );
        graph.add(
          new GPUHistogram({
            id: 'gi-star-counts',
            input: bins,
            output: importGraphBuffer(graph, 'counts', output.counts, 'uint32', GI_BIN_COUNT),
            edges: [-3, -2, -1, 0, 1, 2, 3, 4]
          })
        );
      } else {
        const quadrants = importGraphBuffer(
          graph,
          'quadrants',
          output.classes,
          'uint32',
          pointCount
        );
        graph.add(
          new GPULocalMoran({
            id: 'local-moran',
            positions,
            values,
            parameters: parameterView,
            gridSize: GRID_SIZE,
            zScores,
            quadrants,
            globalStatistics: statistics,
            falseDiscoveryRate: fdr
          })
        );
        graph.add(
          new GPUHistogram({
            id: 'local-moran-counts',
            input: quadrants,
            output: importGraphBuffer(graph, 'counts', output.counts, 'uint32', MORAN_CLASS_COUNT),
            edges: [0, 1, 2, 3, 4, 5]
          })
        );
      }
      return {compiled: resources.track(graph.compile())};
    }

    const variants: Record<`${Statistic}:${'plain' | 'fdr'}`, Variant> = {
      'gi-star:plain': compileVariant('gi-star', false),
      'gi-star:fdr': compileVariant('gi-star', true),
      'local-moran:plain': compileVariant('local-moran', false),
      'local-moran:fdr': compileVariant('local-moran', true)
    };
    const getActiveVariant = () => variants[`${statistic}:${falseDiscoveryRate ? 'fdr' : 'plain'}`];

    const writeParameters = () => {
      parameters.write(
        getGPUSpatialAutocorrelationParameterValues({
          bounds,
          radius: radiusMeters,
          significanceLevel
        })
      );
      dirty = true;
      needsReadback = true;
    };

    const statisticSelect = context.controls.addSelect<Statistic>({
      label: 'Statistic (buffer-level switch between compiled graphs)',
      options: [
        {value: 'gi-star', label: 'Getis-Ord Gi* hot and cold spots'},
        {value: 'local-moran', label: 'Local Moran quadrants'}
      ],
      value: statistic,
      onChange: value => {
        statistic = value;
        significanceControl.setDisabled(statistic === 'gi-star');
        dirty = true;
        needsReadback = true;
        updateLegendNote();
        context.updateLayers();
      }
    });
    void statisticSelect;
    context.controls.addSlider({
      label: 'Neighborhood radius (per-frame parameter)',
      min: 100,
      max: 1500,
      step: 25,
      value: radiusMeters,
      format: value => `${value} m`,
      onChange: value => {
        radiusMeters = value;
        writeParameters();
      }
    });
    const significanceControl = context.controls.addSlider({
      label: 'Moran significance level (per-frame parameter)',
      min: 0.001,
      max: 0.2,
      step: 0.001,
      value: significanceLevel,
      format: value => `p ≤ ${value.toFixed(3)}`,
      onChange: value => {
        significanceLevel = value;
        writeParameters();
      }
    });
    significanceControl.setDisabled(true);
    context.controls.addToggle({
      label: 'Benjamini-Hochberg FDR (compile-time option, both variants precompiled)',
      value: falseDiscoveryRate,
      onChange: value => {
        falseDiscoveryRate = value;
        dirty = true;
        needsReadback = true;
      }
    });
    context.controls.addToggle({
      label: 'Show not significant points',
      value: showNotSignificant,
      onChange: value => {
        showNotSignificant = value;
        context.updateLayers();
      }
    });
    context.controls.addLegend({
      title: 'Gi* confidence bin (hot to cold)',
      entries: [
        {color: GI_COLORS[3], label: 'Hot 99%'},
        {color: GI_COLORS[2], label: 'Hot 95%'},
        {color: GI_COLORS[1], label: 'Hot 90%'},
        {color: NEUTRAL, label: 'Not significant'},
        {color: GI_COLORS[7], label: 'Cold 90%'},
        {color: GI_COLORS[6], label: 'Cold 95%'},
        {color: GI_COLORS[5], label: 'Cold 99%'}
      ]
    });
    context.controls.addLegend({
      title: 'Local Moran quadrant',
      entries: [
        {color: MORAN_COLORS[1], label: 'High-High'},
        {color: MORAN_COLORS[3], label: 'Low-Low'},
        {color: MORAN_COLORS[4], label: 'High-Low outlier'},
        {color: MORAN_COLORS[2], label: 'Low-High outlier'}
      ]
    });
    const legendNote = context.controls.addNote('');
    const updateLegendNote = () =>
      legendNote.setValue(
        statistic === 'gi-star'
          ? 'Gi*: binary weights within the radius, the point itself included. Bins are 90/95/99% ' +
              'two-sided confidence, or BH-FDR corrected.'
          : 'Moran: the point itself excluded, conditional-randomization z-score; quadrants are ' +
              'shown where p is at most the significance level (BH-FDR corrected if on).'
      );
    updateLegendNote();
    context.controls.addReadout('Points', formatCount(pointCount));
    context.controls.addReadout('Value', 'bike-parking spaces');
    const hotReadout = context.controls.addReadout('Hot 99 / 95 / 90% (HH)');
    const coldReadout = context.controls.addReadout('Cold 99 / 95 / 90% (LL)');
    const outlierReadout = context.controls.addReadout('Outliers LH / HL');
    const insignificantReadout = context.controls.addReadout('Not significant');
    const momentsReadout = context.controls.addReadout('Mean / std. deviation');
    context.controls.addReadout('Data', parking.attribution);

    writeParameters();

    const readSummary = async (commandEncoder: Parameters<MapGraphsModeInstance['encode']>[0]) => {
      const ticket = readbackRing.tryAcquire();
      if (!ticket) return;
      const output = outputs[statistic];
      const reportedStatistic = statistic;
      const classCount = reportedStatistic === 'gi-star' ? GI_BIN_COUNT : MORAN_CLASS_COUNT;
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: output.counts,
        destinationBuffer: ticket.buffer,
        destinationOffset: 0,
        size: classCount * 4
      });
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: output.statistics,
        destinationBuffer: ticket.buffer,
        destinationOffset: GI_BIN_COUNT * 4,
        size: GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH * 4
      });
      ticket.markEncoded({byteOffset: 0, byteLength: SUMMARY_WORDS * 4});
      readbackPending = true;
      needsReadback = false;
      try {
        const bytes = await ticket.read();
        if (destroyed) return;
        const words = new Uint32Array(bytes.buffer, bytes.byteOffset, SUMMARY_WORDS);
        const floats = new Float32Array(bytes.buffer, bytes.byteOffset, SUMMARY_WORDS);
        const count = (index: number) => formatCount(words[index]);
        if (reportedStatistic === 'gi-star') {
          // Bins -3..3 are histogram rows 0..6.
          hotReadout.setValue(`${count(6)} / ${count(5)} / ${count(4)}`);
          coldReadout.setValue(`${count(0)} / ${count(1)} / ${count(2)}`);
          outlierReadout.setValue('n/a (Moran only)');
          insignificantReadout.setValue(count(3));
        } else {
          // Quadrant codes 0 (none), 1 HH, 2 LH, 3 LL, 4 HL are histogram rows 0..4.
          hotReadout.setValue(count(1));
          coldReadout.setValue(count(3));
          outlierReadout.setValue(`${count(2)} / ${count(4)}`);
          insignificantReadout.setValue(count(0));
        }
        const mean = floats[GI_BIN_COUNT + 1];
        const deviation = floats[GI_BIN_COUNT + 3];
        momentsReadout.setValue(`${mean.toFixed(2)} / ${deviation.toFixed(2)} spaces`);
      } catch {
        // The ring or device was destroyed while the read was in flight.
        needsReadback = true;
      } finally {
        readbackPending = false;
      }
    };

    const instance: MapGraphsModeInstance = {
      getCompiledGraphs: () => Object.values(variants).map(variant => variant.compiled),
      encode(commandEncoder, frame) {
        // The data is static: results only change with the radius, level, statistic or FDR.
        if (dirty || frame.frameIndex < 2) {
          getActiveVariant().compiled.encode(commandEncoder, {parameters: undefined});
          dirty = false;
        }
        if (needsReadback && !readbackPending && frame.frameIndex >= 1) {
          void readSummary(commandEncoder);
        }
      },
      getLayers() {
        const coordinateOrigin: [number, number, number] = [
          parking.origin[0],
          parking.origin[1],
          0
        ];
        const classes = outputs[statistic].classes;
        const colors = statistic === 'gi-star' ? GI_COLORS : MORAN_COLORS;
        const significantPalette = Array.from({length: 8}, (_, index) => colors[index] ?? HIDDEN);
        const insignificantPalette = Array.from({length: 8}, (_, index) =>
          index === 0 ? NEUTRAL : HIDDEN
        );
        const common = {
          coordinateOrigin,
          positions: positionsBuffer,
          instanceCount: pointCount,
          values: classes,
          valueFormat: 'uint32' as const,
          colormap: 'category' as const,
          noDataValue: NO_DATA_VALUE,
          noDataColor: HIDDEN
        };
        const layers: Layer[] = [];
        if (showNotSignificant) {
          layers.push(
            new MapGraphsPointLayer({
              ...common,
              id: `hot-spots-insignificant-${statistic}`,
              radiusPixels: 3,
              palette: insignificantPalette
            })
          );
        }
        layers.push(
          new MapGraphsPointLayer({
            ...common,
            id: `hot-spots-significant-${statistic}`,
            radiusPixels: 5.5,
            palette: significantPalette
          })
        );
        return layers;
      },
      destroy() {
        destroyed = true;
        resources.destroy();
      }
    };
    return instance;
  }
};
