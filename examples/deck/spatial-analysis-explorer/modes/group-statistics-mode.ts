// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Per-zone statistics and period-over-period compare maps over New York trip vertices, on the GPU.
 *
 * Zones are Quadbin cells. `GPUPointToCell` keys every vertex and `GPUGroupStatistics` computes
 * count, sum, mean, minimum, maximum, standard deviation, median, a per-frame percentile, mode and
 * unique count per zone for two columns (speed and vendor), plus a per-vertex z-score. All outputs
 * land in one metrics buffer; a small kernel picks the chosen column and statistic from a
 * two-word selection buffer, so switching statistic never recompiles. `GPUClassBreaks` and
 * `GPUColorScale` turn it into packed colors that a Quadbin cell layer draws straight from the
 * GPU.
 *
 * The compare view aggregates two time windows of the same vertices with `GPUCellAggregation`,
 * joins the two cell tables with `GPUCellTableCompare` (delta, ratio, percent change and a
 * Poisson z-score), and `GPUKeyJoin` attaches the zone statistics to the compare cells by key.
 * Window sliders rewrite two mask buffers; each graph is re-encoded only when its inputs change.
 */

import type {Layer} from '@deck.gl/core';
import type {Buffer, CommandEncoder} from '@luma.gl/core';
import {
  GPUCommandGraph,
  GPUReadbackRing,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUCellAggregation,
  GPUCellTableCompare,
  GPUPointToCell
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  GPUClassBreaks,
  GPUColorScale,
  GPUGroupStatistics,
  GPUKeyJoin,
  getGPUClassBreaksParameterLength,
  getGPUClassBreaksParameterValues,
  getGPUColorScaleParameterValues,
  GPU_COLOR_SCALE_PARAMETER_LENGTH,
  type GPUClassBreaksMethod,
  type GPUGroupStatistic
} from '@luma.gl/experimental/gpu-dataframe';
import {importGraphBuffer} from '../graph-buffers';
import {LocalMetricProjection} from '../spatial-analysis-data';
import {SpatialAnalysisPointLayer} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {
  createPanelBlock,
  formatCompact,
  getRampPalette,
  packColor,
  PackedQuadbinCellLayer
} from './classification-layers';
import {
  addMetricSelectPass,
  addSlotPackPass,
  getClassLegendHtml,
  SlotSlices
} from './group-statistics-layers';

/** Quadbin resolution of the zones (cell edge about 0.9 km at New York's latitude). */
const ZONE_RESOLUTION = 15;
/** Zone table rows; a multiple of 64 so every metrics slot starts on a 256-byte boundary. */
const ZONE_CAPACITY = 2048;
/** Compare table rows: the union of two zone-sized tables. */
const COMPARE_CAPACITY = 4096;
const MAXIMUM_CLASS_COUNT = 9;
const READBACK_INTERVAL_FRAMES = 6;
const ENCODED_FRAMES_AFTER_START = 2;

/** Slots (zone-table-sized columns) per statistics column inside the zone metrics buffer. */
const ZONE_SLOTS_PER_COLUMN = 12;
const ZONE_SLOT = {
  counts: 0,
  sumValues: 1,
  sums: 2,
  means: 4,
  minimums: 5,
  maximums: 6,
  standardDeviations: 7,
  medians: 8,
  percentiles: 9,
  modes: 10,
  uniqueCounts: 11
} as const;
const ZONE_ROWS_SLOT = ZONE_SLOTS_PER_COLUMN * 2;
const ZONE_SLOT_COUNT = ZONE_ROWS_SLOT + 1;

const COMPARE_SLOT = {
  presence: 0,
  before: 1,
  after: 2,
  delta: 3,
  ratio: 4,
  percentChange: 5,
  zScore: 6,
  zoneMedian: 7,
  zoneMean: 8,
  zoneRows: 9,
  matched: 10
} as const;
const COMPARE_SLOT_COUNT = 11;

type View = 'zones' | 'compare';
type ZoneStatistic =
  | 'rows'
  | 'sum'
  | 'mean'
  | 'median'
  | 'standardDeviation'
  | 'mode'
  | 'uniqueCount'
  | 'percentile'
  | 'minimum'
  | 'maximum';
type CompareMetric =
  | 'delta'
  | 'percentChange'
  | 'zScore'
  | 'ratio'
  | 'before'
  | 'after'
  | 'zoneMedian'
  | 'zoneMean'
  | 'zoneRows';

const ZONE_STATISTICS: Record<
  ZoneStatistic,
  {label: string; slot: (column: number) => number; unsigned: boolean}
> = {
  rows: {label: 'Count (rows per zone)', slot: () => ZONE_ROWS_SLOT, unsigned: true},
  sum: {
    label: 'Sum',
    slot: column => column * ZONE_SLOTS_PER_COLUMN + ZONE_SLOT.sumValues,
    unsigned: false
  },
  mean: {
    label: 'Mean',
    slot: column => column * ZONE_SLOTS_PER_COLUMN + ZONE_SLOT.means,
    unsigned: false
  },
  median: {
    label: 'Median',
    slot: column => column * ZONE_SLOTS_PER_COLUMN + ZONE_SLOT.medians,
    unsigned: false
  },
  standardDeviation: {
    label: 'Standard deviation',
    slot: column => column * ZONE_SLOTS_PER_COLUMN + ZONE_SLOT.standardDeviations,
    unsigned: false
  },
  mode: {
    label: 'Mode',
    slot: column => column * ZONE_SLOTS_PER_COLUMN + ZONE_SLOT.modes,
    unsigned: false
  },
  uniqueCount: {
    label: 'Count unique',
    slot: column => column * ZONE_SLOTS_PER_COLUMN + ZONE_SLOT.uniqueCounts,
    unsigned: true
  },
  percentile: {
    label: 'Percentile (slider)',
    slot: column => column * ZONE_SLOTS_PER_COLUMN + ZONE_SLOT.percentiles,
    unsigned: false
  },
  minimum: {
    label: 'Minimum',
    slot: column => column * ZONE_SLOTS_PER_COLUMN + ZONE_SLOT.minimums,
    unsigned: false
  },
  maximum: {
    label: 'Maximum',
    slot: column => column * ZONE_SLOTS_PER_COLUMN + ZONE_SLOT.maximums,
    unsigned: false
  }
};

const COMPARE_METRICS: Record<
  CompareMetric,
  {label: string; slot: number; unsigned: boolean; diverging: boolean}
> = {
  delta: {label: 'Delta (B - A)', slot: COMPARE_SLOT.delta, unsigned: false, diverging: true},
  percentChange: {
    label: 'Percent change',
    slot: COMPARE_SLOT.percentChange,
    unsigned: false,
    diverging: true
  },
  zScore: {label: 'Poisson z-score', slot: COMPARE_SLOT.zScore, unsigned: false, diverging: true},
  ratio: {label: 'Ratio (B / A)', slot: COMPARE_SLOT.ratio, unsigned: false, diverging: true},
  before: {label: 'Period A rows', slot: COMPARE_SLOT.before, unsigned: false, diverging: false},
  after: {label: 'Period B rows', slot: COMPARE_SLOT.after, unsigned: false, diverging: false},
  zoneMedian: {
    label: 'Zone median speed (key join)',
    slot: COMPARE_SLOT.zoneMedian,
    unsigned: false,
    diverging: false
  },
  zoneMean: {
    label: 'Zone mean speed (key join)',
    slot: COMPARE_SLOT.zoneMean,
    unsigned: false,
    diverging: false
  },
  zoneRows: {
    label: 'Zone rows (key join)',
    slot: COMPARE_SLOT.zoneRows,
    unsigned: true,
    diverging: false
  }
};

const COLUMN_NAMES = ['Speed (km/h)', 'Vendor (0/1)'] as const;

const SEQUENTIAL_PALETTES: Record<string, {label: string; stops: readonly (readonly number[])[]}> =
  {
    viridis: {
      label: 'Viridis',
      stops: [
        [68, 1, 84],
        [59, 82, 139],
        [33, 145, 140],
        [94, 201, 98],
        [253, 231, 37]
      ]
    },
    ylorrd: {
      label: 'Yellow orange red',
      stops: [
        [255, 255, 178],
        [254, 204, 92],
        [253, 141, 60],
        [240, 59, 32],
        [189, 0, 38]
      ]
    },
    blues: {
      label: 'Blues',
      stops: [
        [222, 235, 247],
        [158, 202, 225],
        [66, 146, 198],
        [8, 81, 156],
        [8, 48, 107]
      ]
    }
  };
const DIVERGING_STOPS: readonly (readonly number[])[] = [
  [33, 102, 172],
  [103, 169, 207],
  [209, 229, 240],
  [247, 247, 247],
  [253, 219, 199],
  [239, 138, 98],
  [178, 24, 43]
];
/** Cells that exist in only one period have no percent change or ratio: show them amber. */
const NO_DATA_COLOR = packColor(255, 176, 0, 170);

const METHOD_LABELS: Record<string, string> = {
  quantile: 'Quantile',
  'equal-interval': 'Equal interval',
  'natural-breaks': 'Natural breaks (Jenks)',
  'standard-deviation': 'Standard deviation'
};
const DISPLAY_METHODS = [
  'quantile',
  'equal-interval',
  'standard-deviation',
  'natural-breaks'
] as const;

/** Compact description of one classified display graph; see {@link createClassifiedDisplay}. */
type ClassifiedDisplay = {
  graph: CompiledGPUCommandGraph<void>;
  colors: Buffer;
  /** Parts a summary readback copies, in order: breaks, class count, class counts. */
  readbackParts: {buffer: Buffer; byteLength: number}[];
  setSelection: (slot: number, unsigned: boolean) => void;
  setClassification: (props: {
    method: GPUClassBreaksMethod;
    classCount: number;
    palette: Uint32Array;
  }) => void;
};

export const groupStatisticsMode: SpatialAnalysisModeDefinition = {
  id: 'group-statistics',
  title: 'Group stats',
  contributors: [
    'GPUPointToCell',
    'GPUGroupStatistics',
    'GPUKeyJoin',
    'GPUCellAggregation',
    'GPUCellTableCompare',
    'GPUClassBreaks',
    'GPUColorScale'
  ],
  description:
    'Per-zone statistics of New York trip vertices (Quadbin zones): pick a column and a ' +
    'statistic, drag the percentile, or switch to the period compare map (two time windows, ' +
    'delta / percent change / z-score) with zone statistics joined on by key. Every choice is a ' +
    'buffer write.',
  initialViewState: {longitude: -73.985, latitude: 40.735, zoom: 11.6},

  async create(context) {
    const trips = await context.data.getNewYorkTrips();
    context.signal.throwIfAborted();
    const {device} = context;
    const resources = new SpatialAnalysisResources(device, 'group-statistics');
    const pointCount = trips.vertexTimestamps.length;
    const [firstTime, lastTime] = trips.timeRange;

    // Quadbin keys need longitude/latitude; the dataset stores planar meters around its origin.
    const metricProjection = new LocalMetricProjection(trips.origin);
    const lngLat = new Float32Array(pointCount * 2);
    for (let point = 0; point < pointCount; point++) {
      const [longitude, latitude] = metricProjection.unproject(
        trips.vertexPositions[point * 2],
        trips.vertexPositions[point * 2 + 1]
      );
      lngLat[point * 2] = longitude;
      lngLat[point * 2 + 1] = latitude;
    }
    // Speed (whole km/h, so mode and unique count are meaningful) and vendor per vertex.
    const speeds = new Float32Array(pointCount);
    const vendors = new Float32Array(pointCount);
    for (let trip = 0; trip + 1 < trips.tripOffsets.length; trip++) {
      const first = trips.tripOffsets[trip];
      const last = trips.tripOffsets[trip + 1];
      let speed = 0;
      for (let vertex = first; vertex < last; vertex++) {
        if (vertex + 1 < last) {
          const seconds = trips.vertexTimestamps[vertex + 1] - trips.vertexTimestamps[vertex];
          if (seconds > 0) {
            const distance = Math.hypot(
              trips.vertexPositions[vertex * 2 + 2] - trips.vertexPositions[vertex * 2],
              trips.vertexPositions[vertex * 2 + 3] - trips.vertexPositions[vertex * 2 + 1]
            );
            speed = Math.min(Math.round((distance / seconds) * 3.6), 120);
          }
        }
        speeds[vertex] = speed;
        vendors[vertex] = trips.vendors[trip];
      }
    }

    // Inputs.
    const lngLatBuffer = resources.createBuffer('lng-lat', lngLat);
    const positionsBuffer = resources.createBuffer('positions', trips.vertexPositions);
    const speedsBuffer = resources.createBuffer('speeds', speeds);
    const vendorsBuffer = resources.createBuffer('vendors', vendors);
    const maskValuesA = new Uint32Array(pointCount);
    const maskValuesB = new Uint32Array(pointCount);
    const maskBufferA = resources.createBuffer('mask-a', maskValuesA);
    const maskBufferB = resources.createBuffer('mask-b', maskValuesB);
    const percentileParameters = resources.createParameterBuffer(
      'percentile',
      'float32',
      1,
      Float32Array.of(0.9)
    );

    // Zone table (graph "zones").
    const vertexKeys = resources.createBuffer('vertex-keys', pointCount * 8);
    const zoneKeys = resources.createBuffer('zone-keys', ZONE_CAPACITY * 8);
    const zoneMetrics = resources.createBuffer('zone-metrics', ZONE_SLOT_COUNT * ZONE_CAPACITY * 4);
    const zoneCount = resources.createBuffer('zone-count', 4);
    const zoneOverflow = resources.createBuffer('zone-overflow', 4);
    const zoneTotal = resources.createBuffer('zone-total', 4);
    const zScores = resources.createBuffer('z-scores', pointCount * 4);

    // Period tables and the compare table (graph "compare").
    const tableBuffers = ['a', 'b'].map(name => ({
      cells: resources.createBuffer(`table-${name}-cells`, ZONE_CAPACITY * 8),
      counts: resources.createBuffer(`table-${name}-counts`, ZONE_CAPACITY * 4),
      count: resources.createBuffer(`table-${name}-count`, 4),
      overflow: resources.createBuffer(`table-${name}-overflow`, 4),
      total: resources.createBuffer(`table-${name}-total`, 4)
    }));
    const compareCells = resources.createBuffer('compare-cells', COMPARE_CAPACITY * 8);
    const compareMetrics = resources.createBuffer(
      'compare-metrics',
      COMPARE_SLOT_COUNT * COMPARE_CAPACITY * 4
    );
    const compareOutputs = Object.fromEntries(
      ['presence', 'before', 'after', 'delta', 'ratio', 'percentChange', 'zScore'].map(name => [
        name,
        resources.createBuffer(`compare-${name}`, COMPARE_CAPACITY * 4)
      ])
    ) as Record<
      'presence' | 'before' | 'after' | 'delta' | 'ratio' | 'percentChange' | 'zScore',
      Buffer
    >;
    const compareCount = resources.createBuffer('compare-count', 4);
    const compareOverflow = resources.createBuffer('compare-overflow', 4);
    const compareTotal = resources.createBuffer('compare-total', 4);

    /**
     * One classified display: select a metric slot, classify it, color it. It owns its parameter
     * and output buffers and one compiled graph.
     */
    function createClassifiedDisplay(props: {
      id: string;
      capacity: number;
      metrics: Buffer;
      count: Buffer;
    }): ClassifiedDisplay {
      const {id, capacity} = props;
      const selection = resources.createParameterBuffer(`${id}-selection`, 'uint32', 2);
      const breaksParameters = resources.createParameterBuffer(
        `${id}-breaks-parameters`,
        'float32',
        getGPUClassBreaksParameterLength(MAXIMUM_CLASS_COUNT)
      );
      const scaleParameters = resources.createParameterBuffer(
        `${id}-scale-parameters`,
        'float32',
        GPU_COLOR_SCALE_PARAMETER_LENGTH
      );
      const palette = resources.createBuffer(`${id}-palette`, MAXIMUM_CLASS_COUNT * 4);
      const display = resources.createBuffer(`${id}-display`, capacity * 4);
      const colors = resources.createBuffer(`${id}-colors`, capacity * 4);
      const breaks = resources.createBuffer(`${id}-breaks`, (MAXIMUM_CLASS_COUNT + 1) * 4);
      const classCount = resources.createBuffer(`${id}-class-count`, 4);
      const classCounts = resources.createBuffer(`${id}-class-counts`, MAXIMUM_CLASS_COUNT * 4);

      const graph = new GPUCommandGraph<void>(device, {id: `${id}-display`});
      const displayValues = importGraphBuffer(graph, 'io-display', display, 'float32', capacity);
      addMetricSelectPass(graph, {
        id: `${id}-select`,
        capacity,
        metrics: importGraphBuffer(
          graph,
          'io-metrics',
          props.metrics,
          'uint32',
          props.metrics.byteLength / 4
        ),
        selection: selection.importToGraph(graph),
        count: importGraphBuffer(graph, 'io-count', props.count, 'uint32', 1),
        display: displayValues
      });
      const breaksView = importGraphBuffer(
        graph,
        'io-breaks',
        breaks,
        'float32',
        MAXIMUM_CLASS_COUNT + 1
      );
      const classCountView = importGraphBuffer(graph, 'io-class-count', classCount, 'uint32', 1);
      graph.add(
        new GPUClassBreaks({
          id: `${id}-class-breaks`,
          values: displayValues,
          parameters: breaksParameters.importToGraph(graph),
          maximumClassCount: MAXIMUM_CLASS_COUNT,
          methods: DISPLAY_METHODS,
          naturalBreaksBinCount: 256,
          output: {breaks: breaksView, classCount: classCountView}
        })
      );
      graph.add(
        new GPUColorScale({
          id: `${id}-color-scale`,
          values: displayValues,
          domain: breaksView,
          domainCount: classCountView,
          palette: importGraphBuffer(graph, 'io-palette', palette, 'uint32', MAXIMUM_CLASS_COUNT),
          parameters: scaleParameters.importToGraph(graph),
          maximumDomainCount: MAXIMUM_CLASS_COUNT + 1,
          maximumPaletteCount: MAXIMUM_CLASS_COUNT,
          output: {
            colors: importGraphBuffer(graph, 'io-colors', colors, 'uint32', capacity),
            classCounts: importGraphBuffer(
              graph,
              'io-class-counts',
              classCounts,
              'uint32',
              MAXIMUM_CLASS_COUNT
            )
          }
        })
      );
      const compiled = resources.track(graph.compile());
      return {
        graph: compiled,
        colors,
        readbackParts: [
          {buffer: breaks, byteLength: (MAXIMUM_CLASS_COUNT + 1) * 4},
          {buffer: classCount, byteLength: 4},
          {buffer: classCounts, byteLength: MAXIMUM_CLASS_COUNT * 4}
        ],
        setSelection: (slot, unsigned) => selection.write(Uint32Array.of(slot, unsigned ? 1 : 0)),
        setClassification: ({method, classCount: requestedCount, palette: colorsToWrite}) => {
          breaksParameters.write(
            getGPUClassBreaksParameterValues(
              {method, classCount: requestedCount},
              MAXIMUM_CLASS_COUNT
            )
          );
          const padded = new Uint32Array(MAXIMUM_CLASS_COUNT);
          padded.set(colorsToWrite);
          palette.write(padded);
          scaleParameters.write(
            getGPUColorScaleParameterValues({
              scale: 'threshold',
              domainCount: requestedCount + 1,
              paletteCount: requestedCount,
              clamp: true,
              noDataColor: NO_DATA_COLOR
            })
          );
        }
      };
    }

    // Zone graph: key every vertex, then group statistics per zone.
    const zoneGraph = new GPUCommandGraph<void>(device, {id: 'zone-statistics'});
    {
      const keys = importGraphBuffer(
        zoneGraph,
        'io-vertex-keys',
        vertexKeys,
        'uint32x2',
        pointCount
      );
      zoneGraph.add(
        new GPUPointToCell({
          id: 'zone-cell-keys',
          family: 'quadbin',
          resolution: ZONE_RESOLUTION,
          positions: importGraphBuffer(
            zoneGraph,
            'io-lng-lat',
            lngLatBuffer,
            'float32x2',
            pointCount
          ),
          output: {cells: keys}
        })
      );
      const slices = new SlotSlices(zoneGraph, zoneMetrics, ZONE_CAPACITY);
      const columnValues = [
        importGraphBuffer(zoneGraph, 'io-speeds', speedsBuffer, 'float32', pointCount),
        importGraphBuffer(zoneGraph, 'io-vendors', vendorsBuffer, 'float32', pointCount)
      ];
      const statistics: GPUGroupStatistic[] = [
        'count',
        'sum',
        'mean',
        'minimum',
        'maximum',
        'standardDeviation',
        'median',
        'percentiles',
        'mode',
        'uniqueCount'
      ];
      const columns = columnValues.map((values, column) => {
        const base = column * ZONE_SLOTS_PER_COLUMN;
        return {
          values,
          statistics: column === 0 ? [...statistics, 'zScore' as const] : statistics,
          output: {
            counts: slices.view(base + ZONE_SLOT.counts, 'uint32'),
            sumValues: slices.view(base + ZONE_SLOT.sumValues, 'float32'),
            sums: slices.view(base + ZONE_SLOT.sums, 'uint32x2'),
            means: slices.view(base + ZONE_SLOT.means, 'float32'),
            minimums: slices.view(base + ZONE_SLOT.minimums, 'float32'),
            maximums: slices.view(base + ZONE_SLOT.maximums, 'float32'),
            standardDeviations: slices.view(base + ZONE_SLOT.standardDeviations, 'float32'),
            medians: slices.view(base + ZONE_SLOT.medians, 'float32'),
            percentiles: slices.view(base + ZONE_SLOT.percentiles, 'float32'),
            modes: slices.view(base + ZONE_SLOT.modes, 'float32'),
            uniqueCounts: slices.view(base + ZONE_SLOT.uniqueCounts, 'uint32'),
            ...(column === 0
              ? {
                  zScores: importGraphBuffer(
                    zoneGraph,
                    'io-z-scores',
                    zScores,
                    'float32',
                    pointCount
                  )
                }
              : {})
          }
        };
      });
      zoneGraph.add(
        new GPUGroupStatistics({
          id: 'zones',
          keys,
          columns,
          variance: 'sample',
          percentiles: percentileParameters.importToGraph(zoneGraph),
          output: {
            keys: importGraphBuffer(zoneGraph, 'io-zone-keys', zoneKeys, 'uint32x2', ZONE_CAPACITY),
            counts: slices.view(ZONE_ROWS_SLOT, 'uint32'),
            count: importGraphBuffer(zoneGraph, 'io-zone-count', zoneCount, 'uint32', 1),
            overflow: importGraphBuffer(zoneGraph, 'io-zone-overflow', zoneOverflow, 'uint32', 1),
            requiredCount: importGraphBuffer(zoneGraph, 'io-zone-total', zoneTotal, 'uint32', 1)
          }
        })
      );
    }
    const zonesCompiled = resources.track(zoneGraph.compile());
    const zoneDisplay = createClassifiedDisplay({
      id: 'zones',
      capacity: ZONE_CAPACITY,
      metrics: zoneMetrics,
      count: zoneCount
    });

    // Compare graph: two period aggregations, the compare join, then the zone-stat key join.
    const compareGraph = new GPUCommandGraph<void>(device, {id: 'period-compare'});
    {
      const positions = importGraphBuffer(
        compareGraph,
        'io-lng-lat',
        lngLatBuffer,
        'float32x2',
        pointCount
      );
      const tables = tableBuffers.map((table, index) => ({
        cells: importGraphBuffer(
          compareGraph,
          table.cells.id,
          table.cells,
          'uint32x2',
          ZONE_CAPACITY
        ),
        counts: importGraphBuffer(
          compareGraph,
          table.counts.id,
          table.counts,
          'uint32',
          ZONE_CAPACITY
        ),
        count: importGraphBuffer(compareGraph, table.count.id, table.count, 'uint32', 1),
        overflow: importGraphBuffer(compareGraph, table.overflow.id, table.overflow, 'uint32', 1),
        requiredCount: importGraphBuffer(compareGraph, table.total.id, table.total, 'uint32', 1),
        mask: importGraphBuffer(
          compareGraph,
          `mask-${index ? 'b' : 'a'}`,
          index ? maskBufferB : maskBufferA,
          'uint32',
          pointCount
        )
      }));
      tables.forEach(({mask, ...table}, index) => {
        compareGraph.add(
          new GPUCellAggregation({
            id: `period-${index ? 'b' : 'a'}`,
            family: 'quadbin',
            resolution: ZONE_RESOLUTION,
            positions,
            mask,
            output: table
          })
        );
      });
      const slices = new SlotSlices(compareGraph, compareMetrics, COMPARE_CAPACITY);
      const unionCells = importGraphBuffer(
        compareGraph,
        'io-compare-cells',
        compareCells,
        'uint32x2',
        COMPARE_CAPACITY
      );
      const outputViews = Object.fromEntries(
        (['before', 'after', 'delta', 'ratio', 'percentChange', 'zScore'] as const).map(name => [
          name,
          importGraphBuffer(
            compareGraph,
            `io-compare-${name}`,
            compareOutputs[name],
            'float32',
            COMPARE_CAPACITY
          )
        ])
      ) as Record<
        'before' | 'after' | 'delta' | 'ratio' | 'percentChange' | 'zScore',
        GraphDataView<'float32'>
      >;
      compareGraph.add(
        new GPUCellTableCompare({
          id: 'compare',
          before: {...tables[0]},
          after: {...tables[1]},
          measure: 'count',
          zScore: 'poisson',
          output: {
            cells: unionCells,
            presence: importGraphBuffer(
              compareGraph,
              'io-compare-presence',
              compareOutputs.presence,
              'uint32',
              COMPARE_CAPACITY
            ),
            before: outputViews.before,
            after: outputViews.after,
            delta: outputViews.delta,
            ratio: outputViews.ratio,
            percentChange: outputViews.percentChange,
            zScore: outputViews.zScore,
            count: importGraphBuffer(compareGraph, 'io-compare-count', compareCount, 'uint32', 1),
            overflow: importGraphBuffer(
              compareGraph,
              'io-compare-overflow',
              compareOverflow,
              'uint32',
              1
            ),
            requiredCount: importGraphBuffer(
              compareGraph,
              'io-compare-total',
              compareTotal,
              'uint32',
              1
            )
          }
        })
      );
      addSlotPackPass(compareGraph, {
        id: 'compare-pack',
        capacity: COMPARE_CAPACITY,
        sources: [
          outputViews.before,
          outputViews.after,
          outputViews.delta,
          outputViews.ratio,
          outputViews.percentChange,
          outputViews.zScore
        ],
        destination: slices.whole(COMPARE_SLOT_COUNT * COMPARE_CAPACITY),
        firstSlot: COMPARE_SLOT.before
      });
      const zoneSlices = new SlotSlices(compareGraph, zoneMetrics, ZONE_CAPACITY);
      compareGraph.add(
        new GPUKeyJoin({
          id: 'zone-join',
          leftKeys: unionCells,
          rightKeys: importGraphBuffer(
            compareGraph,
            'io-zone-keys',
            zoneKeys,
            'uint32x2',
            ZONE_CAPACITY
          ),
          kind: 'left',
          gather: [
            {
              column: zoneSlices.view(ZONE_SLOT.medians, 'float32'),
              output: slices.view(COMPARE_SLOT.zoneMedian, 'float32')
            },
            {
              column: zoneSlices.view(ZONE_SLOT.means, 'float32'),
              output: slices.view(COMPARE_SLOT.zoneMean, 'float32')
            },
            {
              column: zoneSlices.view(ZONE_ROWS_SLOT, 'uint32'),
              output: slices.view(COMPARE_SLOT.zoneRows, 'uint32')
            }
          ],
          output: {matched: slices.view(COMPARE_SLOT.matched, 'uint32')}
        })
      );
    }
    const compareCompiled = resources.track(compareGraph.compile());
    const compareDisplay = createClassifiedDisplay({
      id: 'compare',
      capacity: COMPARE_CAPACITY,
      metrics: compareMetrics,
      count: compareCount
    });

    // Summary readback: tables' counts, both displays' breaks and class counts, matched flags.
    const summaryParts: {buffer: Buffer; byteLength: number}[] = [
      {buffer: zoneCount, byteLength: 4},
      {buffer: zoneOverflow, byteLength: 4},
      {buffer: zoneTotal, byteLength: 4},
      {buffer: compareCount, byteLength: 4},
      {buffer: compareOverflow, byteLength: 4},
      {buffer: compareTotal, byteLength: 4},
      ...zoneDisplay.readbackParts,
      ...compareDisplay.readbackParts
    ];
    const matchedByteOffset = summaryParts.reduce((total, part) => total + part.byteLength, 0);
    const readbackByteLength = matchedByteOffset + COMPARE_CAPACITY * 4;
    const readbackRing = resources.track(
      new GPUReadbackRing(device, {id: 'group-statistics-summary', byteLength: readbackByteLength})
    );

    // Control state.
    let view: View = 'zones';
    let columnIndex = 0;
    let zoneStatistic: ZoneStatistic = 'median';
    let percentile = 90;
    let compareMetric: CompareMetric = 'delta';
    let zoneMethod: GPUClassBreaksMethod = 'quantile';
    let classCount = 7;
    let paletteKey = 'viridis';
    let windowA: [number, number] = [0, 50];
    let windowB: [number, number] = [50, 100];
    let showOutliers = false;
    let outlierThreshold = 2;
    let zoneDirty = true;
    let zoneDisplayDirty = true;
    let compareDirty = true;
    let compareDisplayDirty = true;
    let readbackWanted = true;
    let readbackPending = false;
    let lastReadbackFrame = -READBACK_INTERVAL_FRAMES;
    let destroyed = false;
    let latestSummary: {
      zoneCount: number;
      zoneOverflow: number;
      zoneTotal: number;
      compareCount: number;
      compareOverflow: number;
      compareTotal: number;
      zoneBreaks: Float32Array;
      zoneClassCount: number;
      zoneClassCounts: Uint32Array;
      compareBreaks: Float32Array;
      compareClassCount: number;
      compareClassCounts: Uint32Array;
      matchedCount: number;
    } | null = null;

    const zonePalette = (): Uint32Array =>
      getRampPalette(SEQUENTIAL_PALETTES[paletteKey].stops, classCount, 225);

    function writeWindows(): void {
      const span = lastTime - firstTime;
      const counts: [number, number] = [0, 0];
      [windowA, windowB].forEach((window, index) => {
        const values = index === 0 ? maskValuesA : maskValuesB;
        const start = firstTime + (span * window[0]) / 100;
        const end = firstTime + (span * window[1]) / 100;
        for (let point = 0; point < pointCount; point++) {
          const time = trips.vertexTimestamps[point];
          const inside = time >= start && (time < end || (window[1] >= 100 && time <= end));
          values[point] = inside ? 1 : 0;
          counts[index] += inside ? 1 : 0;
        }
        (index === 0 ? maskBufferA : maskBufferB).write(values);
      });
      periodReadout.setValue(`${formatCount(counts[0])} / ${formatCount(counts[1])}`);
      compareDirty = true;
      compareDisplayDirty = true;
      readbackWanted = true;
    }

    function writeZoneDisplay(): void {
      const statistic = ZONE_STATISTICS[zoneStatistic];
      zoneDisplay.setSelection(statistic.slot(columnIndex), statistic.unsigned);
      zoneDisplay.setClassification({method: zoneMethod, classCount, palette: zonePalette()});
      zoneDisplayDirty = true;
      readbackWanted = true;
    }

    function writeCompareDisplay(): void {
      const metric = COMPARE_METRICS[compareMetric];
      compareDisplay.setSelection(metric.slot, metric.unsigned);
      if (metric.diverging) {
        // Seven standard-deviation classes centered on the mean give a symmetric scale.
        compareDisplay.setClassification({
          method: 'standard-deviation',
          classCount: 7,
          palette: getRampPalette(DIVERGING_STOPS, 7, 225)
        });
      } else {
        compareDisplay.setClassification({
          method: zoneMethod === 'standard-deviation' ? 'quantile' : zoneMethod,
          classCount,
          palette: zonePalette()
        });
      }
      compareDisplayDirty = true;
      readbackWanted = true;
    }

    function writeAll(): void {
      writeZoneDisplay();
      writeCompareDisplay();
    }

    // Controls.
    context.controls.addSelect<View>({
      label: 'View',
      options: [
        {value: 'zones', label: 'Zone statistics (GPUGroupStatistics)'},
        {value: 'compare', label: 'Period compare (GPUCellTableCompare + GPUKeyJoin)'}
      ],
      value: view,
      onChange: value => {
        view = value;
        readbackWanted = true;
        context.updateLayers();
        renderLegend();
      }
    });
    context.controls.addSelect<string>({
      label: 'Zones: value column',
      options: COLUMN_NAMES.map((label, index) => ({value: String(index), label})),
      value: String(columnIndex),
      onChange: value => {
        columnIndex = Number(value);
        writeZoneDisplay();
      }
    });
    context.controls.addSelect<ZoneStatistic>({
      label: 'Zones: statistic (selection buffer)',
      options: (Object.keys(ZONE_STATISTICS) as ZoneStatistic[]).map(key => ({
        value: key,
        label: ZONE_STATISTICS[key].label
      })),
      value: zoneStatistic,
      onChange: value => {
        zoneStatistic = value;
        percentileControl.setDisabled(value !== 'percentile');
        writeZoneDisplay();
      }
    });
    const percentileControl = context.controls.addSlider({
      label: 'Zones: percentile (re-encodes the statistics)',
      min: 1,
      max: 99,
      step: 1,
      value: percentile,
      format: value => `p${value}`,
      onChange: value => {
        percentile = value;
        percentileParameters.write(Float32Array.of(percentile / 100));
        zoneDirty = true;
        zoneDisplayDirty = true;
        readbackWanted = true;
      }
    });
    percentileControl.setDisabled(true);
    context.controls.addSelect<CompareMetric>({
      label: 'Compare: metric (selection buffer)',
      options: (Object.keys(COMPARE_METRICS) as CompareMetric[]).map(key => ({
        value: key,
        label: COMPARE_METRICS[key].label
      })),
      value: compareMetric,
      onChange: value => {
        compareMetric = value;
        writeCompareDisplay();
        renderLegend();
      }
    });
    const windowLabel = (value: number) => `${value}%`;
    const windowControls: {label: string; get: () => number; set: (value: number) => void}[] = [
      {
        label: 'Period A start',
        get: () => windowA[0],
        set: value => (windowA = [value, windowA[1]])
      },
      {label: 'Period A end', get: () => windowA[1], set: value => (windowA = [windowA[0], value])},
      {
        label: 'Period B start',
        get: () => windowB[0],
        set: value => (windowB = [value, windowB[1]])
      },
      {label: 'Period B end', get: () => windowB[1], set: value => (windowB = [windowB[0], value])}
    ];
    for (const control of windowControls) {
      context.controls.addSlider({
        label: `Compare: ${control.label} (mask buffer)`,
        min: 0,
        max: 100,
        step: 1,
        value: control.get(),
        format: windowLabel,
        onChange: value => {
          control.set(value);
          writeWindows();
        }
      });
    }
    context.controls.addSelect<GPUClassBreaksMethod>({
      label: 'Class method (per-frame code)',
      options: DISPLAY_METHODS.map(value => ({value, label: METHOD_LABELS[value]})),
      value: zoneMethod,
      onChange: value => {
        zoneMethod = value;
        writeAll();
      }
    });
    context.controls.addSlider({
      label: `Classes (up to ${MAXIMUM_CLASS_COUNT} compiled)`,
      min: 2,
      max: MAXIMUM_CLASS_COUNT,
      step: 1,
      value: classCount,
      onChange: value => {
        classCount = value;
        writeAll();
      }
    });
    context.controls.addSelect<string>({
      label: 'Palette (sequential metrics)',
      options: Object.entries(SEQUENTIAL_PALETTES).map(([value, {label}]) => ({value, label})),
      value: paletteKey,
      onChange: value => {
        paletteKey = value;
        writeAll();
      }
    });
    context.controls.addToggle({
      label: 'Zones: show speed outlier vertices (per-row z-score)',
      value: showOutliers,
      onChange: value => {
        showOutliers = value;
        context.updateLayers();
      }
    });
    context.controls.addSlider({
      label: 'Zones: outlier z-score threshold',
      min: 1,
      max: 4,
      step: 0.5,
      value: outlierThreshold,
      onChange: value => {
        outlierThreshold = value;
        context.updateLayers();
      }
    });
    context.controls.addNote(
      'Zone statistics are computed once and again only when the percentile moves; the ' +
        'statistic, column, classes and palette rewrite a selection buffer and re-run the small ' +
        'select, break and color graph. Cells present in one period only show amber for ratio and ' +
        'percent change.'
    );
    const legendBlock = createPanelBlock();
    context.controls.addReadout('Vertices', formatCount(pointCount));
    context.controls.addReadout(
      'Zone size',
      `Quadbin ${ZONE_RESOLUTION} (~${((40075.017 / 2 ** ZONE_RESOLUTION) * Math.cos((40.73 * Math.PI) / 180)).toFixed(2)} km)`
    );
    const zoneReadout = context.controls.addReadout('Zones');
    const periodReadout = context.controls.addReadout('Period A / B vertices');
    const compareReadout = context.controls.addReadout('Compare cells');
    const joinReadout = context.controls.addReadout('Key-joined cells');
    const rangeReadout = context.controls.addReadout('Value range');
    context.controls.addReadout('Data', trips.attribution);

    writeWindows();
    percentileParameters.write(Float32Array.of(percentile / 100));
    writeAll();

    function renderLegend(): void {
      if (!legendBlock) return;
      const summary = latestSummary;
      if (!summary) return;
      const zones = view === 'zones';
      const metricLabel = zones
        ? `${COLUMN_NAMES[columnIndex]} ${ZONE_STATISTICS[zoneStatistic].label.toLowerCase()}`
        : COMPARE_METRICS[compareMetric].label;
      const k = Math.min(
        zones ? summary.zoneClassCount : summary.compareClassCount,
        MAXIMUM_CLASS_COUNT
      );
      const breaks = zones ? summary.zoneBreaks : summary.compareBreaks;
      const counts = zones ? summary.zoneClassCounts : summary.compareClassCounts;
      const palette =
        zones || !COMPARE_METRICS[compareMetric].diverging
          ? zonePalette()
          : getRampPalette(DIVERGING_STOPS, 7, 225);
      legendBlock.innerHTML = getClassLegendHtml({
        heading: `${metricLabel}: zones per class`,
        edges: breaks,
        counts,
        palette,
        classCount: k,
        noDataLabel:
          !zones && (compareMetric === 'percentChange' || compareMetric === 'ratio')
            ? 'amber = cell absent from period A (ratio and percent change are undefined)'
            : undefined
      });
      rangeReadout.setValue(`${formatCompact(breaks[0])} – ${formatCompact(breaks[k])}`);
    }

    function applySummary(words: Uint32Array, floats: Float32Array): void {
      let cursor = 0;
      const nextWord = () => words[cursor++];
      const zoneCountValue = nextWord();
      const zoneOverflowValue = nextWord();
      const zoneTotalValue = nextWord();
      const compareCountValue = nextWord();
      const compareOverflowValue = nextWord();
      const compareTotalValue = nextWord();
      const readDisplay = () => {
        const breaks = floats.slice(cursor, cursor + MAXIMUM_CLASS_COUNT + 1);
        cursor += MAXIMUM_CLASS_COUNT + 1;
        const count = words[cursor++];
        const counts = words.slice(cursor, cursor + MAXIMUM_CLASS_COUNT);
        cursor += MAXIMUM_CLASS_COUNT;
        return {breaks, count, counts};
      };
      const zonesDisplay = readDisplay();
      const comparesDisplay = readDisplay();
      let matchedCount = 0;
      const matchedStart = matchedByteOffset / 4;
      for (let row = 0; row < Math.min(compareCountValue, COMPARE_CAPACITY); row++) {
        matchedCount += words[matchedStart + row] ? 1 : 0;
      }
      latestSummary = {
        zoneCount: zoneCountValue,
        zoneOverflow: zoneOverflowValue,
        zoneTotal: zoneTotalValue,
        compareCount: compareCountValue,
        compareOverflow: compareOverflowValue,
        compareTotal: compareTotalValue,
        zoneBreaks: zonesDisplay.breaks,
        zoneClassCount: zonesDisplay.count,
        zoneClassCounts: zonesDisplay.counts,
        compareBreaks: comparesDisplay.breaks,
        compareClassCount: comparesDisplay.count,
        compareClassCounts: comparesDisplay.counts,
        matchedCount
      };
      zoneReadout.setValue(
        `${formatCount(zoneCountValue)} of ${formatCount(zoneTotalValue)}${zoneOverflowValue ? ' (overflow)' : ''}`
      );
      compareReadout.setValue(
        `${formatCount(compareCountValue)} of ${formatCount(compareTotalValue)}${compareOverflowValue ? ' (overflow)' : ''}`
      );
      joinReadout.setValue(`${formatCount(matchedCount)} of ${formatCount(compareCountValue)}`);
      renderLegend();
    }

    async function readSummary(commandEncoder: CommandEncoder): Promise<void> {
      const ticket = readbackRing.tryAcquire();
      if (!ticket) return;
      let offset = 0;
      for (const part of summaryParts) {
        commandEncoder.copyBufferToBuffer({
          sourceBuffer: part.buffer,
          destinationBuffer: ticket.buffer,
          destinationOffset: offset,
          size: part.byteLength
        });
        offset += part.byteLength;
      }
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: compareMetrics,
        sourceOffset: COMPARE_SLOT.matched * COMPARE_CAPACITY * 4,
        destinationBuffer: ticket.buffer,
        destinationOffset: matchedByteOffset,
        size: COMPARE_CAPACITY * 4
      });
      ticket.markEncoded({byteOffset: 0, byteLength: readbackByteLength});
      readbackPending = true;
      try {
        const bytes = await ticket.read();
        if (destroyed) return;
        const copy = new ArrayBuffer(bytes.byteLength);
        new Uint8Array(copy).set(bytes);
        applySummary(new Uint32Array(copy), new Float32Array(copy));
      } catch {
        // The ring or device was destroyed while the read was in flight.
      } finally {
        readbackPending = false;
      }
    }

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [
        zonesCompiled,
        zoneDisplay.graph,
        compareCompiled,
        compareDisplay.graph
      ],
      encode(commandEncoder, frame) {
        // Each graph depends only on buffers that change on a control event, so each is encoded
        // only when its inputs changed (and for the first frames). The percentile moves the
        // zone statistics; the windows move the compare tables; everything else only the small
        // display graphs.
        const initial = frame.frameIndex < ENCODED_FRAMES_AFTER_START;
        if (zoneDirty || initial) {
          zonesCompiled.encode(commandEncoder, {parameters: undefined});
          zoneDirty = false;
          zoneDisplayDirty = true;
          readbackWanted = true;
        }
        if (zoneDisplayDirty || initial) {
          zoneDisplay.graph.encode(commandEncoder, {parameters: undefined});
          zoneDisplayDirty = false;
          readbackWanted = true;
        }
        if (compareDirty || initial) {
          compareCompiled.encode(commandEncoder, {parameters: undefined});
          compareDirty = false;
          compareDisplayDirty = true;
          readbackWanted = true;
        }
        if (compareDisplayDirty || initial) {
          compareDisplay.graph.encode(commandEncoder, {parameters: undefined});
          compareDisplayDirty = false;
          readbackWanted = true;
        }
        if (
          readbackWanted &&
          !readbackPending &&
          frame.frameIndex - lastReadbackFrame >= READBACK_INTERVAL_FRAMES
        ) {
          lastReadbackFrame = frame.frameIndex;
          readbackWanted = false;
          void readSummary(commandEncoder);
        }
      },
      getLayers() {
        const coordinateOrigin: [number, number, number] = [trips.origin[0], trips.origin[1], 0];
        const layers: Layer[] =
          view === 'zones'
            ? [
                new PackedQuadbinCellLayer({
                  id: 'group-statistics-zones',
                  cells: zoneKeys,
                  colors: zoneDisplay.colors,
                  count: zoneCount,
                  capacity: ZONE_CAPACITY
                })
              ]
            : [
                new PackedQuadbinCellLayer({
                  id: 'group-statistics-compare',
                  cells: compareCells,
                  colors: compareDisplay.colors,
                  count: compareCount,
                  capacity: COMPARE_CAPACITY
                })
              ];
        if (showOutliers && view === 'zones') {
          layers.push(
            new SpatialAnalysisPointLayer({
              id: 'group-statistics-outliers',
              coordinateOrigin,
              positions: positionsBuffer,
              instanceCount: pointCount,
              radiusPixels: 2.6,
              values: zScores,
              valueFormat: 'float32',
              colormap: 'inferno',
              valueRange: [outlierThreshold, outlierThreshold + 3],
              discardAtOrBelow: outlierThreshold,
              noDataColor: [0, 0, 0, 0]
            })
          );
        }
        return layers;
      },
      destroy() {
        destroyed = true;
        legendBlock?.remove();
        resources.destroy();
      }
    };
    return instance;
  }
};
