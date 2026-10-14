// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getGPUClassBreaksParameterLength,
  getGPUColumnQuantilesParameterLength,
  getGPUColumnProfileParameterLength,
  GPUBivariateClassification,
  GPUCalendarBuckets,
  GPUClassBreaks,
  GPUColorScale,
  GPUColumnProfile,
  GPUColumnQuantiles,
  GPUCompositeScore,
  GPUGroupStatistics,
  GPUInequality,
  GPUKeyJoin,
  GPUPointPatternIndices,
  GPURipley,
  GPURipleyDistanceFunctions,
  GPURipleyEnvelope,
  GPUSpatialCorrelogram,
  GPUTemporalReduction,
  GPUTimeWindowFilter,
  GPUVariogram,
  GPU_BIVARIATE_CLASSIFICATION_PARAMETER_LENGTH,
  GPU_CALENDAR_BUCKETS_MATRIX_LENGTH,
  GPU_CALENDAR_BUCKETS_PARAMETER_LENGTH,
  GPU_CLARK_EVANS_LENGTH,
  GPU_CLASS_BREAKS_METHODS,
  GPU_COLOR_SCALE_PARAMETER_LENGTH,
  getGPUColumnProfileMaximumCategoryTotal,
  GPU_COLUMN_PROFILE_MAXIMUM_COLUMN_COUNT,
  GPU_COLUMN_PROFILE_MAXIMUM_TOP_CATEGORY_COUNT,
  GPU_COLUMN_PROFILE_STATISTIC_COUNT,
  GPU_COLUMN_QUANTILES_MAXIMUM_QUANTILE_COUNT,
  GPU_COMPOSITE_SCORE_COLUMN_STATISTICS_STRIDE,
  GPU_COMPOSITE_SCORE_MAXIMUM_INDICATOR_COUNT,
  GPU_COMPOSITE_SCORE_PARAMETER_LENGTH,
  GPU_COMPOSITE_SCORE_PRINCIPAL_COMPONENT_SUMMARY_LENGTH,
  GPU_INEQUALITY_GLOBAL_SUMMARY_LENGTH,
  getGPUInequalityMaximumLorenzKnotTotal,
  getGPUInequalityMaximumZoneCount,
  GPU_INEQUALITY_PARAMETER_LENGTH,
  GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH,
  GPU_QUADRAT_MAXIMUM_COUNT,
  GPU_QUADRAT_STATISTICS_LENGTH,
  GPU_RIPLEY_DISTANCE_FUNCTIONS_PARAMETER_LENGTH,
  GPU_RIPLEY_ENVELOPE_MAXIMUM_SIMULATION_COUNT,
  GPU_RIPLEY_ENVELOPE_STATISTICS_LENGTH,
  GPU_RIPLEY_PARAMETER_LENGTH,
  GPU_SPATIAL_CORRELOGRAM_PARAMETER_LENGTH,
  GPU_SPATIAL_CORRELOGRAM_STATISTICS_LENGTH,
  GPU_TEMPORAL_REDUCTION_PARAMETER_LENGTH,
  GPU_TEMPORAL_REDUCTION_WORD_PARAMETER_LENGTH,
  GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH,
  GPU_TIME_WINDOW_FILTER_WORD_PARAMETER_LENGTH,
  getGPUGroupStatisticsMaximumGroupCount,
  getGPUPairStatisticsMaximumRowCount,
  GPU_VARIOGRAM_PARAMETER_LENGTH,
  GPU_VARIOGRAM_STATISTICS_LENGTH,
  type GPUClassBreaksMethod,
  type GPUGroupStatistic
} from '../../../src/gpu-dataframe';
import type {KernelAuditCase, KernelAuditContext} from './kernel-audit-context';

const ROWS = 300;
/** Point count the pair cases build with, set by {@link withPairRowCount} around each build. */
let pairMaximumRows = 0;

/**
 * Wraps pair-statistics cases so they build with `offset` rows above
 * {@link getGPUPairStatisticsMaximumRowCount} of the context device.
 */
function withPairRowCount(
  cases: readonly KernelAuditCase[],
  offset: number,
  suffix: string = ''
): KernelAuditCase[] {
  return cases.map(
    (auditCase): KernelAuditCase => ({
      exportName: auditCase.exportName,
      name: `${auditCase.name}${suffix}`,
      build: context => {
        pairMaximumRows = getGPUPairStatisticsMaximumRowCount(context.device) + offset;
        try {
          auditCase.build(context);
        } finally {
          pairMaximumRows = 0;
        }
      }
    })
  );
}
/** Largest neighbor-search lattice with `columns * rows < 2^24`. */
const MAXIMUM_GRID: [number, number] = [4095, 4095];
/** Largest row count of the column-classification and profile contributors. */
const COLUMN_MAXIMUM_ROWS = 2 ** 24;

type View = KernelAuditContext['view'];

/** Caller-owned compact output of `capacity` ids with every status scalar. */
function compactOutput(view: View, capacity: number) {
  return {
    ids: view('uint32', capacity),
    status: {
      count: view('uint32', 1),
      overflow: view('uint32', 1),
      requiredCount: view('uint32', 1)
    }
  };
}

const CLASS_BREAKS_CASES: KernelAuditCase[] = [
  ...GPU_CLASS_BREAKS_METHODS.map(
    (method: GPUClassBreaksMethod): KernelAuditCase => ({
      exportName: 'GPUClassBreaks',
      name: `method ${method}, 64 classes, 2048 bins`,
      build: ({graph, view}) =>
        graph.add(
          new GPUClassBreaks({
            values: view('float32', ROWS),
            mask: view('uint32', ROWS),
            parameters: view('float32', getGPUClassBreaksParameterLength(64)),
            maximumClassCount: 64,
            methods: [method],
            naturalBreaksBinCount: 2048,
            output: {
              breaks: view('float32', 65),
              classCount: view('uint32', 1),
              classCounts: view('uint32', 64)
            }
          })
        )
    })
  ),
  ...(['uint32', 'sint32'] as const).map(
    (format): KernelAuditCase => ({
      exportName: 'GPUClassBreaks',
      name: `every method, ${format} values`,
      build: ({graph, view}) =>
        graph.add(
          new GPUClassBreaks({
            values: view(format, ROWS),
            mask: view('uint32', ROWS),
            parameters: view('float32', getGPUClassBreaksParameterLength(64)),
            maximumClassCount: 64,
            naturalBreaksBinCount: 2048,
            output: {
              breaks: view('float32', 65),
              classCount: view('uint32', 1),
              classCounts: view('uint32', 64)
            }
          })
        )
    })
  ),
  {
    exportName: 'GPUClassBreaks',
    name: 'every method at maximum rows (2^24), 64 classes',
    build: ({graph, view}) =>
      graph.add(
        new GPUClassBreaks({
          values: view('float32', COLUMN_MAXIMUM_ROWS),
          mask: view('uint32', COLUMN_MAXIMUM_ROWS),
          parameters: view('float32', getGPUClassBreaksParameterLength(64)),
          maximumClassCount: 64,
          naturalBreaksBinCount: 2048,
          output: {
            breaks: view('float32', 65),
            classCount: view('uint32', 1),
            classCounts: view('uint32', 64)
          }
        })
      )
  },
  {
    exportName: 'GPUClassBreaks',
    name: 'every method, 6 classes, default bins, no optional views',
    build: ({graph, view}) =>
      graph.add(
        new GPUClassBreaks({
          values: view('float32', ROWS),
          parameters: view('float32', getGPUClassBreaksParameterLength(6)),
          maximumClassCount: 6,
          output: {breaks: view('float32', 7), classCount: view('uint32', 1)}
        })
      )
  },
  ...(
    [
      {classes: 1, bins: 2},
      {classes: 2, bins: 2},
      {classes: 64, bins: 2}
    ] as const
  ).map(
    ({classes, bins}): KernelAuditCase => ({
      exportName: 'GPUClassBreaks',
      name: `natural-breaks only, ${classes} classes, ${bins} bins, masked`,
      build: ({graph, view}) =>
        graph.add(
          new GPUClassBreaks({
            values: view('float32', ROWS),
            mask: view('uint32', ROWS),
            parameters: view('float32', getGPUClassBreaksParameterLength(classes)),
            maximumClassCount: classes,
            methods: ['natural-breaks'],
            naturalBreaksBinCount: bins,
            output: {breaks: view('float32', classes + 1), classCount: view('uint32', 1)}
          })
        )
    })
  )
];

const COLOR_SCALE_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPUColorScale',
    name: 'float32 every output, domainCount view',
    build: ({graph, view}) =>
      graph.add(
        new GPUColorScale({
          values: view('float32', ROWS),
          mask: view('uint32', ROWS),
          domain: view('float32', 65),
          domainCount: view('uint32', 1),
          palette: view('uint32', 64),
          parameters: view('float32', GPU_COLOR_SCALE_PARAMETER_LENGTH),
          maximumDomainCount: 65,
          maximumPaletteCount: 64,
          output: {
            colors: view('uint32', ROWS),
            classIndices: view('uint32', ROWS),
            classCounts: view('uint32', 64)
          }
        })
      )
  },
  {
    exportName: 'GPUColorScale',
    name: 'float32 colors only, domainCount parameter',
    build: ({graph, view}) =>
      graph.add(
        new GPUColorScale({
          values: view('float32', ROWS),
          domain: view('float32', 8),
          palette: view('uint32', 8),
          parameters: view('float32', GPU_COLOR_SCALE_PARAMETER_LENGTH),
          maximumDomainCount: 8,
          maximumPaletteCount: 8,
          output: {colors: view('uint32', ROWS)}
        })
      )
  },
  ...(['ordinal', 'numeric'] as const).map(
    (integerValues): KernelAuditCase => ({
      exportName: 'GPUColorScale',
      name: `uint32 ${integerValues} every output`,
      build: ({graph, view}) =>
        graph.add(
          new GPUColorScale({
            values: view('uint32', ROWS),
            integerValues,
            mask: view('uint32', ROWS),
            domain: view('float32', 65),
            domainCount: view('uint32', 1),
            palette: view('uint32', 64),
            parameters: view('float32', GPU_COLOR_SCALE_PARAMETER_LENGTH),
            maximumDomainCount: 65,
            maximumPaletteCount: 64,
            output: {
              colors: view('uint32', ROWS),
              classIndices: view('uint32', ROWS),
              classCounts: view('uint32', 64)
            }
          })
        )
    })
  ),
  {
    exportName: 'GPUColorScale',
    name: 'sint32 every output',
    build: ({graph, view}) =>
      graph.add(
        new GPUColorScale({
          values: view('sint32', ROWS),
          mask: view('uint32', ROWS),
          domain: view('float32', 65),
          domainCount: view('uint32', 1),
          palette: view('uint32', 64),
          parameters: view('float32', GPU_COLOR_SCALE_PARAMETER_LENGTH),
          maximumDomainCount: 65,
          maximumPaletteCount: 64,
          output: {
            colors: view('uint32', ROWS),
            classIndices: view('uint32', ROWS),
            classCounts: view('uint32', 64)
          }
        })
      )
  },
  {
    exportName: 'GPUColorScale',
    name: 'rows, domain and palette at the binding limit',
    build: ({graph, view, maximumRows}) => {
      const rows = maximumRows(4);
      graph.add(
        new GPUColorScale({
          values: view('float32', rows),
          mask: view('uint32', rows),
          domain: view('float32', rows),
          domainCount: view('uint32', 1),
          palette: view('uint32', rows),
          parameters: view('float32', GPU_COLOR_SCALE_PARAMETER_LENGTH),
          maximumDomainCount: rows,
          maximumPaletteCount: rows,
          output: {
            colors: view('uint32', rows),
            classIndices: view('uint32', rows),
            classCounts: view('uint32', rows)
          }
        })
      );
    }
  }
];

const DATAFRAME_COLUMN_CASES: KernelAuditCase[] = [
  ...[1, 12, 64].map(
    (quantileCount): KernelAuditCase => ({
      exportName: 'GPUColumnQuantiles',
      name: `every output, ${quantileCount} quantiles`,
      build: ({graph, view}) =>
        graph.add(
          new GPUColumnQuantiles({
            values: view('float32', ROWS),
            mask: view('uint32', ROWS),
            parameters: view('float32', getGPUColumnQuantilesParameterLength(quantileCount)),
            quantileCount,
            output: {
              quantiles: view('float32', quantileCount),
              validCount: view('uint32', 1),
              outputMask: view('uint32', ROWS),
              filterBounds: view('float32', 2)
            }
          })
        )
    })
  ),
  {
    exportName: 'GPUColumnQuantiles',
    name: 'maximum rows (2^24), maximum quantiles',
    build: ({graph, view}) => {
      const quantileCount = GPU_COLUMN_QUANTILES_MAXIMUM_QUANTILE_COUNT;
      graph.add(
        new GPUColumnQuantiles({
          values: view('float32', COLUMN_MAXIMUM_ROWS),
          mask: view('uint32', COLUMN_MAXIMUM_ROWS),
          parameters: view('float32', getGPUColumnQuantilesParameterLength(quantileCount)),
          quantileCount,
          output: {
            quantiles: view('float32', quantileCount),
            validCount: view('uint32', 1),
            outputMask: view('uint32', COLUMN_MAXIMUM_ROWS),
            filterBounds: view('float32', 2)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUColumnQuantiles',
    name: 'required outputs only',
    build: ({graph, view}) =>
      graph.add(
        new GPUColumnQuantiles({
          values: view('float32', ROWS),
          parameters: view('float32', getGPUColumnQuantilesParameterLength(3)),
          quantileCount: 3,
          output: {quantiles: view('float32', 3), validCount: view('uint32', 1)}
        })
      )
  },
  {
    exportName: 'GPUBivariateClassification',
    name: 'every option, 16 classes per axis',
    build: ({graph, view}) =>
      graph.add(
        new GPUBivariateClassification({
          valuesX: view('float32', ROWS),
          valuesY: view('float32', ROWS),
          mask: view('uint32', ROWS),
          breaksX: view('float32', 17),
          breaksY: view('float32', 17),
          palette: view('uint32', 256),
          alphaValues: view('float32', ROWS),
          parameters: view('float32', GPU_BIVARIATE_CLASSIFICATION_PARAMETER_LENGTH),
          maximumClassCount: 16,
          output: {
            classIds: view('uint32', ROWS),
            colors: view('uint32', ROWS),
            classCounts: view('uint32', 256)
          }
        })
      )
  },
  {
    exportName: 'GPUBivariateClassification',
    name: 'one output, 3 classes, no optional inputs',
    build: ({graph, view}) =>
      graph.add(
        new GPUBivariateClassification({
          valuesX: view('float32', ROWS),
          valuesY: view('float32', ROWS),
          breaksX: view('float32', 4),
          breaksY: view('float32', 4),
          palette: view('uint32', 9),
          parameters: view('float32', GPU_BIVARIATE_CLASSIFICATION_PARAMETER_LENGTH),
          maximumClassCount: 3,
          output: {colors: view('uint32', ROWS)}
        })
      )
  },
  {
    exportName: 'GPUBivariateClassification',
    name: 'rows at the binding limit',
    build: ({graph, view, maximumRows}) => {
      const rows = maximumRows(4);
      graph.add(
        new GPUBivariateClassification({
          valuesX: view('float32', rows),
          valuesY: view('float32', rows),
          mask: view('uint32', rows),
          breaksX: view('float32', 17),
          breaksY: view('float32', 17),
          palette: view('uint32', 256),
          alphaValues: view('float32', rows),
          parameters: view('float32', GPU_BIVARIATE_CLASSIFICATION_PARAMETER_LENGTH),
          maximumClassCount: 16,
          output: {
            classIds: view('uint32', rows),
            colors: view('uint32', rows),
            classCounts: view('uint32', 256)
          }
        })
      );
    }
  }
];

const CALENDAR_BUCKETS_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPUCalendarBuckets',
    name: 'every output, mask and per-row offsets',
    build: ({graph, view}) =>
      graph.add(
        new GPUCalendarBuckets({
          timestamps: view('uint32x2', ROWS),
          mask: view('uint32', ROWS),
          utcOffsets: view('sint32', ROWS),
          parameters: view('sint32', GPU_CALENDAR_BUCKETS_PARAMETER_LENGTH),
          output: {
            year: view('sint32', ROWS),
            month: view('uint32', ROWS),
            dayOfMonth: view('uint32', ROWS),
            hour: view('uint32', ROWS),
            minute: view('uint32', ROWS),
            weekday: view('uint32', ROWS),
            dayOfYear: view('uint32', ROWS),
            isoWeek: view('uint32', ROWS),
            isoWeekYear: view('sint32', ROWS),
            quarter: view('uint32', ROWS),
            hourWeekdayCounts: view('uint32', GPU_CALENDAR_BUCKETS_MATRIX_LENGTH)
          }
        })
      )
  },
  {
    exportName: 'GPUCalendarBuckets',
    name: 'every output, mask and a 4096-transition time zone table',
    build: ({graph, view}) =>
      graph.add(
        new GPUCalendarBuckets({
          timestamps: view('uint32x2', ROWS),
          mask: view('uint32', ROWS),
          timeZone: {transitions: view('uint32x2', 4096), offsets: view('sint32', 4097)},
          parameters: view('sint32', GPU_CALENDAR_BUCKETS_PARAMETER_LENGTH),
          output: {
            year: view('sint32', ROWS),
            month: view('uint32', ROWS),
            dayOfMonth: view('uint32', ROWS),
            hour: view('uint32', ROWS),
            minute: view('uint32', ROWS),
            weekday: view('uint32', ROWS),
            dayOfYear: view('uint32', ROWS),
            isoWeek: view('uint32', ROWS),
            isoWeekYear: view('sint32', ROWS),
            quarter: view('uint32', ROWS),
            hourWeekdayCounts: view('uint32', GPU_CALENDAR_BUCKETS_MATRIX_LENGTH)
          }
        })
      )
  },
  {
    exportName: 'GPUCalendarBuckets',
    name: 'matrix only, fixed offset',
    build: ({graph, view}) =>
      graph.add(
        new GPUCalendarBuckets({
          timestamps: view('uint32x2', ROWS),
          parameters: view('sint32', GPU_CALENDAR_BUCKETS_PARAMETER_LENGTH),
          output: {hourWeekdayCounts: view('uint32', GPU_CALENDAR_BUCKETS_MATRIX_LENGTH)}
        })
      )
  },
  {
    exportName: 'GPUCalendarBuckets',
    name: 'every output at the binding-limited row count',
    build: ({graph, view, maximumRows}) => {
      const rows = maximumRows(8);
      graph.add(
        new GPUCalendarBuckets({
          timestamps: view('uint32x2', rows),
          mask: view('uint32', rows),
          utcOffsets: view('sint32', rows),
          parameters: view('sint32', GPU_CALENDAR_BUCKETS_PARAMETER_LENGTH),
          output: {
            year: view('sint32', rows),
            month: view('uint32', rows),
            dayOfMonth: view('uint32', rows),
            hour: view('uint32', rows),
            minute: view('uint32', rows),
            weekday: view('uint32', rows),
            dayOfYear: view('uint32', rows),
            isoWeek: view('uint32', rows),
            isoWeekYear: view('sint32', rows),
            quarter: view('uint32', rows),
            hourWeekdayCounts: view('uint32', GPU_CALENDAR_BUCKETS_MATRIX_LENGTH)
          }
        })
      );
    }
  }
];

const COLUMN_PROFILE_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPUColumnProfile',
    name: 'every option, 16 mixed columns',
    build: ({graph, view}) => {
      const columnCount = GPU_COLUMN_PROFILE_MAXIMUM_COLUMN_COUNT;
      const binCount = 1024;
      const precision = 12;
      const topCategoryCount = 10;
      graph.add(
        new GPUColumnProfile({
          columns: Array.from({length: columnCount}, (_, index) =>
            index % 2 === 0
              ? {values: view('float32', ROWS)}
              : {values: view('uint32', ROWS), kind: 'category' as const, categoryCount: 100}
          ),
          mask: view('uint32', ROWS),
          parameters: view('float32', getGPUColumnProfileParameterLength(columnCount)),
          histogramBinCount: binCount,
          hyperLogLogPrecision: precision,
          topCategoryCount,
          output: {
            statistics: view('float32', columnCount * GPU_COLUMN_PROFILE_STATISTIC_COUNT),
            counts: view('uint32', columnCount * 2),
            histograms: view('uint32', columnCount * binCount),
            hyperLogLogRegisters: view('uint32', columnCount * 2 ** precision),
            topCategories: view('uint32', columnCount * topCategoryCount),
            topCategoryCounts: view('uint32', columnCount * topCategoryCount)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUColumnProfile',
    name: 'one numeric column, statistics only',
    build: ({graph, view}) =>
      graph.add(
        new GPUColumnProfile({
          columns: [{values: view('float32', ROWS)}],
          histogramBinCount: 1,
          output: {statistics: view('float32', GPU_COLUMN_PROFILE_STATISTIC_COUNT)}
        })
      )
  },
  {
    exportName: 'GPUColumnProfile',
    name: 'maximum columns, bins, precision, top categories and rows',
    build: ({graph, view, device}) => {
      const categoryCount = getGPUColumnProfileMaximumCategoryTotal(device) / 4;
      const columnCount = GPU_COLUMN_PROFILE_MAXIMUM_COLUMN_COUNT;
      const binCount = 2 ** 20;
      const precision = 16;
      const topCategoryCount = GPU_COLUMN_PROFILE_MAXIMUM_TOP_CATEGORY_COUNT;
      graph.add(
        new GPUColumnProfile({
          columns: Array.from({length: columnCount}, (_, index) =>
            index < 12
              ? {values: view('float32', COLUMN_MAXIMUM_ROWS)}
              : {
                  values: view('uint32', COLUMN_MAXIMUM_ROWS),
                  kind: 'category' as const,
                  categoryCount
                }
          ),
          mask: view('uint32', COLUMN_MAXIMUM_ROWS),
          parameters: view('float32', getGPUColumnProfileParameterLength(columnCount)),
          histogramBinCount: binCount,
          hyperLogLogPrecision: precision,
          topCategoryCount,
          output: {
            statistics: view('float32', columnCount * GPU_COLUMN_PROFILE_STATISTIC_COUNT),
            counts: view('uint32', columnCount * 2),
            histograms: view('uint32', columnCount * binCount),
            hyperLogLogRegisters: view('uint32', columnCount * 2 ** precision),
            topCategories: view('uint32', columnCount * topCategoryCount),
            topCategoryCounts: view('uint32', columnCount * topCategoryCount)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUColumnProfile',
    name: 'category total one above the maximum, 16 columns',
    build: ({graph, view, device}) => {
      const categoryCount = getGPUColumnProfileMaximumCategoryTotal(device) / 4 + 1;
      const columnCount = GPU_COLUMN_PROFILE_MAXIMUM_COLUMN_COUNT;
      const binCount = 2 ** 20;
      const precision = 16;
      const topCategoryCount = GPU_COLUMN_PROFILE_MAXIMUM_TOP_CATEGORY_COUNT;
      graph.add(
        new GPUColumnProfile({
          columns: Array.from({length: columnCount}, (_, index) =>
            index < 12
              ? {values: view('float32', COLUMN_MAXIMUM_ROWS)}
              : {
                  values: view('uint32', COLUMN_MAXIMUM_ROWS),
                  kind: 'category' as const,
                  categoryCount
                }
          ),
          mask: view('uint32', COLUMN_MAXIMUM_ROWS),
          parameters: view('float32', getGPUColumnProfileParameterLength(columnCount)),
          histogramBinCount: binCount,
          hyperLogLogPrecision: precision,
          topCategoryCount,
          output: {
            statistics: view('float32', columnCount * GPU_COLUMN_PROFILE_STATISTIC_COUNT),
            counts: view('uint32', columnCount * 2),
            histograms: view('uint32', columnCount * binCount),
            hyperLogLogRegisters: view('uint32', columnCount * 2 ** precision),
            topCategories: view('uint32', columnCount * topCategoryCount),
            topCategoryCounts: view('uint32', columnCount * topCategoryCount)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUColumnProfile',
    name: 'two category columns of 2^24 categories (2^25 in total)',
    build: ({graph, view}) => {
      const topCategoryCount = GPU_COLUMN_PROFILE_MAXIMUM_TOP_CATEGORY_COUNT;
      graph.add(
        new GPUColumnProfile({
          columns: [0, 1].map(() => ({
            values: view('uint32', COLUMN_MAXIMUM_ROWS),
            kind: 'category' as const,
            categoryCount: COLUMN_MAXIMUM_ROWS
          })),
          histogramBinCount: 256,
          topCategoryCount,
          output: {
            statistics: view('float32', 2 * GPU_COLUMN_PROFILE_STATISTIC_COUNT),
            topCategories: view('uint32', 2 * topCategoryCount),
            topCategoryCounts: view('uint32', 2 * topCategoryCount)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUColumnProfile',
    name: 'one category column with 2^24 categories',
    build: ({graph, view}) => {
      const topCategoryCount = GPU_COLUMN_PROFILE_MAXIMUM_TOP_CATEGORY_COUNT;
      graph.add(
        new GPUColumnProfile({
          columns: [
            {
              values: view('uint32', COLUMN_MAXIMUM_ROWS),
              kind: 'category',
              categoryCount: COLUMN_MAXIMUM_ROWS
            }
          ],
          histogramBinCount: 256,
          topCategoryCount,
          output: {
            statistics: view('float32', GPU_COLUMN_PROFILE_STATISTIC_COUNT),
            topCategories: view('uint32', topCategoryCount),
            topCategoryCounts: view('uint32', topCategoryCount)
          }
        })
      );
    }
  }
];

const COMPOSITE_CASES: KernelAuditCase[] = [
  ...(
    [
      [false, false],
      [true, false],
      [false, true],
      [true, true]
    ] as const
  ).map(
    ([enableRank, enablePrincipalComponent]): KernelAuditCase => ({
      exportName: 'GPUCompositeScore',
      name: `16 indicators, rank ${enableRank}, principal component ${enablePrincipalComponent}`,
      build: ({graph, view}) => {
        const count = GPU_COMPOSITE_SCORE_MAXIMUM_INDICATOR_COUNT;
        graph.add(
          new GPUCompositeScore({
            indicators: view('float32', ROWS * count),
            indicatorCount: count,
            mask: view('uint32', ROWS),
            parameters: view('float32', GPU_COMPOSITE_SCORE_PARAMETER_LENGTH),
            enableRank,
            enablePrincipalComponent,
            output: {
              score: view('float32', ROWS),
              scaled: view('float32', ROWS * count),
              columnStatistics: view(
                'float32',
                count * GPU_COMPOSITE_SCORE_COLUMN_STATISTICS_STRIDE
              ),
              ...(enablePrincipalComponent
                ? {
                    loadings: view('float32', count),
                    principalComponentSummary: view(
                      'float32',
                      GPU_COMPOSITE_SCORE_PRINCIPAL_COMPONENT_SUMMARY_LENGTH
                    )
                  }
                : {})
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUCompositeScore',
    name: 'one indicator, score only',
    build: ({graph, view}) =>
      graph.add(
        new GPUCompositeScore({
          indicators: view('float32', ROWS),
          indicatorCount: 1,
          parameters: view('float32', GPU_COMPOSITE_SCORE_PARAMETER_LENGTH),
          output: {score: view('float32', ROWS)}
        })
      )
  },
  {
    exportName: 'GPUCompositeScore',
    name: '16 indicators, everything on, rows at the binding limit',
    build: ({graph, view, maximumRows}) => {
      const count = GPU_COMPOSITE_SCORE_MAXIMUM_INDICATOR_COUNT;
      const rows = maximumRows(count * 4);
      graph.add(
        new GPUCompositeScore({
          indicators: view('float32', rows * count),
          indicatorCount: count,
          mask: view('uint32', rows),
          parameters: view('float32', GPU_COMPOSITE_SCORE_PARAMETER_LENGTH),
          enableRank: true,
          enablePrincipalComponent: true,
          output: {
            score: view('float32', rows),
            scaled: view('float32', rows * count),
            columnStatistics: view('float32', count * GPU_COMPOSITE_SCORE_COLUMN_STATISTICS_STRIDE),
            loadings: view('float32', count),
            principalComponentSummary: view(
              'float32',
              GPU_COMPOSITE_SCORE_PRINCIPAL_COMPONENT_SUMMARY_LENGTH
            )
          }
        })
      );
    }
  }
];

const INEQUALITY_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPUInequality',
    name: 'every output, weights and mask, 11 knots',
    build: ({graph, view}) => {
      const zones = 40;
      graph.add(
        new GPUInequality({
          values: view('float32', ROWS),
          zoneIds: view('uint32', ROWS),
          zoneCount: zones,
          weights: view('float32', ROWS),
          mask: view('uint32', ROWS),
          parameters: view('float32', GPU_INEQUALITY_PARAMETER_LENGTH),
          output: {
            gini: view('float32', zones),
            theilT: view('float32', zones),
            theilL: view('float32', zones),
            atkinson: view('float32', zones),
            hoover: view('float32', zones),
            palma: view('float32', zones),
            mean: view('float32', zones),
            counts: view('uint32', zones),
            lorenzKnots: view('float32', zones * 11),
            globalSummary: view('float32', GPU_INEQUALITY_GLOBAL_SUMMARY_LENGTH)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUInequality',
    name: 'every output, 1025 knots, no weights',
    build: ({graph, view}) => {
      const zones = 40;
      graph.add(
        new GPUInequality({
          values: view('float32', ROWS),
          zoneIds: view('uint32', ROWS),
          zoneCount: zones,
          parameters: view('float32', GPU_INEQUALITY_PARAMETER_LENGTH),
          lorenzKnotCount: 1025,
          output: {
            gini: view('float32', zones),
            theilT: view('float32', zones),
            theilL: view('float32', zones),
            atkinson: view('float32', zones),
            hoover: view('float32', zones),
            palma: view('float32', zones),
            mean: view('float32', zones),
            counts: view('uint32', zones),
            lorenzKnots: view('float32', zones * 1025),
            globalSummary: view('float32', GPU_INEQUALITY_GLOBAL_SUMMARY_LENGTH)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUInequality',
    name: 'gini only, one zone',
    build: ({graph, view}) =>
      graph.add(
        new GPUInequality({
          values: view('float32', ROWS),
          zoneIds: view('uint32', ROWS),
          zoneCount: 1,
          parameters: view('float32', GPU_INEQUALITY_PARAMETER_LENGTH),
          output: {gini: view('float32', 1)}
        })
      )
  },
  {
    exportName: 'GPUInequality',
    name: 'zones x 1025 knots at the maximum',
    build: ({graph, view, maximumRows, device}) => {
      const rows = maximumRows(4);
      const zones = Math.floor(getGPUInequalityMaximumLorenzKnotTotal(device) / 1025);
      graph.add(
        new GPUInequality({
          values: view('float32', rows),
          zoneIds: view('uint32', rows),
          zoneCount: zones,
          weights: view('float32', rows),
          mask: view('uint32', rows),
          parameters: view('float32', GPU_INEQUALITY_PARAMETER_LENGTH),
          lorenzKnotCount: 1025,
          output: {
            gini: view('float32', zones),
            theilT: view('float32', zones),
            theilL: view('float32', zones),
            atkinson: view('float32', zones),
            hoover: view('float32', zones),
            palma: view('float32', zones),
            mean: view('float32', zones),
            counts: view('uint32', zones),
            lorenzKnots: view('float32', zones * 1025),
            globalSummary: view('float32', GPU_INEQUALITY_GLOBAL_SUMMARY_LENGTH)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUInequality',
    name: 'zones x 1025 knots one zone above the maximum',
    build: ({graph, view, maximumRows, device}) => {
      const rows = maximumRows(4);
      const zones = Math.floor(getGPUInequalityMaximumLorenzKnotTotal(device) / 1025) + 1;
      graph.add(
        new GPUInequality({
          values: view('float32', rows),
          zoneIds: view('uint32', rows),
          zoneCount: zones,
          weights: view('float32', rows),
          mask: view('uint32', rows),
          parameters: view('float32', GPU_INEQUALITY_PARAMETER_LENGTH),
          lorenzKnotCount: 1025,
          output: {
            gini: view('float32', zones),
            theilT: view('float32', zones),
            theilL: view('float32', zones),
            atkinson: view('float32', zones),
            hoover: view('float32', zones),
            palma: view('float32', zones),
            mean: view('float32', zones),
            counts: view('uint32', zones),
            lorenzKnots: view('float32', zones * 1025),
            globalSummary: view('float32', GPU_INEQUALITY_GLOBAL_SUMMARY_LENGTH)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUInequality',
    name: 'zone count at the largest per-zone scratch binding (44 bytes per zone)',
    build: ({graph, view, maximumRows}) => {
      const rows = maximumRows(4);
      const zones = maximumRows(44);
      graph.add(
        new GPUInequality({
          values: view('float32', rows),
          zoneIds: view('uint32', rows),
          zoneCount: zones,
          parameters: view('float32', GPU_INEQUALITY_PARAMETER_LENGTH),
          output: {
            gini: view('float32', zones),
            theilT: view('float32', zones),
            theilL: view('float32', zones),
            atkinson: view('float32', zones),
            hoover: view('float32', zones),
            palma: view('float32', zones),
            mean: view('float32', zones),
            counts: view('uint32', zones),
            globalSummary: view('float32', GPU_INEQUALITY_GLOBAL_SUMMARY_LENGTH)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUInequality',
    name: 'zone count at the maximum, per-zone views only',
    build: ({graph, view, maximumRows, device}) => {
      const rows = maximumRows(4);
      const zones = getGPUInequalityMaximumZoneCount(device);
      graph.add(
        new GPUInequality({
          values: view('float32', rows),
          zoneIds: view('uint32', rows),
          zoneCount: zones,
          parameters: view('float32', GPU_INEQUALITY_PARAMETER_LENGTH),
          output: {
            gini: view('float32', zones),
            theilT: view('float32', zones),
            theilL: view('float32', zones),
            atkinson: view('float32', zones),
            hoover: view('float32', zones),
            palma: view('float32', zones),
            mean: view('float32', zones),
            counts: view('uint32', zones),
            globalSummary: view('float32', GPU_INEQUALITY_GLOBAL_SUMMARY_LENGTH)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUInequality',
    name: 'zone count one above the maximum, per-zone views only',
    build: ({graph, view, maximumRows, device}) => {
      const rows = maximumRows(4);
      const zones = getGPUInequalityMaximumZoneCount(device) + 1;
      graph.add(
        new GPUInequality({
          values: view('float32', rows),
          zoneIds: view('uint32', rows),
          zoneCount: zones,
          parameters: view('float32', GPU_INEQUALITY_PARAMETER_LENGTH),
          output: {
            gini: view('float32', zones),
            theilT: view('float32', zones),
            theilL: view('float32', zones),
            atkinson: view('float32', zones),
            hoover: view('float32', zones),
            palma: view('float32', zones),
            mean: view('float32', zones),
            counts: view('uint32', zones),
            globalSummary: view('float32', GPU_INEQUALITY_GLOBAL_SUMMARY_LENGTH)
          }
        })
      );
    }
  }
];

const ALL_GROUP_STATISTICS: readonly GPUGroupStatistic[] = [
  'count',
  'sum',
  'mean',
  'minimum',
  'maximum',
  'variance',
  'standardDeviation',
  'skewness',
  'kurtosis',
  'median',
  'percentiles',
  'mode',
  'uniqueCount',
  'zScore'
];

/** One value column requesting every statistic over `capacity` groups and `rows` rows. */
function groupStatisticsColumn(
  view: View,
  rows: number,
  capacity: number,
  percentileCount: number
) {
  return {
    values: view('float32', rows),
    statistics: ALL_GROUP_STATISTICS,
    output: {
      counts: view('uint32', capacity),
      sums: view('uint32x2', capacity),
      sumValues: view('float32', capacity),
      means: view('float32', capacity),
      minimums: view('float32', capacity),
      maximums: view('float32', capacity),
      variances: view('float32', capacity),
      standardDeviations: view('float32', capacity),
      skewness: view('float32', capacity),
      kurtosis: view('float32', capacity),
      medians: view('float32', capacity),
      percentiles: view('float32', capacity * percentileCount),
      modes: view('float32', capacity),
      uniqueCounts: view('uint32', capacity),
      zScores: view('float32', rows)
    }
  };
}

const GROUP_STATISTICS_CASES: KernelAuditCase[] = [
  ...(['sample', 'population'] as const).map(
    (variance): KernelAuditCase => ({
      exportName: 'GPUGroupStatistics',
      name: `uint32 keys, 4 columns, every statistic, ${variance}`,
      build: ({graph, view}) => {
        const capacity = 64;
        graph.add(
          new GPUGroupStatistics({
            keys: view('uint32', ROWS),
            mask: view('uint32', ROWS),
            columns: Array.from({length: 4}, () => groupStatisticsColumn(view, ROWS, capacity, 5)),
            variance,
            percentiles: view('float32', 5),
            sumScale: 1024,
            output: {
              keys: view('uint32', capacity),
              counts: view('uint32', capacity),
              status: {
                count: view('uint32', 1),
                overflow: view('uint32', 1),
                requiredCount: view('uint32', 1)
              }
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUGroupStatistics',
    name: 'uint32x2 keys, 4 columns, every statistic',
    build: ({graph, view}) => {
      const capacity = 64;
      graph.add(
        new GPUGroupStatistics({
          keys: view('uint32x2', ROWS),
          mask: view('uint32', ROWS),
          columns: Array.from({length: 4}, () => groupStatisticsColumn(view, ROWS, capacity, 5)),
          percentiles: view('float32', 5),
          output: {
            keys: view('uint32x2', capacity),
            counts: view('uint32', capacity),
            status: {
              count: view('uint32', 1),
              overflow: view('uint32', 1),
              requiredCount: view('uint32', 1)
            }
          }
        })
      );
    }
  },
  {
    exportName: 'GPUGroupStatistics',
    name: 'dense keyCount, 4 columns, every statistic',
    build: ({graph, view}) => {
      const keyCount = 128;
      graph.add(
        new GPUGroupStatistics({
          keys: view('uint32', ROWS),
          mask: view('uint32', ROWS),
          keyCount,
          columns: Array.from({length: 4}, () => groupStatisticsColumn(view, ROWS, keyCount, 5)),
          percentiles: view('float32', 5),
          output: {
            keys: view('uint32', keyCount),
            counts: view('uint32', keyCount),
            status: {count: view('uint32', 1), overflow: view('uint32', 1)}
          }
        })
      );
    }
  },
  {
    exportName: 'GPUGroupStatistics',
    name: 'no value columns',
    build: ({graph, view}) =>
      graph.add(
        new GPUGroupStatistics({
          keys: view('uint32', ROWS),
          columns: [],
          output: {
            keys: view('uint32', 16),
            counts: view('uint32', 16),
            status: {count: view('uint32', 1), overflow: view('uint32', 1)}
          }
        })
      )
  },
  {
    exportName: 'GPUGroupStatistics',
    name: 'uint32x2 keys, groups one above the maximum',
    build: ({graph, view, maximumRows, device}) => {
      const rows = maximumRows(8);
      const capacity = getGPUGroupStatisticsMaximumGroupCount(device) + 1;
      graph.add(
        new GPUGroupStatistics({
          keys: view('uint32x2', rows),
          mask: view('uint32', rows),
          columns: [
            {
              values: view('float32', rows),
              statistics: ALL_GROUP_STATISTICS.filter(statistic => statistic !== 'percentiles'),
              output: {
                counts: view('uint32', capacity),
                sums: view('uint32x2', capacity),
                sumValues: view('float32', capacity),
                means: view('float32', capacity),
                minimums: view('float32', capacity),
                maximums: view('float32', capacity),
                variances: view('float32', capacity),
                standardDeviations: view('float32', capacity),
                skewness: view('float32', capacity),
                kurtosis: view('float32', capacity),
                medians: view('float32', capacity),
                modes: view('float32', capacity),
                uniqueCounts: view('uint32', capacity),
                zScores: view('float32', rows)
              }
            }
          ],
          output: {
            keys: view('uint32x2', capacity),
            counts: view('uint32', capacity),
            status: {
              count: view('uint32', 1),
              overflow: view('uint32', 1),
              requiredCount: view('uint32', 1)
            }
          }
        })
      );
    }
  },
  {
    exportName: 'GPUGroupStatistics',
    name: 'uint32x2 keys, groups at the maximum',
    build: ({graph, view, maximumRows, device}) => {
      const rows = maximumRows(8);
      const capacity = getGPUGroupStatisticsMaximumGroupCount(device);
      graph.add(
        new GPUGroupStatistics({
          keys: view('uint32x2', rows),
          mask: view('uint32', rows),
          columns: [
            {
              values: view('float32', rows),
              statistics: ALL_GROUP_STATISTICS.filter(statistic => statistic !== 'percentiles'),
              output: {
                counts: view('uint32', capacity),
                sums: view('uint32x2', capacity),
                sumValues: view('float32', capacity),
                means: view('float32', capacity),
                minimums: view('float32', capacity),
                maximums: view('float32', capacity),
                variances: view('float32', capacity),
                standardDeviations: view('float32', capacity),
                skewness: view('float32', capacity),
                kurtosis: view('float32', capacity),
                medians: view('float32', capacity),
                modes: view('float32', capacity),
                uniqueCounts: view('uint32', capacity),
                zScores: view('float32', rows)
              }
            }
          ],
          output: {
            keys: view('uint32x2', capacity),
            counts: view('uint32', capacity),
            status: {
              count: view('uint32', 1),
              overflow: view('uint32', 1),
              requiredCount: view('uint32', 1)
            }
          }
        })
      );
    }
  },
  {
    exportName: 'GPUGroupStatistics',
    name: 'uint32 keys, percentiles at the binding limit',
    build: ({graph, view, maximumRows}) => {
      const percentileCount = 64;
      const capacity = maximumRows(4 * percentileCount);
      const rows = maximumRows(4);
      graph.add(
        new GPUGroupStatistics({
          keys: view('uint32', rows),
          columns: [
            {
              values: view('float32', rows),
              statistics: ['percentiles', 'median'],
              output: {
                percentiles: view('float32', capacity * percentileCount),
                medians: view('float32', capacity)
              }
            }
          ],
          percentiles: view('float32', percentileCount),
          output: {
            keys: view('uint32', capacity),
            counts: view('uint32', capacity),
            status: {count: view('uint32', 1), overflow: view('uint32', 1)}
          }
        })
      );
    }
  }
];

const KEY_JOIN_CASES: KernelAuditCase[] = (['left', 'inner'] as const)
  .flatMap(kind =>
    (['uint32', 'uint32x2'] as const).map(
      (keyFormat): KernelAuditCase => ({
        exportName: 'GPUKeyJoin',
        name: `${kind} join, ${keyFormat} keys, every option`,
        build: ({graph, view}) => {
          const left = ROWS;
          const right = 200;
          graph.add(
            new GPUKeyJoin({
              leftKeys: view(keyFormat, left),
              rightKeys: view(keyFormat, right),
              leftMask: view('uint32', left),
              rightMask: view('uint32', right),
              kind,
              sumScale: 1024,
              gather: [
                ...Array.from({length: 4}, () => ({
                  column: view('float32', right),
                  output: view('float32', left)
                })),
                ...Array.from({length: 4}, () => ({
                  column: view('uint32', right),
                  output: view('uint32', left)
                }))
              ],
              aggregates: [
                {operation: 'count' as const, output: view('float32', left)},
                {
                  operation: 'count' as const,
                  column: view('float32', right),
                  output: view('float32', left)
                },
                {
                  operation: 'sum' as const,
                  column: view('float32', right),
                  output: view('float32', left),
                  sums: view('uint32x2', left)
                },
                {
                  operation: 'mean' as const,
                  column: view('float32', right),
                  output: view('float32', left),
                  sums: view('uint32x2', left)
                },
                {
                  operation: 'minimum' as const,
                  column: view('float32', right),
                  output: view('float32', left)
                },
                {
                  operation: 'maximum' as const,
                  column: view('float32', right),
                  output: view('float32', left)
                }
              ],
              output: {
                rightRows: view('uint32', left),
                matchCounts: view('uint32', left),
                matched: view('uint32', left),
                rightMatched: view('uint32', right),
                ...(kind === 'inner' ? {rows: compactOutput(view, left)} : {})
              }
            })
          );
        }
      })
    )
  )
  .concat([
    {
      exportName: 'GPUKeyJoin',
      name: 'left join, matched output only',
      build: ({graph, view}) =>
        graph.add(
          new GPUKeyJoin({
            leftKeys: view('uint32', ROWS),
            rightKeys: view('uint32', ROWS),
            output: {matched: view('uint32', ROWS)}
          })
        )
    },
    ...(['left', 'inner'] as const).map(
      (kind): KernelAuditCase => ({
        exportName: 'GPUKeyJoin',
        name: `${kind} join, uint32x2 keys, rows at the binding limit`,
        build: ({graph, view, maximumRows}) => {
          const rows = maximumRows(8);
          graph.add(
            new GPUKeyJoin({
              leftKeys: view('uint32x2', rows),
              rightKeys: view('uint32x2', rows),
              leftMask: view('uint32', rows),
              rightMask: view('uint32', rows),
              kind,
              gather: [{column: view('float32', rows), output: view('float32', rows)}],
              aggregates: [
                {
                  operation: 'sum',
                  column: view('float32', rows),
                  output: view('float32', rows),
                  sums: view('uint32x2', rows)
                }
              ],
              output: {
                rightRows: view('uint32', rows),
                matchCounts: view('uint32', rows),
                matched: view('uint32', rows),
                rightMatched: view('uint32', rows),
                ...(kind === 'inner' ? {rows: compactOutput(view, rows)} : {})
              }
            })
          );
        }
      })
    )
  ]);

const VARIOGRAM_CASES: KernelAuditCase[] = [
  ...(
    [
      [64, 4, true],
      [128, 4, false],
      [512, 1, false],
      [1, 512, false],
      [409, 1, true]
    ] as const
  ).map(
    ([lagCount, directionCount, robust]): KernelAuditCase => ({
      exportName: 'GPUVariogram',
      name: `${lagCount} lags x ${directionCount} sectors, robust ${robust}, every output`,
      build: ({graph, view}) => {
        const binCount = lagCount * directionCount;
        graph.add(
          new GPUVariogram({
            positions: view('float32x2', ROWS),
            values: view('float32', ROWS),
            mask: view('uint32', ROWS),
            parameters: view('float32', GPU_VARIOGRAM_PARAMETER_LENGTH),
            gridSize: [16, 16],
            lagCount,
            directionCount,
            output: {
              semivariances: view('float32', binCount),
              pairCounts: view('uint32', binCount),
              meanDistances: view('float32', binCount),
              ...(robust ? {robustSemivariances: view('float32', binCount)} : {}),
              statistics: view('float32', GPU_VARIOGRAM_STATISTICS_LENGTH),
              status: {overflow: view('uint32', 1)}
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUVariogram',
    name: 'semivariances only',
    build: ({graph, view}) =>
      graph.add(
        new GPUVariogram({
          positions: view('float32x2', ROWS),
          values: view('float32', ROWS),
          parameters: view('float32', GPU_VARIOGRAM_PARAMETER_LENGTH),
          gridSize: [1, 1],
          lagCount: 1,
          output: {semivariances: view('float32', 1)}
        })
      )
  },
  {
    exportName: 'GPUVariogram',
    name: 'maximum rows and lattice, 409 bins with robust channel',
    build: ({graph, view}) =>
      graph.add(
        new GPUVariogram({
          positions: view('float32x2', pairMaximumRows),
          values: view('float32', pairMaximumRows),
          mask: view('uint32', pairMaximumRows),
          parameters: view('float32', GPU_VARIOGRAM_PARAMETER_LENGTH),
          gridSize: MAXIMUM_GRID,
          lagCount: 409,
          output: {
            semivariances: view('float32', 409),
            pairCounts: view('uint32', 409),
            meanDistances: view('float32', 409),
            robustSemivariances: view('float32', 409),
            statistics: view('float32', GPU_VARIOGRAM_STATISTICS_LENGTH),
            status: {overflow: view('uint32', 1)}
          }
        })
      )
  }
];

const CORRELOGRAM_CASES: KernelAuditCase[] = [
  ...(['cumulative', 'annulus'] as const).map(
    (bandMode): KernelAuditCase => ({
      exportName: 'GPUSpatialCorrelogram',
      name: `${bandMode} bands, 64 bands, every output`,
      build: ({graph, view}) =>
        graph.add(
          new GPUSpatialCorrelogram({
            positions: view('float32x2', ROWS),
            values: view('float32', ROWS),
            mask: view('uint32', ROWS),
            parameters: view('float32', GPU_SPATIAL_CORRELOGRAM_PARAMETER_LENGTH),
            gridSize: [16, 16],
            bandCount: 64,
            bandMode,
            output: {
              moransI: view('float32', 64),
              zScores: view('float32', 64),
              pValues: view('float32', 64),
              expectedI: view('float32', 64),
              varianceI: view('float32', 64),
              pairCounts: view('uint32', 64),
              peakBands: view('uint32', 2),
              statistics: view('float32', GPU_SPATIAL_CORRELOGRAM_STATISTICS_LENGTH),
              status: {overflow: view('uint32', 1)}
            }
          })
        )
    })
  ),
  {
    exportName: 'GPUSpatialCorrelogram',
    name: 'one band, Moran I only',
    build: ({graph, view}) =>
      graph.add(
        new GPUSpatialCorrelogram({
          positions: view('float32x2', ROWS),
          values: view('float32', ROWS),
          parameters: view('float32', GPU_SPATIAL_CORRELOGRAM_PARAMETER_LENGTH),
          gridSize: [1, 1],
          bandCount: 1,
          output: {moransI: view('float32', 1)}
        })
      )
  },
  ...(['cumulative', 'annulus'] as const).map(
    (bandMode): KernelAuditCase => ({
      exportName: 'GPUSpatialCorrelogram',
      name: `${bandMode} bands, maximum rows, lattice and bands`,
      build: ({graph, view}) =>
        graph.add(
          new GPUSpatialCorrelogram({
            positions: view('float32x2', pairMaximumRows),
            values: view('float32', pairMaximumRows),
            mask: view('uint32', pairMaximumRows),
            parameters: view('float32', GPU_SPATIAL_CORRELOGRAM_PARAMETER_LENGTH),
            gridSize: MAXIMUM_GRID,
            bandCount: 64,
            bandMode,
            output: {
              moransI: view('float32', 64),
              zScores: view('float32', 64),
              pValues: view('float32', 64),
              expectedI: view('float32', 64),
              varianceI: view('float32', 64),
              pairCounts: view('uint32', 64),
              peakBands: view('uint32', 2),
              statistics: view('float32', GPU_SPATIAL_CORRELOGRAM_STATISTICS_LENGTH),
              status: {overflow: view('uint32', 1)}
            }
          })
        )
    })
  )
];

const RIPLEY_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPURipley',
    name: '256 radii, every output',
    build: ({graph, view}) =>
      graph.add(
        new GPURipley({
          positions: view('float32x2', ROWS),
          mask: view('uint32', ROWS),
          parameters: view('float32', GPU_RIPLEY_PARAMETER_LENGTH),
          gridSize: [16, 16],
          radiusCount: 256,
          output: {
            k: view('float32', 256),
            l: view('float32', 256),
            lMinusR: view('float32', 256),
            pairCorrelation: view('float32', 256),
            pairCounts: view('uint32', 256),
            radii: view('float32', 256)
          }
        })
      )
  },
  {
    exportName: 'GPURipley',
    name: 'one radius, K only',
    build: ({graph, view}) =>
      graph.add(
        new GPURipley({
          positions: view('float32x2', ROWS),
          parameters: view('float32', GPU_RIPLEY_PARAMETER_LENGTH),
          gridSize: [1, 1],
          radiusCount: 1,
          output: {k: view('float32', 1)}
        })
      )
  },
  {
    exportName: 'GPURipley',
    name: 'maximum rows and lattice, 256 radii, every output',
    build: ({graph, view}) =>
      graph.add(
        new GPURipley({
          positions: view('float32x2', pairMaximumRows),
          mask: view('uint32', pairMaximumRows),
          parameters: view('float32', GPU_RIPLEY_PARAMETER_LENGTH),
          gridSize: MAXIMUM_GRID,
          radiusCount: 256,
          output: {
            k: view('float32', 256),
            l: view('float32', 256),
            lMinusR: view('float32', 256),
            pairCorrelation: view('float32', 256),
            pairCounts: view('uint32', 256),
            radii: view('float32', 256)
          }
        })
      )
  }
];

const ENVELOPE_CASES: KernelAuditCase[] = [
  ...(['k', 'l', 'l-minus-r'] as const).map(
    (statistic): KernelAuditCase => ({
      exportName: 'GPURipleyEnvelope',
      name: `statistic ${statistic}, 99 simulations, rank 5, every output`,
      build: ({graph, view}) =>
        graph.add(
          new GPURipleyEnvelope({
            positions: view('float32x2', ROWS),
            mask: view('uint32', ROWS),
            parameters: view('float32', GPU_RIPLEY_PARAMETER_LENGTH),
            gridSize: [16, 16],
            radiusCount: 64,
            simulationCount: 99,
            rank: 5,
            statistic,
            observedK: view('float32', 64),
            output: {
              lower: view('float32', 64),
              upper: view('float32', 64),
              globalLower: view('float32', 64),
              globalUpper: view('float32', 64),
              theoretical: view('float32', 64),
              statistics: view('float32', GPU_RIPLEY_ENVELOPE_STATISTICS_LENGTH)
            }
          })
        )
    })
  ),
  {
    exportName: 'GPURipleyEnvelope',
    name: 'maximum simulations (199), rank 99, 256 radii, every output',
    build: ({graph, view}) =>
      graph.add(
        new GPURipleyEnvelope({
          positions: view('float32x2', ROWS),
          mask: view('uint32', ROWS),
          parameters: view('float32', GPU_RIPLEY_PARAMETER_LENGTH),
          gridSize: [16, 16],
          radiusCount: 256,
          simulationCount: GPU_RIPLEY_ENVELOPE_MAXIMUM_SIMULATION_COUNT,
          rank: 99,
          observedK: view('float32', 256),
          output: {
            lower: view('float32', 256),
            upper: view('float32', 256),
            globalLower: view('float32', 256),
            globalUpper: view('float32', 256),
            theoretical: view('float32', 256),
            statistics: view('float32', GPU_RIPLEY_ENVELOPE_STATISTICS_LENGTH)
          }
        })
      )
  },
  {
    exportName: 'GPURipleyEnvelope',
    name: 'two simulations, one radius, lower only',
    build: ({graph, view}) =>
      graph.add(
        new GPURipleyEnvelope({
          positions: view('float32x2', ROWS),
          parameters: view('float32', GPU_RIPLEY_PARAMETER_LENGTH),
          gridSize: [1, 1],
          radiusCount: 1,
          simulationCount: 2,
          output: {lower: view('float32', 1)}
        })
      )
  },
  {
    exportName: 'GPURipleyEnvelope',
    name: 'maximum rows and lattice, 199 simulations, 256 radii',
    build: ({graph, view}) =>
      graph.add(
        new GPURipleyEnvelope({
          positions: view('float32x2', pairMaximumRows),
          mask: view('uint32', pairMaximumRows),
          parameters: view('float32', GPU_RIPLEY_PARAMETER_LENGTH),
          gridSize: MAXIMUM_GRID,
          radiusCount: 256,
          simulationCount: GPU_RIPLEY_ENVELOPE_MAXIMUM_SIMULATION_COUNT,
          rank: 99,
          observedK: view('float32', 256),
          output: {
            lower: view('float32', 256),
            upper: view('float32', 256),
            globalLower: view('float32', 256),
            globalUpper: view('float32', 256),
            theoretical: view('float32', 256),
            statistics: view('float32', GPU_RIPLEY_ENVELOPE_STATISTICS_LENGTH)
          }
        })
      )
  }
];

const POINT_PATTERN_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPUPointPatternIndices',
    name: 'every output, 256 x 256 quadrats',
    build: ({graph, view}) =>
      graph.add(
        new GPUPointPatternIndices({
          positions: view('float32x2', ROWS),
          mask: view('uint32', ROWS),
          parameters: view('float32', GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH),
          gridSize: [16, 16],
          quadratGrid: [256, 256],
          output: {
            nearestNeighborDistances: view('float32', ROWS),
            nearestNeighborIds: view('uint32', ROWS),
            clarkEvans: view('float32', GPU_CLARK_EVANS_LENGTH),
            quadratCounts: view('uint32', GPU_QUADRAT_MAXIMUM_COUNT),
            quadratStatistics: view('float32', GPU_QUADRAT_STATISTICS_LENGTH)
          }
        })
      )
  },
  {
    exportName: 'GPUPointPatternIndices',
    name: 'nearest neighbor outputs only',
    build: ({graph, view}) =>
      graph.add(
        new GPUPointPatternIndices({
          positions: view('float32x2', ROWS),
          parameters: view('float32', GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH),
          gridSize: [16, 16],
          output: {
            nearestNeighborDistances: view('float32', ROWS),
            nearestNeighborIds: view('uint32', ROWS),
            clarkEvans: view('float32', GPU_CLARK_EVANS_LENGTH)
          }
        })
      )
  },
  {
    exportName: 'GPUPointPatternIndices',
    name: 'quadrat statistics only, 65536 x 1 quadrats',
    build: ({graph, view}) =>
      graph.add(
        new GPUPointPatternIndices({
          positions: view('float32x2', ROWS),
          parameters: view('float32', GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH),
          gridSize: [16, 16],
          quadratGrid: [GPU_QUADRAT_MAXIMUM_COUNT, 1],
          output: {quadratStatistics: view('float32', GPU_QUADRAT_STATISTICS_LENGTH)}
        })
      )
  },
  {
    exportName: 'GPUPointPatternIndices',
    name: 'maximum rows and lattice, every output',
    build: ({graph, view}) =>
      graph.add(
        new GPUPointPatternIndices({
          positions: view('float32x2', pairMaximumRows),
          mask: view('uint32', pairMaximumRows),
          parameters: view('float32', GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH),
          gridSize: MAXIMUM_GRID,
          quadratGrid: [256, 256],
          output: {
            nearestNeighborDistances: view('float32', pairMaximumRows),
            nearestNeighborIds: view('uint32', pairMaximumRows),
            clarkEvans: view('float32', GPU_CLARK_EVANS_LENGTH),
            quadratCounts: view('uint32', GPU_QUADRAT_MAXIMUM_COUNT),
            quadratStatistics: view('float32', GPU_QUADRAT_STATISTICS_LENGTH)
          }
        })
      )
  }
];

const DISTANCE_FUNCTIONS_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPURipleyDistanceFunctions',
    name: '256 radii, G F J and radii',
    build: ({graph, view}) =>
      graph.add(
        new GPURipleyDistanceFunctions({
          positions: view('float32x2', ROWS),
          mask: view('uint32', ROWS),
          parameters: view('float32', GPU_RIPLEY_DISTANCE_FUNCTIONS_PARAMETER_LENGTH),
          gridSize: [16, 16],
          referenceGrid: [64, 64],
          radiusCount: 256,
          output: {
            g: view('float32', 256),
            f: view('float32', 256),
            j: view('float32', 256),
            radii: view('float32', 256)
          }
        })
      )
  },
  ...(['g', 'f', 'j'] as const).map(
    (name): KernelAuditCase => ({
      exportName: 'GPURipleyDistanceFunctions',
      name: `${name} only, 8 radii`,
      build: ({graph, view}) =>
        graph.add(
          new GPURipleyDistanceFunctions({
            positions: view('float32x2', ROWS),
            parameters: view('float32', GPU_RIPLEY_DISTANCE_FUNCTIONS_PARAMETER_LENGTH),
            gridSize: [16, 16],
            referenceGrid: [8, 8],
            radiusCount: 8,
            output: {[name]: view('float32', 8)}
          })
        )
    })
  ),
  {
    exportName: 'GPURipleyDistanceFunctions',
    name: 'maximum rows, lattice and reference grid (2^24)',
    build: ({graph, view}) =>
      graph.add(
        new GPURipleyDistanceFunctions({
          positions: view('float32x2', pairMaximumRows),
          mask: view('uint32', pairMaximumRows),
          parameters: view('float32', GPU_RIPLEY_DISTANCE_FUNCTIONS_PARAMETER_LENGTH),
          gridSize: MAXIMUM_GRID,
          referenceGrid: [4096, 4096],
          radiusCount: 256,
          output: {
            g: view('float32', 256),
            f: view('float32', 256),
            j: view('float32', 256),
            radii: view('float32', 256)
          }
        })
      )
  }
];

/** Builds a temporal reduction with every output and `cellCount * bucketCount` slots. */
function temporalReduction(
  {view}: KernelAuditContext,
  options: {
    rows: number;
    cellCount: number;
    bucketCount: number;
    words: boolean;
    capacity?: number;
    sorted?: boolean;
  }
) {
  const {rows, cellCount, bucketCount, words} = options;
  const slots = cellCount * bucketCount;
  const capacity = options.capacity ?? slots;
  return new GPUTemporalReduction({
    sorted: options.sorted,
    cellIds: view('uint32', rows),
    timestamps: words ? view('uint32x2', rows) : view('float32', rows),
    values: view('float32', rows),
    mask: view('uint32', rows),
    parameters: words
      ? view('uint32', GPU_TEMPORAL_REDUCTION_WORD_PARAMETER_LENGTH)
      : view('float32', GPU_TEMPORAL_REDUCTION_PARAMETER_LENGTH),
    cellCount,
    bucketCount,
    output: {
      counts: view('uint32', slots),
      minimums: view('float32', slots),
      maximums: view('float32', slots),
      first: view('float32', slots),
      last: view('float32', slots),
      occupiedSlots: compactOutput(view, capacity)
    }
  });
}

const TEMPORAL_REDUCTION_CASES: KernelAuditCase[] = [
  ...[false, true].flatMap(words =>
    [
      [16, 16],
      [1, 257],
      [64, 4]
    ].map(
      ([cellCount, bucketCount]): KernelAuditCase => ({
        exportName: 'GPUTemporalReduction',
        name: `${words ? 'word' : 'float'} times, ${cellCount} cells x ${bucketCount} buckets`,
        build: context =>
          context.graph.add(temporalReduction(context, {rows: ROWS, cellCount, bucketCount, words}))
      })
    )
  ),
  ...[false, true].map(
    (words): KernelAuditCase => ({
      exportName: 'GPUTemporalReduction',
      name: `${words ? 'word' : 'float'} times, slots at the binding limit`,
      build: context => {
        const slots = context.maximumRows(4);
        const bucketCount = words ? 1024 : 2 ** 20;
        const cellCount = Math.floor(slots / bucketCount);
        context.graph.add(
          temporalReduction(context, {
            rows: context.maximumRows(8),
            cellCount,
            bucketCount,
            words
          })
        );
      }
    })
  ),
  ...[false, true].flatMap(words =>
    [
      [16, 16],
      [64, 64]
    ].map(
      ([cellCount, bucketCount]): KernelAuditCase => ({
        exportName: 'GPUTemporalReduction',
        name: `sorted ${words ? 'word' : 'float'} times, ${cellCount} cells x ${bucketCount} buckets`,
        build: context =>
          context.graph.add(
            temporalReduction(context, {rows: ROWS, cellCount, bucketCount, words, sorted: true})
          )
      })
    )
  ),
  ...[false, true].map(
    (words): KernelAuditCase => ({
      exportName: 'GPUTemporalReduction',
      name: `sorted ${words ? 'word' : 'float'} times, rows at the binding limit`,
      build: context =>
        context.graph.add(
          temporalReduction(context, {
            rows: context.maximumRows(8),
            cellCount: 4,
            bucketCount: 4,
            words,
            sorted: true
          })
        )
    })
  ),
  {
    exportName: 'GPUTemporalReduction',
    name: 'one cell with the maximum float bucket count (2^20)',
    build: context =>
      context.graph.add(
        temporalReduction(context, {
          rows: ROWS,
          cellCount: 1,
          bucketCount: 2 ** 20,
          words: false
        })
      )
  }
];

/** Compact output sized `capacity` plus every optional time-window output of `rows` rows. */
function timeWindowFilter(
  {view}: KernelAuditContext,
  options: {rows: number; words: boolean; interval: boolean; trackCount: number}
) {
  const {rows, words, interval, trackCount} = options;
  const time = () => (words ? view('uint32x2', rows) : view('float32', rows));
  return new GPUTimeWindowFilter({
    timestamps: time() as never,
    ...(!words ? {timestampsLow: view('float32', rows)} : {}),
    ...(interval ? {endTimestamps: time() as never} : {}),
    ...(interval && !words ? {endTimestampsLow: view('float32', rows)} : {}),
    window: words
      ? view('uint32', GPU_TIME_WINDOW_FILTER_WORD_PARAMETER_LENGTH)
      : view('float32', GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH),
    additionalPredicates: [
      {kind: 'selection', mask: view('uint32', rows)},
      {kind: ['bounds', 'lod'], mask: view('uint32', rows)}
    ],
    sourceIds: view('uint32', rows),
    trackIds: view('uint32', rows),
    output: {
      ...compactOutput(view, rows),
      outputMask: view('uint32', rows),
      fadeWeights: view('float32', rows),
      ...(interval ? {clipFractions: view('float32x2', rows)} : {}),
      trackVisibleCounts: view('uint32', trackCount),
      drawInstanceCount: view('uint32', 1)
    }
  } as never);
}

const TIME_WINDOW_FILTER_CASES: KernelAuditCase[] = [
  ...[false, true].flatMap(words =>
    [false, true].map(
      (interval): KernelAuditCase => ({
        exportName: 'GPUTimeWindowFilter',
        name: `${words ? 'word' : 'float'} times, ${interval ? 'interval' : 'instant'}, every option`,
        build: context =>
          context.graph.add(
            timeWindowFilter(context, {rows: ROWS, words, interval, trackCount: 50})
          )
      })
    )
  ),
  ...[false, true].map(
    (words): KernelAuditCase => ({
      exportName: 'GPUTimeWindowFilter',
      name: `sorted ${words ? 'word' : 'double-single float'} instants, every option`,
      build: ({graph, view}) => {
        const rows = ROWS;
        graph.add(
          new GPUTimeWindowFilter({
            sorted: true,
            validateOrder: true,
            timestamps: (words ? view('uint32x2', rows) : view('float32', rows)) as never,
            ...(!words ? {timestampsLow: view('float32', rows)} : {}),
            window: words
              ? view('uint32', GPU_TIME_WINDOW_FILTER_WORD_PARAMETER_LENGTH)
              : view('float32', GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH),
            sourceIds: view('uint32', rows),
            output: {
              ...compactOutput(view, rows),
              outputMask: view('uint32', rows),
              fadeWeights: view('float32', rows),
              drawInstanceCount: view('uint32', 1),
              rowRange: view('uint32', 2),
              drawFirstInstance: view('uint32', 1),
              orderViolations: view('uint32', 1)
            }
          } as never)
        );
      }
    })
  ),
  ...[false, true].map(
    (words): KernelAuditCase => ({
      exportName: 'GPUTimeWindowFilter',
      name: `sorted ${words ? 'word' : 'float'} instants, rows at the binding limit`,
      build: ({graph, view, maximumRows}) => {
        const rows = maximumRows(8);
        graph.add(
          new GPUTimeWindowFilter({
            sorted: true,
            timestamps: (words ? view('uint32x2', rows) : view('float32', rows)) as never,
            window: words
              ? view('uint32', GPU_TIME_WINDOW_FILTER_WORD_PARAMETER_LENGTH)
              : view('float32', GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH),
            sourceIds: view('uint32', rows),
            output: {
              ...compactOutput(view, rows),
              rowRange: view('uint32', 2)
            }
          } as never)
        );
      }
    })
  ),
  {
    exportName: 'GPUTimeWindowFilter',
    name: 'float instants, ids only',
    build: ({graph, view}) =>
      graph.add(
        new GPUTimeWindowFilter({
          timestamps: view('float32', ROWS),
          window: view('float32', GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH),
          output: {
            ids: view('uint32', ROWS),
            status: {count: view('uint32', 1), overflow: view('uint32', 1)}
          }
        })
      )
  },
  ...[false, true].map(
    (words): KernelAuditCase => ({
      exportName: 'GPUTimeWindowFilter',
      name: `${words ? 'word' : 'float'} intervals, rows and tracks at the binding limit`,
      build: context =>
        context.graph.add(
          timeWindowFilter(context, {
            rows: context.maximumRows(8),
            words,
            interval: true,
            trackCount: context.maximumRows(4)
          })
        )
    })
  )
];

/**
 * Repeats the pair-statistics maximum-size cases one row above
 * {@link getGPUPairStatisticsMaximumRowCount}, which the contributors must refuse by name.
 */
const PAIR_ABOVE_MAXIMUM_CASES: KernelAuditCase[] = withPairRowCount(
  [
    ...VARIOGRAM_CASES,
    ...CORRELOGRAM_CASES,
    ...RIPLEY_CASES,
    ...ENVELOPE_CASES,
    ...POINT_PATTERN_CASES,
    ...DISTANCE_FUNCTIONS_CASES
  ].filter(auditCase => auditCase.name.includes('maximum rows')),
  1,
  ' (points above the maximum)'
);

/** Kernel-audit cases for dataframe. */
export const DATAFRAME_CASES: KernelAuditCase[] = [
  ...CALENDAR_BUCKETS_CASES,
  ...DATAFRAME_COLUMN_CASES,
  ...CLASS_BREAKS_CASES,
  ...COLOR_SCALE_CASES,
  ...COLUMN_PROFILE_CASES,
  ...COMPOSITE_CASES,
  ...INEQUALITY_CASES,
  ...GROUP_STATISTICS_CASES,
  ...KEY_JOIN_CASES,
  ...withPairRowCount(VARIOGRAM_CASES, 0),
  ...withPairRowCount(CORRELOGRAM_CASES, 0),
  ...withPairRowCount(RIPLEY_CASES, 0),
  ...withPairRowCount(ENVELOPE_CASES, 0),
  ...withPairRowCount(POINT_PATTERN_CASES, 0),
  ...withPairRowCount(DISTANCE_FUNCTIONS_CASES, 0),
  ...TEMPORAL_REDUCTION_CASES,
  ...TIME_WINDOW_FILTER_CASES,
  ...PAIR_ABOVE_MAXIMUM_CASES
];
