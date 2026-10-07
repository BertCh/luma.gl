// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {sampleRamp} from '../../engine/ramps';
import type {LegendSpec} from '../scene';
import {defineScene} from '../scene';
import {CATEGORY_SELECT_OPTIONS, CHICAGO_VIEW, MONTH_NAMES} from './b8-common';
import type {CellPyramidOptions} from './cell-pyramid.compute';

const MONTROSE = [-87.6325, 41.9625] as const;
const DEFAULT_ROLLUP = 14;

const inView =
  (...views: CellPyramidOptions['view'][]) =>
  (state: CellPyramidOptions) =>
    !views.includes(state.view);

const monthFormat = (value: number) => MONTH_NAMES[Math.round(value) - 1];

/** Quadbin cell pyramid, roll-up, class map and period comparison. GPU work is in `cell-pyramid.compute.ts`. */
export default defineScene<CellPyramidOptions>({
  id: 'cell-pyramid',
  title: 'Nature change between seasons',
  chapter: 'cells',
  order: 2,
  summary:
    'One Quadbin pyramid serves every zoom level of 43,600 Chicago wildlife observations. Roll it up exactly, classify the cells with class breaks, and compare the first and second half of 2023 as a cell-by-cell change map.',
  contributors: [
    'GPUCellPyramid',
    'GPUCellLevelSelection',
    'GPUCellRollup',
    'GPUCellAggregation',
    'GPUCellTableCompare',
    'addPeriodComparisonRecipe',
    'addCellTableColumnsNode',
    'GPUClassBreaks',
    'GPUColorScale'
  ],
  datasets: [
    {id: 'chicago-nature', role: 'iNaturalist observations (2023)'},
    {id: 'chicago-community-areas', role: 'place names for readouts and tooltips'}
  ],
  initialView: {...CHICAGO_VIEW},

  options: [
    {
      kind: 'select',
      id: 'view',
      label: 'What to compute',
      group: 'View',
      apply: 'compile',
      default: 'pyramid',
      help: 'Each view runs different compiled graphs over the same observation points. Roll-up, class and compare graphs are compiled the first time you open them (the badge counts that), then reused.',
      options: [
        {value: 'pyramid', label: 'Zoom pyramid (level chosen by the map zoom)'},
        {value: 'rollup', label: 'Exact roll-up to a coarser resolution'},
        {value: 'classes', label: 'Class map of the roll-up (class breaks)'},
        {value: 'compare', label: 'Compare two periods'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showAreas',
      label: 'Community-area borders',
      group: 'View',
      apply: 'param',
      default: true,
      help: 'Thin white borders of the 77 community areas for orientation. Hover the map for the area name.'
    },
    {
      kind: 'select',
      id: 'category',
      label: 'Group',
      group: 'Observations',
      apply: 'param',
      default: 'all',
      help: 'Keeps one iNaturalist group (plants, birds, insects, ...) in every view. A mask write: the graphs re-encode, nothing recompiles.',
      options: CATEGORY_SELECT_OPTIONS
    },
    {
      kind: 'slider',
      id: 'sampledPercent',
      label: 'Sampled share of observations',
      group: 'Observations',
      apply: 'param',
      min: 5,
      max: 100,
      step: 5,
      default: 100,
      unit: '%',
      help: 'Thins the observations with a deterministic hash, to see how stable the patterns are.'
    },
    {
      kind: 'slider',
      id: 'resolutionOffset',
      label: 'Cells finer than the zoom level',
      group: 'Pyramid (GPUCellLevelSelection)',
      apply: 'param',
      min: 0,
      max: 6,
      step: 1,
      default: 4,
      disabledWhen: state => state.view !== 'pyramid' || state.manualLevel,
      format: value => `+${value} (cells about ${Math.round(512 / 2 ** value)} px)`,
      help: 'The pyramid level shown is round(map zoom) + this offset. The zoom is written into a one-word buffer every frame; nothing is rebuilt.'
    },
    {
      kind: 'toggle',
      id: 'manualLevel',
      label: 'Choose the level by hand',
      group: 'Pyramid (GPUCellLevelSelection)',
      apply: 'param',
      default: false,
      disabledWhen: inView('pyramid'),
      help: 'Overrides the zoom-driven level, to compare levels at one camera.'
    },
    {
      kind: 'slider',
      id: 'manualResolution',
      label: 'Quadbin resolution',
      group: 'Pyramid (GPUCellLevelSelection)',
      apply: 'param',
      min: 4,
      max: 17,
      step: 1,
      default: 12,
      disabledWhen: state => state.view !== 'pyramid' || !state.manualLevel,
      help: 'Quadbin zoom level of the cells: 12 is about 7 km wide at Chicago, 15 about 900 m, 17 about 230 m.'
    },
    {
      kind: 'slider',
      id: 'densityCeiling',
      label: 'Density ceiling (log10 observations per km2)',
      group: 'Pyramid (GPUCellLevelSelection)',
      apply: 'param',
      min: 1,
      max: 6,
      step: 0.5,
      default: 4,
      disabledWhen: state =>
        !(state.view === 'pyramid' || (state.view === 'rollup' && state.rollupColor === 'density')),
      help: 'Densities are shown as log10(observations per km2) so levels are comparable. The ramp spans 1 to 10^ceiling; the busiest cells at Montrose Point reach about 10^4.'
    },
    {
      kind: 'slider',
      id: 'rollupResolution',
      label: 'Parent resolution',
      group: 'Roll-up (GPUCellRollup)',
      apply: 'compile',
      min: 10,
      max: 16,
      step: 1,
      default: DEFAULT_ROLLUP,
      disabledWhen: inView('rollup', 'classes'),
      format: value => `res ${value} (tiles about ${((40075 / 2 ** value) * 0.744).toFixed(1)} km)`,
      help: 'Resolution of the parent table. It is a kernel constant, so each resolution is its own small compiled graph, built the first time you pick it. The roll-up reads the finest table (res 17), not the points.'
    },
    {
      kind: 'select',
      id: 'rollupColor',
      label: 'Color the roll-up by',
      group: 'Roll-up (GPUCellRollup)',
      apply: 'param',
      default: 'density',
      disabledWhen: inView('rollup'),
      help: 'Observation density, or the research-grade share: the roll-up carries an exact fixed-point sum of research-grade observations per cell, so mean = research-grade observations / observations.',
      options: [
        {value: 'density', label: 'Observations per km2'},
        {
          value: 'researchShare',
          label: 'Research-grade share (research-grade observations per observation)'
        }
      ]
    },
    {
      kind: 'slider',
      id: 'rateCeiling',
      label: 'Research-grade share ceiling',
      group: 'Roll-up (GPUCellRollup)',
      apply: 'param',
      min: 20,
      max: 100,
      step: 5,
      default: 80,
      unit: '%',
      disabledWhen: state => state.view !== 'rollup' || state.rollupColor !== 'researchShare',
      help: 'Research-grade share mapped to the top of the ramp. Citywide 63% of observations are research grade.'
    },
    {
      kind: 'select',
      id: 'classValue',
      label: 'Classify',
      group: 'Class map (GPUClassBreaks)',
      apply: 'compile',
      default: 'confirmed',
      disabledWhen: inView('classes'),
      help: 'Observations per cell (the count column) or research-grade observations per cell (the fixed-point sums as f32). It changes which columns the adapter node binds, so each choice is its own graph.',
      options: [
        {value: 'confirmed', label: 'Research-grade observations per cell'},
        {value: 'observations', label: 'Observations per cell'}
      ]
    },
    {
      kind: 'select',
      id: 'classMethod',
      label: 'Classification method',
      group: 'Class map (GPUClassBreaks)',
      apply: 'param',
      default: 'quantile',
      disabledWhen: inView('classes'),
      help: 'All six methods are compiled into the graph; the choice is a parameter write. Quantile gives equal numbers of cells per class; equal interval equal widths; natural breaks (Fisher-Jenks on a histogram) minimise within-class variance; standard deviation centres classes on the mean; head/tail suits heavy tails; box plot gives six classes.',
      options: [
        {value: 'quantile', label: 'Quantile'},
        {value: 'equal-interval', label: 'Equal interval'},
        {value: 'natural-breaks', label: 'Natural breaks (Jenks)'},
        {value: 'standard-deviation', label: 'Standard deviation'},
        {value: 'head-tail', label: 'Head/tail breaks'},
        {value: 'box-plot', label: 'Box plot (6 classes)'}
      ]
    },
    {
      kind: 'slider',
      id: 'classCount',
      label: 'Number of classes',
      group: 'Class map (GPUClassBreaks)',
      apply: 'param',
      min: 2,
      max: 8,
      step: 1,
      default: 5,
      disabledWhen: state => state.view !== 'classes' || state.classMethod === 'box-plot',
      help: 'Requested classes; head/tail returns fewer when the data run out of heads. Box plot is always six.'
    },
    {
      kind: 'range',
      id: 'periodA',
      label: 'Earlier period (months)',
      group: 'Period comparison',
      apply: 'param',
      min: 1,
      max: 12,
      step: 1,
      default: [1, 6],
      format: monthFormat,
      disabledWhen: inView('compare'),
      help: 'The observations of these months (inclusive) make the "before" table. Only 2023 is in the dataset, so periods are seasons of one year.'
    },
    {
      kind: 'range',
      id: 'periodB',
      label: 'Later period (months)',
      group: 'Period comparison',
      apply: 'param',
      min: 1,
      max: 12,
      step: 1,
      default: [7, 12],
      format: monthFormat,
      disabledWhen: inView('compare'),
      help: 'The "after" table. Periods can overlap or differ in length; unequal lengths bias the totals, so prefer equal windows.'
    },
    {
      kind: 'slider',
      id: 'compareResolution',
      label: 'Comparison resolution',
      group: 'Period comparison',
      apply: 'compile',
      min: 13,
      max: 16,
      step: 1,
      default: 15,
      disabledWhen: inView('compare'),
      format: value =>
        `Quadbin ${value} (tiles about ${((40075 / 2 ** value) * 0.744).toFixed(1)} km)`,
      help: 'Both periods are keyed at this resolution. It is baked into the recipe graph, so each value compiles once, on first use.'
    },
    {
      kind: 'select',
      id: 'compareMeasure',
      label: 'Compared measure',
      group: 'Period comparison',
      apply: 'compile',
      default: 'count',
      disabledWhen: inView('compare'),
      help: 'Observations per cell (count), or research-grade observations per cell (sum of the 0/1 research-grade flag, compared as exact 64-bit fixed point). Changes the recipe topology.',
      options: [
        {value: 'count', label: 'Observations (count)'},
        {value: 'sum', label: 'Research-grade observations (sum of values)'}
      ]
    },
    {
      kind: 'select',
      id: 'zScoreKind',
      label: 'z-score',
      group: 'Period comparison',
      apply: 'compile',
      default: 'poisson',
      disabledWhen: inView('compare'),
      help: 'Poisson: delta / sqrt(before + after), the natural score for counts. Standardized: (delta - mean) / sd across all cells, the default for sums. Compile-time in GPUCellTableCompare.',
      options: [
        {value: 'poisson', label: 'Poisson (counts)'},
        {value: 'standardized', label: 'Standardized (across cells)'}
      ]
    },
    {
      kind: 'select',
      id: 'changeColumn',
      label: 'Show',
      group: 'Period comparison',
      apply: 'param',
      default: 'classes',
      disabledWhen: inView('compare'),
      help: 'The classified delta from the recipe, or any column of the comparison table as a continuous diverging color: delta (after - before), ratio (after / before, log scale), percent change, or the z-score.',
      options: [
        {value: 'classes', label: 'Classified change (recipe colors)'},
        {value: 'delta', label: 'Delta (after - before)'},
        {value: 'ratio', label: 'Ratio (after / before)'},
        {value: 'percentChange', label: 'Percent change'},
        {value: 'zScore', label: 'z-score'}
      ]
    },
    {
      kind: 'select',
      id: 'changeClassing',
      label: 'Class breaks for the change',
      group: 'Period comparison',
      apply: 'param',
      default: 'band',
      disabledWhen: state => state.view !== 'compare' || state.changeColumn !== 'classes',
      help: 'Fixed band: edges at plus and minus b and 4b around zero (b is the no-change band below). Standard deviation: five classes centred on the mean change. Both are compiled; the choice is a parameter write.',
      options: [
        {value: 'band', label: 'Fixed no-change band'},
        {value: 'standard-deviation', label: 'Standard deviation'}
      ]
    },
    {
      kind: 'slider',
      id: 'stableBand',
      label: 'No-change band',
      group: 'Period comparison',
      apply: 'param',
      min: 1,
      max: 60,
      step: 1,
      default: 8,
      format: value => `+/- ${value}`,
      disabledWhen: inView('compare'),
      help: 'Cells whose change is within this many observations (or research-grade observations) count as unchanged (gray). Also the threshold of the gained / lost readout.'
    },
    {
      kind: 'slider',
      id: 'colorLimit',
      label: 'Color limit',
      group: 'Period comparison',
      apply: 'param',
      min: 10,
      max: 100,
      step: 5,
      default: 60,
      unit: '% of the natural range',
      disabledWhen: state => state.view !== 'compare' || state.changeColumn === 'classes',
      help: "Where the diverging ramp saturates, as a share of the column's range (the largest change for delta, 100% for percent change, 4 for z, a factor of 4 for the ratio)."
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp',
      group: 'Color',
      apply: 'param',
      default: 'viridis',
      disabledWhen: inView('pyramid', 'rollup', 'classes'),
      help: 'Ramp of the density, rate and class maps. The comparison always uses a diverging ramp (blue fewer, red more).',
      options: [
        {value: 'viridis', label: 'Viridis'},
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'cividis', label: 'Cividis'},
        {value: 'grayscale', label: 'Grayscale'}
      ]
    }
  ],

  readouts: [
    {id: 'points', label: 'Observations in the filter'},
    {id: 'pyramid', label: 'Pyramid'},
    {id: 'zoom', label: 'Map zoom to cell level'},
    {
      id: 'activeCells',
      label: 'Cells drawn',
      help: 'Row count of the active level, published by GPUCellLevelSelection into the indirect draw record.'
    },
    {id: 'levelsFine', label: 'Cells per level, fine'},
    {id: 'levelsMiddle', label: 'Cells per level, middle'},
    {id: 'levelsCoarse', label: 'Cells per level, coarse'},
    {id: 'overflow', label: 'Overflowed levels'},
    {id: 'encodes', label: 'Graph encodes'},
    {id: 'rollupCells', label: 'Roll-up cells'},
    {id: 'rollupCounts', label: 'Roll-up observations, fine to parent'},
    {id: 'rollupSums', label: 'Roll-up research-grade observations, fine to parent'},
    {id: 'classEdges', label: 'Class edges'},
    {id: 'periods', label: 'Period sizes'},
    {id: 'compareCells', label: 'Cells before, after, union'},
    {id: 'compareTotals', label: 'Total change'},
    {id: 'compareGainLoss', label: 'Cells gained and lost'},
    {id: 'compareSignificant', label: 'Significant cells'},
    {id: 'compareLargestGain', label: 'Largest gain'},
    {id: 'compareLargestLoss', label: 'Largest loss'}
  ],

  legends: state => {
    const legends: LegendSpec[] = [];
    if (state.view === 'pyramid' || (state.view === 'rollup' && state.rollupColor === 'density')) {
      legends.push({
        kind: 'ramp',
        title: 'Observations per km2, log scale',
        ramp: state.ramp,
        extent: [1, 10 ** state.densityCeiling],
        unit: 'observations per km2',
        labels: ['1', `10^${state.densityCeiling}`]
      });
    } else if (state.view === 'rollup') {
      legends.push({
        kind: 'ramp',
        title: 'Research-grade share per cell',
        ramp: state.ramp,
        extent: [0, state.rateCeiling / 100],
        format: value => `${(value * 100).toFixed(0)}%`
      });
    } else if (state.view === 'classes') {
      const classCount = state.classMethod === 'box-plot' ? 6 : state.classCount;
      legends.push({
        kind: 'categories',
        title: `${state.classValue === 'confirmed' ? 'Research-grade observations' : 'Observations'} per cell, ${classCount} classes`,
        entries: Array.from({length: classCount}, (_, index) => {
          const [r, g, b] = sampleRamp(
            state.ramp,
            classCount === 1 ? 0.5 : index / (classCount - 1)
          );
          return {
            color: [r, g, b, 235] as const,
            label:
              index === 0
                ? 'Lowest class'
                : index === classCount - 1
                  ? 'Highest class'
                  : `Class ${index + 1}`
          };
        }),
        note: 'Edges are computed on the GPU every time the table changes; the "Class edges" readout lists them.'
      });
    } else if (state.changeColumn === 'classes') {
      legends.push({
        kind: 'categories',
        title: 'Change between periods',
        entries: [
          {color: [33, 102, 172, 235], label: 'Much fewer'},
          {color: [103, 169, 207, 235], label: 'Fewer'},
          {color: [210, 210, 210, 235], label: 'About the same'},
          {color: [239, 138, 98, 235], label: 'More'},
          {color: [178, 24, 43, 235], label: 'Much more'}
        ],
        note:
          state.changeClassing === 'band'
            ? `Edges at -4b, -b, +b, +4b with b = ${state.stableBand} ${state.compareMeasure === 'sum' ? 'confirmed' : 'observations'} per cell.`
            : 'Standard-deviation classes of the change, centred on the mean change.'
      });
    } else {
      const unit = {
        delta:
          state.compareMeasure === 'sum'
            ? 'research-grade observations (after - before)'
            : 'observations (after - before)',
        ratio: 'after / before',
        percentChange: '% change',
        zScore: `z (${state.zScoreKind})`
      }[state.changeColumn];
      legends.push({
        kind: 'ramp',
        id: 'change',
        title: 'Change between periods',
        ramp: 'diverging',
        extent: 'gpu',
        unit,
        format:
          state.changeColumn === 'ratio'
            ? value => `${(2 ** value).toFixed(2)}x`
            : value => (value > 0 ? '+' : '') + Math.round(value * 10) / 10
      });
      legends.push({
        kind: 'categories',
        title: 'No baseline',
        entries: [{color: [150, 150, 150, 200], label: 'Absent before (ratio undefined)'}]
      });
    }
    return legends;
  },

  snippet: state => {
    if (state.view === 'pyramid') {
      return `import {GPUCellPyramid, GPUCellLevelSelection} from '@luma.gl/experimental/gpu-spatial-analysis';

// One aggregation at res 17, then roll-ups to res 4, written into per-level slices of one slab.
graph.add(new GPUCellPyramid({
  family: 'quadbin', positions, mask, values: researchGrade,
  levels,   // [{resolution: 17, output: {cells, counts, sums, count, overflow}}, {resolution: 16, ...}, ...]
  levelCounts
}));

// A second tiny graph, every frame: the zoom is a one-word buffer.
activeLevel.write(Uint32Array.of(17 - Math.round(zoom) - ${state.resolutionOffset}));
graph.add(new GPUCellLevelSelection({
  levelCounts, activeLevel: activeLevel.importToGraph(graph), levelFirstRows,
  output: {count: drawRecordInstanceCount, firstRow}   // indirect draw record read by the layer
}));`;
    }
    if (state.view === 'rollup') {
      return `import {GPUCellRollup, GPU_CELL_DEFAULT_SUM_SCALE} from '@luma.gl/experimental/gpu-spatial-analysis';

// Roll the finest table up to res ${state.rollupResolution}. Counts and fixed-point sums are integers,
// so the parent totals equal the child totals exactly.
graph.add(new GPUCellRollup({
  family: 'quadbin', sourceResolution: 17, resolution: ${state.rollupResolution},
  source: finestTable,                    // {cells, counts, sums, count, overflow}
  sumScale: GPU_CELL_DEFAULT_SUM_SCALE,   // 65536 = 16 fractional bits
  output: {cells, counts, sums, sumValues, count: drawRecordInstanceCount, overflow}
}));`;
    }
    if (state.view === 'classes') {
      return `import {GPUClassBreaks, GPUColorScale} from '@luma.gl/experimental/gpu-dataframe';

// Cell table to a value column and an occupied-row mask (adapter node, as in the hot-spot recipe).
addCellTableColumnsNode(graph, 'classes', {counts, ${state.classValue === 'confirmed' ? 'sumValues' : ''}}, {values, mask});

graph.add(new GPUClassBreaks({
  values, mask, parameters,               // getGPUClassBreaksParameterValues({method: '${state.classMethod}', classCount: ${state.classCount}}, 8)
  maximumClassCount: 8, methods: ['quantile', 'natural-breaks', /* ... */],
  output: {breaks, classCount}
}));
graph.add(new GPUColorScale({
  values, mask, domain: breaks, domainCount: classCount, palette,
  parameters: colorScaleParameters,       // scale: 'threshold'
  maximumDomainCount: 9, maximumPaletteCount: 8,
  output: {colors}                        // packed rgba8, read by the layer
}));`;
    }
    return `import {addPeriodComparisonRecipe} from '@luma.gl/experimental/gpu-spatial-analysis';

// Two periods = two masks over the same observation rows.
const recipe = addPeriodComparisonRecipe(graph, {
  family: 'quadbin', resolution: ${state.compareResolution},
  tableCapacity: 1 << 14, unionCapacity: 1 << 15,
  before: {positions${state.compareMeasure === 'sum' ? ', values: researchGrade' : ''}, mask: firstHalf},
  after:  {positions${state.compareMeasure === 'sum' ? ', values: researchGrade' : ''}, mask: secondHalf},
  measure: '${state.compareMeasure}', zScore: '${state.zScoreKind}',
  output: {cells, delta, ratio, percentChange, zScore, count, overflow},  // GPUCellTableCompare columns
  classify: 'delta',
  classBreaksParameters,   // custom edges [-4b, -b, b, 4b], or standard-deviation
  maximumClassCount: 5, methods: ['custom', 'standard-deviation'],
  palette, colorScaleParameters, maximumPaletteCount: 5, colors
});`;
  },

  about: {
    what: '`GPUCellPyramid` aggregates the points once at the finest Quadbin resolution and rolls the table up level by level; `GPUCellLevelSelection` picks the level for the current zoom on the GPU. `GPUCellRollup` rolls a table up exactly to any coarser resolution. The `addCellTableColumnsNode` adapter turns a table into a value column, which `GPUClassBreaks` and `GPUColorScale` classify and color. `addPeriodComparisonRecipe` aggregates two periods and `GPUCellTableCompare` outer-joins them into delta, ratio, percent change and a z-score.',
    why: 'Zoomable maps normally re-aggregate on every zoom, or fetch pre-built tiles. A GPU pyramid keeps every level resident, so zooming is a one-word write. Change maps answer the planner\'s question "where did it get worse" without reducing two maps to two choropleths by eye.',
    howToRead:
      'Pyramid and roll-up: brighter cells hold more observations per km2 (log scale). Class map: each color is a class of cells, by research-grade observations or observations. Comparison: red cells have more observations in the later period, blue fewer, gray no meaningful change; switch to the z-score to see which changes exceed what Poisson noise would produce.'
  },

  create: async ctx => (await import('./cell-pyramid.compute')).createCellPyramid(ctx),

  story: [
    {
      id: 'question',
      title: 'Can one table serve every zoom level?',
      body: 'A city-wide nature map should show neighbourhoods when you are zoomed out and blocks when you are zoomed in, without waiting for a server to re-aggregate. Here **`GPUCellPyramid`** keys all 43,557 observations into **Quadbin** cells once, at tile zoom 17 (about 230 m), then rolls the table up level by level to zoom 4.\n\nBright cells hold more observations per km2 (log scale, so levels stay comparable). Zoom and pan: the cells change resolution but nothing is recomputed. **Group** below filters the observations in every view (try Birds or Fungi).',
      options: {view: 'pyramid'},
      camera: {...CHICAGO_VIEW, transitionMs: 1200},
      controls: ['category'],
      readouts: ['points', 'pyramid']
    },
    {
      id: 'levels',
      title: 'The level is chosen on the GPU',
      body: "Every frame, the map zoom goes into a one-word buffer and **`GPUCellLevelSelection`** publishes that level's row count and first row into an indirect draw record. The layer simply replays the record: no CPU work, no rebuild. The **Under the hood** drawer shows the pyramid graph encoded once (and again only when you change the filter).\n\nSlide **Cells finer than the zoom level** below to trade detail for noise, or tick **Choose the level by hand** and then slide **Quadbin resolution** to compare resolutions at one camera.",
      options: {resolutionOffset: 4},
      camera: {longitude: MONTROSE[0], latitude: MONTROSE[1], zoom: 13.4, transitionMs: 2000},
      callout: {coordinate: MONTROSE, text: 'Montrose Point: the cells follow your zoom'},
      highlight: {readout: 'activeCells'},
      controls: ['resolutionOffset', 'manualLevel', 'manualResolution'],
      readouts: ['activeCells', 'zoom', 'pyramid']
    },
    {
      id: 'rollup',
      title: 'Roll up exactly',
      body: '**`GPUCellRollup`** reads the finest table and merges it to a coarser **parent resolution**. Counts and fixed-point sums are integers, so a parent holds exactly the sum of its children: the readouts compare the totals before and after and say *conserved exactly* and *bit-exact*. Floating-point sums could not promise that.\n\nThe table also carries the sum of the 0/1 research-grade flag, so the **research-grade share** (research-grade observations / observations) is available per cell for free. Raise **Parent resolution** below, or switch **Color the roll-up by** between density and research-grade share: each resolution is a small graph compiled the first time you use it.',
      options: {view: 'rollup', rollupResolution: 14, rollupColor: 'researchShare'},
      camera: {...CHICAGO_VIEW, transitionMs: 1400},
      highlight: {readout: 'rollupCounts'},
      controls: ['rollupResolution', 'rollupColor', 'rateCeiling'],
      readouts: ['rollupCounts', 'rollupSums', 'rollupCells']
    },
    {
      id: 'classes',
      title: 'From a table to a classified map',
      body: 'An analyst usually wants a choropleth, not raw counts. The adapter **`addCellTableColumnsNode`** turns the roll-up table into a value column plus a mask of occupied rows; **`GPUClassBreaks`** computes the class edges on the GPU and **`GPUColorScale`** colors the cells, with no readback in between.\n\nSwitch the **Classification method** below: quantile gives every class the same number of cells, natural breaks minimises within-class variance, head/tail suits the heavy-tailed research-grade observations. The **Class edges** readout below lists the edges the GPU found.',
      options: {
        view: 'classes',
        classValue: 'confirmed',
        classMethod: 'quantile',
        classCount: 5,
        rollupResolution: 15
      },
      camera: {...CHICAGO_VIEW, transitionMs: 1400},
      highlight: {readout: 'classEdges'},
      controls: ['classMethod', 'classCount', 'classValue'],
      readouts: ['classEdges']
    },
    {
      id: 'compare',
      title: 'Which half of the year was busier, where?',
      body: 'Chicago has one year of data here (2023), so the comparison is **January to June against July to December**. **`addPeriodComparisonRecipe`** keys each period into its own cell table (`GPUCellAggregation`), outer-joins them with **`GPUCellTableCompare`** and classifies the change around zero.\n\nRed cells gained observations in the second half, blue lost them, gray changed less than the **No-change band**. Move **Earlier period (months)** and **Later period (months)** to compare other seasons. The second half has 23,167 observations against 20,390 in the first, but the year is lopsided: the spring surge around the City Nature Challenge (late April) lands in the first half, so lakefront and park cells can go either way. Look for the cells that go against the citywide trend.',
      options: {view: 'compare', changeColumn: 'classes', stableBand: 8},
      camera: {...CHICAGO_VIEW, transitionMs: 1400},
      highlight: {readout: 'compareTotals'},
      controls: ['periodA', 'periodB', 'stableBand', 'changeClassing'],
      readouts: ['compareTotals', 'compareGainLoss']
    },
    {
      id: 'columns',
      title: 'Delta, ratio, percent or z-score?',
      body: 'The join writes four columns and each answers a different question. **Delta** (after - before) finds absolute change, which the lakefront parks dominate. **Ratio** and **percent change** find relative change, but are undefined where there was nothing before (gray) and explode on tiny counts. The **z-score** divides by the Poisson noise, `delta / sqrt(before + after)`, so only changes larger than chance stand out.\n\nHere is winter (Jan-Mar, 3,672 observations) against summer (Jul-Sep, 17,145) as z-scores, so nearly everything is red (**Show** below picks the column). Hover a cell for its before and after counts. Set **z-score** to *Standardized (across cells)* to score each cell against the spread of all changes instead.',
      options: {
        view: 'compare',
        periodA: [1, 3],
        periodB: [7, 9],
        changeColumn: 'zScore',
        colorLimit: 60
      },
      camera: {longitude: -87.66, latitude: 41.86, zoom: 11.2, transitionMs: 1600},
      highlight: {readout: 'compareSignificant'},
      controls: ['changeColumn', 'zScoreKind', 'periodA', 'periodB'],
      readouts: ['compareSignificant', 'compareLargestGain', 'compareLargestLoss']
    },
    {
      id: 'limits',
      title: 'Limits, and things to try',
      body: 'Caveats: seasons are not years, so you cannot call a half-year difference a trend; periods of unequal length bias the totals; observations follow the observers, so counts measure effort as much as wildlife, and tile edges change which cell a sighting lands in (the modifiable areal unit problem: change the **Comparison resolution** and watch the pattern move); and the Poisson score ignores that observations are clustered, so treat it as a screening, not a test.\n\nTry it: set **Compared measure** to *Research-grade observations (sum of values)*, set **Group** to *Birds* (spring migration) or *Fungi*, or compare Mar-May (**Earlier period (months)**) with Jun-Aug (**Later period (months)**).',
      options: {view: 'compare', changeColumn: 'classes', periodA: [1, 6], periodB: [7, 12]},
      camera: {...CHICAGO_VIEW, transitionMs: 1400},
      controls: ['compareResolution', 'compareMeasure', 'category', 'periodA', 'periodB']
    }
  ]
});
