// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
import {COS_GROUPS, COS_VARIABLES, getCosLegendText, getCosVariable} from './b7-cos-style';
import type {ChangeOfSupportOptions} from './change-of-support.compute';

const ITERATION_LADDER = [0, 2, 8, 32, 128, 512];

/** Chapter `interpolation`, scene 2: census tracts moved to hexagons and community areas. */
export default defineScene<ChangeOfSupportOptions>({
  id: 'change-of-support',
  title: 'Move Chicago’s census numbers onto a hexagon grid',
  chapter: 'interpolation',
  order: 2,
  summary:
    'Chicago publishes data by census tract, but planners work in community areas and grids. Move counts, rates and categories between zone systems on the GPU with area-share weights, refine them with a street-density (dasymetric) raster, and build Tobler’s smooth mass-preserving surface.',
  contributors: ['GPUArealInterpolation', 'GPUPycnophylactic', 'addChangeOfSupportRecipe'],
  datasets: [
    {
      id: 'chicago-tracts',
      role: 'source zones with SVI, ACS, PLACES, nature observation and jobs columns'
    },
    {id: 'chicago-community-areas', role: 'target zones'},
    {id: 'chicago-roads', role: 'street density for dasymetric weights'},
    {id: 'chicago-places', role: 'point-of-interest density for dasymetric weights'}
  ],
  initialView: {longitude: -87.78, latitude: 41.84, zoom: 9.9},

  options: [
    {
      kind: 'select',
      id: 'variable',
      label: 'Tract variable',
      group: 'Source data',
      apply: 'param',
      default: 'population',
      help: 'What is moved. Counts are extensive (they add up); rates are intensive (they must be re-derived from counts); the category shows how a label is transferred. Switching rewrites a table of three columns per tract; nothing is recompiled.',
      options: COS_VARIABLES.map(variable => ({
        value: variable.id,
        label: variable.label,
        help: variable.help
      }))
    },
    {
      kind: 'select',
      id: 'ancillary',
      label: 'Dasymetric weights',
      group: 'Source data',
      apply: 'param',
      default: 'none',
      help: 'An ancillary raster that says where within a zone the mass really is. With weights, every area becomes a sum of cell weights (the cellWeights input) instead of a count of cells. Without, a zone is assumed uniform.',
      options: [
        {value: 'none', label: 'None: zones are uniform', help: 'Every raster cell weighs 1.'},
        {
          value: 'streets',
          label: 'OpenStreetMap street density',
          help: 'Street length per cell, smoothed: a proxy for built-up land. Parks, rail yards, water and the airport get little weight.'
        },
        {
          value: 'places',
          label: 'Places (points of interest) density',
          help: 'Overture Maps places per cell, smoothed: where businesses and services are. A better proxy for jobs than for residents.'
        }
      ]
    },
    {
      kind: 'select',
      id: 'target',
      label: 'Target zones',
      group: 'Target zones',
      apply: 'compile',
      default: 'hexagon',
      help: 'The system the values are moved to. Each choice is its own compiled recipe graph, compiled the first time it is selected and cached.',
      options: [
        {value: 'hexagon', label: 'Hexagon grid', help: 'GPUGridGenerator, pointy-top cells.'},
        {value: 'square', label: 'Square grid'},
        {value: 'triangle', label: 'Triangle grid'},
        {
          value: 'community',
          label: 'Community areas (77)',
          help: 'The city’s official reporting geography: unequal areas, real boundaries.'
        }
      ]
    },
    {
      kind: 'slider',
      id: 'cellWidth',
      label: 'Grid cell width',
      group: 'Target zones',
      apply: 'param',
      min: 700,
      max: 3000,
      step: 100,
      default: 1200,
      unit: 'm',
      disabledWhen: state => state.target === 'community',
      help: 'Flat-to-flat width of a hexagon, side of a square, base of a triangle. A per-frame grid parameter: the lattice is regenerated and re-rasterized without recompiling. Below about 5 raster cells the boundary error dominates.'
    },
    {
      kind: 'slider',
      id: 'gridShift',
      label: 'Grid offset',
      group: 'Target zones',
      apply: 'param',
      min: 0,
      max: 1,
      step: 0.1,
      default: 0,
      disabledWhen: state => state.target === 'community',
      format: value => `${Math.round(value * 100)}% of a cell`,
      help: 'Slides the grid diagonally. The numbers in each cell change although the data did not: the modifiable areal unit problem.'
    },
    {
      kind: 'select',
      id: 'denominator',
      label: 'Area denominator',
      group: 'Transfer',
      apply: 'compile',
      default: 'overlap',
      help: 'Which area a share is divided by. Compile-time in GPUArealInterpolation.',
      options: [
        {
          value: 'overlap',
          label: 'Overlap only',
          help: 'Divide by the part of the zone that overlaps the other system: mass is conserved over the shared extent.'
        },
        {
          value: 'zone',
          label: 'Whole zone (tobler)',
          help: 'Divide by the full zone area: mass that falls outside the other system is lost, and intensive values are diluted at coasts.'
        }
      ]
    },
    {
      kind: 'select',
      id: 'rule',
      label: 'Transfer rule',
      group: 'Transfer',
      apply: 'param',
      default: 'conserving',
      help: 'Which of the computed transfers is drawn. All three come out of the same graph run (columns of the same weights).',
      options: [
        {
          value: 'conserving',
          label: 'By variable type (extensive, ratio of extensives)',
          help: 'Counts: mass split by area share. Rates: transfer numerator and denominator as counts, then divide.'
        },
        {
          value: 'area-mean',
          label: 'Area-weighted mean of tract values (intensive)',
          help: 'Tobler’s intensive rule: right for a density or an average, not mass-conserving for counts.'
        },
        {
          value: 'naive',
          label: 'Treat the tract value as a count (wrong)',
          help: 'Splits a rate or density by area like mass. Shown on purpose.'
        }
      ]
    },
    {
      kind: 'slider',
      id: 'iterations',
      label: 'Pycnophylactic iterations',
      group: 'Smooth surface',
      apply: 'compile',
      min: 0,
      max: ITERATION_LADDER.length - 1,
      step: 1,
      default: 3,
      format: value => `${ITERATION_LADDER[value]} iterations`,
      help: 'Rounds of focal-mean smoothing with mass restoration. Compile-time, so each step is its own cached graph. 0 is the flat tract density.'
    },
    {
      kind: 'select',
      id: 'kernel',
      label: 'Smoothing neighbourhood',
      group: 'Smooth surface',
      apply: 'compile',
      default: 'rook',
      help: 'Tobler’s original averages the 4 edge neighbours (rook); the box kernel averages the 3 × 3 block.',
      options: [
        {value: 'rook', label: 'Rook (4 neighbours)'},
        {value: 'box', label: 'Box (3 × 3)'}
      ]
    },
    {
      kind: 'button',
      id: 'measure',
      label: 'Time the unweighted fast path against the generic path',
      group: 'Compare',
      help: 'Without dasymetric weights GPUArealInterpolation reads pair areas off run lengths and zone areas off counts. Times both on scratch outputs outside the frame.'
    },
    {
      kind: 'select',
      id: 'view',
      label: 'Show',
      group: 'View',
      apply: 'param',
      default: 'target',
      help: 'Source tracts, the transferred target zones, or the smooth pycnophylactic surface.',
      options: [
        {value: 'source', label: 'Source: census tracts'},
        {value: 'target', label: 'Target: transferred values'},
        {
          value: 'surface',
          label: 'Smooth surface (pycnophylactic)',
          help: 'Tobler’s mass-preserving surface from the tract totals. Categories have none.'
        }
      ]
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp',
      group: 'Display',
      apply: 'param',
      default: 'viridis',
      help: 'The legend range is the 1st to 98.5th percentile of the drawn values, so a few extreme tracts do not wash out the map.',
      options: [
        {value: 'viridis', label: 'Viridis'},
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'cividis', label: 'Cividis (color-blind optimised)'}
      ]
    },
    {
      kind: 'slider',
      id: 'opacity',
      label: 'Fill opacity',
      group: 'Display',
      apply: 'param',
      min: 0.3,
      max: 1,
      step: 0.05,
      default: 0.88,
      help: 'Lower it to read the basemap.'
    },
    {
      kind: 'toggle',
      id: 'showTractOutlines',
      label: 'Tract outlines',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Thin outlines of the 791 source tracts.'
    },
    {
      kind: 'toggle',
      id: 'showTargetOutlines',
      label: 'Target outlines',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Thick outlines of the target zones that overlap any tract (drawn from GPU vertices by a kernel pass).'
    }
  ],

  readouts: [
    {id: 'tracts', label: 'Source zones'},
    {
      id: 'raster',
      label: 'Shared raster',
      help: 'Both zone systems are rasterized to this one grid; every area is a count (or a weighted sum) of its cells.'
    },
    {
      id: 'contributors',
      label: 'Recipe chain',
      help: 'The contributors addChangeOfSupportRecipe added to this graph, in order.'
    },
    {
      id: 'conservation',
      label: 'Totals, tracts → targets',
      help: 'The sum of the transferred counts against the sum of the tract counts. Extensive transfers conserve mass up to the raster resolution.'
    },
    {
      id: 'pairs',
      label: 'Overlap pairs (target, tract)',
      help: 'Distinct (target, source) pairs sharing at least one cell, against the compiled slot capacity.'
    },
    {id: 'rasterCrossings', label: 'Rasterization'},
    {id: 'coverage', label: 'Coverage'},
    {id: 'resolution', label: 'Resolution'},
    {
      id: 'pycnoCheck',
      label: 'Pycnophylactic check',
      help: 'Sums the smooth surface over every tract and compares it with the tract total.'
    },
    {id: 'fastPath', label: 'Unweighted fast path'}
  ],

  legends: state => {
    const meta = getCosVariable(state.variable);
    if (meta.kind === 'category' && state.view !== 'surface') {
      return [
        {
          kind: 'categories',
          title:
            state.view === 'source'
              ? 'Largest group in the tract'
              : 'Group with the largest overlap area',
          entries: COS_GROUPS.map(group => ({color: group.color, label: group.label})),
          note: 'Census 2020 / ACS counts from the CDC SVI 2022 file.'
        }
      ];
    }
    const text = getCosLegendText(state);
    return [
      {
        kind: 'ramp',
        id: 'values',
        title: text.title,
        ramp: state.ramp,
        unit: text.unit,
        extent: 'gpu',
        format: value =>
          Math.abs(value) >= 100 ? Math.round(value).toLocaleString('en-US') : value.toFixed(1)
      }
    ];
  },

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUPolygonRasterization} from '@luma.gl/experimental/gpu-raster';
import {
  addChangeOfSupportRecipe, GPUGridGenerator, GPUPycnophylactic
} from '@luma.gl/experimental/gpu-spatial-analysis';

const graph = new GPUCommandGraph(device, {id: 'change-of-support'});
${
  state.target === 'community'
    ? '// target zones: the 77 community-area polygons (GeoArrow offsets)'
    : `graph.add(new GPUGridGenerator({
  gridType: '${state.target === 'hexagon' ? 'hex' : state.target}', columns: 56, rows: 76,
  parameters: gridParameters.importToGraph(graph),   // [minX, minY, ${state.cellWidth} m, pitch]
  output: {positions: cellVertices}
}));`
}
const result = addChangeOfSupportRecipe(graph, {
  width: 320, height: 384, extent: extent.importToGraph(graph),
  source: {...tractPolygons, crossingCapacity: 1 << 18, raster: sourceZones},
  target: {...${state.target === 'community' ? 'communityPolygons' : 'gridPolygons'}, crossingCapacity: 1 << 18, raster: targetZones},${
    state.ancillary === 'none'
      ? ''
      : `
  cellWeights,                    // dasymetric raster: ${state.ancillary === 'streets' ? 'street length' : 'places'} per cell`
  }
  denominator: '${state.denominator}',
  pairCapacity: 1 << 15,
  sourceValues, columnCount: 3,   // numerator, denominator, the tract's own rate
  extensiveValues, intensiveValues,
  categories: {sourceCategories, categoryCount: 5, output: shares}
});
graph.add(new GPUPycnophylactic({
  width: 320, height: 384, zones: sourceZones, zoneCount: 791, totals,
  iterations: ${ITERATION_LADDER[state.iterations]}, kernel: '${state.kernel}', output: surface
}));
const compiled = graph.compile();             // once
compiled.encode(commandEncoder, {parameters: undefined});
// rule: extensive = rows of extensiveValues; rate = extensive[0] / extensive[1] * scale`,

  about: {
    what: '`addChangeOfSupportRecipe` rasterizes a source and a target zone system onto one raster, lets `GPUArealInterpolation` count (or weight) how much of each source lies in each target, and transfers values through `GPUSpatialLag` over those area-share weights. Extensive transfer splits a source total by area: `w = a_st / A_s`. Intensive transfer takes an area-weighted mean: `w = a_st / B_t`. `GPUPycnophylactic` builds Tobler’s smooth surface that keeps every tract total.',
    why: 'Data arrive in the zones someone else drew: tracts, ZIP codes, school districts. Plans, models and grids need other zones. Moving a number across mismatched boundaries is the modifiable areal unit problem in practice, and the method (count versus rate, uniform versus dasymetric) decides the answer. It matches tobler’s area_interpolate and pycno in Python.',
    howToRead:
      'Each target zone shows the transferred value in the legend units. A hexagon over several tracts is an area-share blend; thick outlines are the targets and thin lines the tracts. The Totals readout tells you whether the transfer conserved mass, and the Resolution readout bounds the error at the zone boundaries.'
  },

  create: async ctx => (await import('./change-of-support.compute')).createChangeOfSupport(ctx),

  story: [
    {
      id: 'the-problem',
      title: 'How many people live in each hexagon?',
      body: 'Chicago’s 2020 census counts are published for **791 tracts**: irregular zones drawn to hold about 4,000 people, so a tract in the Loop is tiny and one on the Southwest Side is huge. A planner wants the same numbers on a **regular hexagon grid**, or for the 77 **community areas** the city uses for reporting. The boundaries do not line up, so someone has to decide how to cut a tract in two.\n\nThat is **change of support**. Here are the raw tract populations (2.7 million residents), with the tract outlines; **Tract variable**, **Target zones** and **Show** below choose what is moved, where to and what is drawn. Hover a tract for its identifier and population.',
      camera: {longitude: -87.78, latitude: 41.84, zoom: 9.9, transitionMs: 1500},
      options: {view: 'source', variable: 'population', target: 'hexagon'},
      controls: ['variable', 'target', 'view'],
      readouts: ['tracts']
    },
    {
      id: 'extensive',
      title: 'Counts are split by area share',
      body: 'A count is **extensive**: it adds up. `GPUArealInterpolation` rasterizes tracts and hexagons onto one fine grid and counts the cells where tract *s* and hexagon *t* overlap, `aₛₜ`. The share of tract *s* that lands in hexagon *t* is `wₜₛ = aₛₜ / Aₛ`, so a hexagon holds `Σₛ wₜₛ xₛ` residents and every tract’s people are divided, never duplicated.\n\nThe **Totals, tracts → targets** readout checks it: tracts in, hexagons out, within a rounding fraction of a percent. Slide the **Grid cell width** (a per-frame parameter) and try **Grid offset**: the same data on a shifted grid gives different cell values, the modifiable areal unit problem. This is the recipe `addChangeOfSupportRecipe` assembles from a rasterizer, `GPUArealInterpolation` and `GPUSpatialLag`; it matches `tobler.area_interpolate` with extensive variables.',
      camera: {longitude: -87.78, latitude: 41.84, zoom: 10.1, transitionMs: 1500},
      options: {view: 'target', cellWidth: 1200},
      controls: ['cellWidth', 'gridShift'],
      readouts: ['conservation']
    },
    {
      id: 'rates',
      title: 'Rates must be rebuilt from counts',
      body: 'A poverty rate is **not** extensive: two tracts at 25% do not make a 50% hexagon. The textbook alternative is the **intensive** rule, the area-weighted mean of tract rates `wₜₛ = aₛₜ / Bₜ`, which is what you see now. It treats a nearly empty lakefront tract like a crowded block of the same size.\n\nThe better answer here moves the *counts* (people below 150% of poverty, and residents) as extensive transfers, then divides: switch **Transfer rule** to *By variable type (extensive, ratio of extensives)*. The map changes where tracts are very unequal in population, and the **Totals, tracts → targets** readout shows the two counts it moved (people below the line, and residents). The third choice sums rates as if they were counts: the numbers are meaningless, which is why the label says wrong.',
      camera: {longitude: -87.7, latitude: 41.85, zoom: 10.4, transitionMs: 1500},
      options: {variable: 'poverty', rule: 'area-mean'},
      controls: ['variable', 'rule'],
      readouts: ['conservation']
    },
    {
      id: 'dasymetric',
      title: 'Tell the weights where people actually are',
      body: 'Uniform zones put people in parks, rail yards and runways. A **dasymetric** raster fixes that: `cellWeights` makes each area a *sum of weights*, so a tract’s residents are split in proportion to where the weight is. Here the weight is OpenStreetMap street length per cell, a rough proxy for built-up land.\n\nWatch the lakefront parks (Jackson and Washington Park) and the big industrial blocks lose people to the neighbouring streets, while totals stay conserved. Honest limit: street density is only a proxy; real dasymetric work uses land cover, building footprints or night lights. Set **Dasymetric weights** to *Places (points of interest) density* with **Tract variable** *Jobs by workplace (count)*, where businesses are the better guide.',
      camera: {longitude: -87.62, latitude: 41.78, zoom: 11.0, transitionMs: 1800},
      options: {variable: 'population', rule: 'conserving', ancillary: 'streets', cellWidth: 700},
      callout: {coordinate: [-87.6, 41.79], text: 'Jackson & Washington Parks'},
      controls: ['ancillary', 'variable'],
      readouts: ['conservation']
    },
    {
      id: 'community-areas',
      title: 'Real boundaries, and the denominator',
      body: 'Now move the tracts to the **77 community areas**. Tract and community-area boundaries mostly align, but the two files do not cover exactly the same ground, so the choice of **Area denominator** matters. With *Whole zone (tobler)* (the tobler default) a source’s mass is divided by its entire area, and the part that falls outside every target is lost; *Overlap only* divides by the shared part, which conserves what overlaps.\n\nCompare the **Totals, tracts → targets** readout between the two denominators. Community areas have unequal areas, so a count per area is not comparable between them: for fair comparison use a rate. Notice also that the thick outlines are real polygons, rasterized by the same `GPUPolygonRasterization` used for the grids.',
      camera: {longitude: -87.7, latitude: 41.84, zoom: 10.0, transitionMs: 1800},
      options: {target: 'community', denominator: 'zone', ancillary: 'none'},
      controls: ['target', 'denominator'],
      readouts: ['conservation']
    },
    {
      id: 'categories',
      title: 'Even a label can be transferred',
      body: 'Categories cannot be added or averaged. `GPUArealInterpolation` also takes a categorical source (here, the largest of five race and ethnicity groups in each tract) and returns the **share of each target covered by each category**, `share(t, k) = Σₛ∈k aₛₜ / Σₛ aₛₜ`. The map shows the group with the largest area share; hover for all five shares.\n\nThe pattern is the city’s well-known segregation: broad Black, Hispanic and White areas with tracts that change character quickly. Remember that area shares are not population shares: a large, sparse tract can outvote a dense small one.',
      camera: {longitude: -87.7, latitude: 41.84, zoom: 10.0, transitionMs: 1500},
      options: {
        variable: 'dominantGroup',
        target: 'hexagon',
        denominator: 'overlap',
        cellWidth: 1200
      },
      controls: ['variable']
    },
    {
      id: 'surface',
      title: 'A smooth surface that keeps every total, then try your own',
      body: '**`GPUPycnophylactic`** (Tobler 1979) builds a continuous density surface instead of zones. It starts with each tract’s density, repeatedly replaces every cell by the average of its neighbours, then rescales each tract so its total is restored and nothing goes negative. Boundaries between tracts melt away and people are spread smoothly, yet **every tract total is exact**; the readout sums the surface over every tract to prove it.\n\n**Pycnophylactic iterations** is compile-time (each step is its own cached graph): 0 is the flat tract density, 512 is nearly a smooth hill. Limits: it assumes people spread smoothly, ignoring parks and the lake, and it has no ancillary data or barriers; the dasymetric weights above do not apply to it. Try the *Introduced share of observations (rate)* variable, **Smoothing neighbourhood** *Box (3 × 3)*.',
      camera: {longitude: -87.7, latitude: 41.84, zoom: 10.0, transitionMs: 1500},
      options: {variable: 'population', view: 'surface', iterations: 3, ramp: 'inferno'},
      controls: ['view', 'iterations', 'kernel', 'variable'],
      readouts: ['pycnoCheck']
    }
  ]
});
