// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
import {DOT_VALUE_SETS} from './b7-dot-style';
import type {DotDensityOptions} from './dot-density.compute';

const formatPeople = (log: number) => {
  const value = 10 ** log;
  return value >= 10 ? `${Math.round(value)}` : value.toFixed(1);
};

/** Chapter `interpolation`, scene 3: a dasymetric dot map of Chicago. */
export default defineScene<DotDensityOptions>({
  id: 'dot-density',
  title: 'One dot per person: a dot map of Chicago',
  chapter: 'interpolation',
  order: 3,
  summary:
    'Draw census counts as stable random dots inside each tract, colored by group, on the GPU. Dots are added and never move as you zoom or change the dot value; a street-density mask pulls them out of parks and rail yards; a second contributor scatters uniform random points for comparison.',
  contributors: ['GPUDotDensity', 'GPURandomPointsInPolygon'],
  datasets: [
    {id: 'chicago-tracts', role: 'tract polygons with race, poverty and vehicle counts'},
    {id: 'chicago-roads', role: 'street density for the dasymetric mask'},
    {id: 'chicago-places', role: 'place density for the dasymetric mask'}
  ],
  initialView: {longitude: -87.78, latitude: 41.84, zoom: 10},

  options: [
    {
      kind: 'select',
      id: 'valueSet',
      label: 'What the dots count',
      group: 'Dots',
      apply: 'param',
      default: 'race',
      help: 'Each set is up to five categories per tract. Switching rewrites the values buffer; the graph is not recompiled (the category count is a compile-time five, unused categories are zero).',
      options: (Object.keys(DOT_VALUE_SETS) as (keyof typeof DOT_VALUE_SETS)[]).map(id => ({
        value: id,
        label: DOT_VALUE_SETS[id].label,
        help: DOT_VALUE_SETS[id].help
      }))
    },
    {
      kind: 'slider',
      id: 'logUnitsPerDot',
      label: 'Residents per dot at zoom 10',
      group: 'Dots',
      apply: 'param',
      min: 0,
      max: 2.3,
      step: 0.1,
      default: 1.4,
      format: formatPeople,
      help: 'The dot value, on a log scale. A per-frame parameter (dots per unit): a larger value draws fewer dots. Smaller values are capped so the dot buffer never overflows.'
    },
    {
      kind: 'slider',
      id: 'zoomCoupling',
      label: 'Zoom coupling',
      group: 'Dots',
      apply: 'param',
      min: 0,
      max: 2,
      step: 0.25,
      default: 1.5,
      format: value =>
        value === 0
          ? 'off: fixed dot value'
          : `dots per resident ×${(2 ** value).toFixed(2)} per zoom level`,
      help: 'Ties the dot value to the zoom: zooming in appends dots (each zoom level multiplies dots per resident by 2 to this power). 2 keeps about the same dots per screen pixel at every zoom. Existing dots never move.'
    },
    {
      kind: 'select',
      id: 'mask',
      label: 'Dasymetric mask',
      group: 'Dasymetric mask',
      apply: 'param',
      default: 'none',
      help: 'A raster of weights in [0, 1]. A candidate dot position is kept with probability equal to the weight of its cell, so dots concentrate where the weight is high. The mask raster is a compile-time input; its contents change here without a recompile.',
      options: [
        {value: 'none', label: 'None: uniform inside each tract'},
        {
          value: 'streets',
          label: 'Street density',
          help: 'OpenStreetMap street length per cell, smoothed. Parks, rail yards, industry and the airport thin out.'
        },
        {
          value: 'places',
          label: 'Places density',
          help: 'Overture Maps points of interest per cell, smoothed: pulls dots toward commercial corridors.'
        }
      ]
    },
    {
      kind: 'select',
      id: 'display',
      label: 'Show',
      group: 'Compare random points',
      apply: 'param',
      default: 'dots',
      help: 'GPURandomPointsInPolygon is the same sampler without categories: N uniform points per polygon.',
      options: [
        {value: 'dots', label: 'Dots by category (GPUDotDensity)'},
        {value: 'random', label: 'Uniform random points (GPURandomPointsInPolygon)'}
      ]
    },
    {
      kind: 'select',
      id: 'randomCounts',
      label: 'Points per tract',
      group: 'Compare random points',
      apply: 'param',
      default: 'match',
      disabledWhen: state => state.display !== 'random',
      help: 'Match the dot map (one point per dot, uncolored), or the same fixed number in every tract.',
      options: [
        {value: 'match', label: 'Same count as the dot map'},
        {value: 'fixed', label: 'The same in every tract'}
      ]
    },
    {
      kind: 'slider',
      id: 'randomFixed',
      label: 'Fixed points per tract',
      group: 'Compare random points',
      apply: 'param',
      min: 10,
      max: 400,
      step: 10,
      default: 100,
      disabledWhen: state => state.display !== 'random' || state.randomCounts !== 'fixed',
      help: 'The per-feature counts buffer is rewritten; raising it only appends points.'
    },
    {
      kind: 'button',
      id: 'reseed',
      label: 'New seed (every dot moves)',
      group: 'Randomness',
      help: 'Each dot is a pure function of the seed, its tract, its category and its rank. A new seed redraws every dot; the pattern at the tract level stays the same.'
    },
    {
      kind: 'button',
      id: 'check',
      label: 'Check that dots never move (GPU read-back)',
      group: 'Compare',
      help: 'Runs the dot graph at a coarse and a finer dot value, reads both outputs back and compares them slot by slot.'
    },
    {
      kind: 'button',
      id: 'measure',
      label: 'Time both graphs',
      group: 'Compare',
      help: 'Runs GPUDotDensity and GPURandomPointsInPolygon outside the frame and reports GPU time.'
    },
    {
      kind: 'slider',
      id: 'dotRadius',
      label: 'Dot size',
      group: 'Display',
      apply: 'param',
      min: 0.8,
      max: 4,
      step: 0.2,
      default: 1.4,
      unit: 'px',
      help: 'Radius in CSS pixels.'
    },
    {
      kind: 'toggle',
      id: 'showTractOutlines',
      label: 'Tract outlines',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Thin outlines of the 791 tracts that hold the dots.'
    }
  ],

  readouts: [
    {id: 'tracts', label: 'Tracts'},
    {
      id: 'dotValue',
      label: 'Dot value now',
      help: 'Follows the zoom when the coupling is above 0.'
    },
    {
      id: 'dots',
      label: 'Dots drawn',
      help: 'Read back from the GPU draw command, against the compiled capacity.'
    },
    {id: 'perCategory', label: 'Dots per category'},
    {
      id: 'failed',
      label: 'Failed dots',
      help: 'A dot whose rejection-sampling attempts all missed (a thin tract, or a mask that is zero over it) keeps its slot with no position.'
    },
    {id: 'mask', label: 'Mask raster'},
    {id: 'seed', label: 'Seed'},
    {id: 'randomPoints', label: 'Random points'},
    {id: 'stability', label: 'Dot stability'},
    {id: 'graphTime', label: 'Graph timing'}
  ],

  legends: state => {
    const set = DOT_VALUE_SETS[state.valueSet];
    if (state.display === 'random') {
      return [
        {
          kind: 'categories',
          title: 'Uniform random points',
          entries: [{color: [200, 120, 0, 255], label: 'One point (no category)'}],
          note: 'The same sampler as the dot map, without values or categories.'
        }
      ];
    }
    return [
      {
        kind: 'categories',
        title: `One dot = ${formatPeople(state.logUnitsPerDot)} ${set.unit} at zoom 10`,
        entries: set.categories.map(category => ({color: category.color, label: category.label})),
        note: state.zoomCoupling > 0 ? 'Dots per resident rise as you zoom in.' : 'Fixed dot value.'
      }
    ];
  },

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUDotDensity, GPURandomPointsInPolygon, getGPUDotDensityParameterValues
} from '@luma.gl/experimental/gpu-spatial-analysis';

const graph = new GPUCommandGraph(device, {id: 'dot-density'});
graph.add(new GPUDotDensity({
  ...tractPolygons,                     // polygonPositions, featureOffsets, polygonOffsets, ringOffsets
  values,                               // 791 tracts × 5 categories, feature-major
  categoryCount: 5,
  parameters: parameters.importToGraph(graph),
  mask: {weights: maskWeights, width: 480, height: 576},   // dasymetric raster, row 0 = south
  output: {
    positions, categories, slotCounts, slotOffsets,
    dots: {ids, count: graph.importGPUData('count', drawCommands.getInstanceCountData(0)),
           overflow, totalCount},
    failedCount
  }
}));
const compiled = graph.compile();       // once

// Every frame: dots per unit follows the zoom; the dots already drawn never move
const dotsPerUnit = 2 ** (${state.zoomCoupling} * (zoom - 10)) / ${formatPeople(state.logUnitsPerDot)};
parameters.write(getGPUDotDensityParameterValues({seed, dotsPerUnit, maskExtent}));
compiled.encode(commandEncoder, {parameters: undefined});
// draw: SpatialAnalysisPointLayer({positions, drawCommands, values: categories, colormap: 'category'})

// Uniform random points for comparison
const random = new GPURandomPointsInPolygon({
  ...tractPolygons, counts, parameters: randomParameters.importToGraph(graph2), mask, output: randomOutput
});`,

  about: {
    what: '`GPUDotDensity` draws `value × dotsPerUnit` dots for every (tract, category), at random positions inside the tract (rejection sampling against the polygon), optionally thinned by a dasymetric weight raster. Counter-based random numbers make each dot a pure function of (seed, tract, category, rank). `GPURandomPointsInPolygon` places N uniform points per polygon, using the same sampler.',
    why: 'A choropleth hides how people are distributed inside a zone and makes big, empty tracts dominate. A dot map shows density and mixing at once, and because dots are stable, zooming in refines the picture instead of reshuffling it. Dasymetric weights keep dots out of parks and industry.',
    howToRead:
      'Each dot stands for the number of residents shown in the legend; the colors are the categories. The dots inside a tract are random, so look at the overall mixture and density, never at an individual dot: a dot says nothing about where a person lives.'
  },

  create: async ctx => (await import('./dot-density.compute')).createDotDensity(ctx),

  story: [
    {
      id: 'the-map',
      title: 'Who lives where in Chicago?',
      body: 'A choropleth of Chicago’s 791 census tracts hides two things: how many people live in a tract, and how mixed it is. A **dot map** shows both. **`GPUDotDensity`** draws one dot per N residents of each group, at a random position inside the tract, all on the GPU: about 100,000 dots here, colored by race and ethnicity from the 2020 Census (**What the dots count**, below).\n\nThe broad pattern is the one Chicago is known for: large, nearly uniform areas on the South and West sides, mixed neighbourhoods on the North Side and along the lakefront, and the Loop’s thin population. Hover a tract for its counts and shares.',
      camera: {longitude: -87.78, latitude: 41.84, zoom: 10, transitionMs: 1500},
      options: {valueSet: 'race', display: 'dots'},
      controls: ['valueSet'],
      readouts: ['tracts', 'dots']
    },
    {
      id: 'stable-dots',
      title: 'Zoom in: dots are added, and never move',
      body: 'The number of dots in a (tract, category) slot is `ceil(value × dotsPerUnit − u)`, where `u` is a per-slot random number, so it never decreases as the dot value shrinks. Dot *j* of a slot is placed by rejection sampling from random numbers that depend on the seed, the slot and *j*, not on the dot value. So zooming in only **appends** dots; the ones you see keep their exact position.\n\n**Zoom coupling** ties the dot value to the zoom: at 1.5, each zoom level multiplies dots per resident by about 2.8. Press **Check that dots never move (GPU read-back)** below: the GPU runs the graph at two dot values, reads both back, and the **Dot stability** readout reports how many dots moved (zero).',
      camera: {longitude: -87.65, latitude: 41.85, zoom: 12.4, transitionMs: 2200},
      options: {zoomCoupling: 1.5},
      callout: {coordinate: [-87.6325, 41.8528], text: 'Chinatown'},
      controls: ['zoomCoupling', 'check'],
      readouts: ['dotValue', 'dots', 'stability']
    },
    {
      id: 'segregation',
      title: 'Boundaries you can see',
      body: 'At neighbourhood scale the mixture becomes a map of boundaries: Pilsen and Little Village (Hispanic or Latino, green), Bronzeville and Englewood (Black, orange), Chinatown (Asian, yellow) and the Near North lakefront (White, blue) meet along a few streets. Dots are random inside a tract, so a sharp edge is a *tract* edge, not a street.\n\nSwitch **What the dots count** to *Poverty status* or *Household vehicle access* below to see the same tracts through a different lens; the graph is not recompiled, only the values buffer is rewritten.',
      camera: {longitude: -87.66, latitude: 41.835, zoom: 11.6, transitionMs: 2200},
      options: {valueSet: 'race'},
      controls: ['valueSet'],
      readouts: ['perCategory']
    },
    {
      id: 'dasymetric',
      title: 'Keep dots out of parks and rail yards',
      body: 'Inside a tract, `GPUDotDensity` is uniform: it will put people in a park or on a rail line. A **dasymetric mask** fixes that. A raster of weights in [0, 1] thins candidate positions: a position is kept with probability equal to its cell’s weight. Here the weight is OpenStreetMap street length per cell: where there are no streets, there are few dots.\n\nSet **Dasymetric mask** to *Street density* and watch Washington and Jackson Parks, the rail yards and the lakefront empty out and the streets fill up. Street density is only a proxy for housing: with real building footprints or land cover you would do better. **Failed dots** counts dots whose sampling attempts all missed; a very restrictive mask makes it grow.',
      camera: {longitude: -87.612, latitude: 41.793, zoom: 12.6, transitionMs: 2200},
      options: {mask: 'streets'},
      callout: {coordinate: [-87.6155, 41.7935], text: 'Washington Park'},
      controls: ['mask'],
      readouts: ['failed', 'mask']
    },
    {
      id: 'random-points',
      title: 'The same sampler without categories',
      body: '**`GPURandomPointsInPolygon`** is the sampler on its own: N uniform random points per polygon, with an optional mask, in its own compiled graph. Switch **Show** to *Uniform random points* and the colors disappear: you see where the dots are placed, not who they represent. With **Points per tract** set to *The same in every tract* (**Fixed points per tract**) every tract gets 100 points, which paints small tracts black and big ones sparse, a reminder of why the dot value should be per resident, not per polygon.\n\nThis is also how you build a synthetic population to feed a simulation, or a Monte Carlo sample of a polygon for a statistic.',
      options: {display: 'random', randomCounts: 'fixed', randomFixed: 100},
      controls: ['display', 'randomCounts', 'randomFixed'],
      readouts: ['randomPoints']
    },
    {
      id: 'limits',
      title: 'Limits, and things to try',
      body: 'A dot map is not a census of people: dots are random inside a tract and say nothing about where a person lives, and tract boundaries still cut neighbourhoods. The map shows 2020 counts; a tract’s mix can change in a year. Dasymetric weights from streets or places only shift dots; they do not know where housing is.\n\nTry: **New seed (every dot moves)** (the picture changes in detail, not in pattern), a larger **Residents per dot at zoom 10** at full view, *Places density* as the **Dasymetric mask** (dots follow commercial streets), a **Zoom coupling** of 0 (fixed dot value: zooming in shows the same dots, bigger), and *Poverty status* in **What the dots count** at the city scale.',
      camera: {longitude: -87.78, latitude: 41.84, zoom: 10, transitionMs: 1800},
      options: {display: 'dots', mask: 'none', randomCounts: 'match'},
      controls: ['reseed', 'logUnitsPerDot', 'mask', 'zoomCoupling', 'valueSet']
    }
  ]
});
