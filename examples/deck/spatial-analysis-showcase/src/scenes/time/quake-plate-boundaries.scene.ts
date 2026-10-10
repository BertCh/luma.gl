// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene, type LegendSpec} from '../scene';
import type {QuakePlateBoundariesOptions} from './quake-plate-boundaries.compute';
import {
  BOUNDARY_FAMILY_NAMES,
  QUAKE_DEPTH_CLASS_COLORS,
  QUAKE_REGION_OPTIONS,
  QUAKE_REGIONS
} from './quake-regions';

/** Family colors, as in `BOUNDARY_FAMILY_COLORS` of the compute module (kept here: no engine imports). */
const FAMILY_COLORS = [
  [230, 57, 70],
  [66, 135, 245],
  [240, 200, 40]
] as const;

export default defineScene<QuakePlateBoundariesOptions>({
  id: 'quake-plate-boundaries',
  title: 'Earthquake depth by distance from plate boundaries',
  chapter: 'time',
  order: 6,
  summary:
    'A GPU distance field and grouped statistics combine PB2002 boundaries with 77,000 USGS events to map depth-distance profiles and magnitude frequency; model boundaries, planar windows and catalog completeness limit precision.',
  contributors: ['GPUDistanceField', 'GPUGroupStatistics'],
  datasets: [
    {id: 'poopdeck-earthquakes', role: 'USGS M4+ catalog, 2020 to 2024'},
    {id: 'plate-boundaries', role: 'PB2002 boundary steps (seeds)'}
  ],
  initialView: QUAKE_REGIONS.japan.view,
  basemap: ground('night'),
  furniture: {
    title: {
      title: 'Earthquake depth and boundary distance',
      subtitle: 'USGS events sampled against PB2002 boundaries'
    },
    scaleBar: {units: 'metric'},
    credit: 'US Geological Survey; PB2002 plate-boundary model',
    caveat: 'Boundary positions, planar distance and catalog completeness limit precision.'
  },

  options: [
    {
      kind: 'select',
      id: 'region',
      label: 'Region',
      group: 'Region',
      apply: 'compile',
      default: 'japan',
      options: QUAKE_REGION_OPTIONS,
      help: 'The analysis window: a planar metric frame (meters around its center, distances exact at the center latitude and about 12% off at the top and bottom) with a 1,024 by 1,024 distance grid that extends 450 km past the window so the nearest boundary is never cut off. Built the first time you pick a region.'
    },
    {
      kind: 'select',
      id: 'mapShows',
      label: 'Map shows',
      group: 'Region',
      apply: 'param',
      default: 'distance',
      options: [
        {
          value: 'distance',
          label: 'Distance to the boundary',
          help: 'The distance field, bright at the boundary and darker with distance.'
        },
        {
          value: 'allocation',
          label: 'Nearest boundary family',
          help: 'Euclidean allocation: every cell takes the family of its nearest seed (convergent, divergent or transform).'
        },
        {value: 'events', label: 'Events and boundaries only'}
      ],
      help: 'Which GPUDistanceField output is drawn under the events.'
    },
    {
      kind: 'slider',
      id: 'mapOpacity',
      label: 'Field opacity',
      group: 'Region',
      apply: 'param',
      min: 0.1,
      max: 1,
      step: 0.05,
      default: 0.55,
      help: 'Opacity of the distance or allocation raster.'
    },
    {
      kind: 'select',
      id: 'boundaryClasses',
      label: 'Measure distance to',
      group: 'Distance field',
      apply: 'param',
      default: 'all',
      options: [
        {value: 'all', label: 'Every boundary'},
        {
          value: 'convergent',
          label: 'Convergent boundaries',
          help: 'Subduction zones plus oceanic and continental convergent boundaries.'
        },
        {
          value: 'subduction',
          label: 'Subduction zones only',
          help: 'The PB2002 class SUB: trenches where oceanic lithosphere descends.'
        },
        {
          value: 'divergent',
          label: 'Spreading ridges and rifts',
          help: 'Oceanic spreading ridges and continental rifts.'
        },
        {
          value: 'transform',
          label: 'Transform faults',
          help: 'Oceanic and continental transform faults.'
        }
      ],
      help: 'Which boundary steps are seeds. The seed points and their family ids are rewritten and the field recomputed; nothing recompiles. With every boundary the allocation map shows which family is nearest.'
    },
    {
      kind: 'slider',
      id: 'maximumDistance',
      label: 'Distance cap',
      group: 'Distance field',
      apply: 'param',
      min: 100,
      max: 2000,
      step: 50,
      default: 2000,
      format: value => (value >= 2000 ? 'no limit' : `${value} km`),
      help: 'Cells (and events) farther than this from every seed get no distance (maxDistance). Events beyond the cap drop out of the distance tables and count as far in the Gutenberg-Richter subset.'
    },
    {
      kind: 'select',
      id: 'algorithm',
      label: 'Distance algorithm',
      group: 'Distance field',
      apply: 'compile',
      default: 'exact',
      options: [
        {
          value: 'exact',
          label: 'Exact (Felzenszwalb-Huttenlocher)',
          help: 'Separable lower envelopes: the exact nearest seed of every cell.'
        },
        {
          value: 'jump-flood-0',
          label: 'Jump flooding, no refinement',
          help: 'About log2(1024) = 10 passes; a fast preview with occasional wrong cells.'
        },
        {
          value: 'jump-flood-1',
          label: 'Jump flooding + 1 refinement pass',
          help: 'JFA+1: one extra step-1 pass that fixes most errors.'
        },
        {
          value: 'jump-flood-2',
          label: 'Jump flooding + 2 refinement passes',
          help: 'JFA+2: two extra passes.'
        }
      ],
      help: 'Compile-time choice of GPUDistanceField mode. The jump-flood variants are approximate; compare the depth and distance chart against the exact one. Each variant is compiled the first time you pick it.'
    },
    {
      kind: 'slider',
      id: 'minimumMagnitude',
      label: 'Minimum magnitude (depth tables)',
      group: 'Group statistics',
      apply: 'param',
      min: 4,
      max: 7,
      step: 0.1,
      default: 4.5,
      help: 'Events below this are masked out of the two distance tables and drawn no longer. The mask is a per-row buffer rewritten on the GPU. The Gutenberg-Richter table is never masked: it needs the small events.'
    },
    {
      kind: 'select',
      id: 'binWidth',
      label: 'Distance bin',
      group: 'Group statistics',
      apply: 'param',
      default: '10',
      options: [
        {value: '5', label: '5 km (to 320 km)'},
        {value: '10', label: '10 km (to 640 km)'},
        {value: '20', label: '20 km (to 1,280 km)'},
        {value: '25', label: '25 km (to 1,600 km)'}
      ],
      help: 'Width of the distance bins. The tables hold 64 bins, so a wider bin reaches further. Distances beyond the last bin are left out of the tables.'
    },
    {
      kind: 'slider',
      id: 'shallowLimit',
      label: 'Shallow / intermediate limit',
      group: 'Group statistics',
      apply: 'param',
      min: 20,
      max: 150,
      step: 5,
      default: 70,
      unit: 'km',
      help: 'Events shallower than this are shallow. 70 km is the conventional limit.'
    },
    {
      kind: 'slider',
      id: 'deepLimit',
      label: 'Intermediate / deep limit',
      group: 'Group statistics',
      apply: 'param',
      min: 150,
      max: 500,
      step: 10,
      default: 300,
      unit: 'km',
      help: 'Events deeper than this are deep. 300 km is the conventional limit.'
    },
    {
      kind: 'slider',
      id: 'lowerFraction',
      label: 'Lower percentile',
      group: 'Group statistics',
      apply: 'param',
      min: 0.01,
      max: 0.45,
      step: 0.01,
      default: 0.1,
      format: value => `${Math.round(value * 100)}%`,
      help: 'Shallow edge of the depth band in the chart. The percentile fractions are a parameter buffer, so this recomputes without recompiling.'
    },
    {
      kind: 'slider',
      id: 'upperFraction',
      label: 'Upper percentile',
      group: 'Group statistics',
      apply: 'param',
      min: 0.55,
      max: 0.99,
      step: 0.01,
      default: 0.9,
      format: value => `${Math.round(value * 100)}%`,
      help: 'Deep edge of the depth band in the chart.'
    },
    {
      kind: 'select',
      id: 'variance',
      label: 'Deviation kind',
      group: 'Group statistics',
      apply: 'compile',
      default: 'sample',
      options: [
        {value: 'sample', label: 'Sample (n - 1)'},
        {value: 'population', label: 'Population (n)'}
      ],
      help: 'Whether the standard deviation of depth per distance bin divides by n - 1 or n. A compile-time property of GPUGroupStatistics; it changes the depth spread readout only.'
    },
    {
      kind: 'select',
      id: 'eventColor',
      label: 'Color events by',
      group: 'Events',
      apply: 'param',
      default: 'class',
      options: [
        {value: 'class', label: 'Depth class'},
        {value: 'depth', label: 'Depth (continuous)'}
      ],
      help: 'Depth class uses the three colors of the charts; continuous depth uses the magma ramp (bright shallow, dark deep).'
    },
    {
      kind: 'slider',
      id: 'sizeScale',
      label: 'Marker size',
      group: 'Events',
      apply: 'param',
      min: 0.5,
      max: 3,
      step: 0.1,
      default: 1,
      format: value => `${value.toFixed(1)}x`,
      help: 'Scales every disc; radius grows with magnitude.'
    },
    {
      kind: 'toggle',
      id: 'showLinks',
      label: 'Link deep events to their boundary',
      group: 'Events',
      apply: 'param',
      default: false,
      help: 'Draws a line from each event deeper than the limit below to the center of the cell of its nearest boundary seed (the nearestCells output of the distance field).'
    },
    {
      kind: 'slider',
      id: 'linkMinimumDepth',
      label: 'Link events deeper than',
      group: 'Events',
      apply: 'param',
      min: 20,
      max: 600,
      step: 10,
      default: 100,
      unit: 'km',
      disabledWhen: state => !state.showLinks,
      help: 'Only events at least this deep get a link.'
    },
    {
      kind: 'select',
      id: 'grSubset',
      label: 'Magnitude table uses',
      group: 'Magnitude and frequency',
      apply: 'param',
      default: 'all',
      options: [
        {value: 'all', label: 'All events'},
        {value: 'near', label: 'Events near a boundary'},
        {value: 'far', label: 'Events far from a boundary'}
      ],
      help: 'Which events feed the Gutenberg-Richter table. The subset is chosen on the GPU from the distance at each event.'
    },
    {
      kind: 'slider',
      id: 'grDistance',
      label: 'Near / far distance',
      group: 'Magnitude and frequency',
      apply: 'param',
      min: 25,
      max: 1000,
      step: 25,
      default: 100,
      unit: 'km',
      disabledWhen: state => state.grSubset === 'all',
      help: 'Events within this distance of a boundary are near; the rest are far.'
    },
    {
      kind: 'slider',
      id: 'completeness',
      label: 'Completeness magnitude (Mc)',
      group: 'Magnitude and frequency',
      apply: 'param',
      min: 4,
      max: 6.5,
      step: 0.1,
      default: 4.5,
      format: value => `M${value.toFixed(1)}`,
      help: 'The b-value is fitted to events at or above Mc. Below the completeness magnitude a catalog misses events, so the curve flattens and the fit would be too low. Fitting happens on the CPU from the counts the table returned.'
    }
  ],

  story: [
    {
      id: 'boundaries',
      title: 'Earthquakes follow plate boundaries',
      headline: 'Earthquakes concentrate along mapped plate boundaries',
      textAlternative:
        'Magnitude-scaled, depth-colored earthquakes trace convergent, divergent and transform boundaries around Japan.',
      body: "Every disc is a magnitude 4.5+ earthquake of 2020 to 2024 from the USGS catalog; radius is **magnitude** and color is **depth class**. The lines are the 5,824 boundary steps of Peter Bird's PB2002 plate model, drawn in red where plates converge, blue where they separate and yellow where they slide past. Around Japan the events trace the Kuril, Japan, Izu-Bonin and Ryukyu trenches.\n\nThe question here is quantitative: *how far* from a boundary does an earthquake happen, and does that distance say anything about depth? Change **Region** below to look at another subduction margin; **Map shows** switches the raster under the events.",
      camera: {...QUAKE_REGIONS.japan.view, pitch: 0, bearing: 0, transitionMs: 1500},
      options: {
        region: 'japan',
        mapShows: 'events',
        boundaryClasses: 'all',
        minimumMagnitude: 4.5
      },
      controls: ['region', 'mapShows'],
      readouts: ['events', 'seeds']
    },
    {
      id: 'distance',
      title: 'Distance to the boundary, everywhere',
      headline: 'Most events occur near a boundary',
      textAlternative:
        'A sequential raster brightens near mapped plate boundaries while earthquake symbols show sampled event locations.',
      body: '`GPUDistanceField` computes, for each of 1,048,576 cells, the exact distance to the nearest boundary seed and which seed it is. Seeds are points sampled along the boundary steps every 0.8 cells. The map is bright on the boundary and darkens away from it; the exact algorithm is the separable Felzenszwalb-Huttenlocher transform.\n\nA kernel then reads the grid at each event, so every earthquake has a distance. **Measure distance to** picks which boundaries count (try *Subduction zones only*), **Distance cap** limits how far the field reaches, and **Distance algorithm** swaps to the jump-flood preview. The readout is the share of events within 100 km of a boundary.',
      camera: {...QUAKE_REGIONS.japan.view, transitionMs: 1000},
      options: {mapShows: 'distance', mapOpacity: 0.55},
      controls: ['boundaryClasses', 'maximumDistance', 'algorithm', 'mapOpacity'],
      readouts: ['seeds', 'eventsAnalysed', 'nearShare']
    },
    {
      id: 'slab',
      title: 'Depth grows with distance: the slab',
      headline: 'Andean earthquake depth increases inland',
      textAlternative:
        'Andean earthquakes deepen with distance from the trench, with lines connecting deep events to nearest subduction boundaries.',
      body: 'At a subduction zone the oceanic plate dips beneath the other, so earthquakes along the sinking slab get deeper the farther they are from the trench. Peru and Chile is the clearest case: the Nazca plate descends beneath South America at about 7 cm per year, and the catalog has events down to 600 km, far inland. The chart shows median depth (and the band between the two percentiles) in each distance bin, computed by `GPUGroupStatistics`.\n\nTurn on **Link deep events to their boundary** to draw a line from each deep event to the cell of its nearest trench seed, and slide **Link events deeper than** to see how far they reach. The apparent dip is the slope of median depth against distance: it mixes segments with a steep slab and the flat-slab segments of Peru and central Chile, so treat it as a rough number.',
      camera: {...QUAKE_REGIONS.andes.view, transitionMs: 2000},
      options: {
        region: 'andes',
        boundaryClasses: 'subduction',
        mapShows: 'distance',
        showLinks: true,
        linkMinimumDepth: 100,
        binWidth: '10'
      },
      controls: ['boundaryClasses', 'showLinks', 'linkMinimumDepth', 'binWidth'],
      readouts: ['apparentDip', 'depthSpread', 'depthByDistance'],
      highlight: {readout: 'apparentDip'}
    },
    {
      id: 'classes',
      title: 'Where each depth class lives',
      headline: 'Deep classes occur farther from trenches',
      textAlternative:
        'Depth-class event colors around Japan accompany distributions showing intermediate and deep events farther from trenches.',
      body: 'The second group table counts events by depth class and distance bin; the chart shows, for each class, the share of its events in each bin. **Shallow** events (above 70 km) pile up at the boundary: they include the megathrust itself and the crust either side. **Intermediate** (70 to 300 km) and **deep** (below 300 km) events sit where the slab is, tens to hundreds of kilometers away; in Japan the deep ones lie under the Sea of Japan and the Izu-Bonin arc, hundreds of kilometers from the trench.\n\nMove **Shallow / intermediate limit** and **Intermediate / deep limit** to redefine the classes: the keys are recomputed on the GPU. **Minimum magnitude** below thins the small events, and the medians per class are in the readout.',
      camera: {...QUAKE_REGIONS.japan.view, transitionMs: 1800},
      options: {
        region: 'japan',
        boundaryClasses: 'subduction',
        mapShows: 'events',
        showLinks: false,
        binWidth: '20'
      },
      controls: ['shallowLimit', 'deepLimit', 'minimumMagnitude', 'binWidth'],
      readouts: ['classMedians', 'depthClasses']
    },
    {
      id: 'gutenberg',
      title: 'Small quakes outnumber big ones: b-value',
      headline: 'Magnitude counts decline approximately exponentially',
      textAlternative:
        'Japan’s earthquake map accompanies a cumulative magnitude-frequency chart and fitted Gutenberg-Richter line above completeness magnitude.',
      body: 'Across the world, each step up in magnitude has about ten times fewer events: **log10 N = a - b M** with **b** close to 1 (Gutenberg and Richter, 1944). The third group table counts events per 0.1 magnitude bin; the chart is the cumulative count on a log axis, and the dashed line is the Aki maximum-likelihood fit above the completeness magnitude **Mc**. Below Mc the catalog misses events and the curve bends away from the line.\n\nThe catalog peaks at M4.4 to 4.5, so it is complete above about **Mc** M4.5. Around Japan the fit gives b of about 1.3, a little above the textbook 1, and the estimate slides toward 1 as you raise **Mc** (it is near 1.0 to 1.2 above M5.2): the M4.5 to 5 range is probably still a little incomplete, so judge by the standard error and the bend. Compare **Magnitude table uses** near and far from boundaries: differences within the error shown are noise.',
      camera: {...QUAKE_REGIONS.japan.view, transitionMs: 1000},
      options: {
        region: 'japan',
        boundaryClasses: 'all',
        mapShows: 'allocation',
        grSubset: 'all',
        completeness: 4.5
      },
      controls: ['completeness', 'grSubset', 'grDistance'],
      readouts: ['bValue', 'grEvents', 'gutenberg']
    },
    {
      id: 'limits',
      title: 'What the distance does not tell you',
      headline: 'Unsigned nearest-boundary distance can misassign events',
      textAlternative:
        'The Sunda allocation raster partitions locations by nearest boundary family, including abrupt changes unrelated to tectonic side.',
      body: 'The distance is **unsigned** and to the *nearest* boundary: an event behind an arc may be measured to a transform fault on the wrong side, and the allocation map shows where the nearest family changes. PB2002 is a model whose boundary positions are good to tens of kilometers at best, so distances under that are in the noise; the grid is 3 to 4 km per cell; intraplate earthquakes are far from every boundary by construction.\n\nThe frame is planar, so distances are exact on the center latitude and drift by up to about 12% toward the top and bottom of a window, and the windows avoid the antimeridian. Depths of 10 km are often a network default. Try **Distance algorithm** (jump flooding) against the exact one on the slab chart, or **Deviation kind**, which only changes the spread readout.',
      camera: {...QUAKE_REGIONS.sunda.view, transitionMs: 1800},
      options: {
        region: 'sunda',
        boundaryClasses: 'all',
        mapShows: 'allocation',
        mapOpacity: 0.45
      },
      controls: ['algorithm', 'variance', 'mapShows'],
      readouts: ['eventsAnalysed', 'depthSpread']
    }
  ],

  about: {
    what: '`GPUDistanceField` turns boundary seed points into a Euclidean distance transform, a nearest-seed allocation and a nearest-cell index on a raster grid. A kernel samples it at each earthquake; `GPUGroupStatistics` then groups the events by distance bin (depth mean, deviation and percentiles), by depth class and distance bin (counts) and by magnitude bin (counts).',
    why: 'Distance to the nearest boundary is the simplest explanatory variable in seismology: it separates plate-boundary from intraplate seismicity, and depth against distance reveals the geometry of subducting slabs. The magnitude table gives the b-value, the standard check of catalog completeness.',
    howToRead:
      'On the map, bright is close to a boundary. In the first chart the line is median depth per distance bin and the band holds the chosen percentiles: a rising line is a dipping slab. In the second chart each curve integrates to 100%: the further right it peaks, the further from the boundary that depth class lives. In the third, the slope of the dashed line is minus b.'
  },

  legends: (state): readonly LegendSpec[] => {
    const rangeKm = Math.min(state.maximumDistance, 1000);
    const entries: LegendSpec[] = [];
    if (state.mapShows === 'distance') {
      entries.push({
        kind: 'ramp',
        title: 'Distance to the nearest boundary',
        ramp: 'cividis',
        extent: [0, rangeKm],
        unit: 'km',
        labels: [`${rangeKm} km`, 'at a boundary']
      });
    } else if (state.mapShows === 'allocation') {
      entries.push({
        kind: 'categories',
        title: 'Nearest boundary family',
        entries: BOUNDARY_FAMILY_NAMES.map((label, index) => ({
          color: [...FAMILY_COLORS[index], 230] as const,
          label
        })),
        note: 'Zones of the Euclidean allocation; the same colors draw the selected boundaries.'
      });
    }
    if (state.mapShows !== 'allocation') {
      entries.push({
        kind: 'categories',
        title: 'Boundary families',
        entries: BOUNDARY_FAMILY_NAMES.map((label, index) => ({
          color: [...FAMILY_COLORS[index], 255] as const,
          label
        })),
        note: 'Selected boundaries; the others are drawn faintly.'
      });
    }
    if (state.eventColor === 'class') {
      entries.push({
        kind: 'categories',
        title: 'Event depth',
        entries: [
          {
            color: [...QUAKE_DEPTH_CLASS_COLORS.dark[0], 255] as const,
            label: `Shallow (above ${state.shallowLimit} km)`
          },
          {
            color: [...QUAKE_DEPTH_CLASS_COLORS.dark[1], 255] as const,
            label: `Intermediate (${state.shallowLimit} to ${state.deepLimit} km)`
          },
          {
            color: [...QUAKE_DEPTH_CLASS_COLORS.dark[2], 255] as const,
            label: `Deep (below ${state.deepLimit} km)`
          }
        ]
      });
    } else {
      entries.push({
        kind: 'ramp',
        title: 'Event depth',
        ramp: 'magma',
        extent: [0, 300],
        unit: 'km',
        labels: ['300 km or more', 'surface']
      });
    }
    entries.push({
      kind: 'size',
      title: 'Radius is magnitude',
      entries: [4, 5, 6, 7, 8].map(magnitude => ({
        radiusPixels: 2.1 * state.sizeScale * 1.8 ** (magnitude - 4),
        label: `M${magnitude}`
      })),
      color: [190, 190, 200, 255]
    });
    return entries;
  },

  readouts: [
    {id: 'events', label: 'Events in the window'},
    {
      id: 'seeds',
      label: 'Distance field seeds',
      help: 'Seed points along the selected boundary steps and the size of one grid cell.'
    },
    {
      id: 'eventsAnalysed',
      label: 'Events with a distance',
      help: 'Events at or above the minimum magnitude whose cell has a seed within the distance cap and a distance inside the 64 bins.'
    },
    {
      id: 'nearShare',
      label: 'Near a boundary',
      help: 'Share of those events within 100 km of a seed.'
    },
    {
      id: 'apparentDip',
      label: 'Apparent slab dip',
      help: 'Slope of the median depth against distance, a least-squares line over bins with at least 8 events out to 500 km. It mixes slab segments and unsigned distances.'
    },
    {
      id: 'depthSpread',
      label: 'Depth spread',
      help: 'Standard deviation of depth per distance bin, averaged with the event counts as weights.'
    },
    {
      id: 'depthByDistance',
      label: 'Depth against distance',
      kind: 'chart'
    },
    {
      id: 'depthClasses',
      label: 'Distance by depth class',
      kind: 'chart'
    },
    {
      id: 'classMedians',
      label: 'Median distance per depth class',
      help: 'Median of the distance histogram of each class, to the bin center.'
    },
    {
      id: 'bValue',
      label: 'Gutenberg-Richter b-value',
      help: 'Aki (1965) maximum-likelihood estimate with Utsu correction for 0.1-unit bins; the error is b / sqrt(N).'
    },
    {
      id: 'grEvents',
      label: 'Events in the fit'
    },
    {
      id: 'gutenberg',
      label: 'Magnitude-frequency',
      kind: 'chart'
    }
  ],

  snippet:
    state => `import {GPUDistanceField, getGPUDistanceFieldParameterValues} from '@luma.gl/experimental/gpu-raster';
import {GPUGroupStatistics} from '@luma.gl/experimental/gpu-dataframe';

// Seeds: points along the PB2002 steps every 0.8 cells, family id as the seed id.
distanceGraph.add(new GPUDistanceField({
  width: 1024, height: 1024, mode: '${state.algorithm === 'exact' ? 'exact' : 'jump-flood'}',
  settings, seedPositions, seedIds, seedCount,
  output: {distances, allocation, nearestCells}
}));

// Depth statistics per distance bin (dense keys 0..63, rows masked by magnitude).
statisticsGraph.add(new GPUGroupStatistics({
  keys: keyDistance, mask: maskMagnitude, keyCount: 64, variance: '${state.variance}',
  percentiles,                                  // [${state.lowerFraction}, 0.5, ${state.upperFraction}]
  columns: [{values: depth, statistics: ['mean', 'standardDeviation', 'percentiles'], output}],
  output: {keys, counts, count, overflow}
}));
// Counts by (depth class, bin) and by magnitude bin: the same contributor with no value column.

// Per change (parameter writes, no recompile):
settings.write(getGPUDistanceFieldParameterValues({
  origin, cellSize: [cell, cell], maxDistance: ${state.maximumDistance >= 2000 ? 'Infinity' : state.maximumDistance * 1000}
}));`,

  create: async ctx =>
    (await import('./quake-plate-boundaries.compute')).createQuakePlateBoundaries(ctx)
});
