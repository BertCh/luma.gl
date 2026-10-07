// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
import {HURRICANE_CATEGORY_COLORS, HURRICANE_CATEGORY_LABELS} from './hurricane-data';
import type {HurricaneLandfallOptions} from './hurricane-landfall.compute';

const BASIN_VIEW = {longitude: -66, latitude: 28, zoom: 3.35};
const COAST_VIEW = {longitude: -83, latitude: 30.5, zoom: 4.6};

export default defineScene<HurricaneLandfallOptions>({
  id: 'hurricane-landfall',
  title: 'Where do hurricanes pass, and where do they come ashore?',
  chapter: 'earth',
  order: 11,
  summary:
    'Line density of 46 seasons of Atlantic storm tracks by intensity, the distance of every cell to the nearest coast from a GPU distance field, and landfalls per US state found as zone entries with interpolated crossing wind.',
  contributors: ['GPULineDensity', 'GPUDistanceField', 'GPUZoneEvents'],
  datasets: [
    {id: 'ibtracs-north-atlantic', role: 'storm tracks (IBTrACS, 1980-2025)'},
    {id: 'us-states', role: 'state polygons for landfall zones'},
    {id: 'naturalearth-atlantic-coast', role: 'coastline seeds for the distance field'}
  ],
  initialView: BASIN_VIEW,

  options: [
    {
      kind: 'select',
      id: 'view',
      label: 'Map layer',
      group: 'Layers',
      apply: 'param',
      default: 'density',
      help: 'Which analysis result is drawn as a raster: where tracks pass, the distance to the coast, or landfalls per state.',
      options: [
        {value: 'density', label: 'Where tracks pass (line density)'},
        {value: 'coast', label: 'Distance to the nearest coast'},
        {value: 'landfalls', label: 'Landfalls per state'},
        {value: 'none', label: 'No raster'}
      ]
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp',
      group: 'Layers',
      apply: 'param',
      default: 'magma',
      help: 'Ramp of the raster layer.',
      options: [
        {value: 'magma', label: 'Magma'},
        {value: 'viridis', label: 'Viridis'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'cividis', label: 'Cividis'}
      ]
    },
    {
      kind: 'slider',
      id: 'rasterOpacity',
      label: 'Raster opacity',
      group: 'Layers',
      apply: 'param',
      min: 0.2,
      max: 1,
      step: 0.05,
      default: 0.8,
      help: 'Opacity of the raster over the basemap.'
    },
    {
      kind: 'toggle',
      id: 'showTracks',
      label: 'Show storm tracks',
      group: 'Layers',
      apply: 'param',
      default: false,
      help: 'Draws every track, colored by the Saffir-Simpson class of the wind at each fix.'
    },
    {
      kind: 'slider',
      id: 'trackOpacity',
      label: 'Track opacity',
      group: 'Layers',
      apply: 'param',
      min: 0.05,
      max: 1,
      step: 0.05,
      default: 0.4,
      disabledWhen: state => !state.showTracks,
      help: 'Opacity of the tracks; low values show where they pile up.'
    },
    {
      kind: 'toggle',
      id: 'showLandfalls',
      label: 'Show landfall points',
      group: 'Layers',
      apply: 'param',
      default: true,
      help: 'One dot per landfall at the interpolated crossing position, colored by the wind there.'
    },
    {
      kind: 'toggle',
      id: 'showStateOutlines',
      label: 'Show state outlines',
      group: 'Layers',
      apply: 'param',
      default: true,
      help: 'The state polygons used as landfall zones.'
    },
    {
      kind: 'toggle',
      id: 'showCoastline',
      label: 'Show coastline seeds',
      group: 'Layers',
      apply: 'param',
      default: false,
      help: 'The Natural Earth 1:50m coastline that seeds the distance field.'
    },
    {
      kind: 'select',
      id: 'intensity',
      label: 'Intensity',
      group: 'Line density',
      apply: 'param',
      default: 'all',
      help: 'GPULineDensity sums length, not wind, so each strength gets its own layer: the parts of tracks at or above a wind threshold, all computed up front. The weighted layer adds the four.',
      options: [
        {value: 'all', label: 'All fixes'},
        {value: 'storm', label: 'Tropical storm or stronger (34 kt)'},
        {value: 'hurricane', label: 'Hurricane or stronger (64 kt)'},
        {value: 'major', label: 'Major hurricane (96 kt, Category 3+)'},
        {
          value: 'weighted',
          label: 'Wind-weighted (sum of the four)',
          help: 'A stretch of track counts once for each threshold it reaches, from 1 for a depression to 4 for a major hurricane.'
        }
      ]
    },
    {
      kind: 'toggle',
      id: 'sqrtScale',
      label: 'Square-root scale',
      group: 'Line density',
      apply: 'param',
      default: true,
      help: 'Lifts the low densities so the sparse edges of the cloud stay visible next to the busy Caribbean and Gulf.'
    },
    {
      kind: 'slider',
      id: 'coastSearchKm',
      label: 'Search limit',
      group: 'Distance to coast',
      apply: 'param',
      min: 200,
      max: 3000,
      step: 100,
      default: 1500,
      unit: 'km',
      help: 'Cells farther than this from every coast get no distance and are left transparent. It is the maxDistance setting of the distance field (a parameter write that re-runs the field) and the end of the color ramp.'
    },
    {
      kind: 'select',
      id: 'distanceMode',
      label: 'Distance algorithm',
      group: 'Distance to coast',
      apply: 'compile',
      default: 'exact',
      help: 'Both graphs are compiled up front; this chooses which one runs.',
      options: [
        {
          value: 'exact',
          label: 'Exact (separable envelopes)',
          help: 'The exact nearest coast seed of every cell.'
        },
        {
          value: 'jump-flood',
          label: 'Jump flooding (approximate)',
          help: 'A cheaper preview in log2(size) passes; cells may pick a slightly farther seed.'
        }
      ]
    },
    {
      kind: 'select',
      id: 'stateMetric',
      label: 'State value',
      group: 'Landfalls (zone events)',
      apply: 'param',
      default: 'count',
      help: 'What the state fill shows: how many landfalls, or the strongest crossing wind.',
      options: [
        {value: 'count', label: 'Landfalls'},
        {value: 'strongest', label: 'Strongest wind at landfall'}
      ]
    },
    {
      kind: 'slider',
      id: 'minimumLandfallWind',
      label: 'Minimum wind at landfall',
      group: 'Landfalls (zone events)',
      apply: 'param',
      min: 0,
      max: 130,
      step: 5,
      default: 0,
      unit: 'kt',
      help: 'Keeps landfalls where the interpolated wind at the crossing was at least this. 34 is tropical storm, 64 hurricane, 96 major hurricane.'
    },
    {
      kind: 'select',
      id: 'landfallCounting',
      label: 'Count',
      group: 'Landfalls (zone events)',
      apply: 'param',
      default: 'every-crossing',
      help: 'A storm can come ashore, go back out and come ashore again (Florida, then Louisiana). Count every crossing from open water, or only the first of each storm.',
      options: [
        {value: 'every-crossing', label: 'Every crossing from open water'},
        {value: 'first-per-storm', label: 'First landfall of each storm'}
      ]
    },
    {
      kind: 'select',
      id: 'eventsPerStorm',
      label: 'Events kept per storm',
      group: 'Landfalls (zone events)',
      apply: 'compile',
      default: '16',
      help: 'Capacity of the event list per storm, fixed when GPUZoneEvents is compiled. A track that crosses state lines more often is truncated and the readout says so; the landfalls of a storm that wanders over several states need the larger sizes.',
      options: [
        {value: '4', label: '4 events'},
        {value: '8', label: '8 events'},
        {value: '16', label: '16 events'},
        {value: '32', label: '32 events'}
      ]
    }
  ],

  legends: (state, data) => {
    const entries: {kind: 'ramp' | 'categories'}[] = [];
    void entries;
    const specs = [];
    if (state.view === 'density') {
      specs.push({
        kind: 'ramp' as const,
        id: 'density',
        title: 'Track length per area, average season',
        ramp: state.ramp,
        extent: 'gpu' as const,
        sqrtScale: state.sqrtScale,
        unit: 'km per 10,000 km2'
      });
    } else if (state.view === 'coast') {
      specs.push({
        kind: 'ramp' as const,
        title: 'Distance to the nearest coast',
        ramp: state.ramp,
        extent: [0, state.coastSearchKm] as const,
        unit: 'km',
        format: (value: number) => value.toFixed(0)
      });
    } else if (state.view === 'landfalls') {
      specs.push({
        kind: 'ramp' as const,
        title: state.stateMetric === 'count' ? 'Landfalls per state' : 'Strongest wind at landfall',
        ramp: state.ramp,
        extent: [
          state.stateMetric === 'count' ? 0 : 34,
          Math.max(1, (data.landfallMaximum as number | undefined) ?? 1)
        ] as const,
        unit: state.stateMetric === 'count' ? 'landfalls' : 'kt',
        format: (value: number) => value.toFixed(0)
      });
    }
    if (state.showTracks || state.showLandfalls) {
      specs.push({
        kind: 'categories' as const,
        title:
          state.showTracks && state.showLandfalls
            ? 'Wind at each fix (tracks) and at landfall (dots)'
            : state.showTracks
              ? 'Wind at each fix'
              : 'Wind at landfall',
        entries: HURRICANE_CATEGORY_LABELS.map((label, index) => ({
          color: HURRICANE_CATEGORY_COLORS[index],
          label
        })),
        note: 'Saffir-Simpson classes come from the sustained wind.'
      });
    }
    return specs;
  },

  readouts: [
    {
      id: 'stateChart',
      label: 'Landfalls per state',
      kind: 'chart',
      help: 'The twelve states with the most landfalls from open water at the chosen minimum wind. The selected state is highlighted.'
    },
    {
      id: 'aliveChart',
      label: 'Storms alive through the season',
      kind: 'chart',
      help: 'Average number of storms with a fix at the chosen strength on each day of the year. It rises through July, peaks around 8 September and fades by December.'
    },
    {
      id: 'coastChart',
      label: 'How close do storms get to a coast?',
      kind: 'chart',
      help: 'Distance to the nearest coast at every fix of the chosen intensity, sampled from the GPU distance field.'
    },
    {
      id: 'grid',
      label: 'Analysis grid',
      help: 'Grid shared by the density and distance layers, in deck.gl meters about 28 N, 56 W.'
    },
    {
      id: 'density',
      label: 'Line density',
      help: 'Segment-cell pieces produced by the four density layers and whether the piece capacity overflowed.'
    },
    {
      id: 'nearCoast',
      label: 'Near the coast',
      help: 'Share of the fixes of the chosen intensity within 100 km of a coastline (sampled at the center of a 25 km cell).'
    },
    {
      id: 'events',
      label: 'State entries',
      help: 'Enter events found by GPUZoneEvents, and how many came from open water.'
    },
    {
      id: 'candidates',
      label: 'Candidate capacity',
      help: 'Segment-edge bounding-box candidates tested, against the compiled capacity. Over capacity, events can be missing.'
    },
    {
      id: 'landfalls',
      label: 'Landfalls',
      help: 'Landfalls kept at the current minimum wind and counting rule.'
    },
    {
      id: 'strongest',
      label: 'Strongest landfall',
      help: 'Highest interpolated wind at the crossing.'
    },
    {
      id: 'selectedState',
      label: 'Selected state',
      help: 'Click a state to read its landfalls and strongest storm.'
    }
  ],

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPULineDensity, GPUZoneEvents, getGPULineDensityParameterValues, GPU_ZONE_EVENT_TYPE
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUDistanceField, getGPUDistanceFieldParameterValues} from '@luma.gl/experimental/gpu-raster';

// Where tracks pass: one density node per wind threshold (paths split where wind < threshold)
for (const [index, threshold] of [0, 34, 64, 96].entries()) {
  densityGraph.add(new GPULineDensity({
    id: 'density-' + index, positions: pathsAbove[index], pathOffsets: offsetsAbove[index],
    columns: ${360}, rows: ${232}, parameters: gridParameters,   // [minX, minY, cellWidth, cellHeight]
    output: {lengths: lengths[index], overflow}
  }));
}

// Distance to the coast: seeds are coastline vertices (planar meters)
coastGraph.add(new GPUDistanceField({
  width: ${360}, height: ${232}, mode: '${state.distanceMode}',   // compile-time
  settings: coastSettings, seedPositions: coastSeeds, seedCount,
  output: {distances}
}));
coastSettings.write(getGPUDistanceFieldParameterValues({
  bounds, gridSize: [${360}, ${232}], maxDistance: ${state.coastSearchKm * 1000}   // per frame
}));

// Landfalls: every state boundary edge tagged with its state, one thread per storm
eventGraph.add(new GPUZoneEvents({
  positions, timestamps, trackOffsets, edgeStarts, edgeEnds, edgeZones, zoneCount: 49,
  candidateCapacity: ${1 << 19}, maxEventsPerTrack: ${state.eventsPerStorm},
  events: {output: {ids, count, overflow}, eventZones, eventTypes, eventTimes, eventRows, eventPositions}
}));
// CPU: keep GPU_ZONE_EVENT_TYPE.enter events whose approach (a few km back) is not inside a state,
// then interpolate the wind between the two fixes of eventRows[i] with the crossing time.`,

  about: {
    what: '`GPULineDensity` clips every track segment to a grid and sums the track length per cell, so the map shows where storms spend their path rather than where they began. `GPUDistanceField` computes, for every cell, the distance to the nearest of tens of thousands of coastline seed points. `GPUZoneEvents` finds every moment a storm track crosses a state boundary, with the interpolated time and position of the crossing.',
    why: 'Emergency managers, insurers and coastal planners care about exposure, not just frequency: how often a stretch of ocean carries intense storms, how close the strongest ones get to land, and which states take the landfalls. These three measures, each a GPU pass over 46 seasons, answer those three questions.',
    howToRead:
      'Density is in kilometers of track per 10,000 square kilometers in an average season: a bright cell was crossed by many tracks. Distance rings show how far open water is from any coast. State fills count landfalls from open water; dots mark each crossing, colored by the wind there. These are best-track positions every six hours joined by straight lines, so landfall points are good to about a cell, not a street.'
  },

  create: async ctx => (await import('./hurricane-landfall.compute')).createHurricaneLandfall(ctx),

  story: [
    {
      id: 'the-question',
      title: 'Where do hurricanes actually pass?',
      body: 'A map of tracks is a tangle. **`GPULineDensity`** turns it into a number per place: it clips every segment of every storm to a grid and adds up the **kilometers of track in each cell**, divided by the cell area and the number of seasons. Bright means "an average season sends a lot of track over here".\n\nThe data are the **739 Atlantic storms of 1980-2025** (NOAA IBTrACS, six-hourly). Start with **All fixes**: the densest cells sit near 12 N between 40 W and 52 W, where waves leave Africa and spin up, and along the Southeast coast around 28 to 32 N; the track then curves out to sea. Turn off **Square-root scale** to see only the busiest cells. The grid is 25 km, so features finer than that are not resolved.',
      camera: {...BASIN_VIEW, transitionMs: 1200},
      options: {
        view: 'density',
        intensity: 'all',
        showTracks: false,
        showLandfalls: false,
        showStateOutlines: true
      },
      controls: ['sqrtScale', 'ramp', 'rasterOpacity'],
      readouts: ['grid', 'density']
    },
    {
      id: 'intensity',
      title: 'Weighted by wind: where do the strong ones go?',
      body: 'Line density sums **length**, not wind. To ask where the *dangerous* storms pass, the scene runs it four times on the pieces of track at or above a wind threshold (every fix, 34 kt tropical storm, 64 kt hurricane, 96 kt major hurricane), all compiled together and computed once. **Intensity** just picks which result is drawn.\n\nSlide through the thresholds: the dense patch near 12 N drops out because most of those storms are still weak, and the **major hurricanes** concentrate farther west, with the densest cells near the Lesser Antilles (about 16 N, 56 W) and across the Bahamas, Greater Antilles and Caribbean, where warm water lets storms peak. **Wind-weighted** adds the four layers, so a stretch of track counts up to four times.',
      camera: {...BASIN_VIEW, transitionMs: 1200},
      options: {view: 'density', intensity: 'major', showStateOutlines: true},
      controls: ['intensity', 'sqrtScale'],
      readouts: ['density']
    },
    {
      id: 'coast',
      title: 'How far is the nearest coast?',
      body: 'Wind is not the only thing that matters: a hurricane over open ocean harms ships, one near land harms people. **`GPUDistanceField`** gives every cell the distance to the **nearest coastline point** (the Natural Earth 1:50m coast, densified to a seed every 11 km), exactly, with one thread per column and then per row. The color is that distance, so the coast itself is the darkest band.\n\nThe histogram reads that field at every fix of the chosen **Intensity**: only about one hurricane-strength fix in six was within 100 km of a coast. Move **Search limit** down to 400 km and cells farther than that go transparent (a parameter write that re-runs the field). Switch **Distance algorithm** to *Jump flooding* to see the approximate version, which is cheaper but can pick a slightly wrong seed.',
      camera: {...BASIN_VIEW, transitionMs: 1200},
      options: {view: 'coast', intensity: 'hurricane', showCoastline: true, showTracks: false},
      controls: ['coastSearchKm', 'distanceMode', 'intensity'],
      readouts: ['nearCoast', 'coastChart']
    },
    {
      id: 'landfalls',
      title: 'Who gets the landfalls?',
      body: '**`GPUZoneEvents`** treats every state boundary edge as part of a zone and finds each time a storm segment crosses one, with the interpolated time and position of the crossing. This scene keeps only the entries that come **from open water** (it steps a few kilometers back along the track and drops the crossings that start inside another state), and reads the wind at the crossing from the two fixes around it.\n\nRoughly 350 landfalls by about 190 storms of the 739: **Florida** leads by a wide margin, then **Texas** and **Louisiana**, with North Carolina and South Carolina behind. Slide **Minimum wind at landfall** up to 64 kt: the Gulf and Florida still dominate, but the Carolinas and the Northeast thin out. Change **State value** to the strongest wind to see where the most intense landfalls were. Click a state for its count and strongest storm.',
      camera: {...COAST_VIEW, transitionMs: 1500},
      options: {
        view: 'landfalls',
        showLandfalls: true,
        showStateOutlines: true,
        showCoastline: false,
        showTracks: false,
        ramp: 'viridis'
      },
      controls: ['minimumLandfallWind', 'stateMetric', 'landfallCounting'],
      readouts: ['landfalls', 'strongest', 'stateChart', 'selectedState']
    },
    {
      id: 'season',
      title: 'When is the season?',
      body: 'The same tracks, read through time: for every day of the year, how many storms were alive on average? The curve is built from the fix times of all 46 seasons. It starts rising in June, **peaks around 8 September** with almost two storms alive at once on average, and is nearly gone by December. At **Hurricane or stronger** the curve is lower, under one storm alive on average, with its peak around 11 September: strong storms need the warmest water of late summer.',
      options: {view: 'landfalls', intensity: 'hurricane', showLandfalls: true},
      controls: ['intensity'],
      readouts: ['aliveChart']
    },
    {
      id: 'limits',
      title: 'What to remember, and what to try',
      body: 'Tracks are six-hourly best-track fixes joined by straight lines, so a storm that skims the coast between two fixes can be missed, and a crossing is placed to within a few tens of kilometers. State outlines are generalised (1:20 million), landfalls in Mexico, Cuba, the Bahamas and Canada are not counted, and an entry from inland Mexico into Texas counts as a landfall. Meters in the grid are deck.gl meters about 28 N: a small kernel rescales each row by cos(latitude) so lengths and distances are true, and the density grid cannot resolve features under 25 km. 46 seasons is a small sample for rare events.\n\n**Try it:** compare *Major hurricane* density with *All fixes*; set **Count** to *First landfall of each storm* and **Minimum wind at landfall** to 96 kt; lower **Events kept per storm** to 4 and watch the truncation readout; click Maine and then Texas and compare their strongest storms.',
      camera: {...BASIN_VIEW, transitionMs: 1200},
      options: {view: 'density', intensity: 'weighted', showLandfalls: true, showTracks: false},
      controls: ['eventsPerStorm', 'landfallCounting', 'minimumLandfallWind'],
      readouts: ['events', 'candidates', 'landfalls']
    }
  ]
});
