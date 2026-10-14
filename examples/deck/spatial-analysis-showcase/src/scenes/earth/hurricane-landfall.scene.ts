// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CREDITS, joinCredits} from '../../cartography/credits';
import {US, WORLD, labelsFor} from '../../cartography/gazetteer';
import {ground} from '../../cartography/grounds';
import {HURRICANE_CLASS} from '../../cartography/hue-registry';
import {defineScene} from '../scene';
import {HURRICANE_CATEGORY_LABELS} from './hurricane-data';
import type {HurricaneLandfallOptions} from './hurricane-landfall.compute';

const BASIN_VIEW = {longitude: -66, latitude: 28, zoom: 3.35};
const COAST_VIEW = {longitude: -83, latitude: 30.5, zoom: 4.6};
const cartouche = (title: string, subtitle: string) => ({
  title,
  subtitle,
  chips: ['best-track crossings'] as const
});
const LANDFALL_LABELS = labelsFor(WORLD, ['atlantic-ocean', 'gulf-of-mexico', 'caribbean-sea']);
const COAST_LABELS = labelsFor(US, ['state-fl', 'state-tx', 'state-nc']);

export default defineScene<HurricaneLandfallOptions>({
  id: 'hurricane-landfall',
  title: 'Where do Atlantic storms cross the coast?',
  chapter: 'earth',
  order: 4,
  summary:
    'Line density of 46 seasons of Atlantic storm tracks by intensity, great-circle distances from cell centres to coast cells picked by an AEQD GPU distance field, and landfalls per US state found as zone entries with interpolated crossing wind.',
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
      help: 'Cells whose great-circle distance to the picked coast cell exceeds this limit are transparent. Parameter writes widen the AEQD candidate search by the maximum grid scale factor, then apply this limit in great-circle kilometers without recompiling. The limit also sets the end of the color ramp.'
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
          help: 'The nearest coast cell in the AEQD grid; the GPU then measures its great-circle distance. Scale variation can make another coast cell closer on the sphere.'
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
          color: HURRICANE_CLASS.light[index],
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
      id: 'coastChart',
      label: 'How close do storms get to a coast?',
      kind: 'chart',
      help: 'Great-circle distance from each fix’s grid-cell centre to the AEQD-picked coast-cell centre, sampled at fixes of the chosen intensity.'
    },
    {
      id: 'grid',
      label: 'Analysis grid',
      help: 'Density uses WGS84 Lambert equal-area meters. The coast grid uses AEQD meters about 28 N, 56 W to pick coast cells; displayed distances are great-circle kilometers between cell centres on the mean-radius sphere.'
    },
    {
      id: 'density',
      label: 'Line density',
      help: 'Segment-cell pieces produced by the four density layers and whether the piece capacity overflowed.'
    },
    {
      id: 'nearCoast',
      label: 'Near the coast',
      help: 'Share of fixes whose grid-cell centre lies within 100 great-circle km of the picked coast-cell centre. Cell snapping and AEQD coast-cell selection limit accuracy.'
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

  pipeline: [
    {
      id: 'density',
      label: 'Line density',
      detail: 'Clip six-hour track segments into the analysis grid'
    },
    {
      id: 'distance',
      label: 'Distance field',
      detail: 'Pick a coast cell, then measure great-circle distance'
    },
    {id: 'events', label: 'Zone events', detail: 'Interpolate crossings of a state edge'},
    {id: 'draw', label: 'Draw', detail: 'One measure at a time: length, distance or crossings'}
  ],

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPULineDensity, GPUZoneEvents, getGPULineDensityParameterValues, GPU_ZONE_EVENT_TYPE,
  GPU_EARTH_MEAN_RADIUS_METERS
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUDistanceField, GPU_DISTANCE_FIELD_NONE, getGPUDistanceFieldParameterValues, makeGPURasterMetadata} from '@luma.gl/experimental/gpu-raster';
import {addKernelPass} from '../../engine/mode-kernels';

// Where tracks pass: one density node per wind threshold (paths split where wind < threshold)
for (const [index, threshold] of [0, 34, 64, 96].entries()) {
  densityGraph.add(new GPULineDensity({spatialContext: densityProjection.spatialContext,
    id: 'density-' + index, positions: pathsAbove[index], lineOffsets: offsetsAbove[index],
    columns: ${360}, rows: ${232}, parameters: gridParameters,   // [minX, minY, cellWidth, cellHeight] in WGS84 Lambert meters
    output: {lengths: lengths[index], densities: densities[index], status: {overflow}}   // densities: projected length / equal-area cell area
  }));
}

// Distance to the coast: coastline seeds in AEQD meters about 28 N, 56 W
const coastGrid = makeGPURasterMetadata({     // AEQD frame meters, row 0 = south, rows advance north
  width: ${360}, height: ${232}, cellSize: [cellWidth, cellHeight], origin: [bounds[0], bounds[1]], rowDirection: 'north'
});
coastGraph.add(new GPUDistanceField({
  grid: coastGrid, mode: '${state.distanceMode}',   // compile-time
  parameters: coastSettings, seedPositions: coastSeeds, seedCount,
  output: {distances, nearestCells}   // distances is required; only nearestCells drives the display
}));
const maximumScaleFactor = maximumAngularDistance / Math.sin(maximumAngularDistance);
coastSettings.write(getGPUDistanceFieldParameterValues({
  maximumDistance: ${state.coastSearchKm * 1000} * maximumScaleFactor   // per frame, conservative AEQD cutoff
}, coastGrid));
coastCutoff.write(Float32Array.of(${state.coastSearchKm}));   // true km, parameter write without recompilation

// cellCenters: static float32x2 lon/lat, storms.unproject(x, y) at every grid-cell centre.
// Both distance modes feed this pass; bindings below are graph views (float32x2 read as scalar pairs).
addKernelPass(coastGraph, {
  id: 'coast-km', invocationCount: ${360 * 232},
  bindings: [
    {name: 'nearestCells', view: nearestCells, type: 'u32', access: 'read'},
    {name: 'cellCenters', view: cellCenters, type: 'f32', access: 'read'},
    {name: 'cutoff', view: coastCutoff.importToGraph(coastGraph), type: 'f32', access: 'read'},
    {name: 'kilometers', view: coastKm, type: 'f32', access: 'read_write'}
  ],
  declarations: 'const NONE: u32 = ' + GPU_DISTANCE_FIELD_NONE + 'u;\\n' +
    'const EARTH_RADIUS_METERS: f32 = ' + GPU_EARTH_MEAN_RADIUS_METERS.toFixed(1) + ';',
  body: \`
    kilometers[kilometersOffset + index] = bitcast<f32>(0x7f800000u); // +Infinity: transparent / beyond limit
    let nearestCell = nearestCells[nearestCellsOffset + index];
    if (nearestCell == NONE) { return; }
    let centerOffset = cellCentersOffset + index * 2u;
    let coastOffset = cellCentersOffset + nearestCell * 2u;
    let latitude = radians(cellCenters[centerOffset + 1u]);
    let coastLatitude = radians(cellCenters[coastOffset + 1u]);
    let longitudeDifference = cellCenters[coastOffset] - cellCenters[centerOffset];
    let wrappedLongitude = longitudeDifference - 360.0 * floor((longitudeDifference + 180.0) / 360.0);
    let latitudeSine = sin((coastLatitude - latitude) * 0.5);
    let longitudeSine = sin(radians(wrappedLongitude) * 0.5);
    let haversine = latitudeSine * latitudeSine + cos(latitude) * cos(coastLatitude) * longitudeSine * longitudeSine;
    let distanceKm = EARTH_RADIUS_METERS * 0.001 * 2.0 * asin(min(1.0, sqrt(haversine)));
    if (distanceKm > cutoff[cutoffOffset]) { return; }
    kilometers[kilometersOffset + index] = distanceKm;
  \`
});

// Landfalls: every state boundary edge tagged with its state, one thread per storm
eventGraph.add(new GPUZoneEvents({
  spatialContext: storms.spatialContext,
  tracks: {kind: 'trajectories', positions, timestamps, trackOffsets},
  zones: {kind: 'polygons', positions: statePositions, featureOffsets, polygonOffsets, ringOffsets},   // 49 states, one feature each
  candidateCapacity: ${1 << 19}, trackEventCapacity: ${state.eventsPerStorm},
  output: {events: {ids, status: {count, overflow}, eventZones, eventTypes, eventTimes, eventRows, eventPositions}}
}));
// CPU: keep GPU_ZONE_EVENT_TYPE.enter events whose approach (a few km back) is not inside a state,
// then interpolate the wind between the two fixes of eventRows[i] with the crossing time.`,

  about: {
    what: '`GPULineDensity` clips every track segment to a grid and sums the track length per cell, so the map shows where storms spend their path rather than where they began. `GPUDistanceField` picks a coast cell in the AEQD grid; a GPU haversine pass measures the great-circle distance between cell centres on the mean-radius sphere. `GPUZoneEvents` finds every moment a storm track crosses a state boundary, with the interpolated time and position of the crossing.',
    why: 'Emergency managers, insurers and coastal planners care about exposure, not just frequency: how often a stretch of ocean carries intense storms, how close the strongest ones get to land, and which states take the landfalls. These three measures, each a GPU pass over 46 seasons, answer those three questions.',
    howToRead:
      'Density is in kilometers of track per 10,000 square kilometers in an average season: a bright cell was crossed by many tracks. Distance rings show how far open water is from any coast. State fills count landfalls from open water; dots mark each crossing, colored by the wind there. These are best-track positions every six hours joined by straight lines, so landfall points are good to about a cell, not a street.'
  },

  basemap: ground('paperSheet'),
  furniture: {
    title: cartouche(
      'Where do Atlantic storms cross the coast?',
      'Track length · km / area · crossings · 1980–2025'
    ),
    scaleBar: {units: 'metric'},
    northArrow: 'always',
    credit: joinCredits(CREDITS.noaaNhc, 'NOAA IBTrACS', CREDITS.usCensus, CREDITS.naturalEarth),
    caveat:
      'Six-hour best-track chords and generalised coastlines locate crossings approximately. Track density uses projected lengths and exact WGS84 Lambert cell areas. Coast distances are great-circle distances on the mean-radius sphere from each cell centre to the centre of the coast cell picked by the AEQD field. Beyond cell discretization and floating-point rounding, residual error comes from picking a different cell than the true nearest coast cell where frame scale varies across candidates: it is zero for most cells, under 1.6% for 99% of them, and up to about 14% in a few cells far from the frame centre (European side of the basin). Jump flooding adds approximate selection error.'
  },
  annotations: [...LANDFALL_LABELS, ...COAST_LABELS],

  create: async ctx => (await import('./hurricane-landfall.compute')).createHurricaneLandfall(ctx),

  story: [
    {
      id: 'paths',
      title: 'Where do hurricanes actually pass?',
      headline: 'Track length shows where storms travel',
      textAlternative: 'A paper Atlantic atlas shows classed storm-track density.',
      optionsMode: 'fresh',
      body: 'A map of tracks is a tangle. **`GPULineDensity`** clips every segment to a grid and sums **kilometers of track in each cell**, normalized by cell area and chosen seasons. Bright means an average season sends more track through that place.\n\nStart with **All fixes** and use the live grid and density readouts to inspect the corridor. Turn off **Square-root scale** to concentrate contrast on the busiest cells. The selected grid size sets the smallest pattern this view can resolve.',
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
      id: 'strength',
      title: 'Weighted by wind: where do the strong ones go?',
      headline: 'Intensity changes the corridor',
      textAlternative: 'Major-hurricane track length appears over a quiet Atlantic paper map.',
      optionsMode: 'fresh',
      body: 'Line density sums **length**, not wind. To inspect stronger segments, the scene precomputes thresholds from all fixes through major-hurricane strength; **Intensity** selects the result rather than recomputing the track archive.\n\nSlide through thresholds and compare corridors. **Wind-weighted** adds the threshold layers, so a strong stretch contributes more than a weak one; it is an emphasis measure, not a count of storms.',
      camera: {...BASIN_VIEW, transitionMs: 1200},
      options: {view: 'density', intensity: 'major', showStateOutlines: true},
      controls: ['intensity', 'sqrtScale'],
      readouts: ['density']
    },
    {
      id: 'coast',
      title: 'How far is the nearest coast?',
      headline: 'Distance turns coastline into a field',
      textAlternative: 'Blue coast-distance bands surround Florida and the Gulf coast.',
      optionsMode: 'fresh',
      body: 'Wind is not the only exposure: a storm over open ocean and one near land have different consequences. **`GPUDistanceField`** picks the nearest coast cell in AEQD grid meters, then a GPU haversine pass measures the great-circle distance between cell centres; the coast is therefore the darkest band. Scale variation can make a different coast cell closer on the sphere.\n\nThe histogram samples that field at fixes for the chosen **Intensity**. Use **Search limit** to hide distant cells, then compare the exact method with *Jump flooding*, which is cheaper but can choose a nearby rather than nearest seed.',
      camera: {...BASIN_VIEW, transitionMs: 1200},
      options: {view: 'coast', intensity: 'hurricane', showCoastline: true, showTracks: false},
      controls: ['coastSearchKm', 'distanceMode', 'intensity'],
      readouts: ['nearCoast', 'coastChart']
    },
    {
      id: 'crossing',
      title: 'How does a crossing become a landfall?',
      headline: 'A landfall is an interpolated crossing',
      textAlternative: 'Classed landfall rings sit over Atlantic-facing US states.',
      optionsMode: 'fresh',
      body: '**`GPUZoneEvents`** treats state boundary edges as zones and finds every storm-segment crossing, interpolating its time, position, and wind. The view keeps approaches **from open water** and rejects crossings that begin inside another state.\n\nThe live landfall card, bars, and state click report the current filter. Raise **Minimum wind at landfall** to see which corridors remain; switch **State value** to strongest wind to prioritize intensity over crossing count.',
      camera: {...COAST_VIEW, transitionMs: 1500},
      options: {
        view: 'landfalls',
        showLandfalls: true,
        showStateOutlines: true,
        showCoastline: false,
        showTracks: false,
        ramp: 'cividis'
      },
      controls: ['minimumLandfallWind', 'stateMetric', 'landfallCounting'],
      readouts: ['landfalls', 'strongest', 'stateChart', 'selectedState']
    },
    {
      id: 'states',
      title: 'Which state edges receive crossings?',
      headline: 'Counts need a coastline denominator',
      textAlternative:
        'State landfall counts are shown with category-coloured crossing rings and a ranked state chart.',
      optionsMode: 'fresh',
      body: 'This state choropleth counts data-derived entries from open water and keeps the landfall rings above it. Choose **State value** to compare event count with strongest interpolated crossing wind.\n\nA coastline-length rate is deliberately omitted: this implementation has not derived a reliable state coastline denominator at the loaded generalisation. Dividing by land area would answer a different question.',
      camera: {...COAST_VIEW, transitionMs: 1200},
      options: {view: 'landfalls', showLandfalls: true, showStateOutlines: true, showTracks: false},
      controls: ['stateMetric', 'minimumLandfallWind', 'landfallCounting'],
      readouts: ['landfalls', 'strongest', 'stateChart', 'selectedState']
    },
    {
      id: 'limits',
      title: 'What to remember, and what to try',
      headline: 'Best-track crossings are approximate',
      textAlternative: 'A paper Atlantic map keeps event rings and state boundaries in view.',
      optionsMode: 'fresh',
      body: 'Best-track fixes are joined by straight lines, so a coast skim between fixes can be missed and every crossing is approximate. State boundaries are generalized and this regional state view does not count every coastline in the basin. Grid resolution also limits what density can show.\n\n**Try it:** compare *Major hurricane* density with *All fixes*; count only each storm’s first landfall; tighten the wind threshold; lower **Events kept per storm** and inspect truncation; then click contrasting states and compare their current summaries.',
      camera: {...BASIN_VIEW, transitionMs: 1200},
      options: {view: 'density', intensity: 'weighted', showLandfalls: true, showTracks: false},
      controls: ['eventsPerStorm', 'landfallCounting', 'minimumLandfallWind'],
      readouts: ['events', 'candidates', 'landfalls']
    }
  ]
});
