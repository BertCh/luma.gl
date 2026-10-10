// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {joinCredits} from '../../cartography/credits';
import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {FlightSpeedContrastOptions} from './flight-corridors-jet-stream.compute';
import {MOVEMENT_CREDITS} from './movement-style';

const US_VIEW = {longitude: -96, latitude: 38.5, zoom: 3.9};

export default defineScene<FlightSpeedContrastOptions>({
  id: 'flight-corridors-jet-stream',
  title: 'East–West Cruise-Speed Contrast',
  chapter: 'movement',
  order: 11,
  summary:
    'Measure speed and heading for 38,135 US flights on the GPU. Half the difference between unmatched eastbound and westbound median ground speeds is an effective along-track wind proxy, confounded by aircraft mix, route and altitude selection, and unequal sampling in space and time.',
  contributors: ['GPUTrajectoryMetrics'],
  datasets: [{id: 'poopdeck-adsb-paths', role: 'flight trajectories (OpenSky ADS-B, 6 Jan 2020)'}],
  initialView: US_VIEW,
  basemap: ground('night', {labels: 'none'}),
  furniture: {
    title: {
      title: 'East–West Cruise-Speed Contrast',
      subtitle: 'Effective along-track proxy from unmatched ADS-B samples · 6 January 2020',
      chips: ['OBSERVED ground speed', 'DERIVED proxy']
    },
    scaleBar: {units: 'metric'},
    credit: joinCredits(MOVEMENT_CREDITS.openSky),
    caveat:
      'The half-difference is not a wind observation: aircraft mix, route choice, altitude, and space–time sampling differ by direction.'
  },

  options: [
    {
      kind: 'select',
      id: 'show',
      label: 'Show steps heading',
      group: 'Map',
      apply: 'param',
      default: 'all',
      help: 'Keeps steps whose heading lies inside the direction cone around east or west. The test runs in the vertex shader on the heading column GPUTrajectoryMetrics wrote. The charts always compare both directions.',
      options: [
        {value: 'all', label: 'Every direction'},
        {value: 'east', label: 'Eastbound only'},
        {value: 'west', label: 'Westbound only'}
      ]
    },
    {
      kind: 'range',
      id: 'speedRange',
      label: 'Color range',
      group: 'Map',
      apply: 'param',
      min: 150,
      max: 700,
      step: 10,
      default: [300, 650],
      unit: 'kn',
      help: 'Ground speeds mapped to the two ends of the color ramp.'
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Speed color ramp',
      group: 'Map',
      apply: 'param',
      default: 'magma',
      help: 'Each ramp is perceptually uniform.',
      options: [
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'cividis', label: 'Cividis (color-blind optimised)'}
      ]
    },
    {
      kind: 'slider',
      id: 'widthPixels',
      label: 'Line width',
      group: 'Map',
      apply: 'param',
      min: 0.5,
      max: 3,
      step: 0.25,
      default: 1.25,
      unit: 'px',
      help: 'Width of the step lines in screen pixels.'
    },
    {
      kind: 'slider',
      id: 'minAltitude',
      label: 'Cruise altitude from',
      group: 'Filters',
      apply: 'param',
      min: 6000,
      max: 11000,
      step: 500,
      default: 9000,
      unit: 'm',
      format: value => `${value.toLocaleString('en-US')} m`,
      help: 'Only steps whose lower end is at or above this altitude count (about 30,000 feet at 9,000 m). It hides climbs and descents on the map and drops them from the statistics, since an aircraft in a climb is not at its cruise speed.'
    },
    {
      kind: 'slider',
      id: 'coneDegrees',
      label: 'Direction cone',
      group: 'Filters',
      apply: 'param',
      min: 15,
      max: 75,
      step: 5,
      default: 45,
      unit: '°',
      help: 'Half angle around east and around west: a step counts as eastbound when its heading is within this angle of east. Angles are measured on the map projection grid, which differs from true north by up to about 16 degrees at the coasts, so keep it above 25 degrees.'
    },
    {
      kind: 'select',
      id: 'speedSource',
      label: 'Ground speed from',
      group: 'Filters',
      apply: 'param',
      default: 'derived',
      help: 'Derived: distance over time between two stored positions, measured by GPUTrajectoryMetrics. Reported: the ground speed each aircraft broadcast (the ADS-B velocity message). They are independent, so agreement is a check on the whole pipeline.',
      options: [
        {value: 'derived', label: 'Measured from positions (GPU)'},
        {value: 'reported', label: 'Reported by the aircraft'}
      ]
    },
    {
      kind: 'select',
      id: 'chartBy',
      label: 'Profile chart by',
      group: 'Charts',
      apply: 'param',
      default: 'longitude',
      help: 'Bins the directional ground-speed contrast by longitude, latitude or altitude. The half-difference is an effective along-track wind proxy, not a matched wind estimate. Bins with fewer than 30 steps are blank.',
      options: [
        {value: 'longitude', label: 'Longitude (4 degree bins)'},
        {value: 'latitude', label: 'Latitude (2 degree bins)'},
        {value: 'altitude', label: 'Altitude (500 m bins)'}
      ]
    }
  ],

  readouts: [
    {
      id: 'speedChart',
      label: 'Ground speed, eastbound against westbound',
      kind: 'chart',
      help: 'Share of cruise steps in each 10 knot bin, separately for each direction. The marked lines are the medians.'
    },
    {
      id: 'profileChart',
      label: 'Directional medians and proxy by bin',
      kind: 'chart',
      help: 'Median ground speed by direction. The dashed half-difference is an effective along-track wind proxy; aircraft, routes, altitudes, locations and times are not matched.'
    },
    {id: 'eastMedian', label: 'Eastbound median'},
    {id: 'westMedian', label: 'Westbound median'},
    {
      id: 'difference',
      label: 'Directional median difference',
      help: 'Eastbound minus westbound median ground speed from unmatched samples.'
    },
    {
      id: 'midpoint',
      label: 'Median midpoint',
      help: 'Average of the two directional medians. It is not an airspeed estimate because aircraft and sampling differ by direction.'
    },
    {
      id: 'windProxy',
      label: 'Effective along-track wind proxy',
      help: 'Half the unmatched directional median difference. Aircraft performance, route choice, altitude, location and time sampling remain confounders.'
    },
    {id: 'samples', label: 'Steps used'},
    {
      id: 'agreement',
      label: 'Derived against reported',
      help: 'How far the speed measured from positions is from the speed the aircraft reported, for the same steps.'
    },
    {id: 'flights', label: 'Flights'},
    {id: 'steps', label: 'Steps measured'}
  ],

  legends: state => [
    {
      kind: 'ramp' as const,
      title:
        state.speedSource === 'derived'
          ? 'Measured ground speed; unmatched proxy input'
          : 'Reported ground speed; unmatched proxy input',
      ramp: state.ramp,
      extent: state.speedRange,
      unit: 'kn',
      format: (value: number) => value.toFixed(0)
    }
  ],

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUTrajectoryMetrics} from '@luma.gl/experimental/gpu-spatial-analysis';

// positions: float32x2 planar metres (azimuthal equidistant around 38.5 N 96 W),
// timestamps: float32 seconds, trackOffsets: uint32 (flights + 1)
const graph = new GPUCommandGraph(device, {id: 'flight-speed-contrast'});
graph.add(new GPUTrajectoryMetrics({spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'native'},
  positions, timestamps, trackOffsets,
  stepSpeeds,      // per row: speed of the step that ends there, m/s
  stepHeadings     // per row: atan2(dy, dx) of that step, radians counter-clockwise from +x
}));
const compiled = graph.compile();
compiled.encode(commandEncoder, {parameters: undefined});   // once: the tracks never change

// shader-side filter, ${state.show}: cos(heading) ${state.show === 'east' ? '> ' : state.show === 'west' ? '< -' : 'ignored '}cos(${state.coneDegrees} deg), lower end >= ${state.minAltitude} m
// speed in knots = stepSpeeds[row] * 3600 / 1852
// Half the unmatched east-west median difference is an effective along-track wind proxy.
// Aircraft type, route, altitude, location, and time are not matched between directions.`,

  about: {
    what: '`GPUTrajectoryMetrics` computes step distance over time and projected heading for every flight. The columns remain on the GPU for map filtering and are read back once for directional distributions and binned summaries.',
    why: 'Ground speed combines airspeed with the along-track wind component. The east–west half-difference is therefore a useful effective proxy, but these are unmatched observational samples rather than reciprocal measurements of the same aircraft, route, altitude, place and time.',
    howToRead:
      'Brightness encodes ground speed. The first chart compares unmatched directional distributions. The second reports directional medians and their half-difference by bin. Treat the half-difference as an effective along-track wind proxy, not a jet-stream detection or direct wind estimate. Projection distance error reaches about 4% at the coasts.'
  },

  create: async ctx =>
    (await import('./flight-corridors-jet-stream.compute')).createFlightSpeedContrast(ctx),

  story: [
    {
      id: 'the-question',
      headline: 'Directional ground speeds define an indirect proxy',
      textAlternative:
        'Flight steps over the contiguous United States are coloured by ground speed; direction is not encoded until filtered.',
      optionsMode: 'fresh',
      title: 'What does the east–west cruise-speed difference contain?',
      body: 'Each line is one step between two recorded positions on 6 January 2020, colored by **ground speed** and filtered to cruise altitude. The data contains **38,135 US flights** from OpenSky ADS-B; it contains no wind observations.\n\nEastbound and westbound steps are not matched by aircraft type, route, altitude, location or time. Their speed difference therefore combines along-track wind with aircraft-performance and sampling effects. Set **Show steps heading** to *Eastbound only* and then *Westbound only* to inspect the two input populations.',
      camera: {...US_VIEW, transitionMs: 1200},
      options: {show: 'all', speedSource: 'derived', minAltitude: 9000},
      controls: ['show', 'speedRange', 'ramp'],
      readouts: ['flights', 'steps']
    },
    {
      id: 'metrics',
      headline: 'One pass measures every flight step',
      textAlternative:
        'Each flight step is coloured by position-derived or reported ground speed after the selected altitude filter.',
      optionsMode: 'fresh',
      title: 'One pass measures every step',
      body: '**`GPUTrajectoryMetrics`** computes the projected distance, elapsed time and heading of the step ending at every row. The result is two float columns that the map reads directly. The graph runs once because the tracks are static; controls change shader filtering and statistical selection.\n\nSet **Cruise altitude from** to 6,000 m to include more climb and descent segments. The 9,000 m default restricts the sample to approximately 30,000 feet and above; it does not equalize altitude or flight phase between directions.',
      options: {minAltitude: 9000, show: 'all'},
      controls: ['minAltitude', 'speedSource'],
      readouts: ['samples', 'agreement']
    },
    {
      id: 'east-west',
      headline: 'The directional distributions differ',
      textAlternative:
        'A line chart compares unmatched eastbound and westbound ground-speed distributions and marks each median.',
      optionsMode: 'fresh',
      title: 'Compare the directional ground-speed distributions',
      body: 'The chart includes steps within **Direction cone** of east (blue) or west (orange). The **directional median difference** is an observed contrast between two unmatched populations.\n\nHalf of that contrast is reported as an **effective along-track wind proxy**. It is not a direct wind measurement and does not identify a jet stream: aircraft mix, route choice, altitude, location and observation time differ between directions. Change **Direction cone** to test sensitivity to the heading definition.',
      camera: {longitude: -96, latitude: 39, zoom: 4.2, transitionMs: 1400},
      options: {show: 'east', coneDegrees: 45, minAltitude: 9000, speedSource: 'derived'},
      highlight: {readout: 'difference'},
      controls: ['show', 'coneDegrees'],
      readouts: ['eastMedian', 'westMedian', 'difference', 'speedChart']
    },
    {
      id: 'airspeed-and-wind',
      headline: 'The half-difference is a conditional proxy',
      textAlternative:
        'Readouts show the midpoint and half-difference of unmatched directional medians; neither is a direct airspeed or wind observation.',
      optionsMode: 'fresh',
      title: 'State the assumptions behind the proxy',
      body: 'For reciprocal observations with equal airspeed and opposite, equal-magnitude along-track wind, half the east–west ground-speed difference equals the wind component. This dataset does not provide those matched observations. The reported half-difference is therefore an effective proxy, and the median midpoint is only an algebraic summary, not inferred airspeed.\n\nSwitch **Ground speed from** to *Reported by the aircraft*. The **agreement** readout compares reported speed with position-derived speed for the same steps; agreement checks the speed calculation, not the assumptions required to isolate wind.',
      options: {show: 'all', speedSource: 'derived', minAltitude: 9000, coneDegrees: 45},
      highlight: {readout: 'windProxy'},
      controls: ['speedSource', 'minAltitude'],
      readouts: ['midpoint', 'windProxy', 'agreement']
    },
    {
      id: 'where',
      headline: 'The proxy varies across the sampled dimensions',
      textAlternative:
        'A profile chart shows directional median ground speeds and their dashed half-difference proxy by the selected bin dimension.',
      optionsMode: 'fresh',
      title: 'Stratify the contrast by longitude, latitude and altitude',
      body: 'The second chart bins steps by **Profile chart by**. Solid lines are directional median ground speeds; the dashed line is their half-difference proxy. A bin is omitted when either direction has fewer than 30 steps.\n\nChanging the bin variable changes both the physical conditions and the sample composition. Longitude and latitude bins contain different routes and aircraft; altitude bins contain different flight phases and route choices. The profiles describe this sample and do not locate or measure a jet-stream core.',
      options: {show: 'all', chartBy: 'longitude'},
      controls: ['chartBy'],
      readouts: ['profileChart', 'windProxy']
    },
    {
      id: 'limits',
      headline: 'Interpret the proxy within its sampling limits',
      textAlternative:
        'The altitude profile shows unmatched directional medians and their half-difference; missing bins have fewer than 30 steps in either direction.',
      optionsMode: 'fresh',
      title: 'Sampling and measurement constraints',
      body: 'The sample covers one day. Steps can span up to 10 minutes after trajectory simplification. Speeds and headings use an azimuthal-equidistant planar projection; distance error reaches about 4% at the coasts and projected headings differ from true bearings. Coverage ends at the contiguous-US boundary.\n\nThe principal confounders are aircraft performance, fleet composition, route selection, altitude selection, and unequal location and time sampling by direction. Compare altitude and latitude profiles, change the minimum altitude, and widen the direction cone to evaluate sensitivity. These checks do not convert the proxy into a wind observation.',
      camera: {...US_VIEW, transitionMs: 1400},
      options: {show: 'all', chartBy: 'altitude'},
      controls: ['chartBy', 'coneDegrees', 'minAltitude'],
      readouts: ['difference', 'samples']
    }
  ]
});
