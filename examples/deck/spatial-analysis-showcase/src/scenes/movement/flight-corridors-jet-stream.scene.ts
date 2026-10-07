// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {joinCredits} from '../../cartography/credits';
import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {JetStreamOptions} from './flight-corridors-jet-stream.compute';
import {MOVEMENT_CREDITS} from './movement-style';

const US_VIEW = {longitude: -96, latitude: 38.5, zoom: 3.9};

export default defineScene<JetStreamOptions>({
  id: 'flight-corridors-jet-stream',
  title: 'Can aircraft measure the jet stream?',
  chapter: 'movement',
  order: 11,
  summary:
    'Measure the speed and heading of every step of 38,135 US flights on the GPU, then compare eastbound with westbound ground speed at cruise altitude: the jet stream, found from aircraft alone.',
  contributors: ['GPUTrajectoryMetrics'],
  datasets: [{id: 'poopdeck-adsb-paths', role: 'flight trajectories (OpenSky ADS-B, 6 Jan 2020)'}],
  initialView: US_VIEW,
  basemap: ground('night', {labels: 'none'}),
  furniture: {
    title: {
      title: 'Aircraft as a wind instrument',
      subtitle: 'Along-track wind from ADS-B ground speed · 6 January 2020',
      chips: ['OBSERVED ground speed', 'DERIVED wind']
    },
    scaleBar: {units: 'metric'},
    credit: joinCredits(MOVEMENT_CREDITS.openSky)
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
      help: 'All four are perceptually uniform.',
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
      help: 'What the second chart bins the steps by: where the jet stream is strong (longitude, latitude) or how it changes with flight level (altitude). Bins with fewer than 30 steps are left blank.',
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
      label: 'Median ground speed by position',
      kind: 'chart',
      help: 'Median ground speed of each direction in bins of longitude, latitude or altitude. The dashed line is half the difference, the average tailwind component.'
    },
    {id: 'eastMedian', label: 'Eastbound median'},
    {id: 'westMedian', label: 'Westbound median'},
    {
      id: 'difference',
      label: 'Median difference',
      help: 'Eastbound median minus westbound median ground speed.'
    },
    {
      id: 'airspeed',
      label: 'Implied airspeed',
      help: 'Average of the two medians. If the tailwind is the same size either way, it cancels: what is left is the speed through the air, about 450 knots for an airliner at cruise.'
    },
    {
      id: 'wind',
      label: 'Implied wind',
      help: 'Half the median difference: the average tailwind component along the flight axis, with no wind data used.'
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
        state.speedSource === 'derived' ? 'Ground speed (measured)' : 'Ground speed (reported)',
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
const graph = new GPUCommandGraph(device, {id: 'jet-stream'});
graph.add(new GPUTrajectoryMetrics({
  positions, timestamps, trackOffsets,
  stepSpeeds,      // per row: speed of the step that ends there, m/s
  stepHeadings     // per row: atan2(dy, dx) of that step, radians counter-clockwise from +x
}));
const compiled = graph.compile();
compiled.encode(commandEncoder, {parameters: undefined});   // once: the tracks never change

// shader-side filter, ${state.show}: cos(heading) ${state.show === 'east' ? '> ' : state.show === 'west' ? '< -' : 'ignored '}cos(${state.coneDegrees} deg), lower end >= ${state.minAltitude} m
// speed in knots = stepSpeeds[row] * 3600 / 1852
// eastbound median - westbound median, over steps at or above ${state.minAltitude} m, is twice the average tailwind`,

  about: {
    what: '`GPUTrajectoryMetrics` measures every step of every flight at once: the distance between two stored positions divided by the time between them, and the heading of the step. The two columns stay on the GPU, where the map colors steps from them with a cruise-altitude and direction filter; they are read back once so the statistics can be binned.',
    why: 'An aircraft flies at a roughly fixed speed through the air, so its speed over the ground is that plus the wind along its track. Flying one way and then the other cancels the aircraft out and leaves the wind. A day of flights is therefore a wind instrument: the jet stream measured with no weather data at all.',
    howToRead:
      'Bright steps are fast over the ground. With eastbound only, the cruise tracks are bright; with westbound only, dim. The first chart is the speed distribution for each direction and the second shows where the difference is largest. Half the difference is the average tailwind; half the sum is the airspeed. Angles are on the map projection grid and speeds are planar speeds, up to about 4% too long at the coasts.'
  },

  create: async ctx => (await import('./flight-corridors-jet-stream.compute')).createJetStream(ctx),

  story: [
    {
      id: 'the-question',
      headline: 'Ground speeds can measure a wind',
      textAlternative: 'Aircraft ground-speed lines form a dark US map.',
      optionsMode: 'fresh',
      title: 'How fast is an aircraft really moving, and does it depend on direction?',
      body: 'Each line is one step between two recorded positions of a flight on Monday 6 January 2020, colored by the **speed over the ground** it covered (legend), and only at cruise altitude. Look at the colors: some tracks are much brighter than others.\n\nThe data is **38,135 US flights** from the OpenSky ADS-B network. Nothing about wind is in it, only where each aircraft was and when. Set **Show steps heading** below to *Eastbound only* and then *Westbound only*, and compare.',
      camera: {...US_VIEW, transitionMs: 1200},
      options: {show: 'all', speedSource: 'derived', minAltitude: 9000},
      controls: ['show', 'speedRange', 'ramp'],
      readouts: ['flights', 'steps']
    },
    {
      id: 'metrics',
      headline: 'One pass measures every flight step',
      textAlternative: 'A selected flight shows measured ground speed along its route.',
      optionsMode: 'fresh',
      title: 'One pass measures every step',
      body: '**`GPUTrajectoryMetrics`** computes, for every row of every track, the speed and the heading of the step that ends there: a distance and a time difference per row, no loops over flights on the CPU. The result is two float columns that the map reads directly. The graph runs **once**, because the tracks never change; every control here just changes what the shader and the statistics keep.\n\nSlide **Cruise altitude from** down to 6,000 m and climbs and descents join the map; they are slower and head every which way. The default of 9,000 m keeps level flight at about 30,000 feet and above.',
      options: {minAltitude: 9000, show: 'all'},
      controls: ['minAltitude', 'speedSource'],
      readouts: ['samples', 'agreement']
    },
    {
      id: 'east-west',
      headline: 'Eastbound ground speeds are faster',
      textAlternative: 'Eastbound and westbound speed distributions are compared.',
      optionsMode: 'fresh',
      title: 'Eastbound jets are faster, westbound are slower',
      body: 'The chart shows ground speed for the steps that head within **Direction cone** of east (blue) and of west (orange). The two humps barely overlap: the median **eastbound** step covers the ground far faster than the median **westbound** one, over the same country, by the same airliner types.\n\nThat gap is the **median difference** readout. It is the jet stream, a river of fast westerly air at cruise altitude: it carries eastbound flights and holds back westbound ones. Narrow **Direction cone** to 25 degrees to check that it is not an artifact of the boundary.',
      camera: {longitude: -96, latitude: 39, zoom: 4.2, transitionMs: 1400},
      options: {show: 'east', coneDegrees: 45, minAltitude: 9000, speedSource: 'derived'},
      highlight: {readout: 'difference'},
      controls: ['show', 'coneDegrees'],
      readouts: ['eastMedian', 'westMedian', 'difference', 'speedChart']
    },
    {
      id: 'airspeed-and-wind',
      headline: 'Subtract airspeed and reveal wind',
      textAlternative: 'A signed wind map uses a zero-centred diverging legend.',
      optionsMode: 'fresh',
      title: 'Cancel the aircraft, keep the wind',
      body: 'If every aircraft flew at the same speed through the air, eastbound ground speed would be airspeed plus wind and westbound airspeed minus wind. Half the difference is then the average **wind**; half the sum, the **implied airspeed**. Read both in the panel: the airspeed lands near the 450 knots an airliner cruises at, which is a sign the reasoning holds.\n\nNow switch **Ground speed from** to *Reported by the aircraft*. Each aircraft broadcast its own ground speed, independent of our position arithmetic, and the medians barely move. The **agreement** readout gives the typical difference between the two for the same step.',
      options: {show: 'all', speedSource: 'derived', minAltitude: 9000, coneDegrees: 45},
      highlight: {readout: 'wind'},
      controls: ['speedSource', 'minAltitude'],
      readouts: ['airspeed', 'wind', 'agreement']
    },
    {
      id: 'where',
      headline: 'The strongest wind forms a band',
      textAlternative: 'A gridded wind map and profile identify the jet band.',
      optionsMode: 'fresh',
      title: 'Where, and how high, is it strongest?',
      body: 'The second chart bins the steps by position. Each point is the median ground speed in one bin of **Profile chart by**; the dashed line is half the difference, the wind. Start with *Longitude*: how does the wind change from the west coast to the east?\n\nThen try *Latitude*: the jet stream follows a meandering track, so the headwind and tailwind depend on which state you cross. Finally *Altitude*: the wind increases with height up to the jet core, which is why long-haul flights climb to the same flight levels. Bins with too few steps are left blank.',
      options: {show: 'all', chartBy: 'longitude'},
      controls: ['chartBy'],
      readouts: ['profileChart', 'wind']
    },
    {
      id: 'limits',
      headline: 'One Monday needs careful reading',
      textAlternative: 'The wind estimate is shown with sampling limits.',
      optionsMode: 'fresh',
      title: 'What to remember, and what to try',
      body: 'This is one winter Monday, so the jet stream is at its strongest; a summer day would show a much smaller gap. The steps are long (up to 10 minutes, simplified from ADS-B pings), speeds are planar speeds in an azimuthal projection that stretches distances up to about 4% at the coasts, and angles are on that grid. Aircraft do not all fly the same airspeed, and pilots choose routes and altitudes to use the wind, so this is the effective wind they experienced, not a weather model. The data stops at the edge of the contiguous US, so there is no Atlantic here.\n\n**Try it:** compare *Altitude* and *Latitude* profiles; set **Cruise altitude from** to 11,000 m and see what changes; widen **Direction cone** to 75 degrees and watch the two humps mix.',
      camera: {...US_VIEW, transitionMs: 1400},
      options: {show: 'all', chartBy: 'altitude'},
      controls: ['chartBy', 'coneDegrees', 'minAltitude'],
      readouts: ['difference', 'samples']
    }
  ]
});
