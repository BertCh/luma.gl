// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {SourceFinderOptions} from './source-finder.compute';

/** An inverse problem: infer a staged release from sparse sensors at real Chicago facilities. */
export default defineScene<SourceFinderOptions>({
  id: 'source-finder',
  title: 'Source Finder: where did the plume begin?',
  chapter: 'interpolation',
  order: 20,
  summary:
    'A Gaussian-plume inverse model scores 36,000 candidate origins against synthetic readings at real Chicago hospitals and fire stations, then publishes its best estimate entirely on the GPU.',
  contributors: ['FrontierSourceInference'],
  datasets: [
    {
      id: 'chicago-facilities',
      role: 'real hospital and fire-station sites used as a staged sensor network'
    }
  ],
  initialView: {longitude: -87.67, latitude: 41.88, zoom: 10.2},
  basemap: ground('night'),
  furniture: {
    title: {title: 'Source Finder', subtitle: 'Candidate likelihood from 32 staged sensors'},
    scaleBar: {units: 'metric'},
    credit: 'City of Chicago; Overture Maps Foundation',
    caveat: 'Facility sites are real; concentrations and source are synthetic.'
  },
  options: [
    {
      kind: 'slider',
      id: 'windBearing',
      label: 'Wind blows toward',
      group: 'Model assumptions',
      apply: 'param',
      min: 0,
      max: 355,
      step: 5,
      default: 65,
      unit: '°',
      help: 'Direction of plume travel clockwise from north. A wrong wind direction moves and broadens the inferred origin.'
    },
    {
      kind: 'slider',
      id: 'emissionRate',
      label: 'Emission strength',
      group: 'Model assumptions',
      apply: 'param',
      min: 0.25,
      max: 2,
      step: 0.05,
      default: 1,
      help: 'Assumed source concentration scale. The staged readings were generated at 1.0.'
    },
    {
      kind: 'slider',
      id: 'dispersion',
      label: 'Initial plume width',
      group: 'Model assumptions',
      apply: 'param',
      min: 100,
      max: 1200,
      step: 25,
      default: 450,
      unit: 'm',
      help: 'Crosswind standard deviation at the source; it grows with square-root distance.'
    },
    {
      kind: 'slider',
      id: 'decayLength',
      label: 'Downwind half-scale',
      group: 'Model assumptions',
      apply: 'param',
      min: 2000,
      max: 20000,
      step: 500,
      default: 9000,
      unit: 'm',
      help: 'Distance scale of centerline concentration decay.'
    },
    {
      kind: 'slider',
      id: 'noiseSigma',
      label: 'Sensor noise',
      group: 'Uncertainty',
      apply: 'param',
      min: 0.02,
      max: 0.3,
      step: 0.01,
      default: 0.08,
      help: 'Standard deviation used by the likelihood. Larger values admit a wider set of plausible origins.'
    },
    {
      kind: 'toggle',
      id: 'showSensors',
      label: 'Show sensor sites',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Shows the real facility locations colored by their staged concentration reading.'
    },
    {
      kind: 'toggle',
      id: 'revealSource',
      label: 'Reveal staged source',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Adds the known red source marker so the cyan GPU estimate can be checked.'
    },
    {
      kind: 'slider',
      id: 'opacity',
      label: 'Likelihood opacity',
      group: 'Display',
      apply: 'param',
      min: 0.25,
      max: 1,
      step: 0.05,
      default: 0.78,
      help: 'Opacity of the candidate likelihood field.'
    }
  ],
  story: [
    {
      id: 'incident',
      title: 'Thirty-two alarms, one unknown origin',
      headline: 'Sparse sensors leave a field of plausible origins',
      body: 'Hospitals and fire stations are real City of Chicago and Overture sites; their readings are a clearly staged incident so this story remains reproducible. `FrontierSourceInference` asks the inverse question: which source would have produced all of those readings together?\n\nGold points are high readings, dark points are low. The magenta surface is candidate likelihood and the cyan marker is its GPU-selected maximum.',
      controls: ['showSensors'],
      readouts: ['sensors', 'candidates']
    },
    {
      id: 'wind',
      title: 'Turn the wind and the answer moves',
      headline: 'One assumption can move the maximum across the city',
      body: 'Every candidate predicts zero plume upwind and a widening Gaussian profile downwind. Change **Wind blows toward** below: 36,000 candidates × 32 sensors are rescored on the next frame, and the cyan maximum moves without rebuilding or downloading the field.',
      controls: ['windBearing'],
      readouts: ['candidates']
    },
    {
      id: 'uncertainty',
      title: 'Assumptions are part of the map',
      headline: 'A sharp likelihood peak is not the same as certainty',
      body: '**Sensor noise** below controls how harshly residuals are penalized; **Initial plume width** controls how much crosswind disagreement the model tolerates. A crisp peak does not mean certainty if these assumptions are wrong. Try a very wide plume and then rotate the wind ten degrees.',
      controls: ['noiseSigma', 'dispersion', 'decayLength'],
      readouts: ['sensors']
    },
    {
      id: 'check',
      title: 'Reveal the answer—then break the model',
      headline: 'The right answer depends on the right transport model',
      body: 'Turn on **Reveal staged source** below. The red dot is the known synthetic origin; the cyan dot is the best candidate written by the reduction graph. At the default assumptions they nearly coincide. Now change **Emission strength** or wind: the failure is visible, which is exactly why inverse maps should expose their assumptions.',
      controls: ['revealSource', 'emissionRate', 'windBearing'],
      readouts: ['candidates']
    }
  ],
  legends: () => [
    {
      kind: 'ramp',
      title: 'Candidate likelihood',
      ramp: 'magma',
      extent: [0.05, 1],
      labels: ['unlikely', 'best fit']
    },
    {
      kind: 'categories',
      title: 'Markers',
      entries: [
        {color: [0, 190, 255, 255], label: 'GPU estimate'},
        {color: [255, 88, 70, 255], label: 'staged truth (when revealed)'}
      ]
    }
  ],
  readouts: [
    {id: 'sensors', label: 'Sensor sites', format: 'integer'},
    {id: 'candidates', label: 'Candidates per frame', format: 'integer'}
  ],
  snippet: state => `const inference = new FrontierSourceInference({
  width: 180, height: 200, sensorCount: 32,
  sensorPositions, sensorValues, parameters, likelihoods, summary, bestPosition
});
parameters.write(getFrontierSourceParameterValues({
  bounds, wind: windFromBearing(${state.windBearing}), noiseSigma: ${state.noiseSigma}
}));`,
  about: {
    what: 'A browser-sized inverse problem using a steady Gaussian plume and a deterministic staged incident at real facility sites.',
    why: 'Inverse mapping turns sparse observations into a spatial hypothesis and makes thousands of independent candidate evaluations ideal for GPU compute.',
    howToRead:
      'Brighter cells fit all sensor readings better under the chosen assumptions. Cyan is the GPU maximum; red, when enabled, is the staged truth.'
  },
  create: async ctx => (await import('./source-finder.compute')).createSourceFinder(ctx)
});
