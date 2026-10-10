// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {SourceFinderOptions} from './source-finder.compute';

/** An inverse problem: localize a synthetic release from sparse sensors at real facilities. */
export default defineScene<SourceFinderOptions>({
  id: 'source-finder',
  title: 'Source Finder: inverse plume localization',
  chapter: 'interpolation',
  order: 20,
  summary:
    'A GPU inverse model evaluates 36,000 candidate origins against 32 synthetic sensor readings. The observations use a distinct forward model with deterministic spatial and heteroscedastic error; the surface is a relative fit score, not a probability.',
  contributors: ['FrontierSourceInference'],
  datasets: [
    {
      id: 'chicago-facilities',
      role: 'real hospital and fire-station sites used in a synthetic sensor-placement experiment'
    }
  ],
  initialView: {longitude: -87.67, latitude: 41.88, zoom: 10.2},
  basemap: ground('night'),
  furniture: {
    title: {title: 'Source Finder', subtitle: 'Relative fit across 36,000 candidate origins'},
    scaleBar: {units: 'metric'},
    credit: 'City of Chicago; Overture Maps Foundation',
    caveat:
      'Facility sites are real. Sensor placement, concentrations and source are synthetic; the model is not calibrated for operational atmospheric inference.'
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
      help: 'Assumed source concentration scale. The fitted default is 1.0; the synthetic generator uses 1.04.'
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
      help: 'Residual scale used by the relative fit score. Larger values reduce the score penalty for disagreement.'
    },
    {
      kind: 'toggle',
      id: 'showSensors',
      label: 'Show sensor sites',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Shows the selected facility locations colored by synthetic concentration.'
    },
    {
      kind: 'toggle',
      id: 'revealSource',
      label: 'Reveal synthetic source',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Adds the known red source marker so the cyan GPU estimate can be checked.'
    },
    {
      kind: 'slider',
      id: 'opacity',
      label: 'Fit surface opacity',
      group: 'Display',
      apply: 'param',
      min: 0.25,
      max: 1,
      step: 0.05,
      default: 0.78,
      help: 'Opacity of the relative fit score surface.'
    }
  ],
  story: [
    {
      id: 'incident',
      title: 'Synthetic observations at 32 facility sites',
      headline: 'A maximin placement design samples the Chicago study area',
      textAlternative:
        'A relative-fit raster covers Chicago. Thirty-two spatially distributed facility sensors are colored by synthetic concentration, and a cyan marker identifies the maximum-score candidate origin.',
      body: 'Hospital and fire-station coordinates are real City of Chicago and Overture sites. A deterministic maximin design selects 32 spatially separated facilities; it is a sensor-placement experiment, not an operational network. Synthetic concentrations use a plume with 525 m initial dispersion, 7.5 m per square-root metre growth and 11,200 m decay length, plus deterministic spatially correlated, heteroscedastic perturbations. The fitted model uses a different dispersion and decay specification to avoid an exact inverse crime.\n\nGold points have higher readings. The magenta raster is a relative fit score and the cyan marker is its GPU-selected maximum. The score is not normalized over candidates and is not a posterior probability.',
      controls: ['showSensors'],
      readouts: ['sensors', 'candidates', 'sensorProfile', 'bestScore'],
      evidence:
        'The sensor profile plots every synthetic observation against its signed downwind distance from the known generating source; the background level and source crosswind plane are explicit references.',
      caveat:
        'The observations are synthetic and the facility locations are an experimental maximin subset, not a deployed monitoring network.'
    },
    {
      id: 'wind',
      title: 'Turn the wind and the answer moves',
      headline: 'One assumption can move the maximum across the city',
      textAlternative:
        'The relative-fit surface and cyan maximum shift when the assumed wind bearing changes; all 36,000 candidate origins are evaluated against the same 32 sensor values.',
      body: 'Each candidate predicts background concentration upwind and a widening Gaussian profile downwind. Changing **Wind blows toward** triggers one evaluation of 36,000 candidates × 32 sensors. Display frames reuse the GPU outputs until an analytic option changes; the field is not downloaded for rendering.',
      controls: ['windBearing'],
      readouts: ['candidates', 'bestScore'],
      evidence:
        'The reported best score and cyan marker come from the same GPU reduction over the complete 180 × 200 candidate grid.',
      caveat:
        'A high relative score only ranks candidates under the selected transport parameters; it is not a calibrated probability.'
    },
    {
      id: 'uncertainty',
      title: 'Assumptions are part of the map',
      headline: 'Residual scale controls contrast, not inferential certainty',
      textAlternative:
        'Changing residual scale, plume width, or decay length changes the contrast and location of the relative-fit surface; the display is not a probability or confidence region.',
      body: '**Sensor noise** sets the residual scale in the score; **Initial plume width** controls crosswind dispersion. The score omits parameter uncertainty, atmospheric variability, sensor bias and a probability normalization. A concentrated score surface therefore does not establish a confidence region.',
      controls: ['noiseSigma', 'dispersion', 'decayLength'],
      readouts: ['bestScore', 'localizationError'],
      evidence:
        'Every analytic control re-evaluates all candidates against the same 32 readings, making changes in the maximum attributable to the stated assumption.',
      caveat:
        'The surface conditions on one parameter setting at a time and does not marginalize over uncertain wind, dispersion, decay or sensor error.'
    },
    {
      id: 'check',
      title: 'Compare the estimate with synthetic truth',
      headline: 'Localization error measures displacement from the generating source',
      textAlternative:
        'A cyan maximum-score marker is compared with the revealed red synthetic source; the localization-error readout reports their planar separation in metres.',
      body: 'Enable **Reveal synthetic source**. The red marker is the synthetic source and the cyan marker is the maximum-score grid cell written by the GPU reduction. **Localization error** is their planar distance in metres. Changing **Emission strength** or wind quantifies sensitivity to misspecified transport assumptions.',
      controls: ['revealSource', 'emissionRate', 'windBearing'],
      readouts: ['bestScore', 'localizationError'],
      evidence:
        'Localization error is calculated directly from the winning grid-cell center and the known synthetic source in the scene projection.',
      caveat:
        'The grid imposes a finite spatial resolution, so even a correctly specified model cannot localize more precisely than its candidate cells.'
    }
  ],
  legends: () => [
    {
      kind: 'ramp',
      title: 'Relative fit score',
      ramp: 'magma',
      extent: [0.05, 1],
      labels: ['lower fit', 'higher fit']
    },
    {
      kind: 'categories',
      title: 'Markers',
      entries: [
        {color: [0, 190, 255, 255], label: 'GPU estimate'},
        {color: [255, 88, 70, 255], label: 'synthetic truth (when revealed)'}
      ]
    }
  ],
  readouts: [
    {id: 'sensors', label: 'Sensor sites', format: 'integer'},
    {id: 'candidates', label: 'Candidates per evaluation', format: 'integer'},
    {
      id: 'sensorProfile',
      label: 'Observation profile',
      kind: 'chart',
      help: 'Each synthetic reading plotted against signed distance along the known generating wind direction.'
    },
    {id: 'bestScore', label: 'Best fit score'},
    {id: 'localizationError', label: 'Localization error'}
  ],
  snippet: state => `const inference = new FrontierSourceInference({
  width: 180, height: 200, sensorCount: 32,
  sensorPositions, sensorValues, parameters, scores, summary, bestPosition
});
parameters.write(getFrontierSourceParameterValues({
  bounds, wind: windFromBearing(${state.windBearing}), noiseSigma: ${state.noiseSigma}
}));`,
  about: {
    what: 'A controlled inverse-source experiment using real facility coordinates, synthetic concentrations from a distinct forward model, deterministic spatial and heteroscedastic perturbations, and a 180 × 200 candidate grid.',
    why: 'The implementation measures GPU evaluation of independent candidate origins and exposes localization sensitivity to transport assumptions. It does not estimate a calibrated posterior distribution.',
    howToRead:
      'Brighter cells have a higher relative residual score under the selected assumptions. Cyan is the maximum-score cell; red is synthetic truth when enabled. Use the localization-error readout for the planar displacement in metres.'
  },
  create: async ctx => (await import('./source-finder.compute')).createSourceFinder(ctx)
});
