// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {ResidualRepairOptions} from './neural-growth.compute';

/** Temporal cloud repair over paired Sentinel-2 observations. */
export default defineScene<ResidualRepairOptions>({
  id: 'neural-growth',
  title: 'GPU Residual Diffusion for Cloud Repair',
  chapter: 'raster',
  order: 20,
  summary:
    'Repair Sentinel-2 cloud masks over Oʻahu or Venice with deterministic GPU diffusion. Clear target pixels pin the observed target-minus-prior residual; a fixed four-neighbor Laplacian propagates that residual into masked pixels. No parameters are learned.',
  contributors: ['FrontierResidualDiffusion'],
  datasets: [
    {
      id: 'sentinel-cloud-repair',
      role: 'two natural-colour cloudy/clear Sentinel-2 pairs with official scene-classification masks'
    }
  ],
  initialView: {longitude: -157.71496, latitude: 21.33493, zoom: 12.55},
  basemap: ground('paperCity'),
  furniture: {
    title: {
      title: 'GPU Residual Diffusion',
      subtitle: 'Deterministic cellular repair from paired observations'
    },
    scaleBar: {units: 'metric'},
    credit: 'Contains modified Copernicus Sentinel data (2026)',
    caveat: 'The result estimates the cloudy date; changes hidden by cloud cannot be verified.'
  },
  options: [
    {
      kind: 'select',
      id: 'place',
      label: 'Place',
      group: 'Observation pair',
      apply: 'param',
      default: 'oahu',
      options: [
        {
          value: 'oahu',
          label: 'Waimānalo, Oʻahu',
          help: 'Eight-month prior; vegetation and illumination changed.'
        },
        {
          value: 'venice',
          label: 'Venice Lagoon',
          help: 'One-day prior; less temporal separation than the Oʻahu pair.'
        }
      ],
      help: 'Switches all resident image, prior and mask buffers, then moves the camera.'
    },
    {
      kind: 'select',
      id: 'display',
      label: 'Map shows',
      group: 'Comparison',
      apply: 'param',
      default: 'cloudy',
      options: [
        {value: 'cloudy', label: 'Cloudy observation'},
        {value: 'diffusion', label: 'Residual diffusion repair'},
        {value: 'naive', label: 'Naive older-image paste'},
        {value: 'prior', label: 'Older clear reference'}
      ],
      help: 'Compare the observed target, the live GPU repair, a hard paste inside the same mask, and the complete prior.'
    },
    {
      kind: 'toggle',
      id: 'play',
      label: 'Run residual diffusion',
      group: 'Residual diffusion',
      apply: 'param',
      default: true,
      help: 'Runs four deterministic four-neighbor diffusion updates per frame while the residual repair is visible.'
    },
    {
      kind: 'slider',
      id: 'stepScale',
      label: 'Repair rate',
      group: 'Residual diffusion',
      apply: 'param',
      min: 0.1,
      max: 1,
      step: 0.05,
      default: 0.8,
      help: 'How quickly the target-minus-prior colour residual propagates across masked pixels.'
    },
    {
      kind: 'button',
      id: 'restart',
      label: 'Restart diffusion',
      group: 'Residual diffusion',
      help: 'Restores the mean-residual initialization so the diffusion and naive results can be compared again.'
    },
    {
      kind: 'slider',
      id: 'opacity',
      label: 'Image opacity',
      group: 'Display',
      apply: 'param',
      min: 0.3,
      max: 1,
      step: 0.05,
      default: 1
    }
  ],
  story: [
    {
      id: 'clouds',
      title: 'Two real cloudy observations',
      headline: 'Clouds hide Oʻahu ridges and Venice docks',
      textAlternative:
        'A natural-colour Sentinel-2 target crop shows cloud-obscured terrain at 10 metres per pixel.',
      body: 'Choose **Waimānalo, Oʻahu** or **Venice Lagoon**. Each view is a natural-colour, 512 × 512 Sentinel‑2 Level‑2A crop covering **5.12 km** at **10 metres per pixel**. These are observed clouds—not a painted circular hole. The target Scene Classification Layer marks cloud, cirrus, shadow, uncertainty and defective pixels, then a one-cell halo catches mixed cloud edges.',
      options: {display: 'cloudy'},
      controls: ['place', 'display'],
      readouts: ['image', 'cloud']
    },
    {
      id: 'naive',
      title: 'Establish the direct-substitution baseline',
      headline: 'The baseline copies prior pixels inside the mask',
      textAlternative:
        'Masked target pixels show the older clear observation; unmasked pixels retain target-date colour.',
      body: '**Naive older-image paste** copies the prior observation where the SCL mask is invalid and leaves every clear target pixel unchanged. Venice uses the previous day. Oʻahu uses an observation acquired eight months earlier, increasing exposure to changes in water colour, vegetation and illumination.',
      options: {display: 'naive'},
      controls: ['display', 'place'],
      readouts: ['cloud']
    },
    {
      id: 'diffusion',
      title: 'Diffuse observed residuals into the mask',
      headline: 'A fixed Laplacian propagates date-to-date colour residuals',
      textAlternative:
        'Masked pixels show the prior image plus an iteratively diffused RGB residual; observed target pixels remain fixed.',
      body: 'Switch to **Residual diffusion repair**. Each clear target pixel stores its observed target-minus-prior RGB residual and confidence 1; these values remain fixed. Masked pixels start at the scene-wide mean residual and confidence 0. Each GPU update adds 0.24 times the four-neighbor Laplacian, scaled by **Repair rate**, to all four channels. This is a deterministic finite-difference rule with fixed coefficients, not a trained model. The display adds the resulting RGB residual to the older clear image.',
      options: {display: 'diffusion'},
      controls: ['display', 'play', 'stepScale', 'restart'],
      readouts: ['iterations', 'coefficients']
    },
    {
      id: 'compare',
      title: 'Compare diffusion with direct substitution',
      headline: 'Boundary continuity does not validate hidden content',
      textAlternative:
        'Residual diffusion and direct prior substitution reuse the same older detail; no target-date reference is available under cloud.',
      body: 'Toggle between **Residual diffusion repair** and **Naive older-image paste** after the iteration count rises. Diffusion transfers the observed surrounding residual into the mask; the naive result uses the unadjusted historical measurement. No target-date reference exists under the cloud, so this comparison cannot measure reconstruction accuracy or recover changes that occurred while hidden. Use **Older clear reference** to inspect the detail both methods reuse.',
      controls: ['display', 'place'],
      readouts: ['iterations', 'image']
    }
  ],
  legends: () => [],
  readouts: [
    {id: 'coefficients', label: 'Fixed coefficient matrix', format: 'text'},
    {id: 'iterations', label: 'Diffusion iterations', format: 'integer'},
    {id: 'cloud', label: 'Masked pixels', format: 'text'},
    {id: 'cells', label: 'Pixels per pass', format: 'integer'},
    {id: 'image', label: 'Observation pair', format: 'text'}
  ],
  snippet:
    state => `// Fixed coefficients implement 0.24 × the four-neighbor Laplacian; no training occurs.
const diffusion = new FrontierResidualDiffusion({
  width, height, state, nextState, coefficients, parameters, habitat, prior, display
});
parameters.write(getFrontierResidualDiffusionParameterValues({
  stepScale: ${state.stepScale}, mutation: 0, damping: 0
}));`,
  about: {
    what: 'A deterministic cellular residual-diffusion rule repairs pixels flagged by Sentinel‑2 scene classification using a registered older clear observation.',
    why: 'The paired observations expose the contribution of residual diffusion relative to direct prior-image substitution. The operation is reproducible and contains no learned parameters.',
    howToRead:
      'The cloudy image and dated prior are observed. Cloud-masked target pixels do not contribute colour. The repaired output is older detail plus a spatially diffused residual; target-date content hidden by cloud is not observed or validated.'
  },
  create: async ctx => (await import('./neural-growth.compute')).createResidualRepair(ctx)
});
