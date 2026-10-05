// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {GPUParticleAdvection} from './gpu-particle-advection';
export type {
  GPUParticleAdvectionProps,
  GPUParticleAdvectionTrails
} from './gpu-particle-advection';
export {
  getGPUParticleAdvectionParameterValues,
  getGPUParticleAdvectionWordParameterValues,
  GPU_PARTICLE_ADVECTION_PARAMETER_LENGTH,
  GPU_PARTICLE_ADVECTION_WORD_PARAMETER_LENGTH
} from './particle-advection-parameters';
export type {
  GPUParticleAdvectionSettings,
  GPUParticleAdvectionWordSettings
} from './particle-advection-parameters';
export {
  advanceParticlesOnCPU,
  createParticleAdvectionCPUState
} from './particle-advection-cpu';
export type {ParticleAdvectionCPUState} from './particle-advection-cpu';
