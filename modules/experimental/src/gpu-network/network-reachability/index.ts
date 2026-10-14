// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {
  GPUNetworkReachability,
  getGPUNetworkReachabilityInflation,
  GPU_NETWORK_REACHABILITY_NONE,
  GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS,
  getGPUNetworkReachabilityMaximumNodeCount,
  GPU_NETWORK_REACHABILITY_MAXIMUM_LOCAL_ITERATIONS,
  GPU_NETWORK_REACHABILITY_MAXIMUM_TIE_ITERATIONS
} from './gpu-network-reachability';
export type {
  GPUNetworkReachabilityOutput,
  GPUNetworkReachabilityProps
} from './gpu-network-reachability';
export {
  adaptReachabilityIterations,
  recommendReachabilityIterations
} from './network-reachability-tuning';
export type {
  ReachabilityIterationsAdaptationProps,
  ReachabilityIterationsRecommendation,
  ReachabilityIterationsRecommendationProps
} from './network-reachability-tuning';

// The shared network input of every gpu-network contributor; defined below the entries.
export type {
  GPUNetworkAdjacency,
  GPUNetworkTopology
} from '../../analysis-core/network-topology';
