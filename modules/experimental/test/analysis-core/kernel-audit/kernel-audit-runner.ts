// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphBufferHandle, GraphDataView, GraphTextureHandle} from '@luma.gl/gpgpu/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {
  createKernelAuditContext,
  getDeclaredNodes,
  recordShaderModules,
  type KernelAuditCase
} from './kernel-audit-context';
import {
  GENERIC_REFUSAL_PATTERNS,
  KERNEL_AUDIT_REJECTIONS,
  type KernelAuditRejection
} from './kernel-audit-rejections';
import {auditWGSLModule, type WGSLFinding} from './kernel-audit-wgsl';

/** The rejection-table entry of a case that must be refused at declaration. */
export function getExpectedRejection(auditCase: KernelAuditCase): KernelAuditRejection | undefined {
  return KERNEL_AUDIT_REJECTIONS.find(
    entry => entry.exportName === auditCase.exportName && entry.name === auditCase.name
  );
}

/**
 * Findings of a declaration error against its expected refusal: the message must match
 * `rejects`, name the documented `limit`, and come from the contributor's own prop check rather
 * than a generic guard.
 */
function getRefusalFindings(
  message: string,
  rejection: KernelAuditRejection | undefined
): KernelAuditFinding[] {
  if (!rejection) {
    return [{location: 'build', check: 'build throws', detail: message}];
  }
  const findings: KernelAuditFinding[] = [];
  if (!rejection.rejects.test(message)) {
    findings.push({location: 'build', check: 'unexpected refusal', detail: message});
  }
  if (rejection.limit === undefined || !message.includes(String(rejection.limit))) {
    findings.push({location: 'build', check: 'refusal names no documented limit', detail: message});
  }
  if (GENERIC_REFUSAL_PATTERNS.some(pattern => pattern.test(message))) {
    findings.push({location: 'build', check: 'generic refusal', detail: message});
  }
  return findings;
}

/** Core `maxBufferSize`, 256 MiB. */
const CORE_MAX_BUFFER_SIZE = 268435456;
/** Core `maxUniformBufferBindingSize`, 64 KiB. */
const CORE_MAX_UNIFORM_BINDING_SIZE = 65536;
/** Core `maxTextureDimension2D`. */
const CORE_MAX_TEXTURE_DIMENSION_2D = 8192;
/** Largest invocation index a `u32` global ID can hold. */
const MAXIMUM_INVOCATION_COUNT = 0xffffffff;

/** One audit finding of one case, with where it came from. */
export type KernelAuditFinding = WGSLFinding & {
  /** Node ID (and shader module ID for WGSL findings). */
  location: string;
};

type AuditedNode = {
  id: string;
  resources?: {
    buffer?: GraphBufferHandle | GraphDataView;
    texture?: GraphTextureHandle | {texture: GraphTextureHandle};
    usage: string;
  }[];
  workload?: {maximumInvocationCount?: number};
};

/** WGSL findings memoized within one case; never retain sources across the entire suite. */
function getModuleFindings(
  source: string,
  constants: Readonly<Record<string, number>>,
  moduleFindings: Map<string, WGSLFinding[]>
) {
  const key = `${source}\n${JSON.stringify(Object.entries(constants).sort())}`;
  let findings = moduleFindings.get(key);
  if (!findings) {
    findings = auditWGSLModule(source, constants);
    try {
      // wgsl_reflect treats the old attribute name `block` as a keyword and the enumerant
      // `workgroup` as reserved outside `var<...>`; WGSL allows both as identifiers, so rename them
      // for the parse only. (The GPU compile sweep confirmed Dawn accepts both.)
      new WgslReflect(
        source
          .replace(/\bblock\b/g, 'block_')
          .replace(/(?<!<\s*)\bworkgroup\b(?!\s*>)/g, 'workgroup_')
      );
    } catch (error) {
      // wgsl_reflect's const evaluator mistypes `const C: u32 = W - 1u` (W a u32 const) as f32;
      // that is not a syntax error, so only report parse failures.
      if (!/Invalid cast/.test(String(error))) {
        findings.push({check: 'wgsl_reflect parse', detail: String(error).slice(0, 200)});
      }
    }
    moduleFindings.set(key, findings);
  }
  return findings;
}

function getBindingBytes(buffer: GraphBufferHandle | GraphDataView): number {
  if ('rowByteLength' in buffer) {
    return (Math.max(buffer.length, 1) - 1) * buffer.byteStride + buffer.rowByteLength;
  }
  return buffer.byteLength;
}

/** Binding, buffer, texture and invocation-count findings of the declared nodes. */
function getResourceFindings(
  nodes: readonly AuditedNode[],
  maxStorageBufferBindingSize: number
): KernelAuditFinding[] {
  const findings: KernelAuditFinding[] = [];
  const add = (location: string, check: string, value: number, limit: number) => {
    if (value > limit) findings.push({location, check, detail: `${value} > ${limit}`});
  };
  for (const node of nodes) {
    for (const use of node.resources ?? []) {
      if (use.buffer) {
        const id = 'rowByteLength' in use.buffer ? use.buffer.buffer.id : use.buffer.id;
        const location = `${node.id} ${id}`;
        const bytes = getBindingBytes(use.buffer);
        if (use.usage.startsWith('storage')) {
          add(location, 'storage binding bytes', bytes, maxStorageBufferBindingSize);
        } else if (use.usage === 'uniform') {
          add(location, 'uniform binding bytes', bytes, CORE_MAX_UNIFORM_BINDING_SIZE);
        }
        const handle = 'rowByteLength' in use.buffer ? use.buffer.buffer : use.buffer;
        add(location, 'buffer bytes', handle.byteLength, CORE_MAX_BUFFER_SIZE);
      }
      if (use.texture) {
        const handle = 'width' in use.texture ? use.texture : use.texture.texture;
        const location = `${node.id} ${handle.id}`;
        add(location, 'texture width', handle.width, CORE_MAX_TEXTURE_DIMENSION_2D);
        add(location, 'texture height', handle.height, CORE_MAX_TEXTURE_DIMENSION_2D);
      }
    }
    const invocations = node.workload?.maximumInvocationCount ?? 0;
    add(node.id, 'capacity x lanes (invocations)', invocations, MAXIMUM_INVOCATION_COUNT);
  }
  return findings;
}

/** Result of auditing one case. */
export type KernelAuditResult = {
  /** Every finding; empty when the case passes. */
  findings: KernelAuditFinding[];
  /** Number of declared command nodes. */
  nodeCount: number;
  /** Number of recorded shader modules. */
  moduleCount: number;
  /** Whether the case was rejected at declaration as its `rejects` pattern expects. */
  rejected: boolean;
};

/**
 * Builds one case on a fresh fake core device, compiles every node on a recording device and
 * returns every finding: a build or compile throw (a prop the constructor accepted but the graph
 * cannot run, such as `capacity x lanes > 2^32 - 1`), WGSL traps and core-limit violations.
 */
export function auditKernelCase(auditCase: KernelAuditCase): KernelAuditResult {
  const context = createKernelAuditContext(auditCase.exportName);
  const rejection = getExpectedRejection(auditCase);
  try {
    try {
      auditCase.build(context);
    } catch (error) {
      const findings = getRefusalFindings(String(error), rejection);
      return {findings, nodeCount: 0, moduleCount: 0, rejected: findings.length === 0};
    }
    if (rejection) {
      return {
        findings: [
          {location: 'build', check: 'expected rejection', detail: String(rejection.rejects)}
        ],
        nodeCount: getDeclaredNodes(context).length,
        moduleCount: 0,
        rejected: false
      };
    }
    const nodes = getDeclaredNodes(context) as AuditedNode[];
    const findings = getResourceFindings(nodes, context.device.limits.maxStorageBufferBindingSize);
    const nodeCount = nodes.length;
    let moduleCount = 0;
    const moduleFindings = new Map<string, WGSLFinding[]>();
    try {
      recordShaderModules(nodes, module => {
        moduleCount++;
        for (const finding of getModuleFindings(module.source, module.constants, moduleFindings)) {
          findings.push({...finding, location: `${module.nodeId} ${module.id} (${module.stage})`});
        }
      });
    } catch (error) {
      findings.push({location: 'compile', check: 'compile throws', detail: String(error)});
    }
    return {findings, nodeCount, moduleCount, rejected: false};
  } finally {
    context.device.destroy();
  }
}

/** Formats findings one per line for an assertion message. */
export function formatKernelAuditFindings(findings: readonly KernelAuditFinding[]): string {
  return findings
    .map(({location, check, detail}) => `${check}: ${detail} @ ${location}`)
    .join('\n');
}
