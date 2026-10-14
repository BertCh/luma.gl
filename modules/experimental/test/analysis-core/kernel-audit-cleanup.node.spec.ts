// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device, type Shader} from '@luma.gl/core';
import {describe, expect, it} from 'vitest';
import {auditKernelCase} from './kernel-audit/kernel-audit-runner';
import {recordShaderModules} from './kernel-audit/kernel-audit-context';

describe('kernel audit resource cleanup', () => {
  it('audits modules and releases processed nodes before compiling the next node', () => {
    const events: string[] = [];
    const shaders: Shader[] = [];
    const nodes: unknown[] = ['first', 'second'].map((id, nodeIndex) => ({
      id,
      compile: ({device}: {device: Device}) => {
        if (nodeIndex > 0) {
          expect(nodes[nodeIndex - 1]).toBeUndefined();
          expect(shaders[nodeIndex - 1].destroyed).toBe(true);
        }
        events.push(`compile ${id}`);
        const shader = device.createShader({
          stage: 'compute',
          source: '@compute @workgroup_size(1) fn main() {}'
        });
        shaders.push(shader);
        device.createComputePipeline({id, shader, constants: {COUNT: nodeIndex + 1}});
        return {destroy: () => shader.destroy()};
      }
    }));
    recordShaderModules(nodes, module => {
      events.push(`audit ${module.nodeId}`);
      expect(module.id).toBe(module.nodeId);
      expect(module.stage).toBe('compute');
      expect(module.constants.COUNT).toBe(shaders.length);
    });
    expect(events).toEqual(['compile first', 'audit first', 'compile second', 'audit second']);
    expect(nodes).toEqual([]);
    expect(shaders.every(shader => shader.destroyed)).toBe(true);
  });

  it('releases unprocessed nodes and resources when a module visitor throws', () => {
    let shader: Shader | undefined;
    let recordingDevice: Device | undefined;
    const nodes: unknown[] = [
      {
        id: 'visitor-failure',
        compile: ({device}: {device: Device}) => {
          recordingDevice = device;
          shader = device.createShader({
            stage: 'compute',
            source: '@compute @workgroup_size(1) fn main() {}'
          });
          device.createComputePipeline({shader});
        }
      },
      {id: 'unprocessed', compile: () => expect.unreachable('compilation must stop')}
    ];
    expect(() =>
      recordShaderModules(nodes, () => {
        throw new Error('visitor failure');
      })
    ).toThrow('node visitor-failure compile failed: Error: visitor failure');
    expect(nodes).toEqual([]);
    expect(shader?.destroyed).toBe(true);
    expect(recordingDevice?.isLost).toBe(true);
  });

  it.each([
    'success',
    'build failure',
    'compile failure'
  ])('releases caller-owned and compiled resources after %s', outcome => {
    const buffers: Buffer[] = [];
    const devices: Device[] = [];
    let executableDestroyed = false;
    const result = auditKernelCase({
      exportName: 'cleanup',
      name: outcome,
      build: ({device, graph}) => {
        devices.push(device);
        // Like recipe parameters and draw commands, this buffer is borrowed by the graph.
        const buffer = device.createBuffer({byteLength: 16, usage: Buffer.STORAGE});
        buffers.push(buffer);
        graph.importBuffer(buffer);
        if (outcome === 'build failure') throw new Error('build failure');
        graph.add({
          id: 'cleanup',
          type: 'compute',
          compile: ({device: recordingDevice}) => {
            devices.push(recordingDevice);
            buffers.push(recordingDevice.createBuffer({byteLength: 16}));
            if (outcome === 'compile failure') throw new Error('compile failure');
            return {
              encode: () => {},
              destroy: () => {
                executableDestroyed = true;
              }
            };
          }
        });
      }
    });
    expect(result.findings.map(finding => finding.check)).toEqual(
      outcome === 'success' ? [] : [outcome === 'build failure' ? 'build throws' : 'compile throws']
    );
    expect(buffers.length).toBe(outcome === 'build failure' ? 1 : 2);
    expect(buffers.every(buffer => buffer.destroyed)).toBe(true);
    expect(devices.every(device => device.isLost)).toBe(true);
    expect(executableDestroyed).toBe(outcome === 'success');
  });
});
