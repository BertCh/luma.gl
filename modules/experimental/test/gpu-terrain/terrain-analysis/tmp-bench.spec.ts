import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUTerrainViewshed,
  getGPUTerrainViewshedParameterValues,
  getGPUTerrainVisibilityToleranceParameterValues
} from '../../../src/gpu-terrain/terrain-analysis';
import {
  createInputBuffer,
  createOutputBuffer,
  isSoftwareDevice,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createFractalTerrain} from './terrain-analysis-oracle';

it('bench', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const lines: string[] = [`software=${isSoftwareDevice(device)}`];
  for (const [size, amplitude] of [
    [512, 0.01],
    [512, 3]
  ]) {
    const terrain = createFractalTerrain(size, size, 3, amplitude);
    const results: Record<string, number[]> = {};
    const codes: Record<string, number[]> = {};
    for (const traversal of ['march', 'pyramid'] as const) {
      const eb = createInputBuffer(device, terrain);
      const vb = createOutputBuffer(device, size * size);
      const sb = new GPUParameterBuffer(device, {
        id: 's',
        format: 'float32',
        length: 8,
        values: getGPUTerrainViewshedParameterValues({
          observer: [size / 2, size / 2],
          observerHeight: 40,
          cellSize: [30, 30],
          curvatureCoefficient: 6.8e-8
        })
      });
      const tb = new GPUParameterBuffer(device, {
        id: 't',
        format: 'float32',
        length: 4,
        values: getGPUTerrainVisibilityToleranceParameterValues({})
      });
      const graph = new GPUCommandGraph(device, {id: 'b'});
      graph.add(
        new GPUTerrainViewshed({
          width: size,
          height: size,
          elevation: {
            id: 'e',
            format: 'float32',
            storage: {
              kind: 'buffer',
              values: importGraphBuffer(graph, 'e', eb, 'float32', size * size)
            }
          },
          settings: sb.importToGraph(graph),
          traversal,
          tolerance: tb.importToGraph(graph),
          visibility: importGraphBuffer(graph, 'v', vb, 'uint32', size * size)
        })
      );
      const compiled = graph.compile();
      const times: number[] = [];
      for (let run = 0; run < 6; run++) {
        const start = performance.now();
        submitGraph(device, compiled, undefined);
        codes[traversal] = await readUint32(vb, size * size);
        times.push(performance.now() - start);
      }
      results[traversal] = times.slice(1).sort((a, b) => a - b);
      compiled.destroy();
    }
    const sum = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length;
    lines.push('meanSamples march=' + sum(codes.march) + ' pyramid=' + sum(codes.pyramid));
    const hidden = codes.march.filter(c => c === 0).length / codes.march.length;
    lines.push(
      `${size}x${size} amp=${amplitude} hidden=${hidden.toFixed(2)} march median=${results.march[2].toFixed(1)}ms pyramid median=${results.pyramid[2].toFixed(1)}ms  march all=${results.march.map(t => t.toFixed(0))} pyramid all=${results.pyramid.map(t => t.toFixed(0))}`
    );
  }
  throw new Error('BENCH\n' + lines.join('\n'));
}, 600000);
