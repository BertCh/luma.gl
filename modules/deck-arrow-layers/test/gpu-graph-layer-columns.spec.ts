// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUGraphEdgeLayer,
  GPUGraphNodeLayer,
  OrthographicView,
  type GPUGraphEdgeLayerProps,
  type GPUGraphNodeLayerProps
} from '@deck.gl-community/arrow-layers';
import {Buffer, Texture, type Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {afterAll, beforeAll, describe, expect, it, vi} from 'vitest';

import {ArrowDeck} from '../../../examples/deck/arrow-deck';

const WIDTH = 320;
const HEIGHT = 240;
const NODE_X = [-60, -20, 20, 60];
const PALETTE = [
  [255, 0, 0],
  [0, 255, 0],
  [0, 0, 255]
] as const;
const NULL_COLOR = [128, 128, 128] as const;

type Pixel = {r: number; g: number; b: number; a: number};
type GraphLayer = GPUGraphNodeLayer | GPUGraphEdgeLayer;

let device: Device;
let deck: ArrowDeck<OrthographicView>;
let framebuffer: ReturnType<Device['createFramebuffer']>;
let cleanup: Array<() => void> = [];
const ownedBuffers: Buffer[] = [];

beforeAll(async () => {
  const testDevice = await getWebGPUTestDevice('core');
  // Never skip silently: the WebGPU branch must execute for these tests to mean anything.
  expect(testDevice, 'a headless WebGPU device is required').toBeTruthy();
  device = testDevice!;
  expect(device.type).toBe('webgpu');

  const canvasContext = device.getDefaultCanvasContext();
  const canvas = canvasContext.canvas;
  if (!(canvas instanceof HTMLCanvasElement)) throw new Error('Expected an HTML canvas');
  const originalParent = canvas.parentNode;
  const originalNextSibling = canvas.nextSibling;
  const originalWidth = canvas.width;
  const originalHeight = canvas.height;
  const originalStyle = canvas.getAttribute('style');
  const originalDrawingBufferSize = canvasContext.getDrawingBufferSize();
  const container = document.createElement('div');
  Object.assign(container.style, {
    position: 'fixed',
    left: '0',
    top: '0',
    width: `${WIDTH}px`,
    height: `${HEIGHT}px`,
    overflow: 'hidden'
  });
  document.body.appendChild(container);
  container.appendChild(canvas);
  canvas.width = WIDTH;
  canvas.height = HEIGHT;
  canvas.style.width = `${WIDTH}px`;
  canvas.style.height = `${HEIGHT}px`;
  canvasContext.setDrawingBufferSize(WIDTH, HEIGHT);

  framebuffer = device.createFramebuffer({
    id: 'gpu-graph-layer-columns-framebuffer',
    width: WIDTH,
    height: HEIGHT,
    colorAttachments: [
      device.createTexture({
        id: 'gpu-graph-layer-columns-color',
        format: device.preferredColorFormat,
        width: WIDTH,
        height: HEIGHT,
        usage: Texture.RENDER | Texture.COPY_SRC | Texture.TEXTURE
      })
    ],
    depthStencilAttachment: 'depth24plus'
  });
  const currentFramebuffer = vi
    .spyOn(canvasContext, 'getCurrentFramebuffer')
    .mockReturnValue(framebuffer);

  let loaded: () => void = () => {};
  const loadedPromise = new Promise<void>(resolve => (loaded = resolve));
  deck = new ArrowDeck<OrthographicView>({
    parent: container,
    device,
    views: new OrthographicView({id: 'graph-columns'}),
    initialViewState: {target: [0, 0, 0], zoom: 0},
    controller: false,
    _animate: false,
    pickAsync: 'auto',
    layers: [],
    onLoad: () => loaded()
  });
  await loadedPromise;

  cleanup = [
    () => deck.finalize(),
    () => currentFramebuffer.mockRestore(),
    () => framebuffer.destroy(),
    () => ownedBuffers.splice(0).forEach(buffer => buffer.destroy()),
    () => {
      canvas.width = originalWidth;
      canvas.height = originalHeight;
      if (originalStyle === null) canvas.removeAttribute('style');
      else canvas.setAttribute('style', originalStyle);
      canvasContext.setDrawingBufferSize(
        originalDrawingBufferSize[0],
        originalDrawingBufferSize[1]
      );
      if (originalParent) originalParent.insertBefore(canvas, originalNextSibling);
      else canvas.remove();
      container.remove();
    }
  ];
});

afterAll(() => {
  for (const step of cleanup) step();
});

describe('GPU Graph layers read analysis-agnostic GPU columns', () => {
  it('colors nodes by a categorical uint32 column with a null color', async () => {
    const layer = makeNodeLayer({
      colorColumn: {
        buffer: makeColumn(Uint32Array.of(0, 1, 2, 0xffffffff)),
        format: 'uint32'
      },
      colorScale: {
        type: 'categorical',
        palette: PALETTE,
        nullColor: NULL_COLOR
      }
    });
    await renderLayers([layer]);
    const pixels = await readFramebuffer();
    const expected = [PALETTE[0], PALETTE[1], PALETTE[2], NULL_COLOR];
    expected.forEach((color, index) => {
      expectColor(getNodePixel(pixels, index), color, `categorical node ${index}`);
    });
  });

  it('colors nodes by a linear float32 column, with NaN as null', async () => {
    const layer = makeNodeLayer({
      colorColumn: {
        buffer: makeColumn(Float32Array.of(0, 0.5, 1, Number.NaN)),
        format: 'float32'
      },
      colorScale: {
        type: 'linear',
        domain: [0, 1],
        palette: [
          [255, 0, 0],
          [0, 0, 255]
        ],
        nullColor: NULL_COLOR
      }
    });
    await renderLayers([layer]);
    const pixels = await readFramebuffer();
    const expected = [[255, 0, 0], [127.5, 0, 127.5], [0, 0, 255], NULL_COLOR];
    expected.forEach((color, index) => {
      expectColor(getNodePixel(pixels, index), color, `linear node ${index}`);
    });
  });

  it('neither draws nor picks nodes removed by filterMask', async () => {
    const layer = makeNodeLayer({
      colorColumn: {
        buffer: makeColumn(Uint32Array.of(0, 1, 2, 0)),
        format: 'uint32'
      },
      colorScale: {type: 'categorical', palette: PALETTE},
      filterMask: makeColumn(Uint32Array.of(1, 0, 1, 1))
    });
    await renderLayers([layer]);
    const pixels = await readFramebuffer();
    expect(getNodePixel(pixels, 1).a, 'filtered node leaves the framebuffer empty').toBe(0);
    expect(getNodePixel(pixels, 0).a, 'unfiltered node is drawn').toBeGreaterThan(200);

    const filteredPick = await pickNode(1);
    expect(filteredPick, 'picking at a filtered node finds nothing').toBeFalsy();
    const unfilteredPick = await pickNode(2);
    expect(unfilteredPick?.index, 'picking an unfiltered node returns its row').toBe(2);
    expect(unfilteredPick?.layer?.id).toBe(layer.id);
  });

  it('dims nodes outside the highlight mask and tints on-path nodes', async () => {
    const layer = makeNodeLayer({
      colorColumn: {
        buffer: makeColumn(Uint32Array.of(0, 0, 0, 0)),
        format: 'uint32'
      },
      colorScale: {type: 'categorical', palette: PALETTE},
      highlightMask: makeColumn(Uint32Array.of(1, 0, 0, 0)),
      pathRanks: makeColumn(Uint32Array.of(0, 0, 1, 0)),
      pathColor: [0, 255, 255]
    });
    await renderLayers([layer]);
    const pixels = await readFramebuffer();
    expectColor(getNodePixel(pixels, 0), [255, 0, 0], 'highlighted node keeps full brightness');
    expectColor(getNodePixel(pixels, 1), [102, 0, 0], 'unhighlighted node dims to 0.40');
    expectColor(getNodePixel(pixels, 2), [0, 255, 255], 'on-path node takes the path color');
  });

  it('skips edges whose endpoint is filtered out', async () => {
    const layer = makeEdgeLayer({
      filterMask: makeColumn(Uint32Array.of(1, 1, 0, 1, 1, 1, 1, 1))
    });
    await renderLayers([layer]);
    const pixels = await readFramebuffer();
    expect(getEdgePixel(pixels, 0).a, 'unfiltered edge draws').toBeGreaterThan(30);
    expect(getEdgePixel(pixels, 1).a, 'edge with a filtered endpoint is not drawn').toBe(0);
    expect(getEdgePixel(pixels, 2).a).toBeGreaterThan(30);
  });

  it('highlights only consecutive path pairs and lets the path win', async () => {
    const layer = makeEdgeLayer({
      highlightMask: makeColumn(Uint32Array.of(1, 1, 0, 0, 1, 1, 1, 0)),
      pathRanks: makeColumn(Uint32Array.of(1, 2, 2, 3, 1, 3, 0, 2))
    });
    await renderLayers([layer]);
    const pixels = await readFramebuffer();
    const path = [255, 214, 97];
    const highlight = [242, 184, 82];
    const base = [82, 130, 179];
    expectColor(getEdgePixel(pixels, 0), path, 'ranks 1,2 are on the path and highlighted');
    expectColor(getEdgePixel(pixels, 1), path, 'ranks 2,3 are on the path');
    expectColor(
      getEdgePixel(pixels, 2),
      highlight,
      'ranks 1,3 are not consecutive but highlighted'
    );
    const dimmed = getEdgePixel(pixels, 3);
    expect(dimmed.a, 'unhighlighted edges dim but still draw').toBeGreaterThan(5);
    expect(dimmed.a).toBeLessThan(40);
    expectColor(dimmed, base, 'rank 0 is off the path', 24);
  });

  it('writes style uniforms only when their inputs change', async () => {
    const colorColumn = {
      buffer: makeColumn(Uint32Array.of(0, 1, 2, 0)),
      format: 'uint32'
    } as const;
    const colorScale = {type: 'categorical', palette: PALETTE} as const;
    const props = {colorColumn, colorScale};
    let layer = makeNodeLayer(props, 'counter-nodes');
    const edgeProps = {
      filterMask: makeColumn(Uint32Array.of(1, 1, 1, 1, 1, 1, 1, 1))
    };
    let edgeLayer = makeEdgeLayer(edgeProps, 'counter-edges');
    await renderLayers([edgeLayer, layer]);
    const pipeline = layer.getModels()[0].pipeline;
    expect(pipeline).toBeTruthy();
    const first = layer.getRenderStats();
    const firstEdge = edgeLayer.getRenderStats();
    expect(first.styleUniformWriteCount, 'one upload on creation').toBe(1);
    expect(first.bindingUpdateCount, 'no rebinding after creation').toBe(0);

    // Steady state: count every write that reaches a UNIFORM-usage GPU buffer, however it got
    // there: through luma's Buffer.write() or straight through GPUQueue.writeBuffer().
    const bufferPrototype = Object.getPrototypeOf(layer.getModels()[0].bindings.nodeStyle);
    const lumaWriteSpy = vi.spyOn(bufferPrototype, 'write');
    const queue = (device.handle as GPUDevice).queue;
    const queueWriteSpy = vi.spyOn(queue, 'writeBuffer');
    const perFrameWrites: number[] = [];
    const perFrameLumaWrites: number[] = [];
    const perFrameStyleWrites: number[] = [];
    for (let frame = 0; frame < 5; frame++) {
      lumaWriteSpy.mockClear();
      queueWriteSpy.mockClear();
      deck.redraw('steady state');
      const lumaWrites = lumaWriteSpy.mock.contexts.filter(
        context => ((context as Buffer).usage & Buffer.UNIFORM) !== 0
      );
      const queueWrites = queueWriteSpy.mock.calls.filter(
        ([gpuBuffer]) => (gpuBuffer.usage & GPUBufferUsage.UNIFORM) !== 0
      );
      perFrameLumaWrites.push(lumaWrites.length);
      perFrameWrites.push(queueWrites.length);
      perFrameStyleWrites.push(
        lumaWrites.filter(context => String((context as Buffer).id).includes('style-uniforms'))
          .length
      );
    }
    lumaWriteSpy.mockRestore();
    queueWriteSpy.mockRestore();
    console.log(
      `[gpu-graph-layer-columns] uniform-buffer writes per steady frame: queue.writeBuffer ${perFrameWrites}, Buffer.write ${perFrameLumaWrites}`
    );
    expect(new Set(perFrameWrites).size, 'steady-state uniform writes are constant').toBe(1);
    expect(perFrameStyleWrites, 'layers never write their style block while drawing').toEqual([
      0, 0, 0, 0, 0
    ]);

    const steady = layer.getRenderStats();
    expect(steady.drawCount - first.drawCount, 'every redraw draws once').toBe(5);
    expect(steady.styleUniformWriteCount).toBe(first.styleUniformWriteCount);
    expect(edgeLayer.getRenderStats().styleUniformWriteCount).toBe(
      firstEdge.styleUniformWriteCount
    );

    // Re-created layer with equal (new-identity) scale objects: no style write, no rebinding.
    layer = makeNodeLayer(
      {
        colorColumn,
        colorScale: {
          type: 'categorical',
          palette: PALETTE.map(color => [...color] as const)
        }
      },
      'counter-nodes'
    );
    edgeLayer = makeEdgeLayer(edgeProps, 'counter-edges');
    await renderLayers([edgeLayer, layer]);
    expect(layer.getRenderStats().styleUniformWriteCount, 'equal props write nothing').toBe(1);
    expect(layer.getRenderStats().bindingUpdateCount).toBe(0);

    // Changing the scale domain rewrites exactly the style block.
    layer = makeNodeLayer(
      {
        colorColumn,
        colorScale: {type: 'linear', domain: [0, 4], palette: PALETTE}
      },
      'counter-nodes'
    );
    const controlSpy = vi.spyOn((device.handle as GPUDevice).queue, 'writeBuffer');
    await renderLayers([edgeLayer, layer]);
    const gpuStyleWrites = controlSpy.mock.calls.filter(([gpuBuffer]) =>
      gpuBuffer.label.includes('counter-nodes-style-uniforms')
    );
    controlSpy.mockRestore();
    expect(gpuStyleWrites.length, 'the spy sees the one real GPU upload of the changed scale').toBe(
      1
    );
    expect(layer.getRenderStats().styleUniformWriteCount, 'new scale is one write').toBe(2);
    expect(layer.getRenderStats().bindingUpdateCount, 'scale change does not rebind').toBe(0);
    expect(layer.getModels()[0].pipeline, 'scale change keeps the pipeline').toBe(pipeline);

    // Swapping to a column of another format changes the format word and the binding: one each.
    layer = makeNodeLayer(
      {
        colorColumn: {
          buffer: makeColumn(Float32Array.of(0, 1, 2, 3)),
          format: 'float32'
        },
        colorScale: {type: 'linear', domain: [0, 4], palette: PALETTE}
      },
      'counter-nodes'
    );
    await renderLayers([edgeLayer, layer]);
    expect(layer.getRenderStats().styleUniformWriteCount, 'format swap is one write').toBe(3);
    expect(layer.getRenderStats().bindingUpdateCount, 'column swap is one rebinding').toBe(1);
    expect(layer.getModels()[0].pipeline, 'column swap keeps the pipeline').toBe(pipeline);

    // Same-format swap: one rebinding, no style write.
    layer = makeNodeLayer(
      {
        colorColumn: {
          buffer: makeColumn(Float32Array.of(3, 2, 1, 0)),
          format: 'float32'
        },
        colorScale: {type: 'linear', domain: [0, 4], palette: PALETTE}
      },
      'counter-nodes'
    );
    await renderLayers([edgeLayer, layer]);
    expect(layer.getRenderStats().styleUniformWriteCount).toBe(3);
    expect(layer.getRenderStats().bindingUpdateCount).toBe(2);
    expect(layer.getModels()[0].pipeline).toBe(pipeline);
    expect(edgeLayer.getRenderStats().styleUniformWriteCount, 'edge style is untouched').toBe(1);
  });

  it('throws a documented error on WebGL2 devices', () => {
    const source = GPUGraphNodeLayer.prototype.initializeState.toString();
    expect(source).toContain('requires WebGPU');
    expect(source).toContain('render CPU-side columns with a standard deck.gl layer instead');
  });
});

// Fixture helpers

let layerSerial = 0;
let sharedPositions: Buffer | null = null;
let sharedEdges: {
  positions: Buffer;
  sources: Buffer;
  targets: Buffer;
} | null = null;

function makeColumn(data: Uint32Array | Float32Array): Buffer {
  const buffer = device.createBuffer({
    data,
    usage: Buffer.STORAGE | Buffer.COPY_DST
  });
  ownedBuffers.push(buffer);
  return buffer;
}

function getNodePositions(): Buffer {
  if (!sharedPositions) {
    sharedPositions = device.createBuffer({
      data: Float32Array.from(NODE_X.flatMap(x => [x, 0])),
      usage: Buffer.VERTEX | Buffer.STORAGE | Buffer.COPY_DST
    });
    ownedBuffers.push(sharedPositions);
  }
  return sharedPositions;
}

/** Eight nodes in four horizontal pairs at rows y = -45, -15, 15, 45 (offset to pixel centers). */
function getEdgeGeometry() {
  if (!sharedEdges) {
    const rows = [-45.5, -15.5, 14.5, 44.5];
    const positions = device.createBuffer({
      data: Float32Array.from(rows.flatMap(y => [-60, y, 60, y])),
      usage: Buffer.VERTEX | Buffer.STORAGE | Buffer.COPY_DST
    });
    ownedBuffers.push(positions);
    sharedEdges = {
      positions,
      sources: makeColumn(Uint32Array.of(0, 2, 4, 6)),
      targets: makeColumn(Uint32Array.of(1, 3, 5, 7))
    };
  }
  return sharedEdges;
}

/** Layer state (including counters) survives re-instantiation by id, so tests pick their id. */
function makeNodeLayer(props: Partial<GPUGraphNodeLayerProps>, id?: string): GPUGraphNodeLayer {
  return new GPUGraphNodeLayer({
    id: id ?? `columns-nodes-${++layerSerial}`,
    positions: getNodePositions(),
    vertexCount: NODE_X.length,
    pointMode: false,
    radiusPixels: 8,
    pickable: true,
    ...props
  } as GPUGraphNodeLayerProps);
}

function makeEdgeLayer(props: Partial<GPUGraphEdgeLayerProps>, id?: string): GPUGraphEdgeLayer {
  const {positions, sources, targets} = getEdgeGeometry();
  return new GPUGraphEdgeLayer({
    id: id ?? `columns-edges-${++layerSerial}`,
    positions,
    sourceVertices: sources,
    targetVertices: targets,
    edgeCount: 4,
    ...props
  } as GPUGraphEdgeLayerProps);
}

/** Mounts layers, waits for pipelines to link, then draws one explicit frame. */
async function renderLayers(layers: GraphLayer[]): Promise<void> {
  layerSerial++;
  deck.setProps({layers});
  const deadline = performance.now() + 5_000;
  const getPipeline = (layer: GraphLayer) =>
    layer.getModels()[0]?.pipeline as {linkStatus?: string} | undefined;
  while (layers.some(layer => getPipeline(layer)?.linkStatus !== 'success')) {
    if (layers.some(layer => getPipeline(layer)?.linkStatus === 'error')) {
      throw new Error('A GPU graph layer pipeline failed to link');
    }
    if (performance.now() > deadline) throw new Error('GPU graph layer pipelines did not link');
    await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
  }
  deck.redraw(`gpu-graph-layer-columns frame ${layerSerial}`);
}

function getPixelRatio(): number {
  return device.getDefaultCanvasContext().cssToDeviceRatio();
}

function projectToDevice(x: number, y: number): [number, number] {
  const [px, py] = deck.getViewports()[0].project([x, y, 0]);
  const ratio = getPixelRatio();
  return [Math.floor(px * ratio), Math.floor(py * ratio)];
}

async function pickNode(index: number) {
  const [x, y] = deck.getViewports()[0].project([NODE_X[index], 0, 0]);
  return deck.pickObjectAsync({
    x: Math.floor(x),
    y: Math.floor(y),
    radius: 3
  });
}

type Pixels = {width: number; height: number; data: Uint8Array};

async function readFramebuffer(): Promise<Pixels> {
  const texture = framebuffer.colorAttachments[0].texture;
  const layout = texture.computeMemoryLayout({width: WIDTH, height: HEIGHT});
  const buffer = device.createBuffer({
    byteLength: layout.byteLength,
    usage: Buffer.COPY_DST | Buffer.MAP_READ
  });
  try {
    texture.readBuffer({width: WIDTH, height: HEIGHT}, buffer);
    const bytes = await buffer.readAsync(0, layout.byteLength);
    const data = new Uint8Array(WIDTH * HEIGHT * 4);
    const swapRedBlue = texture.format.startsWith('bgra');
    for (let row = 0; row < HEIGHT; row++) {
      for (let column = 0; column < WIDTH; column++) {
        const source = bytes.byteOffset + row * layout.bytesPerRow + column * 4;
        const target = (row * WIDTH + column) * 4;
        data[target] = bytes[source - bytes.byteOffset + (swapRedBlue ? 2 : 0)];
        data[target + 1] = bytes[source - bytes.byteOffset + 1];
        data[target + 2] = bytes[source - bytes.byteOffset + (swapRedBlue ? 0 : 2)];
        data[target + 3] = bytes[source - bytes.byteOffset + 3];
      }
    }
    return {width: WIDTH, height: HEIGHT, data};
  } finally {
    buffer.destroy();
  }
}

/** Un-premultiplies: blending stores `rgb * alpha` because the layers use src-alpha blending. */
function readPixel(pixels: Pixels, x: number, y: number): Pixel {
  const offset = (y * pixels.width + x) * 4;
  const a = pixels.data[offset + 3];
  const scale = a > 0 ? 255 / a : 0;
  return {
    r: pixels.data[offset] * scale,
    g: pixels.data[offset + 1] * scale,
    b: pixels.data[offset + 2] * scale,
    a
  };
}

function getNodePixel(pixels: Pixels, index: number): Pixel {
  const [x, y] = projectToDevice(NODE_X[index], 0);
  return readPixel(pixels, x, y);
}

/** Strongest pixel in the column at x = 0 within two rows of the edge's pixel row. */
function getEdgePixel(pixels: Pixels, edge: number): Pixel {
  const rows = [-45.5, -15.5, 14.5, 44.5];
  const [x, y] = projectToDevice(0, rows[edge]);
  let best: Pixel = {r: 0, g: 0, b: 0, a: 0};
  for (let row = y - 2; row <= y + 2; row++) {
    const pixel = readPixel(pixels, x, row);
    if (pixel.a > best.a) best = pixel;
  }
  return best;
}

function expectColor(
  pixel: Pixel,
  expected: readonly number[],
  label: string,
  tolerance = 10
): void {
  expect(
    [pixel.r, pixel.g, pixel.b].map((channel, index) => Math.abs(channel - expected[index])),
    `${label}: got ${JSON.stringify(pixel)}`
  ).toSatisfy((differences: number[]) => differences.every(difference => difference <= tolerance));
}
