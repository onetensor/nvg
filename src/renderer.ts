import postprocessShader from '../shaders/postprocess.wgsl?raw';
import readbackShader from '../shaders/readback.wgsl?raw';
import type { ModelGeometry, OutputStats } from './types';

interface ReadbackSlot {
  buffer: GPUBuffer;
  busy: boolean;
  pixels: ImageData | null;
  width: number;
  height: number;
}

export class Canvas2DRenderer {
  private readonly context: CanvasRenderingContext2D;
  private readonly pipeline: GPUComputePipeline;
  private readonly params: GPUBuffer;
  private readonly packed: GPUBuffer;
  private readonly slots: ReadbackSlot[];
  private source: GPUBuffer | null = null;
  private bindGroup: GPUBindGroup | null = null;
  private sequence = 0;
  private presentedSequence = 0;
  private lastStats: OutputStats | undefined;
  private disposed = false;

  constructor(private readonly canvas: HTMLCanvasElement, private readonly device: GPUDevice) {
    const context = canvas.getContext('2d', { alpha: false });
    if (!context) throw new Error('Could not create a Canvas2D display context.');
    this.context = context;
    this.pipeline = device.createComputePipeline({
      label: 'SID RGBA readback packer',
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ label: 'SID RGBA readback shader', code: readbackShader }),
        entryPoint: 'main',
      },
    });
    this.params = device.createBuffer({
      label: 'SID readback parameters',
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // Production exposes 180p and 360p. Three staging buffers let mapping
    // overlap subsequent inference without allocating memory per frame.
    const maximumBytes = 640 * 360 * 4;
    this.packed = device.createBuffer({
      label: 'SID packed RGBA',
      size: maximumBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.slots = Array.from({ length: 3 }, (_, index) => ({
      buffer: device.createBuffer({
        label: `SID asynchronous readback ${index + 1}`,
        size: maximumBytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }),
      busy: false,
      pixels: null,
      width: 0,
      height: 0,
    }));
  }

  render(image: GPUBuffer, geometry: ModelGeometry): OutputStats | undefined {
    if (this.canvas.width !== geometry.width || this.canvas.height !== geometry.height) {
      this.canvas.width = geometry.width;
      this.canvas.height = geometry.height;
    }

    const slot = this.slots.find((candidate) => !candidate.busy);
    if (!slot) return this.lastStats;
    const byteLength = geometry.width * geometry.height * 4;
    const values = new Uint32Array([
      geometry.paddedWidth,
      geometry.paddedHeight,
      geometry.width,
      geometry.height,
      geometry.cropX,
      geometry.cropY,
      0,
      0,
    ]);
    this.device.queue.writeBuffer(this.params, 0, values);
    if (!this.bindGroup || this.source !== image) {
      this.source = image;
      this.bindGroup = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: image } },
          { binding: 1, resource: { buffer: this.packed } },
          { binding: 2, resource: { buffer: this.params } },
        ],
      });
    }
    const encoder = this.device.createCommandEncoder({ label: 'SID asynchronous display encoder' });
    const pass = encoder.beginComputePass({ label: 'SID RGBA packing pass' });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(geometry.width / 8), Math.ceil(geometry.height / 8));
    pass.end();
    encoder.copyBufferToBuffer(this.packed, 0, slot.buffer, 0, byteLength);
    this.device.queue.submit([encoder.finish()]);

    slot.busy = true;
    const sequence = ++this.sequence;
    void this.present(slot, byteLength, geometry.width, geometry.height, sequence);
    return this.lastStats;
  }

  dispose(): void {
    this.disposed = true;
    this.params.destroy();
    this.packed.destroy();
    for (const slot of this.slots) {
      if (slot.buffer.mapState === 'mapped') slot.buffer.unmap();
      slot.buffer.destroy();
      slot.pixels = null;
    }
    this.bindGroup = null;
    this.source = null;
  }

  private async present(
    slot: ReadbackSlot,
    byteLength: number,
    width: number,
    height: number,
    sequence: number,
  ): Promise<void> {
    try {
      await slot.buffer.mapAsync(GPUMapMode.READ, 0, byteLength);
      if (this.disposed) return;
      if (!slot.pixels || slot.width !== width || slot.height !== height) {
        slot.pixels = this.context.createImageData(width, height);
        slot.width = width;
        slot.height = height;
      }
      const mapped = new Uint8Array(slot.buffer.getMappedRange(0, byteLength));
      slot.pixels.data.set(mapped);

      // Debug statistics do not need to tax every displayed frame.
      if (sequence % 30 === 1) {
        let minimum = 255;
        let maximum = 0;
        let sum = 0;
        for (let offset = 0; offset < mapped.length; offset += 4) {
          const red = mapped[offset]!;
          const green = mapped[offset + 1]!;
          const blue = mapped[offset + 2]!;
          minimum = Math.min(minimum, red, green, blue);
          maximum = Math.max(maximum, red, green, blue);
          sum += red + green + blue;
        }
        this.lastStats = {
          min: minimum / 255,
          mean: sum / (width * height * 3 * 255),
          max: maximum / 255,
          nonFinite: 0,
        };
      }
      slot.buffer.unmap();
      if (sequence >= this.presentedSequence &&
          this.canvas.width === width && this.canvas.height === height) {
        this.presentedSequence = sequence;
        this.context.putImageData(slot.pixels, 0, 0);
      }
    } catch (error) {
      if (!this.disposed) console.warn('Asynchronous SID display readback failed.', error);
    } finally {
      if (slot.buffer.mapState === 'mapped') slot.buffer.unmap();
      slot.busy = false;
    }
  }
}

export class WebGPURenderer {
  private readonly context: GPUCanvasContext;
  private readonly format: GPUTextureFormat;
  private readonly pipeline: GPURenderPipeline;
  private readonly params: GPUBuffer;
  private source: GPUBuffer | null = null;
  private bindGroup: GPUBindGroup | null = null;

  constructor(private readonly canvas: HTMLCanvasElement, private readonly device: GPUDevice) {
    const context = canvas.getContext('webgpu');
    if (!context) throw new Error('Could not create a WebGPU canvas context.');
    this.context = context;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.resizeCanvas();
    this.context.configure({ device, format: this.format, alphaMode: 'opaque' });
    const module = device.createShaderModule({ label: 'SID display shader', code: postprocessShader });
    this.pipeline = device.createRenderPipeline({
      label: 'SID display pipeline',
      layout: 'auto',
      vertex: { module, entryPoint: 'vertex_main' },
      fragment: { module, entryPoint: 'fragment_main', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list' },
    });
    this.params = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  render(buffer: GPUBuffer, geometry: ModelGeometry): void {
    if (this.resizeCanvas()) {
      // WebKit may discard the presentation backing store when the canvas size
      // changes. Explicitly reconfigure it before requesting the next texture.
      this.context.configure({ device: this.device, format: this.format, alphaMode: 'opaque' });
    }
    const values = new Uint32Array([
      geometry.paddedWidth,
      geometry.paddedHeight,
      geometry.width,
      geometry.height,
      geometry.cropX,
      geometry.cropY,
      0,
      0,
    ]);
    this.device.queue.writeBuffer(this.params, 0, values);
    if (!this.bindGroup || this.source !== buffer) {
      this.source = buffer;
      this.bindGroup = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer } },
          { binding: 1, resource: { buffer: this.params } },
        ],
      });
    }
    const encoder = this.device.createCommandEncoder({ label: 'SID display encoder' });
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        loadOp: 'clear',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        storeOp: 'store',
      }],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  dispose(): void {
    this.params.destroy();
    this.bindGroup = null;
    this.source = null;
    this.context.unconfigure();
  }

  private resizeCanvas(): boolean {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const height = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
      return true;
    }
    return false;
  }
}
