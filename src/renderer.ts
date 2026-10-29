import postprocessShader from '../shaders/postprocess.wgsl?raw';
import type { ModelGeometry, OutputStats } from './types';

export class Canvas2DRenderer {
  private readonly context: CanvasRenderingContext2D;
  private pixels: ImageData | null = null;

  constructor(private readonly canvas: HTMLCanvasElement) {
    const context = canvas.getContext('2d', { alpha: false });
    if (!context) throw new Error('Could not create a Canvas2D display context.');
    this.context = context;
  }

  render(image: Float32Array, geometry: ModelGeometry): OutputStats {
    if (this.canvas.width !== geometry.width || this.canvas.height !== geometry.height) {
      this.canvas.width = geometry.width;
      this.canvas.height = geometry.height;
      this.pixels = null;
    }
    this.pixels ??= this.context.createImageData(geometry.width, geometry.height);
    const rgba = this.pixels.data;
    const plane = geometry.paddedWidth * geometry.paddedHeight;
    let minimum = Number.POSITIVE_INFINITY;
    let maximum = Number.NEGATIVE_INFINITY;
    let sum = 0;
    let nonFinite = 0;
    let destination = 0;
    for (let y = 0; y < geometry.height; y += 1) {
      const sourceRow = (y + geometry.cropY) * geometry.paddedWidth + geometry.cropX;
      for (let x = 0; x < geometry.width; x += 1) {
        const source = sourceRow + x;
        for (let channel = 0; channel < 3; channel += 1) {
          const raw = image[channel * plane + source]!;
          const value = Number.isFinite(raw) ? Math.min(1, Math.max(0, raw)) : 0;
          if (!Number.isFinite(raw)) nonFinite += 1;
          else {
            minimum = Math.min(minimum, raw);
            maximum = Math.max(maximum, raw);
            sum += raw;
          }
          rgba[destination + channel] = Math.round(value * 255);
        }
        rgba[destination + 3] = 255;
        destination += 4;
      }
    }
    this.context.putImageData(this.pixels, 0, 0);
    const finite = geometry.width * geometry.height * 3 - nonFinite;
    return {
      min: finite ? minimum : 0,
      mean: finite ? sum / finite : 0,
      max: finite ? maximum : 0,
      nonFinite,
    };
  }

  dispose(): void {
    this.pixels = null;
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
