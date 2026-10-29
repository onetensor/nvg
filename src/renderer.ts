import postprocessShader from '../shaders/postprocess.wgsl?raw';
import type { ModelGeometry } from './types';

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
    this.resizeCanvas();
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

  private resizeCanvas(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const height = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
  }
}
