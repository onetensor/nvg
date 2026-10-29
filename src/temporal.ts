import temporalShader from '../shaders/temporal.wgsl?raw';
import type { TemporalLevel } from './types';

const HISTORY: Record<TemporalLevel, number> = { off: 0, low: 0.18, medium: 0.35, high: 0.55 };

export class TemporalFilter {
  private readonly pipeline: GPUComputePipeline;
  private readonly params: GPUBuffer;
  private previous: GPUBuffer | null = null;
  private bytes = 0;
  private initialized = false;

  constructor(private readonly device: GPUDevice) {
    this.pipeline = device.createComputePipeline({
      label: 'motion-aware temporal filter',
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: temporalShader }), entryPoint: 'main' },
    });
    this.params = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  apply(current: GPUBuffer, width: number, height: number, level: TemporalLevel): GPUBuffer {
    if (level === 'off') return current;
    const plane = width * height;
    const count = plane * 3;
    this.ensurePrevious(count * 4);
    const values = new ArrayBuffer(32);
    const view = new DataView(values);
    view.setUint32(0, count, true);
    view.setUint32(4, plane, true);
    view.setUint32(8, this.initialized ? 1 : 0, true);
    view.setFloat32(16, HISTORY[level], true);
    this.device.queue.writeBuffer(this.params, 0, values);

    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: current } },
        { binding: 1, resource: { buffer: this.previous! } },
        { binding: 2, resource: { buffer: this.params } },
      ],
    });
    const encoder = this.device.createCommandEncoder({ label: 'temporal filter encoder' });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(plane / 256));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    this.initialized = true;
    return this.previous!;
  }

  reset(): void {
    this.initialized = false;
  }

  dispose(): void {
    this.previous?.destroy();
    this.params.destroy();
  }

  private ensurePrevious(bytes: number): void {
    const size = Math.ceil(bytes / 16) * 16;
    if (this.previous && this.bytes === size) return;
    this.previous?.destroy();
    this.previous = this.device.createBuffer({
      label: 'previous enhanced frame',
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.bytes = size;
    this.initialized = false;
  }
}
