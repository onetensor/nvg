import temporalShader from '../shaders/temporal.wgsl?raw';
import type { TemporalLevel } from './types';

/** Maximum frames averaged per static pixel. */
const MAX_FRAMES: Record<TemporalLevel, number> = { off: 1, low: 2, medium: 4, high: 8 };
// Thresholds on |sqrt(now) - sqrt(history)|: below LOW a pixel is static,
// above HIGH it restarts from the current frame.
const MOTION_LOW = 0.02;
const MOTION_HIGH = 0.06;

/**
 * Input-side temporal denoising. Averaging the linear camera tensor before the
 * model raises its signal-to-noise ratio like a longer exposure would, which is
 * the main lever available without RAW sensor data.
 */
export class TemporalAccumulator {
  private readonly pipeline: GPUComputePipeline;
  private readonly params: GPUBuffer;
  private history: GPUBuffer | null = null;
  private frames: GPUBuffer | null = null;
  private bindGroup: GPUBindGroup | null = null;
  private source: GPUBuffer | null = null;
  private key = '';
  private initialized = false;

  constructor(private readonly device: GPUDevice) {
    this.pipeline = device.createComputePipeline({
      label: 'input temporal accumulation',
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: temporalShader }), entryPoint: 'main' },
    });
    this.params = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  /** Accumulates `frame` (NCHW, batch 1) in place. */
  apply(frame: GPUBuffer, dims: readonly number[], level: TemporalLevel): void {
    if (level === 'off') {
      this.initialized = false;
      return;
    }
    const channels = dims[1]!;
    const plane = dims[2]! * dims[3]!;
    this.ensureBuffers(frame, channels, plane);
    const values = new ArrayBuffer(32);
    const view = new DataView(values);
    view.setUint32(0, plane, true);
    view.setUint32(4, channels, true);
    view.setUint32(8, this.initialized ? 1 : 0, true);
    view.setFloat32(16, MAX_FRAMES[level], true);
    view.setFloat32(20, MOTION_LOW, true);
    view.setFloat32(24, MOTION_HIGH, true);
    this.device.queue.writeBuffer(this.params, 0, values);

    const encoder = this.device.createCommandEncoder({ label: 'input temporal accumulation encoder' });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup!);
    pass.dispatchWorkgroups(Math.ceil(plane / 256));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    this.initialized = true;
  }

  reset(): void {
    this.initialized = false;
  }

  dispose(): void {
    this.history?.destroy();
    this.frames?.destroy();
    this.params.destroy();
    this.bindGroup = null;
    this.source = null;
  }

  private ensureBuffers(frame: GPUBuffer, channels: number, plane: number): void {
    const key = `${channels}x${plane}`;
    if (this.key !== key) {
      this.history?.destroy();
      this.frames?.destroy();
      this.history = this.device.createBuffer({
        label: 'input temporal history',
        size: Math.ceil(channels * plane * 4 / 16) * 16,
        usage: GPUBufferUsage.STORAGE,
      });
      this.frames = this.device.createBuffer({
        label: 'input temporal frame counts',
        size: Math.ceil(plane * 4 / 16) * 16,
        usage: GPUBufferUsage.STORAGE,
      });
      this.key = key;
      this.bindGroup = null;
      this.initialized = false;
    }
    if (!this.bindGroup || this.source !== frame) {
      this.source = frame;
      this.bindGroup = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: frame } },
          { binding: 1, resource: { buffer: this.history! } },
          { binding: 2, resource: { buffer: this.frames! } },
          { binding: 3, resource: { buffer: this.params } },
        ],
      });
    }
  }
}
