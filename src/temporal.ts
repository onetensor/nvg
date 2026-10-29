import temporalShader from '../shaders/temporal.wgsl?raw';
import type { TemporalLevel } from './types';

/** Maximum frames averaged per static pixel; `hold` is for a steadied phone. */
const MAX_FRAMES: Record<TemporalLevel, number> = { off: 1, low: 2, medium: 4, high: 8, hold: 32 };
// Thresholds on the neighbourhood-averaged signed change in sqrt space (see
// the `motion` pass): below LOW a cell is static, above HIGH it restarts from
// the current frame. Averaging ~100 differences keeps noise far below LOW.
const MOTION_LOW = 0.012;
const MOTION_HIGH = 0.04;

/**
 * Input-side temporal denoising. Averaging the linear camera tensor before the
 * model raises its signal-to-noise ratio like a longer exposure would, which is
 * the main lever available without RAW sensor data.
 */
export class TemporalAccumulator {
  private readonly pipeline: GPUComputePipeline;
  private readonly motionPipeline: GPUComputePipeline;
  private readonly params: GPUBuffer;
  private history: GPUBuffer | null = null;
  private frames: GPUBuffer | null = null;
  private motion: GPUBuffer | null = null;
  private bindGroup: GPUBindGroup | null = null;
  private motionBindGroup: GPUBindGroup | null = null;
  private source: GPUBuffer | null = null;
  private key = '';
  private initialized = false;

  constructor(private readonly device: GPUDevice) {
    const module = device.createShaderModule({ label: 'input temporal accumulation', code: temporalShader });
    this.pipeline = device.createComputePipeline({
      label: 'input temporal accumulation',
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    this.motionPipeline = device.createComputePipeline({
      label: 'input temporal motion score',
      layout: 'auto',
      compute: { module, entryPoint: 'motion' },
    });
    this.params = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  /**
   * Accumulates `frame` (NCHW, batch 1) in place. With `noisePlane`, the last
   * plane is not image data: it receives 1/sqrt(frames averaged) per pixel,
   * so it is written even when accumulation is off.
   */
  apply(frame: GPUBuffer, dims: readonly number[], level: TemporalLevel, noisePlane = false): void {
    if (level === 'off' && !noisePlane) {
      this.initialized = false;
      return;
    }
    if (level === 'off') this.initialized = false;
    const channels = dims[1]! - (noisePlane ? 1 : 0);
    const plane = dims[2]! * dims[3]!;
    this.ensureBuffers(frame, dims[1]!, plane);
    const values = new ArrayBuffer(32);
    const view = new DataView(values);
    view.setUint32(0, plane, true);
    view.setUint32(4, channels, true);
    view.setUint32(8, this.initialized ? 1 : 0, true);
    view.setUint32(12, noisePlane ? 1 : 0, true);
    view.setFloat32(16, MAX_FRAMES[level], true);
    view.setFloat32(20, MOTION_LOW, true);
    view.setFloat32(24, MOTION_HIGH, true);
    view.setUint32(28, dims[3]!, true);
    this.device.queue.writeBuffer(this.params, 0, values);

    const encoder = this.device.createCommandEncoder({ label: 'input temporal accumulation encoder' });
    if (this.initialized) {
      // Separate passes: the score reads neighbours that pass 2 overwrites.
      const scoring = encoder.beginComputePass({ label: 'input temporal motion score' });
      scoring.setPipeline(this.motionPipeline);
      scoring.setBindGroup(0, this.motionBindGroup!);
      scoring.dispatchWorkgroups(Math.ceil(plane / 256));
      scoring.end();
    }
    const pass = encoder.beginComputePass({ label: 'input temporal accumulation' });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup!);
    pass.dispatchWorkgroups(Math.ceil(plane / 256));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    this.initialized = level !== 'off';
  }

  reset(): void {
    this.initialized = false;
  }

  dispose(): void {
    this.history?.destroy();
    this.frames?.destroy();
    this.motion?.destroy();
    this.params.destroy();
    this.bindGroup = null;
    this.motionBindGroup = null;
    this.source = null;
  }

  private ensureBuffers(frame: GPUBuffer, channels: number, plane: number): void {
    const key = `${channels}x${plane}`;
    if (this.key !== key) {
      this.history?.destroy();
      this.frames?.destroy();
      this.motion?.destroy();
      this.motion = this.device.createBuffer({
        label: 'input temporal motion scores',
        size: Math.ceil(plane * 4 / 16) * 16,
        usage: GPUBufferUsage.STORAGE,
      });
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
          { binding: 4, resource: { buffer: this.motion! } },
        ],
      });
      this.motionBindGroup = this.device.createBindGroup({
        layout: this.motionPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: frame } },
          { binding: 1, resource: { buffer: this.history! } },
          { binding: 3, resource: { buffer: this.params } },
          { binding: 4, resource: { buffer: this.motion! } },
        ],
      });
    }
  }
}
