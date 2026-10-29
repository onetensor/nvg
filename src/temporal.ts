import alignShader from '../shaders/align.wgsl?raw';
import temporalShader from '../shaders/temporal.wgsl?raw';
import type { TemporalLevel } from './types';

/** Maximum frames averaged per static pixel; `hold` is for a steadied phone. */
const MAX_FRAMES: Record<TemporalLevel, number> = { off: 1, low: 2, medium: 4, high: 8, hold: 32 };
// Thresholds on the neighbourhood-averaged signed change in sqrt space (see
// the `motion` pass): below LOW a cell is static, above HIGH it restarts from
// the current frame. Averaging ~100 differences keeps noise far below LOW.
const MOTION_LOW = 0.012;
const MOTION_HIGH = 0.04;
// Alignment search: +/-4 coarse pixels (16 packed cells, 32 output pixels)
// then +/-2 cells around the best coarse match.
const COARSE_RADIUS = 4;
const FINE_RADIUS = 2;
const COARSE_CANDIDATES = (2 * COARSE_RADIUS + 1) ** 2;
const FINE_CANDIDATES = (2 * FINE_RADIUS + 1) ** 2;
const ALIGN_PASSES = ['cell_maps', 'coarse_maps', 'coarse_search', 'coarse_pick', 'fine_search', 'fine_pick'] as const;

interface Side {
  history: GPUBuffer;
  frames: GPUBuffer;
}

/**
 * Input-side temporal denoising. Averaging the linear camera tensor before the
 * model raises its signal-to-noise ratio like a longer exposure would, which is
 * the main lever available without RAW sensor data. Each frame is first
 * aligned to the history by a global translation so hand shake does not reset
 * the average at every edge (`?align=0` disables this for comparison).
 */
export class TemporalAccumulator {
  private readonly pipeline: GPUComputePipeline;
  private readonly motionPipeline: GPUComputePipeline;
  private readonly alignLayout: GPUBindGroupLayout;
  private readonly alignPipelines: Record<(typeof ALIGN_PASSES)[number], GPUComputePipeline>;
  private readonly params: GPUBuffer;
  private readonly alignParams: GPUBuffer;
  private readonly shift: GPUBuffer;
  private readonly costs: GPUBuffer;
  private readonly alignEnabled = new URLSearchParams(globalThis.location?.search ?? '').get('align') !== '0';
  private sides: [Side, Side] | null = null;
  private motion: GPUBuffer | null = null;
  private maps: GPUBuffer | null = null;
  private parity = 0;
  private bindGroups: { main: GPUBindGroup; motion: GPUBindGroup; align: GPUBindGroup }[] = [];
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
    // Each alignment pass touches a different subset of bindings, so they
    // share one explicit layout instead of per-pipeline automatic ones.
    const compute = GPUShaderStage.COMPUTE;
    this.alignLayout = device.createBindGroupLayout({
      label: 'temporal alignment',
      entries: [
        { binding: 0, visibility: compute, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: compute, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: compute, buffer: { type: 'uniform' } },
        { binding: 3, visibility: compute, buffer: { type: 'storage' } },
        { binding: 4, visibility: compute, buffer: { type: 'storage' } },
        { binding: 5, visibility: compute, buffer: { type: 'storage' } },
      ],
    });
    const alignModule = device.createShaderModule({ label: 'temporal alignment', code: alignShader });
    const alignPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.alignLayout] });
    this.alignPipelines = Object.fromEntries(ALIGN_PASSES.map((entryPoint) => [
      entryPoint,
      device.createComputePipeline({
        label: `temporal alignment ${entryPoint}`,
        layout: alignPipelineLayout,
        compute: { module: alignModule, entryPoint },
      }),
    ])) as Record<(typeof ALIGN_PASSES)[number], GPUComputePipeline>;
    this.params = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.alignParams = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.shift = device.createBuffer({
      label: 'temporal alignment shift',
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.costs = device.createBuffer({
      label: 'temporal alignment costs',
      size: (128 + FINE_CANDIDATES) * 4,
      usage: GPUBufferUsage.STORAGE,
    });
  }

  /** The alignment shift buffer (dx, dy in packed cells, then the coarse estimate). */
  get shiftBuffer(): GPUBuffer {
    return this.shift;
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
    const height = dims[2]!;
    const width = dims[3]!;
    const plane = height * width;
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
    view.setUint32(28, width, true);
    this.device.queue.writeBuffer(this.params, 0, values);

    const groups = this.bindGroups[this.parity]!;
    const encoder = this.device.createCommandEncoder({ label: 'input temporal accumulation encoder' });
    if (this.initialized && this.alignEnabled && width >= 16 && height >= 16) {
      const coarseWidth = Math.floor(width / 4);
      const coarseHeight = Math.floor(height / 4);
      this.device.queue.writeBuffer(this.alignParams, 0, new Int32Array([
        width, height, channels, coarseWidth, coarseHeight, COARSE_RADIUS, FINE_RADIUS, 0,
      ]));
      const workgroups: Record<(typeof ALIGN_PASSES)[number], number> = {
        cell_maps: Math.ceil(plane / 256),
        coarse_maps: Math.ceil(coarseWidth * coarseHeight / 256),
        coarse_search: COARSE_CANDIDATES,
        coarse_pick: 1,
        fine_search: FINE_CANDIDATES,
        fine_pick: 1,
      };
      for (const entry of ALIGN_PASSES) {
        const pass = encoder.beginComputePass({ label: `temporal alignment ${entry}` });
        pass.setPipeline(this.alignPipelines[entry]);
        pass.setBindGroup(0, groups.align);
        pass.dispatchWorkgroups(workgroups[entry]);
        pass.end();
      }
    } else {
      this.device.queue.writeBuffer(this.shift, 0, new Int32Array(4));
    }
    if (this.initialized) {
      // Separate passes: the score reads neighbours that pass 2 overwrites.
      const scoring = encoder.beginComputePass({ label: 'input temporal motion score' });
      scoring.setPipeline(this.motionPipeline);
      scoring.setBindGroup(0, groups.motion);
      scoring.dispatchWorkgroups(Math.ceil(plane / 256));
      scoring.end();
    }
    const pass = encoder.beginComputePass({ label: 'input temporal accumulation' });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, groups.main);
    pass.dispatchWorkgroups(Math.ceil(plane / 256));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    // The history just written becomes next frame's input.
    this.parity = 1 - this.parity;
    this.initialized = level !== 'off';
  }

  reset(): void {
    this.initialized = false;
  }

  dispose(): void {
    for (const side of this.sides ?? []) {
      side.history.destroy();
      side.frames.destroy();
    }
    this.motion?.destroy();
    this.maps?.destroy();
    this.params.destroy();
    this.alignParams.destroy();
    this.shift.destroy();
    this.costs.destroy();
    this.bindGroups = [];
    this.source = null;
  }

  private ensureBuffers(frame: GPUBuffer, channels: number, plane: number): void {
    const key = `${channels}x${plane}`;
    if (this.key !== key) {
      for (const side of this.sides ?? []) {
        side.history.destroy();
        side.frames.destroy();
      }
      this.motion?.destroy();
      this.maps?.destroy();
      const storage = (label: string, floats: number): GPUBuffer => this.device.createBuffer({
        label,
        size: Math.ceil(floats * 4 / 16) * 16,
        usage: GPUBufferUsage.STORAGE,
      });
      this.sides = [0, 1].map((index) => ({
        history: storage(`input temporal history ${index}`, channels * plane),
        frames: storage(`input temporal frame counts ${index}`, plane),
      })) as [Side, Side];
      this.motion = storage('input temporal motion scores', plane);
      // Cell maps for frame and history, then two coarse maps of plane / 16 each.
      this.maps = storage('temporal alignment luma maps', plane * 2 + Math.ceil(plane / 8) + 2);
      this.key = key;
      this.bindGroups = [];
      this.parity = 0;
      this.initialized = false;
    }
    if (!this.bindGroups.length || this.source !== frame) {
      this.source = frame;
      const sides = this.sides!;
      this.bindGroups = [0, 1].map((parity) => {
        const input = sides[parity]!;
        const output = sides[1 - parity]!;
        return {
          main: this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: frame } },
              { binding: 1, resource: { buffer: input.history } },
              { binding: 2, resource: { buffer: input.frames } },
              { binding: 3, resource: { buffer: this.params } },
              { binding: 4, resource: { buffer: this.motion! } },
              { binding: 5, resource: { buffer: this.shift } },
              { binding: 6, resource: { buffer: output.history } },
              { binding: 7, resource: { buffer: output.frames } },
            ],
          }),
          motion: this.device.createBindGroup({
            layout: this.motionPipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: frame } },
              { binding: 1, resource: { buffer: input.history } },
              { binding: 3, resource: { buffer: this.params } },
              { binding: 4, resource: { buffer: this.motion! } },
              { binding: 5, resource: { buffer: this.shift } },
            ],
          }),
          align: this.device.createBindGroup({
            layout: this.alignLayout,
            entries: [
              { binding: 0, resource: { buffer: frame } },
              { binding: 1, resource: { buffer: input.history } },
              { binding: 2, resource: { buffer: this.alignParams } },
              { binding: 3, resource: { buffer: this.maps! } },
              { binding: 4, resource: { buffer: this.costs } },
              { binding: 5, resource: { buffer: this.shift } },
            ],
          }),
        };
      });
    }
  }
}
