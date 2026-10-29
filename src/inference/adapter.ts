import gainShader from '../../shaders/gain.wgsl?raw';

const aligned = (bytes: number): number => Math.ceil(bytes / 16) * 16;

export class PseudoRawGain {
  private readonly pipeline: GPUComputePipeline;
  private readonly params: GPUBuffer;
  private output: GPUBuffer | null = null;
  private outputBytes = 0;
  private source: GPUBuffer | null = null;
  private sigmaSource: GPUBuffer | null = null;
  private bindGroup: GPUBindGroup | null = null;
  /** Bound when no noise estimate is used; the shader's binding is static. */
  private readonly unusedSigma: GPUBuffer;

  constructor(private readonly device: GPUDevice) {
    this.pipeline = device.createComputePipeline({
      label: 'pseudo-RAW gain',
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: gainShader }), entryPoint: 'main' },
    });
    this.params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.unusedSigma = device.createBuffer({ label: 'unused noise estimate', size: 16, usage: GPUBufferUsage.STORAGE });
  }

  /**
   * Amplifies and clips the first `count` floats of `input`, copying the rest
   * up to `total`; with `sigma`, the copied noise plane is scaled by
   * sqrt(gain) times the per-frame noise estimate it holds.
   */
  apply(input: GPUBuffer, gain: number, count: number, total = count, sigma?: GPUBuffer): GPUBuffer {
    this.ensureOutput(total * 4);
    const values = new ArrayBuffer(16);
    new DataView(values).setUint32(0, count, true);
    new DataView(values).setFloat32(4, gain, true);
    new DataView(values).setUint32(8, total, true);
    new DataView(values).setUint32(12, sigma ? 1 : 0, true);
    const sigmaBuffer = sigma ?? this.unusedSigma;
    this.device.queue.writeBuffer(this.params, 0, values);
    if (!this.bindGroup || this.source !== input || this.sigmaSource !== sigmaBuffer) {
      this.source = input;
      this.sigmaSource = sigmaBuffer;
      this.bindGroup = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: input } },
          { binding: 1, resource: { buffer: this.output! } },
          { binding: 2, resource: { buffer: this.params } },
          { binding: 3, resource: { buffer: sigmaBuffer } },
        ],
      });
    }
    const encoder = this.device.createCommandEncoder({ label: 'pseudo-RAW gain encoder' });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(total / 256));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return this.output!;
  }

  dispose(): void {
    this.output?.destroy();
    this.params.destroy();
    this.unusedSigma.destroy();
    this.bindGroup = null;
    this.source = null;
  }

  private ensureOutput(bytes: number): void {
    const size = aligned(bytes);
    if (this.output && this.outputBytes === size) return;
    this.output?.destroy();
    this.bindGroup = null;
    this.source = null;
    this.output = this.device.createBuffer({
      label: 'gained pseudo-RAW',
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.outputBytes = size;
  }
}
