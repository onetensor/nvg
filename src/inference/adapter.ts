import gainShader from '../../shaders/gain.wgsl?raw';
import type { InferenceTensor } from './backend';

const aligned = (bytes: number): number => Math.ceil(bytes / 16) * 16;

export class PseudoRawGain {
  private readonly pipeline: GPUComputePipeline;
  private readonly params: GPUBuffer;
  private output: GPUBuffer | null = null;
  private outputBytes = 0;
  private source: GPUBuffer | null = null;
  private bindGroup: GPUBindGroup | null = null;

  constructor(private readonly device: GPUDevice) {
    this.pipeline = device.createComputePipeline({
      label: 'pseudo-RAW gain',
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: gainShader }), entryPoint: 'main' },
    });
    this.params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  apply(input: InferenceTensor, gain: number, count: number): GPUBuffer {
    if (input.location !== 'gpu-buffer') throw new Error('Adapter output is not GPU-resident.');
    this.ensureOutput(count * 4);
    const values = new ArrayBuffer(16);
    new DataView(values).setUint32(0, count, true);
    new DataView(values).setFloat32(4, gain, true);
    this.device.queue.writeBuffer(this.params, 0, values);
    if (!this.bindGroup || this.source !== input.gpuBuffer) {
      this.source = input.gpuBuffer;
      this.bindGroup = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: input.gpuBuffer } },
          { binding: 1, resource: { buffer: this.output! } },
          { binding: 2, resource: { buffer: this.params } },
        ],
      });
    }
    const encoder = this.device.createCommandEncoder({ label: 'pseudo-RAW gain encoder' });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(count / 256));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return this.output!;
  }

  dispose(): void {
    this.output?.destroy();
    this.params.destroy();
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
