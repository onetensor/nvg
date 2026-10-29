import type { InferenceTensor } from './backend';
import { ONNXWebGPUBackend } from './onnxWebgpu';
import { PseudoRawGain } from './adapter';
import { GPUPreprocessor } from '../preprocessing/gpuPreprocess';
import { TemporalFilter } from '../temporal';
import { Canvas2DRenderer, WebGPURenderer } from '../renderer';
import type { AdapterMode, ModelGeometry, ModelMetadata, PipelineResult, TemporalLevel } from '../types';

export class SIDPipeline {
  private readonly sid: ONNXWebGPUBackend;
  private readonly adapter: ONNXWebGPUBackend | null;
  private device!: GPUDevice;
  private preprocessor!: GPUPreprocessor;
  private gainTransform!: PseudoRawGain;
  private temporal!: TemporalFilter;
  private renderer!: WebGPURenderer;
  private canvas2dRenderer!: Canvas2DRenderer;
  private initialized = false;

  constructor(
    private readonly metadata: ModelMetadata,
    private readonly canvas: HTMLCanvasElement,
    graphCapture = false,
    sidOutputShape?: readonly number[],
    adapterOutputShape?: readonly number[],
    private readonly cpuDisplay = false,
  ) {
    this.sid = new ONNXWebGPUBackend({
      asset: metadata.sid,
      label: 'SID',
      graphCapture,
      ...(sidOutputShape ? { outputShape: sidOutputShape } : {}),
    });
    this.adapter = metadata.adapter.available
      ? new ONNXWebGPUBackend({
        asset: metadata.adapter,
        label: 'pseudo-RAW adapter',
        graphCapture,
        ...(adapterOutputShape ? { outputShape: adapterOutputShape } : {}),
      })
      : null;
  }

  get backendName(): string { return this.sid.name; }
  get fp16Available(): boolean { return this.device?.features.has('shader-f16') ?? false; }
  get learnedAdapterAvailable(): boolean { return this.adapter !== null; }

  async initialize(report?: (message: string) => void): Promise<void> {
    report?.('Loading selected joint SID model (1/2)…');
    await this.sid.initialize();
    report?.('Loading learned pseudo-RAW adapter (2/2)…');
    await this.adapter?.initialize();
    report?.('Preparing camera GPU pipeline…');
    this.device = await this.sid.getDevice();
    this.preprocessor = new GPUPreprocessor(this.device, this.cpuDisplay);
    this.gainTransform = new PseudoRawGain(this.device);
    this.temporal = new TemporalFilter(this.device);
    if (this.cpuDisplay) this.canvas2dRenderer = new Canvas2DRenderer(this.canvas, this.device);
    else this.renderer = new WebGPURenderer(this.canvas, this.device);
    this.initialized = true;
  }

  async process(
    video: HTMLVideoElement,
    geometry: ModelGeometry,
    mode: AdapterMode,
    gain: number,
    temporalLevel: TemporalLevel,
  ): Promise<PipelineResult> {
    if (!this.initialized) throw new Error('SID pipeline is not initialized.');
    if (mode === 'learned' && !this.adapter) throw new Error('The learned adapter is not packaged.');
    const totalStart = performance.now();

    const preprocessStart = performance.now();
    const prepared = this.preprocessor.prepare(video, geometry, mode, gain);
    const inputBackend = mode === 'learned' ? this.adapter! : this.sid;
    const input = inputBackend.tensorFromGpuBuffer(prepared.buffer, prepared.dims);
    const preprocess = performance.now() - preprocessStart;

    let adapterMs = 0;
    let adapterOutput: InferenceTensor | null = null;
    let sidInput = input;
    let sidOutput: InferenceTensor | null = null;
    try {
      if (mode === 'learned') {
        const adapterStart = performance.now();
        adapterOutput = await this.adapter!.run(input);
        const rawCount = geometry.rawWidth * geometry.rawHeight * 4;
        const gained = this.gainTransform.apply(adapterOutput, gain, rawCount);
        sidInput = this.sid.tensorFromGpuBuffer(gained, [1, 4, geometry.rawHeight, geometry.rawWidth]);
        adapterMs = performance.now() - adapterStart;
      }

      const sidStart = performance.now();
      sidOutput = await this.sid.run(sidInput);
      const sidMs = performance.now() - sidStart;
      if (sidOutput.location !== 'gpu-buffer') throw new Error('SID output is not GPU-resident.');
      const expected = [1, 3, geometry.paddedHeight, geometry.paddedWidth];
      if (sidOutput.dims.length !== expected.length || sidOutput.dims.some((value, i) => value !== expected[i])) {
        throw new Error(`Unexpected SID output shape: ${sidOutput.dims.join('×')}`);
      }

      const postStart = performance.now();
      let outputStats;
      if (this.cpuDisplay) {
        // Pack and map the smaller RGBA8 image asynchronously. Inference does
        // not wait for Safari's slow GPU-to-CPU presentation boundary.
        outputStats = this.canvas2dRenderer.render(sidOutput.gpuBuffer, geometry);
      } else {
        const displayBuffer = this.temporal.apply(
          sidOutput.gpuBuffer,
          geometry.paddedWidth,
          geometry.paddedHeight,
          temporalLevel,
        );
        this.renderer.render(displayBuffer, geometry);
        await this.device.queue.onSubmittedWorkDone();
      }
      const postprocess = performance.now() - postStart;
      const stats = prepared.stats ? await prepared.stats : undefined;

      return {
        timing: {
          preprocess,
          adapter: adapterMs,
          sid: sidMs,
          postprocess,
          total: performance.now() - totalStart,
        },
        ...(stats ? { stats } : {}),
        ...(outputStats ? { outputStats } : {}),
        gain,
      };
    } finally {
      if (sidOutput) this.sid.releaseOutput(sidOutput);
      if (adapterOutput) this.adapter!.releaseOutput(adapterOutput);
    }
  }

  resetTemporal(): void { this.temporal?.reset(); }

  async dispose(): Promise<void> {
    if (!this.initialized) return;
    // Static graph capture keeps references to the preallocated GPU inputs.
    // Release the sessions before destroying those buffers.
    await Promise.all([this.sid.dispose(), this.adapter?.dispose()]);
    this.preprocessor.dispose();
    this.gainTransform.dispose();
    this.temporal.dispose();
    if (this.cpuDisplay) this.canvas2dRenderer.dispose();
    else this.renderer.dispose();
    this.initialized = false;
  }
}
