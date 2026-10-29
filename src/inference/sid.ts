import type { InferenceTensor } from './backend';
import { ONNXWebGPUBackend } from './onnxWebgpu';
import { PseudoRawGain } from './adapter';
import { GPUPreprocessor } from '../preprocessing/gpuPreprocess';
import { TemporalAccumulator } from '../temporal';
import { Canvas2DRenderer, WebGPURenderer } from '../renderer';
import type {
  AdapterMode, DisplayOptions, ModelGeometry, ModelMetadata, MonoTint, OutlineMode, PipelineResult, PopLevel,
  TemporalLevel,
} from '../types';

export class SIDPipeline {
  private readonly sid: ONNXWebGPUBackend;
  private readonly adapter: ONNXWebGPUBackend | null;
  private device!: GPUDevice;
  private preprocessor!: GPUPreprocessor;
  private gainTransform!: PseudoRawGain;
  private temporal!: TemporalAccumulator;
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
  /** How a luminance-only model is shown. */
  tint: MonoTint = 'green';
  outline: OutlineMode = 'off';
  pop: PopLevel = 'off';
  /** 1 for luminance-only direct models, 2 with an outline plane, otherwise 3. */
  get outputChannels(): 1 | 2 | 3 {
    return (this.metadata.sid as { outputChannels?: 1 | 2 | 3 }).outputChannels ?? 3;
  }
  /** What the 13th plane encodes: frames averaged, or the per-frame noise estimate too. */
  private get noiseInput(): 'count' | 'estimate' {
    return (this.metadata.sid as { noiseInput?: 'count' | 'estimate' }).noiseInput ?? 'count';
  }
  /** 12 (packed RGB) or 13 (plus noise level) for direct models. */
  private get directChannels(): number {
    return (this.metadata.sid as { inputChannels?: number }).inputChannels ?? 12;
  }
  /** False while the GPU still owes the maximum number of undisplayed frames. */
  get ready(): boolean { return !this.cpuDisplay || !this.initialized || this.canvas2dRenderer.ready; }

  async initialize(report?: (message: string) => void): Promise<void> {
    report?.('Loading selected joint SID model (1/2)…');
    await this.sid.initialize();
    report?.('Loading learned pseudo-RAW adapter (2/2)…');
    await this.adapter?.initialize();
    report?.('Preparing camera GPU pipeline…');
    this.device = await this.sid.getDevice();
    this.preprocessor = new GPUPreprocessor(this.device, this.cpuDisplay);
    await this.preprocessor.initialize();
    this.gainTransform = new PseudoRawGain(this.device);
    this.temporal = new TemporalAccumulator(this.device);
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
    const prepared = await this.preprocessor.prepare(video, geometry, mode, gain, this.directChannels);
    const noisePlane = mode === 'direct' && this.directChannels === 13;
    const estimateNoise = noisePlane && this.noiseInput === 'estimate';
    // Visible packed cells (padding excluded) for the per-frame noise estimate.
    const visibleCells = [
      geometry.cropX / 2, geometry.cropY / 2,
      (geometry.cropX + geometry.width) / 2, (geometry.cropY + geometry.height) / 2,
    ] as const;
    this.temporal.apply(prepared.buffer, prepared.dims, temporalLevel, noisePlane,
      estimateNoise ? visibleCells : undefined);
    // Direct models receive the accumulated, unamplified packed RGB; gain and
    // clipping come after accumulation so averaging sees unclipped values.
    const plane = prepared.dims[2]! * prepared.dims[3]!;
    const modelInput = mode === 'direct'
      ? this.gainTransform.apply(prepared.buffer, gain, 12 * plane, prepared.dims[1]! * plane,
        estimateNoise ? this.temporal.sigmaBuffer : undefined)
      : prepared.buffer;
    const inputBackend = mode === 'learned' ? this.adapter! : this.sid;
    const input = inputBackend.tensorFromGpuBuffer(modelInput, prepared.dims);
    const preprocess = performance.now() - preprocessStart;

    let adapterMs = 0;
    let adapterOutput: InferenceTensor | null = null;
    let sidInput = input;
    let sidOutput: InferenceTensor | null = null;
    try {
      if (mode === 'learned') {
        const adapterStart = performance.now();
        adapterOutput = await this.adapter!.run(input);
        if (adapterOutput.location !== 'gpu-buffer') throw new Error('Adapter output is not GPU-resident.');
        const rawCount = geometry.rawWidth * geometry.rawHeight * 4;
        const gained = this.gainTransform.apply(adapterOutput.gpuBuffer, gain, rawCount);
        sidInput = this.sid.tensorFromGpuBuffer(gained, [1, 4, geometry.rawHeight, geometry.rawWidth]);
        adapterMs = performance.now() - adapterStart;
      }

      const sidStart = performance.now();
      sidOutput = await this.sid.run(sidInput);
      const sidMs = performance.now() - sidStart;
      if (sidOutput.location !== 'gpu-buffer') throw new Error('SID output is not GPU-resident.');
      const expected = [1, this.outputChannels, geometry.paddedHeight, geometry.paddedWidth];
      if (sidOutput.dims.length !== expected.length || sidOutput.dims.some((value, i) => value !== expected[i])) {
        throw new Error(`Unexpected SID output shape: ${sidOutput.dims.join('×')}`);
      }

      const postStart = performance.now();
      // Learned outlines need the model's outline plane.
      const outline = this.outline === 'learned' && this.outputChannels !== 2 ? 'off' : this.outline;
      const display: DisplayOptions = { channels: this.outputChannels, tint: this.tint, outline, pop: this.pop };
      let outputStats;
      if (this.cpuDisplay) {
        // Pack and map the smaller RGBA8 image asynchronously. Inference does
        // not wait for Safari's slow GPU-to-CPU presentation boundary.
        outputStats = this.canvas2dRenderer.render(sidOutput.gpuBuffer, geometry, display);
      } else {
        this.renderer.render(sidOutput.gpuBuffer, geometry, display);
        await this.device.queue.onSubmittedWorkDone();
      }
      const postprocess = performance.now() - postStart;
      // Do not await the histogram here: mapping it waits for every queued GPU
      // frame, which was the unaccounted time in `total`.
      const stats = prepared.stats;

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
        cameraInputPath: this.preprocessor.cameraInputPath,
        ...(this.cpuDisplay ? { presentation: this.canvas2dRenderer.presentation } : {}),
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
