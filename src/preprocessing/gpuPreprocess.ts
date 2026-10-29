import preprocessShader from '../../shaders/preprocess.wgsl?raw';
import type { AdapterMode, LuminanceStats, ModelGeometry } from '../types';

export interface PreparedFrame {
  buffer: GPUBuffer;
  dims: readonly number[];
  stats?: Promise<LuminanceStats>;
}

const HISTOGRAM_BYTES = 256 * 4;
const HISTOGRAM_INTERVAL = 30;
const zeroHistogram = new Uint32Array(256);

export class GPUPreprocessor {
  private readonly pipeline: GPUComputePipeline;
  private readonly sampler: GPUSampler;
  private readonly params: GPUBuffer;
  private readonly histogram: GPUBuffer;
  private output: GPUBuffer | null = null;
  private outputBytes = 0;
  private frameIndex = 0;
  private statsInFlight = false;

  constructor(private readonly device: GPUDevice, private readonly useVideoFrameSource = false) {
    this.pipeline = device.createComputePipeline({
      label: 'camera to model tensor',
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ label: 'camera preprocess', code: preprocessShader }),
        entryPoint: 'main',
      },
    });
    this.sampler = device.createSampler({ minFilter: 'linear', magFilter: 'linear' });
    this.params = device.createBuffer({
      label: 'preprocess parameters',
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.histogram = device.createBuffer({
      label: 'luminance histogram',
      size: HISTOGRAM_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
  }

  prepare(video: HTMLVideoElement, geometry: ModelGeometry, mode: AdapterMode, gain: number): PreparedFrame {
    const learned = mode === 'learned';
    const width = learned ? geometry.paddedWidth : geometry.rawWidth;
    const height = learned ? geometry.paddedHeight : geometry.rawHeight;
    const channels = learned ? 3 : 4;
    const count = width * height * channels;
    this.ensureOutput(count * 4);

    const collectStats = !this.statsInFlight && this.frameIndex++ % HISTOGRAM_INTERVAL === 0;
    if (collectStats) this.device.queue.writeBuffer(this.histogram, 0, zeroHistogram);

    const values = new ArrayBuffer(32);
    const view = new DataView(values);
    view.setUint32(0, width, true);
    view.setUint32(4, height, true);
    view.setUint32(8, geometry.width, true);
    view.setUint32(12, geometry.height, true);
    view.setUint32(16, learned ? 0 : 1, true);
    view.setUint32(20, collectStats ? 1 : 0, true);
    view.setFloat32(24, gain, true);
    this.device.queue.writeBuffer(this.params, 0, values);

    // WebKit has shipped camera-backed HTMLVideoElement imports that silently
    // sample black. Importing a VideoFrame snapshot is its documented
    // workaround and still keeps the frame on the browser's media/GPU path.
    let videoFrame: VideoFrame | null = null;
    if (this.useVideoFrameSource && typeof VideoFrame !== 'undefined') {
      try {
        videoFrame = new VideoFrame(video);
      } catch (error) {
        console.warn('Could not snapshot the camera as a VideoFrame; using the video element.', error);
      }
    }
    const externalTexture = this.device.importExternalTexture({ source: videoFrame ?? video });
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: externalTexture },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: { buffer: this.output! } },
        { binding: 3, resource: { buffer: this.params } },
        { binding: 4, resource: { buffer: this.histogram } },
      ],
    });

    const encoder = this.device.createCommandEncoder({ label: 'camera preprocess encoder' });
    const pass = encoder.beginComputePass({ label: 'camera preprocess pass' });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
    pass.end();

    let readback: GPUBuffer | undefined;
    if (collectStats) {
      readback = this.device.createBuffer({
        label: 'luminance histogram readback',
        size: HISTOGRAM_BYTES,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      encoder.copyBufferToBuffer(this.histogram, 0, readback, 0, HISTOGRAM_BYTES);
      this.statsInFlight = true;
    }
    this.device.queue.submit([encoder.finish()]);
    videoFrame?.close();
    // mapAsync must start after the copy command is submitted; mapping a buffer
    // first makes it unavailable to the command encoder on strict WebGPU implementations.
    const stats = readback
      ? this.readStats(readback).finally(() => { this.statsInFlight = false; })
      : undefined;

    return {
      buffer: this.output!,
      dims: [1, channels, height, width],
      ...(stats ? { stats } : {}),
    };
  }

  dispose(): void {
    this.output?.destroy();
    this.params.destroy();
    this.histogram.destroy();
  }

  private ensureOutput(bytes: number): void {
    const size = Math.ceil(bytes / 16) * 16;
    if (this.output && this.outputBytes === size) return;
    this.output?.destroy();
    this.output = this.device.createBuffer({
      label: 'model input tensor',
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.outputBytes = size;
  }

  private async readStats(readback: GPUBuffer): Promise<LuminanceStats> {
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const bins = new Uint32Array(readback.getMappedRange().slice(0));
      const total = bins.reduce((sum, count) => sum + count, 0);
      if (!total) return { mean: 0, p10: 0, p50: 0, p90: 0, blackFraction: 1 };

      let weighted = 0;
      let cumulative = 0;
      let p10 = 0;
      let p50 = 0;
      let p90 = 0;
      let found10 = false;
      let found50 = false;
      let found90 = false;
      bins.forEach((count, index) => {
        weighted += count * index / 255;
        cumulative += count;
        if (!found10 && cumulative >= total * 0.1) { p10 = index / 255; found10 = true; }
        if (!found50 && cumulative >= total * 0.5) { p50 = index / 255; found50 = true; }
        if (!found90 && cumulative >= total * 0.9) { p90 = index / 255; found90 = true; }
      });
      const black = (bins[0] ?? 0) + (bins[1] ?? 0) + (bins[2] ?? 0) + (bins[3] ?? 0);
      return { mean: weighted / total, p10, p50, p90, blackFraction: black / total };
    } finally {
      if (readback.mapState === 'mapped') readback.unmap();
      readback.destroy();
    }
  }
}
