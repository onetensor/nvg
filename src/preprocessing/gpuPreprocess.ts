import preprocessShader from '../../shaders/preprocess.wgsl?raw';
import preprocessTextureShader from '../../shaders/preprocessTexture.wgsl?raw';
import { IPHONE_CAMERA_INPUT_PATHS } from '../cameraInput';
import { coverUvScale } from '../geometry';
import type { AdapterMode, CameraInputPath, LuminanceStats, ModelGeometry } from '../types';

export interface PreparedFrame {
  buffer: GPUBuffer;
  dims: readonly number[];
  stats?: Promise<LuminanceStats>;
}

const HISTOGRAM_BINS = 257;
const HISTOGRAM_BYTES = HISTOGRAM_BINS * 4;
const HISTOGRAM_INTERVAL = 30;
const zeroHistogram = new Uint32Array(HISTOGRAM_BINS);

interface CameraResource {
  resource: GPUExternalTexture | GPUTextureView;
  width: number;
  height: number;
}

export class GPUPreprocessor {
  private readonly pipelinePromise: Promise<GPUComputePipeline>;
  private readonly sampler: GPUSampler;
  private readonly params: GPUBuffer;
  private readonly histogram: GPUBuffer;
  private output: GPUBuffer | null = null;
  private outputBytes = 0;
  private cameraTexture: GPUTexture | null = null;
  private cameraTextureView: GPUTextureView | null = null;
  private cameraTextureWidth = 0;
  private cameraTextureHeight = 0;
  private stagingCanvas: HTMLCanvasElement | null = null;
  private stagingContext: CanvasRenderingContext2D | null = null;
  private probeCanvas: HTMLCanvasElement | null = null;
  private probeContext: CanvasRenderingContext2D | null = null;
  private readonly validatedCopyPaths = new Set<CameraInputPath>();
  private preprocessValidated = false;
  private pathIndex = 0;
  private inputValidated: boolean;
  private exactBlackProbes = 0;
  private inputFailure: string | null = null;
  private frameIndex = 0;
  private statsInFlight = false;

  constructor(private readonly device: GPUDevice, private readonly useCopiedCameraTexture = false) {
    const module = device.createShaderModule({
      label: 'camera preprocess',
      code: useCopiedCameraTexture ? preprocessTextureShader : preprocessShader,
    });
    this.pipelinePromise = device.createComputePipelineAsync({
      label: 'camera to model tensor',
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    this.sampler = device.createSampler({ minFilter: 'linear', magFilter: 'linear' });
    this.params = device.createBuffer({
      label: 'preprocess parameters',
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.histogram = device.createBuffer({
      label: 'luminance histogram',
      size: HISTOGRAM_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.inputValidated = !useCopiedCameraTexture;
  }

  get cameraInputPath(): CameraInputPath {
    return this.useCopiedCameraTexture ? IPHONE_CAMERA_INPUT_PATHS[this.pathIndex]! : 'external-texture';
  }

  async initialize(): Promise<void> {
    await this.pipelinePromise;
  }

  async prepare(
    video: HTMLVideoElement,
    geometry: ModelGeometry,
    mode: AdapterMode,
    gain: number,
  ): Promise<PreparedFrame> {
    if (this.inputFailure) throw new Error(this.inputFailure);
    const pipeline = await this.pipelinePromise;
    const learned = mode === 'learned';
    const width = learned ? geometry.paddedWidth : geometry.rawWidth;
    const height = learned ? geometry.paddedHeight : geometry.rawHeight;
    const channels = learned ? 3 : 4;
    const count = width * height * channels;
    this.ensureOutput(count * 4);

    const camera = await this.cameraResource(video, geometry);
    const cpuProbeNonzero = this.useCopiedCameraTexture && !this.inputValidated
      ? this.probeVideo(video)
      : false;
    const collectStats = !this.statsInFlight &&
      (!this.inputValidated || this.frameIndex % HISTOGRAM_INTERVAL === 0);
    this.frameIndex += 1;
    if (collectStats) this.device.queue.writeBuffer(this.histogram, 0, zeroHistogram);

    const uvScale = coverUvScale(camera.width, camera.height, geometry.width, geometry.height);
    const values = new ArrayBuffer(48);
    const view = new DataView(values);
    view.setUint32(0, width, true);
    view.setUint32(4, height, true);
    view.setUint32(8, geometry.width, true);
    view.setUint32(12, geometry.height, true);
    view.setUint32(16, learned ? 0 : 1, true);
    view.setUint32(20, collectStats ? 1 : 0, true);
    view.setFloat32(24, gain, true);
    view.setFloat32(32, uvScale.x, true);
    view.setFloat32(36, uvScale.y, true);
    this.device.queue.writeBuffer(this.params, 0, values);

    const validatePreprocess = !this.preprocessValidated;
    if (validatePreprocess) this.device.pushErrorScope('validation');
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: camera.resource },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: { buffer: this.output! } },
        { binding: 3, resource: { buffer: this.params } },
        { binding: 4, resource: { buffer: this.histogram } },
      ],
    });

    const encoder = this.device.createCommandEncoder({ label: 'camera preprocess encoder' });
    const pass = encoder.beginComputePass({ label: 'camera preprocess pass' });
    pass.setPipeline(pipeline);
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
    if (validatePreprocess) {
      const validationError = await this.device.popErrorScope();
      if (validationError) {
        if (readback) {
          readback.destroy();
          this.statsInFlight = false;
        }
        throw new Error(`Camera preprocessing validation failed: ${validationError.message}`);
      }
      this.preprocessValidated = true;
    }
    const stats = readback
      ? this.readStats(readback, cpuProbeNonzero).finally(() => { this.statsInFlight = false; })
      : undefined;

    return {
      buffer: this.output!,
      dims: [1, channels, height, width],
      ...(stats ? { stats } : {}),
    };
  }

  dispose(): void {
    this.output?.destroy();
    this.cameraTexture?.destroy();
    this.params.destroy();
    this.histogram.destroy();
    this.stagingCanvas = null;
    this.stagingContext = null;
    this.probeCanvas = null;
    this.probeContext = null;
  }

  private async cameraResource(video: HTMLVideoElement, geometry: ModelGeometry): Promise<CameraResource> {
    if (!this.useCopiedCameraTexture) {
      if (!video.videoWidth || !video.videoHeight) throw new Error('The camera has no decoded frame yet.');
      return {
        resource: this.device.importExternalTexture({ source: video }),
        width: video.videoWidth,
        height: video.videoHeight,
      };
    }

    while (this.pathIndex < IPHONE_CAMERA_INPUT_PATHS.length) {
      const path = this.cameraInputPath;
      try {
        return await this.copyCameraFrame(video, geometry, path);
      } catch (error) {
        const reason = error instanceof Error ? error.message : String(error);
        if (!this.advanceCameraPath(`${path} failed validation: ${reason}`)) {
          throw new Error(this.inputFailure!);
        }
      }
    }
    throw new Error(this.inputFailure ?? 'No iPhone camera ingestion path remains.');
  }

  private async copyCameraFrame(
    video: HTMLVideoElement,
    geometry: ModelGeometry,
    path: CameraInputPath,
  ): Promise<CameraResource> {
    let source: HTMLVideoElement | VideoFrame | HTMLCanvasElement = video;
    let frame: VideoFrame | null = null;
    let sourceWidth = video.videoWidth;
    let sourceHeight = video.videoHeight;

    if (path === 'video-frame-copy') {
      if (typeof VideoFrame === 'undefined') throw new Error('VideoFrame is unavailable.');
      frame = new VideoFrame(video);
      source = frame;
      sourceWidth = frame.displayWidth;
      sourceHeight = frame.displayHeight;
    } else if (path === 'canvas-copy') {
      const canvas = this.ensureStagingCanvas(geometry.width, geometry.height);
      this.stagingContext!.drawImage(video, 0, 0, canvas.width, canvas.height);
      source = canvas;
      sourceWidth = canvas.width;
      sourceHeight = canvas.height;
    }

    if (sourceWidth <= 0 || sourceHeight <= 0) {
      frame?.close();
      throw new Error('The camera source has zero dimensions.');
    }
    this.ensureCameraTexture(sourceWidth, sourceHeight);

    const validate = !this.validatedCopyPaths.has(path);
    if (validate) this.device.pushErrorScope('validation');
    let scopePopped = false;
    try {
      this.device.queue.copyExternalImageToTexture(
        { source },
        { texture: this.cameraTexture!, colorSpace: 'srgb' },
        [sourceWidth, sourceHeight],
      );
      frame?.close();
      frame = null;
      if (validate) {
        const validationError = await this.device.popErrorScope();
        scopePopped = true;
        if (validationError) throw new Error(validationError.message);
        this.validatedCopyPaths.add(path);
      }
    } catch (error) {
      if (validate && !scopePopped) {
        await this.device.popErrorScope();
        scopePopped = true;
      }
      throw error;
    } finally {
      frame?.close();
    }

    return { resource: this.cameraTextureView!, width: sourceWidth, height: sourceHeight };
  }

  private advanceCameraPath(reason: string): boolean {
    console.warn(reason);
    this.pathIndex += 1;
    this.inputValidated = false;
    this.exactBlackProbes = 0;
    this.frameIndex = 0;
    if (this.pathIndex < IPHONE_CAMERA_INPUT_PATHS.length) {
      console.warn(`Retrying camera ingestion with ${this.cameraInputPath}.`);
      return true;
    }
    this.inputFailure = `Safari returned black pixels through every camera upload path (${reason}).`;
    return false;
  }

  private probeVideo(video: HTMLVideoElement): boolean {
    this.probeCanvas ??= document.createElement('canvas');
    this.probeCanvas.width = 16;
    this.probeCanvas.height = 16;
    this.probeContext ??= this.probeCanvas.getContext('2d', { alpha: false, willReadFrequently: true });
    if (!this.probeContext) return false;
    try {
      this.probeContext.drawImage(video, 0, 0, 16, 16);
      const pixels = this.probeContext.getImageData(0, 0, 16, 16).data;
      for (let index = 0; index < pixels.length; index += 4) {
        if (pixels[index]! || pixels[index + 1]! || pixels[index + 2]!) return true;
      }
    } catch (error) {
      console.warn('Camera CPU probe failed.', error);
    }
    return false;
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

  private ensureCameraTexture(width: number, height: number): void {
    if (this.cameraTexture &&
        this.cameraTextureWidth === width && this.cameraTextureHeight === height) return;
    this.cameraTexture?.destroy();
    this.cameraTexture = this.device.createTexture({
      label: 'copied iPhone camera frame',
      size: [width, height, 1],
      format: 'rgba8unorm',
      // WebGPU requires RENDER_ATTACHMENT as well as COPY_DST for
      // copyExternalImageToTexture(), even though this code never renders to it.
      usage: GPUTextureUsage.COPY_DST |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.cameraTextureView = this.cameraTexture.createView();
    this.cameraTextureWidth = width;
    this.cameraTextureHeight = height;
  }

  private ensureStagingCanvas(width: number, height: number): HTMLCanvasElement {
    this.stagingCanvas ??= document.createElement('canvas');
    if (this.stagingCanvas.width !== width || this.stagingCanvas.height !== height) {
      this.stagingCanvas.width = width;
      this.stagingCanvas.height = height;
      this.stagingContext = null;
    }
    this.stagingContext ??= this.stagingCanvas.getContext('2d', { alpha: false });
    if (!this.stagingContext) throw new Error('Could not create the camera staging canvas.');
    return this.stagingCanvas;
  }

  private async readStats(readback: GPUBuffer, cpuProbeNonzero: boolean): Promise<LuminanceStats> {
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const allBins = new Uint32Array(readback.getMappedRange().slice(0));
      const bins = allBins.subarray(0, 256);
      const total = bins.reduce((sum, count) => sum + count, 0);
      const exactZeroCount = allBins[256] ?? 0;
      if (!total) {
        const empty = {
          mean: 0,
          p10: 0,
          p50: 0,
          p90: 0,
          blackFraction: 1,
          exactZeroFraction: 1,
        };
        this.observeInputHealth(empty, cpuProbeNonzero);
        return empty;
      }

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
      const stats = {
        mean: weighted / total,
        p10,
        p50,
        p90,
        blackFraction: black / total,
        exactZeroFraction: exactZeroCount / total,
      };
      this.observeInputHealth(stats, cpuProbeNonzero);
      return stats;
    } finally {
      if (readback.mapState === 'mapped') readback.unmap();
      readback.destroy();
    }
  }

  private observeInputHealth(stats: LuminanceStats, cpuProbeNonzero: boolean): void {
    if (!this.useCopiedCameraTexture) return;
    if (stats.exactZeroFraction < 0.999) {
      this.inputValidated = true;
      this.exactBlackProbes = 0;
      return;
    }
    this.exactBlackProbes += 1;
    if (!cpuProbeNonzero && this.exactBlackProbes < 2) return;
    const failedPath = this.cameraInputPath;
    if (!this.advanceCameraPath(
      `${failedPath} produced ${Math.round(stats.exactZeroFraction * 100)}% exact-zero pixels`)) {
      throw new Error(this.inputFailure!);
    }
  }
}
