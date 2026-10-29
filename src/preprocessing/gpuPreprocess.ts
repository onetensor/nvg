import preprocessShader from '../../shaders/preprocess.wgsl?raw';
import preprocessTextureShader from '../../shaders/preprocessTexture.wgsl?raw';
import {
  CALIBRATION_GRID,
  cameraInputPaths,
  estimateOrientation,
  orientationVerdict,
  sourceQuarterTurns,
} from '../cameraInput';
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
const BLACK_BINS = Math.ceil(Math.sqrt(4 / 255) * 255);
const zeroHistogram = new Uint32Array(HISTOGRAM_BINS);
const THUMBNAIL_CELLS = CALIBRATION_GRID * CALIBRATION_GRID;
const THUMBNAIL_BYTES = THUMBNAIL_CELLS * 3 * 4;
// getMappedRange() offsets must be multiples of 8.
const THUMBNAIL_OFFSET = Math.ceil(HISTOGRAM_BYTES / 8) * 8;
const PROBE_SIZE = 64;
// Give up on a conclusive orientation match after this many validation
// samples (a dark or featureless scene) and keep the metadata orientation.
const MAX_INCONCLUSIVE_CALIBRATIONS = 45;

interface CameraProbe {
  nonzero: boolean;
  /** Upright sRGB-encoded luma, CALIBRATION_GRID² cells. */
  luma: Float32Array;
}

const encodedLuma = (r: number, g: number, b: number): number => 0.2126 * r + 0.7152 * g + 0.0722 * b;

interface CameraResource {
  resource: GPUExternalTexture | GPUTextureView;
  /** Upright (display-oriented) source size used for cover sampling. */
  width: number;
  height: number;
  /** Clockwise quarter turns from the copied buffer to the upright image. */
  quarterTurns: number;
}

interface BufferLayout {
  width: number;
  height: number;
  quarterTurns: number;
}

type RotatableVideoFrame = VideoFrame & { rotation?: number };

function videoFrameLayout(frame: RotatableVideoFrame, video: HTMLVideoElement): BufferLayout {
  const width = frame.visibleRect?.width ?? frame.codedWidth;
  const height = frame.visibleRect?.height ?? frame.codedHeight;
  return {
    width,
    height,
    quarterTurns: sourceQuarterTurns(width, height, video.videoWidth, video.videoHeight, frame.rotation),
  };
}

export class GPUPreprocessor {
  private readonly pipelinePromise: Promise<GPUComputePipeline>;
  private readonly thumbnailPipelinePromise: Promise<GPUComputePipeline> | null;
  private readonly thumbnail: GPUBuffer | null;
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
  private readonly paths = cameraInputPaths();
  private pathIndex = 0;
  private videoCopyLayout: BufferLayout | null = null;
  private videoCopyLayoutKey = '';
  private inputValidated: boolean;
  private calibratedTurns: number | null = null;
  private inconclusiveCalibrations = 0;
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
    this.thumbnailPipelinePromise = useCopiedCameraTexture
      ? device.createComputePipelineAsync({
        label: 'camera orientation thumbnail',
        layout: 'auto',
        compute: { module, entryPoint: 'thumbnail' },
      })
      : null;
    this.thumbnail = useCopiedCameraTexture
      ? device.createBuffer({
        label: 'camera orientation thumbnail',
        size: THUMBNAIL_BYTES,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      })
      : null;
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
    return this.useCopiedCameraTexture ? this.paths[this.pathIndex]! : 'external-texture';
  }

  async initialize(): Promise<void> {
    await Promise.all([this.pipelinePromise, this.thumbnailPipelinePromise]);
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
    const channels = learned ? 3 : mode === 'direct' ? 12 : 4;
    const count = width * height * channels;
    this.ensureOutput(count * 4);

    const camera = await this.cameraResource(video, geometry);
    const collectStats = !this.statsInFlight &&
      (!this.inputValidated || this.frameIndex % HISTOGRAM_INTERVAL === 0);
    this.frameIndex += 1;
    // The CPU reference is taken in the same task as the copy, so both see the
    // same current video frame.
    const probe = collectStats && this.useCopiedCameraTexture && !this.inputValidated
      ? this.probeVideo(video)
      : null;
    const calibrate = probe !== null && this.cameraInputPath !== 'canvas-copy';
    if (collectStats) this.device.queue.writeBuffer(this.histogram, 0, zeroHistogram);

    const uvScale = coverUvScale(camera.width, camera.height, geometry.width, geometry.height);
    const values = new ArrayBuffer(48);
    const view = new DataView(values);
    view.setUint32(0, width, true);
    view.setUint32(4, height, true);
    view.setUint32(8, geometry.width, true);
    view.setUint32(12, geometry.height, true);
    view.setUint32(16, learned ? 0 : mode === 'direct' ? 2 : 1, true);
    view.setUint32(20, collectStats ? 1 : 0, true);
    view.setFloat32(24, gain, true);
    view.setUint32(28, camera.quarterTurns, true);
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

    if (calibrate) {
      const thumbnailPipeline = await this.thumbnailPipelinePromise!;
      const thumbnailPass = encoder.beginComputePass({ label: 'camera orientation thumbnail pass' });
      thumbnailPass.setPipeline(thumbnailPipeline);
      thumbnailPass.setBindGroup(0, this.device.createBindGroup({
        layout: thumbnailPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: camera.resource },
          { binding: 1, resource: this.sampler },
          { binding: 5, resource: { buffer: this.thumbnail! } },
        ],
      }));
      thumbnailPass.dispatchWorkgroups(1);
      thumbnailPass.end();
    }

    let readback: GPUBuffer | undefined;
    if (collectStats) {
      readback = this.device.createBuffer({
        label: 'luminance histogram readback',
        size: calibrate ? THUMBNAIL_OFFSET + THUMBNAIL_BYTES : HISTOGRAM_BYTES,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      encoder.copyBufferToBuffer(this.histogram, 0, readback, 0, HISTOGRAM_BYTES);
      if (calibrate) encoder.copyBufferToBuffer(this.thumbnail!, 0, readback, THUMBNAIL_OFFSET, THUMBNAIL_BYTES);
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
      ? this.readStats(readback, probe, calibrate).finally(() => { this.statsInFlight = false; })
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
    this.thumbnail?.destroy();
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
        quarterTurns: 0,
      };
    }

    while (this.pathIndex < this.paths.length) {
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
    if (!video.videoWidth || !video.videoHeight) throw new Error('The camera has no decoded frame yet.');
    let source: HTMLVideoElement | VideoFrame | HTMLCanvasElement = video;
    let frame: VideoFrame | null = null;
    let layout: BufferLayout;

    if (path === 'video-frame-copy') {
      if (typeof VideoFrame === 'undefined') throw new Error('VideoFrame is unavailable.');
      frame = new VideoFrame(video);
      source = frame;
      layout = videoFrameLayout(frame, video);
    } else if (path === 'canvas-copy') {
      // drawImage applies the camera track rotation, and the cover crop plus
      // downscale happen here so only the processing-size image is uploaded.
      const canvas = this.ensureStagingCanvas(geometry.width, geometry.height);
      const scale = coverUvScale(video.videoWidth, video.videoHeight, canvas.width, canvas.height);
      const cropWidth = video.videoWidth * scale.x;
      const cropHeight = video.videoHeight * scale.y;
      this.stagingContext!.drawImage(
        video,
        (video.videoWidth - cropWidth) / 2,
        (video.videoHeight - cropHeight) / 2,
        cropWidth,
        cropHeight,
        0,
        0,
        canvas.width,
        canvas.height,
      );
      source = canvas;
      layout = { width: canvas.width, height: canvas.height, quarterTurns: 0 };
    } else {
      layout = this.directVideoLayout(video);
    }
    if (path !== 'canvas-copy' && this.calibratedTurns !== null) {
      layout = { ...layout, quarterTurns: this.calibratedTurns };
    }

    if (layout.width <= 0 || layout.height <= 0) {
      frame?.close();
      throw new Error('The camera source has zero dimensions.');
    }
    const sourceWidth = layout.width;
    const sourceHeight = layout.height;
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

    const sideways = layout.quarterTurns % 2 === 1;
    return {
      resource: this.cameraTextureView!,
      width: sideways ? sourceHeight : sourceWidth,
      height: sideways ? sourceWidth : sourceHeight,
      quarterTurns: layout.quarterTurns,
    };
  }

  /** The <video> element exposes only upright sizes; a short-lived VideoFrame reveals the buffer. */
  private directVideoLayout(video: HTMLVideoElement): BufferLayout {
    const key = `${video.videoWidth}x${video.videoHeight}`;
    if (this.videoCopyLayout && this.videoCopyLayoutKey === key) return this.videoCopyLayout;
    let layout: BufferLayout = { width: video.videoWidth, height: video.videoHeight, quarterTurns: 0 };
    if (typeof VideoFrame !== 'undefined') {
      try {
        const frame = new VideoFrame(video);
        try {
          layout = videoFrameLayout(frame, video);
        } finally {
          frame.close();
        }
      } catch (error) {
        console.warn('Could not inspect the camera buffer layout.', error);
      }
    }
    this.videoCopyLayout = layout;
    this.videoCopyLayoutKey = key;
    return layout;
  }

  private advanceCameraPath(reason: string): boolean {
    console.warn(reason);
    this.pathIndex += 1;
    this.inputValidated = false;
    this.calibratedTurns = null;
    this.inconclusiveCalibrations = 0;
    this.exactBlackProbes = 0;
    this.frameIndex = 0;
    if (this.pathIndex < this.paths.length) {
      console.warn(`Retrying camera ingestion with ${this.cameraInputPath}.`);
      return true;
    }
    this.inputFailure = `Safari returned black pixels through every camera upload path (${reason}).`;
    return false;
  }

  private probeVideo(video: HTMLVideoElement): CameraProbe | null {
    this.probeCanvas ??= document.createElement('canvas');
    this.probeCanvas.width = PROBE_SIZE;
    this.probeCanvas.height = PROBE_SIZE;
    this.probeContext ??= this.probeCanvas.getContext('2d', { alpha: false, willReadFrequently: true });
    if (!this.probeContext) return null;
    try {
      // drawImage applies the camera track rotation, so this is the upright
      // reference for orientation calibration.
      this.probeContext.drawImage(video, 0, 0, PROBE_SIZE, PROBE_SIZE);
      const pixels = this.probeContext.getImageData(0, 0, PROBE_SIZE, PROBE_SIZE).data;
      const luma = new Float32Array(THUMBNAIL_CELLS);
      const block = PROBE_SIZE / CALIBRATION_GRID;
      let nonzero = false;
      for (let y = 0; y < PROBE_SIZE; y += 1) {
        for (let x = 0; x < PROBE_SIZE; x += 1) {
          const offset = (y * PROBE_SIZE + x) * 4;
          const r = pixels[offset]!;
          const g = pixels[offset + 1]!;
          const b = pixels[offset + 2]!;
          if (r || g || b) nonzero = true;
          const cell = Math.floor(y / block) * CALIBRATION_GRID + Math.floor(x / block);
          luma[cell]! += encodedLuma(r, g, b) / (255 * block * block);
        }
      }
      return { nonzero, luma };
    } catch (error) {
      console.warn('Camera CPU probe failed.', error);
      return null;
    }
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
      // The -srgb view makes sampling and bilinear filtering happen in linear
      // light, which the area-filtered downsampling relies on.
      format: 'rgba8unorm-srgb',
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

  private async readStats(
    readback: GPUBuffer,
    probe: CameraProbe | null,
    calibrate: boolean,
  ): Promise<LuminanceStats> {
    const pathIndex = this.pathIndex;
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const allBins = new Uint32Array(readback.getMappedRange(0, HISTOGRAM_BYTES).slice(0));
      let thumbnailLuma: Float32Array | null = null;
      if (calibrate) {
        const cells = new Float32Array(readback.getMappedRange(THUMBNAIL_OFFSET, THUMBNAIL_BYTES).slice(0));
        thumbnailLuma = new Float32Array(THUMBNAIL_CELLS);
        for (let cell = 0; cell < THUMBNAIL_CELLS; cell += 1) {
          thumbnailLuma[cell] = encodedLuma(cells[cell * 3]!, cells[cell * 3 + 1]!, cells[cell * 3 + 2]!);
        }
      }
      // A copy failure may have switched paths while this readback was queued.
      const observe = (stats: LuminanceStats): void => {
        if (pathIndex === this.pathIndex) this.observeInputHealth(stats, probe, thumbnailLuma);
      };
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
        observe(empty);
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
        // Bins are sqrt-spaced in linear luminance (see record_luminance).
        const luminance = ((index + 0.5) / 255) ** 2;
        weighted += count * luminance;
        cumulative += count;
        if (!found10 && cumulative >= total * 0.1) { p10 = luminance; found10 = true; }
        if (!found50 && cumulative >= total * 0.5) { p50 = luminance; found50 = true; }
        if (!found90 && cumulative >= total * 0.9) { p90 = luminance; found90 = true; }
      });
      // Near-black keeps its previous meaning: linear luminance below 4/255.
      let black = 0;
      for (let index = 0; index < BLACK_BINS; index += 1) black += bins[index] ?? 0;
      const stats = {
        mean: weighted / total,
        p10,
        p50,
        p90,
        blackFraction: black / total,
        exactZeroFraction: exactZeroCount / total,
      };
      observe(stats);
      return stats;
    } finally {
      if (readback.mapState === 'mapped') readback.unmap();
      readback.destroy();
    }
  }

  private observeInputHealth(
    stats: LuminanceStats,
    probe: CameraProbe | null,
    thumbnailLuma: Float32Array | null,
  ): void {
    if (!this.useCopiedCameraTexture || this.inputValidated) return;
    if (stats.exactZeroFraction < 0.999) {
      this.exactBlackProbes = 0;
      if (probe && thumbnailLuma && !this.calibrateOrientation(probe, thumbnailLuma)) return;
      this.inputValidated = true;
      return;
    }
    this.exactBlackProbes += 1;
    if (!probe?.nonzero && this.exactBlackProbes < 2) return;
    const failedPath = this.cameraInputPath;
    if (!this.advanceCameraPath(
      `${failedPath} produced ${Math.round(stats.exactZeroFraction * 100)}% exact-zero pixels`)) {
      throw new Error(this.inputFailure!);
    }
  }

  /** Returns true once the direct copy's orientation is settled. */
  private calibrateOrientation(probe: CameraProbe, thumbnailLuma: Float32Array): boolean {
    const path = this.cameraInputPath;
    const estimate = estimateOrientation(probe.luma, thumbnailLuma);
    const verdict = orientationVerdict(estimate);
    const detail = `r=${estimate.correlation.toFixed(2)}, margin=${estimate.margin.toFixed(2)}, ` +
      `contrast=${estimate.contrast.toFixed(3)}, missing=${(estimate.missing * 100).toFixed(0)}%`;
    if (verdict.kind === 'accept') {
      this.calibratedTurns = verdict.quarterTurns;
      console.info(`${path} orientation: ${verdict.quarterTurns * 90}° clockwise (${detail}).`);
      return true;
    }
    // A forced diagnostic path is never abandoned; use the best estimate.
    if (verdict.kind === 'reject' && this.paths.length > 1) {
      if (!this.advanceCameraPath(`${path} rejected: ${verdict.reason} (${detail})`)) {
        throw new Error(this.inputFailure!);
      }
      return false;
    }
    this.inconclusiveCalibrations += 1;
    if (verdict.kind === 'inconclusive' && this.inconclusiveCalibrations < MAX_INCONCLUSIVE_CALIBRATIONS) {
      return false;
    }
    if (verdict.kind === 'reject') this.calibratedTurns = estimate.quarterTurns;
    console.warn(`${path} orientation unconfirmed; using ${this.calibratedTurns === null
      ? 'buffer metadata'
      : `${this.calibratedTurns * 90}° clockwise`} (${detail}).`);
    return true;
  }
}
