/** learned: adapter + SID; manual: fixed pseudo-Bayer + SID; direct: 12-channel packed RGB SID. */
export type AdapterMode = 'manual' | 'learned' | 'direct';
export type ModelVariant = 'joint' | 'original' | 'direct';
export type ViewMode = 'original' | 'enhanced' | 'split';
export type ResolutionChoice = 'auto' | '180p' | '360p' | '540p' | '720p' | '1080p';
export type TemporalLevel = 'off' | 'low' | 'medium' | 'high' | 'hold';
export type TargetMode = '30' | '15' | 'quality';
export type CameraInputPath = 'external-texture' | 'video-copy' | 'video-frame-copy' | 'canvas-copy';

/** How model output is shown: 3 = RGB; 1 = luminance through a tint. */
export interface DisplayOptions {
  channels: 1 | 3;
  tint: MonoTint;
}

export type MonoTint = 'gray' | 'green';

export interface Resolution {
  label: Exclude<ResolutionChoice, 'auto'>;
  width: number;
  height: number;
}

export interface ModelGeometry extends Resolution {
  paddedWidth: number;
  paddedHeight: number;
  rawWidth: number;
  rawHeight: number;
  cropX: number;
  cropY: number;
}

export interface LuminanceStats {
  mean: number;
  p10: number;
  p50: number;
  p90: number;
  blackFraction: number;
  exactZeroFraction: number;
}

export interface OutputStats {
  min: number;
  mean: number;
  max: number;
  nonFinite: number;
}

export interface ModelAsset {
  available: boolean;
  url: string;
  sha256: string | null;
  inputName: string;
  outputName: string;
  inputLayout: 'NCHW';
  description: string;
  fp16?: {
    available: boolean;
    url: string;
    sha256: string | null;
    mae: number | null;
    maxError: number | null;
  };
  fixed360p?: {
    available: boolean;
    url: string;
    sha256: string | null;
    outputShape: number[];
    fp16?: {
      available: boolean;
      url: string;
      sha256: string | null;
      mae: number | null;
      maxError: number | null;
    };
  };
}

export interface ModelMetadata {
  schemaVersion: number;
  sid: ModelAsset;
  sidJoint?: ModelAsset;
  /** Adapter-free SID taking 2x2-packed linear RGB (12 channels). */
  sidDirect?: ModelAsset & { inputChannels: number; width: number; outputChannels?: 1 | 3 };
  /** Default variant; `?model=` overrides it for on-device comparison. */
  production?: ModelVariant;
  adapter: ModelAsset & { parameterCount: number };
  preprocessing: {
    transfer: 'srgb-inverse';
    bayerOrder: 'RGGB';
    adapterInput: 'linear-rgb';
    adapterOutput: 'unamplified-pseudo-raw';
  };
  validation: {
    verified: boolean;
    referenceSample: string | null;
    mae: number | null;
    maxError: number | null;
    psnr: number | null;
  };
}

export interface TimingSample {
  preprocess: number;
  adapter: number;
  sid: number;
  postprocess: number;
  total: number;
}

export interface PipelineResult {
  timing: TimingSample;
  /** Resolves once the GPU reaches this frame; never awaited per frame. */
  stats?: Promise<LuminanceStats>;
  outputStats?: OutputStats;
  cameraInputPath: CameraInputPath;
  presentation?: PresentationStats;
  gain: number;
}

export interface PresentationStats {
  displayedFps: number;
  readbackMs: number;
  dropped: number;
}

export interface DebugSnapshot {
  cameraWidth: number;
  cameraHeight: number;
  processing: ModelGeometry;
  cameraFps: number;
  processedFps: number;
  timings: TimingSample;
  backend: string;
  webgpu: boolean;
  fp16: boolean;
  adapterMode: AdapterMode;
  modelVariant: ModelVariant;
  gain: number;
  automaticGain: number;
  cameraInputPath?: CameraInputPath;
  inputStats?: LuminanceStats;
  outputStats?: OutputStats;
  presentation?: PresentationStats;
  framesReceived: number;
  framesProcessed: number;
  framesSkipped: number;
}
