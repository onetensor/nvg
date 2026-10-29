export type AdapterMode = 'manual' | 'learned';
export type ModelVariant = 'joint' | 'original';
export type ViewMode = 'original' | 'enhanced' | 'split';
export type ResolutionChoice = 'auto' | '180p' | '360p' | '540p' | '720p' | '1080p';
export type TemporalLevel = 'off' | 'low' | 'medium' | 'high';
export type TargetMode = '30' | '15' | 'quality';

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
  stats?: LuminanceStats;
  outputStats?: OutputStats;
  gain: number;
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
  outputStats?: OutputStats;
  framesReceived: number;
  framesProcessed: number;
  framesSkipped: number;
}
