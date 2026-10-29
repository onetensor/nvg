import type * as ort from 'onnxruntime-web/webgpu';
import type { ModelAsset } from '../types';

export type InferenceTensor = ort.Tensor;

export interface InferenceBackend {
  readonly name: string;
  initialize(): Promise<void>;
  run(input: InferenceTensor): Promise<InferenceTensor>;
  tensorFromGpuBuffer(buffer: GPUBuffer, dims: readonly number[]): InferenceTensor;
  getDevice(): Promise<GPUDevice>;
  releaseOutput(tensor: InferenceTensor): void;
  dispose(): Promise<void>;
}

export interface BackendOptions {
  asset: ModelAsset;
  label: string;
  graphCapture?: boolean;
  outputShape?: readonly number[];
}
