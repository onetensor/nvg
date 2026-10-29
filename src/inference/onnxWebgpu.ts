import * as ort from 'onnxruntime-web/webgpu';
import type { BackendOptions, InferenceBackend, InferenceTensor } from './backend';

export class ONNXWebGPUBackend implements InferenceBackend {
  get name(): string {
    const precision = this.options.asset.url.includes('fp16') ? 'FP16 weights' : 'FP32';
    return `ONNX Runtime Web / WebGPU (${precision}${this.options.graphCapture ? ', graph capture' : ''})`;
  }
  private session: ort.InferenceSession | null = null;
  private outputBuffer: GPUBuffer | null = null;
  private outputTensor: ort.Tensor | null = null;
  private inputBuffer: GPUBuffer | null = null;
  private inputDims = '';
  private inputTensor: ort.Tensor | null = null;

  constructor(private readonly options: BackendOptions) {}

  async initialize(): Promise<void> {
    if (!this.options.asset.available) throw new Error(`${this.options.label} model is not packaged.`);
    if (!navigator.gpu) throw new Error('WebGPU is unavailable in this browser.');

    ort.env.logLevel = 'warning';
    ort.env.webgpu.powerPreference = 'high-performance';
    this.session = await ort.InferenceSession.create(this.options.asset.url, {
      executionProviders: ['webgpu'],
      graphOptimizationLevel: 'all',
      enableGraphCapture: this.options.graphCapture ?? false,
      preferredOutputLocation: { [this.options.asset.outputName]: 'gpu-buffer' },
    });

    if (!this.session.inputNames.includes(this.options.asset.inputName)) {
      throw new Error(`${this.options.label} input '${this.options.asset.inputName}' was not found.`);
    }
    if (!this.session.outputNames.includes(this.options.asset.outputName)) {
      throw new Error(`${this.options.label} output '${this.options.asset.outputName}' was not found.`);
    }
    if (this.options.outputShape) {
      const device = await this.getDevice();
      const elements = this.options.outputShape.reduce((product, value) => product * value, 1);
      this.outputBuffer = device.createBuffer({
        label: `${this.options.label} preallocated output`,
        size: Math.ceil(elements * 4 / 16) * 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      this.outputTensor = ort.Tensor.fromGpuBuffer(this.outputBuffer, {
        dataType: 'float32',
        dims: this.options.outputShape,
      });
    }
  }

  async run(input: InferenceTensor): Promise<InferenceTensor> {
    if (!this.session) throw new Error(`${this.options.label} backend has not initialized.`);
    const fetches = this.outputTensor ? { [this.options.asset.outputName]: this.outputTensor } : undefined;
    const results = await this.session.run({ [this.options.asset.inputName]: input }, fetches);
    const output = results[this.options.asset.outputName];
    if (!(output instanceof ort.Tensor)) throw new Error(`${this.options.label} returned no tensor output.`);
    if (output.location !== 'gpu-buffer') throw new Error(`${this.options.label} output left GPU memory.`);
    return output;
  }

  tensorFromGpuBuffer(buffer: GPUBuffer, dims: readonly number[]): InferenceTensor {
    const key = dims.join('x');
    if (this.inputTensor && this.inputBuffer === buffer && this.inputDims === key) return this.inputTensor;
    this.inputBuffer = buffer;
    this.inputDims = key;
    this.inputTensor = ort.Tensor.fromGpuBuffer(buffer, { dataType: 'float32', dims });
    return this.inputTensor;
  }

  async getDevice(): Promise<GPUDevice> {
    const device = ort.env.webgpu.device;
    if (!device) throw new Error('ONNX Runtime did not expose its WebGPU device.');
    return await device;
  }

  releaseOutput(tensor: InferenceTensor): void {
    if (tensor !== this.outputTensor) tensor.dispose();
  }

  async dispose(): Promise<void> {
    await this.session?.release();
    this.session = null;
    this.inputTensor = null;
    this.inputBuffer = null;
    this.outputTensor = null;
    this.outputBuffer?.destroy();
    this.outputBuffer = null;
  }
}
