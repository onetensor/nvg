import * as ort from 'onnxruntime-web';
import type { BackendOptions, InferenceBackend, InferenceTensor } from './backend';

const modelDownloads = new Map<string, Promise<Uint8Array>>();

export function preloadModel(url: string): Promise<Uint8Array> {
  const existing = modelDownloads.get(url);
  if (existing) return existing;
  const download = fetch(url, { cache: 'force-cache' })
    .then(async (response) => {
      if (!response.ok) throw new Error(`Could not download ${url} (${response.status}).`);
      return new Uint8Array(await response.arrayBuffer());
    })
    .catch((error) => {
      modelDownloads.delete(url);
      throw error;
    });
  modelDownloads.set(url, download);
  return download;
}

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
  private validateNextRun = true;

  constructor(private readonly options: BackendOptions) {}

  async initialize(): Promise<void> {
    if (!this.options.asset.available) throw new Error(`${this.options.label} model is not packaged.`);
    if (!navigator.gpu) throw new Error('WebGPU is unavailable in this browser.');

    ort.env.logLevel = 'warning';
    // GitHub Pages cannot provide COOP/COEP, and Safari's WebGPU/WASM bridge is
    // substantially more reliable without a worker pool during initialization.
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    ort.env.wasm.initTimeout = 20_000;
    ort.env.webgpu.powerPreference = 'high-performance';
    const model = await preloadModel(this.options.asset.url);
    try {
      this.session = await ort.InferenceSession.create(model, {
        executionProviders: ['webgpu'],
        graphOptimizationLevel: 'all',
        enableGraphCapture: this.options.graphCapture ?? false,
        preferredOutputLocation: { [this.options.asset.outputName]: 'gpu-buffer' },
      });
    } finally {
      // Session creation copies the graph into ORT-owned memory. Keep no second
      // 16 MB JavaScript copy once initialization has finished.
      modelDownloads.delete(this.options.asset.url);
    }

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
    const device = this.validateNextRun ? await this.getDevice() : null;
    device?.pushErrorScope('validation');
    let results: ort.InferenceSession.OnnxValueMapType;
    try {
      results = await this.session.run({ [this.options.asset.inputName]: input }, fetches);
      if (device) await device.queue.onSubmittedWorkDone();
    } finally {
      if (device) {
        const gpuError = await device.popErrorScope();
        this.validateNextRun = false;
        if (gpuError) throw new Error(`${this.options.label} WebGPU validation failed: ${gpuError.message}`);
      }
    }
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
    this.validateNextRun = true;
  }
}
