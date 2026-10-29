import * as ort from 'onnxruntime-web';
import type { ModelAsset } from './types';

const runButton = document.querySelector<HTMLButtonElement>('#run')!;
const result = document.querySelector<HTMLElement>('#result')!;

function modelUrl(url: string): string {
  if (url.startsWith('./')) return `${import.meta.env.BASE_URL}${url.slice(2)}`;
  return url;
}

async function benchmark(
  name: string,
  asset: ModelAsset,
  inputShape: number[],
  outputShape: number[],
  useFp16: boolean,
  iterations: number,
): Promise<Record<string, unknown>> {
  const url = useFp16 && asset.fp16?.available ? asset.fp16.url : asset.url;
  const session = await ort.InferenceSession.create(modelUrl(url), {
    executionProviders: ['webgpu'],
    graphOptimizationLevel: 'all',
    enableGraphCapture: false,
    preferredOutputLocation: { [asset.outputName]: 'gpu-buffer' },
  });
  const device = await ort.env.webgpu.device;
  if (!device) throw new Error('ONNX Runtime did not expose its WebGPU device.');
  const elements = (shape: number[]): number => shape.reduce((product, value) => product * value, 1);
  const inputBuffer = device.createBuffer({
    size: Math.ceil(elements(inputShape) * 4 / 16) * 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  const outputBuffer = device.createBuffer({
    size: Math.ceil(elements(outputShape) * 4 / 16) * 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  const input = ort.Tensor.fromGpuBuffer(inputBuffer, { dataType: 'float32', dims: inputShape });
  const output = ort.Tensor.fromGpuBuffer(outputBuffer, { dataType: 'float32', dims: outputShape });
  const feeds = { [asset.inputName]: input };
  const fetches = { [asset.outputName]: output };
  await session.run(feeds, fetches);
  await device.queue.onSubmittedWorkDone();
  const started = performance.now();
  for (let index = 0; index < iterations; index += 1) await session.run(feeds, fetches);
  await device.queue.onSubmittedWorkDone();
  const elapsed = performance.now() - started;
  await session.release();
  inputBuffer.destroy();
  outputBuffer.destroy();
  return { name, iterations, totalMs: elapsed, averageMs: elapsed / iterations, fps: 1000 * iterations / elapsed };
}

async function run(): Promise<void> {
  runButton.disabled = true;
  result.textContent = 'Loading fixed-shape models…';
  try {
    if (!navigator.gpu) throw new Error('navigator.gpu is unavailable');
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    ort.env.wasm.initTimeout = 20_000;
    const response = await fetch(`${import.meta.env.BASE_URL}models/metadata.json`, { cache: 'no-store' });
    if (!response.ok) throw new Error(`metadata: ${response.status}`);
    const metadata = await response.json() as { sidJoint: ModelAsset; adapter: ModelAsset };
    const gpuAdapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!gpuAdapter?.features.has('shader-f16')) {
      throw new Error('This production benchmark is FP16-only, but shader-f16 is unavailable.');
    }
    const useFp16 = true;
    const iterations = 12;
    const adapter = await benchmark(
      'adapter', metadata.adapter, [1, 3, 192, 320], [1, 4, 96, 160], useFp16, iterations,
    );
    const sid = await benchmark(
      'joint SID', metadata.sidJoint, [1, 4, 96, 160], [1, 3, 192, 320], useFp16, iterations,
    );
    result.textContent = JSON.stringify({ provider: 'webgpu', graphCapture: false, useFp16, adapter, sid }, null, 2);
  } catch (error) {
    result.textContent = `FAILED\n${error instanceof Error ? error.stack ?? error.message : String(error)}`;
  } finally {
    runButton.disabled = false;
  }
}

runButton.addEventListener('click', () => { void run(); });
if (new URLSearchParams(window.location.search).has('autorun')) setTimeout(() => runButton.click(), 0);
