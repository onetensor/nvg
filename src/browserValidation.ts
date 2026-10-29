import * as ort from 'onnxruntime-web/webgpu';

interface Fixture {
  name: string;
  inputUrl: string;
  outputUrl: string;
  inputShape: number[];
  outputShape: number[];
  stages: Array<{
    modelUrl: string;
    inputName: string;
    outputName: string;
    gainClamp?: number;
  }>;
}

const run = document.querySelector<HTMLButtonElement>('#run')!;
const result = document.querySelector<HTMLElement>('#result')!;
const fixtureSelect = document.querySelector<HTMLSelectElement>('#fixture')!;

async function floats(url: string): Promise<Float32Array> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  return new Float32Array(await response.arrayBuffer());
}

run.addEventListener('click', async () => {
  run.disabled = true;
  result.textContent = 'Loading fixture and model…';
  try {
    if (!navigator.gpu) throw new Error('navigator.gpu is unavailable');
    const manifestResponse = await fetch(`./validation/${fixtureSelect.value}/manifest.json`, { cache: 'no-store' });
    if (!manifestResponse.ok) throw new Error('Validation fixture is not packaged. Generate it with tools/export_browser_fixture.py.');
    const fixture = await manifestResponse.json() as Fixture;
    const [input, reference] = await Promise.all([floats(fixture.inputUrl), floats(fixture.outputUrl)]);
    const started = performance.now();
    let tensor: ort.Tensor = new ort.Tensor('float32', input, fixture.inputShape);
    for (const stage of fixture.stages) {
      const session = await ort.InferenceSession.create(stage.modelUrl, {
        executionProviders: ['webgpu'],
        graphOptimizationLevel: 'all',
      });
      const outputs = await session.run({ [stage.inputName]: tensor });
      tensor.dispose();
      tensor = outputs[stage.outputName]!;
      if (stage.gainClamp !== undefined) {
        const values = await tensor.getData() as Float32Array;
        const gained = new Float32Array(values.length);
        for (let index = 0; index < values.length; index += 1) {
          gained[index] = Math.min(1, Math.max(0, values[index]! * stage.gainClamp));
        }
        const dims = [...tensor.dims];
        tensor.dispose();
        tensor = new ort.Tensor('float32', gained, dims);
      }
      await session.release();
    }
    const elapsed = performance.now() - started;
    if (tensor.dims.length !== fixture.outputShape.length ||
        tensor.dims.some((value, index) => value !== fixture.outputShape[index])) {
      throw new Error(`Output shape ${tensor.dims.join('×')} != ${fixture.outputShape.join('×')}`);
    }
    const candidate = await tensor.getData() as Float32Array;
    if (candidate.length !== reference.length) throw new Error(`Output size ${candidate.length} != ${reference.length}`);
    let absolute = 0;
    let maximum = 0;
    let squared = 0;
    for (let index = 0; index < candidate.length; index += 1) {
      const difference = Math.abs(candidate[index]! - reference[index]!);
      absolute += difference;
      squared += difference * difference;
      maximum = Math.max(maximum, difference);
    }
    const mae = absolute / candidate.length;
    const mse = squared / candidate.length;
    result.textContent = JSON.stringify({
      provider: 'webgpu',
      fixture: fixture.name,
      elements: candidate.length,
      elapsedMs: elapsed,
      mae,
      maxError: maximum,
      psnr: -10 * Math.log10(Math.max(mse, 1e-16)),
    }, null, 2);
    tensor.dispose();
  } catch (error) {
    result.textContent = `FAILED\n${error instanceof Error ? error.stack ?? error.message : String(error)}`;
  } finally {
    run.disabled = false;
  }
});

const requestedFixture = new URLSearchParams(window.location.search).get('fixture');
if (requestedFixture && [...fixtureSelect.options].some((option) => option.value === requestedFixture)) {
  fixtureSelect.value = requestedFixture;
  setTimeout(() => run.click(), 0);
}
