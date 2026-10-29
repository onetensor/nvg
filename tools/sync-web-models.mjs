import { cp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const source = resolve(root, 'models');
const destination = resolve(root, 'public', 'models');
const metadata = JSON.parse(await readFile(resolve(source, 'metadata.json'), 'utf8'));

await rm(destination, { recursive: true, force: true });
await mkdir(destination, { recursive: true });
await writeFile(resolve(destination, 'metadata.json'), `${JSON.stringify(metadata, null, 2)}\n`);

for (const asset of [metadata.sid, metadata.sidJoint, metadata.adapter].filter(Boolean)) {
  if (!asset.available) continue;
  const name = new URL(asset.url, 'https://local.invalid/').pathname.split('/').pop();
  if (!name) throw new Error(`Invalid model URL: ${asset.url}`);
  await cp(resolve(source, name), resolve(destination, name));
  if (asset.fp16?.available) {
    const fp16Name = new URL(asset.fp16.url, 'https://local.invalid/').pathname.split('/').pop();
    if (!fp16Name) throw new Error(`Invalid FP16 model URL: ${asset.fp16.url}`);
    await cp(resolve(source, fp16Name), resolve(destination, fp16Name));
  }
  if (asset.fixed360p?.available) {
    const fixedName = new URL(asset.fixed360p.url, 'https://local.invalid/').pathname.split('/').pop();
    if (!fixedName) throw new Error(`Invalid fixed 360p model URL: ${asset.fixed360p.url}`);
    await cp(resolve(source, fixedName), resolve(destination, fixedName));
    if (asset.fixed360p.fp16?.available) {
      const fixedFp16Name = new URL(asset.fixed360p.fp16.url, 'https://local.invalid/').pathname.split('/').pop();
      if (!fixedFp16Name) throw new Error(`Invalid fixed 360p FP16 model URL: ${asset.fixed360p.fp16.url}`);
      await cp(resolve(source, fixedFp16Name), resolve(destination, fixedFp16Name));
    }
  }
}
