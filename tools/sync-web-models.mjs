import { cp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const source = resolve(root, 'models');
const destination = resolve(root, 'public', 'models');
const metadata = JSON.parse(await readFile(resolve(source, 'metadata.json'), 'utf8'));

const selectedSid = metadata.sidJoint?.available ? metadata.sidJoint : metadata.sid;
if (!selectedSid?.fixed360p?.available || !metadata.adapter?.fixed360p?.available) {
  throw new Error('The production site requires fixed 360p joint SID and adapter graphs.');
}

function fixedOnly(asset) {
  const fixed = asset.fixed360p;
  if (!fixed.fp16?.available) throw new Error(`FP16 fixed graph is unavailable for ${asset.description}.`);
  const fp16 = fixed.fp16;
  return {
    ...asset,
    url: fp16.url,
    sha256: fp16.sha256,
    fp16,
    fixed360p: {
      ...fixed,
      url: fp16.url,
      sha256: fp16.sha256,
      fp16,
    },
  };
}

const deployedSid = fixedOnly(selectedSid);
const deployedAdapter = fixedOnly(metadata.adapter);
const deployedMetadata = {
  ...metadata,
  // Both keys deliberately resolve to the selected joint model. The original
  // checkpoint remains in models/ for reproducibility but is not deployed.
  sid: deployedSid,
  sidJoint: deployedSid,
  adapter: deployedAdapter,
};

await rm(destination, { recursive: true, force: true });
await mkdir(destination, { recursive: true });
await writeFile(resolve(destination, 'metadata.json'), `${JSON.stringify(deployedMetadata, null, 2)}\n`);

const deployedUrls = new Set();
for (const asset of [deployedSid, deployedAdapter]) {
  deployedUrls.add(asset.url);
  if (asset.fp16?.available) deployedUrls.add(asset.fp16.url);
}

for (const url of deployedUrls) {
  const name = new URL(url, 'https://local.invalid/').pathname.split('/').pop();
  if (!name) throw new Error(`Invalid model URL: ${url}`);
  await cp(resolve(source, name), resolve(destination, name));
}
