import { cp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const source = resolve(root, 'models');
const destination = resolve(root, 'public', 'models');
const metadata = JSON.parse(await readFile(resolve(source, 'metadata.json'), 'utf8'));

const selectedSid = metadata.sidJoint?.available ? metadata.sidJoint : metadata.sid;
if (!selectedSid?.fp16?.available || !metadata.adapter?.fp16?.available) {
  throw new Error('The production site requires dynamic FP16 joint SID and adapter graphs.');
}

function fp16Only(asset) {
  const fp16 = asset.fp16;
  const { fixed360p: _fixed360p, ...dynamic } = asset;
  return {
    ...dynamic,
    url: fp16.url,
    sha256: fp16.sha256,
    fp16,
  };
}

const deployedSid = fp16Only(selectedSid);
const deployedAdapter = fp16Only(metadata.adapter);
// The direct model ships beside the joint pair so ?model= can compare them on
// a phone; each page load downloads only the variant it runs.
const deployedDirect = metadata.sidDirect?.fp16?.available ? fp16Only(metadata.sidDirect) : undefined;
if (metadata.production === 'direct' && !deployedDirect) {
  throw new Error('metadata.production is "direct" but no FP16 direct model is packaged.');
}
const { sidDirect: _sidDirect, ...withoutDirect } = metadata;
const deployedMetadata = {
  ...withoutDirect,
  // Both keys deliberately resolve to the selected joint model. The original
  // checkpoint remains in models/ for reproducibility but is not deployed.
  sid: deployedSid,
  sidJoint: deployedSid,
  adapter: deployedAdapter,
  ...(deployedDirect ? { sidDirect: deployedDirect } : {}),
};

await rm(destination, { recursive: true, force: true });
await mkdir(destination, { recursive: true });
await writeFile(resolve(destination, 'metadata.json'), `${JSON.stringify(deployedMetadata, null, 2)}\n`);

const deployedUrls = new Set();
for (const asset of [deployedSid, deployedAdapter, deployedDirect].filter(Boolean)) {
  deployedUrls.add(asset.url);
  if (asset.fp16?.available) deployedUrls.add(asset.fp16.url);
}

for (const url of deployedUrls) {
  const name = new URL(url, 'https://local.invalid/').pathname.split('/').pop();
  if (!name) throw new Error(`Invalid model URL: ${url}`);
  await cp(resolve(source, name), resolve(destination, name));
}
