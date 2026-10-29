const endpoint = process.argv[2] ?? 'http://127.0.0.1:9222';
const pagePattern = process.argv[3] ?? 'benchmark.html';
const selector = process.argv[4] ?? '#result';
const timeoutMs = Number(process.argv[5] ?? 120000);
const started = Date.now();

let target;
while (!target && Date.now() - started < timeoutMs) {
  const pages = await fetch(`${endpoint}/json`).then((response) => response.json()).catch(() => []);
  target = pages.find((page) => page.type === 'page' && page.url.includes(pagePattern));
  if (!target) await new Promise((resolve) => setTimeout(resolve, 200));
}
if (!target) throw new Error(`No Chrome page matching ${pagePattern}`);

const socket = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  socket.addEventListener('open', resolve, { once: true });
  socket.addEventListener('error', reject, { once: true });
});
let id = 0;
const pending = new Map();
socket.addEventListener('message', (event) => {
  const message = JSON.parse(event.data);
  const callback = pending.get(message.id);
  if (callback) {
    pending.delete(message.id);
    callback(message);
  }
});

function evaluate(expression) {
  const requestId = ++id;
  socket.send(JSON.stringify({ id: requestId, method: 'Runtime.evaluate', params: { expression, returnByValue: true } }));
  return new Promise((resolve) => pending.set(requestId, resolve));
}

let value = '';
while (Date.now() - started < timeoutMs) {
  const response = await evaluate(`document.querySelector(${JSON.stringify(selector)})?.textContent ?? ''`);
  value = response.result?.result?.value ?? '';
  if (value && value !== 'Waiting.' && !value.startsWith('Loading ')) break;
  await new Promise((resolve) => setTimeout(resolve, 250));
}
socket.close();
if (!value || value === 'Waiting.' || value.startsWith('Loading ')) throw new Error(`Timed out waiting for ${selector}: ${value}`);
console.log(value);
