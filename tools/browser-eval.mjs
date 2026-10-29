const endpoint = process.argv[2] ?? 'http://127.0.0.1:9222';
const pagePattern = process.argv[3] ?? 'benchmark.html';
const expression = process.argv[4] ?? 'document.title';
const pages = await fetch(`${endpoint}/json`).then((response) => response.json());
const target = pages.find((page) => page.type === 'page' && page.url.includes(pagePattern));
if (!target) throw new Error(`No Chrome page matching ${pagePattern}`);
const socket = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  socket.addEventListener('open', resolve, { once: true });
  socket.addEventListener('error', reject, { once: true });
});
const response = await new Promise((resolve) => {
  socket.addEventListener('message', (event) => {
    const message = JSON.parse(event.data);
    if (message.id === 1) resolve(message);
  });
  socket.send(JSON.stringify({
    id: 1,
    method: 'Runtime.evaluate',
    params: { expression, awaitPromise: true, returnByValue: true },
  }));
});
socket.close();
console.log(JSON.stringify(response.result?.result?.value ?? response, null, 2));
