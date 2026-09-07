// Render the actual HUD against a fixed resolver response, without connecting
// to the user's backend or executing a command. Run with the project's Electron.
const { app, BrowserWindow, session } = require('electron');
const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const output = path.resolve(process.argv[2] || path.join(root, 'docs/results/2026-09-07-hud-correction.png'));
app.setPath('userData', path.join(root, 'data/showcase-electron'));
app.disableHardwareAcceleration();

app.whenReady().then(async () => {
  const python = path.join(root, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
  const result = spawnSync(python, ['-c',
    "import json; from eval.command_resolution import fixture_resolver; print(json.dumps(fixture_resolver().resolve('pyhton train.py --lr 0.001').to_dict()))"],
    { cwd: root, encoding: 'utf8', windowsHide: true });
  if (result.status !== 0) throw new Error(result.stderr);
  const resolution = { ...JSON.parse(result.stdout), type: 'resolution', token: 'showcase-only', revision: 1, client_revision: 1 };
  // Block WebSockets before loading the renderer; no live backend is contacted.
  session.defaultSession.webRequest.onBeforeRequest((details, callback) => {
    callback({ cancel: /^wss?:/.test(details.url) });
  });
  const win = new BrowserWindow({ width: 680, height: 260, show: false,
    webPreferences: { nodeIntegration: true, contextIsolation: false, offscreen: true } });
  await win.loadFile(path.join(root, 'ui/renderer/index.html'));
  await win.webContents.executeJavaScript(`
    document.body.style.background = '#0b1018';
    document.body.style.padding = '32px';
    document.body.style.boxSizing = 'border-box';
    cmdInput.value = ${JSON.stringify(resolution.original)};
    inputRevision = 1;
    onStatus({safe_mode:true, tasks_count:0});
    onResolution(${JSON.stringify(resolution)});
    document.getElementById('hud').style.width = '100%';
    new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  `);
  const screenshot = await win.webContents.capturePage();
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, screenshot.toPNG());
  process.stdout.write(output + '\n');
  win.destroy();
  app.quit();
}).catch(error => { process.stderr.write(String(error.stack || error) + '\n'); app.exit(1); });
