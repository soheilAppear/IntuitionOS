const { app, BrowserWindow, globalShortcut, ipcMain, screen } = require('electron');
const path = require('path');

const HUD_W = 680;
let win = null;

function createWindow() {
  const { width } = screen.getPrimaryDisplay().workAreaSize;

  win = new BrowserWindow({
    width: HUD_W,
    height: 64,
    x: Math.floor((width - HUD_W) / 2),
    y: 28,
    frame: false,
    transparent: true,
    backgroundColor: '#00000000',
    alwaysOnTop: true,
    skipTaskbar: true,
    resizable: false,
    movable: true,
    hasShadow: true,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  // Windows 11 acrylic blur — ignore if unavailable
  try { win.setBackgroundMaterial('acrylic'); } catch (_) {}

  win.loadFile(path.join(__dirname, 'renderer', 'index.html'));

  // Hide instead of destroy on close
  win.on('close', (e) => {
    e.preventDefault();
    win.hide();
  });
}

function toggleWindow() {
  if (!win) return;
  if (win.isVisible()) {
    win.hide();
  } else {
    win.show();
    win.focus();
    win.webContents.send('focus-input');
  }
}

app.whenReady().then(() => {
  createWindow();
  globalShortcut.register('Alt+Space', toggleWindow);
  globalShortcut.register('CommandOrControl+Q', () => app.exit(0));
});

app.on('will-quit', () => globalShortcut.unregisterAll());

// Keep process alive when the HUD window is hidden
app.on('window-all-closed', (e) => e.preventDefault());

// Renderer × button → hide window
ipcMain.on('hide-window', () => { if (win) win.hide(); });

// Renderer → resize native window height
ipcMain.on('resize', (event, height) => {
  if (!win) return;
  const { height: maxH } = screen.getPrimaryDisplay().workAreaSize;
  const h = Math.max(64, Math.min(height, Math.floor(maxH * 0.72)));
  win.setSize(HUD_W, h, false);
});
