/**
 * Electron's native HUD shell.
 *
 * This process owns one window, global shortcuts, and height/visibility IPC.
 * The renderer owns its content and communicates with the local backend, which
 * owns command execution. Hiding the HUD preserves the existing renderer;
 * CommandOrControl+Q exits the process and releases global shortcuts.
 */

const { app, BrowserWindow, globalShortcut, ipcMain, screen } = require('electron');
const path = require('path');
const { DesktopVisibility } = require('./desktop-visibility');
const { EegGuard } = require('./eeg-guard.cjs');
const { requestMultimodal } = require('./renderer/multimodal-http.cjs');

const HUD_W = 680;
let win = null;
let desktopVisibility = null;
let eegGuard = null;
let quitting = false;
let quitReady = false;

/** Create the single HUD on the primary display and retain it when closed. */
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
    // A normal application view allows Windows to pin this exact window.
    skipTaskbar: false,
    title: 'IntuitionOS',
    resizable: false,
    movable: true,
    hasShadow: true,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });
  eegGuard = new EegGuard({
    globalShortcut,
    request: requestMultimodal,
    notify: reason => {
      if (win && !win.isDestroyed() && !win.webContents.isDestroyed()) {
        win.webContents.send('eeg-emergency-stop', { reason });
      }
    },
  });
  win.webContents.on('render-process-gone', () => { void eegGuard.emergencyStop('renderer-crash'); });
  win.webContents.on('destroyed', () => { void eegGuard.emergencyStop('renderer-destroyed'); });

  // Windows 11 acrylic blur — ignore if unavailable
  try {
    win.setBackgroundMaterial('acrylic');
  } catch (_) {}

  const projectRoot = path.resolve(__dirname, '..');
  desktopVisibility = new DesktopVisibility({
    window: win, projectRoot,
    python: process.env.INTUITION_PYTHON || path.join(projectRoot, '.venv', 'Scripts', 'python.exe'),
    onStatus: status => { if (!win.isDestroyed()) win.webContents.send('desktop-visibility', status); },
  });
  win.once('ready-to-show', () => desktopVisibility.start());
  win.webContents.on('did-finish-load', () => {
    desktopVisibility.start();
    win.webContents.send('desktop-visibility', desktopVisibility.status);
  });
  win.on('closed', () => {
    desktopVisibility.stop();
    void eegGuard.emergencyStop('window-destroyed');
  });
  win.loadFile(path.join(__dirname, 'renderer', 'index.html'));

  // Hide instead of destroy on close
  win.on('close', (e) => {
    if (quitting) return;
    e.preventDefault();
    win.hide();
  });
}

/** Toggle native visibility and return keyboard focus to the existing draft. */
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
  globalShortcut.register('CommandOrControl+Q', () => app.quit());
  globalShortcut.register('Alt+V', () => {
    if (win) {
      if (!win.isVisible()) {
        win.show();
        win.focus();
      }
      win.webContents.send('voice-toggle');
    }
  });
});

// Give the bounded emergency request a chance to reach the backend before the
// process exits. Merely hiding the HUD leaves the global Escape lease intact.
app.on('before-quit', event => {
  if (quitReady) return;
  event.preventDefault();
  if (quitting) return;
  quitting = true;
  desktopVisibility?.stop();
  Promise.resolve(eegGuard?.dispose()).finally(() => {
    quitReady = true;
    app.quit();
  });
});

app.on('will-quit', () => {
  desktopVisibility?.stop();
  globalShortcut.unregisterAll();
});

ipcMain.handle('eeg-guard-start', async event => {
  if (!win || win.isDestroyed() || event.sender !== win.webContents || quitting) {
    return { ok: false, error: 'EEG guard is restricted to the active HUD' };
  }
  return eegGuard.start();
});
ipcMain.handle('eeg-guard-stop', async (event, payload = {}) => {
  if (!win || win.isDestroyed() || event.sender !== win.webContents) {
    return { ok: false, error: 'EEG guard is restricted to the active HUD' };
  }
  if (!payload || typeof payload !== 'object' || Array.isArray(payload) ||
      Object.keys(payload).some(key => key !== 'token') ||
      ('token' in payload && (typeof payload.token !== 'string' ||
        !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(payload.token)))) {
    return { ok: false, error: 'Invalid EEG guard cleanup token' };
  }
  return eegGuard.stop(payload.token);
});

// Keep process alive when the HUD window is hidden
app.on('window-all-closed', (e) => e.preventDefault());

// Renderer × button → hide window
ipcMain.on('hide-window', () => {
  if (win) win.hide();
});

// Camera activation returns focus to the application underneath the HUD.
// Feedback must never steal focus from the exact window selected for closing.
ipcMain.on('gesture-release-focus', () => {
  if (!win || !win.isFocused()) return;
  win.hide();
  setTimeout(() => { if (win && !win.isDestroyed()) win.showInactive(); }, 80);
});
ipcMain.on('gesture-attention', () => {
  if (win && !win.isDestroyed()) win.showInactive();
});
ipcMain.on('desktop-visibility-retry', event => {
  if (win && event.sender === win.webContents) desktopVisibility?.refresh();
});

// The renderer requests its content height; the native shell enforces bounds.
ipcMain.on('resize', (event, height) => {
  if (!win) return;
  const { height: maxH } = screen.getPrimaryDisplay().workAreaSize;
  const h = Math.max(64, Math.min(height, Math.floor(maxH * 0.72)));
  win.setSize(HUD_W, h, false);
});
