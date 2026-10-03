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

const HUD_W = 680;
let win = null;
let desktopVisibility = null;

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
  win.on('closed', () => desktopVisibility.stop());
  win.loadFile(path.join(__dirname, 'renderer', 'index.html'));

  // Hide instead of destroy on close
  win.on('close', (e) => {
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
  globalShortcut.register('CommandOrControl+Q', () => app.exit(0));
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

app.on('will-quit', () => {
  desktopVisibility?.stop();
  globalShortcut.unregisterAll();
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
