/** Keep this HUD on every workspace without moving focus or changing desktops.
 * Windows pinning runs in a small isolated Python process because Electron's
 * all-workspaces API is a no-op on Windows. A shell restart can lose the pin,
 * so periodically verify it and retry failures without blocking the UI.
 */
const { execFile } = require('node:child_process');

function nativeHandle(window) {
  const buffer = window.getNativeWindowHandle();
  return buffer.length >= 8 ? buffer.readBigUInt64LE(0).toString() : String(buffer.readUInt32LE(0));
}

class DesktopVisibility {
  constructor({ window, python, projectRoot, onStatus, platform = process.platform,
    pid = process.pid, run = execFile, later = setTimeout, cancel = clearTimeout }) {
    Object.assign(this, { window, python, projectRoot, onStatus, platform, pid, run, later, cancel });
    this.stopped = true;
    this.child = null;
    this.timer = null;
    this.busy = false;
    this.failures = 0;
    this.generation = 0;
    this.status = { state: 'checking', pinned: false, text: 'Preparing the HUD for all desktops…' };
  }

  publish(status) {
    const changed = JSON.stringify(status) !== JSON.stringify(this.status);
    this.status = status;
    if (changed) this.onStatus(status);
  }

  start() {
    if (!this.stopped) return;
    this.stopped = false;
    this.generation += 1;
    this.onStatus(this.status);
    this.refresh();
  }

  refresh() {
    if (this.stopped || this.window.isDestroyed() || this.busy) return;
    if (this.timer !== null) this.cancel(this.timer);
    this.timer = null;
    if (this.platform !== 'win32') {
      try {
        this.window.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });
        if (!this.window.isVisibleOnAllWorkspaces()) throw new Error('Workspace visibility was not acknowledged');
        this.publish({ state: 'pinned', pinned: true, text: 'HUD visible on all virtual desktops.' });
      } catch (_) {
        this.publish({ state: 'unavailable', pinned: false, text: 'Could not show the HUD on all desktops. Click Retry.' });
      }
      return;
    }
    let handle;
    try { handle = nativeHandle(this.window); }
    catch (_) { this.failed(); return; }
    this.busy = true;
    const generation = this.generation;
    try {
      this.child = this.run(this.python,
        ['-m', 'core.hud_desktops', '--hwnd', handle, '--pid', String(this.pid)],
        { cwd: this.projectRoot, windowsHide: true, timeout: 6000, maxBuffer: 16384 },
        (error, stdout) => {
          if (this.stopped || generation !== this.generation || this.window.isDestroyed()) return;
          this.busy = false;
          this.child = null;
          let result;
          try { result = JSON.parse(stdout); } catch (_) { result = null; }
          if (error || !result || result.ok !== true || result.pinned !== true
              || String(result.hwnd) !== handle || result.pid !== this.pid) {
            this.failed();
            return;
          }
          this.failures = 0;
          this.publish({ state: 'pinned', pinned: true, text: 'HUD visible on all virtual desktops.' });
          this.schedule(30000);
        });
    } catch (_) {
      this.busy = false;
      this.child = null;
      this.failed();
    }
  }

  failed() {
    if (this.stopped) return;
    this.failures += 1;
    this.publish({ state: this.failures < 3 ? 'checking' : 'unavailable', pinned: false,
      text: this.failures < 3 ? 'Preparing the HUD for all desktops…'
        : 'Could not keep the HUD on all desktops. Click Retry, or use Win+Tab → IntuitionOS → Show this window on all desktops.' });
    this.schedule(this.failures < 4 ? 1000 * this.failures : 30000);
  }

  schedule(delay) {
    this.timer = this.later(() => { this.timer = null; this.refresh(); }, delay);
  }

  stop() {
    this.stopped = true;
    this.generation += 1;
    this.busy = false;
    if (this.timer !== null) this.cancel(this.timer);
    this.timer = null;
    if (this.child) this.child.kill();
    this.child = null;
  }
}

module.exports = { DesktopVisibility, nativeHandle };
