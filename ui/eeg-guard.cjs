// Main-process global Escape ownership and a short backend arming lease.
const { randomUUID: nativeRandomUUID } = require('node:crypto');

class EegGuard {
  constructor({ globalShortcut, request, notify = () => {}, randomUUID = nativeRandomUUID,
    setInterval: every = setInterval, clearInterval: cancelEvery = clearInterval,
    setTimeout: later = setTimeout, clearTimeout: cancelLater = clearTimeout }) {
    this.shortcuts = globalShortcut;
    this.request = request;
    this.notify = notify;
    this.randomUUID = randomUUID;
    this.every = every;
    this.cancelEvery = cancelEvery;
    this.later = later;
    this.cancelLater = cancelLater;
    this.session = null;
    this.starting = null;
    this.stopping = null;
  }

  _request(action, payload) {
    // The shared HTTP helper has a longer generic POST timeout. A guard cannot
    // wait that long: its registration must fail closed before a 3-second lease.
    return new Promise((resolve, reject) => {
      let settled = false;
      const finish = (error, value) => {
        if (settled) return;
        settled = true;
        this.cancelLater(timer);
        if (error) reject(error); else resolve(value);
      };
      const timer = this.later(() => finish(new Error('Emergency guard request timed out')), 1500);
      Promise.resolve().then(() => this.request(action, payload)).then(
        response => response?.ok && !response.data?.error
          ? finish(null, response) : finish(new Error('Emergency guard request failed')),
        error => finish(error),
      );
    });
  }

  start() {
    if (this.session?.ready) {
      if (!this._registered()) {
        void this.emergencyStop('guard-shortcut-lost');
        return Promise.resolve({ ok: false, error: 'Global Escape is unavailable; EEG remains disarmed' });
      }
      return Promise.resolve({ ok: true, token: this.session.token });
    }
    if (this.starting) return this.starting;
    if (this.stopping) return Promise.resolve({ ok: false, error: 'Emergency stop is still completing' });
    const pending = this._prepare();
    this.starting = pending;
    pending.finally(() => { if (this.starting === pending) this.starting = null; });
    return pending;
  }

  async _prepare() {
    const session = { token: null, registered: false, ready: false, timer: null, renewing: false };
    try {
      const registered = this.shortcuts.register('Escape', () => {
        if (this.session === session) void this.emergencyStop('escape');
      });
      if (!registered) return { ok: false, error: 'Global Escape could not be registered; EEG remains disarmed' };
      session.registered = true;
      this.session = session;
      if (!this._registered()) throw new Error('Global Escape registration was not retained');
      session.token = this.randomUUID();
      await this._request('eeg_guard', { token: session.token });
      if (this.session !== session) return { ok: false, error: 'Emergency guard preparation was cancelled' };
      if (!this._registered()) throw new Error('Global Escape registration was lost');
      session.ready = true;
      session.timer = this.every(() => { void this._renew(session); }, 1000);
      return { ok: true, token: session.token };
    } catch (_) {
      if (this.session === session) {
        this._release(session);
        // The server may have accepted a request whose response was lost.
        void this._request('eeg_arm', { enabled: false }).catch(() => {});
      } else if (session.registered && this.session === null) {
        try { this.shortcuts.unregister('Escape'); } catch (_) {}
      }
      return { ok: false, error: 'Emergency guard could not be confirmed; EEG remains disarmed' };
    }
  }

  async _renew(session) {
    if (this.session !== session || !session.ready) return;
    if (!this._registered()) {
      await this.emergencyStop('guard-shortcut-lost');
      return;
    }
    if (session.renewing) return;
    session.renewing = true;
    try {
      await this._request('eeg_guard', { token: session.token });
    } catch (_) {
      if (this.session === session) await this.emergencyStop('guard-lease-lost');
    } finally {
      session.renewing = false;
    }
  }

  _registered() {
    try {
      return this.session?.registered === true &&
        (typeof this.shortcuts.isRegistered !== 'function' || this.shortcuts.isRegistered('Escape'));
    } catch (_) {
      return false;
    }
  }

  _release(session = this.session) {
    if (!session || this.session !== session) return;
    this.session = null;
    this.starting = null;
    session.ready = false;
    if (session.timer !== null) this.cancelEvery(session.timer);
    if (session.registered) {
      session.registered = false;
      try { this.shortcuts.unregister('Escape'); } catch (_) {}
    }
  }

  async stop(token) {
    if (token !== undefined && this.session?.token !== token) return { ok: true, stale: true };
    this._release();
    try {
      await this._request('eeg_arm', { enabled: false });
      return { ok: true };
    } catch (_) {
      return { ok: false, error: 'EEG disarm could not be confirmed; guard renewal stopped' };
    }
  }

  emergencyStop(reason = 'escape') {
    this._release();
    if (this.stopping) return this.stopping;
    try { this.notify(reason); } catch (_) {}
    const pending = this._request('stop', {}).then(
      () => ({ ok: true }),
      () => ({ ok: false, error: 'Emergency stop could not be confirmed; guard renewal stopped' }),
    );
    this.stopping = pending;
    pending.finally(() => { if (this.stopping === pending) this.stopping = null; });
    return pending;
  }

  dispose() {
    return this.emergencyStop('shutdown');
  }
}

module.exports = { EegGuard };
