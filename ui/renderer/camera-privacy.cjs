// Native capture ownership is independent of tracking, hand input, and panels.
function createCameraPrivacyIndicator({ element, request, isConnected,
  schedule = setTimeout, cancel = clearTimeout, now = Date.now }) {
  let snapshot = null, receivedAt = -Infinity, connected = false;
  let timer = null, epoch = 0, inFlight = false;
  const retiredInstances = new Set();
  const valid = data => data && ['on', 'off', 'starting', 'stopping', 'unknown'].includes(data.state)
    && (data.in_use === null || typeof data.in_use === 'boolean')
    && typeof data.release_confirmed === 'boolean'
    && typeof data.instance_id === 'string' && data.instance_id.length > 0 && data.instance_id.length <= 128
    && Number.isSafeInteger(data.revision) && data.revision >= 0
    && Array.isArray(data.owners) && data.owners.length <= 4
    && data.owners.every(owner => typeof owner === 'string' && owner.length <= 80)
    && (data.state !== 'off' || data.in_use === false && data.release_confirmed && data.owners.length === 0)
    && (data.state !== 'on' || data.in_use === true && !data.release_confirmed)
    && !(data.release_confirmed && (data.in_use !== false || data.state !== 'off'));

  function render() {
    // Silence, missing frames and a closed panel are never proof of release.
    const fresh = connected && snapshot && now() - receivedAt <= 3500;
    const state = fresh ? snapshot.state : 'unknown';
    const off = fresh && state === 'off' && snapshot.release_confirmed;
    const inUse = fresh && snapshot.in_use === true;
    element.textContent = off ? 'CAMERA OFF' : !fresh ? 'CAMERA ?'
      : state === 'starting' ? 'CAMERA STARTING' : state === 'stopping' ? 'CAMERA STOPPING'
      : inUse ? 'CAMERA ON' : 'CAMERA ?';
    element.classList.toggle('in-use', !!inUse);
    element.classList.toggle('uncertain', !off && (!fresh || state !== 'on'));
    element.setAttribute('data-state', state);
    element.title = off ? 'Camera release confirmed. Neither hand controls nor EEG controls owns capture.'
      : !fresh ? 'Camera status is unconfirmed. The camera may still be in use; reconnect to confirm release.'
      : state === 'starting' ? 'Opening the camera. Waiting for native capture confirmation.'
      : state === 'stopping' ? 'Camera is still owned while its capture operation finishes releasing.'
      : state === 'unknown' ? 'Camera release is unconfirmed. Stop the owning controls again to retry cleanup.'
      : `Camera is in use${snapshot.owners.length ? ' by ' + snapshot.owners.join(' and ') : ''}. Closing a preview does not stop capture.`;
    element.setAttribute('aria-label', `${element.textContent}. ${element.title}`);
  }

  function accept(data) {
    if (!connected || !valid(data) || retiredInstances.has(data.instance_id)) return false;
    data = { state: data.state, in_use: data.in_use, release_confirmed: data.release_confirmed,
      owners: [...data.owners], revision: data.revision, instance_id: data.instance_id };
    if (snapshot?.instance_id === data.instance_id && data.revision < snapshot.revision) return false;
    if (snapshot?.instance_id === data.instance_id && data.revision === snapshot.revision
        && JSON.stringify(data) !== JSON.stringify(snapshot)) return false;
    // A late HTTP/WS reply from the previous process cannot turn a new owner's
    // yellow indicator off. Revisions order replies within a backend instance.
    if (snapshot && snapshot.instance_id !== data.instance_id) retiredInstances.add(snapshot.instance_id);
    snapshot = { ...data, owners: [...data.owners] };
    receivedAt = now();
    render();
    return true;
  }

  function tick() {
    if (!connected) return;
    render();
    if (!inFlight) {
      inFlight = true;
      const turn = epoch;
      Promise.resolve().then(request).then(data => {
        if (turn === epoch) accept(data?.camera_privacy);
      }).catch(() => {
        if (turn === epoch) { receivedAt = -Infinity; render(); }
      }).finally(() => { if (turn === epoch) inFlight = false; });
    }
    // Keep polling when either controls panel is closed or the HUD is hidden.
    timer = schedule(tick, 1000);
  }

  function connectionChanged() {
    const next = !!isConnected();
    if (next !== connected) {
      connected = next;
      epoch += 1;
      inFlight = false;
      cancel(timer);
      timer = null;
      receivedAt = -Infinity;
      if (connected) tick();
    }
    render();
  }
  function refresh() {
    // Hidden-window timer throttling must not leave an old OFF badge visible
    // when the HUD returns. Recheck its age immediately, then request metadata.
    render();
    if (connected) { cancel(timer); tick(); }
  }
  render();
  return { accept, connectionChanged, refresh };
}

module.exports = { createCameraPrivacyIndicator };
