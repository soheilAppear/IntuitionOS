import os
import tempfile
import threading
import time
import wave

import numpy as np


class VoiceRecognizer:
    """VAD-based recorder + local Whisper transcriber.

    Calling start_recording_vad() opens the mic and monitors amplitude.
    Recording stops automatically after silence_sec of quiet after speech,
    or when stop_now() is called manually. on_silence fires when the mic
    closes (before transcription); on_complete fires with the transcript.
    """

    def __init__(self, model_size: str = "base", language: str = "en"):
        self.model_size = model_size
        self.language = language
        self._model = None
        self._lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._busy = False
        self._recording = False
        self._stream = None
        self._sample_rate = 16000
        self._stop_event = threading.Event()

    # ── Model ──────────────────────────────────────────────────────────────

    def _load_model(self):
        # Startup preparation and transcription can overlap. Load once; retain
        # first-run model download behavior and report any setup failure.
        with self._model_lock:
            if self._model is not None:
                return
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
            )

    def prepare(self):
        """Prepare local transcription without opening a mic (initial setup may download weights)."""
        import sounddevice  # noqa: F401

        self._load_model()

    # ── Transcribe helper ──────────────────────────────────────────────────

    def _transcribe_frames(self, frames: list) -> str:
        if not frames:
            return ""
        audio = np.concatenate(frames, axis=0).flatten()
        if np.max(np.abs(audio)) < 0.005:
            return ""

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self._sample_rate)
                pcm = (audio * 32767).astype(np.int16)
                wf.writeframes(pcm.tobytes())
            self._load_model()
            segments, _ = self._model.transcribe(
                tmp_path, beam_size=5, language=self.language
            )
            return " ".join(seg.text for seg in segments).strip()
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    # ── VAD recording ──────────────────────────────────────────────────────

    def start_recording_vad(
        self,
        on_silence=None,
        on_complete=None,
        on_error=None,
        silence_sec: float = 0.7,
        max_sec: float = 15.0,
        speech_threshold: float = 0.02,
    ):
        """Open mic, auto-stop on silence, transcribe, and call callbacks.

        on_silence()     — called when mic closes (before transcription)
        on_complete(str) — called with transcript when done
        on_error(str)    — microphone or transcription failure; no empty success
        """
        with self._lock:
            if self._busy:
                raise RuntimeError("Voice is already recording or transcribing")
            self._busy = True
            self._recording = True
            self._stop_event.clear()

        def notify(callback, *args):
            if callback:
                try:
                    callback(*args)
                except Exception:
                    # A disconnected HUD must not leak the stream or busy flag.
                    pass

        def _run():
            frames: list = []
            blocksize = 1024
            sr = self._sample_rate
            silence_limit = int(silence_sec * sr / blocksize)
            max_limit = int(max_sec * sr / blocksize)

            speech_detected = False
            silence_count = 0
            total_count = 0
            vad_done = threading.Event()
            chunk_lock = threading.Lock()

            def _cb(indata, frame_count, time_info, status):
                nonlocal speech_detected, silence_count, total_count
                chunk = indata.copy()
                rms = float(np.sqrt(np.mean(chunk**2)))
                with chunk_lock:
                    frames.append(chunk)
                    total_count += 1
                    if rms > speech_threshold:
                        speech_detected = True
                        silence_count = 0
                    elif speech_detected:
                        silence_count += 1
                        if silence_count >= silence_limit:
                            vad_done.set()
                    if total_count >= max_limit or self._stop_event.is_set():
                        vad_done.set()

            stream = None
            phase = "Microphone"
            try:
                import sounddevice as sd

                stream = sd.InputStream(
                    samplerate=sr,
                    channels=1,
                    dtype="float32",
                    callback=_cb,
                    blocksize=blocksize,
                )
                with self._lock:
                    self._stream = stream
                stream.start()
                # Some disconnected/virtual devices produce no callbacks. A
                # wall-clock deadline and manual stop must still close them.
                deadline = time.monotonic() + max_sec
                while not vad_done.wait(timeout=0.05):
                    if self._stop_event.is_set() or time.monotonic() >= deadline:
                        break
                try:
                    stream.stop()
                finally:
                    stream.close()
                    stream = None
                with self._lock:
                    self._recording = False
                    self._stream = None
                notify(on_silence)
                if not speech_detected:
                    notify(on_complete, "")
                    return
                phase = "Transcription"
                with chunk_lock:
                    captured = list(frames)
                text = self._transcribe_frames(captured)
                notify(on_complete, text)
            except Exception as error:
                if on_error:
                    notify(on_error, f"{phase} error: {error}")
                else:
                    notify(on_complete, "")
            finally:
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
                with self._lock:
                    self._recording = False
                    self._busy = False
                    self._stream = None

        worker = threading.Thread(target=_run, daemon=True, name="voice-capture")
        worker.start()
        return worker

    def stop_now(self):
        """Signal the VAD loop to stop early (triggers transcription of whatever was captured)."""
        self._stop_event.set()

    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    def is_busy(self) -> bool:
        """Capture and transcription share one recognizer and cannot overlap."""
        with self._lock:
            return self._busy

    def available(self) -> bool:
        try:
            import sounddevice  # noqa: F401
            from faster_whisper import WhisperModel  # noqa: F401

            return True
        except Exception:
            return False
