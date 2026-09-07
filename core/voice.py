import os
import tempfile
import threading
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
        self._recording = False
        self._stream = None
        self._sample_rate = 16000
        self._stop_event = threading.Event()

    # ── Model ──────────────────────────────────────────────────────────────

    def _load_model(self):
        if self._model is not None:
            return
        from faster_whisper import WhisperModel
        self._model = WhisperModel(
            self.model_size,
            device="cpu",
            compute_type="int8",
        )

    # ── Transcribe helper ──────────────────────────────────────────────────

    def _transcribe_frames(self, frames: list) -> str:
        if not frames:
            return ""
        audio = np.concatenate(frames, axis=0).flatten()
        if audio.max() < 0.005:
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
        silence_sec: float = 0.7,
        max_sec: float = 15.0,
        speech_threshold: float = 0.02,
    ):
        """Open mic, auto-stop on silence, transcribe, and call callbacks.

        on_silence()     — called when mic closes (before transcription)
        on_complete(str) — called with transcript when done
        """
        self._stop_event.clear()

        def _run():
            import sounddevice as sd

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

            with self._lock:
                if self._recording:
                    return
                self._recording = True

            def _cb(indata, frame_count, time_info, status):
                nonlocal speech_detected, silence_count, total_count
                chunk = indata.copy()
                rms = float(np.sqrt(np.mean(chunk ** 2)))
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

            try:
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
                vad_done.wait()
                stream.stop()
                stream.close()
            except Exception as e:
                with self._lock:
                    self._recording = False
                    self._stream = None
                if on_complete:
                    on_complete("")
                return

            with self._lock:
                self._recording = False
                self._stream = None
                captured = list(frames)

            if on_silence:
                on_silence()

            if not speech_detected:
                if on_complete:
                    on_complete("")
                return

            text = self._transcribe_frames(captured)
            if on_complete:
                on_complete(text)

        threading.Thread(target=_run, daemon=True).start()

    def stop_now(self):
        """Signal the VAD loop to stop early (triggers transcription of whatever was captured)."""
        self._stop_event.set()

    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    def available(self) -> bool:
        try:
            import sounddevice  # noqa: F401
            from faster_whisper import WhisperModel  # noqa: F401
            return True
        except ImportError:
            return False
