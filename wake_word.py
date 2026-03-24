import queue
import json
import numpy as np
import sounddevice as sd
from scipy.signal import resample
from vosk import Model, KaldiRecognizer, SetLogLevel

SetLogLevel(-1)

def wait_for_wake_word_vosk(wake_word="ok google"):
    """
    Wake word detection using Vosk (speech-to-text based)
    """
    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status)
        q.put(bytes(indata))

    model = Model("vosk-model-small-en-us-0.15")

    input_samplerate = 44100
    recognizer_samplerate = 16000
    recognizer = KaldiRecognizer(model, recognizer_samplerate)

    print(
        f"🎤 Listening for wake word: '{wake_word}' "
    )

    with sd.RawInputStream(
        samplerate=input_samplerate,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=callback
    ):
        while True:
            data = q.get()
            pcm_44k = np.frombuffer(data, dtype=np.int16)
            if pcm_44k.size == 0:
                continue

            target_length = int(pcm_44k.size * recognizer_samplerate / input_samplerate)
            if target_length <= 0:
                continue

            pcm_16k = resample(pcm_44k, target_length)
            pcm_16k = np.clip(np.rint(pcm_16k), -32768, 32767).astype(np.int16)
            resampled_bytes = pcm_16k.tobytes()

            if recognizer.AcceptWaveform(resampled_bytes):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").lower()

                if text:
                    print("📝 Heard:", text)

                if wake_word in text:
                    print(f"✨ Wake word detected: '{wake_word}'")
                    return wake_word