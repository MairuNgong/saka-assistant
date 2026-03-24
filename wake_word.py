import numpy as np
import sounddevice as sd
from openwakeword.model import Model

def wait_for_wake_word(wake_word="ok google", threshold=0.5):
    """
    Block until OpenWakeWord detects the configured wake word.
    
    Args:
        wake_word: Wake word to listen for. Built-in options include:
                   "ok google", "hey google", "hey siri", "alexa", "hey cortana"
        threshold: Detection confidence threshold (0.0-1.0)
    """
    model = Model(model_path="tiny" if wake_word != "ok google" else None)
    
    print(f"Listening for wake word: '{wake_word}'")
    
    # OpenWakeWord expects 16kHz mono audio
    samplerate = 16000
    frame_length = 1280  # ~80ms chunk
    
    try:
        with sd.RawInputStream(
            samplerate=samplerate,
            blocksize=frame_length,
            dtype="int16",
            channels=1,
        ) as stream:
            while True:
                pcm, overflowed = stream.read(frame_length)
                if overflowed:
                    continue

                pcm_frame = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                prediction = model.predict(pcm_frame, threshold=threshold)
                
                # prediction is a dict: {"ok google": 0.95, ...}
                if prediction.get(wake_word, 0) > threshold:
                    print(f"✨ Wake word detected: '{wake_word}'")
                    return wake_word
    finally:
        model = None
