import sounddevice as sd
from scipy.signal import resample
from dotenv import load_dotenv
import os


load_dotenv()
_hf_token = os.getenv("HF_TOKEN")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

#Load token before importing faster_whisper to avoid issues with Hugging Face authentication
from faster_whisper import WhisperModel

_model = None
_language = "en"


def get_model(model_size="base", device="cpu", compute_type="int8"):
    global _model
    if _model is None:
        _model = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _model


def record_audio(duration=5, samplerate=44100):
    target_sr = 16000
    print("🎤 Recording...")
    audio = sd.rec(
        int(samplerate * duration),
        samplerate=samplerate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    audio = audio.flatten()

    # Resample
    if samplerate != target_sr:
        print("🔄 Resampling...")
        num_samples = int(len(audio) * target_sr / samplerate)
        audio = resample(audio, num_samples)
    return audio


def transcribe_audio(audio, language=_language, beam_size=5):
    model = get_model()
    print("🧠 Transcribing...")
    segments, _info = model.transcribe(audio, language=language, beam_size=beam_size)
    texts = []
    for segment in segments:
        line = "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)
        texts.append(line)
    return texts


def transcribe_from_mic(duration=5, samplerate=44100, language=_language, beam_size=5):
    audio = record_audio(duration=duration, samplerate=samplerate)
    return transcribe_audio(audio, language=language, beam_size=beam_size)