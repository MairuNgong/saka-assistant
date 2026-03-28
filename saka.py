from wake_word import wait_for_wake_word_vosk
from whisper import transcribe_from_mic
from llama import ask_llama


def run_loop(duration=5, wake_word=None):
    while True:
        wait_for_wake_word_vosk(wake_word=wake_word)
        while True:
          lines = transcribe_from_mic(duration=duration, language="en", beam_size=5)
          if not lines:
              print("⚠️ No speech detected, sleeping...")
              break
          for line in lines:
              print(f"You: {line}")
              try:
                  reply = ask_llama(line)
                  print(f"Saka: {reply}")
              except Exception as exc:
                  print(f"Saka error: {exc}")


if __name__ == "__main__":
    try:
        run_loop(duration=5, wake_word="hey bob")
    except KeyboardInterrupt:
        print("\nStopped.")
