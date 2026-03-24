# from wake_word import wait_for_wake_word
from whisper import transcribe_from_mic


def run_loop(duration=5):
    print("Starting Saka loop. Press Ctrl+C to stop.")
    while True:
        lines = transcribe_from_mic(duration=duration, language="en", beam_size=5)
        for line in lines:
            print(line)


if __name__ == "__main__":
    try:
        run_loop(duration=5)
    except KeyboardInterrupt:
        print("\nStopped.")
