# from wake_word import wait_for_wake_word
from whisper import transcribe_from_mic


def run_loop(duration=5):
    while True:
        user_input = input("Press Enter to listen (or type 'q' then Enter to quit): ").strip().lower()
        if user_input == "q":
            print("Exiting.")
            break
        lines = transcribe_from_mic(duration=duration, language="en", beam_size=5)
        for line in lines:
            print(line)


if __name__ == "__main__":
    try:
        run_loop(duration=10)
    except KeyboardInterrupt:
        print("\nStopped.")
