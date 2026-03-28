import os
from dotenv import load_dotenv
from llama_cpp import Llama

load_dotenv()

DEFAULT_REPO_ID = "Qwen/Qwen2.5-3B-Instruct-GGUF"
DEFAULT_FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"
DEFAULT_SYSTEM_PROMPT = "You are a sarcastic AI who speaks like a cowboy named Bob."

_llm = None


def _get_repo_settings():
	repo_id = os.getenv("LLAMA_REPO_ID", DEFAULT_REPO_ID)
	filename = os.getenv("LLAMA_FILENAME", DEFAULT_FILENAME)
	return repo_id, filename


def load_llama(n_ctx=2048, n_threads=0):
	global _llm
	if _llm is None:
		repo_id, filename = _get_repo_settings()
		kwargs = {
			"repo_id": repo_id,
			"filename": filename,
			"chat_format": "chatml",
			"n_ctx": n_ctx,
			"verbose": False,
		}
		if n_threads and n_threads > 0:
			kwargs["n_threads"] = n_threads
		_llm = Llama.from_pretrained(**kwargs)
	return _llm


def ask_llama(user_text, system_prompt=DEFAULT_SYSTEM_PROMPT, max_tokens=128, temperature=0.7):
	llm = load_llama()
	response = llm.create_chat_completion(
		messages=[
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_text},
		],
		max_tokens=max_tokens,
		temperature=temperature,
		top_p=0.9,
		repeat_penalty=1.1,
	)
	text = response["choices"][0]["message"]["content"].strip()
	if text.lower() == user_text.strip().lower():
		return "Can you rephrase that?"
	return text
