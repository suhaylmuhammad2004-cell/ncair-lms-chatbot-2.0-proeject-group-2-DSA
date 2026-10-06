# Running N-ATLaS offline (quantized)

**What it is.** N-ATLaS (NCAIR1/N-ATLaS) is an 8B Llama-3-based model for English, Hausa, Yoruba and Igbo, released by Awarri Technologies with the Federal Ministry of Communications, Innovation and Digital Economy. The full BF16 weights are ~16 GB, which is why it is slow on ordinary machines.

**Quantized builds found** (GGUF, repo `tosinamuda/N-ATLaS-GGUF`; sizes from the model page):

| Quant | Size | Use |
|---|---|---|
| **Q4_K_M** | 4.92 GB | recommended default, fastest |
| Q5_K_M | 5.73 GB | a bit more quality |
| Q6_K | 6.6 GB | near-lossless |
| Q8_0 | 8.54 GB | high quality, slower |
| F16 | 16.1 GB | same as the full model |

Other repos listed as quantizations of N-ATLaS on Hugging Face: `tosinamuda/N-ATLaS-FP8` (for GPU servers, not Ollama) and `Tushe/AMINI-ASSISTANT-GGUF-Q4-B` / `-F16` (the likely source of the `amini` model used in the old `test_amini.py`; it is a derivative, check its card before relying on it).

## Setup
```bash
# 1. install Ollama from https://ollama.com, then:
pip install -r requirements.txt
scripts/setup_natlas.sh Q4_K_M     # downloads once (needs internet), registers the model as "natlas"
NCAIR_PROFILE=natlas python scripts/check_natlas.py     # router JSON + multilingual replies + latency (checks the models of the chosen profile)
python src/app.py                  # chat UI
```
Or without the script: `ollama run hf.co/tosinamuda/N-ATLaS-GGUF:Q4_K_M`, then `ollama cp hf.co/tosinamuda/N-ATLaS-GGUF:Q4_K_M natlas`.

After the download nothing needs the internet. For the optional dense retriever, run once online so the embedding model is cached by `sentence-transformers`; otherwise the app uses BM25 only.

## Choosing models (profiles)
`NCAIR_PROFILE=llama` (**default**: llama3.2:3b routes and answers everything; N-ATLaS is used only to write Hausa, Yoruba and Igbo answers when it is installed), `natlas` (N-ATLaS routes and answers everything) and `mixed` (llama3.2:3b routes, N-ATLaS answers). Override individually with `NCAIR_ROUTER_MODEL` / `NCAIR_ANSWER_MODEL`, and set `NCAIR_MULTILINGUAL_MODEL=""` to switch the Hausa/Yoruba/Igbo specialist off. Run `python src/eval_benchmark.py --stage routing` once per profile and compare; keep the profile with the best exact-route score you can afford.

## Why routing uses JSON schemas, not native tool-calling
The router asks Ollama for structured output (`format` = a JSON schema generated from the tool registry). That works with any model, including N-ATLaS, and makes it impossible to name a non-existent tool, page or step.

## Things to know
* **Licence.** The N-ATLaS terms cap use at 1000 active end-users per 30 days without a commercial licence, forbid some uses, and require the attribution shown in the app footer ("powered by Awarri"). Read them on the model page before a public deployment.
* **Quality varies by language.** The model card's own human evaluation scored Yoruba lowest (2.69/5 vs 4.21 English, 3.98 Hausa, 3.87 Igbo). Nigerian Pidgin is not listed as a supported language. Treat non-English answers as needing review.
* Context is limited (~8k tokens); prompts here are kept well under 4k.
* Quantization costs some accuracy. Q4_K_M is a good default; if Yoruba/Igbo answers degrade, try Q5_K_M or Q6_K.
