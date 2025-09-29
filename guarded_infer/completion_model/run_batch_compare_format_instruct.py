# run_batch_compare.py
# ---------------------------------------------------------------------------
# Runs non-guarded and guarded generation over a list of prompts.
# Saves detailed JSONL with SAPLMA rejection traces and prints a clean summary.
# ---------------------------------------------------------------------------

import argparse
import json
from pathlib import Path
from typing import List, Optional

from non_guarded_generate_completion import generate_without_guardrail
from saplma_guarded_generate_instruct_auto_format import (
    generate_with_saplma_guardrail,
    GuardedGenerationResult,
)

def load_prompts(prompts_arg: Optional[str]) -> List[str]:
    if prompts_arg is None:
        return [
            "Tell me about the capital of France"
        ]
    p = Path(prompts_arg)
    if not p.exists():
        raise FileNotFoundError(f"Prompts file not found: {p}")
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

def guarded_to_dict(obj: GuardedGenerationResult) -> dict:
    return {
        "final_text": obj.final_text,
        "accepted_sentences": obj.accepted_sentences,
        "rejected": [
            {
                "type": r.type,
                "attempt": r.attempt,
                "mode": r.mode,
                "sentence_out": r.sentence_out,
                "clean_sentence": r.clean_sentence,
                "classify_text": r.classify_text,
                "prob_true": r.prob_true,
                "threshold": r.threshold,
                "reason": r.reason,
            } for r in obj.rejected
        ],
        "summary": obj.summary,
    }

def main():
    ap = argparse.ArgumentParser(description="Compare non-guarded vs guarded generations over prompts.")
    ap.add_argument("--model", required=True, help="Path or HF repo id of the base causal LM")
    ap.add_argument("--bundle", required=True, help="Path to SAPLMA bundle directory")
    ap.add_argument("--prompts", default=None, help="Optional path to prompts.txt (one prompt per line)")
    ap.add_argument("--out", default="results.jsonl", help="Output JSONL file with full details")
    ap.add_argument("--max_sentences", type=int, default=3)
    ap.add_argument("--max_tokens_per_sentence", type=int, default=128)
    ap.add_argument("--max_new_tokens_total", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.8)
    ap.add_argument("--device", default="auto", choices=["auto","cpu"], help="Force CPU if desired")
    args = ap.parse_args()

    prompts = load_prompts(args.prompts)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning {len(prompts)} prompts...\n")
    with out_path.open("w", encoding="utf-8") as fjsonl:
        for i, prompt in enumerate(prompts, 1):
            print("="*90)
            print(f"[{i}/{len(prompts)}] PROMPT: {prompt!r}")

            # # ---- Non-guarded ----
            # ng_text = generate_without_guardrail(
            #     prompt=prompt,
            #     model_path=args.model,
            #     temperature=args.temperature,
            #     top_p=args.top_p,
            #     max_sentences=args.max_sentences,
            #     max_new_tokens_total=args.max_new_tokens_total,
            #     max_tokens_per_sentence=args.max_tokens_per_sentence,
            #     device=args.device,
            #     log_level="WARNING",
            # )
            ng_text = ""

            # ---- Guarded (with details) ----
            g_result = generate_with_saplma_guardrail(
                prompt=prompt,
                bundle_path=args.bundle,
                model_path=args.model,
                decode_mode="hybrid",
                temperature=args.temperature,
                top_p=args.top_p,
                min_tokens_to_keep=5,
                classification_mode="cumulative",
                cumulative_exclude_prompt=False,
                max_sentences=args.max_sentences,
                max_new_tokens_total=args.max_new_tokens_total,
                max_tokens_per_sentence=args.max_tokens_per_sentence,
                retries_per_sentence=5,
                min_sentence_chars=1,
                min_alpha_chars=3,
                require_space_in_sentence=True,
                require_keywords=None,
                relax_filters_on_last_retry=True,
                threshold_offset=0.0,
                saplma_threshold=0.45,    # Uncomment to force a specific threshold
                device=args.device,
                log_level="WARNING",
                return_details=True,
            )
            assert isinstance(g_result, GuardedGenerationResult), "guarded call must return details"
            # g_result = ""

            # ---- Pretty console summary ----
            print("\nNON-GUARDED OUTPUT:")
            print(ng_text.strip())

            print("\nGUARDED OUTPUT:")
            print(g_result.final_text.strip())

            print("\nREJECTED SENTENCES (in order):")
            if not g_result.rejected:
                print("  (none)")
            else:
                for j, r in enumerate(g_result.rejected, 1):
                    head = f"  {j:02d}. [{r.type}] attempt={r.attempt} mode={r.mode}"
                    if r.type == "saplma" and r.prob_true is not None and r.threshold is not None:
                        tail = f"prob={r.prob_true:.3f} thr={r.threshold:.3f} -> {r.reason}"
                    else:
                        tail = r.reason or ""
                    snippet_src = r.clean_sentence or r.sentence_out or ""
                    snippet = snippet_src.replace("\n"," ")
                    if len(snippet) > 120:
                        snippet = snippet[:117] + "..."
                    print(f"{head} | {tail}\n      text: {snippet}")

            # ---- Persist JSONL record ----
            record = {
                "prompt": prompt,
                "non_guarded_text": ng_text,
                "guarded": guarded_to_dict(g_result),
            }
            fjsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nDone. Full results saved to: {out_path.resolve()}")
    print("Tip: `jq` is handy for browsing JSONL. Example:")
    print(f"  jq '.prompt, .guarded.summary, (.guarded.rejected|length)' {out_path.name}\n")

if __name__ == "__main__":
    main()



"""
python completion_model/run_batch_compare_format_instruct.py \
  --model ../models/Llama-2-7b-chat-hf \
  --bundle ../pretrained_saplma/instruct/format_3/saplma_checkpoints_LLAMA7/BEST_layer12__heldout_data/capitals \
  --prompts completion_model/prompts_instruct.txt \
  --out completion_model/results_format.jsonl

"""
