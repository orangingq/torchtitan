import argparse
import json
import os
from datetime import datetime
from lm_eval import evaluator
from setproctitle import setproctitle
from transformers import AutoConfig

DEFAULT_TASKS = ["mmlu", "hellaswag", "arc_challenge", "truthfulqa_mc1"] # "hendrycks_math" "arc_easy", 

def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate a local Hugging Face checkpoint (sharded safetensors) with lm-eval-harness")
    ap.add_argument("--model_path", type=str, required=True,
                    help="Local model directory path (the one containing model.safetensors.index.json)")
    ap.add_argument("--model_type", type=str, default="Llama-3.2-1B",
                    help="Model type (default: Llama-3.2-1B)")
    ap.add_argument("--tasks", type=str, default=",".join(DEFAULT_TASKS),
                    help="Comma-separated list of tasks (default: mmlu,hellaswag,arc_easy,arc_challenge,hendrycks_math)")
    ap.add_argument("--batch_size", type=str, default="auto",
                    help="Batch size for lm-eval (integer or 'auto')")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit the number of samples per task (for debugging)")
    ap.add_argument("--num_fewshot", type=int, default=None,
                    help="Global few-shot examples for all tasks (overridden by --num_fewshot_task)")
    ap.add_argument("--num_fewshot_task", action="append", default=None,
                    help="Per-task few-shot like 'task=K'. Repeatable. Also accepts comma-separated pairs or JSON.")
    ap.add_argument("--dtype", type=str, default="auto",
                    choices=["auto", "float16", "bfloat16", "float32"],
                    help="Data type for model loading")
    ap.add_argument("--device_map", type=str, default="auto",
                    help="accelerate/transformers device_map (e.g. 'auto', 'cuda', 'cpu')")
    # ap.add_argument("--trust_remote_code", action="store_true",
    #                 help="Trust remote code when loading custom model classes")
    ap.add_argument("--output_json", type=str, default=None,
                    help="Path to save results JSON (default: ./eval_results_<timestamp>.json)")
    ap.add_argument("--tokenizer_path", type=str, default='Llama-3.2-1B',
                help="Optional separate tokenizer folder name under assets/tokenizer/")
    return ap.parse_args()

def main():
    args = parse_args()
    setproctitle(f"[Eval] TimelyFreeze⏰ - {args.model_path}") # set the process title
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    # Parse per-task fewshot mapping
    fewshot_map = {}
    if args.num_fewshot_task:
        for entry in args.num_fewshot_task:
            try:
                if entry.strip().startswith("{"):
                    obj = json.loads(entry)
                    for k, v in obj.items():
                        fewshot_map[str(k).strip()] = int(v)
                else:
                    parts = [p for p in entry.split(",") if p]
                    for p in parts:
                        if "=" in p:
                            k, v = p.split("=", 1)
                            fewshot_map[k.strip()] = int(v.strip())
            except Exception as e:
                print(f"Warning: failed to parse --num_fewshot_task entry '{entry}': {e}")
    
    config_path = os.path.join(args.model_path, "config.json")
    if not os.path.isfile(config_path):
        print(f"config.json not found in {args.model_path}. Downloading from meta-llama/{args.model_type}...")
        cfg = AutoConfig.from_pretrained(f"meta-llama/{args.model_type}")
        print("Config OK:", cfg.model_type)
        cfg.save_pretrained(args.model_path)
        print(f"Downloaded config.json to {args.model_path}")

    # Transformers backend setup
    model_args = [
        f"pretrained={args.model_path}",
        f"dtype={args.dtype}",
        f"device_map={args.device_map}", 
        f"tokenizer=assets/tokenizer/{args.model_type if args.tokenizer_path is None else args.tokenizer_path}" 
    ]
    # if args.trust_remote_code:
    #     model_args.append("trust_remote_code=True")

    # Run evaluation, grouping tasks by few-shot setting
    groups = {}
    for t in tasks:
        k = fewshot_map.get(t, args.num_fewshot)
        groups.setdefault(k, []).append(t)

    merged = None
    for k, task_group in groups.items():
        part = evaluator.simple_evaluate(
            model="hf",
            model_args=",".join(model_args),
            tasks=task_group,
            batch_size=args.batch_size,
            num_fewshot=k,
            limit=args.limit,
            confirm_run_unsafe_code=True,
            random_seed=42,
            gen_kwargs={ "max_new_tokens": 512, "temperature": 0.0 },
        )
        if merged is None:
            merged = part
        else:
            # merge 'results' and other common dicts if present
            for key in ["results", "versions"]:
                if key in part:
                    merged.setdefault(key, {})
                    merged[key].update(part[key])
            # keep last config/other fields as-is
    results = merged if merged is not None else {}

    def print_metric(task_name, metrics):
        acc = None
        for k in ["acc,none", "acc", "exact_match", "accuracy", "pass@1", "pass@10", "pass_at_1"]:
            if k in metrics:
                acc = metrics[k]
                break
        if acc is not None:
            try:
                print(f"{task_name:20s} | {acc*100:.2f}%")
            except Exception:
                print(f"{task_name:20s} | {acc}")
        else:
            top_items = list(metrics.items())[:3]
            printable = ", ".join([f"{k}={v}" for k, v in top_items])
            print(f"{task_name:20s} | {printable}")
        return

    print("\n=== Summary (main metrics) ===")
    if "results" in results:
        for task_name in tasks:
            if task_name in results["results"]:
                print_metric(task_name, results["results"][task_name])
            else:
                print(f"{task_name:20s} | No results")
        print("\n=== Other metrics ===")
        for task_name, metrics in results["results"].items():
            if task_name not in tasks:
                print_metric(task_name, metrics)
    else:
        print("No 'results' key in lm-eval output. Full dump will be saved.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.output_json or os.path.abspath(f"./eval_results_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved full results to: {out_path}")

    print("\nTips:")
    print("- MMLU is usually run with 5-shot: --num_fewshot 5")
    print("- To see all available task keys: python -m lm_eval --tasks list")
    print("- For MATH errors, try replacing 'math' with 'hendrycks_math' in --tasks")
    print("- If VRAM is low: use --batch_size 1, --dtype float16/bfloat16, --device_map auto")

if __name__ == "__main__":
    main()
