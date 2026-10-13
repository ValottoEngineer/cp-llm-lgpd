"""
Fine-tuning com LoRA do Qwen3 para anonimização LGPD.

A perda é calculada só sobre a resposta (o prompt é mascarado com -100).
A precisão é escolhida automaticamente pela GPU: bf16 (Ampere+), fp16 (T4) ou fp32 (GTX 1060).

Exemplo:
    python src/train.py --out outputs/exp1 --r 8 --alpha 16 --lr 2e-4 --epochs 2
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

sys.path.insert(0, str(Path(__file__).parent))
from common import FIM_RESPOSTA, MODELO_BASE, info_gpu, ler_jsonl, precisao_gpu, prompt_chat, salvar_json

RAIZ = Path(__file__).resolve().parents[1]


def tokenizar(ex, tok, max_len):
    p_ids = tok(prompt_chat(tok, ex["input"]), add_special_tokens=False)["input_ids"]
    r_ids = tok(ex["output"] + FIM_RESPOSTA, add_special_tokens=False)["input_ids"]
    ids = (p_ids + r_ids)[:max_len]
    labels = ([-100] * len(p_ids) + r_ids)[:max_len]
    return {"input_ids": ids, "labels": labels}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids, labels, mask = [], [], []
        for b in batch:
            falta = n - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * falta)
            labels.append(b["labels"] + [-100] * falta)
            mask.append([1] * len(b["input_ids"]) + [0] * falta)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(mask)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODELO_BASE)
    ap.add_argument("--data", default=str(RAIZ / "data"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--r", type=int, default=16, help="rank do LoRA")
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--train-limit", type=int, default=None, help="usa só N exemplos (teste rápido)")
    ap.add_argument("--val-limit", type=int, default=150, help="exemplos de validação para o eval_loss")
    ap.add_argument("--eval-steps", type=int, default=25)
    ap.add_argument("--grad-checkpointing", action="store_true", help="economiza VRAM (GPUs de 6 GB)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    prec = precisao_gpu()
    gpu = info_gpu()
    print(f"GPU: {gpu} | precisão de treino: {prec}")

    tok = AutoTokenizer.from_pretrained(args.model)
    train = [tokenizar(e, tok, args.max_len) for e in ler_jsonl(Path(args.data) / "train.jsonl", args.train_limit)]
    val = [tokenizar(e, tok, args.max_len) for e in ler_jsonl(Path(args.data) / "val.jsonl", args.val_limit)]

    # Pesos em fp32; com fp16/bf16 o Trainer usa precisão mista (autocast).
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lora = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    treinaveis, total = model.get_nb_trainable_parameters()
    print(f"Parâmetros treináveis: {treinaveis:,} de {total:,} ({100 * treinaveis / total:.2f}%)")

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.0,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="no",
        fp16=prec == "fp16",
        bf16=prec == "bf16",
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train, eval_dataset=val,
                      data_collator=Collator(tok.pad_token_id))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    resultado = trainer.train()
    duracao = time.time() - t0
    eval_final = trainer.evaluate()

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    salvar_json(Path(args.out) / "log_history.json", trainer.state.log_history)
    info = {
        "config": vars(args),
        "gpu": gpu,
        "precisao_treino": prec,
        "parametros_treinaveis": treinaveis,
        "parametros_totais": total,
        "tempo_treino_s": round(duracao, 1),
        "train_loss": round(resultado.training_loss, 4),
        "eval_loss": round(eval_final["eval_loss"], 4),
        "pico_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2) if torch.cuda.is_available() else None,
    }
    salvar_json(Path(args.out) / "run_info.json", info)
    print(info)


if __name__ == "__main__":
    main()
