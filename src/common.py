"""Funções compartilhadas: prompt, carregamento do modelo, geração e info da GPU."""
import json
import re
from pathlib import Path

MODELO_BASE = "Qwen/Qwen3-0.6B"
TIPOS = ["NOME", "CPF", "RG", "TELEFONE", "EMAIL", "ENDERECO", "CEP"]
FIM_RESPOSTA = "<|im_end|>"
MARCADOR_RE = re.compile(r"\[(" + "|".join(TIPOS) + r")\]")

SYSTEM_PROMPT = (
    "Você é um anonimizador de dados pessoais conforme a LGPD. "
    "Reescreva o texto do usuário substituindo cada dado pessoal pelo marcador correspondente: "
    "[NOME], [CPF], [RG], [TELEFONE], [EMAIL], [ENDERECO], [CEP]. "
    "Mantenha todo o resto do texto exatamente igual (números de pedido, valores, datas, cidades). "
    "Responda apenas com o texto anonimizado."
)


def ler_jsonl(path, limite=None):
    with open(path, encoding="utf-8") as f:
        dados = [json.loads(l) for l in f if l.strip()]
    return dados[:limite] if limite else dados


def mensagens(texto):
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": texto}]


def prompt_chat(tokenizer, texto):
    # enable_thinking=False: o Qwen3 responde direto, sem o bloco <think>.
    return tokenizer.apply_chat_template(mensagens(texto), tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def precisao_gpu():
    """bf16 (Ampere+), fp16 (Turing/Volta, ex.: T4) ou fp32 (Pascal, ex.: GTX 1060, ou CPU)."""
    import torch
    if not torch.cuda.is_available():
        return "fp32"
    if torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability()[0] >= 8:
        return "bf16"
    if torch.cuda.get_device_capability()[0] >= 7:
        return "fp16"
    return "fp32"


def info_gpu():
    import torch
    if not torch.cuda.is_available():
        return {"gpu": "CPU (CUDA indisponível)", "torch": torch.__version__}
    p = torch.cuda.get_device_properties(0)
    return {
        "gpu": p.name,
        "compute_capability": f"{p.major}.{p.minor}",
        "vram_gb": round(p.total_memory / 1024**3, 2),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "precisao": precisao_gpu(),
    }


def carregar_modelo(base=MODELO_BASE, adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precisao_gpu()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def anonimizar(model, tok, textos, max_new_tokens=256, temperature=0.0, top_p=0.9):
    """Gera as versões anonimizadas. Retorna (saídas, total de tokens gerados)."""
    import torch

    prompts = [prompt_chat(tok, t) for t in textos]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
                  eos_token_id=tok.convert_tokens_to_ids(FIM_RESPOSTA))
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    with torch.inference_mode():
        out = model.generate(**enc, **kwargs)
    novos = out[:, enc["input_ids"].shape[1]:]
    n_tokens = int((novos != tok.pad_token_id).sum())
    saidas = tok.batch_decode(novos, skip_special_tokens=True)
    saidas = [re.sub(r"<think>.*?</think>", "", s, flags=re.S).strip() for s in saidas]
    return saidas, n_tokens


def salvar_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
