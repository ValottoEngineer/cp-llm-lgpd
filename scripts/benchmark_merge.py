"""
Mede a velocidade de geração com o LoRA separado × fundido (merge) × modelo base,
numa requisição por vez (como no frontend), e confere se as saídas são idênticas.

    python scripts/benchmark_merge.py --adapter outputs/melhor
"""
import argparse
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import torch

from common import MODELO_BASE, anonimizar, carregar_modelo, ler_jsonl, modo_base, salvar_json


def medir(model, tok, textos):
    torch.cuda.synchronize()
    saidas, tokens, t0 = [], 0, time.time()
    for t in textos:
        s, n = anonimizar(model, tok, [t], 256, 0.0)
        saidas += s
        tokens += n
    torch.cuda.synchronize()
    dt = time.time() - t0
    return saidas, round(tokens / dt, 1), round(dt / len(textos), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(RAIZ / "outputs" / "melhor"))
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    textos = [e["input"] for e in ler_jsonl(RAIZ / "data" / "desafio2.jsonl", args.n)]
    model, tok = carregar_modelo(MODELO_BASE, args.adapter)
    anonimizar(model, tok, [textos[0]], 16, 0.0)  # aquecimento

    separado, tps_sep, lat_sep = medir(model, tok, textos)
    model.merge_adapter()
    fundido, tps_fun, lat_fun = medir(model, tok, textos)
    with modo_base(model):
        _, tps_base, lat_base = medir(model, tok, textos)
    iguais = sum(a == b for a, b in zip(separado, fundido))

    r = {"textos": len(textos), "gpu": torch.cuda.get_device_name(0),
         "lora_separado": {"tok_s": tps_sep, "latencia_s": lat_sep},
         "lora_fundido": {"tok_s": tps_fun, "latencia_s": lat_fun},
         "modelo_base": {"tok_s": tps_base, "latencia_s": lat_base},
         "saidas_identicas": f"{iguais}/{len(textos)}"}
    salvar_json(RAIZ / "results" / "benchmark_merge.json", r)
    print(r)


if __name__ == "__main__":
    main()
