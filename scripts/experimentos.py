"""
Roda todos os experimentos em sequência:
  1. avalia o modelo base (sem treino)
  2. treina e avalia cada configuração de LoRA
  3. testa parâmetros de geração (temperatura) no melhor adaptador
  4. copia o melhor adaptador para outputs/melhor e gera results/resumo.md

Experimentos já concluídos são pulados, então dá para rodar de novo
depois que a sessão do Colab cair.

    python scripts/experimentos.py              # tudo
    python scripts/experimentos.py --rapido     # teste de fumaça (poucos exemplos)
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PY = sys.executable

# Parâmetros testados: rank do LoRA, learning rate e número de épocas.
CONFIGS = [
    {"nome": "exp1_r8_lr2e-4_ep2", "r": 8, "alpha": 16, "lr": 2e-4, "epochs": 2},
    {"nome": "exp2_r16_lr2e-4_ep2", "r": 16, "alpha": 32, "lr": 2e-4, "epochs": 2},
    {"nome": "exp3_r32_lr2e-4_ep2", "r": 32, "alpha": 64, "lr": 2e-4, "epochs": 2},
    {"nome": "exp4_r16_lr5e-5_ep2", "r": 16, "alpha": 32, "lr": 5e-5, "epochs": 2},
    {"nome": "exp5_r16_lr2e-4_ep1", "r": 16, "alpha": 32, "lr": 2e-4, "epochs": 1},
    # Rodada 2: mesma configuração do melhor (exp2), com o dataset v2 (data/v2).
    {"nome": "exp6_r16_lr2e-4_ep2_v2", "r": 16, "alpha": 32, "lr": 2e-4, "epochs": 2, "data": "data/v2"},
    # Rodada 3: dataset v3 (erros do desafio 2) + perguntas gerais respondidas pelo modelo base (self-distillation).
    {"nome": "exp7_r16_lr2e-4_ep2_v3", "r": 16, "alpha": 32, "lr": 2e-4, "epochs": 2, "data": "data/v3"},
]
TEMPERATURAS = [0.3, 0.7, 1.0]


def impedir_suspensao():
    """Windows: impede o PC de suspender enquanto este processo roda (sem mudar configurações)."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def args_memoria():
    """GPUs com menos de 8 GB (ex.: GTX 1060 6 GB): batch 4 x acumulação 4 + gradient checkpointing.
    O batch efetivo continua 16, então os resultados são comparáveis aos da T4."""
    import torch
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 8 * 1024**3:
        return ["--batch", 4, "--grad-accum", 4, "--grad-checkpointing"]
    return []


def rodar(cmd):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, cwd=RAIZ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true", help="200 exemplos de treino, 40 de teste")
    ap.add_argument("--so", nargs="*", help="rodar só estes experimentos (por nome)")
    args = ap.parse_args()
    impedir_suspensao()

    limite_treino = ["--train-limit", 200] if args.rapido else []
    limite_teste = ["--limit", 40] if args.rapido else []
    res = RAIZ / "results"
    outs = RAIZ / "outputs"

    if not (RAIZ / "data" / "train.jsonl").exists():
        rodar([PY, "data/gerar_dataset.py"])

    if not (res / "base.json").exists():
        rodar([PY, "src/evaluate.py", "--nome", "base", *limite_teste])

    configs = [c for c in CONFIGS if not args.so or c["nome"] in args.so]
    for c in configs:
        out = outs / c["nome"]
        if not (out / "run_info.json").exists():
            extra = args_memoria()
            dados_treino = ["--data", RAIZ / c["data"]] if "data" in c else []
            rodar([PY, "src/train.py", "--out", out, "--r", c["r"], "--alpha", c["alpha"],
                   "--lr", c["lr"], "--epochs", c["epochs"], *dados_treino, *limite_treino, *extra])
        if not (res / f"{c['nome']}.json").exists():
            rodar([PY, "src/evaluate.py", "--nome", c["nome"], "--adapter", out, *limite_teste])

    # Conjuntos de desafio escritos à mão (textos fora dos templates).
    desafio = RAIZ / "data" / "desafio.jsonl"
    if not (RAIZ / "data" / "desafio3.jsonl").exists():
        rodar([PY, "data/gerar_desafio.py"])
    for prefixo, arq in (("desafio_", "desafio.jsonl"), ("desafio2_", "desafio2.jsonl"),
                         ("desafio3_", "desafio3.jsonl")):
        dados = RAIZ / "data" / arq
        if not (res / f"{prefixo}base.json").exists():
            rodar([PY, "src/evaluate.py", "--nome", f"{prefixo}base", "--data", dados])
        for c in CONFIGS:
            if (outs / c["nome"] / "run_info.json").exists() and not (res / f"{prefixo}{c['nome']}.json").exists():
                rodar([PY, "src/evaluate.py", "--nome", f"{prefixo}{c['nome']}", "--adapter", outs / c["nome"],
                       "--data", dados])

    # Melhor experimento, no desafio 3 (o único independente de todos os treinos, inclusive o v3):
    # 1) menor vazamento (para LGPD, deixar passar um dado pessoal é o erro mais grave);
    # 2) se a diferença de vazamento for de no máximo 1 dado (não significativa), vence o maior F1.
    def geral(nome):
        p = res / f"{nome}.json"
        return json.loads(p.read_text(encoding="utf-8"))["metricas"]["geral"] if p.exists() else None

    candidatos = []
    for c in CONFIGS:
        t, d3 = geral(c["nome"]), geral(f"desafio3_{c['nome']}")
        if t and d3:
            n_ent = sum(len(json.loads(l)["entidades"]) for l in open(RAIZ / "data" / "desafio3.jsonl", encoding="utf-8"))
            vazados = round(d3["vazamento_pct"] * n_ent / 100)
            candidatos.append((vazados, d3["f1"], t["f1"], c["nome"]))
    if not candidatos:
        return
    menor = min(c[0] for c in candidatos)
    empatados = [c for c in candidatos if c[0] <= menor + 1]
    melhor = max(empatados, key=lambda c: (c[1], c[2], -c[0]))[-1]
    print(f"\nMelhor experimento: {melhor}")

    # Parâmetros de geração: temperatura, no teste e no desafio.
    for t in TEMPERATURAS:
        for prefixo, dados in (("", []), ("desafio_", ["--data", desafio]),
                               ("desafio2_", ["--data", RAIZ / "data" / "desafio2.jsonl"]),
                               ("desafio3_", ["--data", RAIZ / "data" / "desafio3.jsonl"])):
            nome = f"{prefixo}{melhor}_temp{t}"
            if not (res / f"{nome}.json").exists():
                rodar([PY, "src/evaluate.py", "--nome", nome, "--adapter", outs / melhor,
                       "--temperature", t, *dados, *([] if prefixo else limite_teste)])

    destino = outs / "melhor"
    if destino.exists():
        shutil.rmtree(destino)
    shutil.copytree(outs / melhor, destino)
    (destino / "ORIGEM.txt").write_text(melhor, encoding="utf-8")

    rodar([PY, "scripts/resumo.py"])


if __name__ == "__main__":
    main()
