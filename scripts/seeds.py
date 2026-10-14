"""
Variância entre execuções: treina a configuração do modelo final (exp7) com outras seeds e
reporta média ± desvio padrão (junto com a seed 42 original).

A seed muda a inicialização do LoRA, a ordem dos exemplos e o dropout; o dataset é o mesmo.
Resultados em results/seeds/ e results/seeds_resumo.md (não entram na escolha do modelo).

    python scripts/seeds.py                 # seeds 13 e 2024 (~1h30 na GTX 1060)
    python scripts/seeds.py --seeds 7 99
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PY = sys.executable
BASE = "exp7_r16_lr2e-4_ep2_v3"
CONJUNTOS = [("", "test.jsonl", "Teste sintético"), ("desafio_", "desafio.jsonl", "Desafio 1"),
             ("desafio2_", "desafio2.jsonl", "Desafio 2"), ("desafio3_", "desafio3.jsonl", "Desafio 3")]
METRICAS = [("vazamento_pct", "Vazamento %"), ("f1_deteccao", "F1 de detecção"), ("f1", "F1"),
            ("exato_pct", "Exato %"), ("similaridade", "Similaridade")]


def rodar(cmd):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, cwd=RAIZ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[13, 2024])
    args = ap.parse_args()
    sys.path.insert(0, str(RAIZ / "scripts"))
    from experimentos import args_memoria, impedir_suspensao
    impedir_suspensao()

    res = RAIZ / "results" / "seeds"
    for seed in args.seeds:
        out = RAIZ / "outputs" / f"{BASE}_seed{seed}"
        if not (out / "run_info.json").exists():
            rodar([PY, "src/train.py", "--out", out, "--r", 16, "--alpha", 32, "--lr", 2e-4, "--epochs", 2,
                   "--data", RAIZ / "data" / "v3", "--seed", seed, *args_memoria()])
        for prefixo, arq, _ in CONJUNTOS:
            nome = f"{prefixo}{BASE}_seed{seed}"
            if not (res / f"{nome}.json").exists():
                rodar([PY, "src/evaluate.py", "--nome", nome, "--adapter", out, "--data", RAIZ / "data" / arq,
                       "--out-dir", res])

    # Resumo: seed 42 (results/) + seeds novas (results/seeds/).
    linhas = [f"# Variância entre seeds — configuração do exp7 (r=16, lr=2e-4, 2 épocas, v3)", "",
              f"Seeds: 42 (original), {', '.join(map(str, args.seeds))}. Formato: média ± desvio padrão "
              "(valores individuais entre parênteses).", ""]
    for prefixo, _, titulo in CONJUNTOS:
        valores = {m: [] for m, _ in METRICAS}
        caminhos = [RAIZ / "results" / f"{prefixo}{BASE}.json"] + \
                   [res / f"{prefixo}{BASE}_seed{s}.json" for s in args.seeds]
        for c in caminhos:
            g = json.loads(c.read_text(encoding="utf-8"))["metricas"]["geral"]
            for m, _ in METRICAS:
                valores[m].append(g[m])
        linhas += [f"## {titulo}", "", "| Métrica | Média ± desvio | Por seed |", "|---|---|---|"]
        for m, rotulo in METRICAS:
            v = valores[m]
            linhas.append(f"| {rotulo} | {statistics.mean(v):.3f} ± {statistics.stdev(v):.3f} | "
                          f"{' / '.join(f'{x:g}' for x in v)} |")
        linhas.append("")
    texto = "\n".join(linhas) + "\n"
    (RAIZ / "results" / "seeds_resumo.md").write_text(texto, encoding="utf-8")
    print(texto)


if __name__ == "__main__":
    main()
