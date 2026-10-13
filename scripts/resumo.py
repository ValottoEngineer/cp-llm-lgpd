"""
Gera results/resumo.md (tabelas para o README) e results/curvas_loss.png
a partir de results/*.json e outputs/*/run_info.json.
"""
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "results"
OUTS = RAIZ / "outputs"


def carregar(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main():
    resultados = sorted((carregar(p) for p in RES.glob("*.json")), key=lambda r: r["nome"])
    resultados = [r for r in resultados if "metricas" in r]
    linhas = ["# Resumo dos experimentos", ""]

    gpu = next((r["gpu"] for r in resultados), {})
    linhas += [f"GPU da avaliação: **{gpu.get('gpu', '?')}** · torch {gpu.get('torch', '?')} · "
               f"CUDA {gpu.get('cuda', '?')} · precisão {gpu.get('precisao', '?')}", ""]

    linhas += ["## Treino (LoRA)", "",
               "| Experimento | rank | alpha | lr | épocas | train loss | eval loss | tempo (s) | pico VRAM (GB) |",
               "|---|---|---|---|---|---|---|---|---|"]
    runs = {}
    for p in sorted(OUTS.glob("*/run_info.json")):
        if p.parent.name == "melhor":
            continue
        i = carregar(p)
        runs[p.parent.name] = i
        c = i["config"]
        linhas.append(f"| {p.parent.name} | {c['r']} | {c['alpha']} | {c['lr']} | {c['epochs']} | "
                      f"{i['train_loss']} | {i['eval_loss']} | {i['tempo_treino_s']} | {i['pico_vram_gb']} |")

    linhas += ["", "## Avaliação no conjunto de teste", "",
               "Vazamento = % dos dados pessoais que continuam no texto (menor é melhor).", "",
               "| Modelo | temp. | F1 | Precision | Recall | Vazamento % | Exato % | Similaridade | "
               "F1 templates inéditos | tok/s |",
               "|---|---|---|---|---|---|---|---|---|---|"]
    for r in resultados:
        g = r["metricas"]["geral"]
        ined = r["metricas"].get("templates_ineditos", {}).get("f1", "-")
        linhas.append(f"| {r['nome']} | {r['temperature']} | {g['f1']} | {g['precision']} | {g['recall']} | "
                      f"{g['vazamento_pct']} | {g['exato_pct']} | {g['similaridade']} | {ined} | {r['tokens_por_s']} |")

    melhor_txt = OUTS / "melhor" / "ORIGEM.txt"
    if melhor_txt.exists():
        melhor = melhor_txt.read_text(encoding="utf-8").strip()
        r = next((x for x in resultados if x["nome"] == melhor), None)
        if r:
            linhas += ["", f"## F1 por tipo de dado — melhor modelo ({melhor})", "",
                       "| Tipo | Precision | Recall | F1 | Suporte |", "|---|---|---|---|---|"]
            for t, m in r["metricas"]["geral"]["por_tipo"].items():
                linhas.append(f"| {t} | {m['precision']} | {m['recall']} | {m['f1']} | {m['suporte']} |")
            base = next((x for x in resultados if x["nome"] == "base"), None)
            if base:
                linhas += ["", "## Exemplos: base × ajustado", ""]
                for a, b in list(zip(base["amostras"], r["amostras"]))[:5]:
                    linhas += [f"- **Entrada:** {a['input']}",
                               f"  - Base: {a['previsto']}",
                               f"  - Ajustado: {b['previsto']}",
                               f"  - Esperado: {a['esperado']}", ""]

    (RES / "resumo.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print("\n".join(linhas))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for nome in runs:
        hist = carregar(OUTS / nome / "log_history.json")
        pts = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]
        if pts:
            ax.plot(*zip(*pts), marker="o", ms=3, label=nome)
    ax.set_xlabel("passo")
    ax.set_ylabel("eval loss")
    ax.set_title("Loss de validação por experimento")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "curvas_loss.png", dpi=150)


if __name__ == "__main__":
    main()
