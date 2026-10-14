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


def tabela_avaliacao(resultados, com_ineditos):
    cab = "| Modelo | temp. | F1 | Precision | Recall | Vazamento % | Exato % | Similaridade |"
    sep = "|---|---|---|---|---|---|---|---|"
    if com_ineditos:
        cab += " F1 templates inéditos |"
        sep += "---|"
    cab += " tok/s |"
    sep += "---|"
    linhas = [cab, sep]
    for r in resultados:
        g = r["metricas"]["geral"]
        nome = r["nome"].removeprefix("desafio_")
        linha = (f"| {nome} | {r['temperature']} | {g['f1']} | {g['precision']} | {g['recall']} | "
                 f"{g['vazamento_pct']} | {g['exato_pct']} | {g['similaridade']} |")
        if com_ineditos:
            linha += f" {r['metricas'].get('templates_ineditos', {}).get('f1', '-')} |"
        linhas.append(linha + f" {r['tokens_por_s']} |")
    return linhas


def tabela_por_tipo(r):
    linhas = ["| Tipo | Precision | Recall | F1 | Suporte |", "|---|---|---|---|---|"]
    for t, m in r["metricas"]["geral"]["por_tipo"].items():
        if m["suporte"]:
            linhas.append(f"| {t} | {m['precision']} | {m['recall']} | {m['f1']} | {m['suporte']} |")
    return linhas


def main():
    todos = [carregar(p) for p in sorted(RES.glob("*.json"))]
    todos = [r for r in todos if "metricas" in r]
    teste = [r for r in todos if not r["nome"].startswith("desafio_")]
    desafio = [r for r in todos if r["nome"].startswith("desafio_")]
    por_nome = {r["nome"]: r for r in todos}

    linhas = ["# Resumo dos experimentos", ""]
    gpu = next((r["gpu"] for r in todos), {})
    linhas += [f"GPU: **{gpu.get('gpu', '?')}** · PyTorch {gpu.get('torch', '?')} · "
               f"CUDA {gpu.get('cuda', '?')} · precisão {gpu.get('precisao', '?')}", ""]

    linhas += ["## Treino (LoRA)", "",
               "| Experimento | rank | alpha | lr | épocas | train loss | eval loss | tempo (min) | pico VRAM (GB) |",
               "|---|---|---|---|---|---|---|---|---|"]
    runs = []
    for p in sorted(OUTS.glob("*/run_info.json")):
        if p.parent.name == "melhor":
            continue
        i = carregar(p)
        runs.append(p.parent.name)
        c = i["config"]
        linhas.append(f"| {p.parent.name} | {c['r']} | {c['alpha']} | {c['lr']} | {c['epochs']} | "
                      f"{i['train_loss']} | {i['eval_loss']} | {i['tempo_treino_s'] / 60:.1f} | {i['pico_vram_gb']} |")

    linhas += ["", "## Avaliação — conjunto de teste sintético (400 exemplos)", "",
               "Vazamento = % dos dados pessoais que continuam no texto (menor é melhor).", ""]
    linhas += tabela_avaliacao(teste, com_ineditos=True)

    if desafio:
        n = desafio[0]["metricas"]["geral"]["n"]
        linhas += ["", f"## Avaliação — conjunto de desafio escrito à mão ({n} textos)", "",
                   "Textos com redação livre, fora dos templates do treino (`data/gerar_desafio.py`).", ""]
        linhas += tabela_avaliacao(desafio, com_ineditos=False)

    melhor_txt = OUTS / "melhor" / "ORIGEM.txt"
    if melhor_txt.exists():
        melhor = melhor_txt.read_text(encoding="utf-8").strip()
        linhas += ["", f"## Melhor modelo: `{melhor}`", ""]
        for titulo, nome in (("Teste sintético", melhor), ("Desafio", f"desafio_{melhor}")):
            if nome in por_nome:
                linhas += [f"**F1 por tipo de dado — {titulo}**", ""] + tabela_por_tipo(por_nome[nome]) + [""]

        base, ajust = por_nome.get("desafio_base"), por_nome.get(f"desafio_{melhor}")
        if base and ajust:
            linhas += ["## Exemplos do desafio: base × ajustado", ""]
            for a, b in list(zip(base["amostras"], ajust["amostras"]))[:6]:
                linhas += [f"- **Entrada:** {a['input']}",
                           f"  - Base: {a['previsto']}",
                           f"  - Ajustado: {b['previsto']}",
                           f"  - Esperado: {a['esperado']}", ""]

    texto = "\n".join(linhas) + "\n"
    (RES / "resumo.md").write_text(texto, encoding="utf-8")
    print(texto)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for nome in runs:
        hist = carregar(OUTS / nome / "log_history.json")
        tr = [(h["step"], h["loss"]) for h in hist if "loss" in h]
        ev = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]
        if tr:
            axes[0].plot(*zip(*tr), label=nome)
        if ev:
            axes[1].plot(*zip(*ev), marker="o", ms=3, label=nome)
    for ax, titulo in zip(axes, ("Loss de treino", "Loss de validação")):
        ax.set_title(titulo)
        ax.set_xlabel("passo")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "curvas_loss.png", dpi=150)


if __name__ == "__main__":
    main()
