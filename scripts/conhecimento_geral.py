"""
Mede o catastrophic forgetting: o fine-tuning prejudicou as respostas gerais do modelo?

Usa as 20 perguntas de data/geral_eval.jsonl (nunca usadas no treino), com temperatura 0, no modelo base
e nos adaptadores indicados. Métricas:
- acerto factual: % das perguntas com resposta conhecida cuja resposta contém o trecho esperado;
- similaridade com o base: o quanto a resposta se manteve parecida com a do modelo original (0 a 1);
- cópia do prompt: % de respostas que só repetem a pergunta (comportamento herdado da anonimização).

    python scripts/conhecimento_geral.py --adapters outputs/exp2_r16_lr2e-4_ep2 outputs/exp6_r16_lr2e-4_ep2_v2
"""
import argparse
import sys
from difflib import SequenceMatcher
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from common import MODELO_BASE, carregar_modelo, gerar_livre, ler_jsonl, salvar_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    args = ap.parse_args()

    perguntas = ler_jsonl(RAIZ / "data" / "geral_eval.jsonl")
    model, tok = carregar_modelo(MODELO_BASE, args.adapters[0])

    def responder():
        return [gerar_livre(model, tok, p["prompt"], None, args.max_new_tokens, 0.0)[0] for p in perguntas]

    respostas = {}
    with model.disable_adapter():
        respostas["base"] = responder()
    for i, caminho in enumerate(args.adapters):
        nome = Path(caminho).name
        if i > 0:
            model.load_adapter(caminho, adapter_name=nome)
        model.set_adapter(nome if i > 0 else "default")
        respostas[nome] = responder()

    com_gabarito = [j for j, p in enumerate(perguntas) if p["esperado"]]
    metricas = {}
    for nome, rs in respostas.items():
        acertos = sum(perguntas[j]["esperado"] in rs[j].lower() for j in com_gabarito)
        copias = sum(SequenceMatcher(None, r.lower(), p["prompt"].lower()).ratio() > 0.9 for r, p in zip(rs, perguntas))
        sim = sum(SequenceMatcher(None, r, b).ratio() for r, b in zip(rs, respostas["base"])) / len(rs)
        metricas[nome] = {"acerto_factual_pct": round(100 * acertos / len(com_gabarito), 1),
                          "similaridade_com_base": round(sim, 3),
                          "copia_do_prompt_pct": round(100 * copias / len(rs), 1)}

    linhas = ["# Conhecimento geral: base × ajustados (temperatura 0)", "",
              f"{len(perguntas)} perguntas inéditas (`data/geral_eval.jsonl`), {len(com_gabarito)} com resposta conhecida.", "",
              "| Modelo | Acerto factual % | Similaridade com o base | Cópia do prompt % |", "|---|---|---|---|"]
    for nome, m in metricas.items():
        linhas.append(f"| {nome} | {m['acerto_factual_pct']} | {m['similaridade_com_base']} | {m['copia_do_prompt_pct']} |")
    linhas.append("")
    for j, p in enumerate(perguntas):
        linhas += [f"## {p['prompt']}", ""]
        for nome, rs in respostas.items():
            linhas.append(f"- **{nome}:** {rs[j].replace(chr(10), ' ')}")
        linhas.append("")
    (RAIZ / "results" / "conhecimento_geral.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    salvar_json(RAIZ / "results" / "conhecimento_geral.json", {"metricas": metricas, "respostas": respostas})
    print("\n".join(linhas[:6 + len(metricas)]))


if __name__ == "__main__":
    main()
