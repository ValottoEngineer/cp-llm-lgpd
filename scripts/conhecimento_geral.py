"""
Testa se o fine-tuning prejudicou o conhecimento geral do modelo (catastrophic forgetting).

Faz as mesmas perguntas gerais, com temperatura 0 (determinístico), ao modelo base e aos
adaptadores indicados, e salva as respostas lado a lado em results/conhecimento_geral.md.

    python scripts/conhecimento_geral.py --adapters outputs/exp2_r16_lr2e-4_ep2 outputs/exp6_r16_lr2e-4_ep2_v2
"""
import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from common import MODELO_BASE, carregar_modelo, gerar_livre

PERGUNTAS = [
    "O que significa a sigla LGPD? Responda em uma frase.",
    "Qual é a capital do Brasil?",
    "Explique em duas frases o que é um CPF.",
    "Quanto é 15 vezes 4?",
    "Escreva uma frase curta desejando bom dia a um cliente.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    args = ap.parse_args()

    respostas = {}
    model, tok = carregar_modelo(MODELO_BASE, args.adapters[0])
    with model.disable_adapter():
        respostas["base"] = [gerar_livre(model, tok, p, None, args.max_new_tokens, 0.0)[0] for p in PERGUNTAS]
    for i, caminho in enumerate(args.adapters):
        nome = Path(caminho).name
        if i > 0:
            model.load_adapter(caminho, adapter_name=nome)
        model.set_adapter(nome if i > 0 else "default")
        respostas[nome] = [gerar_livre(model, tok, p, None, args.max_new_tokens, 0.0)[0] for p in PERGUNTAS]

    linhas = ["# Conhecimento geral: base × ajustados (temperatura 0)", ""]
    for j, p in enumerate(PERGUNTAS):
        linhas += [f"## {p}", ""]
        for nome, rs in respostas.items():
            r = rs[j].replace("\n", " ")
            linhas.append(f"- **{nome}:** {r}")
        linhas.append("")
    texto = "\n".join(linhas) + "\n"
    (RAIZ / "results" / "conhecimento_geral.md").write_text(texto, encoding="utf-8")
    print(texto)


if __name__ == "__main__":
    main()
