"""
Pós-processamento das avaliações (não usa GPU: trabalha sobre results/*_previsoes.jsonl).

1. Recalcula as métricas de todos os results/*.json (inclui o vazamento parcial).
2. Avalia a camada de segurança por regex: "só regex" (sem LLM), "só LLM" e "LLM + regex",
   em cada conjunto, e salva results/regex/*.json e results/regex_resumo.md.

    python scripts/pos_processamento.py
"""
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from common import ler_jsonl, salvar_json
from metrics import avaliar_por_grupo
from regex_pii import aplicar_regex

RES = RAIZ / "results"
CONJUNTOS = [("desafio3_", "desafio3.jsonl", "Desafio 3"), ("desafio2_", "desafio2.jsonl", "Desafio 2"),
             ("desafio_", "desafio.jsonl", "Desafio 1"), ("", "test.jsonl", "Teste sintético")]


def conjunto_de(nome):
    for prefixo, arq, titulo in CONJUNTOS:
        if nome.startswith(prefixo):
            return prefixo, arq, titulo


def main():
    dados = {arq: ler_jsonl(RAIZ / "data" / arq) for _, arq, _ in CONJUNTOS}

    # 1. Recalcula as métricas de todos os resultados.
    previsoes = {}
    for p in sorted(RES.glob("*_previsoes.jsonl")):
        nome = p.name.removesuffix("_previsoes.jsonl")
        prefixo, arq, _ = conjunto_de(nome)
        prev = [json.loads(l)["previsto"] for l in open(p, encoding="utf-8")]
        previsoes[nome] = prev
        rj = RES / f"{nome}.json"
        if rj.exists() and len(prev) == len(dados[arq]):
            r = json.loads(rj.read_text(encoding="utf-8"))
            r["metricas"] = avaliar_por_grupo(dados[arq], prev)
            salvar_json(rj, r)

    # 2. Regex: só regex, só LLM e LLM + regex, para o melhor modelo de cada rodada.
    melhor = (RAIZ / "outputs" / "melhor" / "ORIGEM.txt").read_text(encoding="utf-8").strip()
    modelos = list(dict.fromkeys(["exp2_r16_lr2e-4_ep2", "exp6_r16_lr2e-4_ep2_v2", melhor]))
    linhas = ["# Camada de segurança por regex", "",
              "Pós-processamento determinístico (`src/regex_pii.py`) para CPF, e-mail, telefone, CEP e RG "
              "(com a palavra RG). Nomes e endereços dependem de contexto, por isso só a LLM os encontra.", ""]
    for prefixo, arq, titulo in CONJUNTOS:
        ex = dados[arq]
        linhas += [f"## {titulo}", "",
                   "| Abordagem | F1 | Precision | Recall | Vazamento % | Vazamento parcial % | Exato % |",
                   "|---|---|---|---|---|---|---|"]
        variantes = [("Só regex (sem LLM)", "regex_puro", [aplicar_regex(e["input"]) for e in ex])]
        for m in modelos:
            if f"{prefixo}{m}" in previsoes:
                prev = previsoes[f"{prefixo}{m}"]
                curto = m.split("_")[0]
                variantes += [(f"Só LLM ({curto})", f"{m}_llm", prev),
                              (f"LLM ({curto}) + regex", f"{m}_regex", [aplicar_regex(p) for p in prev])]
        for rotulo, chave, prev in variantes:
            g = avaliar_por_grupo(ex, prev)["geral"]
            salvar_json(RES / "regex" / f"{prefixo}{chave}.json", {"nome": rotulo, "conjunto": titulo, "metricas": g})
            linhas.append(f"| {rotulo} | {g['f1']} | {g['precision']} | {g['recall']} | {g['vazamento_pct']} | "
                          f"{g['vazamento_parcial_pct']} | {g['exato_pct']} |")
        linhas.append("")
    texto = "\n".join(linhas) + "\n"
    (RES / "regex_resumo.md").write_text(texto, encoding="utf-8")
    print(texto)


if __name__ == "__main__":
    main()
