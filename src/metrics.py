"""
Métricas de avaliação da anonimização (não dependem de torch).

- precision / recall / F1 por tipo: compara a quantidade de marcadores de cada tipo
  na resposta com a quantidade esperada.
- vazamento: % dos dados pessoais originais que continuam aparecendo na resposta
  (a métrica mais importante para LGPD: quanto menor, melhor).
- exato: % de respostas idênticas à referência.
- similaridade: quanto do texto não sensível foi preservado (difflib, 0 a 1).
"""
from collections import Counter
from difflib import SequenceMatcher

from common import MARCADOR_RE, TIPOS


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def avaliar(exemplos, previsoes):
    cont = {t: Counter() for t in TIPOS}
    total_ent = vazados = exatos = 0
    sim = 0.0
    for ex, pred in zip(exemplos, previsoes):
        esperado = Counter(e["tipo"] for e in ex["entidades"])
        previsto = Counter(MARCADOR_RE.findall(pred))
        for t in TIPOS:
            g, p = esperado[t], previsto[t]
            cont[t]["tp"] += min(g, p)
            cont[t]["fp"] += max(0, p - g)
            cont[t]["fn"] += max(0, g - p)
        for e in ex["entidades"]:
            total_ent += 1
            vazados += e["valor"] in pred
        exatos += pred.strip() == ex["output"].strip()
        sim += SequenceMatcher(None, pred, ex["output"]).ratio()

    por_tipo = {}
    for t in TIPOS:
        p, r, f = _prf(cont[t]["tp"], cont[t]["fp"], cont[t]["fn"])
        por_tipo[t] = {"precision": p, "recall": r, "f1": f,
                       "suporte": cont[t]["tp"] + cont[t]["fn"]}
    tp = sum(c["tp"] for c in cont.values())
    fp = sum(c["fp"] for c in cont.values())
    fn = sum(c["fn"] for c in cont.values())
    p, r, f = _prf(tp, fp, fn)
    n = max(len(previsoes), 1)
    return {
        "precision": p, "recall": r, "f1": f,
        "vazamento_pct": round(100 * vazados / max(total_ent, 1), 2),
        "exato_pct": round(100 * exatos / n, 2),
        "similaridade": round(sim / n, 4),
        "n": len(previsoes),
        "por_tipo": por_tipo,
    }


def avaliar_por_grupo(exemplos, previsoes):
    """Separa templates vistos no treino x inéditos (generalização)."""
    res = {"geral": avaliar(exemplos, previsoes)}
    for nome, flag in (("templates_vistos", True), ("templates_ineditos", False)):
        pares = [(e, p) for e, p in zip(exemplos, previsoes) if e.get("visto", True) == flag]
        if pares:
            res[nome] = avaliar(*zip(*pares))
    return res
