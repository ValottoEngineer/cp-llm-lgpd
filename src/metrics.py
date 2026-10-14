"""
Métricas de avaliação da anonimização (não dependem de torch).

- precision / recall / F1 por tipo: compara a quantidade de marcadores de cada tipo
  na resposta com a quantidade esperada.
- vazamento: % dos dados pessoais originais que continuam aparecendo na resposta
  (a métrica mais importante para LGPD: quanto menor, melhor).
- exato: % de respostas idênticas à referência.
- similaridade: quanto do texto não sensível foi preservado (difflib, 0 a 1).
- F1 de detecção: como o F1, mas ignorando o tipo do marcador (trocar [EMAIL] por [NOME] não é vazamento).
- vazamento parcial: % dos dados pessoais com algum PEDAÇO na resposta (ex.: o sobrenome, o usuário do
  e-mail, um bloco de dígitos do CPF) que também não esteja na resposta esperada.
"""
import re
from collections import Counter
from difflib import SequenceMatcher

from common import MARCADOR_RE, TIPOS

PALAVRAS_GENERICAS = {"rua", "avenida", "travessa", "alameda", "praça", "estrada", "apto", "bloco", "casa",
                      "torre", "condomínio", "residencial", "edifício", "gmail", "hotmail", "outlook", "yahoo",
                      "icloud", "empresa", "andar", "sala", "fundos", "municipal"}


def pedacos(valor):
    """Pedaços identificáveis de um dado: blocos de 3+ dígitos e palavras de 4+ letras (exceto genéricas)."""
    partes = re.findall(r"\d+|[^\W\d_]+", valor.lower())
    return [p for p in partes if (p.isdigit() and len(p) >= 3)
            or (not p.isdigit() and len(p) >= 4 and p not in PALAVRAS_GENERICAS)]


def vazou_parcial(valor, previsto, esperado):
    prev, esp = previsto.lower(), esperado.lower()
    return any(p in prev and p not in esp for p in pedacos(valor))


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def avaliar(exemplos, previsoes):
    cont = {t: Counter() for t in TIPOS}
    total_ent = vazados = parciais = exatos = 0
    det = Counter()  # detecção sem considerar o tipo do marcador
    sim = 0.0
    for ex, pred in zip(exemplos, previsoes):
        esperado = Counter(e["tipo"] for e in ex["entidades"])
        previsto = Counter(MARCADOR_RE.findall(pred))
        for t in TIPOS:
            g, p = esperado[t], previsto[t]
            cont[t]["tp"] += min(g, p)
            cont[t]["fp"] += max(0, p - g)
            cont[t]["fn"] += max(0, g - p)
        g_tot, p_tot = sum(esperado.values()), sum(previsto.values())
        det["tp"] += min(g_tot, p_tot)
        det["fp"] += max(0, p_tot - g_tot)
        det["fn"] += max(0, g_tot - p_tot)
        for e in ex["entidades"]:
            total_ent += 1
            vazados += e["valor"] in pred
            parciais += e["valor"] in pred or vazou_parcial(e["valor"], pred, ex["output"])
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
        "f1_deteccao": _prf(det["tp"], det["fp"], det["fn"])[2],
        "vazamento_pct": round(100 * vazados / max(total_ent, 1), 2),
        "vazamento_parcial_pct": round(100 * parciais / max(total_ent, 1), 2),
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
