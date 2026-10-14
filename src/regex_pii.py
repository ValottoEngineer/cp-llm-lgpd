"""
Camada de segurança por regex (pós-processamento da LLM) e utilitários sem dependência de torch.

A LLM entende contexto (nomes, endereços, o que é ou não dado pessoal); o regex é determinístico
para formatos rígidos (CPF, e-mail, telefone, CEP). Combinados, um cobre as falhas do outro:
se a LLM deixar escapar um CPF, o regex apaga.

Os padrões são conservadores, para não apagar números de pedido, protocolos, CNPJ etc.:
- CPF: formatado (000.000.000-00 / 000 000 000 00), 11 dígitos logo após "CPF", "Pix" ou "chave",
  ou 11 dígitos soltos com dígitos verificadores válidos.
- TELEFONE: com +55, com DDD entre parênteses, "11 98765-4321" ou 11 dígitos (DDD + 9 + 8 dígitos).
- CEP: 00000-000, ou 8 dígitos logo após a palavra "CEP".
- RG: só quando precedido da palavra "RG".
"""
import re
from difflib import SequenceMatcher

D = r"(?<![\d.\-/])"   # não pode haver dígito/ponto/hífen/barra antes (evita pedaços de CNPJ etc.)
F = r"(?![\d\-/]|\.\d)"  # nem depois

PADROES = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("CPF", re.compile(D + r"\d{3}[.\s]\d{3}[.\s]\d{3}[-\s]\d{2}" + F)),
    ("TELEFONE", re.compile(r"\+55\s?\(?\d{2}\)?\s?9?\s?\d{4}[-\s]?\d{4}" + F)),
    ("TELEFONE", re.compile(D + r"\(\d{2}\)\s?9?\s?\d{4}[-\s]?\d{4}" + F)),
    ("TELEFONE", re.compile(D + r"\d{2}\s9?\s?\d{4,5}-\d{4}" + F)),
    ("CEP", re.compile(D + r"\d{5}-\d{3}" + F)),
    ("CEP", re.compile(r"(?i)(?<=cep)[:\s]*(\d{8})" + F)),
    ("CPF", re.compile(r"(?i)\b(?:cpfs?|pix|chave)\b[^\d\[\n]{0,25}?(\d{11})" + F)),
    ("RG", re.compile(r"(?i)(?<=\brg)[:\s]+((?:[A-Z]{2}-)?\d{1,2}\.?\d{3}\.?\d{3}-?[\dX]?)" + F)),
]
ONZE_DIGITOS = re.compile(D + r"\d{11}" + F)
DDDS = {11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 24, 27, 28, 31, 32, 33, 34, 35, 37, 38, 41, 42, 43, 44,
        45, 46, 47, 48, 49, 51, 53, 54, 55, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 73, 74, 75, 77, 79, 81,
        82, 83, 84, 85, 86, 87, 88, 89, 91, 92, 93, 94, 95, 96, 97, 98, 99}


def cpf_valido(digitos):
    if len(digitos) != 11 or len(set(digitos)) == 1:
        return False
    d = list(map(int, digitos))
    for k in (9, 10):
        s = sum(v * p for v, p in zip(d[:k], range(k + 1, 1, -1)))
        if (s * 10) % 11 % 10 != d[k]:
            return False
    return True


def aplicar_regex(texto):
    """Substitui por marcadores os dados pessoais de formato rígido que ainda estiverem no texto."""
    for tipo, padrao in PADROES:
        if padrao.groups:  # padrões com palavra-chave: troca só o grupo (o número)
            texto = padrao.sub(lambda m: m.group(0).replace(m.group(1), f"[{tipo}]"), texto)
        else:
            texto = padrao.sub(f"[{tipo}]", texto)

    def onze(m):
        s = m.group(0)
        if cpf_valido(s):
            return "[CPF]"
        if int(s[:2]) in DDDS and s[2] == "9":
            return "[TELEFONE]"
        return s
    return ONZE_DIGITOS.sub(onze, texto)


def extrair_dados(original, anonimizado, marcador_re):
    """Alinha o texto original com o anonimizado e devolve [(tipo, valor original)] de cada marcador."""
    achados = []
    sm = SequenceMatcher(None, original, anonimizado, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op not in ("replace", "insert"):
            continue
        tipos = marcador_re.findall(anonimizado[j1:j2])
        if not tipos:
            continue
        valor = original[i1:i2].strip(" ,;:")
        if len(tipos) == 1:
            achados.append((tipos[0], valor))
        else:  # vários marcadores colados: devolve o trecho inteiro para cada tipo
            achados.extend((t, valor) for t in tipos)
    return achados
