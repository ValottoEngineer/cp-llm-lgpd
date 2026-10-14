import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path

from common import MARCADOR_RE
from regex_pii import cpf_valido

RAIZ = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gerar_dataset", RAIZ / "data" / "gerar_dataset.py")
gd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gd)


def test_cpf_gerado_tem_digitos_validos():
    rng = random.Random(0)
    for _ in range(200):
        assert cpf_valido(gd.gerar_cpf(rng).replace(".", "").replace("-", ""))


def test_saida_e_entidades_sao_consistentes():
    rng = random.Random(1)
    for tid, t in [(f"t{i}", t) for i, t in enumerate(gd.TEMPLATES + gd.TEMPLATES_V2 + gd.TEMPLATES_V3)]:
        entrada, saida, ents = gd.preencher(t, rng)
        assert "{" not in entrada and "{" not in saida, tid
        assert len(MARCADOR_RE.findall(saida)) == len(ents), tid
        for e in ents:
            assert e["valor"] in entrada and e["valor"] not in saida, (tid, e)


def test_cnpj_e_empresa_nao_viram_marcador():
    rng = random.Random(2)
    _, saida, ents = gd.preencher("A {EMPRESA}, CNPJ {CNPJ}, contratou {NOME}.", rng)
    assert [e["tipo"] for e in ents] == ["NOME"] and "CNPJ [" not in saida


def test_geracao_reprodutivel(tmp_path):
    for pasta in ("a", "b"):
        subprocess.run([sys.executable, str(RAIZ / "data" / "gerar_dataset.py"), "--n-train", "50",
                        "--n-val", "10", "--n-test", "20", "--out", str(tmp_path / pasta)], check=True,
                       capture_output=True)
    for f in ("train", "val", "test"):
        assert (tmp_path / "a" / f"{f}.jsonl").read_text(encoding="utf-8") == \
               (tmp_path / "b" / f"{f}.jsonl").read_text(encoding="utf-8")


def test_desafios_bem_formados():
    for arq in ("desafio.jsonl", "desafio2.jsonl", "desafio3.jsonl"):
        for l in open(RAIZ / "data" / arq, encoding="utf-8"):
            e = json.loads(l)
            assert len(MARCADOR_RE.findall(e["output"])) == len(e["entidades"])
            assert all(x["valor"] in e["input"] for x in e["entidades"])
