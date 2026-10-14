from metrics import avaliar, avaliar_por_grupo, pedacos, vazou_parcial

EX = {"input": "Sou Ana Souza, CPF 123.456.789-09, pedido 4471.",
      "output": "Sou [NOME], CPF [CPF], pedido 4471.",
      "entidades": [{"tipo": "NOME", "valor": "Ana Souza"}, {"tipo": "CPF", "valor": "123.456.789-09"}],
      "visto": True}


def test_resposta_perfeita():
    m = avaliar([EX], [EX["output"]])
    assert m["f1"] == 1.0 and m["vazamento_pct"] == 0 and m["exato_pct"] == 100 and m["similaridade"] == 1.0


def test_copiar_entrada_vaza_tudo():
    m = avaliar([EX], [EX["input"]])
    assert m["f1"] == 0.0 and m["vazamento_pct"] == 100 and m["exato_pct"] == 0


def test_marcador_inventado_reduz_precision():
    m = avaliar([EX], ["Sou [NOME], CPF [CPF], pedido [TELEFONE]."])
    assert m["recall"] == 1.0 and m["precision"] < 1.0 and m["por_tipo"]["TELEFONE"]["precision"] == 0.0


def test_vazamento_parcial():
    assert pedacos("financeiro.ramos@outlook.com") == ["financeiro", "ramos"]
    assert vazou_parcial("Ana Souza", "Sou [NOME] Souza, CPF [CPF]", EX["output"])
    assert not vazou_parcial("Ana Souza", EX["output"], EX["output"])
    m = avaliar([EX], ["Sou [NOME] Souza, CPF [CPF], pedido 4471."])
    assert m["vazamento_pct"] == 0 and m["vazamento_parcial_pct"] == 50


def test_grupos_vistos_e_ineditos():
    inedito = {**EX, "visto": False}
    r = avaliar_por_grupo([EX, inedito], [EX["output"], EX["input"]])
    assert r["templates_vistos"]["f1"] == 1.0 and r["templates_ineditos"]["f1"] == 0.0


def test_f1_deteccao_ignora_tipo():
    m = avaliar([EX], ["Sou [EMAIL], CPF [CPF], pedido 4471."])
    assert m["f1"] < 1.0 and m["f1_deteccao"] == 1.0 and m["vazamento_pct"] == 0
