import pytest

from common import MARCADOR_RE
from regex_pii import aplicar_regex, cpf_valido, extrair_dados


@pytest.mark.parametrize("texto, esperado", [
    ("meu CPF é 123.456.789-09.", "meu CPF é [CPF]."),
    ("cpf: 398 114 027 61 nome", "cpf: [CPF] nome"),
    ("Pix na chave 52833071946, ok", "Pix na chave [CPF], ok"),
    ("e-mail fernanda.rocha@gmail.com.", "e-mail [EMAIL]."),
    ("liga no (11) 98765-4321", "liga no [TELEFONE]"),
    ("zap 11 9 8222-1034", "zap [TELEFONE]"),
    ("Pix +5521987004411.", "Pix [TELEFONE]."),
    ("fala no 85988127766 ou", "fala no [TELEFONE] ou"),
    ("CEP 01310-100, SP", "CEP [CEP], SP"),
    ("cep 91420270", "cep [CEP]"),
    ("RG 12.345.678-9 ok", "RG [RG] ok"),
])
def test_encontra_dados_pessoais(texto, esperado):
    assert aplicar_regex(texto) == esperado


@pytest.mark.parametrize("texto", [
    "CNPJ 12.345.678/0001-95",
    "pedido 4471-88 e inscrição 2026-0457",
    "protocolo 7712093341",
    "protocolo 20260918551 aberto",
    "valor R$ 1.299,00 em 03/10",
    "rastreio QB928374651BR",
    "Olá, [NOME]! Seu CPF [CPF] foi validado.",
])
def test_nao_altera_o_que_nao_e_dado_pessoal(texto):
    assert aplicar_regex(texto) == texto


def test_cpf_valido():
    assert cpf_valido("52998224725")
    assert not cpf_valido("52998224724")
    assert not cpf_valido("11111111111")


def test_extrair_dados():
    original = "Olá, meu nome é Fernanda Lima Rocha e meu CPF é 123.456.789-09. Liga no (11) 98765-4321."
    anonimizado = "Olá, meu nome é [NOME] e meu CPF é [CPF]. Liga no [TELEFONE]."
    assert extrair_dados(original, anonimizado, MARCADOR_RE) == [
        ("NOME", "Fernanda Lima Rocha"), ("CPF", "123.456.789-09"), ("TELEFONE", "(11) 98765-4321")]
