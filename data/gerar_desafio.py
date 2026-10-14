"""
Conjunto de DESAFIO escrito à mão (não usa os templates do dataset sintético).

Mede generalização em textos mais próximos do mundo real: redação livre, textos longos,
vários dados de pessoas diferentes, formatos incomuns e distratores que parecem dados
pessoais (CNPJ, rastreio, cartão de fidelidade, protocolo) e devem ser mantidos.

Cada caso é escrito com marcações {TIPO:valor}; o script gera input/output/entidades.

    python data/gerar_desafio.py   ->  data/desafio.jsonl
"""
import json
import re
from pathlib import Path

CASOS = [
    "Boa noite. Meu nome é {NOME:Letícia Andrade Prado} e estou tentando falar com vocês há três dias. "
    "Já liguei do meu celular {TELEFONE:(21) 99731-0042} e ninguém atende. O pedido é o 4471-88 e o rastreio QB928374651BR não atualiza.",
    "cpf: {CPF:398 114 027 61} nome: {NOME:Rodrigo Tavares} — quero o reembolso do ingresso de R$ 189,90",
    "Pode mandar a nota pro email {EMAIL:financeiro.ramos@outlook.com}? A empresa é Ramos & Filhos Ltda, CNPJ 12.345.678/0001-95.",
    "Oi, sou a {NOME:Bia}, mãe do {NOME:Davi Monteiro}. Ele esqueceu a mochila na sala 12. Meu zap é {TELEFONE:11 9 8222-1034}.",
    "Prezados,\n\nInformo que me mudei. Novo endereço: {ENDERECO:Rua Professor Artur Ramos, 77, apto 904}, Belo Horizonte, "
    "CEP {CEP:30140-120}. Favor atualizar o cadastro do plano Premium 2026.\n\nAtt,\n{NOME:Cláudio Menezes}",
    "o numero do meu rg é {RG:MG-14.552.903} e do cpf {CPF:07152966408}, preciso da segunda via",
    "Chamado 2026-000981 aberto por {NOME:Thaís Correia} ({EMAIL:thais.correia@empresa.com.br}) às 09:12 sobre falha no VPN.",
    "Transferir R$ 2.500,00 da conta de {NOME:Osvaldo Pires} para {NOME:Marina Pires}, chave Pix {EMAIL:marina.pires@gmail.com}.",
    "Entregar na {ENDERECO:Av. Brigadeiro Faria Lima, 3477, 14º andar}, recepção, até 18h. Contato no local: Sr. {NOME:Hélio} {TELEFONE:+55 (11) 3048-7700}.",
    "Meu email é {EMAIL:joao.batista1987 arroba hotmail.com} pq o anterior foi hackeado",
    "Encaminho o laudo da paciente {NOME:Fernanda de Souza Lima}, 54 anos, CPF {CPF:226.407.538-02}, internada no leito 7B.",
    "A reunião de alinhamento com a equipe de São Paulo foi remarcada para quinta às 15h. Pauta: metas do Q4 e orçamento de R$ 40 mil.",
    "Olá! Recebi a cobrança 88213 duas vezes no cartão final 4421. Podem estornar?",
    "Testemunhas: 1) {NOME:Ana Paula Ferreira}, RG {RG:28.391.004-7}; 2) {NOME:Jorge Luiz Mendes}, RG {RG:31.002.876-X}.",
    "{NOME:Kátia} aqui, do RH. O {NOME:Fábio Nunes} pediu para alterar o telefone de emergência para {TELEFONE:(47) 98811-2093}.",
    "Seu código de verificação é 482913. Não compartilhe com ninguém.",
    "Dados para faturamento: {NOME:Luiz Henrique Couto}, {CPF:512.883.097-40}, {ENDERECO:Travessa Santa Rita, 15}, {CEP:40020-030}, Salvador/BA.",
    "fala comigo no {TELEFONE:85988127766} ou no insta, sou o {NOME:caio vasconcelos} da turma B",
    "Prezado Dr. {NOME:Augusto Lemos}, confirmamos sua inscrição no congresso. Número de inscrição: 2026-0457. Valor pago: R$ 650,00.",
    "Remetente: {NOME:Patrícia Gouveia} <{EMAIL:pgouveia@uol.com.br}>\nDestinatário: suporte\nAssunto: senha bloqueada após atualização 3.2.1",
    "O motorista não encontrou o endereço {ENDERECO:Estrada do Campo Limpo, 2200, bloco 3 casa 12}, CEP {CEP:05787000}. Pedido #7781023.",
    "Bom dia, o produto SKU 778-AZ-21 chegou quebrado. Comprei em 03/10 no valor de R$ 1.299,00 na loja de Curitiba.",
    "Para emissão do certificado: {NOME:Maria Eduarda Siqueira Campos}, CPF {CPF:345.019.226-81}, e-mail {EMAIL:duda.siqueira@fiap.com.br}.",
    "Atenção: o celular {TELEFONE:(61) 99404-1188}, que pertence a {NOME:Renan Barros}, foi usado em tentativa de fraude. Protocolo 7712093341.",
    "Matrícula 2026A0193 confirmada. Turno: noite. Início das aulas: 02/02.",
    "Eu, {NOME:Sebastião Rocha Filho}, RG {RG:4.118.229}, CPF {CPF:118.334.905-72}, autorizo {NOME:Lúcia Rocha} a retirar meus documentos.",
    "Contato comercial: {EMAIL:vendas@lojadoze.com.br} / {TELEFONE:(19) 3251-4400}. Atendemos de seg a sex.",
    "Vizinho aqui da {ENDERECO:R. das Acácias, 310} reclamando do barulho de novo. Sou o {NOME:Otávio}, do 2º andar.",
    "Ganhei o sorteio! Código do cupom PROMO2026XZ, válido até 31/12.",
    "Olá, {NOME:Gabriel}! Seu pedido 552-0091 saiu para entrega e chega hoje entre 14h e 18h no CEP {CEP:88015-200}.",
]

MARC_RE = re.compile(r"\{([A-Z]+):([^}]*)\}")


def main():
    exemplos = []
    for caso in CASOS:
        entidades = [{"tipo": t, "valor": v} for t, v in MARC_RE.findall(caso)]
        exemplos.append({
            "input": MARC_RE.sub(lambda m: m.group(2), caso),
            "output": MARC_RE.sub(lambda m: f"[{m.group(1)}]", caso),
            "entidades": entidades,
            "template": "desafio",
            "visto": False,
        })
    out = Path(__file__).parent / "desafio.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for e in exemplos:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    n_ent = sum(len(e["entidades"]) for e in exemplos)
    sem = sum(not e["entidades"] for e in exemplos)
    print(f"{len(exemplos)} casos, {n_ent} entidades, {sem} sem dados pessoais -> {out}")


if __name__ == "__main__":
    main()
