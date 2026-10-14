"""
Gera um dataset sintético em PT-BR para anonimização de dados pessoais (LGPD).

Cada exemplo tem:
  input     -> texto com dados pessoais
  output    -> mesmo texto com os dados trocados por marcadores ([NOME], [CPF], ...)
  entidades -> lista dos dados pessoais presentes (tipo + valor)
  template  -> id do template usado
  visto     -> se o template também aparece no treino (False = template inédito)

Não depende de bibliotecas externas: todos os dados são inventados aqui
(CPFs têm dígitos verificadores válidos, mas são aleatórios).

Uso:
    python data/gerar_dataset.py --n-train 2000 --n-val 300 --n-test 400
"""
import argparse
import json
import random
import re
import unicodedata
from pathlib import Path

PII = ["NOME", "CPF", "RG", "TELEFONE", "EMAIL", "ENDERECO", "CEP"]

PRIMEIROS_NOMES = [
    "Ana", "Bruno", "Carla", "Daniel", "Eduarda", "Felipe", "Gabriela", "Henrique",
    "Isabela", "João", "Juliana", "Kaique", "Larissa", "Lucas", "Mariana", "Matheus",
    "Natália", "Otávio", "Paula", "Rafael", "Renata", "Rodrigo", "Sabrina", "Thiago",
    "Vanessa", "Vinícius", "Yasmin", "Pedro", "Beatriz", "Gustavo", "Camila", "André",
    "Letícia", "Marcos", "Fernanda", "Ricardo", "Aline", "Leonardo", "Patrícia", "Caio",
    "Maria Clara", "José", "Francisco", "Antônia", "Raimundo", "Luana", "Igor", "Débora",
]
SOBRENOMES = [
    "Silva", "Santos", "Oliveira", "Souza", "Rodrigues", "Ferreira", "Alves", "Pereira",
    "Lima", "Gomes", "Costa", "Ribeiro", "Martins", "Carvalho", "Almeida", "Lopes",
    "Soares", "Fernandes", "Vieira", "Barbosa", "Rocha", "Dias", "Nascimento", "Andrade",
    "Moreira", "Nunes", "Marques", "Machado", "Mendes", "Freitas", "Cardoso", "Ramos",
    "Gonçalves", "Santana", "Teixeira", "Araújo", "Batista", "Cavalcanti", "Monteiro", "Pinto",
]
TIPOS_LOGRADOURO = ["Rua", "Avenida", "Av.", "Travessa", "Alameda", "Praça", "Estrada", "R."]
NOMES_LOGRADOURO = [
    "das Flores", "Paulista", "Brasil", "XV de Novembro", "Sete de Setembro", "dos Andradas",
    "Rio Branco", "Getúlio Vargas", "Tiradentes", "das Palmeiras", "Santos Dumont",
    "Marechal Deodoro", "Dom Pedro II", "Augusta", "da Consolação", "Barão de Mauá",
    "Independência", "dos Bandeirantes", "Afonso Pena", "Boa Viagem", "Ipiranga",
    "Joaquim Nabuco", "Castro Alves", "Presidente Vargas", "do Comércio", "São João",
]
COMPLEMENTOS = ["apto 12", "ap. 304", "casa 2", "bloco B", "sala 1201", "fundos", "apto 81 bloco C"]
CIDADES = [
    ("São Paulo", "SP"), ("Rio de Janeiro", "RJ"), ("Belo Horizonte", "MG"), ("Curitiba", "PR"),
    ("Porto Alegre", "RS"), ("Salvador", "BA"), ("Recife", "PE"), ("Fortaleza", "CE"),
    ("Brasília", "DF"), ("Manaus", "AM"), ("Campinas", "SP"), ("Florianópolis", "SC"),
    ("Goiânia", "GO"), ("Belém", "PA"), ("Vitória", "ES"), ("Santos", "SP"),
]
DDDS = [11, 12, 19, 21, 27, 31, 41, 47, 48, 51, 61, 62, 71, 81, 85, 91, 92]
DOMINIOS = ["gmail.com", "hotmail.com", "outlook.com", "yahoo.com.br", "uol.com.br",
            "bol.com.br", "empresa.com.br", "fiap.com.br", "icloud.com"]
PRODUTOS = ["notebook", "fone de ouvido", "ventilador", "micro-ondas", "celular",
            "tênis de corrida", "kit de panelas", "monitor 27 polegadas", "smartwatch", "videogame"]

# Templates: {TIPO} ou {TIPO_2} para dados pessoais; {PEDIDO}, {VALOR}, {DATA}, {HORA},
# {PROTOCOLO}, {CIDADE}, {PRODUTO} são distratores que devem ser mantidos.
TEMPLATES = [
    "Olá, meu nome é {NOME} e meu CPF é {CPF}. Gostaria de cancelar o pedido {PEDIDO}.",
    "Bom dia! Sou {NOME}, CPF {CPF}, e o meu pedido {PEDIDO} ainda não chegou. Meu telefone é {TELEFONE}.",
    "Cliente: {NOME}\nCPF: {CPF}\nE-mail: {EMAIL}\nTelefone: {TELEFONE}\nEndereço: {ENDERECO}, {CIDADE}\nCEP: {CEP}",
    "Por favor, enviar a segunda via do boleto de {VALOR} para {EMAIL}. Titular: {NOME}.",
    "Prezados, solicito a alteração do meu endereço para {ENDERECO}, CEP {CEP}, {CIDADE}. Atenciosamente, {NOME}.",
    "oi, aqui é {NOME}, me liga no {TELEFONE} quando puder",
    "Paciente {NOME}, RG {RG}, consulta agendada para {DATA} às {HORA}. Contato: {TELEFONE}.",
    "Encaminho o currículo de {NOME} ({EMAIL}, {TELEFONE}) para a vaga de desenvolvedor.",
    "A entrega do {PRODUTO} deve ser feita na {ENDERECO}, {CIDADE}, CEP {CEP}, aos cuidados de {NOME}.",
    "Protocolo {PROTOCOLO}: o cliente {NOME} (CPF {CPF}) relatou cobrança indevida de {VALOR}.",
    "Meu e-mail mudou, agora é {EMAIL}. Podem atualizar o cadastro? Obrigado, {NOME}.",
    "Nome completo: {NOME} | RG: {RG} | CPF: {CPF} | Data do pedido: {DATA}",
    "Oi! Comprei um {PRODUTO} no dia {DATA} e veio com defeito. Meu número é {TELEFONE} e meu e-mail {EMAIL}.",
    "Declaro, para os devidos fins, que {NOME}, portador(a) do RG {RG} e CPF {CPF}, reside na {ENDERECO}, {CIDADE}.",
    "Favor transferir a titularidade da conta de {NOME} para {NOME_2}, CPF {CPF_2}.",
    "Reunião com {NOME} e {NOME_2} confirmada para {DATA} às {HORA} em {CIDADE}.",
    "Para recuperar a senha, informe o CPF cadastrado ({CPF}) e o e-mail {EMAIL}.",
    "Ligação recebida do número {TELEFONE} às {HORA}. Solicitante: {NOME}. Protocolo {PROTOCOLO}.",
    "Atualização cadastral: novo telefone {TELEFONE}, novo CEP {CEP}.",
    "O motoboy não achou a {ENDERECO}. Pode confirmar o CEP {CEP}? Pedido {PEDIDO}.",
    "Assinado por {NOME} - {EMAIL} - {TELEFONE}",
    "Matrícula do aluno {NOME}, responsável {NOME_2}, telefone do responsável {TELEFONE}.",
    "Boa tarde, gostaria de saber o status do pedido {PEDIDO} feito em {DATA}. Valor pago: {VALOR}.",
    "Seguem meus dados para o reembolso: {NOME}, CPF {CPF}, valor {VALOR}.",
    "vc pode mandar o contrato pro {EMAIL}? valeu",
    "Recebi uma mensagem estranha do {TELEFONE} pedindo meu CPF {CPF}. É golpe?",
    "Funcionário: {NOME}\nRG: {RG}\nAdmissão: {DATA}\nResidência: {ENDERECO} - {CIDADE}",
    "O cliente informou o CEP {CEP} mas o endereço {ENDERECO} não bate com o cadastro.",
    "Quero cancelar a assinatura. Titular {NOME}, e-mail {EMAIL}, protocolo anterior {PROTOCOLO}.",
    "Indicação: {NOME_2} indicou {NOME} para o programa. Contato do indicado: {TELEFONE}.",
    "Nota fiscal emitida para {NOME}, CPF {CPF}, endereço {ENDERECO}, {CIDADE}, CEP {CEP}, no valor de {VALOR}.",
    "Olá equipe, sou {NOME} e não consigo acessar minha conta com o e-mail {EMAIL}.",
    "Visita técnica agendada: {DATA}, {HORA}. Local: {ENDERECO}, {CIDADE}. Falar com {NOME} ({TELEFONE}).",
    "Meu RG é {RG} e meu CPF é {CPF}, estou mandando para completar o cadastro.",
    "Pedido {PEDIDO} - {PRODUTO} - {VALOR} - entregue para {NOME} em {DATA}.",
    "Prezado(a) {NOME}, recebemos sua solicitação {PROTOCOLO} e retornaremos em até 48 horas.",
    "Contato de emergência: {NOME_2}, {TELEFONE_2}. Paciente: {NOME}.",
    "Pode mandar mensagem no zap {TELEFONE}, sou {NOME} da {CIDADE}.",
    "O e-mail {EMAIL} e o telefone {TELEFONE} pertencem ao mesmo cadastro?",
    "Dados do inquilino: {NOME}, CPF {CPF}, RG {RG}. Imóvel: {ENDERECO}, CEP {CEP}.",
    "Oi, aqui é {NOME}. Troquei de número, o novo é {TELEFONE}. O antigo era {TELEFONE_2}.",
    "Remetente: {NOME} <{EMAIL}>\nAssunto: reclamação sobre o pedido {PEDIDO}",
    "Me chamo {NOME}, moro na {ENDERECO} e quero saber se vocês entregam no CEP {CEP}.",
    "O candidato {NOME}, CPF {CPF}, foi aprovado no processo seletivo de {DATA}.",
    "Segue o comprovante de pagamento de {VALOR} feito por {NOME} em {DATA}.",
    "Para emissão do certificado precisamos do nome completo e CPF: {NOME}, {CPF}.",
]

# Templates da versão 2 (--v2), criados a partir dos erros vistos no desafio: CNPJ e empresas
# (NÃO são dados pessoais), nome isolado no início, títulos (Dr., Sr.), idade e listas sem rótulo.
TEMPLATES_V2 = [
    "{PRIMEIRO} aqui, do financeiro. O {NOME} pediu reembolso de {VALOR}.",
    "{PRIMEIRO} falando. Pode me ligar no {TELEFONE}?",
    "Sou a {PRIMEIRO}, gerente da {EMPRESA}, CNPJ {CNPJ}. Meu contato direto é {EMAIL}.",
    "A nota deve sair em nome da {EMPRESA}, CNPJ {CNPJ}, e ser enviada para {EMAIL}.",
    "Fornecedor: {EMPRESA} (CNPJ {CNPJ}). Responsável: {NOME}, telefone {TELEFONE}.",
    "Prezado Dr. {NOME}, sua consulta foi confirmada para {DATA} às {HORA}.",
    "Falar com o Sr. {PRIMEIRO} na portaria, ramal 214.",
    "A Dra. {NOME} solicitou o prontuário do paciente {NOME_2}, {IDADE}, leito {LEITO}.",
    "Paciente {NOME}, {IDADE}, CPF {CPF}, encaminhado para a cardiologia.",
    "Dados: {NOME}, {CPF}, {CEP}.",
    "{NOME}; {TELEFONE}; {EMAIL}; {CIDADE}",
    "Cadastro rápido -> {NOME} / {CPF} / {ENDERECO} / {CEP}",
    "O rastreio {RASTREIO} do pedido {PEDIDO} não atualiza desde {DATA}. Sou {NOME}.",
    "Transferir {VALOR} da conta de {NOME} para {NOME_2}, chave Pix {EMAIL}.",
    "Oi {PRIMEIRO}! Seu pedido {PEDIDO} chega hoje no CEP {CEP}.",
    "Entregar na {ENDERECO}, recepção, aos cuidados do Sr. {PRIMEIRO}, {TELEFONE}.",
    "Vizinho da {ENDERECO} aqui, reclamando do barulho de novo. Sou o {PRIMEIRO}, do 3º andar.",
    "Contrato entre {EMPRESA}, CNPJ {CNPJ}, e {NOME}, CPF {CPF}.",
]
TEMPLATES_SEM_PII_V2 = [
    "A empresa {EMPRESA}, CNPJ {CNPJ}, foi cadastrada como fornecedora.",
    "O código de rastreio {RASTREIO} foi postado em {DATA}.",
    "Produto SKU {SKU} sem estoque em {CIDADE}.",
    "Emitir nota fiscal para {EMPRESA}, CNPJ {CNPJ}, no valor de {VALOR}.",
]
# Templates da versão 3 (--v3, inclui a v2), criados a partir dos erros vistos no desafio 2:
# chave Pix (CPF/telefone/e-mail sem rótulo), condomínios e residenciais, nome isolado no meio da frase.
TEMPLATES_V3 = [
    "Faz o Pix na chave {CPF_PIX}, está no nome de {NOME}.",
    "Chave Pix {CPF_PIX} ({NOME}). Valor: {VALOR}.",
    "Pode mandar o Pix pro {TELEFONE_PIX}, é do {PRIMEIRO}.",
    "Minha chave Pix é o e-mail {EMAIL}. Obrigado!",
    "Cadastrar chave Pix {CPF_PIX} em nome de {NOME} para o reembolso do pedido {PEDIDO}.",
    "Entregar no {CONDOMINIO}, {CEP}, falar com {NOME}.",
    "Moro no {CONDOMINIO}. Meu nome é {NOME} e o telefone é {TELEFONE}.",
    "Endereço de entrega: {CONDOMINIO} - {CIDADE}. Destinatário: {PRIMEIRO}.",
    "Ontem a {PRIMEIRO} passou aqui e deixou o documento do {NOME}.",
    "A {PRIMEIRO} ligou do {TELEFONE} pedindo para falar com o {PRIMEIRO_2}.",
    "Avisa a {PRIMEIRO} que o {PRIMEIRO_2} chegou na recepção.",
    "Hoje o {PRIMEIRO} não veio; quem cobriu o turno foi a {NOME}.",
    "Ficha: {NOME} ; {RG} ; {CEP} ; {CIDADE}",
    "{NOME} | RG {RG} | {TELEFONE}",
    "Inscrição: {NOME} - {CPF} - nascida em {ANO}.",
]
EMPRESAS = ["Ramos & Filhos Ltda", "Padaria Pão Dourado", "TechNova Sistemas S.A.", "Mercado Bom Preço",
            "Construtora Alicerce Ltda", "Farmácia Vida", "Auto Peças Silva ME", "Clínica Bem Estar",
            "Distribuidora Andrade", "Escola Pequeno Saber"]

# Exemplos sem nenhum dado pessoal: o modelo deve devolver o texto inalterado.
TEMPLATES_SEM_PII = [
    "Gostaria de saber o prazo de entrega para {CIDADE}.",
    "O pedido {PEDIDO} foi faturado em {DATA} no valor de {VALOR}.",
    "Qual o horário de funcionamento da loja de {CIDADE}?",
    "Meu {PRODUTO} parou de funcionar depois de uma semana. Tem garantia?",
    "Protocolo {PROTOCOLO} encerrado às {HORA}.",
    "Vocês aceitam pagamento por Pix? O valor é {VALOR}.",
    "A reunião foi remarcada para {DATA} às {HORA}.",
    "Quero trocar o {PRODUTO} por outro modelo, é possível?",
]

PH_RE = re.compile(r"\{([A-Z]+(?:_[A-Z]+)?)(?:_(\d))?\}")
# Placeholders especiais -> tipo de dado pessoal no marcador.
TIPO_MARCADOR = {"PRIMEIRO": "NOME", "CPF_PIX": "CPF", "TELEFONE_PIX": "TELEFONE", "CONDOMINIO": "ENDERECO"}


def _sem_acento(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def gerar_nome(rng):
    n = rng.choice(PRIMEIROS_NOMES)
    sobrenomes = rng.sample(SOBRENOMES, rng.choice([1, 2, 2, 3]))
    return " ".join([n] + sobrenomes)


def gerar_cpf(rng):
    d = [rng.randint(0, 9) for _ in range(9)]
    for k in (10, 11):
        s = sum(v * p for v, p in zip(d, range(k, 1, -1)))
        r = (s * 10) % 11
        d.append(0 if r == 10 else r)
    s = "".join(map(str, d))
    if rng.random() < 0.75:
        return f"{s[:3]}.{s[3:6]}.{s[6:9]}-{s[9:]}"
    return s


def gerar_rg(rng):
    d = "".join(str(rng.randint(0, 9)) for _ in range(8))
    dv = rng.choice("0123456789X")
    if rng.random() < 0.7:
        return f"{d[:2]}.{d[2:5]}.{d[5:]}-{dv}"
    return d + dv


def gerar_telefone(rng):
    ddd = rng.choice(DDDS)
    if rng.random() < 0.75:  # celular
        p1, p2 = f"9{rng.randint(1000, 9999)}", f"{rng.randint(0, 9999):04d}"
    else:  # fixo
        p1, p2 = f"{rng.randint(2000, 5999)}", f"{rng.randint(0, 9999):04d}"
    fmt = rng.choice([
        "({ddd}) {p1}-{p2}", "({ddd}) {p1}-{p2}", "{ddd} {p1}-{p2}",
        "+55 {ddd} {p1}-{p2}", "({ddd}){p1}{p2}", "{ddd}{p1}{p2}",
    ])
    return fmt.format(ddd=ddd, p1=p1, p2=p2)


def gerar_email(rng, nome=None):
    if nome and rng.random() < 0.7:
        partes = _sem_acento(nome).lower().split()
        usuario = rng.choice([
            f"{partes[0]}.{partes[-1]}", f"{partes[0][0]}{partes[-1]}",
            f"{partes[0]}{rng.randint(1, 99)}", f"{partes[0]}_{partes[-1]}{rng.randint(80, 99)}",
        ])
    else:
        usuario = _sem_acento(rng.choice(PRIMEIROS_NOMES)).lower().replace(" ", "") + str(rng.randint(1, 999))
    return f"{usuario}@{rng.choice(DOMINIOS)}"


CONDOMINIOS = ["Condomínio Jardim das Flores", "Residencial Bela Vista", "Condomínio Parque dos Ipês",
               "Residencial Vila Verde", "Edifício Solar das Águas", "Condomínio Recanto do Sol"]


def gerar_condominio(rng):
    c = rng.choice(CONDOMINIOS)
    return rng.choice([f"{c}, bloco {rng.choice('ABCD123')}, apto {rng.randint(11, 1504)}",
                       f"{c}, torre {rng.randint(1, 6)}, apto {rng.randint(11, 1504)}",
                       f"{c}, casa {rng.randint(1, 200)}"])


def gerar_endereco(rng):
    end = f"{rng.choice(TIPOS_LOGRADOURO)} {rng.choice(NOMES_LOGRADOURO)}, {rng.randint(1, 3500)}"
    if rng.random() < 0.3:
        end += f", {rng.choice(COMPLEMENTOS)}"
    return end


def gerar_cep(rng):
    n = f"{rng.randint(1000, 99999):05d}"
    s = f"{rng.randint(0, 999):03d}"
    return f"{n}-{s}" if rng.random() < 0.8 else n + s


def gerar_distrator(tipo, rng):
    if tipo == "PEDIDO":
        return rng.choice(["#", "nº ", ""]) + str(rng.randint(100000, 9999999))
    if tipo == "VALOR":
        v = rng.randint(10, 9000) + rng.randint(0, 99) / 100
        return "R$ " + f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    if tipo == "DATA":
        return f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}" + rng.choice(["", "/2026", "/2025"])
    if tipo == "HORA":
        return rng.choice([f"{rng.randint(7, 20)}h{rng.choice(['', '30', '15', '45'])}",
                           f"{rng.randint(7, 20):02d}:{rng.choice(['00', '30', '15'])}"])
    if tipo == "PROTOCOLO":
        return str(rng.randint(10**9, 10**10 - 1))
    if tipo == "CIDADE":
        c, uf = rng.choice(CIDADES)
        return rng.choice([c, f"{c}/{uf}", f"{c} - {uf}"])
    if tipo == "PRODUTO":
        return rng.choice(PRODUTOS)
    if tipo == "EMPRESA":
        return rng.choice(EMPRESAS)
    if tipo == "CNPJ":
        d = [rng.randint(0, 9) for _ in range(8)] + [0, 0, 0, 1] + [rng.randint(0, 9) for _ in range(2)]
        s = "".join(map(str, d))
        return f"{s[:2]}.{s[2:5]}.{s[5:8]}/{s[8:12]}-{s[12:]}"
    if tipo == "RASTREIO":
        letras = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return rng.choice(letras) + rng.choice(letras) + str(rng.randint(10**8, 10**9 - 1)) + "BR"
    if tipo == "IDADE":
        return f"{rng.randint(18, 90)} anos"
    if tipo == "LEITO":
        return f"{rng.randint(1, 30)}{rng.choice('ABCD')}"
    if tipo == "ANO":
        return str(rng.randint(1950, 2008))
    if tipo == "SKU":
        return f"{rng.randint(100, 999)}-{rng.choice(['AZ', 'BX', 'KT', 'PR'])}-{rng.randint(10, 99)}"
    raise ValueError(tipo)


def preencher(template, rng, informal=False):
    valores = {}
    nome_principal = None
    entrada, saida, entidades = [], [], []
    pos = 0
    for m in PH_RE.finditer(template):
        literal = template[pos:m.start()]
        if informal:
            literal = literal.lower()
        entrada.append(literal)
        saida.append(literal)

        tipo, chave = m.group(1), m.group(0)
        if chave not in valores:
            if tipo == "NOME":
                v = gerar_nome(rng)
                nome_principal = nome_principal or v
            elif tipo == "PRIMEIRO":  # só o primeiro nome (ex.: "Kátia aqui, do RH")
                v = rng.choice(PRIMEIROS_NOMES)
                nome_principal = nome_principal or v
            elif tipo == "CPF":
                v = gerar_cpf(rng)
            elif tipo == "CPF_PIX":  # chave Pix: CPF só com dígitos
                v = gerar_cpf(rng).replace(".", "").replace("-", "")
            elif tipo == "TELEFONE_PIX":  # chave Pix: telefone no formato +55DDDNÚMERO
                v = f"+55{rng.choice(DDDS)}9{rng.randint(10000000, 99999999)}"
            elif tipo == "CONDOMINIO":
                v = gerar_condominio(rng)
            elif tipo == "RG":
                v = gerar_rg(rng)
            elif tipo == "TELEFONE":
                v = gerar_telefone(rng)
            elif tipo == "EMAIL":
                v = gerar_email(rng, nome_principal)
            elif tipo == "ENDERECO":
                v = gerar_endereco(rng)
            elif tipo == "CEP":
                v = gerar_cep(rng)
            else:
                v = gerar_distrator(tipo, rng)
            if informal and TIPO_MARCADOR.get(tipo, tipo) == "NOME" and rng.random() < 0.5:
                v = v.lower()
            valores[chave] = v

        v = valores[chave]
        entrada.append(v)
        tipo_pii = TIPO_MARCADOR.get(tipo, tipo)
        if tipo_pii in PII:
            saida.append(f"[{tipo_pii}]")
            entidades.append({"tipo": tipo_pii, "valor": v})
        else:
            saida.append(v)
        pos = m.end()

    resto = template[pos:].lower() if informal else template[pos:]
    entrada.append(resto)
    saida.append(resto)
    return "".join(entrada), "".join(saida), entidades


SEM_PII = list(TEMPLATES_SEM_PII)


def gerar_exemplos(n, templates_pii, rng, visto, frac_sem_pii=0.08):
    exemplos = []
    for _ in range(n):
        if rng.random() < frac_sem_pii:
            idx = rng.randrange(len(SEM_PII))
            t, tid = SEM_PII[idx], f"sem_pii_{idx}"
        else:
            tid, t = rng.choice(templates_pii)
        inp, out, ents = preencher(t, rng, informal=rng.random() < 0.12)
        exemplos.append({"input": inp, "output": out, "entidades": ents,
                         "template": tid, "visto": visto})
    return exemplos


def salvar(path, exemplos):
    with open(path, "w", encoding="utf-8") as f:
        for e in exemplos:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--holdout", type=int, default=8,
                    help="templates reservados só para o teste (medem generalização)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--v3", action="store_true", help="inclui os templates da versão 3 (implica --v2)")
    ap.add_argument("--v2", action="store_true",
                    help="inclui os templates da versão 2 (CNPJ/empresas, nome isolado, títulos, listas)")
    ap.add_argument("--out", default=str(Path(__file__).parent))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.v2 = args.v2 or args.v3
    if args.v2:
        SEM_PII.extend(TEMPLATES_SEM_PII_V2)
    indexados = [(f"t{i}", t) for i, t in enumerate(TEMPLATES)]
    rng.shuffle(indexados)
    ineditos, vistos = indexados[:args.holdout], indexados[args.holdout:]
    if args.v2:  # os templates v2 entram sempre no treino
        vistos += [(f"v2_{i}", t) for i, t in enumerate(TEMPLATES_V2)]
    if args.v3:
        vistos += [(f"v3_{i}", t) for i, t in enumerate(TEMPLATES_V3)]

    train = gerar_exemplos(args.n_train, vistos, random.Random(args.seed + 1), True)
    val = gerar_exemplos(args.n_val, vistos, random.Random(args.seed + 2), True)
    metade = args.n_test // 2
    test = (gerar_exemplos(metade, vistos, random.Random(args.seed + 3), True)
            + gerar_exemplos(args.n_test - metade, ineditos, random.Random(args.seed + 4), False, 0.0))
    random.Random(args.seed + 5).shuffle(test)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    salvar(out / "train.jsonl", train)
    salvar(out / "val.jsonl", val)
    salvar(out / "test.jsonl", test)
    print(f"train={len(train)} val={len(val)} test={len(test)} "
          f"(templates vistos={len(vistos)}, inéditos no teste={len(ineditos)}) -> {out}")


if __name__ == "__main__":
    main()
