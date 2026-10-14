"""
Self-distillation contra o catastrophic forgetting.

1. Monta perguntas gerais (conceitos, contas, capitais, frases de atendimento, traduções, dicas).
2. Gera as respostas com o PRÓPRIO modelo base (temperatura 0) e salva em data/v3/geral.jsonl.
   Misturadas ao treino, elas ensinam o adaptador a anonimizar sem esquecer como responder outras coisas.
3. Salva um conjunto SEPARADO de perguntas de avaliação (data/geral_eval.jsonl), que nunca entra no treino.

Assuntos ligados a LGPD, CPF e dados pessoais ficam fora do treino, para que a avaliação nesses temas seja justa.

    python scripts/gerar_geral.py
"""
import json
import random
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

CONCEITOS = ["fotossíntese", "inflação", "um algoritmo", "a gravidade", "a democracia", "um vulcão",
             "a internet", "o ciclo da água", "uma vacina", "a energia solar", "um banco de dados",
             "a reciclagem", "o efeito estufa", "um computador", "a música clássica", "a economia circular",
             "um eclipse lunar", "a computação em nuvem", "o sistema solar", "a inteligência artificial",
             "um contrato de aluguel", "o juro composto", "uma planilha eletrônica", "a evaporação",
             "o futebol", "um terremoto", "a fotografia", "a matemática", "uma bactéria", "o oxigênio"]
CAPITAIS = ["França", "Japão", "Argentina", "Portugal", "Canadá", "Itália", "Alemanha", "Chile",
            "Bahia", "Minas Gerais", "Paraná", "Pernambuco", "Amazonas", "Santa Catarina", "Goiás", "Ceará",
            "Espanha", "México", "Egito", "Austrália"]
FRASES_ATENDIMENTO = ["desejando boa tarde a um cliente", "agradecendo uma compra", "pedindo desculpas pelo atraso na entrega",
                      "confirmando o recebimento de um e-mail", "convidando o cliente para avaliar o atendimento",
                      "informando que o pedido foi enviado", "desejando um bom fim de semana",
                      "pedindo um momento para verificar uma informação", "parabenizando um cliente pelo aniversário",
                      "encerrando um atendimento de forma educada"]
TRADUCOES = ["Bom dia, como posso ajudar?", "O pedido chegará amanhã.", "Obrigado pela preferência.",
             "A reunião foi cancelada.", "Onde fica a estação de trem?", "Eu gosto de ler livros.",
             "O céu está azul hoje.", "Precisamos de mais tempo.", "A loja abre às nove horas.",
             "Feliz aniversário!"]
DICAS = ["estudar melhor", "economizar dinheiro", "dormir bem", "organizar a rotina", "aprender programação",
         "manter uma alimentação saudável", "fazer uma boa apresentação", "trabalhar em equipe",
         "cuidar de plantas", "reduzir o estresse"]


def perguntas_treino(rng):
    ps = []
    for c in CONCEITOS:
        ps.append(f"Explique em poucas frases o que é {c}.")
        ps.append(f"O que é {c}? Responda de forma simples.")
    for c in CAPITAIS:
        ps.append(f"Qual é a capital {'da' if c in ('França', 'Argentina', 'Itália', 'Alemanha', 'Espanha', 'Austrália', 'Bahia') else 'do'} {c}?")
    for _ in range(50):
        a, b = rng.randint(2, 99), rng.randint(2, 30)
        op, simb = rng.choice([("mais", "+"), ("menos", "-"), ("vezes", "x")])
        ps.append(rng.choice([f"Quanto é {a} {op} {b}?", f"Calcule {a} {simb} {b}."]))
    for f in FRASES_ATENDIMENTO:
        ps.append(f"Escreva uma frase curta {f}.")
        ps.append(f"Escreva uma mensagem curta {f}.")
    for t in TRADUCOES:
        ps.append(f"Traduza para o inglês: \"{t}\"")
        ps.append(f"Traduza para o espanhol: \"{t}\"")
    for d in DICAS:
        ps.append(f"Dê 3 dicas para {d}.")
        ps.append(f"Como {d}? Responda em tópicos curtos.")
    return ps


# Avaliação: perguntas inéditas. "esperado" = trecho que a resposta correta deve conter.
PERGUNTAS_EVAL = [
    {"prompt": "O que significa a sigla LGPD? Responda em uma frase.", "esperado": "proteção de dados"},
    {"prompt": "Qual é a capital do Brasil?", "esperado": "brasília"},
    {"prompt": "Explique em duas frases o que é um CPF.", "esperado": None},
    {"prompt": "Quanto é 15 vezes 4?", "esperado": "60"},
    {"prompt": "Escreva uma frase curta desejando bom dia a um cliente.", "esperado": "bom dia"},
    {"prompt": "Qual é a capital do Rio Grande do Sul?", "esperado": "porto alegre"},
    {"prompt": "Qual é a capital da Inglaterra?", "esperado": "londres"},
    {"prompt": "Quanto é 120 dividido por 4?", "esperado": "30"},
    {"prompt": "Quanto é 7 mais 8?", "esperado": "15"},
    {"prompt": "Traduza para o inglês: \"Meu pedido ainda não chegou.\"", "esperado": "order"},
    {"prompt": "Explique em poucas frases o que é a fotossíntese das algas marinhas.", "esperado": None},
    {"prompt": "O que são dados pessoais? Responda em uma frase.", "esperado": None},
    {"prompt": "Dê 3 dicas para proteger sua senha.", "esperado": None},
    {"prompt": "Escreva uma frase curta pedindo o número do pedido ao cliente.", "esperado": "pedido"},
    {"prompt": "Qual planeta é conhecido como planeta vermelho?", "esperado": "marte"},
    {"prompt": "Quantos dias tem uma semana?", "esperado": "7"},
    {"prompt": "Explique o que é um e-mail em uma frase.", "esperado": None},
    {"prompt": "Corrija o português: \"Nós vai na loja amanhã.\"", "esperado": "vamos"},
    {"prompt": "Escreva uma saudação de boas-vindas para um novo funcionário.", "esperado": None},
    {"prompt": "Qual é o maior oceano do mundo?", "esperado": "pacífico"},
]


def main():
    import torch
    from common import FIM_RESPOSTA, MODELO_BASE, carregar_modelo

    rng = random.Random(7)
    prompts = perguntas_treino(rng)
    avaliacao = {p["prompt"] for p in PERGUNTAS_EVAL}
    prompts = [p for p in dict.fromkeys(prompts) if p not in avaliacao]

    model, tok = carregar_modelo(MODELO_BASE)
    fim = tok.convert_tokens_to_ids(FIM_RESPOSTA)
    saida = []
    for i in range(0, len(prompts), 16):
        lote = prompts[i:i + 16]
        textos = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False) for p in lote]
        enc = tok(textos, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=200, do_sample=False, temperature=None, top_p=None,
                                 top_k=None, pad_token_id=tok.pad_token_id, eos_token_id=fim)
        for p, seq in zip(lote, out[:, enc["input_ids"].shape[1]:]):
            terminou = bool((seq == fim).any())  # descarta respostas cortadas pelo limite de tokens
            r = tok.decode(seq, skip_special_tokens=True).strip()
            if terminou and r:
                saida.append({"input": p, "output": r, "geral": True})
        print(f"{min(i + 16, len(prompts))}/{len(prompts)}", end="\r")

    destino = RAIZ / "data" / "v3"
    destino.mkdir(parents=True, exist_ok=True)
    with open(destino / "geral.jsonl", "w", encoding="utf-8") as f:
        for e in saida:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(RAIZ / "data" / "geral_eval.jsonl", "w", encoding="utf-8") as f:
        for e in PERGUNTAS_EVAL:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"\n{len(saida)} de {len(prompts)} respostas completas -> {destino / 'geral.jsonl'}")


if __name__ == "__main__":
    main()
