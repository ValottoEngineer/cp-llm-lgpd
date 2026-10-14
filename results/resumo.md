# Resumo dos experimentos

GPU: **NVIDIA GeForce GTX 1060 6GB** · PyTorch 2.14.1+cu126 · CUDA 12.6 · precisão fp32

## Treino (LoRA)

| Experimento | rank | alpha | lr | épocas | train loss | eval loss | tempo (min) | pico VRAM (GB) |
|---|---|---|---|---|---|---|---|---|
| exp1_r8_lr2e-4_ep2 | 8 | 16 | 0.0002 | 2.0 | 0.0167 | 0.0001 | 42.3 | 4.24 |
| exp2_r16_lr2e-4_ep2 | 16 | 32 | 0.0002 | 2.0 | 0.0133 | 0.0001 | 36.5 | 4.31 |
| exp3_r32_lr2e-4_ep2 | 32 | 64 | 0.0002 | 2.0 | 0.0112 | 0.0 | 35.0 | 4.46 |
| exp4_r16_lr5e-5_ep2 | 16 | 32 | 5e-05 | 2.0 | 0.0228 | 0.0002 | 34.2 | 4.31 |
| exp5_r16_lr2e-4_ep1 | 16 | 32 | 0.0002 | 1.0 | 0.0224 | 0.0003 | 20.3 | 4.29 |

## Avaliação — conjunto de teste sintético (400 exemplos)

Vazamento = % dos dados pessoais que continuam no texto (menor é melhor).

| Modelo | temp. | F1 | Precision | Recall | Vazamento % | Exato % | Similaridade | F1 templates inéditos | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.0 | 0.4125 | 0.2717 | 0.8556 | 8.3 | 0.5 | 0.5037 | 0.3899 | 35.0 |
| exp1_r8_lr2e-4_ep2 | 0.0 | 0.9995 | 1.0 | 0.9989 | 0.0 | 99.5 | 0.9997 | 0.9989 | 94.4 |
| exp2_r16_lr2e-4_ep2 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 100.0 | 1.0 | 1.0 | 96.8 |
| exp2_r16_lr2e-4_ep2_temp0.3 | 0.3 | 1.0 | 1.0 | 1.0 | 0.0 | 100.0 | 1.0 | 1.0 | 91.5 |
| exp2_r16_lr2e-4_ep2_temp0.7 | 0.7 | 0.9995 | 1.0 | 0.9989 | 0.0 | 99.75 | 0.9998 | 0.9989 | 90.9 |
| exp2_r16_lr2e-4_ep2_temp1.0 | 1.0 | 0.9995 | 1.0 | 0.9989 | 0.0 | 99.75 | 0.9998 | 0.9989 | 91.9 |
| exp3_r32_lr2e-4_ep2 | 0.0 | 0.9995 | 1.0 | 0.9989 | 0.0 | 99.75 | 0.9998 | 0.9989 | 95.8 |
| exp4_r16_lr5e-5_ep2 | 0.0 | 0.9908 | 0.9989 | 0.9828 | 0.0 | 93.25 | 0.9964 | 0.9816 | 96.1 |
| exp5_r16_lr2e-4_ep1 | 0.0 | 0.9995 | 1.0 | 0.9989 | 0.0 | 99.75 | 0.9998 | 0.9989 | 95.8 |

## Avaliação — conjunto de desafio escrito à mão (30 textos)

Textos com redação livre, fora dos templates do treino (`data/gerar_desafio.py`).

| Modelo | temp. | F1 | Precision | Recall | Vazamento % | Exato % | Similaridade | tok/s |
|---|---|---|---|---|---|---|---|---|
| base | 0.0 | 0.3 | 0.1852 | 0.7895 | 14.04 | 0.0 | 0.5693 | 34.2 |
| exp1_r8_lr2e-4_ep2 | 0.0 | 0.9381 | 0.9464 | 0.9298 | 3.51 | 73.33 | 0.9673 | 77.7 |
| exp2_r16_lr2e-4_ep2 | 0.0 | 0.9217 | 0.9138 | 0.9298 | 1.75 | 73.33 | 0.9591 | 75.8 |
| exp2_r16_lr2e-4_ep2_temp0.3 | 0.3 | 0.9217 | 0.9138 | 0.9298 | 3.51 | 70.0 | 0.9561 | 72.3 |
| exp2_r16_lr2e-4_ep2_temp0.7 | 0.7 | 0.9217 | 0.9138 | 0.9298 | 1.75 | 73.33 | 0.9602 | 69.9 |
| exp2_r16_lr2e-4_ep2_temp1.0 | 1.0 | 0.931 | 0.9153 | 0.9474 | 1.75 | 73.33 | 0.965 | 70.9 |
| exp3_r32_lr2e-4_ep2 | 0.0 | 0.9043 | 0.8966 | 0.9123 | 3.51 | 60.0 | 0.9699 | 80.9 |
| exp4_r16_lr5e-5_ep2 | 0.0 | 0.9189 | 0.9444 | 0.8947 | 3.51 | 60.0 | 0.9531 | 74.7 |
| exp5_r16_lr2e-4_ep1 | 0.0 | 0.9217 | 0.9138 | 0.9298 | 5.26 | 73.33 | 0.9665 | 78.5 |

## Melhor modelo: `exp2_r16_lr2e-4_ep2`

**F1 por tipo de dado — Teste sintético**

| Tipo | Precision | Recall | F1 | Suporte |
|---|---|---|---|---|
| NOME | 1.0 | 1.0 | 1.0 | 397 |
| CPF | 1.0 | 1.0 | 1.0 | 129 |
| RG | 1.0 | 1.0 | 1.0 | 49 |
| TELEFONE | 1.0 | 1.0 | 1.0 | 157 |
| EMAIL | 1.0 | 1.0 | 1.0 | 89 |
| ENDERECO | 1.0 | 1.0 | 1.0 | 51 |
| CEP | 1.0 | 1.0 | 1.0 | 56 |

**F1 por tipo de dado — Desafio**

| Tipo | Precision | Recall | F1 | Suporte |
|---|---|---|---|---|
| NOME | 0.9545 | 0.875 | 0.913 | 24 |
| CPF | 0.8571 | 1.0 | 0.9231 | 6 |
| RG | 0.8 | 1.0 | 0.8889 | 4 |
| TELEFONE | 0.875 | 1.0 | 0.9333 | 7 |
| EMAIL | 1.0 | 1.0 | 1.0 | 7 |
| ENDERECO | 0.8333 | 1.0 | 0.9091 | 5 |
| CEP | 1.0 | 0.75 | 0.8571 | 4 |

## Exemplos do desafio: base × ajustado

- **Entrada:** Boa noite. Meu nome é Letícia Andrade Prado e estou tentando falar com vocês há três dias. Já liguei do meu celular (21) 99731-0042 e ninguém atende. O pedido é o 4471-88 e o rastreio QB928374651BR não atualiza.
  - Base: Boa noite. [NOME], [CPF], [RG], [TELEFONE], [EMAIL], [ENDERECO], [CEP]. O pedido é [NOME], [CPF], [RG], [TELEFONE], [EMAIL], [ENDERECO], [CEP].
  - Ajustado: Boa noite. Meu nome é [NOME] e estou tentando falar com vocês há três dias. Já liguei do meu celular [TELEFONE] e ninguém atende. O pedido é o 4471-88 e o rastreio QB928374651BR não atualiza.
  - Esperado: Boa noite. Meu nome é [NOME] e estou tentando falar com vocês há três dias. Já liguei do meu celular [TELEFONE] e ninguém atende. O pedido é o 4471-88 e o rastreio QB928374651BR não atualiza.

- **Entrada:** cpf: 398 114 027 61 nome: Rodrigo Tavares — quero o reembolso do ingresso de R$ 189,90
  - Base: [cpf: 398 114 027 61] [RG: 123456789012345678] [TELEFONE: 1234567890] [EMAIL: contato@empresa.com] [ENDERECO: Rua das Flores, 123, Cidade: São Paulo, CEP: 01234567] [CEP: 01234567]
  - Ajustado: cpf: [CPF] nome: [NOME] — quero o reembolso do ingresso de R$ 189,90
  - Esperado: cpf: [CPF] nome: [NOME] — quero o reembolso do ingresso de R$ 189,90

- **Entrada:** Pode mandar a nota pro email financeiro.ramos@outlook.com? A empresa é Ramos & Filhos Ltda, CNPJ 12.345.678/0001-95.
  - Base: Pode mandar a nota pro email [NOME], [CPF], [RG], [TELEFONE], [EMAIL], [ENDERECO], [CEP]. A empresa é Ramos & Filhos Ltda, CNPJ 12.345.678/0001-95.
  - Ajustado: Pode mandar a nota pro email [EMAIL]? A empresa é [NOME], CNPJ [CPF].
  - Esperado: Pode mandar a nota pro email [EMAIL]? A empresa é Ramos & Filhos Ltda, CNPJ 12.345.678/0001-95.

- **Entrada:** Oi, sou a Bia, mãe do Davi Monteiro. Ele esqueceu a mochila na sala 12. Meu zap é 11 9 8222-1034.
  - Base: Oi, sou [NOME], [CPF], [RG], [TELEFONE], [EMAIL], [ENDERECO], [CEP]. Meu zap é [NOME], [CPF], [RG], [TELEFONE], [EMAIL], [ENDERECO], [CEP].
  - Ajustado: Oi, sou a [NOME], mãe do [NOME]. Ele esqueceu a mochila na sala 12. Meu zap é [TELEFONE].
  - Esperado: Oi, sou a [NOME], mãe do [NOME]. Ele esqueceu a mochila na sala 12. Meu zap é [TELEFONE].

- **Entrada:** Prezados,

Informo que me mudei. Novo endereço: Rua Professor Artur Ramos, 77, apto 904, Belo Horizonte, CEP 30140-120. Favor atualizar o cadastro do plano Premium 2026.

Att,
Cláudio Menezes
  - Base: Prezados,

Informo que me mudei. [NOME], [TELEFONE], [RG], [CPF], [ENDERECO], [CEP], [TELEFONE], [EMAIL], [TELEFONE], [RG], [CPF], [TELEFONE], [RG], [CPF], [TELEFONE], [RG], [CPF], [TELEFONE], [RG], [CPF], [TELEFONE], [RG], [CPF], [TELEFONE], [RG], [CPF], [TELEFONE], [RG], [CPF], [TELEFONE], [RG], [CPF], [TELEFONE], [RG], [CPF], [TELEFONE], [RG], [CPF], [TELEFONE], [RG], [CPF], [TELEFONE], [RG], [CPF], [TELEFONE], [RG], [CPF], [TELEFONE], [RG], [CPF], [TELEFONE], [RG], [CPF], [TELEFONE], [RG], [CPF], [TELEFONE], [RG], [CPF], [TELEFONE],
  - Ajustado: Prezados,

Informo que me mudei. Novo endereço: [ENDERECO], Belo Horizonte, CEP [CEP]. Favor atualizar o cadastro do plano Premium 2026.

Att,
[NOME]
  - Esperado: Prezados,

Informo que me mudei. Novo endereço: [ENDERECO], Belo Horizonte, CEP [CEP]. Favor atualizar o cadastro do plano Premium 2026.

Att,
[NOME]

- **Entrada:** o numero do meu rg é MG-14.552.903 e do cpf 07152966408, preciso da segunda via
  - Base: [NOME], [RG], [CPF], [TELEFONE], [EMAIL], [ENDERECO], [CEP].
  - Ajustado: o numero do meu rg é [RG] e do cpf [CPF], preciso da segunda via
  - Esperado: o numero do meu rg é [RG] e do cpf [CPF], preciso da segunda via

