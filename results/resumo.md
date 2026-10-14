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
| exp6_r16_lr2e-4_ep2_v2 | 16 | 32 | 0.0002 | 2.0 | 0.0164 | 0.0001 | 31.4 | 4.28 |

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
| exp3_r32_lr2e-4_ep2_temp0.3 | 0.3 | 1.0 | 1.0 | 1.0 | 0.0 | 100.0 | 1.0 | 1.0 | 90.0 |
| exp3_r32_lr2e-4_ep2_temp0.7 | 0.7 | 0.9984 | 0.9989 | 0.9978 | 0.0 | 99.5 | 0.9997 | 0.9968 | 90.1 |
| exp3_r32_lr2e-4_ep2_temp1.0 | 1.0 | 0.9989 | 1.0 | 0.9978 | 0.0 | 99.5 | 0.9997 | 0.9979 | 90.2 |
| exp4_r16_lr5e-5_ep2 | 0.0 | 0.9908 | 0.9989 | 0.9828 | 0.0 | 93.25 | 0.9964 | 0.9816 | 96.1 |
| exp5_r16_lr2e-4_ep1 | 0.0 | 0.9995 | 1.0 | 0.9989 | 0.0 | 99.75 | 0.9998 | 0.9989 | 95.8 |
| exp6_r16_lr2e-4_ep2_v2 | 0.0 | 0.9978 | 1.0 | 0.9957 | 0.0 | 99.0 | 0.9993 | 0.9957 | 95.2 |
| exp6_r16_lr2e-4_ep2_v2_temp0.3 | 0.3 | 0.9989 | 1.0 | 0.9978 | 0.0 | 99.5 | 0.9997 | 0.9979 | 90.0 |
| exp6_r16_lr2e-4_ep2_v2_temp0.7 | 0.7 | 0.9968 | 1.0 | 0.9935 | 0.0 | 98.5 | 0.999 | 0.9936 | 90.4 |
| exp6_r16_lr2e-4_ep2_v2_temp1.0 | 1.0 | 0.9984 | 1.0 | 0.9968 | 0.0 | 99.25 | 0.9995 | 0.9968 | 90.8 |

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
| exp3_r32_lr2e-4_ep2_temp0.3 | 0.3 | 0.9043 | 0.8966 | 0.9123 | 3.51 | 60.0 | 0.9699 | 74.8 |
| exp3_r32_lr2e-4_ep2_temp0.7 | 0.7 | 0.9217 | 0.9138 | 0.9298 | 3.51 | 60.0 | 0.9719 | 74.5 |
| exp3_r32_lr2e-4_ep2_temp1.0 | 1.0 | 0.9138 | 0.8983 | 0.9298 | 3.51 | 66.67 | 0.9707 | 75.0 |
| exp4_r16_lr5e-5_ep2 | 0.0 | 0.9189 | 0.9444 | 0.8947 | 3.51 | 60.0 | 0.9531 | 74.7 |
| exp5_r16_lr2e-4_ep1 | 0.0 | 0.9217 | 0.9138 | 0.9298 | 5.26 | 73.33 | 0.9665 | 78.5 |
| exp6_r16_lr2e-4_ep2_v2 | 0.0 | 0.973 | 1.0 | 0.9474 | 3.51 | 83.33 | 0.9857 | 78.6 |
| exp6_r16_lr2e-4_ep2_v2_temp0.3 | 0.3 | 0.973 | 1.0 | 0.9474 | 3.51 | 80.0 | 0.9849 | 74.6 |
| exp6_r16_lr2e-4_ep2_v2_temp0.7 | 0.7 | 0.973 | 1.0 | 0.9474 | 3.51 | 86.67 | 0.986 | 74.2 |
| exp6_r16_lr2e-4_ep2_v2_temp1.0 | 1.0 | 0.973 | 1.0 | 0.9474 | 3.51 | 83.33 | 0.9803 | 74.0 |

## Avaliação — desafio 2, escrito antes do exp6 (25 textos)

Único conjunto independente de todos os treinos, inclusive do dataset v2.

| Modelo | temp. | F1 | Precision | Recall | Vazamento % | Exato % | Similaridade | tok/s |
|---|---|---|---|---|---|---|---|---|
| base | 0.0 | 0.3789 | 0.2432 | 0.8571 | 9.52 | 0.0 | 0.4616 | 73.2 |
| exp1_r8_lr2e-4_ep2 | 0.0 | 0.7907 | 0.7727 | 0.8095 | 11.9 | 48.0 | 0.8901 | 86.1 |
| exp2_r16_lr2e-4_ep2 | 0.0 | 0.8235 | 0.814 | 0.8333 | 11.9 | 48.0 | 0.8872 | 84.9 |
| exp3_r32_lr2e-4_ep2 | 0.0 | 0.7816 | 0.7556 | 0.8095 | 7.14 | 36.0 | 0.9004 | 85.5 |
| exp4_r16_lr5e-5_ep2 | 0.0 | 0.7952 | 0.8049 | 0.7857 | 14.29 | 44.0 | 0.8998 | 86.3 |
| exp5_r16_lr2e-4_ep1 | 0.0 | 0.7619 | 0.7619 | 0.7619 | 14.29 | 48.0 | 0.886 | 87.6 |
| exp6_r16_lr2e-4_ep2_v2 | 0.0 | 0.875 | 0.9211 | 0.8333 | 9.52 | 72.0 | 0.9583 | 81.2 |
| exp6_r16_lr2e-4_ep2_v2_temp0.3 | 0.3 | 0.875 | 0.9211 | 0.8333 | 9.52 | 72.0 | 0.9583 | 77.2 |
| exp6_r16_lr2e-4_ep2_v2_temp0.7 | 0.7 | 0.875 | 0.9211 | 0.8333 | 9.52 | 72.0 | 0.9583 | 76.9 |
| exp6_r16_lr2e-4_ep2_v2_temp1.0 | 1.0 | 0.875 | 0.9211 | 0.8333 | 9.52 | 72.0 | 0.9546 | 74.5 |

## Melhor modelo: `exp6_r16_lr2e-4_ep2_v2`

**F1 por tipo de dado — Teste sintético**

| Tipo | Precision | Recall | F1 | Suporte |
|---|---|---|---|---|
| NOME | 1.0 | 1.0 | 1.0 | 397 |
| CPF | 1.0 | 0.969 | 0.9843 | 129 |
| RG | 1.0 | 1.0 | 1.0 | 49 |
| TELEFONE | 1.0 | 1.0 | 1.0 | 157 |
| EMAIL | 1.0 | 1.0 | 1.0 | 89 |
| ENDERECO | 1.0 | 1.0 | 1.0 | 51 |
| CEP | 1.0 | 1.0 | 1.0 | 56 |

**F1 por tipo de dado — Desafio**

| Tipo | Precision | Recall | F1 | Suporte |
|---|---|---|---|---|
| NOME | 1.0 | 1.0 | 1.0 | 24 |
| CPF | 1.0 | 1.0 | 1.0 | 6 |
| RG | 1.0 | 0.75 | 0.8571 | 4 |
| TELEFONE | 1.0 | 1.0 | 1.0 | 7 |
| EMAIL | 1.0 | 0.8571 | 0.9231 | 7 |
| ENDERECO | 1.0 | 1.0 | 1.0 | 5 |
| CEP | 1.0 | 0.75 | 0.8571 | 4 |

**F1 por tipo de dado — Desafio 2**

| Tipo | Precision | Recall | F1 | Suporte |
|---|---|---|---|---|
| NOME | 0.9545 | 0.9545 | 0.9545 | 22 |
| CPF | 1.0 | 0.6667 | 0.8 | 3 |
| RG | 1.0 | 0.5 | 0.6667 | 2 |
| TELEFONE | 0.75 | 1.0 | 0.8571 | 3 |
| EMAIL | 1.0 | 0.75 | 0.8571 | 4 |
| ENDERECO | 0.8 | 0.8 | 0.8 | 5 |
| CEP | 1.0 | 0.3333 | 0.5 | 3 |

## Exemplos do desafio 2: base × ajustado

- **Entrada:** Aqui é a Rosângela, da contabilidade. O boleto da Metalúrgica Forte Ltda (CNPJ 33.104.556/0001-08) venceu ontem.
  - Base: [NOME], [CPF], [RG], [TELEFONE], [EMAIL], [ENDERECO], [CEP]
  - Ajustado: Aqui é a [NOME], da contabilidade. O boleto da Metalúrgica Forte Ltda (CNPJ 33.104.556/0001-08) venceu ontem.
  - Esperado: Aqui é a [NOME], da contabilidade. O boleto da Metalúrgica Forte Ltda (CNPJ 33.104.556/0001-08) venceu ontem.

- **Entrada:** Sr. Valdir Campos, 67 anos, aposentado, mora na Rua Vergueiro, 2045, casa 3 e pediu visita técnica.
  - Base: Sr. Valdir [NOME], [CPF], [RG], [TELEFONE], [EMAIL], [ENDERECO], [CEP].
  - Ajustado: Sr. [NOME], 67 anos, aposentado, mora na [ENDERECO] e pediu visita técnica.
  - Esperado: Sr. [NOME], 67 anos, aposentado, mora na [ENDERECO] e pediu visita técnica.

- **Entrada:** por favor encaminhar para a Dra. Simone Arantes o exame do paciente Wesley Rocha, quarto 12.
  - Base: [EXAME DO PACIENTE Wesley Rocha, QUARTO 12]
  - Ajustado: por favor encaminhar para a Dra. [NOME] o exame do paciente [NOME], quarto 12.
  - Esperado: por favor encaminhar para a Dra. [NOME] o exame do paciente [NOME], quarto 12.

- **Entrada:** Marcelo, 31 99102-7788, marcelo.ti@gmail.com
  - Base: [NOME], [CPF], [RG], [TELEFONE], [EMAIL], [ENDERECO], [CEP].
  - Ajustado: [NOME], [TELEFONE], [EMAIL]
  - Esperado: [NOME], [TELEFONE], [EMAIL]

- **Entrada:** Razão social: Supermercado Estrela do Norte. CNPJ: 08.552.761/0001-44. Endereço comercial: Av. Recife, 900.
  - Base: Razão social: [NOME], [CPF], [RG], [TELEFONE], [EMAIL], [ENDERECO], [CEP].
  - Ajustado: Razão social: Supermercado Estrela do Norte. CNPJ: 08.552.761/0001-44. Endereço comercial: [ENDERECO].
  - Esperado: Razão social: Supermercado Estrela do Norte. CNPJ: 08.552.761/0001-44. Endereço comercial: Av. Recife, 900.

- **Entrada:** Oi Júlia, tudo bem? O seu código de rastreamento é NX448201937BR e a previsão é sexta.
  - Base: Oi [NOME], [CPF], [RG], [TELEFONE], [EMAIL], [ENDERECO], [CEP]. O seu código de rastreamento é [NOME], [CPF], [RG], [TELEFONE], [EMAIL], [ENDERECO], [CEP]. O seu código de rastreamento é [NOME], [CPF], [RG], [TELEFONE], [EMAIL], [ENDERECO], [CEP].
  - Ajustado: Oi [NOME], tudo bem? O seu código de rastreamento é NX448201937BR e a previsão é sexta.
  - Esperado: Oi [NOME], tudo bem? O seu código de rastreamento é NX448201937BR e a previsão é sexta.

