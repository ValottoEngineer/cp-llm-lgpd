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
| exp7_r16_lr2e-4_ep2_v3 | 16 | 32 | 0.0002 | 2.0 | 0.0469 | 0.0002 | 38.4 | 4.31 |

## Avaliação — conjunto de teste sintético (400 exemplos)

Vazamento = % dos dados pessoais que continuam no texto (menor é melhor).

| Modelo | temp. | F1 | Precision | Recall | Vazamento % | Vaz. parcial % | Exato % | Similaridade | F1 templates inéditos | tok/s |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.0 | 0.4125 | 0.2717 | 0.8556 | 8.3 | 11.21 | 0.5 | 0.5037 | 0.3899 | 35.0 |
| exp1_r8_lr2e-4_ep2 | 0.0 | 0.9995 | 1.0 | 0.9989 | 0.0 | 0.0 | 99.5 | 0.9997 | 0.9989 | 94.4 |
| exp2_r16_lr2e-4_ep2 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 100.0 | 1.0 | 1.0 | 96.8 |
| exp2_r16_lr2e-4_ep2_temp0.3 | 0.3 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 100.0 | 1.0 | 1.0 | 91.5 |
| exp2_r16_lr2e-4_ep2_temp0.7 | 0.7 | 0.9995 | 1.0 | 0.9989 | 0.0 | 0.0 | 99.75 | 0.9998 | 0.9989 | 90.9 |
| exp2_r16_lr2e-4_ep2_temp1.0 | 1.0 | 0.9995 | 1.0 | 0.9989 | 0.0 | 0.0 | 99.75 | 0.9998 | 0.9989 | 91.9 |
| exp3_r32_lr2e-4_ep2 | 0.0 | 0.9995 | 1.0 | 0.9989 | 0.0 | 0.0 | 99.75 | 0.9998 | 0.9989 | 95.8 |
| exp3_r32_lr2e-4_ep2_temp0.3 | 0.3 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 100.0 | 1.0 | 1.0 | 90.0 |
| exp3_r32_lr2e-4_ep2_temp0.7 | 0.7 | 0.9984 | 0.9989 | 0.9978 | 0.0 | 0.0 | 99.5 | 0.9997 | 0.9968 | 90.1 |
| exp3_r32_lr2e-4_ep2_temp1.0 | 1.0 | 0.9989 | 1.0 | 0.9978 | 0.0 | 0.0 | 99.5 | 0.9997 | 0.9979 | 90.2 |
| exp4_r16_lr5e-5_ep2 | 0.0 | 0.9908 | 0.9989 | 0.9828 | 0.0 | 0.0 | 93.25 | 0.9964 | 0.9816 | 96.1 |
| exp5_r16_lr2e-4_ep1 | 0.0 | 0.9995 | 1.0 | 0.9989 | 0.0 | 0.0 | 99.75 | 0.9998 | 0.9989 | 95.8 |
| exp6_r16_lr2e-4_ep2_v2 | 0.0 | 0.9978 | 1.0 | 0.9957 | 0.0 | 0.0 | 99.0 | 0.9993 | 0.9957 | 95.2 |
| exp6_r16_lr2e-4_ep2_v2_temp0.3 | 0.3 | 0.9989 | 1.0 | 0.9978 | 0.0 | 0.0 | 99.5 | 0.9997 | 0.9979 | 90.0 |
| exp6_r16_lr2e-4_ep2_v2_temp0.7 | 0.7 | 0.9968 | 1.0 | 0.9935 | 0.0 | 0.0 | 98.5 | 0.999 | 0.9936 | 90.4 |
| exp6_r16_lr2e-4_ep2_v2_temp1.0 | 1.0 | 0.9984 | 1.0 | 0.9968 | 0.0 | 0.0 | 99.25 | 0.9995 | 0.9968 | 90.8 |
| exp7_r16_lr2e-4_ep2_v3 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 100.0 | 1.0 | 1.0 | 94.8 |
| exp7_r16_lr2e-4_ep2_v3_temp0.3 | 0.3 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 100.0 | 1.0 | 1.0 | 91.3 |
| exp7_r16_lr2e-4_ep2_v3_temp0.7 | 0.7 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 100.0 | 1.0 | 1.0 | 90.5 |
| exp7_r16_lr2e-4_ep2_v3_temp1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.11 | 0.11 | 99.5 | 0.9997 | 1.0 | 87.8 |

## Avaliação — conjunto de desafio escrito à mão (30 textos)

Textos com redação livre, fora dos templates do treino (`data/gerar_desafio.py`).

| Modelo | temp. | F1 | Precision | Recall | Vazamento % | Vaz. parcial % | Exato % | Similaridade | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.0 | 0.3 | 0.1852 | 0.7895 | 14.04 | 22.81 | 0.0 | 0.5693 | 34.2 |
| exp1_r8_lr2e-4_ep2 | 0.0 | 0.9381 | 0.9464 | 0.9298 | 3.51 | 3.51 | 73.33 | 0.9673 | 77.7 |
| exp2_r16_lr2e-4_ep2 | 0.0 | 0.9217 | 0.9138 | 0.9298 | 1.75 | 1.75 | 73.33 | 0.9591 | 75.8 |
| exp2_r16_lr2e-4_ep2_temp0.3 | 0.3 | 0.9217 | 0.9138 | 0.9298 | 3.51 | 3.51 | 70.0 | 0.9561 | 72.3 |
| exp2_r16_lr2e-4_ep2_temp0.7 | 0.7 | 0.9217 | 0.9138 | 0.9298 | 1.75 | 1.75 | 73.33 | 0.9602 | 69.9 |
| exp2_r16_lr2e-4_ep2_temp1.0 | 1.0 | 0.931 | 0.9153 | 0.9474 | 1.75 | 1.75 | 73.33 | 0.965 | 70.9 |
| exp3_r32_lr2e-4_ep2 | 0.0 | 0.9043 | 0.8966 | 0.9123 | 3.51 | 3.51 | 60.0 | 0.9699 | 80.9 |
| exp3_r32_lr2e-4_ep2_temp0.3 | 0.3 | 0.9043 | 0.8966 | 0.9123 | 3.51 | 3.51 | 60.0 | 0.9699 | 74.8 |
| exp3_r32_lr2e-4_ep2_temp0.7 | 0.7 | 0.9217 | 0.9138 | 0.9298 | 3.51 | 3.51 | 60.0 | 0.9719 | 74.5 |
| exp3_r32_lr2e-4_ep2_temp1.0 | 1.0 | 0.9138 | 0.8983 | 0.9298 | 3.51 | 3.51 | 66.67 | 0.9707 | 75.0 |
| exp4_r16_lr5e-5_ep2 | 0.0 | 0.9189 | 0.9444 | 0.8947 | 3.51 | 3.51 | 60.0 | 0.9531 | 74.7 |
| exp5_r16_lr2e-4_ep1 | 0.0 | 0.9217 | 0.9138 | 0.9298 | 5.26 | 7.02 | 73.33 | 0.9665 | 78.5 |
| exp6_r16_lr2e-4_ep2_v2 | 0.0 | 0.973 | 1.0 | 0.9474 | 3.51 | 5.26 | 83.33 | 0.9857 | 78.6 |
| exp6_r16_lr2e-4_ep2_v2_temp0.3 | 0.3 | 0.973 | 1.0 | 0.9474 | 3.51 | 5.26 | 80.0 | 0.9849 | 74.6 |
| exp6_r16_lr2e-4_ep2_v2_temp0.7 | 0.7 | 0.973 | 1.0 | 0.9474 | 3.51 | 5.26 | 86.67 | 0.986 | 74.2 |
| exp6_r16_lr2e-4_ep2_v2_temp1.0 | 1.0 | 0.973 | 1.0 | 0.9474 | 3.51 | 7.02 | 83.33 | 0.9803 | 74.0 |
| exp7_r16_lr2e-4_ep2_v3 | 0.0 | 0.9735 | 0.9821 | 0.9649 | 3.51 | 5.26 | 83.33 | 0.9828 | 78.7 |
| exp7_r16_lr2e-4_ep2_v3_temp0.3 | 0.3 | 0.9735 | 0.9821 | 0.9649 | 3.51 | 5.26 | 83.33 | 0.9828 | 74.9 |
| exp7_r16_lr2e-4_ep2_v3_temp0.7 | 0.7 | 0.9735 | 0.9821 | 0.9649 | 3.51 | 5.26 | 80.0 | 0.9827 | 74.9 |
| exp7_r16_lr2e-4_ep2_v3_temp1.0 | 1.0 | 0.9649 | 0.9649 | 0.9649 | 3.51 | 7.02 | 80.0 | 0.978 | 74.6 |

## Avaliação — desafio 2, escrito antes do exp6 (25 textos)

Único conjunto independente de todos os treinos, inclusive do dataset v2.

| Modelo | temp. | F1 | Precision | Recall | Vazamento % | Vaz. parcial % | Exato % | Similaridade | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.0 | 0.3789 | 0.2432 | 0.8571 | 9.52 | 16.67 | 0.0 | 0.4616 | 73.2 |
| exp1_r8_lr2e-4_ep2 | 0.0 | 0.7907 | 0.7727 | 0.8095 | 11.9 | 14.29 | 48.0 | 0.8901 | 86.1 |
| exp2_r16_lr2e-4_ep2 | 0.0 | 0.8235 | 0.814 | 0.8333 | 11.9 | 14.29 | 48.0 | 0.8872 | 84.9 |
| exp3_r32_lr2e-4_ep2 | 0.0 | 0.7816 | 0.7556 | 0.8095 | 7.14 | 9.52 | 36.0 | 0.9004 | 85.5 |
| exp4_r16_lr5e-5_ep2 | 0.0 | 0.7952 | 0.8049 | 0.7857 | 14.29 | 16.67 | 44.0 | 0.8998 | 86.3 |
| exp5_r16_lr2e-4_ep1 | 0.0 | 0.7619 | 0.7619 | 0.7619 | 14.29 | 16.67 | 48.0 | 0.886 | 87.6 |
| exp6_r16_lr2e-4_ep2_v2 | 0.0 | 0.875 | 0.9211 | 0.8333 | 9.52 | 9.52 | 72.0 | 0.9583 | 81.2 |
| exp6_r16_lr2e-4_ep2_v2_temp0.3 | 0.3 | 0.875 | 0.9211 | 0.8333 | 9.52 | 9.52 | 72.0 | 0.9583 | 77.2 |
| exp6_r16_lr2e-4_ep2_v2_temp0.7 | 0.7 | 0.875 | 0.9211 | 0.8333 | 9.52 | 9.52 | 72.0 | 0.9583 | 76.9 |
| exp6_r16_lr2e-4_ep2_v2_temp1.0 | 1.0 | 0.875 | 0.9211 | 0.8333 | 9.52 | 9.52 | 72.0 | 0.9546 | 74.5 |
| exp7_r16_lr2e-4_ep2_v3 | 0.0 | 0.9302 | 0.9091 | 0.9524 | 2.38 | 2.38 | 76.0 | 0.9712 | 82.4 |
| exp7_r16_lr2e-4_ep2_v3_temp0.3 | 0.3 | 0.9195 | 0.8889 | 0.9524 | 2.38 | 2.38 | 76.0 | 0.9625 | 73.1 |
| exp7_r16_lr2e-4_ep2_v3_temp0.7 | 0.7 | 0.9195 | 0.8889 | 0.9524 | 2.38 | 2.38 | 80.0 | 0.9701 | 75.4 |
| exp7_r16_lr2e-4_ep2_v3_temp1.0 | 1.0 | 0.9111 | 0.8542 | 0.9762 | 2.38 | 2.38 | 72.0 | 0.9578 | 71.9 |

## Avaliação — desafio 3, escrito antes do exp7 (25 textos)

Único conjunto independente de todos os treinos, inclusive do dataset v3.

| Modelo | temp. | F1 | Precision | Recall | Vazamento % | Vaz. parcial % | Exato % | Similaridade | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.0 | 0.3388 | 0.2168 | 0.775 | 7.5 | 17.5 | 0.0 | 0.5202 | 95.0 |
| exp1_r8_lr2e-4_ep2 | 0.0 | 0.8974 | 0.9211 | 0.875 | 7.5 | 10.0 | 60.0 | 0.9375 | 92.0 |
| exp2_r16_lr2e-4_ep2 | 0.0 | 0.8974 | 0.9211 | 0.875 | 7.5 | 10.0 | 52.0 | 0.9311 | 92.6 |
| exp3_r32_lr2e-4_ep2 | 0.0 | 0.8148 | 0.8049 | 0.825 | 12.5 | 17.5 | 28.0 | 0.9012 | 93.7 |
| exp4_r16_lr5e-5_ep2 | 0.0 | 0.8267 | 0.8857 | 0.775 | 15.0 | 17.5 | 36.0 | 0.9261 | 91.1 |
| exp5_r16_lr2e-4_ep1 | 0.0 | 0.8 | 0.8571 | 0.75 | 17.5 | 20.0 | 36.0 | 0.9286 | 97.6 |
| exp6_r16_lr2e-4_ep2_v2 | 0.0 | 0.9211 | 0.9722 | 0.875 | 12.5 | 15.0 | 76.0 | 0.9447 | 99.0 |
| exp6_r16_lr2e-4_ep2_v2_temp0.3 | 0.3 | 0.9211 | 0.9722 | 0.875 | 12.5 | 15.0 | 76.0 | 0.9447 | 92.6 |
| exp6_r16_lr2e-4_ep2_v2_temp0.7 | 0.7 | 0.9211 | 0.9722 | 0.875 | 12.5 | 15.0 | 76.0 | 0.9447 | 94.5 |
| exp6_r16_lr2e-4_ep2_v2_temp1.0 | 1.0 | 0.9211 | 0.9722 | 0.875 | 12.5 | 15.0 | 76.0 | 0.9464 | 93.7 |
| exp7_r16_lr2e-4_ep2_v3 | 0.0 | 0.878 | 0.8571 | 0.9 | 0.0 | 0.0 | 72.0 | 0.9805 | 95.2 |
| exp7_r16_lr2e-4_ep2_v3_temp0.3 | 0.3 | 0.878 | 0.8571 | 0.9 | 0.0 | 0.0 | 72.0 | 0.9805 | 82.5 |
| exp7_r16_lr2e-4_ep2_v3_temp0.7 | 0.7 | 0.9048 | 0.8636 | 0.95 | 0.0 | 0.0 | 72.0 | 0.9811 | 87.0 |
| exp7_r16_lr2e-4_ep2_v3_temp1.0 | 1.0 | 0.8642 | 0.8537 | 0.875 | 0.0 | 2.5 | 72.0 | 0.9693 | 86.2 |

## Melhor modelo: `exp7_r16_lr2e-4_ep2_v3`

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
| NOME | 1.0 | 1.0 | 1.0 | 24 |
| CPF | 1.0 | 1.0 | 1.0 | 6 |
| RG | 1.0 | 0.75 | 0.8571 | 4 |
| TELEFONE | 0.875 | 1.0 | 0.9333 | 7 |
| EMAIL | 1.0 | 0.8571 | 0.9231 | 7 |
| ENDERECO | 1.0 | 1.0 | 1.0 | 5 |
| CEP | 1.0 | 1.0 | 1.0 | 4 |

**F1 por tipo de dado — Desafio 2**

| Tipo | Precision | Recall | F1 | Suporte |
|---|---|---|---|---|
| NOME | 0.9545 | 0.9545 | 0.9545 | 22 |
| CPF | 0.75 | 1.0 | 0.8571 | 3 |
| RG | 1.0 | 1.0 | 1.0 | 2 |
| TELEFONE | 1.0 | 1.0 | 1.0 | 3 |
| EMAIL | 1.0 | 0.75 | 0.8571 | 4 |
| ENDERECO | 0.7143 | 1.0 | 0.8333 | 5 |
| CEP | 1.0 | 1.0 | 1.0 | 3 |

**F1 por tipo de dado — Desafio 3**

| Tipo | Precision | Recall | F1 | Suporte |
|---|---|---|---|---|
| NOME | 0.8636 | 1.0 | 0.9268 | 19 |
| CPF | 0.75 | 1.0 | 0.8571 | 3 |
| RG | 1.0 | 1.0 | 1.0 | 1 |
| TELEFONE | 1.0 | 0.8333 | 0.9091 | 6 |
| EMAIL | 1.0 | 0.25 | 0.4 | 4 |
| ENDERECO | 0.8 | 1.0 | 0.8889 | 4 |
| CEP | 0.75 | 1.0 | 0.8571 | 3 |

## Exemplos do desafio 3: base × ajustado

- **Entrada:** Pode fazer o Pix para a chave 52833071946? É da minha mãe, Neusa Ferraz.
  - Base: Pode fazer o Pix para a chave [NOME], [CPF], [RG], [TELEFONE], [EMAIL], [ENDERECO], [CEP]. É da minha mãe, Neusa Ferraz.
  - Ajustado: Pode fazer o Pix para a chave [CPF]? É da minha mãe, [NOME].
  - Esperado: Pode fazer o Pix para a chave [CPF]? É da minha mãe, [NOME].

- **Entrada:** Chave Pix: +5521987004411. Favorecido: Wagner Lopes. Valor: R$ 75,00.
  - Base: Chave Pix: [NOME], [CPF], [RG], [TELEFONE], [EMAIL], [ENDERECO], [CEP].
  - Ajustado: Chave Pix: [CPF]. Favorecido: [NOME]. Valor: R$ 75,00.
  - Esperado: Chave Pix: [TELEFONE]. Favorecido: [NOME]. Valor: R$ 75,00.

- **Entrada:** Mora no Residencial Parque das Árvores, torre 2, apto 1503, 06454-000, Barueri/SP.
  - Base: [NOME], [CPF], [RG], [TELEFONE], [EMAIL], [ENDERECO], [CEP].
  - Ajustado: Mora no [ENDERECO], [CEP], Barueri/SP.
  - Esperado: Mora no [ENDERECO], [CEP], Barueri/SP.

- **Entrada:** Ontem a Sueli veio buscar a encomenda, mas o pacote estava em nome do Henrique Bastos.
  - Base: Ontem [NOME] veio buscar a encomenda, mas o pacote estava em nome do [RG].
  - Ajustado: Ontem a [NOME] veio buscar a encomenda, mas o pacote estava em nome do [NOME].
  - Esperado: Ontem a [NOME] veio buscar a encomenda, mas o pacote estava em nome do [NOME].

- **Entrada:** O técnico da Vivo Empresas, CNPJ 02.558.157/0001-62, visitou o cliente Mauro Teles hoje.
  - Base: O técnico da Vivo Empresas, [NOME], [CPF], [RG], [TELEFONE], [EMAIL], [ENDERECO], [CEP], visitou o cliente Mauro Teles hoje.
  - Ajustado: O técnico da Vivo Empresas, CNPJ 02.558.157/0001-62, visitou o cliente [NOME] hoje.
  - Esperado: O técnico da Vivo Empresas, CNPJ 02.558.157/0001-62, visitou o cliente [NOME] hoje.

- **Entrada:** joana pediu pra avisar que o caio vai atrasar, liga pra ela no (62) 98144-3002
  - Base: [NOME] pediu pra avisar que o caio vai atrasar, liga pra ela no [62] 98144-3002.
  - Ajustado: [NOME] pediu pra avisar que o [NOME] vai atrasar, liga pra ela no [TELEFONE].
  - Esperado: [NOME] pediu pra avisar que o [NOME] vai atrasar, liga pra ela no [TELEFONE]

