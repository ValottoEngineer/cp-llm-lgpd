# Camada de segurança por regex

Pós-processamento determinístico (`src/regex_pii.py`) para CPF, e-mail, telefone, CEP e RG (com a palavra RG). Nomes e endereços dependem de contexto, por isso só a LLM os encontra.

## Desafio 3

| Abordagem | F1 | Precision | Recall | Vazamento % | Vazamento parcial % | Exato % |
|---|---|---|---|---|---|---|
| Só regex (sem LLM) | 0.5965 | 1.0 | 0.425 | 60.0 | 62.5 | 36.0 |
| Só LLM (exp2) | 0.8974 | 0.9211 | 0.875 | 7.5 | 10.0 | 52.0 |
| LLM (exp2) + regex | 0.925 | 0.925 | 0.925 | 5.0 | 7.5 | 56.0 |
| Só LLM (exp6) | 0.9211 | 0.9722 | 0.875 | 12.5 | 15.0 | 76.0 |
| LLM (exp6) + regex | 0.975 | 0.975 | 0.975 | 5.0 | 7.5 | 80.0 |
| Só LLM (exp7) | 0.878 | 0.8571 | 0.9 | 0.0 | 0.0 | 72.0 |
| LLM (exp7) + regex | 0.878 | 0.8571 | 0.9 | 0.0 | 0.0 | 72.0 |

## Desafio 2

| Abordagem | F1 | Precision | Recall | Vazamento % | Vazamento parcial % | Exato % |
|---|---|---|---|---|---|---|
| Só regex (sem LLM) | 0.5 | 1.0 | 0.3333 | 66.67 | 71.43 | 24.0 |
| Só LLM (exp2) | 0.8235 | 0.814 | 0.8333 | 11.9 | 14.29 | 48.0 |
| LLM (exp2) + regex | 0.8506 | 0.8222 | 0.881 | 7.14 | 9.52 | 52.0 |
| Só LLM (exp6) | 0.875 | 0.9211 | 0.8333 | 9.52 | 9.52 | 72.0 |
| LLM (exp6) + regex | 0.9024 | 0.925 | 0.881 | 4.76 | 4.76 | 76.0 |
| Só LLM (exp7) | 0.9302 | 0.9091 | 0.9524 | 2.38 | 2.38 | 76.0 |
| LLM (exp7) + regex | 0.9302 | 0.9091 | 0.9524 | 2.38 | 2.38 | 76.0 |

## Desafio 1

| Abordagem | F1 | Precision | Recall | Vazamento % | Vazamento parcial % | Exato % |
|---|---|---|---|---|---|---|
| Só regex (sem LLM) | 0.6265 | 1.0 | 0.4561 | 54.39 | 59.65 | 26.67 |
| Só LLM (exp2) | 0.9217 | 0.9138 | 0.9298 | 1.75 | 1.75 | 73.33 |
| LLM (exp2) + regex | 0.9217 | 0.9138 | 0.9298 | 1.75 | 1.75 | 73.33 |
| Só LLM (exp6) | 0.973 | 1.0 | 0.9474 | 3.51 | 5.26 | 83.33 |
| LLM (exp6) + regex | 0.9821 | 1.0 | 0.9649 | 1.75 | 3.51 | 86.67 |
| Só LLM (exp7) | 0.9735 | 0.9821 | 0.9649 | 3.51 | 5.26 | 83.33 |
| LLM (exp7) + regex | 0.9825 | 0.9825 | 0.9825 | 1.75 | 3.51 | 86.67 |

## Teste sintético

| Abordagem | F1 | Precision | Recall | Vazamento % | Vazamento parcial % | Exato % |
|---|---|---|---|---|---|---|
| Só regex (sem LLM) | 0.656 | 1.0 | 0.4881 | 51.19 | 53.88 | 15.75 |
| Só LLM (exp2) | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 100.0 |
| LLM (exp2) + regex | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 100.0 |
| Só LLM (exp6) | 0.9978 | 1.0 | 0.9957 | 0.0 | 0.0 | 99.0 |
| LLM (exp6) + regex | 0.9978 | 1.0 | 0.9957 | 0.0 | 0.0 | 99.0 |
| Só LLM (exp7) | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 100.0 |
| LLM (exp7) + regex | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 100.0 |

