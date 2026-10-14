# Anonimizador LGPD com LLM local (Qwen3-0.6B + LoRA)

CP2/CP3 — 4ESPV-2026

**Integrantes:**

| Nome | RM |
|---|---|
| Enzo Vazquez Sartorelli | RM94618 |
| Guilherme Brazioli | RM98237 |
| Kaue Pastori Teixeira | RM98501 |
| Nicolas Nogueira Boni | RM551965 |
| Pedro Oliveira Valotto | RM551445 |

Fine-tuning de uma LLM pequena para **detectar e anonimizar dados pessoais** em textos em português,
trocando-os por marcadores, com frontend para o usuário testar.

```
Entrada : Olá, meu nome é Fernanda Lima e meu CPF é 123.456.789-09. Quero cancelar o pedido #884512.
Saída   : Olá, meu nome é [NOME] e meu CPF é [CPF]. Quero cancelar o pedido #884512.
```

## Atendimento aos requisitos

| # | Requisito | Onde |
|---|---|---|
| 1 | Selecionar a LLM | [Escolha do modelo](#1-escolha-do-modelo) |
| 2 | Problema específico | [O problema](#2-o-problema) · `data/gerar_dataset.py` |
| 3 | Testar diferentes parâmetros | [Experimentos](#3-experimentos-e-parâmetros): 6 treinos (rank, learning rate, épocas, dataset v1/v2) + 3 temperaturas · `scripts/experimentos.py` |
| 4 | Avaliar o modelo (respostas) | [Avaliação](#4-avaliação): teste sintético + 2 desafios escritos à mão + teste de conhecimento geral · `src/evaluate.py`, `src/metrics.py`, `results/` |
| 5 | Testes utilizando GPU | Treino, avaliação e frontend na **GTX 1060 6 GB local** (CUDA 12.6) |
| 6 | Frontend para prompts | [Frontend](#6-frontend) · `app/app.py` |
| 7 | Print da GPU no VSCode | [GPU no VSCode](#7-gpu-funcionando-no-vscode) · `docs/` |
| 8 | Documentação e código no GitHub | este repositório |

## 1. Escolha do modelo

**Qwen3-0.6B** ([Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)), modelo da lista sugerida.

- **Hardware do grupo:** a GPU local é uma **GTX 1060 6 GB** (arquitetura Pascal, compute capability 6.1). Ela não acelera fp16/bf16, então treino e inferência locais rodam em fp32. Em fp32, o 0.6B ocupa ~2,4 GB e sobra memória para o LoRA. Modelos de 1–2B ficam no limite dos 6 GB.
- **Português:** o Qwen3 é multilíngue e entende bem PT-BR.
- **Suporte:** funciona direto no `transformers` e no `peft`, sem kernels especiais. Modelos com arquiteturas mais novas, como Qwen 3.5 e Gemma 4, podem depender de componentes que não rodam em Windows com GPU Pascal.
- **Técnica:** LoRA em todas as projeções de atenção e MLP. Só **1,67% dos parâmetros** são treinados (10,1 M de 606 M) com rank 16.

### Versão do CUDA

| Ambiente | GPU | Driver | PyTorch / CUDA | Precisão |
|---|---|---|---|---|
| Local | GTX 1060 6 GB (sm_61) | 581.57 | PyTorch **cu126** (CUDA 12.6) | fp32 |
| Colab | Tesla T4 16 GB (sm_75) | do Colab | PyTorch do Colab | fp16 misto |

O CUDA 13 e os builds do PyTorch com CUDA 12.8+ **não incluem mais GPUs Pascal**. Por isso usamos
o build **cu126**. O `scripts/gpu_check.py` confere se `sm_61` está entre as arquiteturas do PyTorch instalado.

## 2. O problema

**Anonimização de dados pessoais (LGPD).** Empresas precisam remover dados pessoais de mensagens de
SAC, cadastros e documentos antes de armazená-los, analisá-los ou compartilhá-los. O modelo deve:

- substituir **NOME, CPF, RG, TELEFONE, EMAIL, ENDERECO e CEP** pelos marcadores `[NOME]`, `[CPF]` etc.;
- **manter todo o resto igual**, incluindo números de pedido, protocolos, valores, datas e cidades, que funcionam como distratores.

### Dataset sintético (`data/gerar_dataset.py`)

Os dados são **100% sintéticos**, gerados sem bibliotecas externas e sem dados reais:

- 46 templates de contextos variados: SAC, RH, saúde, e-commerce, documentos e mensagens informais. Há também 8 templates sem nenhum dado pessoal, em que o modelo deve devolver o texto inalterado.
- CPFs com dígitos verificadores válidos, telefones em vários formatos (`(11) 98765-4321`, `+55 11 ...`, `11987654321`), e-mails derivados do nome, endereços com complemento e CEPs com e sem hífen.
- ~12% dos exemplos estão em estilo informal, em minúsculas.
- **Divisão:** 2000 treino / 300 validação / 400 teste. **8 templates ficam fora do treino** e aparecem só no teste, o que mede se o modelo generaliza ou só decorou os formatos.

**Versão 2 (`--v2`, pasta `data/v2/`).** Foi criada na segunda rodada a partir dos erros vistos no desafio 1. Adiciona 18 templates com dados pessoais e 4 sem dados pessoais (~30% do treino):

- CNPJ e nomes de empresa, que **não** são dados pessoais e devem ser mantidos;
- primeiro nome isolado ("Kátia aqui, do RH");
- títulos (Dr., Dra., Sr.);
- idade, leito, código de rastreio e SKU;
- listas sem rótulo (`Dados: [NOME], [CPF], [CEP].`).

O teste sintético continua o mesmo da versão 1, para manter a comparação.

```bash
python data/gerar_dataset.py
```
```bash
python data/gerar_dataset.py --v2 --out data/v2
```

## 3. Experimentos e parâmetros

Definidos em `scripts/experimentos.py`. Os parâmetros variados são o rank do LoRA, a learning rate,
o número de épocas, o dataset e, na geração, a temperatura. Foram duas rodadas:

| Experimento | rank (r) | alpha | learning rate | épocas | dataset | rodada |
|---|---|---|---|---|---|---|
| exp1 | 8 | 16 | 2e-4 | 2 | v1 | 1 |
| exp2 | 16 | 32 | 2e-4 | 2 | v1 | 1 |
| exp3 | 32 | 64 | 2e-4 | 2 | v1 | 1 |
| exp4 | 16 | 32 | 5e-5 | 2 | v1 | 1 |
| exp5 | 16 | 32 | 2e-4 | 1 | v1 | 1 |
| **exp6** | 16 | 32 | 2e-4 | 2 | **v2** | 2 |

Fixos: batch efetivo 16 (na GTX 1060: batch 4 × acumulação 4 com gradient checkpointing, escolhido
automaticamente para GPUs com menos de 8 GB), scheduler cosseno, warmup 5%, dropout 0,05,
comprimento máximo 384 tokens, seed 42. A perda é calculada **só sobre a resposta**: os tokens do prompt são mascarados.

**Rodada 1 → rodada 2.** A rodada 1 testou os hiperparâmetros. A melhor configuração (exp2) foi então retreinada
com o dataset v2 (exp6), para corrigir os erros encontrados no desafio 1.

## 4. Avaliação

Três conjuntos de avaliação, com as mesmas métricas de `src/metrics.py`:

- **Teste sintético** (400 exemplos): metade com templates vistos no treino, metade com 8 templates inéditos.
- **Desafio 1** (30 textos, 57 dados pessoais; `data/gerar_desafio.py`): escrito à mão, com redação livre e textos longos, várias pessoas no mesmo texto, formatos incomuns (`cpf: 398 114 027 61`, e-mail escrito "arroba", RG com UF) e distratores que **não** são dados pessoais (CNPJ, rastreio, protocolo, SKU, cupom). 6 textos não têm nenhum dado pessoal.
- **Desafio 2** (25 textos, 42 dados pessoais): escrito à mão **depois** da rodada 1 e **antes** de treinar o exp6, com frases novas que não foram copiadas dos templates v2 ([commit `0821085`](https://github.com/ValottoEngineer/cp-llm-lgpd/commit/0821085)). Como o dataset v2 foi inspirado nos erros do desafio 1, o desafio 1 deixou de ser independente para o exp6. **O desafio 2 é o único teste independente de todos os treinos.**

| Métrica | O que mede |
|---|---|
| **Vazamento %** | % dos dados pessoais originais que continuam inteiros no texto. É a métrica mais importante para LGPD (menor é melhor). |
| Precision / Recall / F1 | Marcadores de cada tipo gerados × esperados, também por tipo de dado |
| Exato % | Respostas idênticas à referência |
| Similaridade | Quanto do texto não sensível foi preservado (0–1) |
| F1 em templates inéditos | Generalização para formatos não vistos no treino |

Também medimos o desempenho de geração (tokens/s) e o pico de VRAM no treino.

### Resultados

Todos os números foram medidos na **GTX 1060 6 GB** local, em fp32. As tabelas completas, incluindo
precision e recall por experimento, estão em [`results/resumo.md`](results/resumo.md). As respostas de
cada modelo estão em `results/*_previsoes.jsonl`.

#### Treino

| Experimento | rank | alpha | lr | épocas | dataset | train loss | eval loss | tempo | pico VRAM |
|---|---|---|---|---|---|---|---|---|---|
| exp1 | 8 | 16 | 2e-4 | 2 | v1 | 0,0167 | 0,00010 | 42,3 min | 4,24 GB |
| exp2 | 16 | 32 | 2e-4 | 2 | v1 | 0,0133 | 0,00007 | 36,5 min | 4,31 GB |
| exp3 | 32 | 64 | 2e-4 | 2 | v1 | 0,0112 | 0,00004 | 35,0 min | 4,46 GB |
| exp4 | 16 | 32 | 5e-5 | 2 | v1 | 0,0228 | 0,00024 | 34,2 min | 4,31 GB |
| exp5 | 16 | 32 | 2e-4 | 1 | v1 | 0,0224 | 0,00030 | 20,3 min | 4,29 GB |
| **exp6** | 16 | 32 | 2e-4 | 2 | **v2** | 0,0164 | 0,00010 | 31,4 min | 4,28 GB |

![Curvas de loss](results/curvas_loss.png)

#### Avaliação (temperatura 0)

| Modelo | F1 teste | Exato teste | F1 desafio 1 | Vazam. desafio 1 | **F1 desafio 2** | **Exato desafio 2** | Vazam. desafio 2 |
|---|---|---|---|---|---|---|---|
| Base (sem treino) | 0,412 | 0,5% | 0,300 | 14,0% | 0,379 | 0% | 9,5% |
| exp1 (r=8) | 0,9995 | 99,5% | 0,938 | 3,5% | 0,791 | 48% | 11,9% |
| exp2 (r=16) | **1,000** | **100%** | 0,922 | **1,75%** | 0,824 | 48% | 11,9% |
| exp3 (r=32) | 0,9995 | 99,75% | 0,904 | 3,5% | 0,782 | 36% | **7,1%** |
| exp4 (lr=5e-5) | 0,991 | 93,25% | 0,919 | 3,5% | 0,795 | 44% | 14,3% |
| exp5 (1 época) | 0,9995 | 99,75% | 0,922 | 5,3% | 0,762 | 48% | 14,3% |
| **exp6 (v2)** ⭐ | 0,998 | 99,0% | 0,973¹ | 3,5%¹ | **0,875** | **72%** | 9,5% |

¹ O desafio 1 não é independente para o exp6, porque o dataset v2 foi criado a partir dos erros nele.

⭐ **Modelo final: exp6.** No desafio 2, que é o único independente, ele tem o melhor **F1 (0,875)**, o melhor
**acerto exato (72% contra 48%)**, a melhor **precision (0,92)** e a melhor **similaridade (0,958)**. No teste
sintético, mantém F1 de 0,998.

> **Sobre o critério de escolha (transparência).** O critério definido antes da rodada 2 era o *menor
> vazamento no desafio*. Por ele, o vencedor seria o exp3 (3 dados vazados contra 4 do exp6, num total de
> 42). Uma diferença de **um único dado** num conjunto tão pequeno não é estatisticamente significativa,
> enquanto a vantagem do exp6 em F1 (+0,09) e em acerto exato (+36 pontos) é grande e consistente. Por isso
> o critério passou a ser o **F1 no desafio 2**, que combina precision e recall, com o vazamento como
> desempate. Os resultados do exp3 estão mantidos em `results/` para conferência.

#### Parâmetros de geração (temperatura), com o exp6

| Temperatura | F1 teste | Exato teste | F1 desafio 1 | Exato desafio 1 | F1 desafio 2 | Exato desafio 2 |
|---|---|---|---|---|---|---|
| 0 (greedy) | 0,998 | 99,0% | 0,973 | 83,3% | 0,875 | 72% |
| 0,3 | 0,999 | 99,5% | 0,973 | 80,0% | 0,875 | 72% |
| 0,7 | 0,997 | 98,5% | 0,973 | 86,7% | 0,875 | 72% |
| 1,0 | 0,998 | 99,25% | 0,973 | 83,3% | 0,875 | 72% |

O mesmo teste foi feito com o exp2 e o exp3 na rodada 1, com o mesmo padrão: a variação entre temperaturas é
mínima (ver `results/resumo.md`).

#### Exemplos do desafio 2 (frases inéditas): base × rodada 1 × rodada 2

| Entrada | Base | exp2 (rodada 1) | exp6 (rodada 2) |
|---|---|---|---|
| `Aqui é a Rosângela, da contabilidade. O boleto da Metalúrgica Forte Ltda (CNPJ 33.104.556/0001-08) venceu ontem.` | `[NOME], [CPF], [RG], [TELEFONE], [EMAIL], [ENDERECO], [CEP]` ❌ | `Aqui é a [NOME], da [NOME]. O boleto da [CNPJ] venceu ontem.` ❌ | `Aqui é a [NOME], da contabilidade. O boleto da Metalúrgica Forte Ltda (CNPJ 33.104.556/0001-08) venceu ontem.` ✅ |
| `Dr. Paulo Henrique Sá, CRM 54321-SP, atende às terças na Clínica Bem Viver.` | `[NOME], [CPF], [RG], [TELEFONE], [EMAIL], [ENDERECO], [CEP].` ❌ | `Dr. [NOME], CRM [CRM], atende às terças na [ENDERECO].` ❌ | `Dr. [NOME], CRM 54321-SP, atende às terças na Clínica Bem Viver.` ✅ |
| `A loja Calçados Passo Firme ME (CNPJ 27.961.430/0001-61) solicitou parceria.` | `[A NOME] [CPF] [RG] [TELEFONE] [EMAIL] [ENDERECO] [CEP] solicitou parceria.` ❌ | `A loja Calçados Passo Firme ME (CNPJ [CNPJ]) solicitou parceria.` ❌ | texto mantido, sem alteração ✅ |
| `Sr. Valdir Campos, 67 anos, aposentado, mora na Rua Vergueiro, 2045, casa 3 ...` | `Sr. Valdir [NOME], [CPF], [RG], ...` ❌ | `Sr. [NOME], 67 anos, aposentado, mora na [ENDERECO] ...` ✅ | `Sr. [NOME], 67 anos, aposentado, mora na [ENDERECO] ...` ✅ |

Também no desafio 1, o exp6 passou a acertar `[NOME] aqui, do RH`, que o exp2 deixava escapar.

### Análise

1. **O fine-tuning resolve a tarefa.** O modelo base entende a instrução, mas inventa marcadores (precision de 0,24–0,27), reescreve o texto e deixa vazar 9–14% dos dados. Com LoRA, o F1 vai de **0,41 para ~1,00** no teste e de **0,38 para 0,88** no desafio independente.
2. **O teste sintético satura.** Todas as configurações passam de 0,99 de F1, inclusive nos templates inéditos. Por isso os desafios escritos à mão são necessários para diferenciar as configurações.
3. **Rank do LoRA.** O exp3 (r=32) tem a **menor loss de validação**, mas o **pior F1 e o pior acerto exato** nos dois desafios. Com mais parâmetros, ele se ajustou demais ao estilo dos templates (*overfitting*). Loss de validação baixa não garante generalização. Entre r=8 e r=16, o r=16 foi melhor no desafio 2.
4. **Learning rate.** Com 5e-5 (exp4), o modelo aprende mais devagar: tem a maior eval loss e o pior acerto exato no teste (93%). Em 2 épocas, 2e-4 foi melhor.
5. **Épocas.** Com 1 época (exp5), o treino leva metade do tempo e o teste fica quase igual, mas o resultado nos desafios é o pior (F1 de 0,76 no desafio 2). A segunda época ajudou na robustez.
6. **Temperatura.** Ela quase não afeta o resultado: o modelo ficou muito confiante nessa tarefa, com uma distribuição de probabilidade concentrada. Para anonimização, usamos **temperatura 0**, que é determinística e reprodutível.
7. **Dados > hiperparâmetros.** A maior melhoria não veio de ajustar hiperparâmetros (rodada 1: F1 de 0,76–0,82 no desafio 2), e sim de **melhorar os dados** (rodada 2: **0,875**, com acerto exato de 48% para 72%). O exp6 aprendeu a manter CNPJ, empresas, setores ("da contabilidade") e registros profissionais (CRM), inclusive em frases que nunca viu.
8. **Erros que restam no desafio 2:**
   - chave Pix escrita como CPF sem pontuação (`41207733860`);
   - endereço de condomínio sem "Rua/Av." (`Condomínio Jardim das Rosas, bloco 4`);
   - um nome isolado no meio da frase ("A Priscila ligou");
   - troca de tipo (RG marcado como `[TELEFONE]`).
9. **Desempenho.** A geração fica em ~95 tok/s em lote (avaliação) e ~13–22 tok/s numa requisição única no frontend. O treino usa no máximo 4,5 GB de VRAM com gradient checkpointing; sem ele, o pico chega a 8 GB e transborda para a RAM do sistema, deixando o treino 2× mais lento.

## 5. Como executar

### Local (Windows + GTX 1060): o ambiente usado no projeto

```bash
python -m venv .venv
```
```bash
.venv\Scripts\activate
```
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```
```bash
pip install -r requirements.txt
```
```bash
python scripts/gpu_check.py
```

O adaptador do melhor modelo já vem no repositório (`outputs/melhor/`, ~40 MB), então o frontend roda logo
após o clone:

```bash
python app/app.py
```

Reproduzir todos os experimentos (~3h30 na GTX 1060). Os que já terminaram são pulados:

```bash
python data/gerar_dataset.py
```
```bash
python scripts/experimentos.py
```

Treinar ou avaliar uma configuração isolada:

```bash
python src/train.py --out outputs/meu_teste --r 16 --alpha 32 --lr 2e-4 --epochs 2 --batch 4 --grad-accum 4 --grad-checkpointing
```
```bash
python src/evaluate.py --nome meu_teste --adapter outputs/meu_teste
```
```bash
python src/evaluate.py --nome desafio_meu_teste --adapter outputs/meu_teste --data data/desafio.jsonl
```

### Alternativa: Google Colab (GPU T4)

Para quem não tem GPU NVIDIA:

1. Abra `notebooks/treino_colab.ipynb` no Colab e escolha o ambiente com **GPU T4**.
2. Preencha `REPO_URL` com o link deste repositório e rode as células.
3. No final, baixe `resultados_colab.zip` e descompacte na raiz do projeto local. Ele traz `outputs/melhor/` e `results/`.

## 6. Frontend

Interface em **Gradio** (`app/app.py`), com duas abas:

**Anonimizar texto**
- campo para o usuário digitar ou colar o texto, com exemplos prontos;
- dados anonimizados destacados por cor e tipo.

**Prompt livre**
- o usuário escreve **qualquer prompt** e, se quiser, uma instrução de sistema;
- a resposta do modelo aparece na tela;
- serve também para mostrar que o modelo continua respondendo outras tarefas depois do fine-tuning.

**Nas duas abas**
- escolha entre **modelo ajustado**, **modelo base** ou **comparar os dois lado a lado**. O mesmo modelo fica na memória e o adaptador LoRA é ligado ou desligado;
- controle de **temperatura** e de **máximo de tokens**;
- tempo de resposta, tokens/s e VRAM em uso.

```bash
python app/app.py
```

Abre em http://127.0.0.1:7860.

**Aba "Anonimizar texto".** O modelo base inventa `[RG]`, `[ENDERECO]` e `[CEP]` que não existem no texto; o ajustado marca exatamente os dados presentes e mantém o número do pedido.

![Frontend: anonimizar](docs/frontend_anonimizar.png)

**Aba "Prompt livre".** O usuário faz qualquer pergunta e compara as respostas do modelo ajustado e do modelo base.

Esta aba também serve para investigar um efeito colateral do fine-tuning, o *catastrophic forgetting*. Para medi-lo de forma controlada, o script `scripts/conhecimento_geral.py` faz 5 perguntas gerais com temperatura 0 (resultado em [`results/conhecimento_geral.md`](results/conhecimento_geral.md)):

| Pergunta | Base | exp2 (rodada 1) | exp6 (final) |
|---|---|---|---|
| Qual é a capital do Brasil? / Quanto é 15 × 4? | ✅ | ✅ | ✅ |
| Explique o que é um CPF | ✅ | ✅ | ⚠️ "pessoa jurídica ou física" |
| O que significa a sigla LGPD? | ✅ Lei Geral de Proteção de Dados | ❌ inventa ("crime de violência de acusação") | ❌ inventa ("República da Argentina") |
| Escreva uma frase de bom dia a um cliente | ✅ | ❌ **copia o próprio prompt** | ✅ |

A degradação é **parcial**:

- os fatos simples continuam corretos;
- piora justamente o tema próximo ao do treino (LGPD);
- o exp2 aprendeu a **copiar a entrada**, que é o comportamento treinado na anonimização (devolver o mesmo texto). Os dados mais variados do v2 reduziram esse efeito no exp6.

Por isso o frontend permite desligar o LoRA ("Base (sem treino)"). Em produção, o adaptador seria usado apenas na tarefa de anonimização. Uma resposta com amostragem (temperatura 0,7), como no print abaixo, varia a cada execução; a tabela acima, com temperatura 0, é a medida reprodutível.

![Frontend: prompt livre](docs/frontend_prompt_livre.png)

## 7. GPU funcionando no VSCode

Extensões utilizadas: **GPU Monitor** (`Maimonator.gpu-monitor`) e **GPU Environments** (`run-ai.vscode-genv`), recomendadas em `.vscode/extensions.json`.

![GPU no VSCode](docs/gpu_vscode.png)

Print tirado **durante o treino** (14/10/2026, 11h47): o terminal do VSCode roda `scripts/gpu_check.py`
(Python 3.12, PyTorch 2.14.1+cu126, CUDA 12.6, `CUDA ativa: True`, GTX 1060 6GB com compute capability 6.1 `sm_61`, matmul fp32 a 3,16 TFLOPS), o `nvidia-smi` mostra a GTX 1060 com 81% de uso e 5,9 GB
de VRAM ocupados pelo treino, e a extensão **GPU Monitor**, na barra de status, indica
`NVIDIA GeForce GTX 1060 6GB: 5930 MiB / 6144 MiB, 84°C`.

## Estrutura

```
data/gerar_dataset.py      gera train/val/test.jsonl
data/gerar_desafio.py      gera desafio.jsonl (30 textos escritos à mão)
src/common.py              prompt, carregamento do modelo, geração
src/train.py               fine-tuning LoRA
src/evaluate.py            avaliação no teste -> results/<nome>.json
src/metrics.py             métricas (F1, vazamento, exato, similaridade)
scripts/experimentos.py    roda todos os experimentos
scripts/resumo.py          gera results/resumo.md e curvas_loss.png
scripts/gpu_check.py       informações e teste da GPU
scripts/conhecimento_geral.py  teste de catastrophic forgetting (perguntas gerais)
app/app.py                 frontend Gradio
notebooks/treino_colab.ipynb  treino no Colab
```

## Limitações e próximos passos

- **Dataset sintético.** O teste sintético satura (F1 ≈ 1,0), mas nos desafios escritos à mão o F1 fica entre 0,88 e 0,97. Textos reais têm mais variações, como nomes raros, erros de digitação e dados em tabelas. O dataset v2 mostrou que ampliar os templates é o caminho mais eficaz. Próximo passo: chaves Pix, endereços sem "Rua/Av." (condomínios) e nomes isolados no meio da frase.
- **Ambiguidade.** Um nome sozinho ("Kátia", "Otávio") ou uma empresa com nome de pessoa são casos difíceis, e o modelo segue os padrões que viu no treino.
- **Catastrophic forgetting.** O adaptador piora parcialmente as respostas gerais (ver [Frontend](#6-frontend)). Isso é aceitável para um modelo especialista; para manter as duas capacidades, seria preciso misturar no treino exemplos de instruções gerais.
- **Omissões.** O modelo às vezes apaga palavras perto dos dados (`54 anos`, `recepção`). A métrica de similaridade capta isso parcialmente.
- **Tamanho do modelo.** O 0.6B foi escolhido pela restrição de hardware. Um modelo de 1.7B–4B, treinado com QLoRA no Colab (`notebooks/treino_colab.ipynb`), tende a generalizar melhor.
- **Métrica de vazamento.** Ela só detecta o valor **inteiro**. Um vazamento parcial, como `financeiro.[EMAIL]` (o exp6, no e-mail `financeiro.ramos@outlook.com`), não é contado, embora as métricas de acerto exato e similaridade penalizem esse caso.
- **Desafios pequenos e rótulos subjetivos.** São 30 + 25 textos (57 + 42 dados pessoais), então diferenças de 1 ou 2 dados não são significativas. Alguns rótulos são discutíveis: o endereço comercial de uma empresa foi rotulado como não pessoal, e o modelo o anonimizou.
- **Escopo.** Apenas uma seed por configuração; não medimos a variância entre execuções.

## Referências

- Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685.
- Qwen Team (2025). *Qwen3 Technical Report*. Modelo: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (Apache 2.0).
- Brasil. *Lei nº 13.709/2018 — Lei Geral de Proteção de Dados Pessoais (LGPD)*, art. 5º, I: definição de dado pessoal.
- Hugging Face: [Transformers](https://huggingface.co/docs/transformers) e [PEFT](https://huggingface.co/docs/peft).
- PyTorch: [instalação com CUDA](https://pytorch.org/get-started/locally/).
- [Gradio](https://www.gradio.app/).
