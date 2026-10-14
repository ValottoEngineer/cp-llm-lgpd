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
| 3 | Testar diferentes parâmetros | [Experimentos](#3-experimentos-e-parâmetros) · `scripts/experimentos.py` |
| 4 | Avaliar o modelo (respostas) | [Avaliação](#4-avaliação) · `src/evaluate.py`, `src/metrics.py`, `results/` |
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

```bash
python data/gerar_dataset.py --n-train 2000 --n-val 300 --n-test 400
```

## 3. Experimentos e parâmetros

Definidos em `scripts/experimentos.py`. Os parâmetros variados são o rank do LoRA, a learning rate,
o número de épocas e, na geração, a temperatura.

| Experimento | rank (r) | alpha | learning rate | épocas |
|---|---|---|---|---|
| exp1 | 8 | 16 | 2e-4 | 2 |
| exp2 | 16 | 32 | 2e-4 | 2 |
| exp3 | 32 | 64 | 2e-4 | 2 |
| exp4 | 16 | 32 | 5e-5 | 2 |
| exp5 | 16 | 32 | 2e-4 | 1 |

Fixos: batch efetivo 16 (na GTX 1060: batch 4 × acumulação 4 com gradient checkpointing, escolhido
automaticamente para GPUs com menos de 8 GB), scheduler cosseno, warmup 5%, dropout 0,05,
comprimento máximo 384 tokens, seed 42. A perda é calculada **só sobre a resposta**: os tokens do prompt são mascarados.

O melhor adaptador é escolhido pelo **menor vazamento no conjunto de desafio**, com o F1 no desafio e depois o
F1 no teste como desempates. Ele é copiado para `outputs/melhor/` e avaliado também com temperaturas
0,3 / 0,7 / 1,0, para medir o efeito da amostragem.

## 4. Avaliação

Dois conjuntos de avaliação, com as mesmas métricas de `src/metrics.py`:

- **Teste sintético** (400 exemplos): metade com templates vistos no treino, metade com 8 templates inéditos.
- **Desafio escrito à mão** (30 textos, 57 dados pessoais; `data/gerar_desafio.py`): redação livre e textos longos, com várias pessoas no mesmo texto, formatos incomuns (`cpf: 398 114 027 61`, e-mail escrito "arroba", RG com UF) e distratores que **não** são dados pessoais (CNPJ, rastreio, protocolo, SKU, cupom). 6 dos textos não têm nenhum dado pessoal.

| Métrica | O que mede |
|---|---|

| Métrica | O que mede |
|---|---|
| **Vazamento %** | % dos dados pessoais originais que continuam no texto. É a mais importante para LGPD (menor é melhor). |
| Precision / Recall / F1 | Marcadores de cada tipo gerados × esperados, também por tipo de dado |
| Exato % | Respostas idênticas à referência |
| Similaridade | Quanto do texto não sensível foi preservado (0–1) |
| F1 em templates inéditos | Generalização para formatos não vistos no treino |

Também medimos o desempenho de geração (tokens/s) e o pico de VRAM no treino.

### Resultados

Todos os números foram medidos na **GTX 1060 6 GB** local, em fp32. As tabelas completas, incluindo
precision e recall por experimento, estão em [`results/resumo.md`](results/resumo.md), e as respostas de
cada modelo em `results/*.json`.

#### Treino

| Experimento | rank | alpha | lr | épocas | train loss | eval loss | tempo | pico VRAM |
|---|---|---|---|---|---|---|---|---|
| exp1 | 8 | 16 | 2e-4 | 2 | 0,0167 | 0,00010 | 42,3 min | 4,24 GB |
| **exp2** | **16** | **32** | **2e-4** | **2** | 0,0133 | 0,00007 | 36,5 min | 4,31 GB |
| exp3 | 32 | 64 | 2e-4 | 2 | 0,0112 | 0,00004 | 35,0 min | 4,46 GB |
| exp4 | 16 | 32 | 5e-5 | 2 | 0,0228 | 0,00024 | 34,2 min | 4,31 GB |
| exp5 | 16 | 32 | 2e-4 | 1 | 0,0224 | 0,00030 | 20,3 min | 4,29 GB |

![Curvas de loss](results/curvas_loss.png)

#### Avaliação (temperatura 0)

| Modelo | F1 teste | Vazamento teste | Exato teste | **F1 desafio** | **Vazamento desafio** | Exato desafio |
|---|---|---|---|---|---|---|
| Base (sem treino) | 0,412 | 8,3% | 0,5% | 0,300 | 14,0% | 0% |
| exp1 (r=8) | 0,9995 | 0% | 99,5% | **0,938** | 3,5% | 73,3% |
| **exp2 (r=16)** ⭐ | **1,000** | **0%** | **100%** | 0,922 | **1,75%** | 73,3% |
| exp3 (r=32) | 0,9995 | 0% | 99,75% | 0,904 | 3,5% | 60,0% |
| exp4 (lr=5e-5) | 0,991 | 0% | 93,25% | 0,919 | 3,5% | 60,0% |
| exp5 (1 época) | 0,9995 | 0% | 99,75% | 0,922 | 5,3% | 73,3% |

⭐ **Melhor modelo: exp2.** O critério é o **menor vazamento no desafio**, porque, para LGPD, deixar passar
um dado pessoal é o erro mais grave. O desempate é feito pelo F1 no desafio e depois pelo F1 no teste.

#### Parâmetros de geração (temperatura), com o exp2

| Temperatura | F1 teste | Exato teste | F1 desafio | Vazamento desafio | Exato desafio |
|---|---|---|---|---|---|
| 0 (greedy) | 1,000 | 100% | 0,922 | 1,75% | 73,3% |
| 0,3 | 1,000 | 100% | 0,922 | 3,51% | 70,0% |
| 0,7 | 0,9995 | 99,75% | 0,922 | 1,75% | 73,3% |
| 1,0 | 0,9995 | 99,75% | 0,931 | 1,75% | 73,3% |

#### Exemplos do desafio: base × ajustado

| Entrada | Modelo base | Modelo ajustado (exp2) |
|---|---|---|
| `cpf: 398 114 027 61 nome: Rodrigo Tavares — quero o reembolso ...` | `[cpf: 398 114 027 61] [RG: 123456789012345678] [TELEFONE: 1234567890] ...` ❌ não anonimiza e inventa dados | `cpf: [CPF] nome: [NOME] — quero o reembolso ...` ✅ |
| `Pode mandar a nota pro email financeiro.ramos@outlook.com? A empresa é Ramos & Filhos Ltda, CNPJ 12.345.678/0001-95.` | `Pode mandar a nota pro email [NOME], [CPF], [RG], [TELEFONE], [EMAIL], [ENDERECO], [CEP]. ...` ❌ | `Pode mandar a nota pro email [EMAIL]? A empresa é [NOME], CNPJ [CPF].` ⚠️ e-mail certo, mas empresa e CNPJ não são dados pessoais |
| `Kátia aqui, do RH. O Fábio Nunes pediu para alterar o telefone de emergência para (47) 98811-2093.` | `[KATEIA] aqui, do RH. [FABIO] Nunes pediu ... para [47] 98811-2093.` ❌ vaza sobrenome e telefone | `Kátia aqui, do RH. O [NOME] pediu ... para [TELEFONE].` ⚠️ o nome isolado no início escapou |

### Análise

1. **O fine-tuning resolve a tarefa.** O modelo base entende a instrução, mas inventa marcadores (precision de 0,27), reescreve o texto e deixa vazar 8–14% dos dados. Com LoRA, o F1 vai de **0,41 para 1,00** no teste e de **0,30 para 0,92** no desafio, e o vazamento cai de 14% para **1,75%**.
2. **O teste sintético satura.** Quatro das cinco configurações passam de 99,5% de acerto exato, inclusive nos templates inéditos. Por isso o desafio escrito à mão é necessário para diferenciar as configurações.
3. **Rank do LoRA.** Os ranks 8 e 16 generalizaram melhor que o 32. O exp3 tem a **menor loss de validação** e o **pior F1 no desafio**: com mais parâmetros, ele se ajustou demais ao estilo dos templates (*overfitting*). Loss de validação baixa não garante generalização.
4. **Learning rate.** Com 5e-5 (exp4), o modelo aprende mais devagar: tem a maior eval loss e o pior acerto exato no teste (93%). Em 2 épocas, 2e-4 foi melhor.
5. **Épocas.** Com 1 época (exp5), o resultado no teste é quase o mesmo e o treino leva metade do tempo, mas o vazamento no desafio é o maior (5,3%). A segunda época ajudou na robustez.
6. **Temperatura.** Ela quase não afeta o resultado: o modelo ficou muito confiante nessa tarefa, com uma distribuição de probabilidade concentrada. Para anonimização, usamos **temperatura 0**, que é determinística e reprodutível.
7. **Erros que restam no desafio:**
   - tratar CNPJ e nome de empresa como dado pessoal;
   - deixar escapar um nome sozinho no início da frase;
   - apagar palavras próximas aos dados (`54 anos`, `recepção`, `Dr.`);
   - inventar marcadores em listas compactas.

   Todos vêm de padrões ausentes do dataset sintético, e o caminho para corrigi-los é ampliar os templates.
8. **Desempenho.** A geração fica em ~95 tok/s em lote (avaliação) e ~13–22 tok/s numa requisição única no frontend. O treino usa no máximo 4,5 GB de VRAM com gradient checkpointing; sem ele, o pico chega a 8 GB e transborda para a RAM do sistema, deixando o treino 2× mais lento.

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

**Aba "Prompt livre".** Nesta aba aparece um efeito colateral do fine-tuning, o *catastrophic forgetting*. Em perguntas gerais, o modelo ajustado piora: inventa que a LGPD seria "Laws Against Discrimination" e repete itens. O modelo base responde corretamente.

Isso acontece porque o adaptador foi treinado só para anonimizar. Por isso o frontend permite desligar o LoRA ("Base (sem treino)"). Em produção, o adaptador seria usado apenas na tarefa de anonimização.

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
app/app.py                 frontend Gradio
notebooks/treino_colab.ipynb  treino no Colab
```

## Limitações e próximos passos

- **Dataset sintético.** O teste sintético satura (F1 ≈ 1,0), mas no desafio o F1 cai para ~0,92. Textos reais têm mais variações, como nomes raros, erros de digitação e dados em tabelas. Próximo passo: ampliar os templates com CNPJ e empresas (não pessoais), nomes isolados, títulos (Dr., Sr.) e listas compactas.
- **Ambiguidade.** Um nome sozinho ("Kátia", "Otávio") ou uma empresa com nome de pessoa são casos difíceis, e o modelo segue os padrões que viu no treino.
- **Catastrophic forgetting.** O adaptador piora o modelo em tarefas gerais (ver [Frontend](#6-frontend)). Isso é aceitável para um modelo especialista; para manter as duas capacidades, seria preciso misturar no treino exemplos de instruções gerais.
- **Omissões.** O modelo às vezes apaga palavras perto dos dados (`54 anos`, `recepção`). A métrica de similaridade capta isso parcialmente.
- **Tamanho do modelo.** O 0.6B foi escolhido pela restrição de hardware. Um modelo de 1.7B–4B, treinado com QLoRA no Colab (`notebooks/treino_colab.ipynb`), tende a generalizar melhor.
- **Escopo.** Apenas uma seed por configuração; não medimos a variância entre execuções.

## Referências

- Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685.
- Qwen Team (2025). *Qwen3 Technical Report*. Modelo: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (Apache 2.0).
- Brasil. *Lei nº 13.709/2018 — Lei Geral de Proteção de Dados Pessoais (LGPD)*, art. 5º, I: definição de dado pessoal.
- Hugging Face: [Transformers](https://huggingface.co/docs/transformers) e [PEFT](https://huggingface.co/docs/peft).
- PyTorch: [instalação com CUDA](https://pytorch.org/get-started/locally/).
- [Gradio](https://www.gradio.app/).
