"""
Frontend (Gradio) para testar o anonimizador.

    python app/app.py                         # usa outputs/melhor (LoRA fundido ao modelo)
    python app/app.py --adapter outputs/exp2_r16_lr2e-4_ep2
    python app/app.py --sem-merge             # mantém o LoRA separado (mais lento)

Abre em http://127.0.0.1:7860

Abas:
- Anonimizar texto: texto -> versão anonimizada, com dados destacados, tabela dos dados encontrados,
  camada de segurança por regex e comparação com o modelo base.
- Arquivo / lote: anonimiza um .txt (uma mensagem por linha) ou uma coluna de um .csv e devolve o arquivo.
- Prompt livre: qualquer prompt, com resposta do modelo ajustado e/ou do base.
"""
import argparse
import csv
import io
import sys
import tempfile
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import gradio as gr
import torch

from common import MARCADOR_RE, MODELO_BASE, anonimizar, carregar_modelo, gerar_livre, info_gpu, modo_base
from regex_pii import aplicar_regex, extrair_dados

EXEMPLOS = [
    "Olá, meu nome é Fernanda Lima Rocha e meu CPF é 123.456.789-09. Quero cancelar o pedido #884512.",
    "oi, aqui é o joão pedro, me chama no zap (11) 98765-4321 ou manda e-mail pra jp.souza@gmail.com",
    "Entregar na Rua Augusta, 1500, apto 42, São Paulo/SP, CEP 01304-001, aos cuidados de Marcos Alves.",
    "Paciente Beatriz Santos, RG 12.345.678-9, consulta em 20/10 às 14h30. Valor: R$ 350,00.",
    "A nota sai em nome da Padaria Pão Dourado, CNPJ 12.345.678/0001-95. Contato: Rafael, (21) 99876-1234.",
    "Qual o prazo de entrega para Curitiba? O pedido 552190 foi pago em 10/10.",
]
CORES = {"NOME": "#f59e0b", "CPF": "#ef4444", "RG": "#ec4899", "TELEFONE": "#3b82f6",
         "EMAIL": "#8b5cf6", "ENDERECO": "#10b981", "CEP": "#14b8a6"}
MODOS = ["Ajustado (LoRA)", "Base (sem treino)", "Comparar"]


def destacar(texto):
    partes, pos = [], 0
    for m in MARCADOR_RE.finditer(texto):
        if m.start() > pos:
            partes.append((texto[pos:m.start()], None))
        partes.append((m.group(0), m.group(1)))
        pos = m.end()
    if pos < len(texto):
        partes.append((texto[pos:], None))
    return partes


def ler_arquivo(caminho, coluna):
    """Devolve (linhas de texto, função que monta o arquivo de saída, extensão)."""
    bruto = Path(caminho).read_bytes().decode("utf-8-sig", errors="replace")
    if str(caminho).lower().endswith(".csv"):
        dialeto = csv.Sniffer().sniff(bruto[:2000], delimiters=",;\t")
        linhas = list(csv.DictReader(io.StringIO(bruto), dialect=dialeto))
        if not linhas:
            raise gr.Error("CSV vazio.")
        col = coluna.strip() or next(iter(linhas[0]))
        if col not in linhas[0]:
            raise gr.Error(f"Coluna '{col}' não encontrada. Colunas: {', '.join(linhas[0])}")
        textos = [l[col] or "" for l in linhas]

        def montar(saidas):
            buf = io.StringIO()
            campos = list(linhas[0]) + [f"{col}_anonimizado"]
            w = csv.DictWriter(buf, fieldnames=campos, delimiter=dialeto.delimiter)
            w.writeheader()
            for l, s in zip(linhas, saidas):
                w.writerow({**l, f"{col}_anonimizado": s})
            return buf.getvalue()
        return textos, montar, ".csv"
    textos = [l for l in bruto.splitlines() if l.strip()]
    return textos, lambda saidas: "\n".join(saidas) + "\n", ".txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODELO_BASE)
    ap.add_argument("--adapter", default=str(RAIZ / "outputs" / "melhor"))
    ap.add_argument("--sem-merge", action="store_true", help="não funde o LoRA ao modelo")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    tem_adapter = Path(args.adapter).exists()
    if not tem_adapter:
        print(f"Adaptador não encontrado em {args.adapter}: usando só o modelo base.")
    model, tok = carregar_modelo(args.model, args.adapter if tem_adapter else None, merge=not args.sem_merge)
    gpu = info_gpu()
    origem = Path(args.adapter, "ORIGEM.txt")
    nome_adapter = origem.read_text(encoding="utf-8").strip() if origem.exists() else Path(args.adapter).name

    def alvos(modo):
        """Quais modelos rodar: [(chave, rótulo, usar_base)]."""
        if not tem_adapter:
            return [("base", "Base", True)]
        return ([("ajust", "Ajustado", False)] if modo in (MODOS[0], MODOS[2]) else []) + \
               ([("base", "Base", True)] if modo in (MODOS[1], MODOS[2]) else [])

    def rodar(fn, usar_base):
        t0 = time.time()
        if usar_base and tem_adapter:
            with modo_base(model):
                r = fn()
        else:
            r = fn()
        return r, time.time() - t0

    def rodape(infos):
        if torch.cuda.is_available():
            infos.append(f"VRAM em uso: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        return " | ".join(infos)

    # ---------- Aba: anonimizar texto ----------
    def processar(texto, modo, temperatura, max_tokens, usar_regex):
        if not texto.strip():
            return [], [], "", [], "Digite um texto."
        res, infos = {"ajust": "", "base": ""}, []
        for chave, rotulo, usar_base in alvos(modo):
            (s, n), dt = rodar(lambda: anonimizar(model, tok, [texto], int(max_tokens), temperatura), usar_base)
            s = s[0]
            if usar_regex:
                s = aplicar_regex(s)
            res[chave] = s
            infos.append(f"**{rotulo}:** {dt:.2f}s · {n} tokens · {n / dt:.1f} tok/s")
        principal = res["ajust"] or res["base"]
        dados = [[t, v] for t, v in extrair_dados(texto, principal, MARCADOR_RE)]
        return destacar(res["ajust"]), destacar(res["base"]), principal, dados, rodape(infos)

    # ---------- Aba: arquivo / lote ----------
    def processar_lote(arquivo, coluna, usar_regex, progresso=gr.Progress()):
        if not arquivo:
            raise gr.Error("Envie um arquivo .txt ou .csv.")
        textos, montar, ext = ler_arquivo(arquivo, coluna or "")
        if len(textos) > 500:
            raise gr.Error(f"O arquivo tem {len(textos)} linhas; o limite é 500.")
        saidas, t0 = [], time.time()
        for i in progresso.tqdm(range(0, len(textos), 8), desc="Anonimizando"):
            lote = textos[i:i + 8]
            idx = [j for j, t in enumerate(lote) if t.strip()]
            geradas, _ = anonimizar(model, tok, [lote[j] for j in idx], 256, 0.0) if idx else ([], 0)
            out = list(lote)
            for j, g in zip(idx, geradas):
                out[j] = aplicar_regex(g) if usar_regex else g
            saidas += out
        dt = time.time() - t0
        destino = Path(tempfile.mkdtemp()) / f"{Path(arquivo).stem}_anonimizado{ext}"
        destino.write_text(montar(saidas), encoding="utf-8-sig" if ext == ".csv" else "utf-8")
        previa = [[a, b] for a, b in list(zip(textos, saidas))[:20]]
        info = rodape([f"**{len(textos)} textos** em {dt:.1f}s ({dt / max(len(textos), 1):.2f}s por texto)"])
        return previa, str(destino), info

    # ---------- Aba: prompt livre ----------
    def livre(prompt, system, modo, temperatura, max_tokens):
        if not prompt.strip():
            return "", "", "Digite um prompt."
        res, infos = {"ajust": "", "base": ""}, []
        for chave, rotulo, usar_base in alvos(modo):
            (r, n), dt = rodar(lambda: gerar_livre(model, tok, prompt, system or None, int(max_tokens), temperatura),
                               usar_base)
            res[chave] = r
            infos.append(f"**{rotulo}:** {dt:.2f}s · {n} tokens · {n / dt:.1f} tok/s")
        return res["ajust"], res["base"], rodape(infos)

    def controles(temp_padrao):
        modo = gr.Radio(MODOS, value="Comparar", label="Modelo")
        temperatura = gr.Slider(0, 1.5, value=temp_padrao, step=0.1, label="Temperatura (0 = determinístico)")
        max_tokens = gr.Slider(32, 1024, value=256, step=16, label="Máx. tokens gerados")
        return modo, temperatura, max_tokens

    with gr.Blocks(title="Anonimizador LGPD") as demo:
        gr.Markdown(
            "# Anonimizador LGPD com LLM local\n"
            f"Modelo **{args.model}** + LoRA **{nome_adapter if tem_adapter else '(nenhum)'}**"
            f"{' (fundido)' if tem_adapter and not args.sem_merge else ''} · "
            f"GPU: **{gpu.get('gpu')}** · precisão {gpu.get('precisao', 'fp32')}"
        )
        with gr.Tab("Anonimizar texto"):
            with gr.Row():
                with gr.Column(scale=3):
                    entrada = gr.Textbox(label="Texto com dados pessoais", lines=6,
                                         placeholder="Cole aqui um e-mail, mensagem, cadastro...")
                    gr.Examples(EXEMPLOS, inputs=entrada)
                with gr.Column(scale=1):
                    modo, temperatura, max_tokens = controles(0)
                    usar_regex = gr.Checkbox(value=True, label="Camada de segurança (regex para CPF, e-mail, telefone, CEP)")
                    botao = gr.Button("Anonimizar", variant="primary")
            with gr.Row():
                saida_ajust = gr.HighlightedText(label="Modelo ajustado (LoRA)", color_map=CORES,
                                                 show_inline_category=False)
                saida_base = gr.HighlightedText(label="Modelo base", color_map=CORES, show_inline_category=False)
            with gr.Row():
                copiar = gr.Textbox(label="Texto anonimizado (para copiar)", lines=3, show_copy_button=True, scale=3)
                dados = gr.Dataframe(headers=["Tipo", "Valor original"], label="Dados pessoais encontrados",
                                     interactive=False, scale=2)
            info = gr.Markdown()
            entradas = [entrada, modo, temperatura, max_tokens, usar_regex]
            saidas = [saida_ajust, saida_base, copiar, dados, info]
            botao.click(processar, entradas, saidas)
            entrada.submit(processar, entradas, saidas)

        with gr.Tab("Arquivo / lote"):
            gr.Markdown("Envie um **.txt** (uma mensagem por linha) ou um **.csv** (escolha a coluna de texto). "
                        "O resultado é anonimizado com o modelo ajustado, temperatura 0. Limite: 500 linhas.")
            with gr.Row():
                with gr.Column(scale=2):
                    arquivo = gr.File(label="Arquivo", file_types=[".txt", ".csv"], type="filepath")
                with gr.Column(scale=1):
                    coluna = gr.Textbox(label="Coluna do CSV (vazio = primeira coluna)")
                    regex_lote = gr.Checkbox(value=True, label="Camada de segurança (regex)")
                    botao_lote = gr.Button("Anonimizar arquivo", variant="primary")
            previa = gr.Dataframe(headers=["Original", "Anonimizado"], label="Prévia (20 primeiras linhas)",
                                  interactive=False, wrap=True)
            baixar = gr.File(label="Arquivo anonimizado")
            info_lote = gr.Markdown()
            botao_lote.click(processar_lote, [arquivo, coluna, regex_lote], [previa, baixar, info_lote])

        with gr.Tab("Prompt livre"):
            gr.Markdown("Escreva qualquer prompt e veja a resposta do modelo ajustado e/ou do modelo base.")
            with gr.Row():
                with gr.Column(scale=3):
                    prompt = gr.Textbox(label="Prompt", lines=5,
                                        placeholder="Ex.: Explique em 3 frases o que é a LGPD.")
                    system = gr.Textbox(label="Instrução de sistema (opcional)", lines=2,
                                        placeholder="Ex.: Responda sempre em português, de forma objetiva.")
                    gr.Examples([
                        ["Explique em 3 frases o que é a LGPD.", ""],
                        ["Quais dados são considerados pessoais pela LGPD? Responda em tópicos.", ""],
                        ["Anonimize: meu nome é Carla Dias e meu e-mail é carla.dias@gmail.com", ""],
                        ["Escreva um e-mail curto pedindo a segunda via de um boleto.",
                         "Você é um assistente de atendimento educado."],
                    ], inputs=[prompt, system])
                with gr.Column(scale=1):
                    modo_l, temperatura_l, max_tokens_l = controles(0.7)
                    botao_l = gr.Button("Enviar prompt", variant="primary")
            with gr.Row():
                resp_ajust = gr.Textbox(label="Resposta — modelo ajustado (LoRA)", lines=10, show_copy_button=True)
                resp_base = gr.Textbox(label="Resposta — modelo base", lines=10, show_copy_button=True)
            info_l = gr.Markdown()
            botao_l.click(livre, [prompt, system, modo_l, temperatura_l, max_tokens_l], [resp_ajust, resp_base, info_l])

    demo.launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
