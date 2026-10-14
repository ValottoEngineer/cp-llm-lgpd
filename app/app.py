"""
Frontend (Gradio) para testar o anonimizador.

    python app/app.py                         # usa outputs/melhor
    python app/app.py --adapter outputs/exp2_r16_lr2e-4_ep2

Abre em http://127.0.0.1:7860
"""
import argparse
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import gradio as gr
import torch

from common import MARCADOR_RE, MODELO_BASE, anonimizar, carregar_modelo, gerar_livre, info_gpu

EXEMPLOS = [
    "Olá, meu nome é Fernanda Lima Rocha e meu CPF é 123.456.789-09. Quero cancelar o pedido #884512.",
    "oi, aqui é o joão pedro, me chama no zap (11) 98765-4321 ou manda e-mail pra jp.souza@gmail.com",
    "Entregar na Rua Augusta, 1500, apto 42, São Paulo/SP, CEP 01304-001, aos cuidados de Marcos Alves.",
    "Paciente Beatriz Santos, RG 12.345.678-9, consulta em 20/10 às 14h30. Valor: R$ 350,00.",
    "Qual o prazo de entrega para Curitiba? O pedido 552190 foi pago em 10/10.",
]
CORES = {"NOME": "#f59e0b", "CPF": "#ef4444", "RG": "#ec4899", "TELEFONE": "#3b82f6",
         "EMAIL": "#8b5cf6", "ENDERECO": "#10b981", "CEP": "#14b8a6"}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODELO_BASE)
    ap.add_argument("--adapter", default=str(RAIZ / "outputs" / "melhor"))
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    tem_adapter = Path(args.adapter).exists()
    if not tem_adapter:
        print(f"Adaptador não encontrado em {args.adapter}: usando só o modelo base.")
    model, tok = carregar_modelo(args.model, args.adapter if tem_adapter else None)
    gpu = info_gpu()
    origem = Path(args.adapter, "ORIGEM.txt")
    nome_adapter = origem.read_text(encoding="utf-8").strip() if origem.exists() else Path(args.adapter).name

    def gerar(texto, temperatura, max_tokens, base):
        t0 = time.time()
        if base and tem_adapter:
            with model.disable_adapter():
                saida, n = anonimizar(model, tok, [texto], int(max_tokens), temperatura)
        else:
            saida, n = anonimizar(model, tok, [texto], int(max_tokens), temperatura)
        dt = time.time() - t0
        return saida[0], n, dt

    def processar(texto, modo, temperatura, max_tokens):
        if not texto.strip():
            return [], [], "Digite um texto."
        ajust = base = []
        infos = []
        if modo in ("Ajustado (LoRA)", "Comparar") and tem_adapter:
            s, n, dt = gerar(texto, temperatura, max_tokens, base=False)
            ajust = destacar(s)
            infos.append(f"**Ajustado:** {dt:.2f}s · {n} tokens · {n / dt:.1f} tok/s")
        if modo in ("Base (sem treino)", "Comparar") or not tem_adapter:
            s, n, dt = gerar(texto, temperatura, max_tokens, base=True)
            base = destacar(s)
            infos.append(f"**Base:** {dt:.2f}s · {n} tokens · {n / dt:.1f} tok/s")
        if torch.cuda.is_available():
            infos.append(f"VRAM em uso: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        return ajust, base, " | ".join(infos)

    def livre(prompt, system, modo, temperatura, max_tokens):
        if not prompt.strip():
            return "", "", "Digite um prompt."
        respostas = {"ajust": "", "base": ""}
        infos = []
        alvos = []
        if modo in ("Ajustado (LoRA)", "Comparar") and tem_adapter:
            alvos.append(("ajust", "Ajustado"))
        if modo in ("Base (sem treino)", "Comparar") or not tem_adapter:
            alvos.append(("base", "Base"))
        for chave, rotulo in alvos:
            t0 = time.time()
            if chave == "base" and tem_adapter:
                with model.disable_adapter():
                    r, n = gerar_livre(model, tok, prompt, system or None, int(max_tokens), temperatura)
            else:
                r, n = gerar_livre(model, tok, prompt, system or None, int(max_tokens), temperatura)
            dt = time.time() - t0
            respostas[chave] = r
            infos.append(f"**{rotulo}:** {dt:.2f}s · {n} tokens · {n / dt:.1f} tok/s")
        if torch.cuda.is_available():
            infos.append(f"VRAM em uso: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        return respostas["ajust"], respostas["base"], " | ".join(infos)

    def controles(temp_padrao):
        modo = gr.Radio(["Ajustado (LoRA)", "Base (sem treino)", "Comparar"], value="Comparar", label="Modelo")
        temperatura = gr.Slider(0, 1.5, value=temp_padrao, step=0.1, label="Temperatura (0 = determinístico)")
        max_tokens = gr.Slider(32, 1024, value=256, step=16, label="Máx. tokens gerados")
        return modo, temperatura, max_tokens

    with gr.Blocks(title="Anonimizador LGPD") as demo:
        gr.Markdown(
            "# Anonimizador LGPD com LLM local\n"
            f"Modelo **{args.model}** + LoRA **{nome_adapter if tem_adapter else '(nenhum)'}** · "
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
                    botao = gr.Button("Anonimizar", variant="primary")
            with gr.Row():
                saida_ajust = gr.HighlightedText(label="Modelo ajustado (LoRA)", color_map=CORES)
                saida_base = gr.HighlightedText(label="Modelo base", color_map=CORES)
            info = gr.Markdown()
            botao.click(processar, [entrada, modo, temperatura, max_tokens], [saida_ajust, saida_base, info])
            entrada.submit(processar, [entrada, modo, temperatura, max_tokens], [saida_ajust, saida_base, info])

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
                resp_ajust = gr.Textbox(label="Resposta — modelo ajustado (LoRA)", lines=10)
                resp_base = gr.Textbox(label="Resposta — modelo base", lines=10)
            info_l = gr.Markdown()
            botao_l.click(livre, [prompt, system, modo_l, temperatura_l, max_tokens_l], [resp_ajust, resp_base, info_l])

    demo.launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
