"""
Avalia o modelo base ou um adaptador LoRA no conjunto de teste.

Exemplos:
    python src/evaluate.py --nome base
    python src/evaluate.py --nome exp1 --adapter outputs/exp1
    python src/evaluate.py --nome exp1_temp07 --adapter outputs/exp1 --temperature 0.7
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import MODELO_BASE, anonimizar, carregar_modelo, info_gpu, ler_jsonl, salvar_json
from metrics import avaliar_por_grupo

RAIZ = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nome", required=True)
    ap.add_argument("--model", default=MODELO_BASE)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default=str(RAIZ / "data" / "test.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--out-dir", default=str(RAIZ / "results"))
    args = ap.parse_args()

    exemplos = ler_jsonl(args.data, args.limit)
    model, tok = carregar_modelo(args.model, args.adapter)

    previsoes, tokens = [], 0
    t0 = time.time()
    for i in range(0, len(exemplos), args.batch_size):
        lote = [e["input"] for e in exemplos[i:i + args.batch_size]]
        saidas, n = anonimizar(model, tok, lote, args.max_new_tokens, args.temperature, args.top_p)
        previsoes += saidas
        tokens += n
        print(f"{min(i + args.batch_size, len(exemplos))}/{len(exemplos)}", end="\r")
    duracao = time.time() - t0

    metricas = avaliar_por_grupo(exemplos, previsoes)
    out_dir = Path(args.out_dir)
    resultado = {
        "nome": args.nome,
        "modelo": args.model,
        "adapter": args.adapter,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "gpu": info_gpu(),
        "tempo_s": round(duracao, 1),
        "tokens_por_s": round(tokens / duracao, 1),
        "metricas": metricas,
        "amostras": [{"input": e["input"], "esperado": e["output"], "previsto": p}
                     for e, p in list(zip(exemplos, previsoes))[:15]],
    }
    salvar_json(out_dir / f"{args.nome}.json", resultado)
    with open(out_dir / f"{args.nome}_previsoes.jsonl", "w", encoding="utf-8") as f:
        for e, p in zip(exemplos, previsoes):
            f.write(json.dumps({"input": e["input"], "esperado": e["output"], "previsto": p,
                                "visto": e.get("visto")}, ensure_ascii=False) + "\n")

    g = metricas["geral"]
    print(f"\n[{args.nome}] F1={g['f1']:.3f} vazamento={g['vazamento_pct']}% "
          f"exato={g['exato_pct']}% similaridade={g['similaridade']:.3f} "
          f"({duracao:.0f}s, {resultado['tokens_por_s']} tok/s)")


if __name__ == "__main__":
    main()
