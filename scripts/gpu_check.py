"""
Mostra as informações da GPU e comprova que o PyTorch está usando CUDA.
Rode no terminal do VSCode e tire o print (item 7 do trabalho).

    python scripts/gpu_check.py
"""
import platform
import subprocess
import time

import torch

print("=" * 60)
print("Python      :", platform.python_version())
print("PyTorch     :", torch.__version__)
print("CUDA (torch):", torch.version.cuda)
print("cuDNN       :", torch.backends.cudnn.version())
print("CUDA ativa  :", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("CUDA indisponível: instale o PyTorch com CUDA (veja o README).")

p = torch.cuda.get_device_properties(0)
arch = f"sm_{p.major}{p.minor}"
print("GPU         :", p.name)
print("Compute cap.:", f"{p.major}.{p.minor} ({arch})")
print("VRAM        :", f"{p.total_memory / 1024**3:.2f} GB")
print("SMs         :", p.multi_processor_count)
print("Arquiteturas compiladas no torch:", ", ".join(torch.cuda.get_arch_list()))
if arch not in torch.cuda.get_arch_list():
    print(f"AVISO: este build do PyTorch não inclui {arch}. Use o build cu126 (veja o README).")

# Teste rápido: multiplicação de matrizes na GPU.
n = 4096
a = torch.randn(n, n, device="cuda")
b = torch.randn(n, n, device="cuda")
torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):
    c = a @ b
torch.cuda.synchronize()
dt = time.time() - t0
print(f"Matmul {n}x{n} fp32 x10: {dt:.2f}s  ->  {10 * 2 * n**3 / dt / 1e12:.2f} TFLOPS")
print("Memória alocada agora:", f"{torch.cuda.memory_allocated() / 1024**2:.0f} MB")
print("=" * 60)

try:
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
except FileNotFoundError:
    pass
