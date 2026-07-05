"""
utils.py — funções de apoio do projeto MNIST (Fundamentos de IA).

Concentra aqui a parte "de infraestrutura" (seleção de hardware, sementes,
carregamento e divisão dos dados) para que os notebooks fiquem curtos e
focados no que importa: a rede neural e os resultados.

Organização:
    1. Caminhos do projeto        -> PROJECT_ROOT, DATA_DIR, ...
    2. Reprodutibilidade          -> seed_everything()
    3. Seleção de dispositivo     -> get_device(), autocast_ctx(), ...
    4. Dados (MNIST)              -> get_transform(), make_dataloaders()

Compatível com Intel Arc (XPU), NVIDIA (CUDA), AMD (ROCm via CUDA / DirectML),
Apple Silicon (MPS) e CPU — a mesma linha de código roda em qualquer um.
"""

from __future__ import annotations

import contextlib
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ─────────────────────────────────────────────────────────────────────────────
# 1. Caminhos do projeto
#    Resolvidos a partir da localização deste arquivo (src/utils.py), então
#    funcionam independentemente de onde o notebook é executado — inclusive no
#    Google Colab depois de clonar o repositório.
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

for _d in (DATA_DIR, FIGURES_DIR, RESULTS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Estatísticas oficiais do MNIST (média e desvio-padrão do conjunto de treino),
# usadas para normalizar as imagens para média ~0 e desvio ~1.
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

SEED = 42  # semente única do projeto (reprodutibilidade)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Reprodutibilidade
# ─────────────────────────────────────────────────────────────────────────────
def seed_everything(seed: int = SEED) -> torch.Generator:
    """Fixa as sementes de random, numpy e torch para resultados reproduzíveis.

    Retorna um `torch.Generator` semeado, útil para passar ao `random_split` e
    aos `DataLoader` (embaralhamento reproduzível).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


# ─────────────────────────────────────────────────────────────────────────────
# 3. Seleção de dispositivo (Intel Arc / NVIDIA / AMD / Apple / CPU)
# ─────────────────────────────────────────────────────────────────────────────
def _is_available(backend: str) -> bool:
    if backend == "xpu":  # Intel Arc (ex.: B580)
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    if backend == "cuda":  # NVIDIA — e também AMD/ROCm, que reusa a API cuda
        return torch.cuda.is_available()
    if backend == "mps":  # Apple Silicon
        return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    if backend == "cpu":
        return True
    return False


def _device_name(backend: str) -> str:
    try:
        if backend == "xpu":
            return torch.xpu.get_device_name(0)
        if backend == "cuda":
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    if backend == "mps":
        return "Apple Silicon (MPS)"
    import platform

    return platform.processor() or platform.machine() or "CPU"


def get_device(prefer: str | None = None, verbose: bool = True) -> torch.device:
    """Retorna o melhor dispositivo disponível.

    Ordem de preferência: XPU (Intel Arc) > CUDA (NVIDIA/AMD-ROCm) > MPS (Apple) > CPU.
    Passe `prefer="cuda"` para forçar um backend específico, se desejar.

    Observação p/ AMD no Windows: não há PyTorch-ROCm estável no Windows; nesse
    caso use CPU (padrão) ou o pacote `torch-directml` (fora do escopo deste helper).
    """
    order = ["xpu", "cuda", "mps", "cpu"]
    if prefer:
        order = [prefer] + [b for b in order if b != prefer]

    backend = next(b for b in order if _is_available(b))
    device = torch.device(backend)

    if verbose:
        icon = {"xpu": "🔵", "cuda": "🟢", "mps": "🍎", "cpu": "⚪"}[backend]
        print(f"{icon} Dispositivo: {backend.upper()} — {_device_name(backend)}")
        if backend == "cpu":
            print("   (nenhuma GPU detectada — o treino será mais lento, mas funciona)")
    return device


def autocast_ctx(device: torch.device, enabled: bool = True):
    """Context manager de precisão mista (AMP), acelera o treino em GPU.

    CUDA -> float16 · XPU/CPU -> bfloat16 · MPS -> desabilitado (suporte parcial).
    """
    dev = device.type
    if not enabled or dev == "mps":
        return contextlib.nullcontext()
    dtype = torch.float16 if dev == "cuda" else torch.bfloat16
    return torch.autocast(device_type=dev, dtype=dtype)


def loader_kwargs(device: torch.device, num_workers: int | None = None) -> dict:
    """kwargs recomendados para o DataLoader conforme o dispositivo."""
    pin = device.type in ("cuda", "xpu")
    if num_workers is None:
        num_workers = 2 if pin else 0
    kw = {"num_workers": num_workers, "pin_memory": pin}
    if num_workers > 0:
        kw["persistent_workers"] = True
    return kw


def synchronize(device: torch.device) -> None:
    """Sincroniza a GPU antes de medir tempo (no-op em CPU)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Dados (MNIST)
# ─────────────────────────────────────────────────────────────────────────────
def get_transform(normalize: bool = True) -> transforms.Compose:
    """Pipeline de pré-processamento.

    `ToTensor` converte a imagem PIL (0–255) em tensor float (0–1);
    `Normalize` centra os pixels usando média/desvio do MNIST. Para *visualizar*
    imagens na EDA, use `normalize=False` (mantém a escala 0–1, mais legível).
    """
    steps = [transforms.ToTensor()]
    if normalize:
        steps.append(transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)))
    return transforms.Compose(steps)


def load_mnist(normalize: bool = True):
    """Baixa (se preciso) e retorna os datasets (treino_completo, teste)."""
    tfm = get_transform(normalize=normalize)
    train_full = datasets.MNIST(root=str(DATA_DIR), train=True, download=True, transform=tfm)
    test = datasets.MNIST(root=str(DATA_DIR), train=False, download=True, transform=tfm)
    return train_full, test


def make_dataloaders(
    batch_size: int = 128,
    val_size: int = 10_000,
    device: torch.device | None = None,
    seed: int = SEED,
):
    """Cria os três DataLoaders com divisão metodologicamente correta.

    Os 60.000 exemplos de TREINO são divididos em treino (50k) + validação (10k)
    de forma reproduzível. O conjunto de TESTE (10k) permanece intocado e só deve
    ser usado UMA vez, na avaliação final — nunca para escolher hiperparâmetros.

    Retorna: (train_loader, val_loader, test_loader).
    """
    if device is None:
        device = torch.device("cpu")

    train_full, test = load_mnist(normalize=True)

    n_train = len(train_full) - val_size  # 60000 - 10000 = 50000
    gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(train_full, [n_train, val_size], generator=gen)

    kw = loader_kwargs(device)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=gen, **kw)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, **kw)
    test_loader = DataLoader(test, batch_size=256, shuffle=False, **kw)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Diagnóstico rápido: `python src/utils.py`
    seed_everything()
    dev = get_device()
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"loader_kwargs: {loader_kwargs(dev)}")
