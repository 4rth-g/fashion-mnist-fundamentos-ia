# Fashion-MNIST — Classificação com CNN (Fundamentos de IA)

Classificação de imagens de **peças de roupa** do **Fashion-MNIST** (Zalando) com uma
**Rede Neural Convolucional (CNN)** estilo LeNet, em PyTorch. Trabalho da disciplina
de *Fundamentos de IA*.

O Fashion-MNIST é um substituto *drop-in* do MNIST clássico — mesmo formato (70.000
imagens 28×28 em tons de cinza, 10 classes), porém **mais difícil** e representativo
de tarefas reais de visão computacional.

O código foi escrito para ser **simples, legível e reproduzível**, e roda igual em
Intel Arc, NVIDIA, AMD, Apple Silicon ou CPU — e também no **Google Colab**.

## Estrutura

```
fashion-mnist-fundamentos-ia/
├── src/
│   └── utils.py          # device, sementes, split treino/val/teste, loaders
├── notebooks/
│   ├── 01_eda.ipynb      # análise exploratória do dataset
│   └── 02_cnn.ipynb      # modelo central: CNN, treino e avaliação
├── docs/
│   └── documentacao.typ  # doc complementar (dataset, CNN, matemática, glossário) → PDF
├── figures/              # gráficos gerados pelos notebooks
├── results/              # métricas (metrics.json, gerado pelo 02_cnn)
├── models/               # pesos treinados (gitignored)
└── data/                 # Fashion-MNIST (baixado automaticamente, gitignored)
```

As 10 classes: `Camiseta/top`, `Calça`, `Pulôver`, `Vestido`, `Casaco`, `Sandália`,
`Camisa`, `Tênis`, `Bolsa`, `Bota`.

## Como rodar (local, com `uv`)

[`uv`](https://docs.astral.sh/uv/) cuida do ambiente e das dependências a partir do
`uv.lock` — todos os colegas obtêm exatamente o mesmo ambiente.

```bash
uv sync                    # cria o ambiente e instala tudo (PyTorch CPU por padrão)
uv run jupyter lab         # abre os notebooks
```

Os notebooks resolvem os caminhos sozinhos (via `src/utils.py`), então podem ser
executados de qualquer pasta.

## Como rodar no Google Colab

Cada notebook tem uma célula de *setup* no topo que, ao detectar o Colab, clona este
repositório e ajusta o ambiente automaticamente. Depois de publicar o projeto no
GitHub, atualize a variável `REPO_URL` nessa célula e use os links:

[![EDA no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/4rth-g/fashion-mnist-fundamentos-ia/blob/main/notebooks/01_eda.ipynb)
[![CNN no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/4rth-g/fashion-mnist-fundamentos-ia/blob/main/notebooks/02_cnn.ipynb)

## Hardware (aceleração opcional)

O Fashion-MNIST treina em poucos minutos **na CPU** — GPU é só um acelerador opcional. O
`utils.get_device()` detecta e usa automaticamente o melhor dispositivo disponível.
Por padrão o `pyproject.toml` instala o PyTorch para CPU (universal). Para usar GPU,
reinstale o `torch` com o índice correto **depois** do `uv sync`:

| Hardware | Backend | Como instalar o PyTorch |
|---|---|---|
| **Intel Arc B580** (Windows) | XPU | `uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/xpu` + [driver Intel Arc](https://www.intel.com/content/www/us/en/download/785597/) |
| **NVIDIA** (Linux/Windows) | CUDA | `uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124` |
| **Google Colab** | CUDA (T4) | já vem pronto — nada a instalar |
| **AMD** (Linux) | ROCm | `uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2` (aparece como `cuda`) |
| **AMD** (Windows) | — | sem ROCm estável: use CPU (padrão) ou `torch-directml` |
| **Apple Silicon** | MPS | `uv pip install --reinstall torch torchvision` (índice padrão) |
| **Qualquer CPU** | CPU | padrão do `uv sync` — nada a fazer |

## O modelo

CNN estilo **LeNet** (~420 mil parâmetros), esperado **~90–93% de acurácia no teste**
(o Fashion-MNIST é bem mais difícil que o MNIST, que chega a ~99%):

```
Conv(1→32, 3×3) + ReLU + MaxPool     28×28 → 14×14
Conv(32→64, 3×3) + ReLU + MaxPool    14×14 → 7×7
Flatten → Linear(3136→128) + ReLU + Dropout(0.5)
Linear(128→10)
```

**Metodologia:** os 60k de treino são divididos em **50k treino / 10k validação**; a
validação guia o treino e a seleção do melhor modelo; o **teste (10k) é usado uma
única vez**, ao final — evitando vazamento de dados. Uma **augmentation leve**
(recorte + espelhamento) é aplicada *só no treino*. As métricas finais (acurácia,
F1 macro, relatório por classe) são salvas em `results/metrics.json`.

## Documentação complementar

`docs/documentacao.typ` explica o dataset (Fashion-MNIST), cada operação da CNN, a base matemática
(convolução, ReLU, pooling, softmax, entropia cruzada, backpropagation, Adam) e um
glossário. Compile o PDF com:

```bash
typst compile docs/documentacao.typ
```
