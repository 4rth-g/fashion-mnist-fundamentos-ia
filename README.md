# Fashion-MNIST — Classificação com CNN (Fundamentos de IA)

Classificação de imagens de **peças de roupa** do **Fashion-MNIST** (Zalando) com uma
**Rede Neural Convolucional (CNN)** estilo LeNet, em PyTorch, comparada a dois baselines
clássicos de Scikit-Learn (`SGDClassifier`, `RandomForestClassifier`). Trabalho da disciplina
de *Fundamentos de IA*.

O Fashion-MNIST é um substituto *drop-in* do MNIST (70.000 imagens 28×28 em tons de cinza,
10 classes), porém **mais difícil** e representativo de tarefas reais de visão computacional.
O código é reproduzível e roda em CPU, GPU (Intel/NVIDIA/AMD/Apple) ou no **Google Colab**.

## Integrantes

- Arthur de Azevedo Grazzia
- Rafael Rocha da Silva

## Objetivo e tipo da tarefa

- **Objetivo:** reconhecer automaticamente, a partir da imagem, a categoria de uma peça de
  roupa (ex.: distinguir uma camisa de um casaco).
- **Atributo-alvo:** categoria da peça — variável categórica com 10 classes.
- **Atributos preditivos:** os 784 pixels (28×28, tons de cinza) de cada imagem.
- **Tipo da tarefa:** classificação multiclasse.
- **Fonte dos dados:** [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
  (Zalando Research), baixado automaticamente via `torchvision.datasets.FashionMNIST`.

## Estrutura

```
fashion-mnist-fundamentos-ia/
├── src/
│   └── utils.py          # device, sementes, split treino/val/teste, loaders
├── notebooks/
│   ├── 00_baseline.ipynb  # modelos mínimos exigidos: SGDClassifier e RandomForestClassifier
│   ├── 01_eda.ipynb       # análise exploratória do dataset
│   ├── 02_cnn.ipynb       # modelo principal: CNN, treino e avaliação
│   └── 03_tuning.ipynb    # ajuste de hiperparâmetros da CNN (grid search)
├── docs/
│   ├── documentacao.typ         # doc complementar (dataset, CNN, matemática, glossário)
│   ├── documentacao.pdf         # PDF compilado da doc (versionado)
│   ├── fashion-mnist-slides.pdf # slide de apresentação (versionado)
│   └── link-drive.txt           # link da pasta do Google Drive do projeto
├── figures/              # gráficos gerados pelos notebooks
├── results/              # métricas geradas ao rodar os notebooks (baseline_metrics.json,
│                         #   metrics.json, grid_results.json); pasta gitignored
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

Cada notebook detecta o Colab e clona o repositório automaticamente. Abra pelos badges
(confira que o nome do repositório nos links e na variável `REPO_URL` de cada notebook bate
com o do GitHub de vocês):

[![Baseline no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/4rth-g/fashion-mnist-fundamentos-ia/blob/main/notebooks/00_baseline.ipynb)
[![EDA no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/4rth-g/fashion-mnist-fundamentos-ia/blob/main/notebooks/01_eda.ipynb)
[![CNN no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/4rth-g/fashion-mnist-fundamentos-ia/blob/main/notebooks/02_cnn.ipynb)
[![Tuning no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/4rth-g/fashion-mnist-fundamentos-ia/blob/main/notebooks/03_tuning.ipynb)

## Hardware (aceleração opcional)

O Fashion-MNIST treina em poucos minutos **na CPU** (padrão do `uv sync`); GPU é só um
acelerador. O `utils.get_device()` escolhe o melhor dispositivo automaticamente. Para usar
GPU, reinstale o `torch` com o índice correto **depois** do `uv sync`:

| Hardware | Backend | Índice do PyTorch (`uv pip install --reinstall torch torchvision --index-url …`) |
|---|---|---|
| **NVIDIA** | CUDA | `…/whl/cu124` |
| **AMD** (Linux) | ROCm | `…/whl/rocm6.2` |
| **Intel Arc** | XPU | `…/whl/xpu` (Linux/cp313: fixe `torch==2.7.0 torchvision==0.22.0` e runtime Intel) |
| **Apple Silicon** | MPS | índice padrão (nada a fazer além do `uv sync`) |
| **Google Colab** | CUDA (T4) | já vem pronto |

> AMD no Windows não tem ROCm estável — use CPU ou `torch-directml`.

## Modelos utilizados

| Notebook | Modelo | Papel no projeto |
|---|---|---|
| `00_baseline.ipynb` | `SGDClassifier` | Baseline linear — modelo mínimo exigido |
| `00_baseline.ipynb` | `RandomForestClassifier` | Baseline por árvores — modelo mínimo exigido |
| `02_cnn.ipynb` | CNN estilo LeNet | Modelo principal do projeto |
| `03_tuning.ipynb` | CNN (grid search) | Ajuste de hiperparâmetros do modelo principal |

Os baselines tratam cada imagem como um vetor de 784 pixels (achatado), sem noção de estrutura
espacial. A CNN (~420 mil parâmetros) explora essa estrutura via convolução:

```
Conv(1→32, 3×3) + ReLU + MaxPool     28×28 → 14×14
Conv(32→64, 3×3) + ReLU + MaxPool    14×14 → 7×7
Flatten → Linear(3136→128) + ReLU + Dropout(0.5)
Linear(128→10)
```

**Metodologia (comum a todos os notebooks):** os 60k de treino são divididos em **50k treino /
10k validação**; a validação guia o treino e a seleção do melhor modelo (e, no caso do
baseline, escolhe entre `SGDClassifier` e `RandomForestClassifier`); o **teste (10k) é usado
uma única vez**, ao final — evitando vazamento de dados. Uma **augmentation leve** (recorte +
espelhamento) é aplicada *só no treino* da CNN.

## Principais resultados

| Modelo | Acurácia (validação) | Acurácia (teste) | F1 macro (teste) |
|---|---|---|---|
| SGDClassifier | 0,8452 | — | — |
| RandomForestClassifier (vencedor do baseline) | 0,8836 | 0,8731 | 0,8715 |
| CNN (`02_cnn.ipynb`) | 0,9062 | 0,9024 | 0,9011 |
| CNN + tuning (`03_tuning.ipynb`) | 0,9246 | 0,9204 | 0,9203 |

> O `SGDClassifier` não tem acurácia de teste: como perdeu para o `RandomForestClassifier` na
> validação, ele nunca toca o conjunto de teste (só o modelo escolhido pela validação é
> avaliado no teste, para não "gastar" essa reserva com mais de um candidato).

A CNN supera o melhor baseline em ~2,9 p.p. no teste (90,24% vs. 87,31%), evidência de que a
convolução captura padrões que os 784 pixels tratados independentemente não capturam. O ajuste
de hiperparâmetros (`lr=0.002`, `dropout=0.2`, `weight_decay=0`) leva a CNN a 92,04% — mais
~1,8 p.p., como esperado de um ajuste fino sobre uma arquitetura já escolhida. A classe mais
difícil foi **Camisa** (recall 0,628), confundida com Camiseta/top e Pulôver — coerente com a
EDA, onde essas classes já apareciam visualmente parecidas.

## Documentação e materiais

- **`docs/documentacao.pdf`** (compilado de `documentacao.typ`): explica o dataset, cada
  operação da CNN, a base matemática (convolução, ReLU, pooling, softmax, entropia cruzada,
  backpropagation, Adam) e um glossário. Regenerar com `typst compile docs/documentacao.typ`.
- **`docs/fashion-mnist-slides.pdf`**: slide de apresentação (versionado).
- **`docs/link-drive.txt`**: link da pasta do Google Drive com os materiais grandes que não
  vão para o Git (slides em edição, vídeo).

## Divisão das contribuições

- **Rafael Rocha da Silva:** baselines clássicos (`SGDClassifier` e `RandomForestClassifier`
  em `notebooks/00_baseline.ipynb`), incluindo pré-processamento (achatamento, split
  estratificado) e as seções exigidas pelo enunciado; atualização de notebooks e figuras.
- **Arthur de Azevedo Grazzia:** estrutura do projeto e utilitários (`src/utils.py`),
  modelo principal — CNN estilo LeNet (`notebooks/02_cnn.ipynb`) — e o ajuste de
  hiperparâmetros (`notebooks/03_tuning.ipynb`, grid search na GPU); análise exploratória
  (`notebooks/01_eda.ipynb`: intensidade, correlação, PCA, outliers); documentação
  (`docs/documentacao.typ`), README e execução de ponta a ponta dos notebooks.

## Vídeo

O vídeo de apresentação está na **pasta do Google Drive** do projeto (link em
[`docs/link-drive.txt`](docs/link-drive.txt)):

<https://drive.google.com/drive/folders/1fnZc1bltQuY7eVhtHnrGgfs78Gn5RABO?usp=drive_link>

No vídeo, cada integrante se identifica, explica sua parte, justifica ao menos uma decisão
técnica e interpreta ao menos um resultado.

## Declaração sobre uso de Inteligência Artificial

Em conformidade com o [Código de Conduta da SBC](https://www.sbc.org.br/) para autores, o uso
de ferramentas de Inteligência Artificial Generativa neste trabalho é declarado explicitamente
abaixo. Em todos os casos, a IA foi utilizada **apenas como auxílio ao desenvolvimento** — as
decisões técnicas, a validação e a responsabilidade pelo conteúdo são inteiramente dos
integrantes. Nenhuma ferramenta de IA é listada como autora do trabalho, e seu uso não isenta
os integrantes da responsabilidade pelo conteúdo produzido, incluindo em caso de plágio
identificado.

| Integrante | Modelo de IA | Auxílio ao desenvolvimento |
|---|---|---|
| Arthur de Azevedo Grazzia | Claude (Anthropic), via Claude Code | Organização do repositório Git; extensão da EDA (`01_eda.ipynb`: correlação, PCA, outliers); textos de pré-processamento; documentação (`docs/`) e README; execução dos notebooks. |
| Rafael Rocha da Silva | Claude (Anthropic) | Apoio nos baselines (`SGDClassifier`, `RandomForestClassifier`) em `00_baseline.ipynb`. |

**Verificação:** todo código gerado foi executado de ponta a ponta
(`jupyter nbconvert --execute`) antes de ser incorporado, e os números citados nas
interpretações vêm dessa execução — não foram estimados. O texto final foi revisado pelos
integrantes.

> O enunciado do trabalho reforça que a avaliação prioriza a capacidade do grupo de **explicar
> e justificar** o trabalho, não a qualidade aparente do texto/código.
