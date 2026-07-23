# Fashion-MNIST — Classificação com CNN (Fundamentos de IA)

Classificação de imagens de **peças de roupa** do **Fashion-MNIST** (Zalando) com uma
**Rede Neural Convolucional (CNN)** estilo LeNet, em PyTorch, comparada a dois baselines
clássicos de Scikit-Learn (`SGDClassifier`, `RandomForestClassifier`). Trabalho da disciplina
de *Fundamentos de IA*.

O Fashion-MNIST é um substituto *drop-in* do MNIST clássico — mesmo formato (70.000
imagens 28×28 em tons de cinza, 10 classes), porém **mais difícil** e representativo
de tarefas reais de visão computacional.

O código foi escrito para ser **simples, legível e reproduzível**, e roda igual em
Intel Arc, NVIDIA, AMD, Apple Silicon ou CPU — e também no **Google Colab**.

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

Cada notebook tem uma célula de *setup* no topo que, ao detectar o Colab, clona este
repositório e ajusta o ambiente automaticamente. Depois de publicar o projeto no
GitHub, atualize a variável `REPO_URL` nessa célula e use os links:

[![Baseline no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/4rth-g/fashion-mnist-fundamentos-ia/blob/main/notebooks/00_baseline.ipynb)
[![EDA no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/4rth-g/fashion-mnist-fundamentos-ia/blob/main/notebooks/01_eda.ipynb)
[![CNN no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/4rth-g/fashion-mnist-fundamentos-ia/blob/main/notebooks/02_cnn.ipynb)
[![Tuning no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/4rth-g/fashion-mnist-fundamentos-ia/blob/main/notebooks/03_tuning.ipynb)

> ⚠️ Confira se o nome do repositório acima (`fashion-mnist-fundamentos-ia`) bate com o do
> GitHub de vocês — se o remote real tiver outro nome, atualize tanto estes links quanto a
> variável `REPO_URL` no topo de cada notebook.

## Hardware (aceleração opcional)

O Fashion-MNIST treina em poucos minutos **na CPU** — GPU é só um acelerador opcional. O
`utils.get_device()` detecta e usa automaticamente o melhor dispositivo disponível.
Por padrão o `pyproject.toml` instala o PyTorch para CPU (universal). Para usar GPU,
reinstale o `torch` com o índice correto **depois** do `uv sync`:

| Hardware | Backend | Como instalar o PyTorch |
|---|---|---|
| **Intel Arc B580** (Windows) | XPU | `uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/xpu` + [driver Intel Arc](https://www.intel.com/content/www/us/en/download/785597/) |
| **Intel Arc B580** (Linux, cp313) | XPU | `uv pip install --reinstall torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/xpu` — versões travadas em par: é o build XPU mais recente disponível para Linux/cp313 (torchvision `0.22.0+xpu` exige `torch==2.7.0` exato). Requer runtime Intel (`intel-opencl-icd`, `libze-intel-gpu1`) e driver de kernel `xe` já carregado |
| **NVIDIA** (Linux/Windows) | CUDA | `uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124` |
| **Google Colab** | CUDA (T4) | já vem pronto — nada a instalar |
| **AMD** (Linux) | ROCm | `uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2` (aparece como `cuda`) |
| **AMD** (Windows) | — | sem ROCm estável: use CPU (padrão) ou `torch-directml` |
| **Apple Silicon** | MPS | `uv pip install --reinstall torch torchvision` (índice padrão) |
| **Qualquer CPU** | CPU | padrão do `uv sync` — nada a fazer |

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

A CNN supera o melhor baseline em ~2,9 pontos percentuais no teste (90,24% vs. 87,31%),
evidência de que a convolução captura padrões que os 784 pixels tratados independentemente
não capturam. O ajuste de hiperparâmetros (`lr=0.002`, `dropout=0.2`, `weight_decay=0`) leva
a CNN a 92,04% no teste — um ganho adicional de ~1,8 p.p. sobre a CNN sem tuning, como esperado
de um ajuste fino sobre uma arquitetura já escolhida. (Pequenas variações entre execuções do
grid, como as ~0,7 p.p. entre esta rodada e uma anterior, são normais — vêm de aleatoriedade
residual do treino em GPU e não mudam a conclusão: a configuração `lr=0.002`/`dropout=0.2`
consistentemente vence na validação.)

A classe mais difícil para a CNN foi **Camisa** (recall 0,628), frequentemente confundida com
Camiseta/top e Pulôver — coerente com a EDA, onde essas classes já apareciam visualmente
parecidas nas imagens médias por classe.

## Documentação complementar

`docs/documentacao.typ` explica o dataset (Fashion-MNIST), cada operação da CNN, a base matemática
(convolução, ReLU, pooling, softmax, entropia cruzada, backpropagation, Adam) e um
glossário. O PDF já compilado está versionado em `docs/documentacao.pdf`; para
regenerá-lo após editar o `.typ`:

```bash
typst compile docs/documentacao.typ
```

## Slides e pasta do Drive

- **Slide de apresentação:** `docs/fashion-mnist-slides.pdf` (versionado no repositório).
- **Pasta do Google Drive:** os materiais complementares (slides em edição, vídeo de
  apresentação e outros arquivos grandes que não vão para o Git) ficam numa pasta do
  Google Drive, cujo link está em **`docs/link-drive.txt`**.

## Divisão das contribuições

_PREENCHER — o que cada integrante fez (ex.: "Fulano: EDA e baseline; Beltrano: CNN e
tuning; Sicrano: documentação e vídeo"). Todos os integrantes listados aqui precisam também
aparecer participando ativamente no vídeo — quem não participar da explicação é excluído
da avaliação, mesmo constando no repositório._

## Vídeo

_PREENCHER — link do vídeo (YouTube não listado, Google Drive, etc.). Cada integrante deve se
identificar, explicar sua parte, justificar ao menos uma decisão técnica e interpretar ao
menos um resultado._

Link: _PREENCHER_

## Declaração sobre uso de Inteligência Artificial

Em conformidade com o [Código de Conduta da SBC](https://www.sbc.org.br/) para autores, o uso
de ferramentas de Inteligência Artificial Generativa na escrita e/ou revisão deste trabalho é
declarado explicitamente abaixo. Nenhuma ferramenta de IA é listada como autora do trabalho, e
seu uso não isenta os integrantes da responsabilidade pelo conteúdo produzido, incluindo em
caso de plágio identificado.

- **Ferramenta utilizada:** Claude (Anthropic), via Claude Code — Arthur de Azevedo Grazzia.
  _(Rafael Rocha da Silva: PREENCHER — se usou alguma ferramenta de IA em sua parte, declarar
  aqui da mesma forma; se não usou, declarar isso explicitamente também.)_
- **Finalidade:** apoio na organização do repositório Git (merge de branches, resolução de
  conflitos, reescrita de commits), extensão da análise exploratória de dados além do que
  havia sido feito manualmente, e revisão de conformidade do repositório com os critérios do
  enunciado.
- **Parte do trabalho em que foi utilizada:**
  - `notebooks/01_eda.ipynb`: código e texto interpretativo das seções de correlação entre
    regiões da imagem, projeção PCA (gráfico de dispersão) e análise de valores extremos/
    atributos irrelevantes.
  - `notebooks/00_baseline.ipynb` e `notebooks/02_cnn.ipynb`: trechos curtos de texto na seção
    de pré-processamento, referenciando os achados acima.
  - Seção "Integrantes" deste README.
  - Organização do histórico Git do repositório (merge da branch `SGDC/RF`, resolução de
    conflitos em `notebooks/03_tuning.ipynb` e `uv.lock`, remoção de coautoria automática de
    commits antigos).
  - Execução de ponta a ponta de `00_baseline.ipynb` (não estava executado) e `03_tuning.ipynb`
    (grid grande, na GPU) para gerar os outputs e os números da tabela de resultados.
- **Forma de verificação do conteúdo/código produzido:** todo código gerado foi executado de
  ponta a ponta (`jupyter nbconvert --execute`) antes de ser incorporado, e os números citados
  nas interpretações (ex.: 0,06% de imagens atípicas, 1,4% dos pixels com variância quase nula,
  correlações entre quadrantes de 0,44 a 0,91) vêm diretamente dessa execução — não foram
  estimados ou inventados. O texto final foi revisado por Arthur antes de ser commitado.

> O enunciado do trabalho reforça que a avaliação prioriza a capacidade do grupo de **explicar
> e justificar** o trabalho, não a qualidade aparente do texto/código — preencham a parte do
> Rafael com a mesma honestidade.
