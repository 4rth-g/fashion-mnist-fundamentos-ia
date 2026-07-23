// Documentação complementar — Classificação do Fashion-MNIST com CNN
// Compilar com:  typst compile documentacao.typ
// (ou `typst watch documentacao.typ` para recompilar ao salvar)

#set document(
  title: "Fashion-MNIST + CNN — Documentação",
  author: ("Arthur de Azevedo Grazzia", "Rafael Rocha da Silva"),
)
#set page(numbering: "1", margin: 2.2cm)
#set text(font: "New Computer Modern", size: 11pt, lang: "pt")
#set par(justify: true, leading: 0.68em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => { v(0.4em); it; v(0.2em) }
#show link: set text(fill: blue)
#show raw: set text(font: "DejaVu Sans Mono", size: 9.5pt)

#align(center)[
  #text(18pt, weight: "bold")[Classificação de Peças de Roupa (Fashion-MNIST)] \
  #text(14pt)[com uma Rede Neural Convolucional] \
  #v(0.3em)
  #text(11pt, style: "italic")[Documentação complementar — Fundamentos de IA]
]

#v(0.5em)
#outline(title: "Sumário", indent: auto)
#line(length: 100%, stroke: 0.5pt + gray)

#block(inset: 8pt, fill: luma(245), radius: 4pt, width: 100%)[
  *Materiais do projeto.* O slide de apresentação está em
  `docs/fashion-mnist-slides.pdf`. Os demais materiais complementares (slides em
  edição, vídeo e arquivos grandes não versionados no Git) ficam numa pasta do
  Google Drive, cujo link está em `docs/link-drive.txt`.
]

= O dataset Fashion-MNIST

O *Fashion-MNIST* é um conjunto de imagens de *peças de roupa* publicado pela Zalando
como um substituto _drop-in_ do MNIST clássico: mesmo formato e mesma divisão, porém
*mais difícil* e mais representativo de tarefas reais de visão computacional. Contém
*70.000 imagens*:

- *60.000* imagens para *treino* e *10.000* para *teste*;
- cada imagem é *28 × 28 pixels*, em *tons de cinza* (1 canal), com valores de
  intensidade de 0 (preto) a 255 (branco);
- são *10 classes* — Camiseta/top, Calça, Pulôver, Vestido, Casaco, Sandália, Camisa,
  Tênis, Bolsa e Bota — e o dataset é *perfeitamente balanceado*
  (6.000 exemplos por classe no treino).

== Pré-processamento

Antes de entrar na rede, cada imagem passa por duas transformações:

+ *`ToTensor`*: converte a imagem em um tensor de números reais na faixa $[0, 1]$
  (divide por 255);
+ *`Normalize`*: recentraliza os pixels usando a média $mu = 0{.}2860$ e o desvio
  padrão $sigma = 0{.}3530$ do Fashion-MNIST, aplicando
  $ x' = (x - mu) / sigma. $
  Isso deixa os dados com média $approx 0$ e desvio $approx 1$, o que acelera e
  estabiliza o treino (os gradientes ficam melhor condicionados).

== Metodologia: treino, validação e teste

Para medir o desempenho de forma honesta, os 60.000 exemplos de treino são divididos
em *50.000 para treino* e *10.000 para validação*. A validação é usada para
acompanhar o aprendizado e escolher o melhor modelo. O *conjunto de teste é usado uma
única vez*, ao final — nunca para ajustar hiperparâmetros. Ignorar essa separação
causa *vazamento de dados* (_data leakage_) e infla artificialmente a acurácia
reportada.

Aplica-se ainda uma *data augmentation* leve — um recorte aleatório com borda de 2px
(`RandomCrop`) e um espelhamento horizontal (`RandomHorizontalFlip`) — *apenas ao
conjunto de treino*. Isso cria pequenas variações das peças a cada época e reduz o
_overfitting_. Validação e teste são avaliados *sem* augmentation; por isso o
conjunto de treino é carregado com dois _transforms_ diferentes, reaproveitando os
mesmos índices do split para que nenhuma imagem de validação receba augmentation.

Essa mesma divisão (50.000 treino, 10.000 validação, 10.000 teste) é reaproveitada pelos
modelos de *baseline* (`SGDClassifier`, `RandomForestClassifier`) descritos na próxima
seção — garantindo que os números reportados sejam diretamente comparáveis com os da CNN.

= Baseline: SGDClassifier e RandomForestClassifier

Além da CNN, o projeto treina dois modelos clássicos de Scikit-Learn — *SGDClassifier* e
*RandomForestClassifier* — usados como *baseline*: um patamar de comparação mais simples,
que ajuda a mostrar se a complexidade extra da CNN realmente se traduz em ganho de
desempenho. Este é também o requisito mínimo de modelos exigido pelo enunciado do trabalho
para tarefas de classificação.

== Por que um baseline

Diferente da CNN, esses modelos não enxergam a estrutura espacial da imagem — cada pixel é
tratado como um atributo independente, sem noção de vizinhança. Se a CNN não superasse esses
baselines mais simples, isso seria um sinal de que a complexidade extra (convolução, mais
parâmetros, mais tempo de treino) não estaria trazendo benefício real.

== Preparação dos dados

`SGDClassifier` e `RandomForestClassifier` esperam uma matriz 2D `(n_amostras, n_atributos)`,
não um tensor de imagem. O pré-processamento é, portanto, mais simples que o da CNN:

+ *Achatamento (_flatten_)*: cada imagem $28 times 28$ vira um vetor de *784 atributos*
  (um por pixel).
+ *Escala*: os pixels são usados na faixa $[0, 1]$ (após `ToTensor`, sem a normalização
  Z-score usada na CNN). Essa escala já é adequada para os dois modelos — o
  `RandomForestClassifier` é invariante à escala dos atributos (decide por _splits_, não por
  distância), e o `SGDClassifier` já trabalha bem com atributos numa faixa pequena e
  comparável entre si. Por isso, *não se aplica um `StandardScaler` adicional*.
+ *Separação estratificada*: os 60.000 exemplos de treino são divididos em *50.000 treino* /
  *10.000 validação*, preservando a proporção das 10 classes em cada subconjunto
  (`stratify=y`) — mais rigoroso que uma divisão puramente aleatória.

== SGDClassifier

Um classificador *linear*, treinado por *gradiente descendente estocástico*: a cada passo,
ajusta os pesos usando um pequeno lote de exemplos, na direção que reduz o erro. É rápido
mesmo em datasets grandes, mas só consegue separar as classes por combinações *lineares* dos
784 pixels — não captura relações não lineares nem a proximidade espacial entre eles.

== RandomForestClassifier

Uma *floresta aleatória*: um conjunto (_ensemble_) de várias árvores de decisão, cada uma
treinada em um subconjunto aleatório dos dados e dos atributos. A previsão final é a
*votação da maioria* das árvores. Cada árvore consegue aprender combinações simples e não
lineares entre pixels (ex.: "se o pixel $x$ é claro *e* o pixel $y$ é escuro..."), mas ainda
sem noção de vizinhança espacial — algo que só a convolução da CNN explora de fato.

== Metodologia de comparação

A mesma divisão treino/validação/teste descrita na seção anterior ("Metodologia: treino,
validação e teste") é reaproveitada aqui: a *validação* escolhe entre os dois baselines
(nunca o teste), e o vencedor é avaliado *uma única vez* no conjunto de teste, lado a lado
com a CNN.

== Resultados

#table(
  columns: (auto, auto, auto, auto),
  align: (left, center, center, center),
  stroke: 0.5pt + gray,
  table.header([*Modelo*], [*Acurácia (validação)*], [*Acurácia (teste)*], [*F1 macro (teste)*]),
  [SGDClassifier],                    [0,8452], [---],    [---],
  [RandomForestClassifier _(vencedor)_], [0,8836], [0,8731], [0,8715],
  [CNN (`02_cnn.ipynb`)],             [0,9062], [0,9024], [0,9011],
  [CNN + tuning (`03_tuning.ipynb`)], [0,9246], [0,9204], [0,9203],
)

#text(9pt)[_Números da CNN + tuning extraídos de `results/grid_results.json` (grid
executado na GPU em 2026-07-22); os demais, dos respectivos notebooks._]

_O `SGDClassifier` não tem acurácia de teste: como perdeu para o `RandomForestClassifier`
na validação, ele nunca toca o conjunto de teste — tocar o teste com um modelo que não foi
escolhido violaria a regra de "o teste é usado uma única vez", mesmo sem ajustar
hiperparâmetros a partir dele._

O `RandomForestClassifier` venceu o baseline (88,36% na validação) e, avaliado uma única vez
no teste, atingiu *87,31%* de acurácia. A CNN supera esse baseline em cerca de *2,9 pontos
percentuais* (90,24% vs. 87,31% no teste) — evidência de que a convolução realmente ajuda a
capturar padrões que os 784 pixels tratados independentemente não capturam. O ajuste de
hiperparâmetros (`lr = 0,002`, `dropout = 0,2`, `weight_decay = 0`) leva a CNN a *92,04%* no
teste, um ganho adicional de ~1,8 ponto percentual sobre a configuração fixa — consistente
com o que se espera de um _fine-tuning_: o maior salto de desempenho vem da escolha da
arquitetura (CNN vs. baseline), não do ajuste fino de hiperparâmetros.

A classe mais difícil para a CNN foi *Camisa* (recall $approx 0{,}628$), frequentemente
confundida com Camiseta/top e Pulôver — coerente com a análise exploratória, em que essas
três classes já apareciam visualmente parecidas nas imagens médias por classe. Esse é o tipo
de padrão que a *matriz de confusão* (ver glossário) evidencia.

= Arquitetura da CNN

Uma *Rede Neural Convolucional* (CNN) processa a imagem preservando sua estrutura
espacial. Em vez de conectar todos os pixels a todos os neurônios, ela aplica
*filtros* pequenos que deslizam pela imagem detectando padrões locais (bordas,
curvas), e vai combinando esses padrões em representações mais abstratas.

A arquitetura usada é do tipo *LeNet*, com o seguinte fluxo (formato do tensor
$"canais" times "altura" times "largura"$):

#table(
  columns: (auto, 1fr, auto),
  align: (center, left, center),
  stroke: 0.5pt + gray,
  table.header([*\#*], [*Operação*], [*Entrada → Saída*]),
  [1], [`Conv2d(1→32, 3×3, pad=1)` + ReLU], [$1 times 28 times 28 → 32 times 28 times 28$],
  [2], [`MaxPool2d(2)`],                     [$32 times 28 times 28 → 32 times 14 times 14$],
  [3], [`Conv2d(32→64, 3×3, pad=1)` + ReLU], [$32 times 14 times 14 → 64 times 14 times 14$],
  [4], [`MaxPool2d(2)`],                     [$64 times 14 times 14 → 64 times 7 times 7$],
  [5], [`Flatten`],                          [$64 times 7 times 7 → 3136$],
  [6], [`Linear(3136→128)` + ReLU + `Dropout(0.5)`], [$3136 → 128$],
  [7], [`Linear(128→10)`],                   [$128 → 10$ (logits)],
)

== Convolução (`Conv2d`)

Cada filtro é uma pequena matriz de pesos $k$ (aqui $3 times 3$) que percorre a
imagem. Em cada posição, calcula-se a soma dos produtos entre o filtro e a região
correspondente da entrada:
$ y_(i,j) = sum_(m) sum_(n) x_(i+m, space j+n) dot k_(m,n) + b. $
O resultado é um *mapa de características* que realça onde o padrão do filtro
aparece. A camada aprende $32$ filtros na primeira convolução e $64$ na segunda.
O `padding=1` acrescenta uma borda de zeros para que a saída mantenha o mesmo
tamanho espacial da entrada. Em geral, o tamanho de saída é
$ H_"out" = floor((H_"in" + 2p - k) / s) + 1, $
onde $p$ é o _padding_, $k$ o tamanho do filtro e $s$ o _stride_ (passo).

== Ativação ReLU

Após cada convolução aplica-se a não-linearidade *ReLU* (_Rectified Linear Unit_):
$ "ReLU"(x) = max(0, x). $
Ela zera valores negativos e mantém os positivos. Sem uma função não-linear, empilhar
camadas seria equivalente a uma única transformação linear — a não-linearidade é o
que permite à rede aprender relações complexas.

== _Max pooling_ (`MaxPool2d`)

O _pooling_ reduz a resolução espacial pegando o *valor máximo* de cada janela
$2 times 2$:
$ y_(i,j) = max_(0 <= m,n < 2) x_(2i+m, space 2j+n). $
Isso diminui o custo computacional, dá uma pequena *invariância a translações*
(a peça pode estar levemente deslocada) e resume a informação mais forte de cada
região.

== _Flatten_ e camadas totalmente conectadas (`Linear`)

Após os blocos convolucionais, o tensor $64 times 7 times 7$ é *achatado* em um vetor
de $3136$ números. Uma camada `Linear` (densa) faz uma combinação linear
$ y = W x + b, $
onde $W$ é a matriz de pesos e $b$ o viés. A primeira camada densa reduz para 128
neurônios (com ReLU); a última produz *10 logits*, um por classe.

== _Dropout_

Durante o treino, o `Dropout(0.5)` *desativa aleatoriamente 50%* dos neurônios a cada
passo. Isso evita que a rede dependa demais de neurônios específicos, funcionando como
*regularização* contra _overfitting_. Na avaliação, o dropout é desligado
automaticamente.

= Base matemática do treino

== _Softmax_ e logits

A rede devolve um vetor de 10 *logits* $z = (z_0, ..., z_9)$. A função *softmax* os
transforma em probabilidades que somam 1:
$ "softmax"(z)_i = e^(z_i) / (sum_(j=0)^9 e^(z_j)). $
A classe prevista é a de maior probabilidade, $hat(y) = arg max_i "softmax"(z)_i$.

== Função de perda: entropia cruzada

Mede o quão distante a distribuição prevista está do rótulo verdadeiro. Para um
exemplo cuja classe correta é $c$:
$ cal(L) = - log("softmax"(z)_c). $
Quanto maior a probabilidade atribuída à classe certa, menor a perda. No PyTorch, a
`CrossEntropyLoss` combina _softmax_ e _log_ em uma só operação numericamente estável
— por isso a rede emite logits, sem _softmax_ explícito.

== Retropropagação e gradiente descendente

O treino ajusta os pesos $theta$ para minimizar a perda. A cada lote:

+ *forward*: calcula as previsões e a perda $cal(L)$;
+ *backward* (_backpropagation_): usa a regra da cadeia para obter o gradiente
  $nabla_theta cal(L)$ (a direção de maior aumento da perda);
+ *atualização*: move os pesos no sentido contrário ao gradiente,
  $ theta <- theta - eta dot nabla_theta cal(L), $
  onde $eta$ é a *taxa de aprendizado* (_learning rate_, aqui $10^(-3)$).

== Otimizador Adam

Em vez do gradiente descendente puro, usamos o *Adam*, que adapta o passo de cada
parâmetro combinando estimativas da média (momento de 1ª ordem) e da variância
(2ª ordem) dos gradientes recentes. Na prática, converge mais rápido e é menos
sensível à escolha da taxa de aprendizado.

= Glossário de jargões

/ Época (_epoch_): uma passagem completa por todo o conjunto de treino.
/ Lote (_batch_): subconjunto de exemplos processados de uma vez antes de atualizar
  os pesos (aqui, 128 imagens).
/ Hiperparâmetro: valor definido *antes* do treino (taxa de aprendizado, tamanho do
  lote, nº de filtros), em contraste com os *pesos*, que são aprendidos.
/ _Overfitting_: quando o modelo decora o treino e generaliza mal para dados novos;
  combatido com _dropout_ e validação.
/ Regularização: técnicas que reduzem o _overfitting_ (ex.: _dropout_).
/ Logit: saída bruta da rede, antes do _softmax_.
/ Gradiente: vetor de derivadas parciais da perda em relação aos pesos.
/ _Data leakage_ (vazamento): usar informação do teste durante o treino/seleção,
  inflando a acurácia reportada.
/ Acurácia: fração de exemplos classificados corretamente.
/ Matriz de confusão: tabela que mostra, para cada classe real, como o modelo
  distribuiu suas previsões — útil para ver *quais* classes são confundidas.
/ AMP (_mixed precision_): uso de números de 16 bits para acelerar o treino em GPU.
/ _Baseline_: modelo mais simples usado como referência, para avaliar se um modelo mais
  complexo (aqui, a CNN) realmente compensa a complexidade extra.
/ Estratificação: dividir os dados preservando a proporção de cada classe em cada
  subconjunto (treino, validação, teste).
/ _Ensemble_: modelo composto pela combinação de vários modelos mais simples — a floresta
  aleatória, por exemplo, combina a previsão de várias árvores de decisão.

= Referências

- H. Xiao, K. Rasul, R. Vollgraf. *Fashion-MNIST: a Novel Image Dataset for Benchmarking
  Machine Learning Algorithms*. arXiv:1708.07747, 2017.
  #link("https://github.com/zalandoresearch/fashion-mnist")
- Y. LeCun, L. Bottou, Y. Bengio, P. Haffner. *Gradient-Based Learning Applied to Document
  Recognition*. Proceedings of the IEEE, 86(11):2278–2324, 1998. (arquitetura LeNet)
- A. Paszke et al. *PyTorch: An Imperative Style, High-Performance Deep Learning Library*.
  NeurIPS, 2019. #link("https://pytorch.org")
- F. Pedregosa et al. *Scikit-learn: Machine Learning in Python*. JMLR 12:2825–2830, 2011.
  #link("https://scikit-learn.org")
- D. P. Kingma, J. Ba. *Adam: A Method for Stochastic Optimization*. ICLR, 2015.
