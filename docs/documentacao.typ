// Documentação complementar — Classificação do MNIST com CNN
// Compilar com:  typst compile documentacao.typ
// (ou `typst watch documentacao.typ` para recompilar ao salvar)

#set document(title: "MNIST + CNN — Documentação", author: "Arthur Grazzia")
#set page(numbering: "1", margin: 2.2cm)
#set text(font: "New Computer Modern", size: 11pt, lang: "pt")
#set par(justify: true, leading: 0.68em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => { v(0.4em); it; v(0.2em) }
#show link: set text(fill: blue)
#show raw: set text(font: "DejaVu Sans Mono", size: 9.5pt)

#align(center)[
  #text(18pt, weight: "bold")[Classificação de Dígitos Manuscritos (MNIST)] \
  #text(14pt)[com uma Rede Neural Convolucional] \
  #v(0.3em)
  #text(11pt, style: "italic")[Documentação complementar — Fundamentos de IA]
]

#v(0.5em)
#outline(title: "Sumário", indent: auto)
#line(length: 100%, stroke: 0.5pt + gray)

= O dataset MNIST

O *MNIST* (_Modified National Institute of Standards and Technology_) é o conjunto
de dados mais clássico da área de reconhecimento de imagens. Contém *70.000 imagens*
de dígitos manuscritos (0 a 9):

- *60.000* imagens para *treino* e *10.000* para *teste*;
- cada imagem é *28 × 28 pixels*, em *tons de cinza* (1 canal), com valores de
  intensidade de 0 (preto) a 255 (branco);
- as classes são os dez dígitos, e o dataset é *aproximadamente balanceado*
  (~6.000 exemplos por dígito no treino).

== Pré-processamento

Antes de entrar na rede, cada imagem passa por duas transformações:

+ *`ToTensor`*: converte a imagem em um tensor de números reais na faixa $[0, 1]$
  (divide por 255);
+ *`Normalize`*: recentraliza os pixels usando a média $mu = 0{.}1307$ e o desvio
  padrão $sigma = 0{.}3081$ do MNIST, aplicando
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
(o dígito pode estar levemente deslocado) e resume a informação mais forte de cada
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
  distribuiu suas previsões — útil para ver *quais* dígitos são confundidos.
/ AMP (_mixed precision_): uso de números de 16 bits para acelerar o treino em GPU.
