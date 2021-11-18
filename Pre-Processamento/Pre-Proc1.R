# Definindo o diretório de trabalho
setwd("D:/DSA/7.PreparacaoCientistaDados/Cap03_Portfolio_Projetos/Projeto1/Pre-Processamento")
getwd()

# Pacotes
install.packages("slam")
install.packages("tm")
install.packages("SnowballC")
install.packages("wordcloud")
install.packages("gmodels")
library(tm)
library(SnowballC)
library(wordcloud) # Construir uma nuvem de palavras
library(e1071) # É o algorítmo que contém o Naive Bayes
library(gmodels) # Para construirmos a Confusion Matrix

# Carregando os dados
dados <- read.csv("sms_spam.csv", stringsAsFactors = FALSE)

# Examinando a estrutura dos dados
# Temos 5.559 observações e 2 variáveis (colunas)
str(dados)

# Convertendo para fator
dados$type <- factor(dados$type)

# Examinando a estrutura dos dados
# Como convertemos para fator, passamos a ter os níveis "ham" e "spam"
str(dados$type)
table(dados$type) # Temos 4812 registros com ham e 747 com spam

# Construindo um Corpus
# Corpus ---> É um conjunto de documentos de textos
dados_corpus <- VCorpus(VectorSource(dados$text))

# Examinando a estrutura dos dados
# Quanto temos objeto do tipo corpus, podemos utilizar a função inspect, para analisar esse conjunto de dados
print(dados_corpus)
inspect(dados_corpus[1:2])

# Ajustando a estrutura
as.character(dados_corpus[[1]])
lapply(dados_corpus[1:2], as.character)

# Limpeza do Corpus com tm_map() ---> faz transformações no corpus, pertence ao pacote tm
?tm_map
# tolower --> vou converter todos os caracteres para minúsculo.
dados_corpus_clean <- tm_map(dados_corpus, content_transformer(tolower))

# Diferenças entre o Corpus inicial e o Corpus após a limpeza
as.character(dados_corpus[[1]])
as.character(dados_corpus_clean[[1]])

# Outras etapas de limpeza
dados_corpus_clean <- tm_map(dados_corpus_clean, removeNumbers) # remove números
dados_corpus_clean <- tm_map(dados_corpus_clean, removeWords, stopwords()) # remove stop words
dados_corpus_clean <- tm_map(dados_corpus_clean, removePunctuation) # remove pontuação

# Criando uma função para substituir ao invés de remover pontuação
removePunctuation("hello...world") # quando eu removo a pontuação transformo em uma
# uma única palavra: "helloworld". Nesse caso, não é o ideal.

replacePunctuation <- function(x) { gsub("[[:punct:]]+", " ", x) }
replacePunctuation("hello...world") # aqui substitui os 3 pontos por um espaço vazio e tive
# como resultado: "hello world"

# Word stemming (Tratar palavras que são muito parecidas)
?wordStem
wordStem(c("learn", "learned", "learning", "learns")) # Essas palavras representam a mesma informação,
# estão apenas em tempos verbais diferentes
# Converteu as 4 palavras para a mesma palavra.
# Resultado: "learn" "learn" "learn" "learn"


# Aplicando Stem (aqui fazemos a mesma coisa com a função tm_map)
# Vamos aplicar a função tm_map com stemDocument
dados_corpus_clean <- tm_map(dados_corpus_clean, stemDocument)

# Eliminando espaço em branco (espaços duplos ou no final de uma frase, ou no final de uma palavra.
# Ou espaços que aparecem mais de uma vez).
dados_corpus_clean <- tm_map(dados_corpus_clean, stripWhitespace)

# Examinando a versão final do Corpu
# Vamos comparar os dados antes e depois da limpeza:
lapply(dados_corpus[1:3], as.character) # antes da limpeza
lapply(dados_corpus_clean[1:3], as.character) # depois da limpeza


## Criando uma matriz esparsa document-term ###
# Matriz esparsa --> as linhas são os termos e as colunas são os documentos.
?DocumentTermMatrix

# Solução alternartiva 1 (eu só aplico a função DocumentTermMatrix)
dados_dtm <- DocumentTermMatrix(dados_corpus_clean)

# Solução alternartiva 2 - cria uma matriz esparsa  document-term direto a partir do Corpus
# Crio alguns parâmetros
dados_dtm2 <- DocumentTermMatrix(dados_corpus, control = list(tolower = TRUE,
                                                              removeNumbers = TRUE,
                                                              stopwords = TRUE,
                                                              removePunctuation = TRUE,
                                                              stemming = TRUE))

# Solução alternativa 3 - usando stop words customizadas a partir da função
# Estamos usando uma função específica para remover as stop words:
# stopwords = function(x) { removeWords(x, stopwords()) }
dados_dtm3 <- DocumentTermMatrix(dados_corpus, control = list(tolower = TRUE,
                                                              removeNumbers = TRUE,
                                                              stopwords = function(x) { removeWords(x, stopwords()) },
                                                              removePunctuation = TRUE,
                                                              stemming = TRUE))

# Comparando os 3 resultados: (as 3 alternativas acima).
# As alternativas 1 e 3 geraram o mesmo resultado.
# Mas, como a alternativa 1 é a mais fácil, vamos usar essa.
dados_dtm
dados_dtm2 # Esse resultado conseguiu aumentar consideravelmente o número de termos.
# Isso provavelmente indica que ele não conseguiu remover as stop words.
# Pois, aumentou de 6582 para 6971. Então, esse procedimento foi o pior entre os 3.
dados_dtm3


# Criando datasets de treino e de teste (Isso com os atributos, variáveis preditoras)
# Faço um subset
dados_dtm_train <- dados_dtm[1:4169, ]
dados_dtm_test  <- dados_dtm[4170:5559, ]

# Labels (variável target)
# A variável target está na coluna type, que nós convertemos para fator lá no início do script.
dados_train_labels <- dados[1:4169, ]$type
dados_test_labels  <- dados[4170:5559, ]$type

# Verificando se a proporção de Spam é similar
# Vamos usar a função prop.table para comparar a proporção
prop.table(table(dados_train_labels))
prop.table(table(dados_test_labels))

# Word Cloud (Nuvem de palavras)
wordcloud(dados_corpus_clean, min.freq = 50, random.order = FALSE)

# Frequência dos dados
# Coletar apenas as palavras que aparecem com mais frequência (vamos estar diminuindo a nossa matriz esparsa)
# Função removeSparseTerms
# Vamos colocar essas palavras num índice, que nesse caso é o 0.999
sms_dtm_freq_train <- removeSparseTerms(dados_dtm_train, 0.999)
sms_dtm_freq_train

# Indicador de Features para palavras frequentes
findFreqTerms(dados_dtm_train, 5)

# save frequently-appearing terms to a character vector
# E utilizamos essas palavras mais frequentes para gerar um novo dataset.
sms_freq_words <- findFreqTerms(dados_dtm_train, 5)
str(sms_freq_words)

# A partir desse conjunto de dados, das palavras mais frequentes, vamos criar uma divisão em dados de treino e de teste.
# Criando subsets apenas com palavras mais frequentes
sms_dtm_freq_train <- dados_dtm_train[ , sms_freq_words]
sms_dtm_freq_test <- dados_dtm_test[ , sms_freq_words]

# Converte para fator
convert_counts <- function(x) {
  print(x)
  x <- ifelse(x > 0, "Yes", "No")
}

# apply() converte counts para colunas de dados de treino e de teste
sms_train <- apply(sms_dtm_freq_train, MARGIN = 2, convert_counts)
sms_test  <- apply(sms_dtm_freq_test, MARGIN = 2, convert_counts)

str(sms_test)


# Treinando o modelo
?naiveBayes
# sms_train ---> conjunto de dados de treino com os atributos
# dados_train_labels ---> conjunto de dados de treino com os labels, que é a variável target
nb_classifier <- naiveBayes(sms_train, dados_train_labels)

# Avaliando o modelo
# Apresentamos o conjunto de testes ao nosso modelo preditivo
sms_test_pred <- predict(nb_classifier, sms_test)

# Confusion Matrix
CrossTable(sms_test_pred,
           dados_test_labels,
           prop.chisq = FALSE,
           prop.t = FALSE,
           prop.r = FALSE,
           dnn = c('Previsto', 'Observado'))


# Veja que nosso classificador executou com um excelente nível de performance, de acurácia de 99,5% para ham e 83,6% para spam.
# Podemos melhorar isso? Podemos.

### Frequência Zero ---> adiciona o contador = 1. ###
# Um problema comum quando se trabalha com Naive Bayes é a Fequência Zero.
# Frequência Zero, ou seja, quando fizemos a divisão entre treino e teste, algumas palavras podem ter ido
# para os dados de teste, mas não foram para os conjuntos de dados de treino.
# O que acontece é que durante o treino o algorítmo não viu aquela palavra. Não sabe se aquela palavra é spam ou ham.
# Qd vai para o conjunto de dados de teste, ele encontra aquela palavra e vai dar uma probabilidade zero.
# Não aprendeu a classificar aquela palavra durante o treinamento.

# Podemos corrigir esse problema de Frequência Zero aplicando o estimador laplace.
# Na verdade é uma suavização. Quando ele não encontrar uma palavra, ao invés de colocar a probabilidade zero
# ele adiciona o valor 1. Isto é, ele não vai colocar a probabilidade 1, ele vai colocar o contador 1.

# Vou criar a versão 2 do meu modelo:
# Melhorando a performance do modelo aplicando suavização laplace
nb_classifier_v2 <- naiveBayes(sms_train, dados_train_labels, laplace = 1)

# Avaliando o modelo
sms_test_pred2 <- predict(nb_classifier_v2, sms_test)

# Confusion Matrix
CrossTable(sms_test_pred2,
           dados_test_labels,
           prop.chisq = FALSE,
           prop.t = FALSE,
           prop.r = FALSE,
           dnn = c('Previsto', 'Observado'))

# Nota: Para fazer novas previsões com o modelo treinado, gere uma nova massa de dados
# e use a função predict().

