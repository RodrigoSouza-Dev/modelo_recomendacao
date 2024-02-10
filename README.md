Modelo de recomendação por imagem

# Instalação de dependências:
Esta seção instala o pacote kaggle usando o pip, cria um diretório .kaggle no diretório do usuário e configura as credenciais do Kaggle para autenticação usando um arquivo kaggle.json.

# Download do conjunto de dados: 
Utiliza o comando kaggle datasets download para baixar o conjunto de dados "fashion-product-images-small" do Kaggle.

# Descompactação do conjunto de dados: 
Descompacta o arquivo zip do conjunto de dados recém-baixado usando o comando unzip.

# Pré-processamento dos dados: 
Carrega o arquivo styles.csv em um DataFrame Pandas, selecionando apenas as colunas 'id' e 'masterCategory'. Em seguida, organiza as imagens por categoria, movendo os arquivos de imagem para pastas separadas de acordo com a categoria à qual pertencem.

# Treinamento do modelo: 
Utiliza a biblioteca TensorFlow e TensorFlow Hub para criar um modelo de aprendizado profundo. Configura os geradores de dados de imagem usando a classe ImageDataGenerator para carregar e pré-processar as imagens. Em seguida, define a arquitetura do modelo usando uma camada de entrada, uma camada de pré-processamento do TensorFlow Hub, camadas de dropout para regularização e camadas densas para classificação. Compila o modelo com otimizador "adam" e função de perda "categorical_crossentropy" e o treina usando os dados de treinamento e validação.

