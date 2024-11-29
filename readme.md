# Relatório de Treinamento e Testes

**Projeto**: Agrupamento de Dados com PCA e K-Means  


---

## **Introdução**
Este relatório descreve o processo de treinamento e testes utilizando PCA (Análise de Componentes Principais) para redução de dimensionalidade e K-Means para agrupamento. O objetivo foi identificar padrões em um conjunto de dados de dois meses (Outubro e Novembro de 2019), realizando as etapas de limpeza, análise e visualização dos clusters.

---

## **Como rodar**
### **1. pré-requisitos**
- python -> [instalar](https://www.python.org/downloads/).
- planilhas -> [instalar](https://drive.google.com/file/d/10WcuTh0S7DojX1ZA1gUaUVlEQ8tCqsQz/view).
- instalar os imports.
~~~cmd
pip install pandas numpy matplotlib seaborn scikit-learn
~~~ 
### **2. rodando**
Coloque as planilhas junto ao arquivo em py e rode o projeto

---
## **Passo a Passo do Experimento**

### **1. Carregamento e Visualização dos Dados**
Os dados foram carregados de dois arquivos CSV, combinados em um único `DataFrame` para análise. 

- **Ação**: Utilizamos o `pd.concat` para mesclar os dados de outubro e novembro.
- **Resultados**: Verificação inicial dos dados (`head` e `info`):
  - Identificamos colunas categóricas e numéricas.
  - Algumas colunas possuíam valores ausentes.

---

### **2. Limpeza e Seleção de Características**
Decidimos focar em colunas numéricas para evitar ruídos no treinamento do modelo de clustering.

- **Ação**:
  - Seleção de colunas numéricas com `select_dtypes`.
  - Tratamento de valores ausentes substituindo-os pela média da coluna.
  - Padronização dos dados com `StandardScaler`.

- **Desafios**:
  - Algumas colunas categóricas apresentaram relevância potencial para o clustering. Testamos manter essas colunas com codificação (one-hot encoding), mas os resultados de agrupamento se mostraram menos coerentes.

---

### **3. Redução de Dimensionalidade com PCA**
A redução de dimensionalidade foi aplicada para minimizar ruídos e melhorar a eficiência do algoritmo de clustering.

- **Ação**:
  - Reduzimos as dimensões mantendo 95% da variância explicada.
  - O PCA reduziu os dados de *N* dimensões para **6 componentes principais**.

- **Resultados**:
  - O gráfico de variância acumulada demonstrou que 95% da variância foi explicada pelas primeiras 6 dimensões.

---

### **4. Determinação do Número de Clusters**
Testamos diferentes valores de `k` para K-Means (de 2 a 10 clusters) usando dois métodos:
- **Método do Cotovelo**: Avaliando a soma das distâncias ao quadrado dentro dos clusters.
- **Coeficiente de Silhueta**: Medida de qualidade do agrupamento.

- **Ação**:
  - Ajustamos o modelo para cada `k`.
  - Visualizamos os gráficos para identificar o ponto ideal.

- **Resultados**:
  - O cotovelo sugeriu **4 clusters** como ideal.
  - O coeficiente de silhueta foi consistente com essa escolha, apresentando valores elevados para 4 clusters.

---

### **5. Treinamento Final com K-Means**
Baseado na análise anterior, aplicamos o K-Means com 4 clusters.

- **Ação**:
  - Ajustamos o modelo aos dados PCA.
  - Adicionamos os rótulos de cluster ao conjunto original.

- **Resultados**:
  - Perfis médios dos clusters foram calculados, identificando características distintas entre os grupos.
  - Foi possível diferenciar perfis como "alta atividade" e "baixa atividade".

---

### **6. Visualização dos Clusters**
Visualizamos os clusters no espaço bidimensional (2 componentes principais).

- **Ação**:
  - Plotamos os dados coloridos pelos clusters.
  - Destacamos os centróides dos clusters no gráfico.

- **Resultados**:
  - Os clusters apresentaram separação clara no espaço bidimensional.

---

## **Desafios e Soluções**
1. **Valores ausentes**:
   - Solução: Substituição pela média para manter a coesão do conjunto de dados.

2. **Alta dimensionalidade**:
   - Solução: Utilização de PCA para redução de dimensões.

3. **Colunas categóricas**:
   - Solução: Optamos por removê-las após verificar que a codificação aumentava a complexidade sem melhorar os resultados.

4. **Escolha de `k`**:
   - Solução: Combinação dos métodos do cotovelo e coeficiente de silhueta.

---

## **Resultados Finais**
- Número de clusters: 4.
- Perfis dos clusters:
  - Cluster 0: Caracterizado por valores médios baixos em todas as variáveis.
  - Cluster 1: Alta atividade em algumas variáveis principais.
  - Cluster 2: Perfil moderado, intermediário em relação aos outros clusters.
  - Cluster 3: Valores consistentemente altos, sugerindo maior atividade.

- Visualização dos clusters foi bem-sucedida com clara separação entre os grupos.

---

## **Conclusões**

- **Lições aprendidas**:
  - A padronização é essencial para algoritmos sensíveis a escala, como K-Means.
  - PCA simplifica a análise e melhora a separação dos clusters.
  - A análise combinada de métodos de avaliação de `k` gera melhores resultados.

- **Melhorias sugeridas**:
  1. Experimentar outros algoritmos de clustering (como DBSCAN) para avaliar padrões não lineares.
  2. Incluir mais variáveis categóricas relevantes utilizando codificação apropriada.
  3. Automatizar a análise de variância explicada no PCA para facilitar a seleção de componentes.

---

**Autor**: *Seu Nome*  
**Ferramentas Utilizadas**: Python, Pandas, Scikit-learn, Matplotlib, NumPy
