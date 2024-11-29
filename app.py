import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# 1. Carregar os dados
# Ler os arquivos CSV e concatenar os dados
nov_data = pd.read_csv('2019-Nov.csv')
oct_data = pd.read_csv('2019-Oct.csv')

# Combinar os dados
data = pd.concat([nov_data, oct_data], axis=0)

# Visualizar a estrutura dos dados
print(data.head())
print(data.info())

# 2. Limpeza e Seleção de Características
# Remover colunas irrelevantes ou categóricas não tratadas
# (Substitua 'Coluna1', 'Coluna2' pelas colunas relevantes no seu conjunto de dados)
features = data.select_dtypes(include=[np.number])  # Apenas colunas numéricas

# Verificar valores ausentes
print(features.isnull().sum())

# Substituir ou descartar valores ausentes (estratégia depende do seu caso)
features = features.fillna(features.mean())  # Substituir NaN pela média da coluna

# Padronizar os dados (necessário para PCA e K-Means)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 3. Redução de Dimensionalidade com PCA
# Reduzir para componentes principais que expliquem 95% da variância
pca = PCA(n_components=0.95)
features_pca = pca.fit_transform(features_scaled)

# Quantos componentes foram selecionados?
print(f"Componentes principais selecionados: {pca.n_components_}")
print(f"Variância explicada por componente: {pca.explained_variance_ratio_}")

# 4. Determinar o número ideal de clusters
# Método do cotovelo e coeficiente de silhueta
distortions = []
silhouettes = []
range_clusters = range(2, 11)

for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=0)
    y_kmeans = kmeans.fit_predict(features_pca)
    distortions.append(kmeans.inertia_)  # Soma das distâncias ao quadrado
    silhouettes.append(silhouette_score(features_pca, y_kmeans))

# Plotar o método do cotovelo
plt.figure(figsize=(8, 4))
plt.plot(range_clusters, distortions, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de clusters')
plt.ylabel('Distortion')
plt.grid()
plt.show()

# Plotar o coeficiente de silhueta
plt.figure(figsize=(8, 4))
plt.plot(range_clusters, silhouettes, marker='o')
plt.title('Coeficiente de Silhueta')
plt.xlabel('Número de clusters')
plt.ylabel('Silhouette Score')
plt.grid()
plt.show()

# 5. Aplicar K-Means com o número ideal de clusters
# Baseado nos gráficos anteriores, escolha o número ideal de clusters (exemplo: 4)
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
y_kmeans = kmeans.fit_predict(features_pca)

# Adicionar os rótulos ao dataset original
data['Cluster'] = y_kmeans

# 6. Analisar os clusters
# Estatísticas descritivas por cluster
cluster_profiles = data.groupby('Cluster').mean()
print("Perfis de clusters:")
print(cluster_profiles)

# 7. Visualização (se possível, em 2D ou 3D usando PCA)
plt.figure(figsize=(8, 6))
sample_data = features_pca[:1000]  # Exemplo com as primeiras 1000 observações
plt.scatter(sample_data[:, 0], sample_data[:, 1], c=y_kmeans[:1000], cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=200, c='red', marker='*', label='Centroids')
plt.title('Clusters dos Clientes')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()
plt.grid()
plt.show()

