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

# Resumo do PCA
print("Resumo do PCA:")
print(f"Componentes principais selecionados: {pca.n_components_}")
print(f"Variância explicada por componente: {pca.explained_variance_ratio_}")
print(f"Variância total explicada: {sum(pca.explained_variance_ratio_):.2f}")

# 4. Determinar o número ideal de clusters
# Método do cotovelo e coeficiente de silhueta
distortions = []
silhouettes = []
range_clusters = range(2, 11)

for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=0)
    y_kmeans = kmeans.fit_predict(features_pca)
    distortions.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(features_pca, y_kmeans))

# Determinar o número de clusters com o maior coeficiente de silhueta
optimal_clusters = range_clusters[np.argmax(silhouettes)]
print("\nNúmero ideal de clusters:")
print(f"Baseado no coeficiente de silhueta, o número ideal de clusters é {optimal_clusters}.")

# 5. Aplicar K-Means com o número ideal de clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=0)
y_kmeans = kmeans.fit_predict(features_pca)

# Adicionar os rótulos ao dataset original
data['Cluster'] = y_kmeans

# 6. Analisar os clusters
cluster_profiles = data.groupby('Cluster').mean()
print("\nPerfis dos Clusters:")
print(cluster_profiles)




