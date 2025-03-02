#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('dataset.csv')


# In[3]:


df.dropna(how='all', inplace=True)
df['Area Privada'] = df['Area Privada'].str.extract('(\d+\.?\d*)').astype(float)


# In[4]:


df.info()


# In[5]:


df.to_csv('cleaned_dataset.csv', index=False)


# In[6]:


model_df = df.drop(columns=['Descripción', 'Facilities', 'ID', 'Estado'])


# In[7]:


model_df.head()


# In[8]:


model_df['Tipo de inmueble'].value_counts()


# In[9]:


model_df['Tipo de inmueble'] = model_df['Tipo de inmueble'].replace({
	'Apartaestudio': 'Apartamento',
	'Habitación': 'Apartamento'
})


# In[10]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Suponiendo que 'df' es tu DataFrame y 'precio' es la columna objetivo:
X = model_df.drop('Precio (admin_included)', axis=1)
y = model_df['Precio (admin_included)']

# Define las columnas numéricas y categóricas
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols)
    ])


# In[11]:


from sklearn.cluster import KMeans
import numpy as np

# Convierte el precio a un array 2D
precio_array = np.array(y).reshape(-1, 1)

# Define el número de clusters (por ejemplo, 3)
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(precio_array)

# Agrega el cluster al DataFrame
model_df['cluster_precio'] = clusters


# In[12]:


from sklearn.impute import SimpleImputer

# Añade un SimpleImputer al preprocesador para manejar los valores NaN
preprocessor = ColumnTransformer(
	transformers=[
		('num', Pipeline(steps=[
			('imputer', SimpleImputer(strategy='mean')),
			('scaler', StandardScaler())
		]), num_cols),
		('cat', Pipeline(steps=[
			('imputer', SimpleImputer(strategy='most_frequent')),
			('onehot', OneHotEncoder(handle_unknown='ignore'))
		]), cat_cols)
	])

# Aplica el preprocesamiento a todas las variables (incluyendo precio si lo consideras)
X_scaled = preprocessor.fit_transform(model_df.drop('Precio (admin_included)', axis=1))
# Si deseas incluir el precio, puedes concatenarlo:

X_clustering = np.hstack((X_scaled, np.array(y)[:, np.newaxis]))

kmeans_all = KMeans(n_clusters=4, random_state=42)
clusters_all = kmeans_all.fit_predict(X_clustering)
model_df['cluster_total'] = clusters_all


# In[13]:


model_df['cluster_precio'].value_counts()


# In[14]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Divide en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crea un pipeline que incluya el preprocesamiento y el modelo
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', RandomForestRegressor(random_state=42))])
pipeline.fit(X_train, y_train)

# Extrae la importancia de variables
importances = pipeline.named_steps['regressor'].feature_importances_

# Si tienes variables categóricas codificadas, obtén los nombres de las columnas resultantes:
encoded_cols = pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(cat_cols)
feature_names = num_cols + list(encoded_cols)

# Crea un DataFrame para visualizar la importancia
importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
importance_df = importance_df.sort_values(by='importance', ascending=False)
print(importance_df)


# In[15]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Supongamos que 'X_clustering' es tu matriz de características y 'clusters' es el vector de etiquetas
pca = PCA(n_components=2)
components = pca.fit_transform(X_clustering)

plt.figure(figsize=(8, 6))
plt.scatter(components[:, 0], components[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.title("Visualización de clusters usando PCA")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.colorbar(label="Cluster")
plt.show()


# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt

# Suponiendo que tu DataFrame 'df' ya tiene la columna 'cluster_precio'
plt.figure(figsize=(8, 6))
sns.boxplot(x='cluster_precio', y='precio', data=df)
plt.title("Distribución del Precio por Cluster")
plt.xlabel("Cluster")
plt.ylabel("Precio")
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# 'X_clustering' es tu matriz de datos preprocesada
Z = linkage(X_clustering, method='ward')

plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title("Dendrograma del Clustering Jerárquico")
plt.xlabel("Índice de Muestra")
plt.ylabel("Distancia")
plt.show()


# In[ ]:


import plotly.express as px
import pandas as pd

# Crear un DataFrame a partir de los componentes del PCA
df_viz = pd.DataFrame(components, columns=['PC1', 'PC2'])
df_viz['Cluster'] = clusters.astype(str)

fig = px.scatter(df_viz, x='PC1', y='PC2', color='Cluster',
                 title="Clusters visualizados en 2D con PCA")
fig.show()

