import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.cluster import KMeans 

df =  pd.read_csv('Mall_Customers.csv')

df.rename(index=str, columns={
    'Annual Income (k$)' : 'Income',
    'Spending Score (1-100)' : 'Score'
}, inplace=True)

X = df.drop(['CustomerID','Gender'], axis=1)

st.header("Isi DataSet")
st.write(X) 

#panah elbow
clusters=[]
for i in range(1,11):
    km = KMeans(n_clusters=i).fit(X)
    clusters.append(km.inertia_)
    
fig, ax = plt.subplots(figsize=(12,8))
sns.lineplot(x=list(range(1,11)), y=clusters, ax=ax)
ax.set_title('Pencarian Elbow')
ax.set_xlabel('clusters')
ax.set_ylabel('inertia')

#Panah elbow
ax.annotate('Possible elbow points', xy=(3, 140000), xytext=(3, 50000), xycoords='data',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

ax.annotate('Possible elbow points', xy=(5, 80000), xytext=(5, 150000), xycoords='data',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

st.set_option('deprecation.showPyplotGlobalUse', False)
elbow_plot = st.pyplot()

st.sidebar.subheader("Nilai Jumblah Klastering")
clust = st.sidebar.slider("Pilih Jumblah Kluster:", 2,10,3,1)

def k_means(n_clust):
    kmean = KMeans(n_clusters=n_clust).fit(X)
    X['Labels'] = kmean.labels_
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Income', y='Score', hue='Labels', size='Labels', data=X, palette=sns.color_palette('hls', n_clust))

# Mengannotasi titik rata-rata dari setiap label
    for label in X['Labels'].unique():
        plt.annotate(label,
                (X[X['Labels'] == label]['Income'].mean(),
                X[X['Labels'] == label]['Score'].mean()),
                horizontalalignment='center',
                verticalalignment='center',
                size=20, weight='bold',
                color='black')

    plt.show()
    
    st.header('Cluster Plot')
    st.pyplot()
    st.write(X)
    
k_means(clust)