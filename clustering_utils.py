import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi


def plot_parallel_coordinates_clusters(df, cluster_centers, method=None):
    """
    Function that plot the coordinates of the centroids in a parallel fashion way.
    """
    plt.figure(figsize=(8, 4))
    for i in range(0, len(cluster_centers)):
        plt.plot(cluster_centers[i], marker='o', label='Cluster %s' % i)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.xticks(range(0, len(df.columns)), df.columns, fontsize=18, rotation=90)
    plt.legend(fontsize=10)
    plt.savefig(f"images/clustering/{method}_parallel.png")
    plt.show()


def plot_radar_clusters(df, cluster_centers, method=None):
    """
    Function that plot the coordinates of the centroids in a radar fashion way.
    """
    # number of variable
    N = len(df.columns)
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    for i in range(0, len(cluster_centers)):
        angles = [n / float(N) * 2 * pi for n in range(N)]
        values = cluster_centers[i].tolist()
        values += values[:1]
        angles += angles[:1]
        # Initialise the spider plot
        ax = plt.subplot(polar=True)
        # Draw one axe per variable + add labels labels yet
        plt.xticks(angles[:-1], df.columns, color='grey', size=8)
        # Plot data
        ax.plot(angles, values, linewidth=1, linestyle='solid')
        # Fill area
        ax.fill(angles, values, 'b', alpha=0.1)
    plt.savefig(f"images/clustering/{method}_radar.png")
    plt.show()


def plot_date_feature_clusters(df, labels, num_cluster):
    """
    Function that plot the number of the profiles conditioned by the creation date, in an histogram way
    """
    for i in range(num_cluster):
        i_samples = labels==i
        print(i)
        print(np.sum(i_samples))
        df[i_samples].name.groupby(pd.to_datetime(df.loc[i_samples, 'created_at'], format='%Y-%m-%d %H:%M:%S', errors='raise').dt.year).count().plot(kind="bar")
        plt.show()


def categorical_hist_clusters(df, labels, feature_name, method=None):
    """
    Function that plot the number of the profiles conditioned by the value of certain feature
    """
    bot_xt_pct = pd.crosstab(labels, df[feature_name])
    bot_xt_pct.plot(kind='bar', stacked=False, 
                       title=f'{feature_name} per cluster')
    plt.xlabel('Cluster')
    plt.ylabel(feature_name)
    plt.savefig(f"images/clustering/{method}_{feature_name}_hist.png")
    plt.show()


def plot_numerical_features_clusters(df, labels, num_cluster):
    """
    Funciton that print and plot distribution of numerical features conditioned by the cluster
    """
    for i in range(num_cluster):
        i_samples = labels==i
        df[i_samples].boxplot()
        plt.xticks(rotation=90)
        df[i_samples].hist(figsize=(10,10))
        plt.xticks(rotation=90)
        plt.show()





def scatter_features_clusters(df, labels):
    """
    Funciton that print and plot distribution of numerical features conditioned by the cluster
    """
    colors = plt.cm.jet(np.linspace(0,1,len(set(labels))))

    for i in range(len(df.columns)):
        for j in range(i+1, len(df.columns)):
            feature_1 = df.columns[i]
            feature_2 = df.columns[j]

            print(f"{feature_1} - {feature_2}")
            for label in set(labels):
                x = df[feature_1][labels == label]
                y = df[feature_2][labels == label]
                color = labels[labels == label]
                plt.scatter(x, y, c=np.array([colors[label]]), label=str(label))

            plt.legend()
            plt.tick_params(axis='both', which='major', labelsize=22)
            plt.show()


def preprocess_skewed_features(df, skewed_features):
    df_num_not_skewed = df.copy()
    df_num_not_skewed[skewed_features] = df[skewed_features].apply(lambda x: np.log(x + 1))

    return df_num_not_skewed


def reverse_log_skewed(X, df, skewed_features):
    for feature in skewed_features:
        i = list(df.columns).index(feature)
        X[:, i] = np.exp(X[:, i]) - 1
    return X
