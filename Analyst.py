import numpy as np
import pandas
from time import time
import matplotlib.pyplot as plt
import collections, numpy
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, zero_one_loss, accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def count_data_in_label(data_set):
    k = 4
    kmeans = KMeans(n_clusters=k)
    t0 = time()
    kmeans.fit(data_set)
    tt = time() - t0
    print("Clustered in {} seconds".format(round(tt, 3)))
    print(pandas.Series(kmeans.labels_).value_counts())
    labels = data_set['target']
    label_names = list(map(
        lambda x: pandas.Series([labels[i] for i in range(len(kmeans.labels_)) if kmeans.labels_[i] == x]),
        range(k)))
    for i in range(k):
        print("Cluster {} labels:".format(i))
        print(label_names[i].value_counts())


def caculator_K(data_set):
    Sum_of_squared_distances = []
    K = range(1,15)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(data_set)
        Sum_of_squared_distances.append(km.inertia_)
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()


def preprocess_1(file_name):
    col_names = ["duration", "protocol_type", "service", "flag", "src_bytes",
                 "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                 "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                 "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                 "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                 "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                 "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                 "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                 "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                 "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]
    kdd_data_10percent = pandas.read_csv(file_name, header=None, names=col_names)
    kdd_data_10percent.describe()
    num_features = ["duration", "protocol_type", "service", "flag", "src_bytes",
                    "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]
    kdd_data_10percent['service'] = kdd_data_10percent['service'].astype('category')
    kdd_data_10percent['flag'] = kdd_data_10percent['flag'].astype('category')
    kdd_data_10percent['protocol_type'] = kdd_data_10percent['protocol_type'].astype('category')
#   kdd_data_10percent['label'] = kdd_data_10percent['label'].astype('category')
    cat_columns = kdd_data_10percent.select_dtypes(['category']).columns
    kdd_data_10percent[cat_columns] = kdd_data_10percent[cat_columns].apply(lambda x: x.cat.codes)
    features = kdd_data_10percent[num_features].astype(float)
    # labels = kdd_data_10percent['label'].copy()
    features = features.values
    Y = features[:, 41]
    X = features[:, 0:41]
    Y = Y.reshape(-1, 1)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    sScaler = StandardScaler()
    rescaleX = sScaler.fit_transform(X)
    pca = PCA(n_components=2)
    rescaleX1 = pca.fit_transform(rescaleX)
    rescaleX = np.append(rescaleX1, Y, axis=1)

    principalDf = pandas.DataFrame(data=rescaleX, columns=['principal component 1', 'principal component 2', 'target'])
    return principalDf,rescaleX1
    # plt.title('KDD data set - K-means')
    # plt.xlabel('pc1')
    # plt.ylabel('pc2')
    # plt.scatter(principalDf['principal component 1'],principalDf['principal component 2'], s=50)

# def preprocess(file_name):
#     col_names = ["duration", "protocol_type", "service", "flag", "src_bytes",
#                  "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
#                  "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
#                  "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
#                  "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
#                  "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
#                  "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
#                  "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
#                  "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
#                  "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]
#     kdd_data_10percent = pandas.read_csv(file_name, header=None, names=col_names)
#     kdd_data_10percent.describe()
#     num_features = ["duration", "protocol_type", "service", "flag", "src_bytes",
#                     "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
#                     "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
#                     "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
#                     "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
#                     "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
#                     "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
#                     "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
#                     "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
#                     "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]
#     kdd_data_10percent['service'] = kdd_data_10percent['service'].astype('category')
#     kdd_data_10percent['flag'] = kdd_data_10percent['flag'].astype('category')
#     kdd_data_10percent['protocol_type'] = kdd_data_10percent['protocol_type'].astype('category')
#     kdd_data_10percent['label'] = kdd_data_10percent['label'].astype('category')
#     cat_columns = kdd_data_10percent.select_dtypes(['category']).columns
#     kdd_data_10percent[cat_columns] = kdd_data_10percent[cat_columns].apply(lambda x: x.cat.codes)
#     features = kdd_data_10percent[num_features].astype(float)
#     # labels = kdd_data_10percent['label'].copy()
#     features = features.values
#     Y = features[0:410021, 41]
#     X = features[0:410021, 0:41]
#     Y = Y.reshape(-1, 1)
#     from sklearn.decomposition import PCA
#     from sklearn.preprocessing import StandardScaler
#     sScaler = StandardScaler()
#     rescaleX = sScaler.fit_transform(X)
#     pca = PCA(n_components=2)
#     rescaleX = pca.fit_transform(rescaleX)
#     rescaleX = np.append(rescaleX, Y, axis=1)
#
#     principalDf = pandas.DataFrame(data=rescaleX, columns=['principal component 1', 'principal component 2', 'target'])
#     return principalDf
#     # plt.title('KDD data set - K-means')
#     # plt.xlabel('pc1')
#     # plt.ylabel('pc2')
#     # plt.scatter(principalDf['principal component 1'],principalDf['principal component 2'], s=50)

def display_cluster(data_set, rs):
    # k = 4
    # kmeans = KMeans(n_clusters=k)
    k = 4
    kmeans = KMeans(n_clusters=k)
    t0 = time()
    kmeans.fit(data_set)
    tt = time() - t0
    # print("Clustered in {} seconds".format(round(tt, 3)))
    # print(pandas.Series(kmeans.labels_).value_counts())
    # labels = data_set['target']
    # label_names = list(map(
    #     lambda x: pandas.Series([labels[i] for i in range(len(kmeans.labels_)) if kmeans.labels_[i] == x]),
    #     range(k)))
    # for i in range(k):
    #     print("Cluster {} labels:".format(i))
    #     print(label_names[i].value_counts())

    # kmeans.fit(data_set)
    # name = ['Normal', 'Probe', 'DoS', 'U2R', 'R2L']
    # plt.title('KDD data set - K-means')
    # plt.xlabel('pc1')
    # plt.ylabel('pc2')
    # for i in range(len(name)):
    #     bucket = data_set[data_set['target'] == i]
    #     bucket = bucket.iloc[:,[0,1]].values
    #     # print(bucket)
    # #     plt.scatter(bucket[:, 0], bucket[:, 1], label=name[i])
    # #     plt.legend(loc='upper left',
    # #               fontsize=8)
    # # plt.show()

    y_kmeans = kmeans.predict(data_set)
    plt.scatter(data_set['principal component 1'], data_set['principal component 2'], c=y_kmeans, s=50, cmap='viridis')

    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
    plt.show()

    # Label



def main():
    df, rs = preprocess_1("kddcup.data_10_percent_ver1.txt")
    count_data_in_label(df)

if __name__ == '__main__':
    main()
