import pandas
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss, accuracy_score, precision_score, recall_score, \
    classification_report

def preprocess(file_name):
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
    num_features = ["duration", "protocol_type", "service", "flag", "src_bytes",
                    "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                    "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]
    kdd_data_10percent = pandas.read_csv(file_name, header=None, names=col_names)

    kdd_data_10percent['service'] = kdd_data_10percent['service'].astype('category')
    kdd_data_10percent['flag'] = kdd_data_10percent['flag'].astype('category')
    kdd_data_10percent['protocol_type'] = kdd_data_10percent['protocol_type'].astype('category')
    cat_columns = kdd_data_10percent.select_dtypes(['category']).columns
    kdd_data_10percent[cat_columns] = kdd_data_10percent[cat_columns].apply(lambda x: x.cat.codes)
    features = kdd_data_10percent[num_features].astype(float)
    features = features.values
    Y = kdd_data_10percent.values[0:410021, 41]
    X = features[0:410021, 0:41]
    Y = Y.reshape(-1, 1)
    sScaler = StandardScaler()
    rescaleX = sScaler.fit_transform(X)
    pca = PCA(n_components=2)
    rescaleX = pca.fit_transform(rescaleX)
    rescaleX = np.append(rescaleX, Y, axis=1)
    principalDf = pandas.DataFrame(data=rescaleX, columns=['principal component 1', 'principal component 2', 'target'])
    return principalDf
def main():
    with open("training_attack_types") as file:
        lines = [line.strip() +"." for line in file]
    data_tranning = preprocess("kddcup.data_10_percent")
    data_test = preprocess("corrected")
    data_tranning = data_tranning.values
    data_test = data_test.values
    train_X = data_tranning[:, 0:2]
    train_y = data_tranning[:, 2]
    test_X = data_test[:, 0:2]
    test_y = data_test[:, 2]
    print("Training and predicting")
    learner = KNeighborsClassifier(1, n_jobs=-1)
    learner.fit(train_X, train_y)
    pred_y = learner.predict(test_X)
    # result = confusion_matrix(test_y, pred_y, labels= lines)
    # print(pred_y)
    error = zero_one_loss(test_y, pred_y)
    acc = accuracy_score(test_y, pred_y)
    # pre = precision_score(test_y,pred_y)
    # recall = recall_score(test_y, pred_y)
    # print(result)
    print(error)
    print(acc)
    # print(recall)
    print(classification_report(test_y,pred_y, labels= lines, zero_division = 1))


if __name__ == '__main__':
    main()
