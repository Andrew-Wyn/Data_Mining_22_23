from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

#The following function prepares the data, removing all the categorical features and transforming the created_at feature in timestamp. Finally, it returns the result of train_test_split on the transformed dataset.
def prepare_data(data):
    # categorical_features = ["lang", "bot", "created_at", "name"]
    categorical_features = ["lang", "bot", "name"]

    # remove categorical variables
    classification_features = list(data.columns).copy()

    for feat in categorical_features:
    # for feat in ["lang", "bot", "created_at", "name", "reply_count_entropy", "favorite_count_entropy"]:
        classification_features.remove(feat)

    print(f"Features : {classification_features}")

    # convert datetime to timestamp to permit classification
    data["created_at"] = pd.to_datetime(data.created_at).values.astype(np.int64) // 10 ** 9

    data_classification = data[classification_features]
    data_label = data.pop("bot")

    train_set, test_set, train_label, test_label = train_test_split(data_classification, data_label, stratify =data_label, test_size=0.30, random_state=42)

    return train_set, test_set, train_label, test_label