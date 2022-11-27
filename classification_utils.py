from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt

#metrics computed on the test set
def report_scores(test_label, test_pred):
    print(classification_report(test_label, 
                            test_pred, 
                            target_names=['<=50', '>50']))


def print_metrics(train_label, train_pred, test_label, test_pred):
    #evaulate the accuracy on the train set and the test set
    #metrics also contains precision, recall, f1 and the support
    print('Accuracy train set ', metrics.accuracy_score(train_label, train_pred))
    print('Accuracy test set ', metrics.accuracy_score(test_label, test_pred))
    print('Precision train set ', metrics.precision_score(train_label, train_pred, average='weighted'))
    print('Recall train set ', metrics.recall_score(train_label, train_pred, average='weighted'))
    print('F1 score train set ', metrics.f1_score(train_label, train_pred, average='weighted'))
    print('Support train set ', metrics.precision_recall_fscore_support(train_label, train_pred))


#function to discretize the variables
#input: the dataset and the list of variables' names to discretize
def discretize_data(dataset, variables):
    for variable in variables:
        #get the unique variable's values
        var = sorted(dataset[variable].unique())
        
        #generate a mapping from the variable's values to the number representation  
        mapping = dict(zip(var, range(0, len(var) + 1)))

        #add a new colum with the number representation of the variable
        dataset[variable+'_num'] = dataset[variable].map(mapping).astype(int)
    return dataset


def prepare_data(data):
    categorical_features = ["lang", "bot", "created_at", "name"]

    # remove categorical variables
    classification_features = list(data.columns).copy()

    for feat in categorical_features:
    # for feat in ["lang", "bot", "created_at", "name", "reply_count_entropy", "favorite_count_entropy"]:
        classification_features.remove(feat)
        
    print(f"Classification features : {classification_features}")

    data_classification = data[classification_features]
    data_label = data.pop("bot")
    
    train_set, test_set, train_label, test_label = train_test_split(data_classification, data_label, stratify =data_label, test_size=0.30)

    return train_set, test_set, train_label, test_label


def plot_lang_hist(data):
    bot_xt_pct = pd.crosstab(data.lang, data["bot"])
    bot_xt_pct.plot(kind='bar', stacked=False, 
                    title=f'bot per lang')
    plt.xlabel('Lang')
    plt.ylabel("bot")
    plt.show()