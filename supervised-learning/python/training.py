# data manipulation tools
import json
import numpy as np
import pandas as pd
import statistics

# preprocessing tools
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# classifiers used in feature selection and modeling tasks
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# feature selection using filter methods
from sklearn.feature_selection import SelectKBest, VarianceThreshold, chi2, f_classif

# feature selection by feature importance (model coefficients)
from sklearn.feature_selection import SelectFromModel

# model validation
from sklearn.model_selection import StratifiedKFold

# metrics for model validation
from sklearn.metrics import accuracy_score, f1_score, auc, roc_curve, precision_score, recall_score

# processing mixed columns on the dataset
from sklearn.compose import ColumnTransformer

# flow of data transforming and fitting
from sklearn.pipeline import Pipeline



def filter_columns(columns, condition='not'):
    """
        Description: Filter specific elements from a given list.
        Input: Full list of strings 
        Output: Filtered list of strrings. 
    """
    if condition == 'not': # just categorical variables
        return [x for x in columns if x not in ["base_price", "price", "initial_quantity", "sold_quantity", "available_quantity"]] 
    else: # numerical variables
        return [x for x in columns if x in ["base_price", "price", "initial_quantity", "sold_quantity", "available_quantity"]]

def summary_results_metrics(metrics_results, name):
    """
        Description: calculate the overall metrics of the cross validation step.
        Input: List of metrics
        Output: File with mean and std of each available metric
    """
    results = {
        "F1_score": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": []
    }
    for metric in metrics_results:
        results["F1_score"].append(metric["F1_score"])
        results["Accuracy"].append(metric["Accuracy"])
        results["Precision"].append(metric["Precision"])
        results["Recall"].append(metric["Recall"])
    
    with open(f"{name.split('.')[0]}_overall_CV.{name.split('.')[1]}", "w") as f:
        json.dump(
            {
                "acc_mean": np.mean(results["Accuracy"]),
                "acc_std": np.std(results["Accuracy"]),
                "f1_score_mean": np.mean(results["F1_score"]),
                "f1_score_std": np.std(results["F1_score"]),
                "precision_mean": np.mean(results["Precision"]),
                "precision_std": np.std(results["Precision"]),
                "recall_mean": np.mean(results["Recall"]),
                "recall_std": np.std(results["Recall"])
            },
            f
        )

def predict_from_cross_validation_models(test_dataset, pipelines):  
    """
        Description: Predict based on trained models (cross validation step).
        Input: list of trained models and test dataset
        Output: predition for test dataset
    """
    result = []
    for _pipeline in pipelines:
        result.append(_pipeline.predict(test_dataset))
    # voting classifier
    result = np.array(result).T
    return [statistics.mode(x) for x in result]

def model_approach_feature_importance(train_dataset, y_train, test_dataset, y_test, model='lr', feature_selection_approach='from_model'):
    """
        Description - Experiments with Feature Selection approach:
            Feature Importance from single model (linear or decision tree model).
        Input: Train and test datasets.
        Output: Model metrics results (dictionary-like)
    """
    # List of models trained at k-folds cross validation
    models_pipeline = []
    # list of metric results
    metrics_results = []
    # Setting StratifiedKFold for cross-validation
    skf = StratifiedKFold(n_splits=10)
    for train_index, test_index in skf.split(train_dataset, y_train):
        # k-1 train set and k test set
        X_train, X_test = train_dataset.loc[train_index], train_dataset.loc[test_index]
        y_trainset, y_testset = np.asarray(y_train)[train_index], np.asarray(y_train)[test_index]
        # z-score normalization at numeric data fields
        numeric_transformer = Pipeline(
            steps=[("scaler", StandardScaler())]
        )
        # Onehot encoding at categorical data fields\
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")
        # processing numeric and categorical fields
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, filter_columns(X_train.columns, condition="yes")),
                ("cat", categorical_transformer, filter_columns(X_train.columns)),
            ]
        )
        # default model for feature selection
        fs_model = LogisticRegression(penalty='l1', solver='saga', max_iter=200)
        if model != 'lr': # Use DecisionTreeClassifier as wrapper model for feature selection
            fs_model = DecisionTreeClassifier()
        fi = SelectFromModel(estimator=fs_model)
        # define Pipeline object for fitting model
        clf = Pipeline(
            steps=[
                ("preprocessor", preprocessor), 
                ("feature_selection", fi),
                ("classifier", RandomForestClassifier())]
        )       
        # fit pipeline model
        clf.fit(X_train, y_trainset)
        # save model on list of models
        models_pipeline.append(clf)
        # predict Random Forest
        # predict on validation dataset
        y_pred = clf.predict(X_test)
        # save metrics results
        results = validation_metrics(y_testset, y_pred)
        metrics_results.append(results)
    
    # cross validation overall results
    summary_results_metrics(metrics_results=metrics_results, name=f"results/overall_CV_from_model_{model}_.json")

    # predict on unseen dataset using all trained model (pipelines)
    y_pred = predict_from_cross_validation_models(test_dataset=test_dataset, pipelines=models_pipeline)
    # metrics performance on test dataset
    return validation_metrics(y_test=y_test, y_pred=y_pred)

def model_approach_statistical_methods(train_dataset, y_train, test_dataset, y_test):
    """
        Description: Experiment with statistical methods for Features Selection.
        Input: Train and test datasets.
        Output: Model results metrics (dictionary-like)
    """
    # List of models trained at k-folds cross validation
    models_pipeline = []
    # list of metric results
    metrics_results = []
    # Setting StratifiedKFold for cross-validation
    skf = StratifiedKFold(n_splits=10)
    for train_index, test_index in skf.split(train_dataset, y_train):
        # k-1 train set and k test set
        X_train, X_test = train_dataset.loc[train_index], train_dataset.loc[test_index]
        y_trainset, y_testset = np.asarray(y_train)[train_index], np.asarray(y_train)[test_index]
        # z-score normalization at numeric data fields
        numeric_transformer = Pipeline(
            steps=[
                ("high_variance", VarianceThreshold(threshold=0.0)),
                ("feature_selection", SelectKBest(score_func=f_classif, k='all')),
                ("scaler", StandardScaler())
            ]
        )
        # Onehot encoding at categorical data fields\
        categorical_transformer = Pipeline(
            steps=[
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ("high_variance", VarianceThreshold(threshold=0.0)),
                ("feature_selection", SelectKBest(k=500, score_func=chi2))
            ]    
        )
        # processing numeric and categorical fields
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, filter_columns(X_train.columns, condition="yes")),
                ("cat", categorical_transformer, filter_columns(X_train.columns)),
            ]
        )
        # define Pipeline object for fitting model
        clf = Pipeline(
            steps=[
                ("preprocessor", preprocessor), 
                ("classifier", RandomForestClassifier())
            ]
        )      
        # fit pipeline model
        clf.fit(X_train, y_trainset)
        # save model on list of models
        models_pipeline.append(clf)
        # predict on validation dataset
        y_pred = clf.predict(X_test)
        # save metrics results
        results = validation_metrics(y_testset, y_pred)
        metrics_results.append(results)
    
    # cross validation overall results
    summary_results_metrics(metrics_results=metrics_results, name=f"results/overall_CV_statistical_.json")

    # predict on unseen dataset using all trained model (pipelines)
    y_pred = predict_from_cross_validation_models(test_dataset=test_dataset, pipelines=models_pipeline)
    # metrics performance on test dataset
    return validation_metrics(y_test=y_test, y_pred=y_pred)

def validation_metrics(y_test, y_pred, pos_label=1):
    """
        Description: Calculate validation metrics for validation/unseen dataset
        Input: groundtruth labels and predicted labels
        Output: Available validation metrics.
    """
    y_test = [1 if x == 'new' else 0 for x in y_test]
    y_pred = [1 if x == 'new' else 0 for x in y_pred]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=pos_label)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1_score": f1_score(y_test, y_pred, pos_label=pos_label),
        "Precision": precision_score(y_test, y_pred, pos_label=pos_label),
        "Recall": recall_score(y_test, y_pred, pos_label=pos_label),
        "AUC": auc(fpr, tpr),
        "FPR": fpr.tolist(),
        "TPR": tpr.tolist(),
        "thresholds": thresholds.tolist()
    }    

def save_metrics(results, name):
    """
        Description: Save the results metrics into JSON format file.
        Input: Dict object
        Output: JSON file format
    """
    with open(f"results/{name}", "w") as outfile:
        json.dump(results, outfile)

if __name__ == "__main__":
    print("Loading dataset...")
    # TODO: Read your dataset and split as train and test sets
    X_train, y_train, X_test, y_test = ...

    # Run tbe experiments
    print("***********************************************************************************")
    print("Model approach statistical methods using Chi^2 and ANOVA test...")
    save_metrics(model_approach_statistical_methods(
        train_dataset=X_train, 
        y_train=y_train, 
        test_dataset=X_test, 
        y_test=y_test
    ), name="statistical_methods.json")
    print("***********************************************************************************")
    print("model=LogisticRegression, fs=FromModel")
    save_metrics(model_approach_feature_importance(
        train_dataset=X_train, 
        y_train=y_train, 
        test_dataset=X_test, 
        y_test=y_test
    ), name="logistic_regression_from_model.json")
    print("***********************************************************************************")
    print("model=DecisionTreeClassifier, fs=FromModel")
    save_metrics(model_approach_feature_importance(
        train_dataset=X_train, 
        y_train=y_train, 
        test_dataset=X_test, 
        y_test=y_test, 
        model='decision_tree'
    ), name="decision_tree_classifier_from_model.json")
    print("***********************************************************************************")