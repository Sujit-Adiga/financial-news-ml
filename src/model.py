import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from xgboost import XGBClassifier


def make_xgb(random_state=42):
    return XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )


def walk_forward_predict(
    df,
    features,
    min_train_size=500,
    test_size=60,
    random_state=42,
):
    df = df.sort_values("Date").reset_index(drop=True)

    predictions = []
    probabilities = []
    dates = []
    targets = []
    next_returns = []

    start = min_train_size

    while start < len(df):
        end = min(start + test_size, len(df))

        train = df.iloc[:start]
        test = df.iloc[start:end]

        model = make_xgb(random_state)

        model.fit(
            train[features],
            train["target"],
        )

        prob = model.predict_proba(
            test[features]
        )[:, 1]

        predictions.extend((prob >= 0.5).astype(int))
        probabilities.extend(prob)
        dates.extend(test["Date"])
        targets.extend(test["target"])
        next_returns.extend(test["next_day_return"])

        start = end

    result = pd.DataFrame({
        "Date": dates,
        "target": targets,
        "probability": probabilities,
        "prediction": predictions,
        "next_day_return": next_returns,
    })

    metrics = {
        "f1": f1_score(result["target"], result["prediction"]),
        "roc_auc": roc_auc_score(
            result["target"], result["probability"]
        ),
        "accuracy": accuracy_score(
            result["target"], result["prediction"]
        ),
    }

    return result, metrics
