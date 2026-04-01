import os
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("titanic-experiment")


# -------------------------------
# 1. Load Data
# -------------------------------
def load_data():
    return pd.read_csv("data/train.csv")


# -------------------------------
# 2. Preprocessing
# -------------------------------
def preprocess(df):
    df = df.copy()

    # Fill Embarked
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Convert categorical
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # Title extraction
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

    df['Title'] = df['Title'].replace(['Lady','Countess','Capt','Col',
                                       'Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')

    df['Title'] = df['Title'].replace(['Mlle','Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    df['Title'] = df['Title'].map({
        'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4
    })

    # Fill Age smarter (based on Title)
    df['Age'] = df.groupby('Title')['Age'].transform(
        lambda x: x.fillna(x.median())
    )

    # Fill Fare for test.csv
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Feature Engineering
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # 🔥 NEW FEATURE
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']

    return df


# -------------------------------
# 3. Train Model
# -------------------------------
def train_model(X_train, y_train, max_depth, n_estimators):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


# -------------------------------
# 4. Main Pipeline
# -------------------------------
def main():

    df = load_data()
    df = preprocess(df)

    features = [
        'Pclass', 'Sex', 'Age', 'Fare',
        'FamilySize', 'IsAlone', 'Title',
        'FarePerPerson'
    ]

    X = df[features]
    y = df['Survived']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 🔥 MORE POWERFUL SEARCH SPACE
    for max_depth in [6, 7, 8]:
        for n_estimators in [180, 200, 220]:

            run_name = f"depth_{max_depth}_trees_{n_estimators}"

            with mlflow.start_run(run_name=run_name):

                model = train_model(X_train, y_train, max_depth, n_estimators)

                preds = model.predict(X_val)
                acc = accuracy_score(y_val, preds)

                # ---------------- MLflow ----------------
                mlflow.log_param("max_depth", max_depth)
                mlflow.log_param("n_estimators", n_estimators)

                mlflow.log_metric("accuracy", acc)

                # 🔥 Feature Importance Plot
                importances = model.feature_importances_

                plt.figure()
                plt.barh(features, importances)
                plt.title("Feature Importance")
                plt.tight_layout()
                plt.savefig("feature_importance.png")

                mlflow.log_artifact("feature_importance.png")

                # Save model
                mlflow.sklearn.log_model(model, "model")

                print(f"{run_name} → Accuracy: {acc}")

                # Save locally
                os.makedirs("models", exist_ok=True)
                model_path = f"models/{run_name}.pkl"
                pd.to_pickle(model, model_path)


# -------------------------------
if __name__ == "__main__":
    main()