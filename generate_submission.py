import pandas as pd
from train import preprocess, train_model

def main():
    print("Loading data...")
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    print("Preprocessing data...")
    train_df = preprocess(train_df)
    test_df = preprocess(test_df)

    features = [
        'Pclass', 'Sex', 'Age', 'Fare',
        'FamilySize', 'IsAlone', 'Title',
        'FarePerPerson'
    ]

    X_train = train_df[features]
    y_train = train_df['Survived']
    X_test = test_df[features]

    print("Training best model...")
    # Using the best parameters evaluated
    model = train_model(X_train, y_train, max_depth=8, n_estimators=200)

    print("Predicting on test set...")
    predictions = model.predict(X_test)

    print("Creating submission file...")
    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": predictions
    })
    
    submission.to_csv("submission.csv", index=False)
    print("Successfully generated submission.csv!")

if __name__ == "__main__":
    main()
