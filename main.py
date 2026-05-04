from data_preprocessing import load_and_preprocess
from train_model import train
from evaluate_model import evaluate

def main():
    X_train, X_test, y_train, y_test = load_and_preprocess()

    model = train(X_train, y_train)

    evaluate(model, X_test, y_test)

if __name__ == "__main__":
    main()