import joblib
import pandas as pd

def predict(input_data, model_type):

    model_type = model_type.lower()
    
    if model_type == 'ml_a':
        model = joblib.load('ML_A.sav')

    elif model_type == 'ml_b':
        model = joblib.load('ML_B.sav')

    elif model_type == 'mlp':
        model = joblib.load('MLP_Model.sav')

    elif model_type == 'cnn':
        model = joblib.load('CNN_Model.sav')

    else:
        raise ValueError("Invalid value for `model_type`. Please provide one of 'ML_A', 'ML_B', 'MLP', or 'CNN'.")

    output = model.predict(input_data)

    return output

if __name__ == '__main__':

    # Example usage
    
    pass


