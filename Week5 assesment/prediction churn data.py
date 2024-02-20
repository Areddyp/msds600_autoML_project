import pandas as pd
from pycaret.classification import predict_model, load_model

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df


def make_predictions(df):
    model = load_model('lr')
   
    predictions = predict_model(model, data=df)
    
    predictions.rename({'prediction_label':'Churn_prediction'},axis=1,inplace=True)
    predictions['Churn_prediction'].replace({'Yes':1,'No':0},inplace=True)
    
    
    return predictions['Churn_prediction']

if __name__ == "__main__":
    df = load_data("/Users/abhilashreddy/Desktop/weeek 5/new_churn_data.csv")
    predictions = make_predictions(df)
    print('predictions:')
    print(predictions)