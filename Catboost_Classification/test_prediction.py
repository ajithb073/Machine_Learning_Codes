# pip install catboost==0.26
from catboost import CatBoostClassifier
from nltk import sent_tokenize
import pandas as pd


def company_classification(sentence):
    model = CatBoostClassifier()
    model.load_model('/home/ullas/company_classification_model_new')
    sentence = " ".join(sentence.split())
    model_prediction = model.predict([sentence])
    # {'Non-Technology driven(Supplier)': 1, 'Technology driven(Manufacturer)': 2}
    if model_prediction == 1:
      prediction = 'Non-Technology driven(Supplier)'
    else:
      prediction = 'Technology driven(Manufacturer)'  
    return prediction



if __name__ == "__main__":
    statement = "Hi Team, Thanks for the report on Graphene battery. We were able to get all technical details as well a s market details. Could you please clarify below queries. Will there be heating issues while charging a graphene battery? Can you also mention the CAGR of Graphene battery market from 2026 - 2030?"
    print(company_classification(statement))

