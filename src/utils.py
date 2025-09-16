import os
import sys
import dill
from src.exception import Custom_Exception
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    
    try:
        
        dir = os.path.dirname(file_path)
        os.makedirs(dir, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)  # Fixed parameter order
        
    except Exception as e:
        raise Custom_Exception(e, sys) 
    
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    
    try:
        
        model_report = {}
        
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            r2score = r2_score(y_test, y_pred)
            
            model_report[model_name] = r2score
            
        return model_report    
        
        
    except Exception as e:
        raise Custom_Exception(e, sys)
    
    
def load_object(file_path):
    
    try:
        with open(file_path, "rb") as file_obj:
            
            return dill.load(file_obj)
        
    except Exception as e:
        raise Custom_Exception(e, sys)