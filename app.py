from flask import Flask, render_template, request
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=["GET", "POST"])
def predict_data():
    if request.method == "GET":
        return render_template("home.html")
    
    else:
        data = CustomData(
            gender = request.form.get("gender"),
            race_ethnicity=request.form.get("race_ethnicity"),
            parental_level_of_education=request.form.get("parental_level_of_education"),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test_preparation_course"),
            reading_score=request.form.get("reading_score"),
            writing_score=request.form.get("writing_score")
        )
        
        df = data.get_customdata_into_dataframe() 
        
        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(df)
        
        return render_template("home.html", result=result[0])
    
if __name__ == "__main__":
    app.run(debug=True)    
    
    
