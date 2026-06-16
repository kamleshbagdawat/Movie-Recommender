import mlflow
import mlflow.sklearn
from datetime import datetime

mlflow.set_experiment("movie_recommender")

def log_model(model_name, recommendation_count):

    with mlflow.start_run():

        mlflow.log_param("model_name", model_name)

        mlflow.log_param(
            "recommendation_count",
            recommendation_count
        )

        mlflow.log_metric(
            "timestamp",
            datetime.now().timestamp()
        )

        print("MLflow logging completed")
