import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = mlflow.tracking.MlflowClient()

for mv in client.search_model_versions("name='ChurnModel'"):
    print(mv.name, mv.version, mv.current_stage, mv.run_id)