import mlflow
import logging

MLFLOW_URL = "http://127.0.0.1:8080"


class MLFlow_Logger:
    def __init__(self, experiment):
        self.experiment = experiment
        self.run = None

        mlflow.set_tracking_uri(MLFLOW_URL)
        mlflow.set_experiment(experiment_name=self.experiment)
        logging.info(f"Logger for experiment '{self.experiment}' is ready.")

    def create_run(self, run):
        if self.run:
            raise Exception("Logger is already running and therefore can't create a new run!")

        self.run = run
        mlflow.start_run(run_name=self.run)
        logging.info(f"Logger started run '{self.run}'.")

    def end_run(self):
        if not self.run:
            raise Exception("Logger is not running and therefore can't end a run!")

        mlflow.end_run()
        logging.info(f"Logger ended run '{self.run}'.")
        self.run = None

    def log_parameters(self, params):
        if not self.run:
            raise Exception("Logger is not running and therefore can't log a parameter!")

        mlflow.log_params(params)
