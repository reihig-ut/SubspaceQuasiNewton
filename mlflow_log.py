import mlflow
from pathlib import Path
import numpy as np


def extract_params(path):
    objective_properties = path.parent.parent.parent.name
    objective_properties = objective_properties.split("~")
    objective_properties = {
        prop.split("@")[0]: prop.split("@")[1] for prop in objective_properties
    }
    optimizer_properties = path.name
    optimizer_properties = optimizer_properties.split("~")
    optimizer_properties = {
        prop.split("@")[0]: prop.split("@")[1] for prop in optimizer_properties
    }
    return objective_properties, optimizer_properties


paths = [
    p.parent
    for p in Path("/home/meip-users/Projects/SubspaceQuasiNewton/results").glob(
        "**/func_values.npy"
    )
]

for path in paths:
    optimizer_name = "RSTRM" if "RSTRM" in str(path) else "RSRNM"
    objective_properties, optimizer_properties = extract_params(path)

    mlflow.set_experiment("MLP")

    with mlflow.start_run() as run:
        mlflow.log_params(objective_properties)
        mlflow.log_params(optimizer_properties)
        mlflow.log_param("optimizer", optimizer_name)

        # log metrics
        func_values = np.load(path / "func_values.npy")
        grad_norms = np.load(path / "grad_norm.npy")
        timestamps = np.load(path / "time.npy")
        mask = func_values != 0
        func_values = func_values[mask]
        grad_norms = grad_norms[mask]
        timestamps = timestamps[mask]
        for i, (func_value, grad_norm, timestamp) in enumerate(
            zip(func_values, grad_norms, timestamps)
        ):
            timestamp = int(timestamp * 1000)  # convert to milliseconds
            mlflow.log_metric("func_value", func_value, step=i, timestamp=timestamp)
            mlflow.log_metric("grad_norm", grad_norm, step=i, timestamp=timestamp)
