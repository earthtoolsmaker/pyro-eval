from pathlib import Path

def get_prediction_path(model_path):
    abs_model_path = Path(model_path).resolve()
    output_dir = Path("data/predictions")

    relative = Path(*abs_model_path.parts[-4:])

    output_name = "_".join(relative.parts).replace(".pt", "") + ".json"

    return output_dir / output_name