import json
import logging
import re
from glob import glob
from pathlib import Path

import gspread
import pandas as pd
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from oauth2client.service_account import ServiceAccountCredentials

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

credentials = (
    "/Users/theocayla/Documents/Dev/GoogleAPI/pyro-metrics-459816-9ef5ab17ce1c.json"
)


def build_dataframe(run_dirs, csv_path=None):
    rows = []

    for run_dir in run_dirs:
        json_file = Path(run_dir) / "metrics.json"
        with open(json_file, "r") as f:
            data = json.load(f)
        config = data.get("config", {})
        model_config = config.get("model")
        engine_config = config.get("engine")
        model_metrics = data.get("model_metrics", {})
        seq_metrics = data.get("engine_metrics", {}).get("sequence_metrics", {})
        img_metrics = data.get("engine_metrics", {}).get("image_metrics", {})
        dataset_info = data.get("dataset")

        rows.append(
            {
                # Config
                "Run ID": data.get("run_id"),
                "Model Path": config.get("model_path").split("pyro-eval")[-1],
                "Engine Conf thresh": engine_config.get("conf_thresh"),
                "Engine Nb consecutive frames": engine_config.get("nb_consecutive_frames"),
                "Model IoU": model_config.get("iou"),
                "Model Imgsz": model_config.get("imgsz"),
                "Model Conf": model_config.get("conf"),
                "Model Hash" : model_config.get("hash"),
                "Engine Max Bbox Size": engine_config.get("max_bbox_size"),
                "Engine Dataset Hash" : dataset_info.get("engine", {}).get("hash"),
                "Engine Dataset ID" : dataset_info.get("engine", {}).get("ID"),
                "Number of images (Engine dataset)" : dataset_info.get("engine", {}).get("Number of images"),
                "Number of sequences (Engine dataset)" : dataset_info.get("engine", {}).get("Number of sequences"),
                "Model Dataset Hash" : dataset_info.get("model", {}).get("hash"),
                "Model Dataset ID" : dataset_info.get("model", {}).get("ID"),
                "Number of images (Model dataset)" : dataset_info.get("model", {}).get("Number of images"),
                # Model metrics
                "Model Precision": model_metrics.get("precision"),
                "Model Recall": model_metrics.get("recall"),
                "Model F1": model_metrics.get("f1"),
                "Model FP": model_metrics.get("fp"),
                "Model TP": model_metrics.get("tp"),
                "Model FN": model_metrics.get("fn"),
                "Model TN": model_metrics.get("tn"),
                # Engine metrics
                "Sequence Precision": seq_metrics.get("precision"),
                "Sequence Recall": seq_metrics.get("recall"),
                "Sequence F1": seq_metrics.get("f1"),
                "Sequence FP": seq_metrics.get("fp"),
                "Sequence TP": seq_metrics.get("tp"),
                "Sequence FN": seq_metrics.get("fn"),
                "Sequence TN": seq_metrics.get("tn"),
                "Avg Detection Delay (min)" : timedelta_string_to_minutes(seq_metrics.get("avg_detection_delay")),
                "Image Precision": img_metrics.get("precision"),
                "Image Recall": img_metrics.get("recall"),
                "Image F1": img_metrics.get("f1"),
                "Image FP": img_metrics.get("fp"),
                "Image TP": img_metrics.get("tp"),
                "Image FN": img_metrics.get("fn"),
            }
        )

    df = pd.DataFrame(rows)
    if csv_path:
        df.to_csv(csv_path)
    return df

def timedelta_string_to_minutes(timedelta_str):
    match = re.search(r'(\d+) days (\d+):(\d+):(\d+\.\d+)', timedelta_str)

    if not match:
        logging.error(f"Invalid timedelta format, conversion impossible : {timedelta_str}")

    days = int(match.group(1))
    hours = int(match.group(2))
    minutes = int(match.group(3))
    seconds = float(match.group(4))

    total_seconds = (days * 24 * 3600) + (hours * 3600) + (minutes * 60) + seconds

    total_minutes = total_seconds / 60

    return total_minutes

def export_google_sheet(df, sheet_name, key_column="run_id"):
    """
    Dumps a csv uin a google sheet
    If the sheet already exists, update the data and upload
    Use key_column as an identifier to see what's changed
    """
    # Google API Atuthentification
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials, scope)
    client = gspread.authorize(creds)

    # Open google sheet
    try:
        spreadsheet = client.open(sheet_name)
    except Exception as e:
        logging.error(f"Unable to open Google sheet : {e}")

    df[key_column] = df[key_column].astype(str)

    config_cols = [
        "Run ID",
        "Model Path",
        "Model Dataset ID",
        "Model Dataset Hash",
        "Model Hash",
        "Number of images (Model dataset)",
        "Engine Dataset ID",
        "Engine Dataset Hash",
        "Number of images (Engine dataset)",
        "Number of sequences (Engine dataset)",
    ]

    model_cols = [
        "Run ID",
        "Model Precision",
        "Model Recall",
        "Model F1",
        "Model FP",
        "Model TP",
        "Model FN",
        "Model TN",
        "Model Path",
        "Model Conf",
        "Model IoU",
        "Model Imgsz",
        "Model Dataset Hash",
        "Model Dataset ID",
    ]

    engine_cols = [
        "Run ID",
        "Sequence Precision",
        "Sequence Recall",
        "Sequence F1",
        "Sequence FP",
        "Sequence TP",
        "Sequence FN",
        "Sequence TN",
        "Avg Detection Delay (min)",
        "Image Precision",
        "Image Recall",
        "Image F1",
        "Image FP",
        "Image TP",
        "Image FN",
        "Model Path",
        "Engine Conf thresh",
        "Engine Nb consecutive frames",
        "Engine Max Bbox Size",
        "Engine Dataset Hash",
        "Engine Dataset ID",
    ]

    df_model = df[model_cols].copy()
    df_engine = df[engine_cols].copy()
    df_config = df[config_cols].copy()
    logging.info("Updating google sheet")

    def update_worksheet(sheet_name, new_df, key_column="Run ID"):
        try:
            worksheet = spreadsheet.worksheet(sheet_name)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=sheet_name, rows=200, cols=50)

        # Retrieve existing content
        try:
            existing_df = get_as_dataframe(worksheet).dropna(how="all")
            existing_df[key_column] = existing_df[key_column].astype(str)
        except Exception:
            existing_df = pd.DataFrame(columns=new_df.columns)

        new_df[key_column] = new_df[key_column].astype(str)

        # Filter old dataframe to remove runs present in the new dataframe
        filtered_existing = existing_df[
            ~existing_df[key_column].isin(new_df[key_column])
        ]

        # Concatenate new dataframe to the old one
        final_df = pd.concat([filtered_existing, new_df], ignore_index=True)

        # Write everything in the sheet
        worksheet.clear()
        set_with_dataframe(worksheet, final_df)

        nb_updated = len(existing_df) - len(filtered_existing)
        nb_added = len(final_df) - len(existing_df)
        logging.info(f"{sheet_name} sheet updated")
        return (nb_updated, nb_added)

    update_worksheet("Config", df_config)
    update_worksheet("Model", df_model)
    nb_updated, nb_added = update_worksheet("Engine", df_engine)
    logging.info(f"{nb_added} runs added.")
    logging.info(f"{nb_updated} runs updated.")


if __name__ == "__main__":

    run_dirs = [run for run in glob("/Users/theocayla/Documents/Dev/Pyronear/pyro-eval/data/evaluation/*") if "run" in run]

    df = build_dataframe(run_dirs, csv_path=None)
    sheet_name = "Pyro Metrics"
    export_google_sheet(df, sheet_name, key_column="Run ID")
