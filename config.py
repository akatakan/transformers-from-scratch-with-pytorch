from pathlib import Path

def get_config():
    return {
        "batch_size":8,
        "num_epochs":20,
        "lr": 10**-4,
        "max_len": 350,
        "d_model": 512,
        "lang_src": "en",
        "lang_dst": "it",
        "model_folder":"weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weigths_file_path(config, epochs: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epochs}.pt"
    return str(Path('.') / model_folder / model_filename)