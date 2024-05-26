from pathlib import Path


def get_icv_cpk_path(result_dir, model_name, dataset_name, run_name):
    result_dir = Path(result_dir)
    model_cpk_dir = result_dir / "model_cpk" / dataset_name / model_name / run_name
    return model_cpk_dir


def get_inference_paths(result_dir, model_name, dataset_name, run_name):
    result_dir = Path(result_dir)
    save_dir = result_dir / "inference" / model_name / dataset_name / run_name
    meta_info_dir = save_dir / "meta_info"
    metric_file_path = save_dir / "result.json"
    return save_dir, meta_info_dir, metric_file_path
