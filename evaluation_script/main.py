import json
import re
import tempfile
import zipfile
from pathlib import Path
from statistics import mean


ENVIRONMENTS = {
    "eb_alfred": "EB-ALFRED",
    "eb_navigation": "EB-Navigation",
}
SUCCESS_KEYS = ("task_success", "success", "is_success", "episode_success")
STEP_KEYS = ("num_steps", "steps", "episode_steps")
STEP_FILENAME_RE = re.compile(r"_step_(\d+)(?:_\d+)?\.json$")


def _safe_extract(zip_file, destination):
    destination = Path(destination).resolve()
    for member in zip_file.infolist():
        target = (destination / member.filename).resolve()
        if destination not in target.parents and target != destination:
            raise ValueError(f"Unsafe path in submission zip: {member.filename}")
    zip_file.extractall(destination)


def _find_results_root(extract_dir):
    extract_dir = Path(extract_dir)
    direct = extract_dir / "results"
    if direct.is_dir():
        return direct

    candidates = [
        path
        for path in extract_dir.rglob("results")
        if path.is_dir()
        and any((path / env_name).is_dir() for env_name in ENVIRONMENTS)
    ]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return sorted(candidates)[0]

    return extract_dir


def _find_success_value(record):
    for key in SUCCESS_KEYS:
        if key in record:
            value = record[key]
            if isinstance(value, bool):
                return float(value)
            if isinstance(value, (int, float)):
                return float(value)
    return None


def _find_step_value(record):
    for key in STEP_KEYS:
        if key in record:
            value = record[key]
            if isinstance(value, (int, float)):
                return float(value)
    return None


def _load_json_records(path):
    with path.open("r") as file_obj:
        try:
            data = json.load(file_obj)
        except json.JSONDecodeError:
            file_obj.seek(0)
            data = [json.loads(line) for line in file_obj if line.strip()]

    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def _read_episode_record(path):
    records = _load_json_records(path)
    if not records:
        raise ValueError(f"No JSON object records found in {path}")

    merged = dict(records[-1])

    for item in reversed(records):
        success = _find_success_value(item)
        if success is not None:
            merged["task_success"] = success
            break
    else:
        merged["task_success"] = 0.0

    if _find_step_value(merged) is None:
        env_steps = [
            item.get("env_step")
            for item in records
            if isinstance(item.get("env_step"), (int, float))
        ]
        if env_steps:
            merged["num_steps"] = max(env_steps)
        else:
            match = STEP_FILENAME_RE.search(path.name)
            merged["num_steps"] = int(match.group(1)) if match else len(records)

    return merged


def _iter_episode_files(env_root):
    final_result_files = sorted(env_root.glob("**/results/episode_*_final_res.json"))
    if final_result_files:
        yield from final_result_files
        return

    for path in sorted(env_root.glob("**/episode_*.json")):
        if not path.is_file():
            continue
        if "images" in path.parts:
            continue
        if path.name.endswith("_final_res.json"):
            continue
        yield path


def _summarize_environment(env_root):
    successes = []
    steps = []
    for path in _iter_episode_files(env_root):
        record = _read_episode_record(path)
        success = _find_success_value(record)
        if success is None:
            raise ValueError(f"Missing success field in {path}")
        successes.append(success)

        step_value = _find_step_value(record)
        if step_value is not None:
            steps.append(step_value)

    if not successes:
        raise ValueError(f"No episode JSON files found in {env_root}")

    return {
        "success_rate": mean(successes),
        "average_steps": mean(steps) if steps else 0.0,
        "num_tasks": len(successes),
    }


def _score_submission(user_submission_file):
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(user_submission_file, "r") as zip_file:
            _safe_extract(zip_file, temp_dir)

        results_root = _find_results_root(temp_dir)
        summaries = {}
        for env_dir_name, leaderboard_name in ENVIRONMENTS.items():
            env_root = results_root / env_dir_name
            if not env_root.is_dir():
                raise ValueError(
                    "Missing required environment directory: "
                    f"{env_root}. Expected zip layout: "
                    "results/eb_alfred/ and results/eb_navigation/."
                )
            summaries[leaderboard_name] = _summarize_environment(env_root)

    task_counts = [
        summaries[leaderboard_name]["num_tasks"]
        for leaderboard_name in ENVIRONMENTS.values()
    ]
    total_tasks = sum(task_counts)
    weighted_steps = sum(
        summaries[leaderboard_name]["average_steps"]
        * summaries[leaderboard_name]["num_tasks"]
        for leaderboard_name in ENVIRONMENTS.values()
    )

    return {
        "Overall": sum(
            summaries[leaderboard_name]["success_rate"]
            * summaries[leaderboard_name]["num_tasks"]
            for leaderboard_name in ENVIRONMENTS.values()
        )
        / total_tasks
        if total_tasks
        else 0.0,
        "EB-ALFRED": summaries["EB-ALFRED"]["success_rate"],
        "EB-Navigation": summaries["EB-Navigation"]["success_rate"],
        "Average Steps": weighted_steps / total_tasks if total_tasks else 0.0,
    }


def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting EmbodiedBench evaluation")
    scores = _score_submission(user_submission_file)
    split_name = "train_split" if phase_codename == "dev" else "test_split"

    output = {
        "result": [
            {
                split_name: scores,
            }
        ],
        "submission_result": scores,
    }
    print("Completed EmbodiedBench evaluation")
    return output
