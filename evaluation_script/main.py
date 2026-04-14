import json
import os
import re
import glob
import zipfile
import tempfile




def _last_valid_step(lines):
    """Return the last JSON line that contains task_success."""
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            if d.get("task_success") is not None:
                return d
        except json.JSONDecodeError:
            continue
    return None


def eval_alfred(alfred_dir):
    """
    Walk results/eb_alfred/<model>/<category>/episode_N_step_S.json.

    Returns (success_rate_pct, avg_steps) or (None, None) if no files found.
    Steps are read from the filename suffix _step_S; task_success from the
    last line that carries the field.
    """
    files = glob.glob(os.path.join(alfred_dir, "*", "*", "*.json"))
    if not files:
        return None, None

    successes = 0
    total_steps = 0
    total = 0

    for fp in files:
        m = re.search(r"_step_(\d+)\.json$", fp)
        steps = int(m.group(1)) if m else None

        with open(fp) as f:
            lines = f.readlines()

        last = _last_valid_step(lines)
        if last is None:
            continue

        task_success = last["task_success"]
        if steps is None:
            steps = last.get("env_step", 0)

        total += 1
        if task_success == 1.0:
            successes += 1
        total_steps += steps

    if total == 0:
        return None, None

    return round(successes / total * 100, 3), round(total_steps / total, 3)


def eval_navigation(nav_dir):
    """
    Walk results/eb_nav/<model>/<category>/episode_N.json.

    Returns (success_rate_pct, avg_steps) or (None, None) if no files found.
    Both task_success and env_step are taken from the last line.
    """
    files = glob.glob(os.path.join(nav_dir, "*", "*", "*.json"))
    if not files:
        return None, None

    successes = 0
    total_steps = 0
    total = 0

    for fp in files:
        with open(fp) as f:
            lines = f.readlines()

        if not lines:
            continue

        try:
            last = json.loads(lines[-1].strip())
        except json.JSONDecodeError:
            continue

        task_success = last.get("task_success", 0.0)
        steps = last.get("env_step", 0)

        total += 1
        if task_success == 1.0:
            successes += 1
        total_steps += steps

    if total == 0:
        return None, None

    return round(successes / total * 100, 3), round(total_steps / total, 3)


def compute_overall_score(alfred_sr, nav_sr):
    """Simple average success rate across available environments."""
    values = [v for v in [alfred_sr, nav_sr] if v is not None]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


def compute_average_steps(alfred_steps, nav_steps):
    """Simple average of per-environment avg steps."""
    values = [v for v in [alfred_steps, nav_steps] if v is not None]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    """
    EvalAI evaluation entry point.

    Expects the submission to be a ZIP archive with structure:
        results/
            eb_alfred/
                <model_name>/
                    <category>/
                        episode_N_step_S.json   (one JSON object per line)
            eb_nav/
                <model_name>/
                    <category>/
                        episode_N.json          (one JSON object per line)

    Only eb_alfred and eb_nav are currently supported; eb_habitat and
    eb_manipulation folders are silently ignored (scores reported as null).
    """
    print("Starting EmbodiedBench evaluation...")

    split_key = "dev_split" if phase_codename == "dev" else "test_split"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Unzip submission
        try:
            with zipfile.ZipFile(user_submission_file, "r") as zf:
                zf.extractall(tmpdir)
        except zipfile.BadZipFile as e:
            raise ValueError(f"Submission is not a valid ZIP file: {e}")

        # Locate results root (handle both zip-with and zip-without top-level dir)
        results_root = os.path.join(tmpdir, "results")
        if not os.path.isdir(results_root):
            # Try one level deeper (e.g. submission.zip/results/...)
            candidates = glob.glob(os.path.join(tmpdir, "*", "results"))
            if candidates:
                results_root = candidates[0]
            else:
                # Fall back to tmpdir itself
                results_root = tmpdir

        alfred_dir = os.path.join(results_root, "eb_alfred")
        nav_dir = os.path.join(results_root, "eb_nav")

        alfred_sr, alfred_steps = eval_alfred(alfred_dir) if os.path.isdir(alfred_dir) else (None, None)
        nav_sr, nav_steps = eval_navigation(nav_dir) if os.path.isdir(nav_dir) else (None, None)

    print(f"ALFRED     — SR: {alfred_sr}%  Avg Steps: {alfred_steps}")
    print(f"Navigation — SR: {nav_sr}%  Avg Steps: {nav_steps}")

    metrics_clean = {
        "Overall Score": compute_overall_score(alfred_sr, nav_sr),
        "ALFRED SR": alfred_sr if alfred_sr is not None else 0,
        "ALFRED Steps": alfred_steps if alfred_steps is not None else 0,
        "Navigation SR": nav_sr if nav_sr is not None else 0,
        "Navigation Steps": nav_steps if nav_steps is not None else 0,
        "Average Steps": compute_average_steps(alfred_steps, nav_steps),
    }

    output = {
        "result": [{split_key: metrics_clean}],
        "submission_result": metrics_clean,
    }

    print("Evaluation complete:", metrics_clean)
    return output
