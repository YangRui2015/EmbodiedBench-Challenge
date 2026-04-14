import json
import os
import re
import glob
import zipfile
import tempfile


# Leaderboard weights for Overall Score
WEIGHTS = {
    "alfred": 0.20,
    "habitat": 0.20,
    "navigation": 0.25,
    "manipulation": 0.35,
}


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


def compute_overall(alfred_sr, nav_sr, habitat_sr=None, manip_sr=None):
    """
    Weighted Overall Score using available environments.
    Missing environments contribute 0 to numerator; their weight is still
    included in the denominator so scores remain comparable across submissions
    that cover different subsets of environments.
    """
    score = 0.0
    score += WEIGHTS["alfred"] * (alfred_sr if alfred_sr is not None else 0.0)
    score += WEIGHTS["habitat"] * (habitat_sr if habitat_sr is not None else 0.0)
    score += WEIGHTS["navigation"] * (nav_sr if nav_sr is not None else 0.0)
    score += WEIGHTS["manipulation"] * (manip_sr if manip_sr is not None else 0.0)
    return round(score, 3)


def compute_weighted_avg_steps(alfred_steps, nav_steps, habitat_steps=None, manip_steps=None):
    """
    Same weights as Overall Score applied to avg steps (tiebreaker).
    """
    total = 0.0
    total += WEIGHTS["alfred"] * (alfred_steps if alfred_steps is not None else 0.0)
    total += WEIGHTS["habitat"] * (habitat_steps if habitat_steps is not None else 0.0)
    total += WEIGHTS["navigation"] * (nav_steps if nav_steps is not None else 0.0)
    total += WEIGHTS["manipulation"] * (manip_steps if manip_steps is not None else 0.0)
    return round(total, 3)


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

    print(f"ALFRED   — SR: {alfred_sr}%  Avg Steps: {alfred_steps}")
    print(f"Navigation — SR: {nav_sr}%  Avg Steps: {nav_steps}")

    overall_score = compute_overall(alfred_sr, nav_sr)
    weighted_steps = compute_weighted_avg_steps(alfred_steps, nav_steps)

    metrics = {
        "ALFRED SR": alfred_sr,
        "ALFRED Steps": alfred_steps,
        "Habitat SR": None,
        "Habitat Steps": None,
        "Navigation SR": nav_sr,
        "Navigation Steps": nav_steps,
        "Manipulation SR": None,
        "Manipulation Steps": None,
        "Overall Score": overall_score,
        "Weighted Avg Steps": weighted_steps,
    }

    # EvalAI requires None values to be absent or 0 in the result dict.
    # Replace None with 0 so the leaderboard columns still populate.
    metrics_clean = {k: (v if v is not None else 0) for k, v in metrics.items()}

    output = {
        "result": [{split_key: metrics_clean}],
        "submission_result": metrics_clean,
    }

    print("Evaluation complete:", metrics_clean)
    return output
