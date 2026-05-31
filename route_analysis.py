from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.utils.validation import check_random_state
from gplearn.genetic import SymbolicRegressor


@dataclass
class RouteSummary:
    distance_m: int
    duration_s: int
    co2_score: float
    scenic_score: float
    safety_score: float
    preference_score: float


def train_symbolic_regressor(w1, w2, w3, criteria=0.1):
    rng = check_random_state(0)

    x_train = rng.uniform(1, 5, 200).reshape(50, 4)
    y_train = (
        (x_train[:, 0] * w1 + x_train[:, 1] * w2 * 6 + x_train[:, 2] * w3 * 8)
        / (w1 + w2 * 6 + w3 * 8 + 1)
        * x_train[:, 3]
    )

    model = SymbolicRegressor(
        population_size=5000,
        generations=20,
        stopping_criteria=criteria,
        random_state=0,
        verbose=1,
    )
    model.fit(x_train, y_train)
    return model


def format_duration(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, _ = divmod(remainder, 60)

    parts = []
    if hours:
        parts.append(f"{hours} h")
    if minutes:
        parts.append(f"{minutes} min")

    return " ".join(parts)


def evaluate_routes(routes, model, prioritize_time=False):
    summaries = []

    for route in routes:
        steps = route["legs"][0]["steps"]

        distances = np.array([s["distance"]["value"] for s in steps])
        durations = np.array([s["duration"]["value"] for s in steps])

        total_distance = int(distances.sum())
        total_duration = int(durations.sum())

        co2 = np.random.randint(0, 100, len(steps))
        scenic = np.random.randint(1, 10, len(steps))
        safety = np.random.randint(1, 10, len(steps))

        weights = distances / total_distance

        features = np.c_[100 - co2, scenic, safety, weights]
        preference = float(model.predict(features).sum())

        summaries.append(
            RouteSummary(
                distance_m=total_distance,
                duration_s=total_duration,
                co2_score=float(np.average(co2, weights=weights)),
                scenic_score=float(np.average(scenic, weights=weights)),
                safety_score=float(np.average(safety, weights=weights)),
                preference_score=preference,
            )
        )

    if prioritize_time:
        summaries.sort(key=lambda r: r.duration_s)
    else:
        summaries.sort(key=lambda r: r.preference_score, reverse=True)

    return pd.DataFrame([vars(s) for s in summaries])
