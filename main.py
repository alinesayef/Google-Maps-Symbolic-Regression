import argparse
import os
from datetime import datetime

import googlemaps

from route_analysis import evaluate_routes, train_symbolic_regressor


def initialize_google_api_client():
    api_key = "GOOGLE_API_KEY_HERE"

    return googlemaps.Client(api_key)


def get_directions(gmaps, start_location, end_location):
    return gmaps.directions(
        start_location,
        end_location,
        mode="driving",
        departure_time=datetime.now(),
        alternatives=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("start_location")
    parser.add_argument("end_location")
    parser.add_argument("--air", type=float, default=0)
    parser.add_argument("--scene", type=float, default=0)
    parser.add_argument("--safe", type=float, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    weights = [args.air, args.scene, args.safe]

    if sum(weights) == 0:
        prioritize_time = True
    else:
        prioritize_time = False
        total = sum(weights)
        weights = [w / total for w in weights]

    gmaps = initialize_google_api_client()
    routes = get_directions(gmaps, args.start_location, args.end_location)

    model = train_symbolic_regressor(*weights)
    results = evaluate_routes(routes, model, prioritize_time)

    print(results)


if __name__ == "__main__":
    main()
