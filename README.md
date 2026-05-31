# Route Recommendation Using Symbolic Regression

## Overview

This project is a proof of concept demonstrating how route recommendations from Google Maps can be enhanced using Symbolic Regression (Genetic Programming).

Rather than ranking routes solely by travel time, the system allows users to express preferences for additional factors such as:

- Air quality
- Scenic value
- Safety

A symbolic regression model is trained using these preferences and used to generate a custom route ranking score for each available route.

> **Note:** This project uses randomly generated values for environmental and route-quality metrics. These values are placeholders and are intended to demonstrate the recommendation framework rather than provide real-world route optimisation.

---

## Requirements

- Python 3.8+
- Google Maps API Key

### Python Packages

Install the required dependencies:

```bash
pip install googlemaps pandas numpy scikit-learn gplearn
```

---

## Google Maps API Setup

1. Create a project in Google Cloud Console.
2. Enable the required Google Maps services (Directions API and/or Routes API depending on your implementation).
3. Create an API key.
4. Insert the API key into the application configuration.

Example:

```python
API_KEY = "YOUR_GOOGLE_MAPS_API_KEY"
```

Alternatively, use an environment variable:

```bash
export GOOGLE_MAPS_API_KEY="YOUR_GOOGLE_MAPS_API_KEY"
```

---

## Usage

### Route Recommendation Without Preferences

This will rank routes primarily based on travel time.

```bash
python main.py "London" "Manchester"
```

### Route Recommendation With Preferences

Users can assign weights to different route criteria:

```bash
python main.py "London" "Manchester" \
    --air 5 \
    --scene 3 \
    --safe 2
```

### Parameters

| Parameter | Description |
|------------|------------|
| `--air` | Preference for cleaner air quality |
| `--scene` | Preference for more scenic routes |
| `--safe` | Preference for safer routes |

Higher values indicate greater importance.

---

## How It Works

1. Alternative routes are retrieved from Google Maps.
2. Route information is extracted and processed.
3. Environmental and route-quality metrics are assigned to each route segment.
4. A symbolic regression model is trained using the user-defined preference weights.
5. Routes are scored and ranked according to the generated symbolic expression.
6. Results are exported for comparison and analysis.

---

## Limitations

This project is intended as a research and demonstration prototype.

Current limitations include:

- Environmental metrics are generated using placeholder values.
- Route quality data is not sourced from live external datasets.
- The symbolic regression model is trained on synthetic data.
- Results should not be interpreted as real-world route recommendations.

Potential future improvements include:

- Integration of live air quality data
- Road safety and accident statistics
- Real-time traffic information
- Scenic route datasets
- User feedback and preference learning

---

## Example Output

```text
route 2 is the best route
```

or

```text
route 1 is the fastest of the routes
```

Depending on the preferences supplied by the user.

---

## License

This software is distributed under the terms specified in the accompanying license file.
