# Description
This software is a proof of concept of how Google Maps route reccomendations can be improved using symbolic regression (Genetic Programming). 

# Instructions
Please insert your Google API key in the specified place in the code file. You will need to create a project in Google Cloud Console and add the Geocoding API service to that project, there is online help available on how to do it.

When running the code, please provide two valid addresses as arguments, followed by your route preferences, please note that the first preference value you specify will be for air quality score, the second will be for the scenic score and the final value will be for the saftey score. An example of the correct format is shown below (don't include anything in brackets when running the command);

python Google-Maps-Symbolic-Regression.py Origin_Address Destiation_Address 60(air quality) 20(scenic score) 20(safety score)

# License
This software is subject to a license, please refer to the license file for more information.
