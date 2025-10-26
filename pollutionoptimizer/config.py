# config.py

import os

# --- CORE CONFIGURATION ---
LOCATION = "Thyagaraya Nagar, Chennai, India"
NETWORK_DISTANCE = 800  # Distance in meters for the street network download

# --- API KEY (Read from environment variable for security) ---
# Set this variable in your .env file or VS Code settings
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "YOUR_API_KEY") 

# --- CONTROL FLAGS ---
RUN_POLLUTION_ANALYSIS = True
RUN_WEATHER_FETCH = True

# --- ROUTING ENDPOINTS (Sample) ---
# These will be dynamically found in main.py, but here are constants if needed
# Define coordinates near your target area for robust node finding
START_COORDS = (13.0431, 80.2436)
END_COORDS = (13.0465, 80.2500)