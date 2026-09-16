
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Cab Demand & Fare Analytics",
    page_icon="🚕",
    layout="wide"
)


# -----------------------------
# Create Demo Dataset
# -----------------------------
@st.cache_data
def create_data(n=5000):

    np.random.seed(42)

    dates = pd.date_range(
        "2025-01-01",
        "2025-12-31",
        freq="h"
    )

    pickup_locations = [
        "Downtown",
        "Airport",
        "Railway Station",
        "Mall Road",
        "Business Park",
        "University"
    ]

    drop_locations = [
        "Downtown",
        "Airport",
        "Railway Station",
        "Mall Road",
        "Business Park",
        "University"
    ]

    cab_types = [
        "Mini",
        "Sedan",
        "SUV",
        "Premium"
    ]

    pickup = np.random.choice(
        pickup_locations,
        n
    )

    drop = np.random.choice(
        drop_locations,
        n
    )

    cab_type = np.random.choice(
        cab_types,
        n,
        p=[0.35, 0.35, 0.20, 0.10]
    )

    distance = np.random.gamma(
        2.5,
        3.2,
        n
    ) + 1

    distance = np.clip(
        distance,
        1,
        40
    )

    distance = np.round(
        distance,
        2
    )

    passengers = np.random.randint(
        1,
        6,
        n
    )

    date = np.random.choice(
        dates,
        n
    )

    date = pd.to_datetime(date)

    hour = date.hour

    day = date.day_name()

    surge = np.where(
        (
            ((hour >= 8) & (hour <= 10))
            |
            ((hour >= 17) & (hour <= 20))
        ),
        1.25,
        1.0
    )

    cab_multiplier = {
        "Mini": 1.0,
        "Sedan": 1.25,
        "SUV": 1.55,
        "Premium": 2.0
    }

    multiplier = np.array(
        [
            cab_multiplier[x]
            for x in cab_type
        ]
    )

    base_fare = (
        60
        + distance * 18
        + multiplier * 45
    )

    fare = (
        base_fare * surge
        + passengers * 8
        + np.random.normal(0, 45, n)
    )

    fare = np.maximum(
        fare,
        60
    )

    fare = np.round(
        fare,
        2
    )

    df = pd.DataFrame({
        "date": date,
        "pickup": pickup,
        "drop": drop,
        "cab_type": cab_type,
        "distance_km": distance,
        "passengers": passengers,
        "day": day,
        "hour": hour,
        "fare": fare
    })

    return df


# -----------------------------
# Machine Learning Model
# -----------------------------
@st.cache_resource
def train_model(df):

    features = [
        "distance_km",
        "passengers",
        "hour",
        "cab_type",
        "pickup",
        "drop"
    ]

    X = df[features]

    y = df["fare"]

    X = pd.get_dummies(
        X,
        drop_first=False
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=14,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )
