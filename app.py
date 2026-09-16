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
