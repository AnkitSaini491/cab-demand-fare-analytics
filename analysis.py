
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# CAB DEMAND & FARE ANALYSIS
# ==========================================

# Load dataset
df = pd.read_csv("data/cab_data.csv")

# Convert date column
df["date"] = pd.to_datetime(df["date"])

print("\n====================================")
print("     CAB DEMAND & FARE ANALYSIS")
print("====================================\n")


# ==========================================
# 1. BASIC DATA INFORMATION
# ==========================================

print("----- Dataset Information -----")

print("Total Records:", len(df))
print("Total Columns:", len(df.columns))

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())


# ==========================================
# 2. BASIC STATISTICS
# ==========================================

print("\n----- Basic Statistics -----")

print(df.describe())


# ==========================================
# 3. TOTAL REVENUE
# ==========================================

total_revenue = df["fare"].sum()

average_fare = df["fare"].mean()

average_distance = df["distance_km"].mean()

total_rides = len(df)

print("\n----- Business KPIs -----")

print("Total Rides:", total_rides)

print("Total Revenue: ₹", round(total_revenue, 2))

print("Average Fare: ₹", round(average_fare, 2))

print("Average Distance:",
      round(average_distance, 2), "km")


# ==========================================
# 4. HOURLY DEMAND ANALYSIS
# ==========================================

hourly_demand = (
    df.groupby("hour")
    .size()
    .reset_index(name="rides")
)

hourly_demand = hourly_demand.sort_values(
    "rides",
    ascending=False
)

print("\n----- Hourly Demand -----")

print(hourly_demand)

peak_hour = hourly_demand.iloc[0]["hour"]

print(
    "\nPeak Demand Hour:",
    int(peak_hour)
)


# Save report
hourly_demand.to_csv(
    "hourly_demand_report.csv",
    index=False
)


# ==========================================
# 5. CAB TYPE ANALYSIS
# ==========================================

cab_analysis = (
    df.groupby("cab_type")
    .agg(
        total_rides=("fare", "count"),
        total_revenue=("fare", "sum"),
        average_fare=("fare", "mean"),
        average_distance=("distance_km", "mean")
    )
    .reset_index()
)

print("\n----- Cab Type Analysis -----")

print(cab_analysis)


cab_analysis.to_csv(
    "cab_type_analysis.csv",
    index=False
)


# ==========================================
# 6. PICKUP LOCATION ANALYSIS
# ==========================================

pickup_analysis = (
    df.groupby("pickup")
    .agg(
        total_rides=("fare", "count"),
        total_revenue=("fare", "sum"),
        average_fare=("fare", "mean")
    )
    .reset_index()
)

pickup_analysis = pickup_analysis.sort_values(
    "total_rides",
    ascending=False
)

print("\n----- Pickup Location Analysis -----")

print(pickup_analysis)


pickup_analysis.to_csv(
    "pickup_location_analysis.csv",
    index=False
)


# ==========================================
# 7. DAILY ANALYSIS
# ==========================================

daily_analysis = (
    df.groupby("day")
    .agg(
        rides=("fare", "count"),
        revenue=("fare", "sum"),
        average_fare=("fare", "mean")
    )
    .reset_index()
)

print("\n----- Day Analysis -----")

print(daily_analysis)


daily_analysis.to_csv(
    "daily_analysis.csv",
    index=False
)


# ==========================================
# 8. DISTANCE ANALYSIS
# ==========================================

print("\n----- Distance Analysis -----")

print(
    "Minimum Distance:",
    df["distance_km"].min(),
    "km"
)

print(
    "Maximum Distance:",
    df["distance_km"].max(),
    "km"
)

print(
    "Average Distance:",
    round(
        df["distance_km"].mean(),
        2
    ),
    "km"
)


# ==========================================
# 9. FARE ANALYSIS
# ==========================================

print("\n----- Fare Analysis -----")

print(
    "Minimum Fare: ₹",
    df["fare"].min()
)

print(
    "Maximum Fare: ₹",
    df["fare"].max()
)

print(
    "Average Fare: ₹",
    round(
        df["fare"].mean(),
        2
    )
)


# ==========================================
# 10. DISTANCE VS FARE CORRELATION
# ==========================================

correlation = df[
    ["distance_km", "fare"]
].corr()

print("\n----- Distance & Fare Correlation -----")

print(correlation)

print(
    "\nCorrelation:",
    round(
        correlation.loc[
            "distance_km",
            "fare"
        ],
        3
    )
)


# ==========================================
# 11. VISUALIZATION - HOURLY DEMAND
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    hourly_demand["hour"],
    hourly_demand["rides"],
    marker="o"
)

plt.title(
    "Hourly Cab Demand"
)

plt.xlabel(
    "Hour"
)

plt.ylabel(
    "Number of Rides"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "hourly_demand.png"
)

plt.show()


# ==========================================
# 12. VISUALIZATION - CAB REVENUE
# ==========================================

plt.figure(figsize=(8, 5))

plt.bar(
    cab_analysis["cab_type"],
    cab_analysis["total_revenue"]
)

plt.title(
    "Revenue by Cab Type"
)

plt.xlabel(
    "Cab Type"
)

plt.ylabel(
    "Revenue (₹)"
)

plt.tight_layout()

plt.savefig(
    "cab_revenue.png"
)

plt.show()


# ==========================================
# 13. VISUALIZATION - DISTANCE VS FARE
# ==========================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["distance_km"],
    df["fare"],
    alpha=0.4
)

plt.title(
    "Distance vs Fare"
)

plt.xlabel(
    "Distance (km)"
)

plt.ylabel(
    "Fare (₹)"
)

plt.tight_layout()

plt.savefig(
    "distance_vs_fare.png"
)

plt.show()


# ==========================================
# 14. FINAL SUMMARY
# ==========================================

print("\n====================================")
print("             SUMMARY")
print("====================================")

print(
    "Total Rides:",
    total_rides
)

print(
    "Total Revenue: ₹",
    round(total_revenue, 2)
)

print(
    "Average Fare: ₹",
    round(average_fare, 2)
)

print(
    "Average Distance:",
    round(average_distance, 2),
    "km"
)

print(
    "Peak Demand Hour:",
    int(peak_hour)
)

print("\nAnalysis completed successfully!")

print(
    "\nReports and charts have been generated."
)

Important: Is "analysis.py" ko chalane ke liye project ke andar "data/cab_data.csv" hona chahiye.

Run:

python analysis.py

Ye automatically ye reports/charts generate karega:

hourly_demand_report.csv
cab_type_analysis.csv
pickup_location_analysis.csv
daily_analysis.csv

hourly_demand.png
cab_revenue.png
distance_vs_fare.png
