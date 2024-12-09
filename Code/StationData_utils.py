import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from ipywidgets import interact
import matplotlib

matplotlib.use('TkAgg')


def Load_Csv(file_path):
    df = pd.read_csv(file_path)
    return df


def analyze_hourly_sessions(data):
    # Convert time columns to datetime
    for col in ['connectionTime', 'disconnectTime', 'doneChargingTime']:
        data[col] = pd.to_datetime(data[col])

    # Create hourly sessions
    sessions = []

    # Iterate over unique dates
    for date in data['connectionTime'].dt.date.unique():
        for hour in range(24):
            session_start = datetime.combine(date, datetime.min.time()) + timedelta(hours=hour)
            session_end = session_start + timedelta(hours=1)

            session_data = data[
                ((data['connectionTime'] < session_end) & (data['doneChargingTime'] > session_start)) |
                ((data['doneChargingTime'] < session_end) & (data['disconnectTime'] > session_start))
                ]

            if session_data.empty:
                continue

            charging_users = session_data[
                (session_data['connectionTime'] < session_end) & (session_data['doneChargingTime'] > session_start)
                ].shape[0]

            standby_users = session_data[
                (session_data['doneChargingTime'] < session_end) & (session_data['disconnectTime'] > session_start)
                ].shape[0]

            # Calculate Power_New for each session
            def calculate_power(row):
                if row['connectionTime'] <= session_start and session_end <= row['doneChargingTime']:
                    return row['powerkW']
                elif session_start <= row['connectionTime'] < session_end:
                    overlap = (session_end - row['connectionTime']).total_seconds() / 3600
                    session_duration = (session_end - session_start).total_seconds() / 3600
                    return row['powerkW'] * (overlap / session_duration)
                elif session_start <= row['doneChargingTime'] < session_end:
                    overlap = (row['doneChargingTime'] - session_start).total_seconds() / 3600
                    session_duration = (session_end - session_start).total_seconds() / 3600
                    return row['powerkW'] * (overlap / session_duration)
                elif session_start <= row['connectionTime'] and row['doneChargingTime'] <= session_end:
                    overlap = (row['doneChargingTime'] - row['connectionTime']).total_seconds() / 3600
                    session_duration = (session_end - session_start).total_seconds() / 3600
                    return row['powerkW'] * (overlap / session_duration)
                return 0

            # Use .loc to avoid SettingWithCopyWarning
            session_data = session_data.copy()
            session_data.loc[:, 'Power_New'] = session_data.apply(calculate_power, axis=1)
            average_power = session_data['Power_New'].mean()

            for _, row in session_data.iterrows():
                sessions.append({
                    'date': session_start.date(),
                    'session': f'{session_start.hour:02d}:00-{session_end.hour:02d}:00',
                    'session_start': session_start,
                    'session_end': session_end,
                    'clusterID': str(row['clusterID']),
                    'siteID': str(row['siteID']),
                    'stationID': str(row['stationID']),
                    'chargingUsers': charging_users,
                    'standbyUsers': standby_users,
                    'Average_Power': average_power
                })

    # Create a new DataFrame
    sessions_df = pd.DataFrame(sessions)

    # Return the resulting DataFrame
    return sessions_df


def plot_data(df, selected_date):
    """
    Plot the metrics for a specific date separately.

    Args:
        df (pd.DataFrame): The dataset containing the metrics.
        selected_date (str): The date for which metrics should be plotted.
    """
    filtered_data = df[df['date'] == selected_date]

    # Check if there is data for the selected date
    if filtered_data.empty:
        print(f"No data available for {selected_date}.")
        return

    # Plot Charging Users
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_data['session'], filtered_data['chargingUsers'], label='Charging Users', marker='o')
    plt.xlabel('Session')
    plt.ylabel('Charging Users')
    plt.title(f"Charging Users for {selected_date}")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot Standby Users
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_data['session'], filtered_data['standbyUsers'], label='Standby Users', marker='o')
    plt.xlabel('Session')
    plt.ylabel('Standby Users')
    plt.title(f"Standby Users for {selected_date}")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot Average Power
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_data['session'], filtered_data['Average_Power'], label='Average Power', marker='o')
    plt.xlabel('Session')
    plt.ylabel('Average Power')
    plt.title(f"Average Power for {selected_date}")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def get_user_input_plot(df):
    """
    Continuously ask the user for a date input and plot the data for that date until the user exits.

    Args:
        df (pd.DataFrame): The dataset containing the metrics.
    """
    dates = df['date'].unique()
    print("Available dates:")
    for date in dates:
        print(date)

    while True:
        selected_date = input("Enter a date from the above list (or type 'e', 'exit', or 'Exit' to quit): ")
        if selected_date.lower() in ['e', 'exit']:
            print("Exiting the program.")
            break
        if selected_date not in dates:
            print("Invalid date entered. Please try again or type 'e' to exit.")
            continue

        plot_data(df, selected_date)


def create_interactive_plot(df):
    """
    Create an interactive plot for metrics with a dropdown to select the date.

    Args:
        df (pd.DataFrame): The dataset containing the metrics.
    """
    dates = df['date'].unique()
    interact(lambda selected_date: plot_data(df, selected_date), selected_date=dates)


def preprocess_data(data):
    """
    Preprocess the data for EV charging behavior analysis.
    Args:
        data (pd.DataFrame): Input data.
    Returns:
        pd.DataFrame: Preprocessed data with added columns for month, weekday, and energy consumed.
    """
    data['date'] = pd.to_datetime(data['date'])
    data['month'] = data['date'].dt.month
    data['weekday'] = data['date'].dt.day_name()
    data['year'] = data['date'].dt.year

    # Calculate session duration (assuming session_end and session_start are in the dataset)
    if 'session_start' in data.columns and 'session_end' in data.columns:
        data['session_start'] = pd.to_datetime(data['session_start'])
        data['session_end'] = pd.to_datetime(data['session_end'])
        data['session_duration'] = (data['session_end'] - data['session_start']).dt.total_seconds() / 3600  # Duration in hours

    # Calculate Energy Consumed
    if 'Average_Power' in data.columns:
        data['Energy_Consumed'] = data['Average_Power'] * data['session_duration']
    return data

def aggregate_behavior(data):
    """
    Aggregate behavior for charging users, standby users, and energy consumed by month, weekday, and daily/yearly.
    Args:
        data (pd.DataFrame): Preprocessed data.
    Returns:
        dict: Aggregated data for monthly, weekday, daily, and yearly behavior for each metric.
    """
    monthly_behavior = {}
    weekday_behavior = {}
    daily_behavior = {}
    yearly_behavior = {}

    if 'chargingUsers' in data.columns:
        monthly_behavior['chargingUsers'] = data.groupby('month')['chargingUsers'].sum()
        weekday_behavior['chargingUsers'] = data.groupby('weekday')['chargingUsers'].sum()
        daily_behavior['chargingUsers'] = data.groupby('date')['chargingUsers'].sum()
        yearly_behavior['chargingUsers'] = data.groupby('year')['chargingUsers'].sum()

    if 'standbyUsers' in data.columns:
        monthly_behavior['standbyUsers'] = data.groupby('month')['standbyUsers'].sum()
        weekday_behavior['standbyUsers'] = data.groupby('weekday')['standbyUsers'].sum()
        daily_behavior['standbyUsers'] = data.groupby('date')['standbyUsers'].sum()
        yearly_behavior['standbyUsers'] = data.groupby('year')['standbyUsers'].sum()

    if 'Energy_Consumed' in data.columns:
        monthly_behavior['Energy_Consumed'] = data.groupby('month')['Energy_Consumed'].sum()
        weekday_behavior['Energy_Consumed'] = data.groupby('weekday')['Energy_Consumed'].sum()
        daily_behavior['Energy_Consumed'] = data.groupby('date')['Energy_Consumed'].sum()
        yearly_behavior['Energy_Consumed'] = data.groupby('year')['Energy_Consumed'].sum()

    # Map month numbers to names
    month_names = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April',
        5: 'May', 6: 'June', 7: 'July', 8: 'August',
        9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }
    for key in monthly_behavior:
        monthly_behavior[key].index = monthly_behavior[key].index.map(month_names)

    # Sort weekdays in correct order
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for key in weekday_behavior:
        weekday_behavior[key] = weekday_behavior[key].reindex(weekday_order)

    return monthly_behavior, weekday_behavior, daily_behavior, yearly_behavior

def plot_behavior(monthly_behavior, weekday_behavior, daily_behavior, yearly_behavior, metric):
    """
    Plot the monthly, weekday, daily, and yearly behavior for a given metric.
    Args:
        monthly_behavior (dict): Aggregated monthly data.
        weekday_behavior (dict): Aggregated weekday data.
        daily_behavior (dict): Aggregated daily data.
        yearly_behavior (dict): Aggregated yearly data.
        metric (str): Metric to plot (e.g., 'chargingUsers', 'standbyUsers', 'Energy_Consumed').
    """
    # Monthly behavior
    if metric in monthly_behavior:
        plt.figure(figsize=(12, 6))
        plt.bar(monthly_behavior[metric].index, monthly_behavior[metric].values, color='skyblue')
        plt.title(f'Monthly {metric.replace("_", " ").title()} Behavior')
        plt.xlabel('Month')
        plt.ylabel(f'Total {metric.replace("_", " ").title()}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Weekday behavior
    if metric in weekday_behavior:
        plt.figure(figsize=(12, 6))
        plt.bar(weekday_behavior[metric].index, weekday_behavior[metric].values, color='orange')
        plt.title(f'Weekly {metric.replace("_", " ").title()} Behavior')
        plt.xlabel('Day of the Week')
        plt.ylabel(f'Total {metric.replace("_", " ").title()}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Daily behavior
    if metric in daily_behavior:
        plt.figure(figsize=(12, 6))
        plt.plot(daily_behavior[metric].index, daily_behavior[metric].values, color='green')
        plt.title(f'Daily {metric.replace("_", " ").title()} Behavior')
        plt.xlabel('Date')
        plt.ylabel(f'Total {metric.replace("_", " ").title()}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Yearly behavior
    if metric in yearly_behavior:
        plt.figure(figsize=(12, 6))
        plt.bar(yearly_behavior[metric].index.astype(str), yearly_behavior[metric].values, color='purple')
        plt.title(f'Yearly {metric.replace("_", " ").title()} Behavior')
        plt.xlabel('Year')
        plt.ylabel(f'Total {metric.replace("_", " ").title()}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

