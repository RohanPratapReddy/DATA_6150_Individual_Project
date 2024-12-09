import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from multiprocessing import Pool


def preprocess_data(acndata, stationdata):
    stationdata['session_start'] = pd.to_datetime(stationdata['session_start'])
    stationdata['session_end'] = pd.to_datetime(stationdata['session_end'])
    acndata['connectionTime'] = pd.to_datetime(acndata['connectionTime'])
    acndata['doneChargingTime'] = pd.to_datetime(acndata['doneChargingTime'])
    return acndata, stationdata


def optimize_batch(batch):
    def objective(x):
        return np.sum(x)

    constraints = [
        {'type': 'ineq', 'fun': lambda x: x - batch['Power_Limit']},  # Power constraint
        {'type': 'ineq', 'fun': lambda x: batch['Duration_Limit'] - x / batch['Power_Limit']},  # Time constraint
        {'type': 'ineq', 'fun': lambda x: batch['Session_Energy'] - x}  # Energy cannot exceed original
    ]

    x0 = batch['Session_Energy'].values

    result = minimize(objective, x0, constraints=constraints, method='SLSQP')
    return result.x if result.success else batch['Session_Energy'].values


def calculate_power_new(stationdata, acndata):
    stationdata = stationdata.merge(
        acndata.groupby('stationID').agg(
            connectionTime=('connectionTime', 'min'),
            doneChargingTime=('doneChargingTime', 'max')
        ).reset_index(),
        on='stationID',
        how='left'
    )

    conditions = [
        (stationdata['connectionTime'] < stationdata['session_start']) & (
                stationdata['session_end'] < stationdata['doneChargingTime']),
        (stationdata['session_start'] < stationdata['connectionTime']) & (
                stationdata['connectionTime'] < stationdata['session_end']),
        (stationdata['session_start'] < stationdata['doneChargingTime']) & (
                stationdata['doneChargingTime'] < stationdata['session_end']),
        (stationdata['session_start'] < stationdata['connectionTime']) & (
                stationdata['doneChargingTime'] < stationdata['session_end'])
    ]

    choices = [
        stationdata['Average_Power'],
        stationdata['Average_Power'] * (
                stationdata['session_end'] - stationdata['connectionTime']).dt.total_seconds() / (
                stationdata['session_end'] - stationdata['session_start']).dt.total_seconds(),
        stationdata['Average_Power'] * (
                stationdata['doneChargingTime'] - stationdata['session_start']).dt.total_seconds() / (
                stationdata['session_end'] - stationdata['session_start']).dt.total_seconds(),
        stationdata['Average_Power'] * (
                stationdata['doneChargingTime'] - stationdata['connectionTime']).dt.total_seconds() / (
                stationdata['session_end'] - stationdata['session_start']).dt.total_seconds()
    ]

    stationdata['Power_New'] = np.select(conditions, choices, default=0)
    stationdata['Total_Power'] = stationdata.groupby('stationID')['Power_New'].transform('sum')
    stationdata.drop(columns=['Average_Power'], inplace=True)
    return stationdata


def calculate_session_energy(acndata, stationdata):
    stationdata['session_duration'] = (stationdata['session_end'] - stationdata[
        'session_start']).dt.total_seconds() / 3600
    stationdata['Session_Energy'] = stationdata['Total_Power'] * stationdata['session_duration']

    acndata['session_duration'] = pd.to_timedelta(acndata['sessionDuration']).dt.total_seconds() / 3600
    acndata['Session_Energy'] = acndata['kWhDelivered'] * acndata['session_duration']
    return acndata, stationdata


def link_pricing_data(acndata, stationdata, electricity_pricing):
    def get_season(month):
        if month in [1, 2, 3, 4, 5]:
            return 'January-May'
        elif month in [6, 7, 8, 9]:
            return 'June-September'
        else:
            return 'October-December'

    # Ensure Season is computed for both datasets
    stationdata['Year'] = stationdata['session_start'].dt.year
    stationdata['Month'] = stationdata['session_start'].dt.month
    stationdata['Season'] = stationdata['Month'].apply(get_season)

    acndata['Year'] = acndata['connectionTime'].dt.year
    acndata['Month'] = acndata['connectionTime'].dt.month
    acndata['Season'] = acndata['Month'].apply(get_season)

    # Verify existence of keys in electricity_pricing
    if not all(key in electricity_pricing.columns for key in ['Year', 'Season']):
        raise KeyError("The 'electricity_pricing' DataFrame must contain 'Year' and 'Season' columns.")

    # Melt and check for necessary columns
    required_rate_columns = ['Tier 1 Rate ($/kWh)', 'Tier 2 Rate ($/kWh)', 'Tier 3 Rate ($/kWh)']
    if not all(col in electricity_pricing.columns for col in required_rate_columns):
        raise KeyError("Missing one or more required rate columns: " + ", ".join(required_rate_columns))

    # Merge electricity pricing
    stationdata = stationdata.merge(
        electricity_pricing,
        left_on=['Year', 'Season'],
        right_on=['Year', 'Season'],
        how='left'
    )

    acndata = acndata.merge(
        electricity_pricing,
        left_on=['Year', 'Season'],
        right_on=['Year', 'Season'],
        how='left'
    )

    # Check for presence of 'Rate' or compute from melted columns
    if 'Rate' not in stationdata.columns or 'Rate' not in acndata.columns:
        melted = electricity_pricing.melt(
            id_vars=['Year', 'Season'],
            value_vars=required_rate_columns,
            var_name='Tier',
            value_name='Rate'
        )
        # Example fallback to assign a default rate
        acndata['Rate'] = melted['Rate'].mean()
        stationdata['Rate'] = melted['Rate'].mean()

    stationdata['Session_Cost'] = stationdata['Rate']
    stationdata['Energy_Cost'] = stationdata['Session_Energy'] * stationdata['Session_Cost']

    acndata['Session_Cost'] = acndata['Rate']
    acndata['Energy_Cost'] = acndata['Session_Energy'] * acndata['Session_Cost']
    return acndata, stationdata


def aggregate_bimonthly(acndata, stationdata):
    bimonthly_map = {
        1: 'Jan-Feb', 2: 'Jan-Feb', 3: 'Mar-Apr', 4: 'Mar-Apr',
        5: 'May-Jun', 6: 'May-Jun', 7: 'Jul-Aug', 8: 'Jul-Aug',
        9: 'Sep-Oct', 10: 'Sep-Oct', 11: 'Nov-Dec', 12: 'Nov-Dec'
    }

    acndata['Bimonthly'] = acndata['Month'].map(bimonthly_map)
    stationdata['Bimonthly'] = stationdata['Month'].map(bimonthly_map)

    acndata_agg = acndata.groupby(['stationID', 'Bimonthly']).agg(
        Total_Energy=('Session_Energy', 'sum'),
        Total_Cost=('Energy_Cost', 'sum')
    ).reset_index()

    stationdata_agg = stationdata.groupby(['stationID', 'Bimonthly']).agg(
        Total_Energy=('Session_Energy', 'sum'),
        Total_Cost=('Energy_Cost', 'sum')
    ).reset_index()

    return acndata_agg, stationdata_agg


def batch_optimization_and_scheduling(acndata, stationdata, batch_size=1000, n_processes=4):
    combined_data = pd.concat([
        acndata[['session_duration', 'powerkW', 'kWhDelivered']].rename(columns={
            'session_duration': 'Duration_Limit',
            'powerkW': 'Power_Limit',
            'kWhDelivered': 'Session_Energy'
        }),
        stationdata[['session_duration', 'Power_New', 'Total_Power']].rename(columns={
            'session_duration': 'Duration_Limit',
            'Power_New': 'Power_Limit',
            'Total_Power': 'Session_Energy'
        })
    ], ignore_index=True)

    batches = [combined_data.iloc[i:i + batch_size] for i in range(0, len(combined_data), batch_size)]
    with Pool(n_processes) as pool:
        results = pool.map(optimize_batch, batches)

    combined_data['Energy_Scheduled'] = np.concatenate(results)

    acndata_results = combined_data.iloc[:len(acndata)].copy()
    stationdata_results = combined_data.iloc[len(acndata):].copy()

    acndata['Energy_Scheduled'] = acndata_results['Energy_Scheduled']
    stationdata['Energy_Scheduled'] = stationdata_results['Energy_Scheduled']

    acndata['charging_power_scheduled'] = acndata['Energy_Scheduled'] / acndata['session_duration']
    acndata['chargingTime_scheduled'] = acndata['Energy_Scheduled'] / acndata['powerkW']

    acndata['chargingTime_scheduled'] = np.minimum(acndata['chargingTime_scheduled'], acndata['session_duration'])
    acndata['charging_power_scheduled'] = np.maximum(acndata['charging_power_scheduled'], acndata['powerkW'])

    return acndata, stationdata


def fast_optimization(acndata, stationdata):
    # Combine data
    combined_data = pd.concat([
        acndata[['session_duration', 'powerkW', 'kWhDelivered']].rename(columns={
            'session_duration': 'Duration_Limit',
            'powerkW': 'Power_Limit',
            'kWhDelivered': 'Session_Energy'
        }),
        stationdata[['session_duration', 'Power_New', 'Total_Power']].rename(columns={
            'session_duration': 'Duration_Limit',
            'Power_New': 'Power_Limit',
            'Total_Power': 'Session_Energy'
        })
    ], ignore_index=True)

    # Compute scheduled energy vectorized
    scheduled_energy = np.minimum(
        combined_data['Session_Energy'],  # Can't exceed original energy
        combined_data['Power_Limit'] * combined_data['Duration_Limit']  # Power * time constraint
    )

    combined_data['Energy_Scheduled'] = scheduled_energy

    # Split results
    acndata['Energy_Scheduled'] = combined_data.iloc[:len(acndata)]['Energy_Scheduled'].values
    stationdata['Energy_Scheduled'] = combined_data.iloc[len(acndata):]['Energy_Scheduled'].values

    # Compute charging power and time
    acndata['charging_power_scheduled'] = acndata['Energy_Scheduled'] / acndata['session_duration']
    acndata['chargingTime_scheduled'] = acndata['Energy_Scheduled'] / acndata['powerkW']

    # Ensure constraints are satisfied
    acndata['chargingTime_scheduled'] = np.minimum(acndata['chargingTime_scheduled'], acndata['session_duration'])
    acndata['charging_power_scheduled'] = np.maximum(acndata['charging_power_scheduled'], acndata['powerkW'])

    return acndata, stationdata


# Ensure chargingTime_scheduled is not used prematurely
def generate_charging_time_chart(acndata):
    if 'chargingTime_scheduled' not in acndata.columns:
        raise KeyError("'chargingTime_scheduled' column is missing. Ensure it is computed first.")

    acndata['chargingTime%'] = (acndata['chargingTime_scheduled'] / acndata['session_duration']) * 100
    station_agg = acndata.groupby(['stationID', 'Month']).agg({'chargingTime%': 'mean'}).reset_index()

    stations = station_agg['stationID'].unique()
    for i in range(0, len(stations), 5):
        subset = station_agg[station_agg['stationID'].isin(stations[i:i + 5])]
        for station in subset['stationID'].unique():
            plt.plot(subset[subset['stationID'] == station]['Month'],
                     subset[subset['stationID'] == station]['chargingTime%'],
                     label=f"Station {station}")
        plt.xlabel('Month')
        plt.ylabel('Charging Time (%)')
        plt.title('Charging Time Percentage by Station')
        plt.legend()
        plt.show()


def plot_tariff_trends(electricity_pricing):
    # Melt the dataframe to plot tariffs by tier across periods
    pricing_melted = electricity_pricing.melt(
        id_vars=['Year', 'Season'],
        value_vars=['Tier 1 Rate ($/kWh)', 'Tier 2 Rate ($/kWh)', 'Tier 3 Rate ($/kWh)'],
        var_name='Tier',
        value_name='Rate'
    )

    # Create a unique time period for easier plotting
    pricing_melted['Time_Period'] = pricing_melted['Year'].astype(str) + ' - ' + pricing_melted['Season']

    # Plot tariffs across time
    plt.figure(figsize=(12, 6))
    for tier in pricing_melted['Tier'].unique():
        tier_data = pricing_melted[pricing_melted['Tier'] == tier]
        plt.plot(tier_data['Time_Period'], tier_data['Rate'], marker='o', label=tier)

    plt.xlabel('Time Period')
    plt.ylabel('Rate ($/kWh)')
    plt.title('Electricity Tariffs Across Different Periods')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Tier')
    plt.tight_layout()
    plt.show()


def shift_charging_sessions(stationdata, shift_percentage=30):
    # Define peak and off-peak hours
    peak_hours = list(range(18, 22))  # 6 PM to 10 PM
    off_peak_hours = list(range(22, 24)) + list(range(0, 6))  # 10 PM to 6 AM

    # Add hour column for session start
    stationdata['start_hour'] = stationdata['session_start'].dt.hour

    # Identify peak-hour sessions
    peak_sessions = stationdata[stationdata['start_hour'].isin(peak_hours)]

    # Select a percentage of peak-hour sessions to shift
    num_shift = int(len(peak_sessions) * shift_percentage / 100)
    sessions_to_shift = peak_sessions.sample(num_shift)

    # Shift these sessions to off-peak hours
    sessions_to_shift['session_start'] += pd.Timedelta(hours=8)  # Shift by 8 hours to off-peak
    sessions_to_shift['session_end'] += pd.Timedelta(hours=8)

    # Explicitly cast to match original data types
    for col in ['session_start', 'session_end', 'start_hour']:
        sessions_to_shift[col] = sessions_to_shift[col].astype(stationdata[col].dtype)

    # Update original data
    stationdata.update(sessions_to_shift)

    return stationdata


def calculate_grid_load_and_cost(stationdata, electricity_pricing):
    try:
        # Group data by hour to calculate grid load
        stationdata['hour'] = stationdata['session_start'].dt.hour
        hourly_load = stationdata.groupby('hour')['Session_Energy'].sum()

        # Merge with pricing to calculate costs
        stationdata = stationdata.merge(electricity_pricing, on=['Year', 'Season'], how='left')
        if 'Rate' not in stationdata.columns:
            raise KeyError("'Rate' column missing after merge. Check electricity_pricing data structure.")

        stationdata['Cost'] = stationdata['Session_Energy'] * stationdata['Rate']

        total_cost = stationdata['Cost'].sum()

        # Validate results
        print(f"Hourly Load (Sample):\n{hourly_load.head()}")
        print(f"Total Cost: ${total_cost:.2f}")

        return hourly_load, total_cost
    except Exception as e:
        print(f"Error in calculate_grid_load_and_cost: {e}")
        raise


def plot_comparison(original_load, shifted_load):
    plt.figure(figsize=(10, 6))
    plt.plot(original_load.index, original_load.values, label='Original Grid Load')
    plt.plot(shifted_load.index, shifted_load.values, label='Shifted Grid Load', linestyle='--')
    plt.xlabel('Hour of Day')
    plt.ylabel('Grid Load (kWh)')
    plt.title('Grid Load Before and After Shifting Charging Sessions')
    plt.legend()
    plt.show()


def plot_tariffs_over_time(electricity_pricing):
    # Melt the pricing data to long format
    pricing_melted = electricity_pricing.melt(
        id_vars=['Year', 'Season'],
        value_vars=['Tier 1 Rate ($/kWh)', 'Tier 2 Rate ($/kWh)', 'Tier 3 Rate ($/kWh)'],
        var_name='Tariff Tier',
        value_name='Rate ($/kWh)'
    )

    # Create a combined time period column for easier plotting
    pricing_melted['Time Period'] = pricing_melted['Year'].astype(str) + ' - ' + pricing_melted['Season']

    # Sort by time period for consistent plotting
    pricing_melted.sort_values(by=['Year', 'Season'], inplace=True)

    # Plot tariffs across time
    plt.figure(figsize=(12, 6))
    for tier in pricing_melted['Tariff Tier'].unique():
        tier_data = pricing_melted[pricing_melted['Tariff Tier'] == tier]
        plt.plot(tier_data['Time Period'], tier_data['Rate ($/kWh)'], marker='o', label=tier)

    plt.xlabel('Time Period')
    plt.ylabel('Rate ($/kWh)')
    plt.title('Tariffs Across Different Periods of Time')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Tariff Tier')
    plt.tight_layout()
    plt.show()
