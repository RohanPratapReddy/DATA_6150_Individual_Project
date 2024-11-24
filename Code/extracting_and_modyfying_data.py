# -*- coding: utf-8 -*-
"""Extracting and Modyfying Data.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1SRhduwyFu-YeK6MsIGMopige1bBcXM2z
"""

try:
  from acnportal.acndata import DataClient
except ModuleNotFoundError:
  !pip install acnportal

import datetime
Start = datetime.datetime(2020,4,25)
End = datetime.datetime.now()

site_id = 'caltech'
client = DataClient(api_token="CGr-ywZlC1J4gg5tjINtf4RDgz3MD6e5jT7xlauI4GY")
sessions = client.get_sessions_by_time(start=Start,end=End,site=site_id,timeseries=True)

# prompt: how to get json file from above sessions

import json

# Assuming 'sessions' is the variable holding the session data from the previous code.

# Convert the sessions data to a JSON string
sessions_json = json.dumps(sessions, indent=4, default=str)  # Use default=str to handle datetime objects

# Print or save the JSON data to a file
print(sessions_json)
# or
#with open('sessions_data.json', 'w') as f:
    #f.write(sessions_json)

# prompt: create a dictionary from the sessions data above

data = {}
for session in sessions:
  data[session.session_id] = {
      'start_time': session.start_time,
      'end_time': session.end_time,
      'Charging_time': session.charging_time,
      'Connection_time': session.connection_time,
      'Disconnect_time': session.disconnect_time,
      'Done_charging_time': session.done_charging_time,
      'Energy_delivered': session.energy_delivered,
      'Energy_remaining': session.energy_remaining,
      'Energy_limit': session.energy_limit,
      'Energy_rate': session.energy_rate,
      'Energy_rate_unit': session.energy_rate_unit,
      'Energy_rate_multiplier': session.energy_rate_multiplier,
      'Energy_rate_multiplier_unit': session.energy_rate_multiplier_unit,
      'Session_id': session.session_id,
      'Site_id': session.site,
      'Station_id': session.station_id,
      'Cluster_id': session.cluster_id,
      'Space_id': session.space_id,
      'Status': session.status,
      'Timeseries': session.timeseries,
      'User_id': session.user_id,
  }

import pandas as pd
df = pd.DataFrame(data)

data

df.head()