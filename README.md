# DATA_6150_Individual_Project
# Optimization of EV Charging Stations

## Project Overview

This repository is dedicated to the optimization of electric vehicle (EV) charging stations using data-driven methodologies. The project investigates EV charging patterns, forecasts demand, and optimizes scheduling to reduce grid strain, enhance energy efficiency, and minimize reliance on non-renewable energy sources.

### Key Objectives:
- Analyze EV charging patterns and behaviors.
- Implement forecasting models for demand prediction.
- Develop optimization strategies to improve station efficiency.
- Enable renewable energy integration and sustainable practices.

---

## Repository Contents

### 1. **Documents**
- **`DATA 6150_Individual_Project_Report.docx`**: Comprehensive report outlining the methodology, findings, and implications of the project.

### 2. **Datasets**
- **`acndata_sessions_Caltech.json`**: Charging session data from Caltech stations.
- **`acndata_sessions_JPL.json`**: Charging session data from JPL stations.
- **`acndata_sessions_Office_1.json`**: Charging session data from Office 1 stations.

### 3. **Notebooks**
- **`Individual_Project.ipynb`**: Python notebook containing data preprocessing, analysis, and visualization steps.

### 4. **Utilities**
- **`StationData_utils.py`**: Functions for processing station data, analyzing sessions, and visualizing user behaviors.
- **`StationForecast_utils.py`**: Forecasting utilities implementing models like ARIMA, SARIMAX, Prophet, and tree-based methods.
- **`acndata_utils.py`**: Tools for handling and repairing JSON data, extracting user inputs, and interacting with APIs.
- **`Optimization_utils.py`**: Optimization routines using constraint-based methods to improve scheduling and resource allocation.

---

## Setup and Usage

### Prerequisites
Ensure the following libraries are installed:
- `pandas`
- `numpy`
- `matplotlib`
- `scipy`
- `sklearn`
- `tensorflow` (for LSTM models)
- `xgboost`
- `prophet`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Code
1. Load the datasets:
   - Place the `json` files in the `data/` directory.
   - Modify file paths in scripts as needed.
2. Execute the `Individual_Project.ipynb` notebook to:
   - Preprocess datasets.
   - Visualize charging patterns.
   - Train forecasting models.
   - Optimize scheduling strategies.

---

## Key Features

### 1. Data Analysis
- Hourly session analysis.
- User behavior visualization.
- Energy consumption trends.

### 2. Forecasting
- Time-series analysis using ARIMA, SARIMAX, Prophet, and tree-based methods.
- Demand prediction with interactive visualizations.

### 3. Optimization
- Session scheduling based on energy and time constraints.
- Integration with renewable energy sources.
- Cost minimization strategies using advanced optimization algorithms.

---

## Contributions
Contributions are welcome! Feel free to submit issues, feature requests, or pull requests.

---

## Contact
- **Author**: Rohan Pratap Reddy Ravula
- **Email**: ravular@wit.edu
- **Institution**: Wentworth Institute of Technology

---

## Acknowledgement
This project was made possible with support from the Wentworth Institute of Technology and data provided by the Caltech EV Data Platform and the US Energy Information Administration.

---

## References
- [Caltech EV Data Platform](https://ev.caltech.edu/dataset)
- [US Energy Information Administration](https://www.eia.gov)

