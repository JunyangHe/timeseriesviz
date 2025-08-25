# timeseriesviz

**timeseriesviz** is a lightweight Python package for **visualizing time series model performance**, with a focus on **spatio-temporal datasets** (e.g., multiple locations, stations, or sensors). It helps researchers and practitioners quickly assess the accuracy of time series forecasts and compare real vs. predicted values across multiple locations.

---

## ✨ Features

- Plot **aggregated performance** across all locations.  
- Generate **detailed subplots** with zoomed-in sections for better error analysis. 
- Plot **error** calculated by (error = actual - forecasted) 
- Support for:
  - **Numpy arrays** (`plot_numpy`)
  - **Pandas DataFrames** from [Nixtla’s NeuralForecast](https://github.com/Nixtla/neuralforecast) (`plot_neuralforecast`) 
- Customizable (`splitsize`) parameter to specify number of detailed plots to generate.
- Option to **save plots to disk**.  

---

## 📦 Installation

```bash
pip install timeseriesviz
```

---

## 🚀 Usage

### Example 1: With Numpy arrays
```python
import numpy as np
from timeseriesviz import plot_numpy

# Simulated data: 100 time steps, 5 locations
y = np.random.rand(100, 5)
pred = y + np.random.normal(0, 0.1, size=y.shape)

fig, axs = plot_numpy(y, pred, title="Forecast vs Actual", splitsize=6)
```

---

### Example 2: With NeuralForecast DataFrame
```python
import pandas as pd
from timeseriesviz import plot_neuralforecast

# Example NeuralForecast output DataFrame
df = pd.DataFrame({
    "unique_id": ["loc1"]*100 + ["loc2"]*100,
    "ds": list(range(100))*2,
    "y": np.random.rand(200),
    "my_model": np.random.rand(200)
})

fig, axs = plot_neuralforecast(df, model_name="my_model", title="NeuralForecast Results", splitsize=6)
```

---

## 📊 Output Example

The generated plots contain:
- **Main Plot**: Entire aggregated time series (real, predicted, and error).  
- **Detailed Plots**: Split into smaller chunks for clearer inspection.  

---

## ⚠️ Requirements

- `pandas`
- `numpy`
- `matplotlib`

---

## 📄 License

MIT License © 2025
