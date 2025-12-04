import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

# ---------------- Synthetic time series ----------------
def generate_series(T=600, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    series = 0.6*np.sin(2*np.pi*t/24) + 0.3*np.sin(2*np.pi*t/7) + 0.01*t + rng.normal(0, 0.1, T)
    return series.astype(np.float32)

def make_supervised(series, window=24, horizon=1):
    X, y = [], []
    for i in range(len(series) - window - horizon + 1):
        X.append(series[i:i+window])
        y.append(series[i+window+horizon-1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def train_test_split(X, y, train_frac=0.6, valid_frac=0.2):
    N = len(X)
    n_train = int(train_frac*N)
    n_valid = int(valid_frac*N)
    X_train, y_train = X[:n_train], y[:n_train]
    X_valid, y_valid = X[n_train:n_train+n_valid], y[n_train:n_train+n_valid]
    X_test, y_test = X[n_train+n_valid:], y[n_train+n_valid:]
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def naive_forecast(X):
    return X[:, -1]

# ---------------- GUI ----------------
class MLPTimeSeriesGUI:
    def __init__(self, root):
        self.root = root
        root.title("MLP Time Series Forecast")

        # Parameters
        tk.Label(root, text="Window Size").grid(row=0, column=0)
        self.window_entry = tk.Entry(root)
        self.window_entry.insert(0, "24")
        self.window_entry.grid(row=0, column=1)

        tk.Label(root, text="Horizon").grid(row=1, column=0)
        self.horizon_entry = tk.Entry(root)
        self.horizon_entry.insert(0, "1")
        self.horizon_entry.grid(row=1, column=1)

        tk.Label(root, text="Hidden Layers").grid(row=2, column=0)
        self.hidden_var = tk.StringVar(value="64,32")
        ttk.Combobox(root, textvariable=self.hidden_var, values=["64", "64,32", "128,64,32"]).grid(row=2, column=1)

        tk.Label(root, text="Activation").grid(row=3, column=0)
        self.activation_var = tk.StringVar(value="relu")
        ttk.Combobox(root, textvariable=self.activation_var, values=["relu","tanh"]).grid(row=3, column=1)

        tk.Button(root, text="Run Forecast", command=self.run_forecast).grid(row=4, column=0, columnspan=2, pady=10)

        self.output = tk.Text(root, height=20, width=70)
        self.output.grid(row=5, column=0, columnspan=2)

    def run_forecast(self):
        try:
            window = int(self.window_entry.get())
            horizon = int(self.horizon_entry.get())
            hidden_layers = tuple(int(h) for h in self.hidden_var.get().split(","))
            activation = self.activation_var.get()
        except ValueError:
            messagebox.showerror("Error", "Invalid input values")
            return

        series = generate_series(T=600)
        X, y = make_supervised(series, window=window, horizon=horizon)
        X_train, y_train, X_valid, y_valid, X_test, y_test = train_test_split(X, y)

        # Naive baseline
        y_pred_naive = naive_forecast(X_test)
        mae_naive = mean_absolute_error(y_test, y_pred_naive)
        rmse_naive = sqrt(mean_squared_error(y_test, y_pred_naive))

        # MLP pipeline
        mlp = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=hidden_layers,
                         activation=activation,
                         solver='adam',
                         learning_rate_init=1e-3,
                         max_iter=400,
                         random_state=42,
                         early_stopping=True,
                         n_iter_no_change=20,
                         validation_fraction=0.1)
        )
        X_tr_all = np.concatenate([X_train, X_valid], axis=0)
        y_tr_all = np.concatenate([y_train, y_valid], axis=0)
        mlp.fit(X_tr_all, y_tr_all)

        y_pred = mlp.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = sqrt(mean_squared_error(y_test, y_pred))

        # Display results
        self.output.delete(1.0, tk.END)
        self.output.insert(tk.END, f"Naive baseline: MAE={mae_naive:.4f}, RMSE={rmse_naive:.4f}\n")
        self.output.insert(tk.END, f"MLP Forecast:   MAE={mae:.4f}, RMSE={rmse:.4f}\n\n")
        self.output.insert(tk.END, "Sample predictions (first 5):\n")
        for i in range(5):
            self.output.insert(tk.END, f"y_true={y_test[i]:.3f} | naive={y_pred_naive[i]:.3f} | mlp={y_pred[i]:.3f}\n")

        # Plot predictions
        plt.figure(figsize=(10,4))
        plt.plot(y_test, label="True", marker='o')
        plt.plot(y_pred_naive, label="Naive", marker='x')
        plt.plot(y_pred, label="MLP", marker='s')
        plt.title("Forecast vs True Values")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()

# ---------------- Run GUI ----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = MLPTimeSeriesGUI(root)
    root.mainloop()
