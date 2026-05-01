# Cell Fate Engine (CFE) v2.1
### Real-time Stochastic Autopilot for Proteomic Flux Simulation

The **Cell Fate Engine (CFE)** is a biophysical simulation tool designed to model and visualize protein concentration dynamics. It uses Ordinary Differential Equations (ODEs) to simulate how cells maintain homeostasis or drift into pathological states like oncogenic overexpression.



## 🧬 Biophysical Model
The simulation is driven by the Hill Equation, which accounts for cooperative binding kinetics:

$$\frac{dX}{dt} = \left( O_2 \cdot \frac{S^n}{K^n + S^n} \right) - (d \cdot X)$$

- **S**: Signal strength
- **n**: Hill coefficient (cooperativity)
- **K**: Threshold constant (0.5)
- **d**: Degradation rate

## ✨ Key Features
- **Real-time Visualization:** Live plotting of proteomic flux using Matplotlib and CustomTkinter.
- **Scenario Director:** Switch instantly between biological modes:
  - **Homeostasis:** Healthy balanced state.
  - **Oncogenic Drift:** Simulated protein overexpression.
  - **Treatment Modes:** Simulation of PROTACs and traditional Inhibitors.
- **Stochasticity:** Integrated Gaussian noise to represent natural biological variability.
- **Clean Architecture:** Refactored with a strict separation between the biophysics engine and the UI.

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.9+ installed along with the following libraries:
```bash
pip install customtkinter numpy scipy matplotlib
