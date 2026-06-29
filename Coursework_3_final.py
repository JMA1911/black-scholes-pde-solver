import numpy as np
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D 
import pandas as pd

# Part 1 - Linear System Solver
def tridiagonal_solve(lambda_, b):
    n = len(b)
    
    # Construct the diagonals
    upper = -lambda_ * np.ones(n - 1)
    main = (1 + 2 * lambda_) * np.ones(n)
    lower = -lambda_ * np.ones(n - 1)
    
    # Assemble the banded matrix
    ab = np.zeros((3, n))
    ab[0, 1:] = upper      # upper diagonal (1 above main)
    ab[1, :] = main        # main diagonal
    ab[2, :-1] = lower     # lower diagonal (1 below main)
    
    # Solve the system
    x = solve_banded((1, 1), ab, b)
    return x

# Example with known solution
lambda_ = 1
n = 4
A = np.array([
    [3, -1,  0,  0],
    [-1, 3, -1,  0],
    [0, -1, 3, -1],
    [0,  0, -1, 3]
])
x_true = np.array([1, 2, 3, 4])
b = A @ x_true

# Solve using our function
x_computed = tridiagonal_solve(lambda_, b)
print("Computed solution:", x_computed)
print("True solution:    ", x_true)

# Part 2 - Bounde Power Put Option Boundary Conditions
# Theoretical - only for the report 

# Part 3 - Numerical Solution of the Black-Scholes PDE - Implicit Method
def bounded_power_implicit(r, sigma, K, L, p, T, M, N, xmin, xmax):
    T = T / 12
    delta_tau = (sigma**2 * T) / (2 * N)
    dx = (xmax - xmin) / M
    lambda_ = delta_tau / dx**2

    x = np.linspace(xmin, xmax, M + 1)       # spatial grid in x
    tau = np.linspace(0, delta_tau * N, N + 1)
    t = T - (2 / sigma**2) * tau                     # inverse time grid (for plotting)

    # NEW: Create 2D stock price grid according to inverse transformation
    X, TAU = np.meshgrid(x, tau, indexing='ij')   # shape (M+1, N+1)
    S = np.exp(X - ((2 * r / sigma**2) - 1) * TAU) # shape (M+1, N+1)

    # Step 2: Initial condition at τ = 0
    w = np.zeros((M + 1, N + 1))
    # Use the stock prices at τ = 0 (which corresponds to t = T)
    S_initial = S[:, 0]
    w[:, 0] = np.exp(-r * T) * np.minimum(L, np.maximum(K - S_initial, 0)**p)

    # Step 3: Time stepping
    for n in range(0, N):
        w0_next = min(L, K**p) * np.exp(-r * (T - tau[n + 1]))
        wM_next = 0

        b = w[1:-1, n].copy()
        b[0] += lambda_ * w0_next
        b[-1] += lambda_ * wM_next

        w_inner_next = tridiagonal_solve(lambda_, b)

        w[0, n + 1] = w0_next
        w[1:-1, n + 1] = w_inner_next
        w[M, n + 1] = wM_next

    # Now return both option grid and full stock price matrix
    return w, S, t

option_values_3, S_3, t_3 = bounded_power_implicit(r=0.08, sigma=0.1, K=10, L=11, p=1, T=18, M=100, N=100, xmin=np.log(0.01), xmax=np.log(100))

# Black-Scholes formula for a European put option
def black_scholes_put_price(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Plot the option value at t=0 as a function of S, comparing numerical PDE solution vs Black-Scholes formula.
def plot_option_profile_vs_true(option_grid, S, t, K, r, sigma, filename):

    V_numerical = option_grid[:, -1]  # last column = t = 0
    S_final = S[:, -1]                # stock prices at t = 0

    # Restrict to S <= 20 for clearer plotting
    mask = S_final <= 20
    S_plot = S_final[mask]
    V_plot = V_numerical[mask]

    # Compute exact Black-Scholes put prices
    T0 = t[0]
    V_true = black_scholes_put_price(S_plot, K, r, sigma, T0)

    plt.figure(figsize=(10, 6))
    plt.plot(S_plot, V_plot, label="Heat Equation (Implicit Method)", color="#1f77b4", lw=2)
    plt.plot(S_plot, V_true, label="Black-Scholes Formula", color="#d62728", lw=2, linestyle="--")
    plt.xlabel("Stock Price $S$")
    plt.ylabel("Option Value")
    plt.title("Vanilla European Put Option at $t=0$")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 20)
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()

plot_option_profile_vs_true(option_values_3, S_3, t_3, K=10, r=0.08, sigma=0.1, filename="vanilla_put_plot.pdf")

# Part 4 - Analysis of the Case p=2
# Obtaining the values for the case where p=2 and L=81
option_values_4, S_4, t_4 = bounded_power_implicit(r=0.08, sigma=0.1, K=10, L=81, p=2, T=18, M=100, N=100, xmin=np.log(0.01), xmax=np.log(100))

# Function to make a surface plot for option values, stock prices and time
def plot_option_surface(option_values, S, t, p, L, filename):
    V_surface = option_values

    S_mesh = S
    t_mesh = np.tile(t, (S.shape[0], 1))

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(S_mesh, t_mesh, V_surface, cmap="viridis", edgecolor="none")

    ax.set_xlabel("Stock Price $S$")
    ax.set_ylabel("Time $t$")
    ax.set_zlabel("Option Value $V(S, t)$")
    ax.set_title(f"Surface Plot of Bounded Power Put Option when $p={p}$ and $L={L}$")

    fig.colorbar(surf, shrink=0.5, aspect=10)

    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()

plot_option_surface(option_values_4, S_4, t_4, 2, 81, filename="surface_p2.pdf")

# Plotting the current value of the option compared to the pay-off
def plot_option_vs_payoff(option_values, S, K, L, p, T, r, filename):
    S_final = S[:, -1]
    option_at_t0 = option_values[:, -1]

    payoff = np.minimum(L, np.maximum(K - S_final, 0)**p)
    discounted_payoff = np.exp(-r * T) * payoff

    plt.figure(figsize=(8, 5))
    plt.plot(S_final, option_at_t0, label="Numerical Option Value at $t = 0$", color="#1f77b4", linewidth=2)
    plt.plot(S_final, discounted_payoff, "--", label="Discounted Payoff Function", color="#d62728", linewidth=2)

    plt.xlabel("Stock Price $S$")
    plt.ylabel("Option Value")
    plt.title(f"Option Value vs Payoff ($p = {p}$, $L = {L}$)")

    # Clean legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()

plot_option_vs_payoff(option_values_4, S_4, K=10, L=81, p=2, T=1.5, r=0.08, filename="value_vs_payoff_p2.pdf")

# Estimating the value of the option given the current stock price 4.73EUR using interpolation
def estimate_option_value(S, option_values, S_query, filename):
    option_at_t0 = option_values[:, -1]  # V(S, t=0)
    S_final = S[:, -1]                   # S at t=0

    value = np.interp(S_query, S_final, option_at_t0)

    labels = [r"$S$ at $t=0$ (\EUR)", r"$V(S, 0)$ (\EUR)"]
    values = [f"{S_query:.2f}", f"{value:.4f}"]
    df = pd.DataFrame([values], columns=labels)
    df.to_latex(filename, index=False, escape=False, column_format="c" * len(labels))
    print(df)
    return value

estimate_option_value(S_4, option_values_4, 4.73, filename="interpolated_value_p2.tex")

# Estimating Theta using forward differencing and plot stock price against theta
def estimate_theta(S_grid, option_values, t, p, L, filename):

    # Time slices
    V_t0 = option_values[:, -1]       # Option value at t = 0
    V_t1_raw = option_values[:, -2]   # Option value at previous time step

    # Corresponding stock price slices
    S_t0 = S_grid[:, -1]              # S at t = 0
    S_t1 = S_grid[:, -2]              # S at t = dt

    # Interpolate V_t1 onto the S_t0 grid
    V_t1_interp = np.interp(S_t0, S_t1, V_t1_raw)

    # Correct Theta: account for ∂V/∂t = -∂V/∂τ
    dt = t[-1] - t[-2]
    theta_values = -(V_t1_interp - V_t0) / dt

    # Identify max Theta for annotation
    max_index = np.argmax(theta_values)
    S_max_theta = S_t0[max_index]
    max_theta = theta_values[max_index]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(S_t0, theta_values, label=r"Estimated $\Theta(S)$", color="#1f77b4", linewidth=2)
    plt.axvline(S_max_theta, linestyle=":", color="#d62728", linewidth=1.5)
    plt.plot(S_max_theta, max_theta, "ro", label=fr"$\max \Theta = {max_theta:.4f}$ at $S = {S_max_theta:.2f}$")

    plt.xlabel("Stock Price $S$")
    plt.ylabel(r"Greek $\Theta = \frac{\partial V}{\partial t}$")
    plt.title(f"Theta vs Stock Price ($p = {p}$, $L = {L}$)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()

    return theta_values

estimate_theta(S_4, option_values_4, t_4, p=2, L=81, filename="theta_plot_p2.pdf")

# Part 5 - Analysis of the Case p=0.5
# Obtaining the values for the case where p=0.5 and L=3
option_values_5, S_5, t_5 = bounded_power_implicit(r=0.08, sigma=0.1, K=10, L=3, p=0.5, T=18, M=100, N=100, xmin=np.log(0.01), xmax=np.log(100))

# Making a surface plot for option values, stock prices and time
plot_option_surface(option_values_5, S_5, t_5, 0.5, 3, filename="surface_p5.pdf")

# Plotting the current value of the option compared to the pay-off
plot_option_vs_payoff(option_values_5, S_5, K=10, L=3, p=0.5, T=1.5, r=0.08, filename="value_vs_payoff_p5.pdf")

# Estimating the value of the option given the current stock price 4.73EUR using interpolation
estimate_option_value(S_5, option_values_5, 4.73, filename="interpolated_value_p5.tex")

# Estimating Theta using forward differencing and plot stock price against theta
estimate_theta(S_5, option_values_5, t_5, p=0.5, L=3, filename="theta_plot_p5.pdf")

# Additional bonus plot for checking how realistic theta values are based on comparison to analytical solution for vanilla European put option
def analytical_theta(S, K, r, sigma, T):

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    theta = r * K * np.exp(-r * T) * norm.cdf(-d2) - (sigma * S * norm.pdf(d1)) / (2 * np.sqrt(T))
    return theta

def compare_theta_given_values(S_values, theta_numerical, K, r, sigma, T, filename):
    # Analytical theta
    theta_analytical = analytical_theta(S_values, K, r, sigma, T)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(S_values, theta_numerical, label=r"Numerical $\Theta$", color="#1f77b4", linewidth=2)
    plt.plot(S_values, theta_analytical, "--", label=r"Analytical $\Theta$", color="#d62728", linewidth=2)

    plt.xlabel("Stock Price $S$")
    plt.ylabel(r"Greek $\Theta$")
    plt.title("Comparison of Theta: Numerical vs Analytical")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()

theta_numerical = estimate_theta(S_3, option_values_3, t_3, p=1, L=11, filename="theta_correct_p1.pdf")
S_t0 = S_3[:, -1]
T0 = t_3[0]

compare_theta_given_values(S_t0, theta_numerical, K=10, r=0.08, sigma=0.1, T=T0, filename="theta_comparison_reuse.pdf")
