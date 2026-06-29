# Black-Scholes PDE Solver

## MSc Financial Mathematics – Computational Finance Project (University of Leeds)

This repository contains a Python implementation and accompanying report for solving the Black–Scholes partial differential equation using an implicit finite difference method.

The project develops a numerical PDE solver for pricing bounded power put options, validates the implementation against the analytical Black–Scholes solution, and analyses option sensitivities.

---

## Project Overview

The implementation transforms the Black–Scholes PDE into the heat equation before solving it using an implicit finite difference scheme.

The project includes:
- Construction of an efficient tridiagonal linear system solver
- Derivation and implementation of boundary conditions for bounded power put options
- Numerical pricing using an implicit finite difference method
- Linear interpolation to estimate option values
- Numerical estimation of the Greek Theta
- Validation against the analytical Black–Scholes solution

---

## Repository Contents

- **black_scholes_pde_solver.py** – Python implementation
- **Black-Scholes PDE Solver Report.pdf** – Technical report describing the methodology, implementation and numerical results

---

## Technologies

- Python
- NumPy
- SciPy
- Matplotlib
- Pandas

---

## Key Results

- Implemented an implicit finite difference solver for the Black–Scholes PDE.
- Successfully replicated analytical Black–Scholes prices for vanilla European put options.
- Extended the framework to price bounded power put options with different payoff structures.
- Estimated the Greek Theta using finite difference methods and compared numerical results with analytical solutions.

---

*Academic coursework completed as part of the MSc Financial Mathematics programme at the University of Leeds.*
