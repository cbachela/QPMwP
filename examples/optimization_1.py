############################################################################
### QPMwP CODING EXAMPLES - OPTIMIZATION 1
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     23.12.2024
# First version:    23.12.2024
# --------------------------------------------------------------------------



# pip install pandas
# pip install qpsolvers[open_source_solvers]
# pip install scipy


# Standard library imports
import os

# Third party imports
import numpy as np
import pandas as pd
import qpsolvers
import matplotlib.pyplot as plt






# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------

# Load msci country index return series

path_to_data = '../data/'
# N = 24
N = 10
df = pd.read_csv(os.path.join(path_to_data, 'msci_country_indices.csv'),
                    index_col=0,
                    header=0,
                    parse_dates=True,
                    date_format='%d-%m-%Y')
series_id = df.columns[0:N]
X = df[series_id]








# --------------------------------------------------------------------------
# Estimates of the expected returns and covariance matrix (using sample mean and covariance)
# --------------------------------------------------------------------------

scalefactor = 1  # could be set to 252 (trading days) for annualized returns

# This would be wrong:
# mu = X.mean()

# This is correct:
mu = np.exp(np.log(1 + X).mean(axis=0) * scalefactor) - 1

covmat = X.cov() * scalefactor




# --------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------


# We represent the portfolio domain with the form P = {x | Gx <= h, Ax = b, lb <= x <= ub}


# Lower and upper bounds
lb = np.zeros(covmat.shape[0])
ub = np.repeat(0.2, N)
lb, ub


# Budget constraint
A = np.ones((1, N))
b = np.array(1.0)

A, b


# LInear inequality constraints
G = np.zeros((2, N))
G[0, 0:5] = 1
G[1, 6:10] = 1
h = np.array([0.5, 0.5])

G, h





# --------------------------------------------------------------------------
# Solve for the mean-variance optimal portfolio with fixed risk aversion parameter
# --------------------------------------------------------------------------

# See: https://qpsolvers.github.io/qpsolvers/quadratic-programming.html




# Scale the covariance matrix by the risk aversion parameter
risk_aversion = 1
P = covmat * risk_aversion


# Define problem and solve
problem = qpsolvers.Problem(
    P = P.to_numpy(),
    q = mu.to_numpy(),
    G = G,
    h = h,
    A = A,
    b = b,
    lb = lb,
    ub = ub
)

solution = qpsolvers.solve_problem(
    problem = problem,
    solver = 'cvxopt',
    initvals = None,
    verbose = False,
)


# Inspect the solution object
solution
dir(solution)
solution.obj
solution.found
solution.is_optimal
solution.primal_residual
solution.dual_residual
solution.duality_gap

solution.x



# Extract weights
weights_mv = {col: float(solution.x[i]) for i, col in enumerate(X.columns)}
weights_mv
weights_mv = pd.Series(weights_mv)
weights_mv.plot(kind='bar')






# --------------------------------------------------------------------------
# Solve for the minimum tracking error portfolio, setup as a Least Squares problem
# --------------------------------------------------------------------------

# See: https://qpsolvers.github.io/qpsolvers/least-squares.html




# # Load msci world index return series
# y = pd.read_csv(f'{path_to_data}NDDLWI.csv',
#                 index_col=0,
#                 header=0,
#                 parse_dates=True,
#                 date_format='%d-%m-%Y')

# Create an equally weighted benchmark
y = X.mean(axis=1)
y


# Coefficients of the least squares problem

P = 2 * (X.T @ X)
q = -2 * X.T @ y
constant = y.T @ y

# Define problem and solve
problem = qpsolvers.Problem(
    P = P.to_numpy(),
    q = q.to_numpy(),
    G = G,
    h = h,
    A = A,
    b = b,
    lb = lb,
    ub = ub,
)

solution = qpsolvers.solve_problem(
    problem = problem,
    solver = 'cvxopt',
    initvals = None,
    verbose = False,
)

# Extract weights
weights_ls = pd.Series(solution.x, X.columns)
weights_ls.plot(kind='bar')



# Inspect portfolio simulations

sim_mv = (X @ weights_mv).rename('Mean-Variance Portfolio')
sim_ls = (X @ weights_ls).rename('Min Tracking Error Portfolio (by Least Squares)')

sim = pd.concat({
    'benchmark': y,
    'mean-variance': sim_mv,
    'least-squares': sim_ls,
}, axis=1).dropna()
sim

np.log((1 + sim).cumprod()).plot()




