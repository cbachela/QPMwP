############################################################################
### QPMwP CODING EXAMPLES - OPTIMIZATION 2
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
import sys

# Third party imports
import numpy as np
import pandas as pd
import qpsolvers
import scipy

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.append(project_root)
sys.path.append(src_path)

# Local modules imports
from helper_functions import load_data_msci






# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------

N = 24
data = load_data_msci(path = '../data/', n = N)
data





# --------------------------------------------------------------------------
# Estimates of the expected returns and covariance matrix (using sample mean and covariance)
# --------------------------------------------------------------------------

X = data['return_series']
scalefactor = 1  # could be set to 252 (trading days) for annualized returns

# This would be wrong:
# mu = X.mean()

# This is correct:
mu = np.exp(np.log(1 + X).mean(axis=0) * scalefactor) - 1

covmat = X.cov() * scalefactor




# --------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------

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
# Solve for the mean-variance portfolio with fixed risk aversion parameter
# --------------------------------------------------------------------------

# Scale the covariance matrix by the risk aversion parameter
risk_aversion = 1
P = covmat * risk_aversion


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

solution
solution.x




# qp = QuadraticProgram(
#     solver_name = 'cvxopt',
#     P = covmat,
#     q = np.zeros(covmat.shape[0])
# )










ALL_SOLVERS = {'clarabel', 'cvxopt', 'daqp', 'ecos', 'gurobi', 'highs', 'mosek', 'osqp', 'piqp', 'proxqp', 'qpalm', 'quadprog', 'scs'}
SPARSE_SOLVERS = {'clarabel', 'ecos', 'gurobi', 'mosek', 'highs', 'qpalm', 'osqp', 'qpswift', 'scs'}
IGNORED_SOLVERS = {
    'gurobi',  # Commercial solver
    'mosek',  # Commercial solver
    'ecos',
    'scs',
    'piqp',
    'proxqp',
    'clarabel'
}
USABLE_SOLVERS = ALL_SOLVERS - IGNORED_SOLVERS



class QuadraticProgram(dict):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.solver = self['params']['solver_name']

    def solve(self) -> None:

        if self.solver in ['ecos', 'scs', 'clarabel']:
            if self.get('b').size == 1:
                self['b'] = np.array(self.get('b')).reshape(-1)

        P = self.get('P')
        # if P is not None and not isPD(P):
        #     self['P'] = nearestPD(P)
        problem = qpsolvers.Problem(
            P=self.get('P'),
            q=self.get('q'),
            G=self.get('G'),
            h=self.get('h'),
            A=self.get('A'),
            b=self.get('b'),
            lb=self.get('lb'),
            ub=self.get('ub')
        )

        # Convert to sparse matrices for best performance
        if self.solver in SPARSE_SOLVERS:
            if self['params'].get('sparse'):
                if problem.P is not None:
                    problem.P = scipy.sparse.csc_matrix(problem.P)
                if problem.A is not None:
                    problem.A = scipy.sparse.csc_matrix(problem.A)
                if problem.G is not None:
                    problem.G = scipy.sparse.csc_matrix(problem.G)

        solution = qpsolvers.solve_problem(
            problem=problem,
            solver=self.solver,
            initvals=self.get('x0'),
            verbose=False
        )
        self['solution'] = solution
        return None



qp = QuadraticProgram(P=covmat, q=np.zeros(covmat.shape[0]), solver_name='cvxopt')









