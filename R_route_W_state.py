import numpy as np
import numpy.random as rnd
import pandas as pd
import random
import math
from math import prod
import cvxpy as cp
import sympy as sp
from scipy.optimize import fsolve
import scipy.linalg as la
from scipy.optimize import minimize


## 1. Generate Random Parameters

# Randomly generate variable values that meet the conditions
def generate_variables(R):
    random.seed(15) # Set random seed to ensure experiment reproducibility

    # Randomly select R numbers from 1 to 5 and ensure they are in ascending order
    values = sorted(random.sample([round(x * 0.1, 1) for x in range(10, 51)], R))

    # Use a dictionary to store variable names and corresponding values
    variables = {f'm{i+1}': values[i] for i in range(R)}

    return variables

# Randomly generate W weights that meet the conditions
def generate_weights(W):
    random.seed(0)  # Set random seed to ensure experiment reproducibility
    weights = [random.random() for _ in range(W)]
    total = sum(weights)
    normalized_weights = [w / total for w in weights]

    # Use a dictionary to store variable names and corresponding values
    weight_variables = {f'w{i+1}': round(normalized_weights[i], 2) for i in range(W)}

    return weight_variables

# Randomly generate alpha values ranging from 0 to 1
def generate_alphas(R, W):
    random.seed(0)  # Set random seed to ensure experiment reproducibility
    alpha_variables = {f'alpha{i+1}{j+1}': round(random.random(), 2) for i in range(R) for j in range(W)}

    return alpha_variables

betamax = 100
W = 3  # 3 states
R = 3  # 3 routes

variables = generate_variables(R)

weight_variables = generate_weights(W)

alpha_variables = generate_alphas(R, W)

# Dynamically define tau variables and assign values (equal to the corresponding alpha values)
tau_variables = {f'tau{i+1}{j+1}': alpha_variables[f'alpha{i+1}{j+1}'] for i in range(R) for j in range(W)}

# Dynamically define variables and assign values
for key, value in variables.items():
    globals()[key] = value

# Dynamically define variables and assign values
for key, value in weight_variables.items():
    globals()[key] = value

# Dynamically define alpha variables and assign values
for key, value in alpha_variables.items():
    globals()[key] = value

# Dynamically define tau variables and assign values
for key, value in tau_variables.items():
    globals()[key] = value

# Print directly defined variables
for key in variables:
    print(f"{key} = {globals()[key]}")

# Print directly defined variables
for key in weight_variables:
    print(f"{key} = {globals()[key]}")

# Print directly defined alpha variables
for key in alpha_variables:
    print(f"{key} = {globals()[key]}")

# Print directly defined tau variables
for key in tau_variables:
    print(f"{key} = {globals()[key]}")


## 2. Calculate First Best (FB) and Complete Information (CI) Equilibrium 

# Function to calculate l values
def calculate_l_values(alpha, tau):
    R = len(alpha)
    l_values = []
    for r in range(R):
        numerator_term1 = 2 * prod([alpha[i] for i in range(R) if i != r])
        numerator_term2 = sum([tau[j] * prod([alpha[i] for i in range(R) if i != r and i != j]) for j in range(R) if j != r])
        numerator_term3 = sum([tau[r] * prod([alpha[i] for i in range(R) if i != r and i != j]) for j in range(R) if j != r])
        numerator = numerator_term1 + numerator_term2 - numerator_term3
        denominator = 2 * sum([prod([alpha[i] for i in range(R) if i != j]) for j in range(R)])
        l_r = numerator / denominator
        l_values.append(l_r)
    return l_values

# Function to calculate C_r values
def calculate_C_values(alpha, tau):
    R = len(alpha)
    C_values = []
    for r in range(R):
        term1 = 2 * prod([alpha[i] for i in range(R) if i != r])
        term2 = sum([tau[j] * prod([alpha[i] for i in range(R) if i != r and i != j]) for j in range(R) if j != r])
        term3 = sum([tau[r] * prod([alpha[i] for i in range(R) if i != r and i != j]) for j in range(R) if j != r])
        C_r = term1 + term2 - term3
        C_values.append(C_r)
    return C_values

# Function to calculate l values based on conditions
def calculate_fb_equilibrium(alpha, tau, m, betamax):
    R = len(alpha)
    C_values = calculate_C_values(alpha, tau)
    valid_routes = [r for r in range(R) if C_values[r] >= 0]

    if len(valid_routes) == R:
        # Condition from label {22}
        l_values = calculate_l_values(alpha, tau)
    else:
        # Condition from label {23} and label {24}
        R_prime = len(valid_routes)
        alpha_prime = [alpha[r] for r in valid_routes]
        tau_prime = [tau[r] for r in valid_routes]
        l_values = calculate_l_values(alpha_prime, tau_prime)
        # Expand l_values to match original R by setting excluded routes to 0
        full_l_values = [0] * R
        for idx, r in enumerate(valid_routes):
            full_l_values[r] = l_values[idx]
        l_values = full_l_values

    return l_values

# Given values for alpha, tau, and m - state 1
alpha = [alpha11, alpha21, alpha31]
tau = [tau11, tau21, tau31]
m = [m1, m2, m3]
# Calculate l values for FB equilibrium
l_values = calculate_fb_equilibrium(alpha, tau, m, betamax)
[l11fb, l21fb, l31fb] = l_values
print('FB load in state 1: ', l_values)

# Given values for alpha, tau, and m - state 2
alpha = [alpha12, alpha22, alpha32]
tau = [tau12, tau22, tau32]
m = [m1, m2, m3]
# Calculate l values for FB equilibrium
l_values = calculate_fb_equilibrium(alpha, tau, m, betamax)
[l12fb, l22fb, l32fb] = l_values
print('FB load in state 2: ', l_values)

# Given values for alpha, tau, and m - state 3
alpha = [alpha13, alpha23, alpha33]
tau = [tau13, tau23, tau33]
m = [m1, m2, m3]
# Calculate l values for FB equilibrium
l_values = calculate_fb_equilibrium(alpha, tau, m, betamax)
[l13fb, l23fb, l33fb] = l_values
print('FB load in state 3: ', l_values)

# General function definition for solving complete information equilibrium
def solve_complete_information_equilibrium(R, W, betamax):
    results = {}
    for w in range(1, W + 1):
        l_values = [0] * R
        remaining_routes = list(range(R))
        while remaining_routes:
            def func(x):
                equations = []
                cumulative_l = 0
                for r_idx, r in enumerate(remaining_routes[:-1]):
                    l = x[r_idx]
                    cumulative_l += l
                    l_next = 1 - cumulative_l if r_idx == len(remaining_routes) - 2 else x[r_idx + 1]
                    beta = betamax * cumulative_l
                    equation = (
                        beta * (globals()[f'alpha{r+1}{w}'] * l + globals()[f'tau{r+1}{w}']) + globals()[f'm{r+1}']
                        - beta * (globals()[f'alpha{remaining_routes[r_idx + 1] + 1}{w}'] * l_next + globals()[f'tau{remaining_routes[r_idx + 1] + 1}{w}']) - globals()[f'm{remaining_routes[r_idx + 1] + 1}']
                    )
                    equations.append(equation)
                return np.array(equations)

            initial_guess = [1 / len(remaining_routes)] * (len(remaining_routes) - 1)
            solution = fsolve(func, initial_guess)
            solution = np.append(solution, 1 - sum(solution))

            if all(l >= 0 for l in solution):
                for r_idx, r in enumerate(remaining_routes):
                    l_values[r] = solution[r_idx]
                break
            else:
                # Find the first negative flow and remove the corresponding route
                for r_idx, l in enumerate(solution):
                    if l < 0:
                        remaining_routes.pop(r_idx)
                        break

        results[f'l{w}ci'] = l_values
    return results

# Calculate complete information equilibrium
ci_results = solve_complete_information_equilibrium(R, W, betamax)

# Print equilibrium results
for w in range(1, W + 1):
    print(f"l{w}ci: {ci_results[f'l{w}ci']}")


## 3. Information design section

# Define objective function and constraints
def objective(l):
    l = l.reshape((R, W))
    return np.sum([weight_variables[f'w{j+1}'] * l[k, j] * (alpha_variables[f'alpha{k+1}{j+1}'] * l[k, j] + tau_variables[f'tau{k+1}{j+1}'])
                   for j in range(W) for k in range(R)])

# Constraint function (default constraints are greater than or equal to 0)
def constraints():
    cons = []

    # Normalization constraint: for each j, the sum of all k flows equals 1
    for j in range(W):
        cons.append({
            'type': 'eq',
            'fun': lambda l, j=j: np.sum(l.reshape((R, W))[:, j]) - 1
        })

    # Given constraints
    cons.append({
        'type': 'ineq',
        'fun': lambda l: -weight_variables['w1'] * (betamax*l[0] * (alpha_variables['alpha11'] * l[0] + tau_variables['tau11']) + variables['m1']) -
                         weight_variables['w3'] * (betamax*l[0] * (alpha_variables['alpha13'] * l[2] + tau_variables['tau13']) + variables['m1']) +
                         (weight_variables['w1'] * (betamax*l[0] * (alpha_variables['alpha21'] * l[3] + tau_variables['tau21']) + variables['m2']) +
                          weight_variables['w3'] * (betamax*l[0] * (alpha_variables['alpha23'] * l[5] + tau_variables['tau23']) + variables['m2']))
    })

    cons.append({
        'type': 'ineq',
        'fun': lambda l: -weight_variables['w2'] * (betamax*l[1] * (alpha_variables['alpha12'] * l[1] + tau_variables['tau12']) + variables['m1']) -
                         weight_variables['w3'] * (betamax*l[1] * (alpha_variables['alpha13'] * l[2] + tau_variables['tau13']) + variables['m1']) +
                         (weight_variables['w2'] * (betamax*l[1] * (alpha_variables['alpha22'] * l[4] + tau_variables['tau22']) + variables['m2']) +
                          weight_variables['w3'] * (betamax*l[1] * (alpha_variables['alpha23'] * l[5] + tau_variables['tau23']) + variables['m2']))
    })

    cons.append({
        'type': 'ineq',
        'fun': lambda l: weight_variables['w2'] * (betamax*l[1] * (alpha_variables['alpha12'] * l[1] + tau_variables['tau12']) + variables['m1']) +
                         weight_variables['w1'] * (betamax*l[1] * (alpha_variables['alpha11'] * l[0] + tau_variables['tau11']) + variables['m1']) -
                         (weight_variables['w2'] * (betamax*l[1] * (alpha_variables['alpha22'] * l[4] + tau_variables['tau22']) + variables['m2']) +
                          weight_variables['w1'] * (betamax*l[1] * (alpha_variables['alpha21'] * l[3] + tau_variables['tau21']) + variables['m2']))
    })

    cons.append({
        'type': 'ineq',
        'fun': lambda l: weight_variables['w1'] * (betamax*l[2] * (alpha_variables['alpha11'] * l[0] + tau_variables['tau11']) + variables['m1']) +
                         weight_variables['w3'] * (betamax*l[2] * (alpha_variables['alpha13'] * l[2] + tau_variables['tau13']) + variables['m1']) -
                         (weight_variables['w1'] * (betamax*l[2] * (alpha_variables['alpha21'] * l[3] + tau_variables['tau21']) + variables['m2']) +
                          weight_variables['w3'] * (betamax*l[2] * (alpha_variables['alpha23'] * l[5] + tau_variables['tau23']) + variables['m2']))
    })

    cons.append({
        'type': 'ineq',
        'fun': lambda l: -weight_variables['w1'] * (betamax*(l[1]+l[4]) * (alpha_variables['alpha21'] * l[3] + tau_variables['tau21']) + variables['m2']) -
                         weight_variables['w2'] * (betamax*(l[1]+l[4]) * (alpha_variables['alpha22'] * l[4] + tau_variables['tau22']) + variables['m2']) +
                         (weight_variables['w1'] * (betamax*(l[1]+l[4]) * (alpha_variables['alpha31'] * l[6] + tau_variables['tau31']) + variables['m3']) +
                          weight_variables['w2'] * (betamax*(l[1]+l[4]) * (alpha_variables['alpha32'] * l[7] + tau_variables['tau32']) + variables['m3']))
    })

    cons.append({
        'type': 'ineq',
        'fun': lambda l: weight_variables['w3'] * (betamax*(l[2]+l[5]) * (alpha_variables['alpha23'] * l[5] + tau_variables['tau23']) + variables['m2']) +
                         weight_variables['w2'] * (betamax*(l[2]+l[5]) * (alpha_variables['alpha22'] * l[4] + tau_variables['tau22']) + variables['m2']) -
                         (weight_variables['w3'] * (betamax*(l[2]+l[5]) * (alpha_variables['alpha33'] * l[8] + tau_variables['tau33']) + variables['m3']) +
                          weight_variables['w2'] * (betamax*(l[2]+l[5]) * (alpha_variables['alpha32'] * l[7] + tau_variables['tau32']) + variables['m3']))
    })

    cons.append({
        'type': 'ineq',
        'fun': lambda l: -weight_variables['w1'] * (betamax*(l[2]+l[5]) * (alpha_variables['alpha21'] * l[3] + tau_variables['tau21']) + variables['m2']) -
                         weight_variables['w3'] * (betamax*(l[2]+l[5]) * (alpha_variables['alpha23'] * l[5] + tau_variables['tau23']) + variables['m2']) +
                         (weight_variables['w1'] * (betamax*(l[2]+l[5]) * (alpha_variables['alpha31'] * l[6] + tau_variables['tau31']) + variables['m3']) +
                          weight_variables['w3'] * (betamax*(l[2]+l[5]) * (alpha_variables['alpha33'] * l[8] + tau_variables['tau33']) + variables['m3']))
    })

    cons.append({
        'type': 'ineq',
        'fun': lambda l: -weight_variables['w1'] * (betamax*l[0] * (alpha_variables['alpha11'] * l[0] + tau_variables['tau11']) + variables['m1']) -
                         weight_variables['w2'] * (betamax*l[0] * (alpha_variables['alpha12'] * l[1] + tau_variables['tau12']) + variables['m1']) -
                         weight_variables['w3'] * (betamax*l[0] * (alpha_variables['alpha13'] * l[2] + tau_variables['tau13']) + variables['m1']) +
                         weight_variables['w1'] * (betamax*l[0] * (alpha_variables['alpha21'] * l[3] + tau_variables['tau21']) + variables['m2']) +
                         weight_variables['w2'] * (betamax*l[0] * (alpha_variables['alpha22'] * l[4] + tau_variables['tau22']) + variables['m2']) +
                         weight_variables['w3'] * (betamax*l[0] * (alpha_variables['alpha23'] * l[5] + tau_variables['tau23']) + variables['m2'])
    })

    cons.append({
        'type': 'ineq',
        'fun': lambda l: weight_variables['w1'] * (betamax*l[2] * (alpha_variables['alpha11'] * l[0] + tau_variables['tau11']) + variables['m1']) +
                         weight_variables['w2'] * (betamax*l[2] * (alpha_variables['alpha12'] * l[1] + tau_variables['tau12']) + variables['m1']) +
                         weight_variables['w3'] * (betamax*l[2] * (alpha_variables['alpha13'] * l[2] + tau_variables['tau13']) + variables['m1']) -
                         (weight_variables['w1'] * (betamax*l[2] * (alpha_variables['alpha21'] * l[3] + tau_variables['tau21']) + variables['m2']) +
                          weight_variables['w2'] * (betamax*l[2] * (alpha_variables['alpha22'] * l[4] + tau_variables['tau22']) + variables['m2']) +
                          weight_variables['w3'] * (betamax*l[2] * (alpha_variables['alpha23'] * l[5] + tau_variables['tau23']) + variables['m2']))
    })

    cons.append({
        'type': 'ineq',
        'fun': lambda l: -weight_variables['w1'] * (betamax*(l[1]+l[4]) * (alpha_variables['alpha21'] * l[3] + tau_variables['tau21']) + variables['m2']) -
                         weight_variables['w2'] * (betamax*(l[1]+l[4]) * (alpha_variables['alpha22'] * l[4] + tau_variables['tau22']) + variables['m2']) -
                         weight_variables['w3'] * (betamax*(l[1]+l[4]) * (alpha_variables['alpha23'] * l[5] + tau_variables['tau23']) + variables['m2']) +
                         (weight_variables['w1'] * (betamax*(l[1]+l[4]) * (alpha_variables['alpha31'] * l[6] + tau_variables['tau31']) + variables['m3']) +
                          weight_variables['w2'] * (betamax*(l[1]+l[4]) * (alpha_variables['alpha32'] * l[7] + tau_variables['tau32']) + variables['m3']) +
                          weight_variables['w3'] * (betamax*(l[1]+l[4]) * (alpha_variables['alpha33'] * l[8] + tau_variables['tau33']) + variables['m3']))
    })

    return cons

# Initial guess values
initial_guess = np.ones((R, W)).flatten() / W

# Set variable bounds (l11 ≥ l11ci, l13 ≤ l13ci, l21 ≤ l21ci, l22 ≥ l22ci, l31 ≥ l31ci, l32 ≤ l32ci)
bounds = [(ci_results['l1ci'][0], 1), (0, 1), (0, ci_results['l3ci'][0]), (0, ci_results['l1ci'][1]),
          (ci_results['l2ci'][1], 1), (0, 1), (ci_results['l1ci'][2], 1), (0, ci_results['l2ci'][2]), (0, 1)]

# Use scipy.optimize.minimize for solving
result = minimize(objective, initial_guess, method='SLSQP', constraints=constraints(), bounds=bounds)

# Print results
if result.success:
    print("Global minimization of the objective function result: ", result.fun)
    l_optimal = result.x.reshape((R, W))
    for k in range(R):
        for j in range(W):
            print(f"l_{k+1}^{j+1}: {l_optimal[k, j]}")
else:
    print("Optimization unsuccessful: ", result.message)

# comparision: ci
print('Complete information equilibrium objective value: ', objective(np.array([ci_results['l1ci'][0], ci_results['l2ci'][0], ci_results['l3ci'][0],
        ci_results['l1ci'][1], ci_results['l2ci'][1], ci_results['l3ci'][1],
        ci_results['l1ci'][2], ci_results['l2ci'][2], ci_results['l3ci'][2]])))

# comparison: FB
print('FB objective value: ', objective(np.array([0, 0, 0.3547008547008547, 1, 0.24074074074074067, 0.39743589743589736, 0, 0.7592592592592592, 0.24786324786324795])))


## 4. Semidefinite Relaxation for Finding SDP Optimal Solution

# Simplify original inequality constraints into a matrix
def simplify_constraint_to_matrix_form(weights, alpha, tau, m, betamax):

    l1_1, l1_2, l1_3, l2_1, l2_2, l2_3, l3_1, l3_2, l3_3 = sp.symbols('l1_1 l1_2 l1_3 l2_1 l2_2 l2_3 l3_1 l3_2 l3_3')

    l = [l1_1, l1_2, l1_3, l2_1, l2_2, l2_3, l3_1, l3_2, l3_3]

    simplified_constraints = []

    # Process each constraint in the loop
    for con in constraints():
        if con['type'] == 'ineq':
            constraint_expr = con['fun'](l)
            simplified_expr = sp.expand(constraint_expr)
            simplified_constraints.append(simplified_expr)

    return simplified_constraints

# Print the formalized transformation process
alpha = {f'alpha{i+1}{j+1}': alpha_variables[f'alpha{i+1}{j+1}'] for i in range(R) for j in range(W)}

tau = {f'tau{i+1}{j+1}': tau_variables[f'tau{i+1}{j+1}'] for i in range(R) for j in range(W)}

weights = {f'w{i+1}': weight_variables[f'w{i+1}'] for i in range(W)}

m = {f'm{i+1}': variables[f'm{i+1}'] for i in range(R)}

simplified_constraints = simplify_constraint_to_matrix_form(weights, alpha, tau, m, betamax)

for idx, constraint in enumerate(simplified_constraints):
    print(f"Formalized constraint expression {idx + 1}: ")
    print(constraint)


def build_Q_matrices(simplified_constraints):
    Q_list = []
    y_symbols = [sp.Symbol(f'l{i}_{j}') for i in range(1, 4) for j in range(1, 4)] + [1]  # Define y matrix, including [l11, l12, l13, l21, l22, l23, l31, l32, l33, 1]

    for constraint in simplified_constraints:
        # Initialize Q matrix
        Q = sp.zeros(10, 10)

        # Extract each term in the constraint expression and fill in the Q matrix
        for term in constraint.as_ordered_terms():
            # Check if it is a square term
            if isinstance(term, sp.Mul):
                coeff, rest = term.as_coeff_Mul()
                if isinstance(rest, sp.Pow):
                    base, exp = rest.as_base_exp()
                    if exp == 2 and base in y_symbols:
                        idx = y_symbols.index(base)
                        Q[idx, idx] += coeff
                        continue

            coeff, varis = term.as_coeff_mul(*y_symbols)
            indices = [y_symbols.index(var) for var in varis if var in y_symbols]

            if len(indices) == 1:
                # Handle linear terms
                idx = indices[0]
                Q[idx, -1] += coeff / 2
                Q[-1, idx] += coeff / 2
            elif len(indices) == 2:
                # Handle cross terms
                Q[indices[0], indices[1]] += coeff / 2
                Q[indices[1], indices[0]] += coeff / 2
            elif len(indices) == 0:
                # Constant terms
                Q[-1, -1] = constraint.as_ordered_terms()[-1]

        Q_list.append(Q)

    return Q_list

# Generate Q matrix list
Q_matrices = build_Q_matrices(simplified_constraints)

for idx, Q in enumerate(Q_matrices):
    print(f"Q matrix {idx + 1}: ")
    sp.pprint(Q)

# Build SDP optimization problem
# Define variable dimensions
n = 9  # R*W variables: l11, l12, l13, l21, l22, l23, l31, l32, l33
N = n + 1  # Dimension of lifted matrix, including constant terms

# Construct the lifted matrix variable (semidefinite matrix) L = (l11, l12, ..., 1)T @ (l11, l12, ..., 1)
L = cp.Variable((N, N), PSD=True)

# Construct coefficient matrix Q0, vector c0, and constant d0 for the objective function
Q0 = np.zeros((N, N))

# w1 part
Q0[0,0] = w1 * alpha11
Q0[0,9] = 1/2 * w1 * tau11
Q0[9,0] = 1/2 * w1 * tau11
Q0[3,3] = w1 * alpha21
Q0[3,9] = 1/2 * w1 * tau21
Q0[9,3] = 1/2 * w1 * tau21
Q0[6,6] = w1 * alpha31
Q0[6,9] = 1/2 * w1 * tau31
Q0[9,6] = 1/2 * w1 * tau31

# w2 part
Q0[1,1] = w2 * alpha12
Q0[1,9] = 1/2 * w2 * tau12
Q0[9,1] = 1/2 * w2 * tau12
Q0[4,4] = w2 * alpha22
Q0[4,9] = 1/2 * w2 * tau22
Q0[9,4] = 1/2 * w2 * tau22
Q0[7,7] = w2 * alpha32
Q0[7,9] = 1/2 * w2 * tau32
Q0[9,7] = 1/2 * w2 * tau32

# w3 part
Q0[2,2] = w3 * alpha13
Q0[2,9] = 1/2 * w3 * tau13
Q0[9,2] = 1/2 * w3 * tau13
Q0[5,5] = w3 * alpha23
Q0[5,9] = 1/2 * w3 * tau23
Q0[9,5] = 1/2 * w3 * tau23
Q0[8,8] = w3 * alpha33
Q0[8,9] = 1/2 * w3 * tau33
Q0[9,8] = 1/2 * w3 * tau33

# Construct coefficient matrix Q, vector c, and constant d for constraints
num_constraints = len(Q_matrices)
Q = np.zeros((N, N, num_constraints))

# Constraint: yTQy >= 0
for num,Q_mat in enumerate(Q_matrices):
    Q[:,:,num] = Q_mat

# Construct constraint list
constraints = []

# Variable range constraints
constraints += [
    L[i,9] >= 0 for i in range(N) 
]

constraints += [
    L[i,9] <= 1 for i in range(N)
]

constraints += [
    L[0,9] >= ci_results['l1ci'][0], # l11 ≥ l11ci
    L[2,9] <= ci_results['l3ci'][0],  # l13 ≤ l13ci
    L[3,9] <= ci_results['l1ci'][1],  # l21 ≤ l21ci
    L[4,9] >= ci_results['l2ci'][1],  # l22 ≥ l22ci
    L[6,9] >= ci_results['l1ci'][2],  # l31 ≥ l31ci
    L[7,9] <= ci_results['l2ci'][2]  # l32 ≤ l32ci
]

# l11+l21+l31 = l12+l22+l32 = l13+l23+l33 = 1
constraints += [
    L[0,9] + L[3,9] + L[6,9] == 1,
    L[1,9] + L[4,9] + L[7,9] == 1,
    L[2,9] + L[5,9] + L[8,9] == 1
]

# L(-1,-1) == 1 (corresponding to constant term)
constraints += [L[9,9] == 1]

# Ensure diagonal elements satisfy L(ii) ≥ L(i0)^2
constraints += [
    L[i,i] >= cp.square(L[i,0]) for i in range(N-1)
]

# Add linear constraints
for i in range(num_constraints):
    constraints += [
        cp.trace(Q[:,:,i] @ L) >= 0
    ]

# Construct the objective function
obj = cp.Minimize(cp.trace(Q0 @ L))

# Define and solve the optimization problem
prob = cp.Problem(obj, constraints)
prob.solve(solver=cp.MOSEK)
# prob.solve(solver=cp.SCS)

# Check solution status
if prob.status not in ["infeasible", "unbounded"]:
    # Extract solution
    l11 = L.value[0,9]
    l12 = L.value[1,9]
    l13 = L.value[2,9]
    l21 = L.value[3,9]
    l22 = L.value[4,9]
    l23 = L.value[5,9]
    l31 = L.value[6,9]
    l32 = L.value[7,9]
    l33 = L.value[8,9]
    print("SDP relaxed solution is as follows——")
    print(f"Optimal l_1^1: {l11}")
    print(f"Optimal l_1^2: {l12}")
    print(f"Optimal l_1^3: {l13}")
    print(f"Optimal l_2^1: {l21}")
    print(f"Optimal l_2^2: {l22}")
    print(f"Optimal l_2^3: {l23}")
    print(f"Optimal l_3^1: {l31}")
    print(f"Optimal l_3^2: {l32}")
    print(f"Optimal l_3^3: {l33}")
    print(f"Minimum Objective Value: {prob.value}")
else:
    print("Problem is infeasible or unbounded.")


## 5. Finding Feasible Solution Based on SDP Relaxation — Randomization Strategy

# Number of random samples
num_samples = 10000

best_obj = np.inf

best_l = None

# Set mean and standard deviation
mean = L.value[:6, -1]  # Mean

std_dev = 0.01  # Standard deviation is the square root of variance

# Set random seed to ensure reproducibility
np.random.seed(15)

# Generate samples from normal distribution
random_vectors = np.random.normal(loc=mean, scale=std_dev, size=(num_samples, 6))

# Check if constraints are satisfied
for n_sample in range(num_samples):

    l = np.array([random_vectors[n_sample][0], random_vectors[n_sample][1], random_vectors[n_sample][2],
                  random_vectors[n_sample][3], random_vectors[n_sample][4], random_vectors[n_sample][5],
                  1 - random_vectors[n_sample][0] - random_vectors[n_sample][3],
                  1 - random_vectors[n_sample][1] - random_vectors[n_sample][4],
                  1 - random_vectors[n_sample][2] - random_vectors[n_sample][5]])

    # Check variable range constraints
    if ci_results['l1ci'][0] <= l[0] <= 1 and 0 <= l[1] <= 1 and 0 <= l[2] <= ci_results['l3ci'][0] and \
       0 <= l[3] <= ci_results['l1ci'][1] and ci_results['l2ci'][1] <= l[4] <= 1 and 0 <= l[5] <= 1 and \
       ci_results['l1ci'][2] <= l[6] <= 1 and 0 <= l[7] <= ci_results['l2ci'][2] and 0 <= l[8] <= 1:

        # Check original constraints
        con1 = -weight_variables['w1'] * (betamax * l[0] * (alpha_variables['alpha11'] * l[0] + tau_variables['tau11']) + variables['m1']) - \
               weight_variables['w3'] * (betamax * l[0] * (alpha_variables['alpha13'] * l[2] + tau_variables['tau13']) + variables['m1']) + \
               (weight_variables['w1'] * (betamax * l[0] * (alpha_variables['alpha21'] * l[3] + tau_variables['tau21']) + variables['m2']) + \
                weight_variables['w3'] * (betamax * l[0] * (alpha_variables['alpha23'] * l[5] + tau_variables['tau23']) + variables['m2']))
        if con1 < 0:
            continue  # Constraint not satisfied

        con2 = -weight_variables['w2'] * (betamax * l[1] * (alpha_variables['alpha12'] * l[1] + tau_variables['tau12']) + variables['m1']) - \
               weight_variables['w3'] * (betamax * l[1] * (alpha_variables['alpha13'] * l[2] + tau_variables['tau13']) + variables['m1']) + \
               (weight_variables['w2'] * (betamax * l[1] * (alpha_variables['alpha22'] * l[4] + tau_variables['tau22']) + variables['m2']) + \
                weight_variables['w3'] * (betamax * l[1] * (alpha_variables['alpha23'] * l[5] + tau_variables['tau23']) + variables['m2']))
        if con2 < 0:
            continue  # Constraint not satisfied

        con3 = weight_variables['w2'] * (betamax * l[1] * (alpha_variables['alpha12'] * l[1] + tau_variables['tau12']) + variables['m1']) + \
               weight_variables['w1'] * (betamax * l[1] * (alpha_variables['alpha11'] * l[0] + tau_variables['tau11']) + variables['m1']) - \
               (weight_variables['w2'] * (betamax * l[1] * (alpha_variables['alpha22'] * l[4] + tau_variables['tau22']) + variables['m2']) + \
                weight_variables['w1'] * (betamax * l[1] * (alpha_variables['alpha21'] * l[3] + tau_variables['tau21']) + variables['m2']))
        if con3 < 0:
            continue  # Constraint not satisfied

        con4 = weight_variables['w1'] * (betamax * l[2] * (alpha_variables['alpha11'] * l[0] + tau_variables['tau11']) + variables['m1']) + \
               weight_variables['w3'] * (betamax * l[2] * (alpha_variables['alpha13'] * l[2] + tau_variables['tau13']) + variables['m1']) - \
               (weight_variables['w1'] * (betamax * l[2] * (alpha_variables['alpha21'] * l[3] + tau_variables['tau21']) + variables['m2']) + \
                weight_variables['w3'] * (betamax * l[2] * (alpha_variables['alpha23'] * l[5] + tau_variables['tau23']) + variables['m2']))
        if con4 < 0:
            continue  # Constraint not satisfied

        con5 = -weight_variables['w1'] * (betamax * (l[1] + l[4]) * (alpha_variables['alpha21'] * l[3] + tau_variables['tau21']) + variables['m2']) - \
               weight_variables['w2'] * (betamax * (l[1] + l[4]) * (alpha_variables['alpha22'] * l[4] + tau_variables['tau22']) + variables['m2']) + \
               (weight_variables['w1'] * (betamax * (l[1] + l[4]) * (alpha_variables['alpha31'] * l[6] + tau_variables['tau31']) + variables['m3']) + \
                weight_variables['w2'] * (betamax * (l[1] + l[4]) * (alpha_variables['alpha32'] * l[7] + tau_variables['tau32']) + variables['m3']))
        if con5 < 0:
            continue  # Constraint not satisfied

        con6 = weight_variables['w3'] * (betamax * (l[2] + l[5]) * (alpha_variables['alpha23'] * l[5] + tau_variables['tau23']) + variables['m2']) + \
               weight_variables['w2'] * (betamax * (l[2] + l[5]) * (alpha_variables['alpha22'] * l[4] + tau_variables['tau22']) + variables['m2']) - \
               (weight_variables['w3'] * (betamax * (l[2] + l[5]) * (alpha_variables['alpha33'] * l[8] + tau_variables['tau33']) + variables['m3']) + \
                weight_variables['w2'] * (betamax * (l[2] + l[5]) * (alpha_variables['alpha32'] * l[7] + tau_variables['tau32']) + variables['m3']))
        if con6 < 0:
            continue  # Constraint not satisfied

        con7 = -weight_variables['w1'] * (betamax * (l[2] + l[5]) * (alpha_variables['alpha21'] * l[3] + tau_variables['tau21']) + variables['m2']) - \
               weight_variables['w3'] * (betamax * (l[2] + l[5]) * (alpha_variables['alpha23'] * l[5] + tau_variables['tau23']) + variables['m2']) + \
               (weight_variables['w1'] * (betamax * (l[2] + l[5]) * (alpha_variables['alpha31'] * l[6] + tau_variables['tau31']) + variables['m3']) + \
                weight_variables['w3'] * (betamax * (l[2] + l[5]) * (alpha_variables['alpha33'] * l[8] + tau_variables['tau33']) + variables['m3']))
        if con7 < 0:
            continue  # Constraint not satisfied

        con8 = -weight_variables['w1'] * (betamax * l[0] * (alpha_variables['alpha11'] * l[0] + tau_variables['tau11']) + variables['m1']) - \
               weight_variables['w2'] * (betamax * l[0] * (alpha_variables['alpha12'] * l[1] + tau_variables['tau12']) + variables['m1']) - \
               weight_variables['w3'] * (betamax * l[0] * (alpha_variables['alpha13'] * l[2] + tau_variables['tau13']) + variables['m1']) + \
               weight_variables['w1'] * (betamax * l[0] * (alpha_variables['alpha21'] * l[3] + tau_variables['tau21']) + variables['m2']) + \
               weight_variables['w2'] * (betamax * l[0] * (alpha_variables['alpha22'] * l[4] + tau_variables['tau22']) + variables['m2']) + \
               weight_variables['w3'] * (betamax * l[0] * (alpha_variables['alpha23'] * l[5] + tau_variables['tau23']) + variables['m2'])
        if con8 < 0:
            continue  # Constraint not satisfied

        con9 = weight_variables['w1'] * (betamax * l[2] * (alpha_variables['alpha11'] * l[0] + tau_variables['tau11']) + variables['m1']) + \
               weight_variables['w2'] * (betamax * l[2] * (alpha_variables['alpha12'] * l[1] + tau_variables['tau12']) + variables['m1']) + \
               weight_variables['w3'] * (betamax * l[2] * (alpha_variables['alpha13'] * l[2] + tau_variables['tau13']) + variables['m1']) - \
               (weight_variables['w1'] * (betamax * l[2] * (alpha_variables['alpha21'] * l[3] + tau_variables['tau21']) + variables['m2']) + \
                weight_variables['w2'] * (betamax * l[2] * (alpha_variables['alpha22'] * l[4] + tau_variables['tau22']) + variables['m2']) + \
                weight_variables['w3'] * (betamax * l[2] * (alpha_variables['alpha23'] * l[5] + tau_variables['tau23']) + variables['m2']))
        if con9 < 0:
            continue  # Constraint not satisfied

        con10 = -weight_variables['w1'] * (betamax * (l[1] + l[4]) * (alpha_variables['alpha21'] * l[3] + tau_variables['tau21']) + variables['m2']) - \
                weight_variables['w2'] * (betamax * (l[1] + l[4]) * (alpha_variables['alpha22'] * l[4] + tau_variables['tau22']) + variables['m2']) - \
                weight_variables['w3'] * (betamax * (l[1] + l[4]) * (alpha_variables['alpha23'] * l[5] + tau_variables['tau23']) + variables['m2']) + \
                (weight_variables['w1'] * (betamax * (l[1] + l[4]) * (alpha_variables['alpha31'] * l[6] + tau_variables['tau31']) + variables['m3']) + \
                 weight_variables['w2'] * (betamax * (l[1] + l[4]) * (alpha_variables['alpha32'] * l[7] + tau_variables['tau32']) + variables['m3']) + \
                 weight_variables['w3'] * (betamax * (l[1] + l[4]) * (alpha_variables['alpha33'] * l[8] + tau_variables['tau33']) + variables['m3']))
        if con10 < 0:
            continue  # Constraint not satisfied

        # Calculate objective function value
        obj_value = objective(l)
        if obj_value < best_obj:
            best_obj = obj_value
            best_l = l

if best_l is not None:
    print("Approximate solution found through randomization:")
    print(f"l11 = {best_l[0]}")
    print(f"l12 = {best_l[1]}")
    print(f"l13 = {best_l[2]}")
    print(f"l21 = {best_l[3]}")
    print(f"l22 = {best_l[4]}")
    print(f"l23 = {best_l[5]}")
    print(f"l31 = {best_l[6]}")
    print(f"l32 = {best_l[7]}")
    print(f"l33 = {best_l[8]}")
    print(f"Objective value = {best_obj}")
else:
    print("No feasible solution found through randomization.")