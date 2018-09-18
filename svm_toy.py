import cvxpy as cvx
# Create two scalar optimization variables.
x = cvx.Variable()
y = cvx.Variable()
# Create two constraints.
constraints = [x + y == 1,
x - y >= 1]
# Form objective.
obj = cvx.Minimize((x - y)**2)
# Form and solve problem.
prob = cvx.Problem(obj, constraints)
prob.solve()
# The optimal dual variable (Lagrange multiplier) for
# a constraint is stored in constraint.dual_value.
print("optimal (x + y == 1) dual variable", constraints[0].dual_value)
print("optimal (x - y >= 1) dual variable", constraints[1].dual_value)
print("x - y value:", (x - y).value)