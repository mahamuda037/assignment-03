import numpy as np
import pandas as pd

def least_cost_method_step_by_step(supply, demand, cost):
    m, n = len(supply), len(demand)
    allocation = np.zeros((m, n))
    total_cost = 0
    steps = []  # To save steps as tables

    while sum(supply) > 0 and sum(demand) > 0:
        min_cost = float('inf')
        min_pos = (0, 0)

        # Find the least cost cell
        for i in range(m):
            for j in range(n):
                if supply[i] > 0 and demand[j] > 0 and cost[i][j] < min_cost:
                    min_cost = cost[i][j]
                    min_pos = (i, j)

        i, j = min_pos
        allocation_amount = min(supply[i], demand[j])
        allocation[i][j] = allocation_amount
        total_cost += allocation_amount * cost[i][j]

        # Save table for the current step
        table = pd.DataFrame(data=allocation, columns=["P", "Q", "R", "S", "T"], 
                             index=["A", "B", "C", "D"]).astype(int)
        steps.append({"step_table": table, "supply": supply.copy(), "demand": demand.copy(), "cost": total_cost})

        supply[i] -= allocation_amount
        demand[j] -= allocation_amount

    return allocation, total_cost, steps

# Problem data
cost = np.array([[4, 3, 1, 2, 6],
                 [5, 2, 3, 4, 5],
                 [3, 5, 6, 3, 2],
                 [2, 4, 4, 5, 3]])
supply = [80, 60, 40, 20]
demand = [60, 60, 30, 40, 10]

# Solve problem
allocation, total_cost, steps = least_cost_method_step_by_step(supply, demand, cost)

# Print each step as a table
for idx, step in enumerate(steps):
    print(f"Step {idx + 1}:")
    print(step["step_table"])
    print("Remaining Supply:", step["supply"])
    print("Remaining Demand:", step["demand"])
    print("Cumulative Cost:", step["cost"])
    print("---")
