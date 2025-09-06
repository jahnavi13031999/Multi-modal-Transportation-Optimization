from docplex.mp.model import Model
import numpy as np

class MultiModalTransport:
    def __init__(self, alpha=0.6):
        self.model = Model(name="Multi-Modal Transportation Optimization")
        self.alpha = alpha  # Trade-off factor between cost and delivery time

        # Dimensions
        self.portDim = 4  # Reduced dimensions for example output
        self.goodsDim = 2
        self.timeDim = 5

        # Decision Variables
        self.x = self.model.binary_var_list(self.portDim * self.portDim * self.timeDim * self.goodsDim, name='x')
        self.y = self.model.integer_var_list(self.portDim * self.portDim * self.timeDim, name='y')
        self.z = self.model.binary_var_list(self.portDim * self.portDim * self.timeDim, name='z')
        
        # Reshape variables for indexing convenience
        self.x = np.array(self.x).reshape(self.portDim, self.portDim, self.timeDim, self.goodsDim)
        self.y = np.array(self.y).reshape(self.portDim, self.portDim, self.timeDim)
        self.z = np.array(self.z).reshape(self.portDim, self.portDim, self.timeDim)

        # Parameters (example values for demonstration)
        self.perCtnCost = np.random.randint(10, 100, size=(self.portDim, self.portDim, self.timeDim))
        self.tranFixedCost = np.random.randint(10, 50, size=(self.portDim, self.portDim, self.timeDim))
        self.warehouse_fee = np.random.randint(5, 20, size=(self.portDim,))
        self.taxPct = np.random.rand(self.goodsDim)
        self.transitDuty = np.random.rand(self.portDim, self.portDim)
        self.kValue = np.random.randint(100, 1000, size=self.goodsDim)
        self.kVol = np.random.randint(1, 10, size=self.goodsDim)
        self.ctnVol = np.random.randint(10, 100, size=(self.portDim, self.portDim))
        self.OrderDate = np.random.randint(1, 5, size=self.goodsDim)
        self.DeliveryDeadline = np.random.randint(20, 30, size=self.goodsDim)
        self.tranTime = np.random.randint(1, 5, size=(self.portDim, self.portDim, self.timeDim))

    def build_objective(self):
        # Calculate costs
        transport_cost = np.sum(self.y * self.perCtnCost) + np.sum(self.z * self.tranFixedCost)
        warehouse_cost = np.sum(self.warehouse_fee)  # Adjust as needed
        tax_cost = np.sum(self.taxPct * self.kValue) + np.sum(self.x * self.transitDuty.reshape(self.portDim, self.portDim, 1, 1) * self.kValue.reshape(1, 1, 1, self.goodsDim))
        total_cost = transport_cost + warehouse_cost + tax_cost

        # Calculate delivery time as sum of transit times for all goods
        arrival_times = np.arange(self.timeDim).reshape(1, 1, self.timeDim, 1) * self.x + self.tranTime.reshape(self.portDim, self.portDim, self.timeDim, 1) * self.x
        delivery_time = np.sum(arrival_times)

        # Multi-objective function with alpha as a trade-off factor
        objective = self.alpha * total_cost + (1 - self.alpha) * delivery_time
        self.model.minimize(objective)

    def add_constraints(self):
        # constraints
        for k in range(self.goodsDim):
            self.model.add_constraint(np.sum(self.x[:, :, :, k]) == 1)

        for i in range(self.portDim):
            for j in range(self.portDim):
                for t in range(self.timeDim):
                    self.model.add_constraint(self.y[i, j, t] >= np.dot(self.x[i, j, t, :], self.kVol) / max(1, self.ctnVol[i, j]))  # Avoid division by zero

    def solve(self):
        self.model.solve()
        results = {
            "total_cost": self.alpha * self.model.solution.get_objective_value(),
            "total_delivery_time": (1 - self.alpha) * self.model.solution.get_objective_value(),
            "objective_value": self.model.solution.get_objective_value()
        }
        return results

# Running the model and produce output in text file
model = MultiModalTransport(alpha=0.6)
model.build_objective()
model.add_constraints()
results = model.solve()

# Save the output to a text file
output_text = (
    "Multi-Objective Optimization Results\n"
    "------------------------------------\n"
    f"Alpha (Trade-off factor between cost and time): {model.alpha}\n"
    f"Total Cost (weighted): {results['total_cost']:.2f}\n"
    f"Total Delivery Time (weighted): {results['total_delivery_time']:.2f}\n"
    f"Objective Value: {results['objective_value']:.2f}\n"
)

output_path = 'E:/modal transport/Solution_Output.txt'
with open(output_path, 'w') as f:
    f.write(output_text)

output_path





