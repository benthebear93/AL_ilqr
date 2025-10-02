class CostCollection:
    def __init__(self, costs):
        self.costs = costs

    def evaluate(self, states, actions, parameters):
        J = 0.0
        for t, c in enumerate(self.costs):
            J += c.evaluate(states[t], actions[t], parameters[t])
        return J

    def __len__(self):
        return len(self.costs)

    def __getitem__(self, idx):
        return self.costs[idx]

    def __iter__(self):
        return iter(self.costs)