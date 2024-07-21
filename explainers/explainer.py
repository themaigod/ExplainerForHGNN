class Explainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.metrics = config.get("metrics", None)

    def explain(self, model):
        self.model = model
        self.model.eval()
        pass

    def evaluate(self):
        pass

    def visualize(self):
        pass

    def get_summary(self):
        pass

    def save_summary(self):
        pass
