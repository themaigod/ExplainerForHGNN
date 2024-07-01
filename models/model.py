import torch


class BaseModel(torch.nn.Module):
    def __init__(self, config, dataset):
        super(BaseModel, self).__init__()
        self.config = config
        self.dataset = dataset
        self.summary = None
        self.metrics = config.get("metrics", None)

        # to be used in Explainer for node explanation or graph explanation
        # single_graph is True if the model is for node classification, link prediction, etc.
        # single_graph is False if the model is for graph classification, etc.
        self.single_graph = getattr(dataset, "single_graph", None)  # bool

    def forward(self):
        """
        For explanation usage, the forward function should not contain any input.
        The input should be passed in dataset, and further process in dataset_adaptation.
        :return:
        """
        raise NotImplementedError

    def fit(self):
        """
        Also for explanation usage, the fit function should not contain any input.
        :return:
        """
        raise NotImplementedError

    def get_summary(self):
        raise NotImplementedError

    def save_summary(self):
        raise NotImplementedError

    def dataset_adaptation(self):
        """
        This function should be used to adapt the dataset to the model input.
        :return:
        """
        raise NotImplementedError
