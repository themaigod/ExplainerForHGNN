import torch


class BaseExplanation:
    pass


class NodeExplanation(BaseExplanation):
    def __init__(self, **kwargs):
        self._other_data = self.get_attrs(kwargs)

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        elif key.startswith("_"):
            super().__setattr__(key, value)
        else:
            self._other_data[key] = self.safe_tensor_copy(value)

    @staticmethod
    def safe_tensor_copy(tensor):
        if tensor is not None:
            if isinstance(tensor, torch.Tensor):
                return tensor.clone().detach()
            if isinstance(tensor, list):
                return [NodeExplanation.safe_tensor_copy(t) for t in tensor]
            if isinstance(tensor, dict):
                return {k: NodeExplanation.safe_tensor_copy(v) for k, v in
                        tensor.items()}
            import copy
            return copy.deepcopy(tensor)
        else:
            return None

    @staticmethod
    def get_attr(kwargs, attr):
        result = kwargs.get(attr, None)
        return NodeExplanation.safe_tensor_copy(result)

    def get_attrs(self, kwargs):
        return {attr: self.get_attr(kwargs, attr) for attr in kwargs}

    def __repr__(self):
        return f"Node Explanation for node {self.node_id}"

    def __getattr__(self, item):
        if item in self._other_data:
            return self._other_data[item]
        else:
            raise None

    def __getitem__(self, item):
        return self._other_data[item]

    def __contains__(self, item):
        return item in self._other_data


class NodeExplanationCombination(BaseExplanation):
    def __init__(self, node_explanations=None, **kargs):
        self.node_explanations = node_explanations if node_explanations is not None else []
        self.control_data = kargs  # control settings of the metrics

    def __repr__(self):
        return f"Node Explanation Combination with {len(self.node_explanations)} nodes"

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        elif key.startswith("_"):
            super().__setattr__(key, value)
        elif key == "node_explanations" or key == "control_data":
            self.__dict__[key] = value
        else:
            self.control_data[key] = value

    def __getitem__(self, item):
        return self.node_explanations[item]

    def __len__(self):
        return len(self.node_explanations)

    def __getattr__(self, item):
        if item in self.control_data:
            return self.control_data[item]
        result = [self.node_explanations[i].__getattr__(item) for i in
                  range(len(self.node_explanations))]
        # check if all the elements in the list are None
        if all([r is None for r in result]):
            raise AttributeError(
                f"Attribute {item} not found in NodeExplanationCombination")
        return result

    def __contains__(self, item):
        if item in self.control_data:
            return True
        if item in self.node_explanations[0]:
            return True
