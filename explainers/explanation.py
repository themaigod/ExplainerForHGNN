import torch
import os


def _packaged2tensor_v1(data):
    if isinstance(data, dict):
        if 'type' in data and data['type'] == 'tensor':
            if data['is_sparse']:
                indices = torch.tensor(data['indices'], dtype=torch.long)
                values = torch.tensor(data['values'])
                return torch.sparse_coo_tensor(indices, values, data['size'])
            return torch.tensor(data['tensor'])
        return {k: _packaged2tensor_v1(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_packaged2tensor_v1(d) for d in data]


packaged_deserializer = {
    '1': _packaged2tensor_v1
}


def packaged2tensor(data, version='1'):
    return packaged_deserializer[version](data)


def _tensor2packaged_v1(data):
    if isinstance(data, torch.Tensor):
        # consider the tensor is a sparse tensor
        if data.is_sparse:
            data = data.coalesce()
            return {'tensor':
                {
                    'indices': data.indices().cpu().numpy(),
                    'values': data.values().cpu().numpy(),
                    'size': list(data.size()),
                },
                'type': 'tensor',
                'is_sparse': True}
        return {'tensor': data.cpu().numpy(),
                'type': 'tensor', 'is_sparse': False}
    if isinstance(data, list):
        return [_tensor2packaged_v1(d) for d in data]
    if isinstance(data, dict):
        return {k: _tensor2packaged_v1(v) for k, v in data.items()}
    return data


packaged_serializer = {
    '1': _tensor2packaged_v1
}


def tensor2packaged(data, version='1'):
    return packaged_serializer[version](data)


def load_explanation_related_data2object(data):
    if 'type' in data:
        if data['type'] == 'NodeExplanation':
            return NodeExplanation.from_json(data)
        if data['type'] == 'NodeExplanationCombination':
            return NodeExplanationCombination.from_json(data)
    elif isinstance(data, dict):
        return {k: load_explanation_related_data2object(v) for k, v in data}
    elif isinstance(data, list):
        return [load_explanation_related_data2object(d) for d in data]
    return data


def save_explanation_related_object2data(obj, dir_path=None, file_name=None):
    if isinstance(obj, NodeExplanation):
        if dir_path is None:
            dir_path = "./explainer_results"
        if file_name is None:
            file_name = f"node_explanation_{obj.node_id}.pkl"
        obj.save(os.path.join(dir_path, file_name))
        return
    if isinstance(obj, NodeExplanationCombination):
        obj.save(dir_path)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            save_explanation_related_object2data(v, os.path.join(dir_path, k))
        return
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            save_explanation_related_object2data(v, os.path.join(dir_path, str(i)))
        return
    raise ValueError(f"Object {obj} is not a valid explanation object")


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
        elif callable(value):
            super().__setattr__(key, value)
        else:
            self._other_data[key] = self.safe_tensor_copy(value)

    @staticmethod
    def safe_tensor_copy(tensor):
        if tensor is not None:
            if isinstance(tensor, torch.Tensor):
                return tensor.cpu().clone().detach()
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

    def __len__(self):
        return len(self._other_data)

    def to_dict(self):
        return self._other_data

    @staticmethod
    def packaged_like(data, version='1'):
        return tensor2packaged(data, version)

    @staticmethod
    def packaged2tensor(data, version='1'):
        return packaged2tensor(data, version)

    def to_packaged(self, version='1'):
        return self.packaged_like(self.to_dict(), version)

    @classmethod
    def from_packaged(cls, data):
        result = cls()
        result._other_data = packaged2tensor(data)
        return result

    def save(self, file_path):
        import pickle as pkl
        with open(file_path, 'wb') as f:
            pkl.dump(self.to_packaged(), f)

    @classmethod
    def from_file(cls, file_path):
        import pickle as pkl
        with open(file_path, 'rb') as f:
            data = pkl.load(f)
            return cls.from_packaged(data)


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
        elif callable(value):
            super().__setattr__(key, value)
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
        return False

    def __iter__(self):
        return iter(self.node_explanations)

    def to_dict(self):
        return {"node_explanations": [ne.to_dict() for ne in self.node_explanations],
                "control_data": self.control_data}

    @staticmethod
    def packaged_like(data, version='1'):
        return tensor2packaged(data, version)

    @staticmethod
    def packaged2tensor(data, version='1'):
        return packaged2tensor(data, version)

    def save(self, dir_path):
        import pickle as pkl
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path, 'node_explanations_meta.pkl'), 'wb') as f:
            pkl.dump({
                'control_data': self.packaged_like(self.control_data),
                'node_explanations': [ne.node_id for ne in self.node_explanations]
                },
                f)

        for i, ne in enumerate(self.node_explanations):
            ne.save(os.path.join(dir_path, f'node_explanation_{ne.node_id}.pkl'))
