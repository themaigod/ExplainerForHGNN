import copy

import numpy as np
import pandas as pd
from scipy import stats
import scipy

from .explainer import Explainer, ExplainerCore
from .explanation import NodeExplanation, NodeExplanationCombination
from .prepare_explanation_for_node_scores import standard_explanation
from .prepare_explanation_for_node_dataset_scores import \
    prepare_explanation_fn_for_node_dataset_scores
import torch


def power_divergence(X, Y, Z, data, boolean=True, lambda_="cressie-read", **kwargs):
    """
    From pgmpy package, we do not want to add the dependency, so we copy the code here.
    Computes the Cressie-Read power divergence statistic [1]. The null hypothesis
    for the test is X is independent of Y given Z. A lot of the frequency comparision
    based statistics (eg. chi-square, G-test etc) belong to power divergence family,
    and are special cases of this test.

    Parameters
    ----------
    X: int, string, hashable object
        A variable name contained in the data set

    Y: int, string, hashable object
        A variable name contained in the data set, different from X

    Z: list, array-like
        A list of variable names contained in the data set, different from X and Y.
        This is the separating set that (potentially) makes X and Y independent.
        Default: []

    data: pandas.DataFrame
        The dataset on which to test the independence condition.

    lambda_: float or string
        The lambda parameter for the power_divergence statistic. Some values of
        lambda_ results in other well known tests:
            "pearson"             1          "Chi-squared test"
            "log-likelihood"      0          "G-test or log-likelihood"
            "freeman-tuckey"     -1/2        "Freeman-Tuckey Statistic"
            "mod-log-likelihood"  -1         "Modified Log-likelihood"
            "neyman"              -2         "Neyman's statistic"
            "cressie-read"        2/3        "The value recommended in the paper[1]"

    boolean: bool
        If boolean=True, an additional argument `significance_level` must
            be specified. If p_value of the test is greater than equal to
            `significance_level`, returns True. Otherwise returns False.

        If boolean=False, returns the chi2 and p_value of the test.

    Returns
    -------
    CI Test Results: tuple or bool
        If boolean = False, Returns a tuple (chi, p_value, dof). `chi` is the
        chi-squared test statistic. The `p_value` for the test, i.e. the
        probability of observing the computed chi-square statistic (or an even
        higher value), given the null hypothesis that X \u27C2 Y | Zs is True.
        If boolean = True, returns True if the p_value of the test is greater
        than `significance_level` else returns False.

    References
    ----------
    [1] Cressie, Noel, and Timothy RC Read. "Multinomial goodness‐of‐fit tests." Journal of the Royal Statistical Society: Series B (Methodological) 46.3 (1984): 440-464.
    """

    # Step 1: Check if the arguments are valid and type conversions.
    if hasattr(Z, "__iter__"):
        Z = list(Z)
    else:
        raise (f"Z must be an iterable. Got object type: {type(Z)}")

    if (X in Z) or (Y in Z):
        raise ValueError(
            f"The variables X or Y can't be in Z. Found {X if X in Z else Y} in Z."
        )

    # Step 2: Do a simple contingency test if there are no conditional variables.
    if len(Z) == 0:
        chi, p_value, dof, expected = stats.chi2_contingency(
            data.groupby([X, Y]).size().unstack(Y, fill_value=0), lambda_=lambda_
        )

    # Step 3: If there are conditionals variables, iterate over unique states and do
    #         the contingency test.
    else:
        chi = 0
        dof = 0
        for z_state, df in data.groupby(Z):
            try:
                c, _, d, _ = stats.chi2_contingency(
                    df.groupby([X, Y]).size().unstack(Y, fill_value=0), lambda_=lambda_
                )
                chi += c
                dof += d
            except ValueError:
                # If one of the values is 0 in the 2x2 table.
                if isinstance(z_state, str):
                    print(
                        f"Skipping the test {X} \u27C2 {Y} | {Z[0]}={z_state}. Not enough samples"
                    )
                else:
                    z_str = ", ".join(
                        [f"{var}={state}" for var, state in zip(Z, z_state)]
                    )
                    print(
                        f"Skipping the test {X} \u27C2 {Y} | {z_str}. Not enough samples"
                    )
        p_value = 1 - stats.chi2.cdf(chi, df=dof)

    # Step 4: Return the values
    if boolean:
        return p_value >= kwargs["significance_level"]
    else:
        return chi, p_value, dof


def g_sq(X, Y, Z, data, boolean=True, **kwargs):
    """
    From pgmpy package, we do not want to add the dependency, so we copy the code here.
    G squared test for conditional independence. Also commonly known as G-test,
    likelihood-ratio or maximum likelihood statistical significance test.
    Tests the null hypothesis that X is independent of Y given Zs.

    Parameters
    ----------
    X: int, string, hashable object
        A variable name contained in the data set

    Y: int, string, hashable object
        A variable name contained in the data set, different from X

    Z: list (array-like)
        A list of variable names contained in the data set, different from X and Y.
        This is the separating set that (potentially) makes X and Y independent.
        Default: []

    data: pandas.DataFrame
        The dataset on which to test the independence condition.

    boolean: bool
        If boolean=True, an additional argument `significance_level` must be
        specified. If p_value of the test is greater than equal to
        `significance_level`, returns True. Otherwise returns False. If
        boolean=False, returns the chi2 and p_value of the test.

    Returns
    -------
    CI Test Results: tuple or bool
        If boolean = False, Returns a tuple (chi, p_value, dof). `chi` is the
        chi-squared test statistic. The `p_value` for the test, i.e. the
        probability of observing the computed chi-square statistic (or an even
        higher value), given the null hypothesis that X \u27C2 Y | Zs is True.
        If boolean = True, returns True if the p_value of the test is greater
        than `significance_level` else returns False.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/G-test

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame(np.random.randint(0, 2, size=(50000, 4)), columns=list('ABCD'))
    >>> data['E'] = data['A'] + data['B'] + data['C']
    >>> g_sq(X='A', Y='C', Z=[], data=data, boolean=True, significance_level=0.05)
    True
    >>> g_sq(X='A', Y='B', Z=['D'], data=data, boolean=True, significance_level=0.05)
    True
    >>> g_sq(X='A', Y='B', Z=['D', 'E'], data=data, boolean=True, significance_level=0.05)
    False
    """
    return power_divergence(
        X=X, Y=Y, Z=Z, data=data, boolean=boolean, lambda_="log-likelihood", **kwargs
    )


class HENCEXCore(ExplainerCore):
    def __init__(self, config):
        super().__init__(config)

    def explain(self, model, **kwargs):
        self.model = model
        self.model.eval()

        if self.model.dataset.single_graph:
            self.node_id = kwargs.get('node_id', None)

        if self.model.dataset.single_graph:
            if self.node_id is None:
                raise ValueError('node_id is required for node-level explanation')
            return self.node_level_explain()
        else:
            return self.graph_level_explain()

    def graph_level_explain(self):
        pass

    def node_level_explain(self):
        _ = self.extract_neighbors_input()
        perturb_result, mask_all, perturb_position = self.get_perturb_result()
        candidates, candidates_features = self.select_candidates(perturb_result,
                                                                 mask_all,
                                                                 perturb_position)
        selected_candidates, candidates_features = self.select_features(candidates,
                                                                        candidates_features,
                                                                        perturb_result,
                                                                        mask_all,
                                                                        perturb_position)
        candidates_features_dict = {candidates[i]: candidates_features[i] for i in
                                    range(len(candidates))}
        selected_candidates = self.drop_nodes(selected_candidates, candidates_features,
                                              perturb_result,
                                              mask_all, perturb_position)
        feature_exp = self.feature_raw2out(candidates_features_dict,
                                           selected_candidates)
        return self.construct_explanation()

    def feature_raw2out(self, candidates_features_dict, selected_candidates):
        gs, features = self.extract_neighbors_input()
        feature_mask = np.zeros_like(features, dtype=bool)
        for node in selected_candidates:
            feature_mask[node, candidates_features_dict[node]] = True
        self.feature_mask = feature_mask
        self.feature_raw = features[feature_mask]
        return self.feature_raw

    def drop_nodes(self, selected_candidates, candidates_features, perturb_result,
                   mask_all, perturb_position
                   ):
        gs, features = self.extract_neighbors_input()
        pd_data = [np.expand_dims(perturb_result, axis=1)]

        # use connectivities to remove nodes
        can_remove = []
        for node in selected_candidates:
            if node == self.mapped_node_id():
                continue
            if self.check_connectivity(node, gs):
                can_remove.append(node)

        for node in selected_candidates:
            mask_u = [i[node] for i in mask_all]
            data = np.zeros((len(mask_u), features.shape[1]))
            data[mask_u] = perturb_position
            data = data[:, candidates_features[selected_candidates.index(node)]]
            data = self.vec2categ(data)
            pd_data.append(data)

        for node in selected_candidates:
            data = self.feature_categ[node]
            pd_data.append(data)

        pd_data = np.concatenate(pd_data, axis=1)
        pd_data = pd.DataFrame(pd_data, columns=['target'] +
                                                [str(i) for i in selected_candidates] +
                                                ["s" + str(i) for i in
                                                 range(len(selected_candidates))]
                               )

        p_values = [g_sq('target', "u" + str(i), [], pd_data, boolean=False)[1]
                    for i in can_remove]

        selected_candidates_set = set(selected_candidates)
        p_threshold = self.config.get('p_threshold', 0.05)
        for idx in np.argsort(-1 * np.array(p_values)):
            if g_sq('target', "u" + str(can_remove[idx]), [str(i) for i in
                                                           range(selected_candidates)],
                    pd_data, boolean=True, significance_level=p_threshold):
                selected_candidates_set.remove(can_remove[idx])

        return list(selected_candidates_set)

    def check_connectivity(self, node, gs):
        g_total = gs[0]
        if len(gs) > 1:
            for g in gs[1:]:
                g_total += g
        g_total = g_total.coalesce()

        # remove node
        indices = g_total.indices()
        values = g_total.values()
        mask = (indices[0] != node) & (indices[1] != node)
        new_g = torch.sparse_coo_tensor(indices[:, mask], values[mask], g_total.size())
        new_g = new_g.coalesce()

        # check connectivity
        return self.check_connectivity_for_g(new_g)

    def check_connectivity_for_g(self, g):
        # convert to scipy
        g = g.coalesce()
        indices = g.indices()
        values = g.values()
        adj = scipy.sparse.coo_matrix((values.cpu().numpy(), (indices[0].cpu().numpy(),
                                                              indices[
                                                                  1].cpu().numpy())),
                                      shape=(g.size(0), g.size(1)))
        n_components, _ = scipy.sparse.csgraph.connected_components(adj,
                                                                    directed=False,
                                                                    connection='weak')
        return n_components == 1

    def select_features(self, candidates, candidates_features, perturb_result, mask_all,
                        perturb_position
                        ):
        gs, features = self.extract_neighbors_input()
        feature_categ = {}
        for idx, node in enumerate(candidates):
            mask_u = [i[node] for i in mask_all]
            data = np.zeros((len(mask_u), features.shape[1]))
            data[mask_u] = perturb_position
            feature_categ_tmp = self.vec2categ(data[:, candidates_features[idx]])
            feature_categ[node] = feature_categ_tmp

        selected = []
        processing_list = [self.mapping_node_id()]
        excluded = set()

        p_values = [-1]
        c = 0

        p_threshold = self.config.get('p_threshold', 0.05)

        new_candidates_features = {}

        while len(processing_list) > 0 and min(p_values) < p_threshold:
            current_node = processing_list[np.argmin(p_values)]
            excluded = excluded.union(
                set([processing_list[i] for i in range(len(p_values)) if
                     p_values[i] > p_threshold]))

            processing_list = set(processing_list)
            processing_list.remove(current_node)

            feature_categ_tmp_all = []
            for node in selected:
                mask_u = [i[node] for i in mask_all]
                data = np.zeros((len(mask_u), features.shape[1]))
                data[mask_u] = perturb_position
                feature_categ_tmp = self.vec2categ(data)
                feature_categ_tmp_all.append(feature_categ_tmp)
            pd_data = [np.expand_dims(perturb_result, axis=1)]
            pd_data.extend(feature_categ_tmp_all)
            pd_data = np.concatenate(pd_data, axis=1)

            mask_u = [i[current_node] for i in mask_all]
            data = np.zeros((len(mask_u), features.shape[1]))
            data[mask_u] = perturb_position
            data = data[:, candidates_features[candidates.index(current_node)]]

            pd_data = np.concatenate((pd_data, data), axis=1)
            pd_data = pd.DataFrame(pd_data, columns=['target'] +
                                                    ["s" + str(i) for i in
                                                     range(len(selected))] +
                                                    [str(i) for i in
                                                     range(data.shape[1])])
            feat_p_values = [g_sq('target', str(i), ["s" + str(i) for i in
                                                     range(len(selected))], pd_data,
                                  boolean=False)[1]
                             for i in range(data.shape[1])]
            feat_p_values_pick = np.array(feat_p_values) < p_threshold
            if np.any(feat_p_values_pick):
                new_candidates_features[current_node] = \
                    candidates_features[candidates.index(current_node)][
                        feat_p_values_pick]
                selected.append(current_node)
                processing_list.union(set(self.get_neighbors(current_node, gs,
                                                             range_=candidates))).difference(
                    selected)
            else:
                excluded.add(current_node)
            processing_list = processing_list.difference(excluded)

            if len(processing_list) > 0:
                processing_list = list(processing_list)
                feature_categ_tmp_all = []
                for node in selected:
                    mask_u = [i[node] for i in mask_all]
                    data = np.zeros((len(mask_u), features.shape[1]))
                    data[mask_u] = perturb_position
                    feature_categ_tmp = self.vec2categ(data)
                    feature_categ_tmp_all.append(feature_categ_tmp)

                pd_data = [np.expand_dims(perturb_result, axis=1)]
                pd_data.extend(feature_categ_tmp_all)

                pd_data = np.concatenate(pd_data, axis=1)

                feature_categ_tmp_all = []
                for node in processing_list:
                    feature_categ_tmp = feature_categ[node]
                    feature_categ_tmp_all.append(feature_categ_tmp)

                pd_data = [pd_data]
                pd_data.extend(feature_categ_tmp_all)
                pd_data = np.concatenate(pd_data, axis=1)

                pd_data = pd.DataFrame(pd_data, columns=['target'] +
                                                        ["s" + str(i) for i in
                                                         range(len(selected))] +
                                                        ["u" + str(i) for i in
                                                         range(len(processing_list))])
                p_values = [g_sq('target', "u" + str(i), ["s" + str(j) for j in
                                                          range(len(selected))],
                                 pd_data, boolean=False)[1]
                            for i in range(len(processing_list))]
                c += 1

        if len(selected) == 0:
            print("Warning: No feature is selected")

        self.feature_categ = feature_categ

        return selected, new_candidates_features

    def vec2categ(self, vec):
        if vec.shape[1] == 1:
            return vec

        base = np.array([2 ** k for k in range(vec.shape[1])])
        base = np.expand_dims(base, axis=0)
        base = np.repeat(base, vec.shape[0], axis=0)
        cat = np.expand_dims(np.sum(base * vec, axis=-1), axis=0).T
        cat = cat - np.min(cat) * np.ones(cat.shape, dtype=np.int8)

        return cat

    def select_candidates(self, perturb_result, mask_all, perturb_position):
        candidates = []
        candidates_features = []

        processing_list = set(self.mapping_node_id())

        processed = set()
        c = 0

        gs, features = self.extract_neighbors_input()

        p_threshold = self.config.get('p_threshold', 0.05)

        while len(processing_list) > 0:

            c += 1

            processing_list_tmp = processing_list.copy()
            for node in processing_list_tmp:
                mask_u = [i[node] for i in mask_all]
                data = np.zeros((len(mask_u), features.shape[1]))
                data[mask_u] = perturb_position
                combined = np.concatenate((perturb_result.reshape(-1, 1), data), axis=1)
                pd_data = pd.DataFrame(combined, columns=['target'] + [str(i) for i in
                                                                       range(data.shape[
                                                                                 1])])
                feat_p_values = [g_sq('target', str(i), [], pd_data, boolean=False)[1]
                                 for i in range(data.shape[1])]

                feat_pick = np.array(feat_p_values) < p_threshold
                if np.any(feat_pick):
                    candidates.append(node)
                    processed.add(node)
                    processing_list.remove(node)
                    # feat_pick = np.where(feat_pick)[0]
                    candidates_features.append(feat_pick)
                    processing_list.union(set(self.get_neighbors(node, gs)))

            processed.union(processing_list_tmp)
            processing_list = processing_list.difference(processed)

        if len(candidates) == 0:
            print("Warning: No candidate features are selected")

        return candidates, candidates_features

    def get_neighbors(self, node, gs, range_=None):
        neighbors = []
        for g in gs:
            g = g.coalesce()
            indices = g.indices()
            mask = (indices[0] == node) | (indices[1] == node)
            neighbors.extend(indices[1][mask].tolist())
        neighbors = list(set(neighbors))
        if range_ is not None:
            neighbors = [i for i in neighbors if i in range_]
        return neighbors

    def get_perturb_result(self):
        min_num_samples = self.config.get('min_num_samples', 1000)
        n_cat_value = self.model.dataset.num_classes

        p_perturb = self.config.get('p_perturb', 0.1)

        # calculate the number of non-zero features
        gs, features = self.extract_neighbors_input()
        num_non_zero_features = torch.count_nonzero(features).item()
        num_RV = n_cat_value + num_non_zero_features
        k = self.config.get('k', 15)  # suggest in paper
        num_samples = max(min_num_samples, k * num_RV)

        num_samples = min(num_samples, 10000)  # avoid too many samples

        def handle_fn(model):
            return gs, features

        ori_pred = self.model.custom_forward(handle_fn)
        ori_pred = ori_pred[self.mapping_node_id()]
        target = ori_pred.argmax().item()

        perturb_score_all = []

        mask_all = []
        perturb_position = []

        for i in range(num_samples):
            features_perturbed = copy.deepcopy(features)
            # positions = torch.nonzero(features)
            # generate a mask that decides which features to be perturbed
            # since torch cannot create it with weights, we use numpy to create it
            mask = torch.from_numpy(
                np.random.choice([False, True], size=features.shape[0],
                                 p=[1 - p_perturb, p_perturb])).to(
                self.device_string)
            mask_all.append(mask)
            num_selected_perturbed = torch.count_nonzero(mask).item()
            # generate the perturbed features
            perturbed_features = torch.randint(0, 2, (
                num_selected_perturbed, features.shape[1])).to(
                self.device_string)
            perturbed_features = perturbed_features.to(dtype=features.dtype)
            tmp = features[mask] != perturbed_features
            perturb_position.append(tmp)
            features_perturbed[mask] = perturbed_features

            def handle_fn(model):
                return gs, features_perturbed

            with torch.no_grad():
                perturb_result = self.model.custom_forward(handle_fn)
                perturb_result = perturb_result[self.mapping_node_id()]
                perturb_score = perturb_result[target].item()
            perturb_score_all.append(perturb_score)

        perturb_score_all = np.array(perturb_score_all)
        perturb_range = perturb_score_all.max() - perturb_score_all.min()

        if perturb_range == 0:
            print("Warning: GNN prediction is not changed after perturbation")
            return

        pred_threshold = self.config.get('pred_threshold', 0.01)
        if perturb_range < pred_threshold:
            pred_threshold /= 2

        perturb_cap = np.where(
            perturb_score_all <= (perturb_score_all.min() + pred_threshold), 0,
            perturb_score_all)
        perturb_cap = np.where(perturb_cap >= (perturb_cap.max() - pred_threshold), 2,
                               perturb_cap)
        perturb_cap = np.where((perturb_cap > 0) & (perturb_cap < 2), 1, perturb_cap)
        perturb_cap = perturb_cap.astype(int)

        # count the number of 0, 1, 2
        perturb_count = np.bincount(perturb_cap)

        bar = 0.001

        how_many_more = np.where(bar * num_samples - perturb_count, 0,
                                 np.ceil(bar * num_samples - perturb_count)).astype(int)

        if np.sum(how_many_more) > 0:
            # if the number of 0, 1, 2 is not enough, we need to generate more samples
            to_substitute = np.argmax(perturb_count)
            step = 0
            if to_substitute == 0 or to_substitute == 2:
                p_perturb = self.adjust_p_perturb(np.argmin(perturb_count),
                                                  perturb_position, perturb_cap)
            while np.sum(how_many_more) > 0 and step < 2 * num_samples:
                features_perturbed = copy.deepcopy(features)
                mask = torch.from_numpy(
                    np.random.choice([False, True], size=features.shape[0],
                                     p=[1 - p_perturb, p_perturb])).to(
                    self.device_string)
                # mask_all.append(mask)
                num_selected_perturbed = torch.count_nonzero(mask).item()
                perturbed_features = torch.randint(0, 2, (
                    num_selected_perturbed, features.shape[1])).to(
                    self.device_string)
                perturbed_features = perturbed_features.to(dtype=features.dtype)
                tmp = features[mask] != perturbed_features
                # perturb_position.append(tmp)
                features_perturbed[mask] = perturbed_features

                def handle_fn(model):
                    return gs, features_perturbed

                with torch.no_grad():
                    perturb_result = self.model.custom_forward(handle_fn)
                    perturb_result = perturb_result[self.mapping_node_id()]
                    perturb_score = perturb_result[target].item()

                if how_many_more[0] > 0 and perturb_score <= (
                    perturb_score_all.min() + pred_threshold):
                    # change one of to_substitute to 0
                    can_be_changed = np.where(perturb_cap == to_substitute)[0]
                    idx = np.random.choice(can_be_changed)
                    perturb_cap[idx] = 0
                    mask_all[idx] = mask
                    how_many_more[0] -= 1
                    perturb_position[idx] = tmp

                elif how_many_more[1] > 0 and perturb_score > (
                    perturb_score_all.min() + pred_threshold) and perturb_score < (
                    perturb_score_all.max() - pred_threshold):
                    # change one of to_substitute to 1
                    can_be_changed = np.where(perturb_cap == to_substitute)[0]
                    idx = np.random.choice(can_be_changed)
                    perturb_cap[idx] = 1
                    mask_all[idx] = mask
                    how_many_more[1] -= 1
                    perturb_position[idx] = tmp

                elif how_many_more[2] > 0 and perturb_score >= (
                    perturb_score_all.max() - pred_threshold):
                    # change one of to_substitute to 2
                    can_be_changed = np.where(perturb_cap == to_substitute)[0]
                    idx = np.random.choice(can_be_changed)
                    perturb_cap[idx] = 2
                    mask_all[idx] = mask
                    how_many_more[2] -= 1
                    perturb_position[idx] = tmp

                step += 1

        return perturb_cap, mask_all, perturb_position

    def adjust_p_perturb(self, perturb_min, perturb_position, perturb_cap):

        non_min_indices = np.where(perturb_cap != perturb_min)[0]
        selected_positions = [perturb_position[idx] for idx in non_min_indices]
        selected_positions_num = len(
            [1 for positions in selected_positions if np.count_nonzero(positions) > 0])
        return selected_positions_num / len(selected_positions)

    def mapping_node_id(self):
        if getattr(self, 'mapped_node_id', None) is not None:
            return self.mapped_node_id
        if not self.config.get('extract_neighbors', True):
            self.mapped_node_id = self.node_id
        else:
            self.mapped_node_id = self.recovery_dict[self.node_id]
        return self.mapped_node_id

    def extract_neighbors_input(self):
        """
        Extract the neighbors of the node to be explained
        :return:
        """
        # the sample number of hencex highly depends on the number of nodes
        # Therefore, we suggests to set it to True to avoid too many samples
        if not self.config.get('extract_neighbors', True):
            gs, features = self.model.standard_input()
            self.neighbor_input = {"gs": gs, "features": features}
            return gs, features

        if getattr(self, 'neighbor_input',
                   None) is not None and self.neighbor_input.get(
            "gs", None) is not None:
            return self.neighbor_input["gs"], self.neighbor_input["features"]

        # we follow the default value in hencex
        self.n_hop = self.config.get('n_hop', 2)

        gs, features = self.model.standard_input()

        used_nodes_set = set()

        for g in gs:
            indices = g.indices()
            # mask = indices[0] == self.node_id

            # consider memory-efficient
            current_nodes = [self.node_id]

            for i in range(self.n_hop):
                new_current_nodes = []
                for node in current_nodes:
                    mask = indices[0] == node
                    used_nodes_set.update(indices[1][mask].tolist())
                    new_current_nodes.extend(indices[1][mask].tolist())

                current_nodes = new_current_nodes

        self.used_nodes = sorted(list(used_nodes_set))
        self.recovery_dict = {node: i for i, node in enumerate(self.used_nodes)}

        # now reconstruct the graph
        temp_used_nodes_tensor = torch.tensor(self.used_nodes).to(self.device_string)
        new_gs = []
        for g in gs:
            indices = g.indices()
            # !TODO: Test it in the future, and then expand it to other algorithms
            mask = torch.isin(indices[0], torch.tensor(temp_used_nodes_tensor,
                                                       device=self.device_string)) & \
                   torch.isin(indices[1], torch.tensor(temp_used_nodes_tensor,
                                                       device=self.device_string))
            new_indices = torch.stack(
                [torch.tensor(
                    [self.recovery_dict[node.item()] for node in indices[0][mask]]),
                 torch.tensor(
                     [self.recovery_dict[node.item()] for node in indices[1][mask]])],
                dim=0)
            new_indices = new_indices.to(self.device_string)
            new_values = g.values()[mask]
            shape = torch.Size([len(self.used_nodes), len(self.used_nodes)])
            new_gs.append(torch.sparse_coo_tensor(new_indices, new_values, shape))

        self.neighbor_input = {"gs": new_gs, "features": features[self.used_nodes]}
        return self.neighbor_input["gs"], self.neighbor_input["features"]

    def construct_explanation(self):
        explanation = NodeExplanation()
        explanation = standard_explanation(explanation, self)
        for metric in self.config['eval_metrics']:
            prepare_explanation_fn_for_node_dataset_scores[metric](explanation, self)
        self.explanation = explanation
        return explanation

    def visualize(self):
        # !TODO: finish it, but not now
        pass

    @property
    def edge_mask_for_output(self):
        return None

    @property
    def feature_mask_for_output(self):
        pass

    def get_custom_input_handle_fn(self, masked_gs=None, feature_mask=None):
        pass
