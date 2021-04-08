import numpy as np
from collections import defaultdict

from externals.LOREM.rule import Rule, Condition


def weight_attribution(dtr1, dtr2, weight_fun):
    if weight_fun == 'avg':
        weight = (dtr1.weight + dtr2.weight) / 2
    elif weight_fun == 'max':
        weight = max(dtr1.weight, dtr2.weight)
    elif weight_fun == 'min':
        weight = min(dtr1.weight, dtr2.weight)
    else:
        weight = (dtr1.weight + dtr2.weight) / 2
    return weight


class DecisionTableRegion(object):
    def __init__(self, rule, feature_names, class_values, class_name, idx=None):
        self.idx = idx
        self.rule = rule
        self.feature_names = feature_names
        self.class_values = class_values
        self.class_name = class_name
        self.label = rule.cons
        self.attr_op = {(c.att, c.op): c for c in rule.premises}
        self.feature_values = defaultdict(list)
        for c in rule.premises:
            self.feature_values[c.att].append(c)
        self.weight = 0
        self.coverage_count = 0
        self.precision_count = 0
        self.coverage = 0
        self.precision = 0
        self.coverage_set = set()

    def __str__(self):
        return str(self.rule) + ' w: %.2f %.2f %.2f - idx: %s' % (
            self.weight, self.coverage, self.precision, self.idx)

    def __eq__(self, other):
        return self.rule == other.rule

    def __hash__(self):
        return hash(str(self))

    def calculate_and_set_weight(self, X):
        for i, x in enumerate(X):
            if self.rule.is_covered(x, self.feature_names):
                self.weight += 1

    def set_weight(self, weight):
        self.weight = weight

    def calculate_coverage_precision(self, X, Y):
        self.coverage_count = 0
        self.precision_count = 0
        self.coverage_set = set()
        for i, x in enumerate(X):
            if self.rule.is_covered(x, self.feature_names):
                self.coverage_count += 1
                self.coverage_set.add(i)
                if self.class_values[Y[i]] == self.label:
                    self.precision_count += 1
        self.coverage = self.coverage_count / len(X)
        self.precision = self.precision_count / self.coverage_count if self.coverage_count > 0 else 0.0

    def merge_regions(self, dtr, ratio_thr=1.0, weight_fun='avg', conflict_fun='max'):
        all_features = list(set(self.feature_values.keys()) | set(dtr.feature_values.keys()))
        for feature in all_features:
            if feature in self.feature_values and feature in dtr.feature_values:
                for c1 in self.feature_values[feature]:
                    for c2 in dtr.feature_values[feature]:
                        if c1.op == '>' and c2.op == '<=' and c1.thr > c2.thr:
                            return None
                        elif c2.op == '>' and c1.op == '<=' and c2.thr > c1.thr:
                            return None
        # not disjoint
        all_attr_op = list(set(self.attr_op.keys()) | set(dtr.attr_op.keys()))

        premises = list()
        for att_op in all_attr_op:
            if att_op not in self.attr_op:
                premises.append(dtr.attr_op[att_op])
            elif att_op not in dtr.attr_op:
                premises.append(self.attr_op[att_op])
            else:
                c1 = self.attr_op[att_op]
                c2 = dtr.attr_op[att_op]
                att, op = att_op
                if c1.op == '>':
                    thr = max(c1.thr, c2.thr)
                    if (thr - min(c1.thr, c2.thr)) / thr > ratio_thr:
                        return None
                else:
                    thr = min(c1.thr, c2.thr)
                    if (max(c1.thr, c2.thr) - thr) / max(c1.thr, c2.thr) > ratio_thr:
                        return None
                c = Condition(att, op, thr)
                premises.append(c)

        if self.label == dtr.label:
            new_label = self.label
        else:
            # print(self.label, dtr.label, self.label == dtr.label, self.weight, dtr.weight, self.weight >= dtr.weight)
            if conflict_fun == 'max':
                new_label = self.label if self.weight >= dtr.weight else dtr.label
            elif conflict_fun == 'min':
                new_label = self.label if self.weight < dtr.weight else dtr.label
            else:
                new_label = self.label if self.weight >= dtr.weight else dtr.label

        cons = new_label
        class_name = self.rule.class_name
        rm = Rule(premises, cons, class_name)
        weight = weight_attribution(self, dtr, weight_fun)
        merged_dtr = DecisionTableRegion(rm, self.feature_names, self.class_values, self.class_name,
                                         idx='%s-%s' % (self.idx, dtr.idx))
        # merged_dtr.calculate_and_set_weight(X, Y)
        merged_dtr.set_weight(weight)
        return merged_dtr

    def generate_data(self, X, nbr_samples=1000, type='sample'):
        feature_upper_limits = np.max(X, axis=0)
        feature_lower_limits = np.min(X, axis=0)
        # print(feature_lower_limits)
        # print(feature_upper_limits)

        for f_idx, f in enumerate(self.feature_names):
            if (f, '<=') in self.attr_op:
                feature_upper_limits[f_idx] = self.attr_op[(f, '<=')].thr
            elif (f, '>') in self.attr_op:
                feature_lower_limits[f_idx] = self.attr_op[(f, '>')].thr

        # print('---')
        # print(feature_lower_limits)
        # print(feature_upper_limits)

        Z = list()
        if type == 'limits':
            Z.append(feature_lower_limits)
            Z.append(feature_upper_limits)
        elif type == 'sample':
            for _ in range(nbr_samples):
                z = np.zeros(len(self.feature_names))
                for i in range(len(self.feature_names)):
                    lb = feature_lower_limits[i]
                    ub = feature_upper_limits[i]
                    z[i] = np.random.uniform(lb, ub, 1)[0]
                    # print(i, z[i], lb, ub)
                Z.append(z)

        # Z = np.array(Z)
        Y = [self.class_values.index(self.label)] * len(Z)
        return Z, Y


class DecisionTable(object):

    def __init__(self, table=None, rules=None, feature_names=None, class_values=None, class_name=None, X=None):
        self.table = list()
        self.feature_names = feature_names
        self.class_values = class_values
        self.class_name = class_name
        if table is None and rules is not None:
            for idx, r in enumerate(rules):
                dtr = DecisionTableRegion(rule=r, feature_names=feature_names, class_values=class_values,
                                          class_name=class_name, idx=idx)
                dtr.calculate_and_set_weight(X)
                self.table.append(dtr)
        elif table is not None:
            for dtr in table:
                self.table.append(dtr)
        self.normalize_weights()

    def __str__(self):
        s = '\n'.join([str(dtr) for dtr in self.table])
        return s

    def set_table(self, table):
        self.table = table

    def normalize_weights(self):
        current_weights = [dtr.weight for dtr in self.table]
        tot_weights = np.sum(current_weights)
        new_weights = [cw/tot_weights for cw in current_weights]
        for dtr, nw in zip(self.table, new_weights):
            dtr.weight = nw

    def merge(self, dt, X=None, ratio_thr=1.0, weight_fun='avg', conflict_fun='max'):
        merged_dtr_list = list()
        for dtr1 in self.table:
            for dtr2 in dt.table:
                merged_dtr = dtr1.merge_regions(dtr2, ratio_thr, weight_fun, conflict_fun)
                if merged_dtr is not None:
                    merged_dtr_list.append(merged_dtr)

        merged_dt = DecisionTable(table=merged_dtr_list, rules=None, feature_names=self.feature_names,
                                  class_values=self.class_values, class_name=self.class_name, X=X)
        return merged_dt

    def remove_redundancies(self):
        filtered_dtr = list()
        for dtr in self.table:
            if dtr not in filtered_dtr:
                filtered_dtr.append(dtr)
        self.table = filtered_dtr

    def calculate_coverage_precision(self, X, Y):
        for dtr in self.table:
            dtr.calculate_coverage_precision(X, Y)

    def generate_data(self, X, size=1000, type='sample'):
        Z = list()
        Y = list()
        self.normalize_weights()
        for dtr in self.table:
            nbr_samples = int(np.round(dtr.weight * size))
            Zi, Yi = dtr.generate_data(X, nbr_samples, type=type)
            Z.extend(Zi)
            Y.extend(Yi)

        Z = np.array(Z)
        Y = np.array(Y)
        return Z, Y

