import copy
import json
import numpy as np

from ..ilore.util import vector2dict
from collections import defaultdict


class Condition(object):

    def __init__(self, att, op, thr, is_continuous=True):
        self.att = att
        self.op = op
        self.thr = thr
        self.is_continuous = is_continuous

    def __str__(self):
        if self.is_continuous:
            return '%s %s %.2f' % (self.att, self.op, self.thr)
        else:
            att_split = self.att.split('=')
            sign = '=' if self.op == '>' else '!='
            return '%s %s %s' % (att_split[0], sign, att_split[1])

    def __eq__(self, other):
        return self.att == other.att and self.op == other.op and self.thr == other.thr

    def __hash__(self):
        return hash(str(self))


class Rule(object):

    def __init__(self, premises, cons, class_name):
        self.premises = premises
        self.cons = cons
        self.class_name = class_name

    def _pstr(self):
        return '{ %s }' % (', '.join([str(p) for p in self.premises]))

    def _cstr(self):
        if not isinstance(self.class_name, list):
            return '{ %s: %s }' % (self.class_name, self.cons)
        else:
            return '{ %s }' % self.cons

    def __str__(self):
        return '%s --> %s' % (self._pstr(), self._cstr())

    def __eq__(self, other):
        return self.premises == other.premises and self.cons == other.cons

    def __len__(self):
        return len(self.premises)

    def __hash__(self):
        return hash(str(self))

    def is_covered(self, x, feature_names):
        xd = vector2dict(x, feature_names)
        for p in self.premises:
            if p.op == '<=' and xd[p.att] > p.thr:
                return False
            elif p.op == '>' and xd[p.att] <= p.thr:
                return False
        return True


def json2cond(obj):
    return Condition(obj['att'], obj['op'], obj['thr'], obj['is_continuous'])


def json2rule(obj):
    premises = [json2cond(p) for p in obj['premise']]
    cons = obj['cons']
    class_name = obj['class_name']
    return Rule(premises, cons, class_name)


class ConditionEncoder(json.JSONEncoder):
    """ Special json encoder for Condition types """
    def default(self, obj):
        if isinstance(obj, Condition):
            json_obj = {
                'att': obj.att,
                'op': obj.op,
                'thr': obj.thr,
                'is_continuous': obj.is_continuous,
            }
            return json_obj
        return json.JSONEncoder.default(self, obj)


class RuleEncoder(json.JSONEncoder):
    """ Special json encoder for Rule types """
    def default(self, obj):
        if isinstance(obj, Rule):
            ce = ConditionEncoder()
            json_obj = {
                'premise': [ce.default(p) for p in obj.premises],
                'cons': obj.cons,
                'class_name': obj.class_name
            }
            return json_obj
        return json.JSONEncoder.default(self, obj)


def get_rule(x, dt, feature_names, class_name, class_values, numeric_columns):

    x = x.reshape(1, -1)
    feature = dt.tree_.feature
    threshold = dt.tree_.threshold

    leave_id = dt.apply(x)
    node_index = dt.decision_path(x).indices

    premises = list()
    for node_id in node_index:
        if leave_id[0] == node_id:
            break
        else:
            op = '<=' if x[0][feature[node_id]] <= threshold[node_id] else '>'
            att = feature_names[feature[node_id]]
            thr = threshold[node_id]
            iscont = att in numeric_columns
            premises.append(Condition(att, op, thr, iscont))

    dt_outcome = dt.predict(x)[0]
    cons = class_values[int(dt_outcome)]
    premises = compact_premises(premises)
    return Rule(premises, cons, class_name)


def get_rules(dt, feature_names, class_name, class_values, numeric_columns):

    n_nodes = dt.tree_.node_count
    feature = dt.tree_.feature
    threshold = dt.tree_.threshold
    children_left = dt.tree_.children_left
    children_right = dt.tree_.children_right
    value = dt.tree_.value

    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    reverse_dt_dict = dict()
    left_right = dict()
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()

        # If we have a test node
        if children_left[node_id] != children_right[node_id]:
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
            reverse_dt_dict[children_left[node_id]] = node_id
            left_right[(node_id, children_left[node_id])] = 'l'
            reverse_dt_dict[children_right[node_id]] = node_id
            left_right[(node_id, children_right[node_id])] = 'r'
        else:
            is_leaves[node_id] = True

    node_index_list = list()
    for node_id in range(n_nodes):
        if is_leaves[node_id]:
            node_index = [node_id]
            parent_node = reverse_dt_dict.get(node_id, None)
            while parent_node:
                node_index.insert(0, parent_node)
                parent_node = reverse_dt_dict.get(parent_node, None)
            if node_index[0] != 0:
                node_index.insert(0, 0)
            node_index_list.append(node_index)

    if len(value) > 1:
        value = np.argmax(value.reshape(len(value), 2), axis=1)

        rules = list()
        for node_index in node_index_list:

            premises = list()
            for i in range(len(node_index) - 1):
                node_id = node_index[i]
                child_id = node_index[i+1]

                op = '<=' if left_right[(node_id, child_id)] == 'l' else '>'
                att = feature_names[feature[node_id]]
                thr = threshold[node_id]
                iscont = att in numeric_columns
                premises.append(Condition(att, op, thr, iscont))

            cons = class_values[int(value[node_index[-1]])]
            premises = compact_premises(premises)
            rules.append(Rule(premises, cons, class_name))

    else:
        x = np.zeros(len(feature_names)).reshape(1, -1)
        dt_outcome = dt.predict(x)[0]
        cons = class_values[int(dt_outcome)]
        rules = [Rule([], cons, class_name)]
    return rules


def compact_premises(plist):
    att_list = defaultdict(list)
    for p in plist:
        att_list[p.att].append(p)

    compact_plist = list()
    for att, alist in att_list.items():
        if len(alist) > 1:
            min_thr = None
            max_thr = None
            for av in alist:
                if av.op == '<=':
                    max_thr = min(av.thr, max_thr) if max_thr else av.thr
                elif av.op == '>':
                    min_thr = max(av.thr, min_thr) if min_thr else av.thr

            if max_thr:
                compact_plist.append(Condition(att, '<=', max_thr))

            if min_thr:
                compact_plist.append(Condition(att, '>', min_thr))
        else:
            compact_plist.append(alist[0])
    return compact_plist


def get_counterfactual_rules(x, y, dt, Z, Y, feature_names, class_name, class_values, numeric_columns, autoencoder,
                             bb_predict=None, eps=0.01):
    clen = np.inf
    crule_list = list()
    delta_list = list()
    Z1 = Z[np.where(Y != y)[0]]
    xd = vector2dict(x, feature_names)
    for z in Z1:
        crule = get_rule(z, dt, feature_names, class_name, class_values, numeric_columns)
        delta, qlen = get_falsified_conditions(xd, crule)
        if qlen < clen:
            clen = qlen
            crule_list = [crule]
            delta_list = [delta]
        elif qlen == clen:
            if delta not in delta_list:
                crule_list.append(crule)
                delta_list.append(delta)

    if bb_predict is not None:
        cleaned_crules = list()
        cleaned_deltas = list()
        for crule, delta in zip(crule_list, delta_list):
            xc = apply_counterfactual(x, delta, feature_names, eps)

            img_xc = autoencoder.decode(xc.reshape(1, -1))[0]

            bb_outcomec = bb_predict(np.array([img_xc]))[0]
            bb_outcomec = class_values[bb_outcomec]
            dt_outcomec = crule.cons
            if bb_outcomec == dt_outcomec:
                cleaned_crules.append(crule)
                cleaned_deltas.append(delta)

        crule_list = cleaned_crules
        delta_list = cleaned_deltas

    return crule_list, delta_list


def get_falsified_conditions(xd, crule):
    delta = list()
    nbr_falsified_conditions = 0
    for p in crule.premises:
        if p.op == '<=' and xd[p.att] > p.thr:
            delta.append(p)
            nbr_falsified_conditions += 1
        elif p.op == '>' and xd[p.att] <= p.thr:
            delta.append(p)
            nbr_falsified_conditions += 1
    return delta, nbr_falsified_conditions


def apply_counterfactual(x, delta, feature_names, eps=0.01):
    xd = vector2dict(x, feature_names)
    xcd = copy.deepcopy(xd)
    for p in delta:
        if p.op == '>':
            xcd[p.att] = p.thr + eps
        else:
            xcd[p.att] = p.thr - eps

    xc = np.zeros(len(xd))
    for i, fn in enumerate(feature_names):
        xc[i] = xcd[fn]

    return xc


