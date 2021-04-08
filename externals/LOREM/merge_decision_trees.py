from sklearn.tree import DecisionTreeClassifier

from externals.LOREM.rule import Rule, Condition, get_rules
from externals.LOREM.decision_tree import prune_duplicate_leaves
from externals.LOREM.decision_table import DecisionTable, DecisionTableRegion, weight_attribution


def merge_decision_trees(dt_list, X, Y, feature_names, class_name, class_values, numeric_columns,
                         ratio_thr=1.0, weight_fun='avg', conflict_fun='max', coverage_thr=0.01, precision_thr=0.6):
    dtab_list = decision_trees2decision_tables(dt_list, X, Y, feature_names, class_name, class_values, numeric_columns)
    dt = merge_models(dtab_list, X, Y, ratio_thr, weight_fun, conflict_fun, coverage_thr, precision_thr,
                      type='sample', size=1000, random_state=None)
    if dt is None:
        dt = dt_list[0]
    return dt


def intersection(m1, m2, X, Y, ratio_thr=1.0, weight_fun='avg', conflict_fun='max'):
    m = m1.merge(m2, X=X, ratio_thr=ratio_thr, weight_fun=weight_fun, conflict_fun=conflict_fun)
    m.calculate_coverage_precision(X, Y)
    return m


def filtering(m, X, Y, coverage_thr=0.01, precision_thr=0.6):
    if coverage_thr is None and precision_thr is None:
        return m
    dtr_list = list()
    for dtr in m.table:
        if coverage_thr is not None and precision_thr is not None:
            if dtr.coverage > coverage_thr and dtr.precision > precision_thr:
                dtr_list.append(dtr)
        elif coverage_thr is not None and precision_thr is None:
            if dtr.coverage > coverage_thr:
                dtr_list.append(dtr)
        elif coverage_thr is None and precision_thr is not None:
            if dtr.precision > precision_thr:
                dtr_list.append(dtr)
    m.set_table(dtr_list)
    m.remove_redundancies()
    m.calculate_coverage_precision(X, Y)
    return m


def reduction(m, X, Y, ratio_thr=1.0, weight_fun='avg'):
    dtr_list = m.table[:]
    there_is_a_change = True
    while there_is_a_change:
        there_is_a_change = False
        if len(dtr_list) == 1:
            break
        for i in range(len(dtr_list) - 1):
            dtr1 = dtr_list[i]
            for j in range(i + 1, len(dtr_list)):
                dtr2 = dtr_list[j]
                if dtr1.label == dtr2.label and set(dtr1.attr_op.keys()) == set(dtr2.attr_op.keys()):
                    premises = list()
                    for att_op in dtr1.attr_op:
                        c1 = dtr1.attr_op[att_op]
                        c2 = dtr2.attr_op[att_op]
                        att, op = att_op
                        if c1.op == '<=':
                            thr = max(c1.thr, c2.thr)
                            if (thr - min(c1.thr, c2.thr)) / thr > ratio_thr:
                                break
                        else:
                            thr = min(c1.thr, c2.thr)
                            if (max(c1.thr, c2.thr) - thr) / max(c1.thr, c2.thr) > ratio_thr:
                                break
                        c = Condition(att, op, thr)
                        premises.append(c)

                    rm = Rule(premises, dtr1.label, dtr1.class_name)
                    weight = weight_attribution(dtr1, dtr2, weight_fun)
                    merged_dtr = DecisionTableRegion(rm, dtr1.feature_names, dtr1.class_values, dtr1.class_name,
                                                     idx='%s-%s' % (dtr1.idx, dtr2.idx))
                    merged_dtr.set_weight(weight)
                    dtr_list.pop(i)
                    dtr_list.pop(j-1)
                    dtr_list.append(merged_dtr)
                    there_is_a_change = True
                if there_is_a_change:
                    break
            if there_is_a_change:
                break

    m.set_table(dtr_list)
    m.remove_redundancies()
    m.calculate_coverage_precision(X, Y)
    return m


def merge_two_models(m1, m2, X, Y, ratio_thr=1.0, weight_fun='avg', conflict_fun='max',
                     coverage_thr=0.01, precision_thr=0.6):
    # print('intersection')
    intersected_merged_model = intersection(m1, m2, X, Y, ratio_thr, weight_fun, conflict_fun)
    # print('filtering')
    filtered_merged_model = filtering(intersected_merged_model, X, Y, coverage_thr, precision_thr)
    # filtered_merged_model = intersected_merged_model
    # print('reduction')
    merged_model = reduction(filtered_merged_model, X, Y, ratio_thr, weight_fun)
    # merged_model = filtered_merged_model
    return merged_model


def merge_models(dt_list, X, Y, ratio_thr=1.0, weight_fun='avg', conflict_fun='max',
                 coverage_thr=0.01, precision_thr=0.6, type='sample', size=1000, random_state=None):
    m = dt_list[0]
    for i in range(1, len(dt_list)):
        # print('merge', i)
        m1 = dt_list[i]
        # print('M1')
        # print(m)
        # print('M2')
        # print(m1)
        m = merge_two_models(m, m1, X, Y, ratio_thr, weight_fun, conflict_fun, coverage_thr, precision_thr)
        # print('New M')
        # count_cons = defaultdict(int)
        # for t in m.table:
        #     count_cons[t.label] += 1
        # print(count_cons)
        # print(m)
        # print('---------------\n')
        # break
    # print('Final')
    # print(m)
    Z, Y = m.generate_data(X, type=type, size=size)

    dt = DecisionTreeClassifier(max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=random_state)
    # print(len(Z), np.unique(Y, return_counts=True))
    if len(Z) > 1:
        dt.fit(Z, Y)
    elif len(Z) == 1:
        dt.fit(Z.reshape(1, -1), Y)
    else:
        return None
    prune_duplicate_leaves(dt)
    return dt


def decision_trees2decision_tables(decision_trees, X, Y, feature_names, class_name, class_values, numeric_columns):
    decision_tables = list()
    for dt in decision_trees:
        rules = get_rules(dt, feature_names, class_name, class_values, numeric_columns, multi_label=False)
        m = DecisionTable(rules=rules, feature_names=feature_names,
                          class_values=class_values, class_name=class_name, X=X)
        m.calculate_coverage_precision(X, Y)
        decision_tables.append(m)
    return decision_tables
