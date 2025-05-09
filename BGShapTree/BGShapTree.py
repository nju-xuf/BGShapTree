from scipy.special import comb
from copy import deepcopy
from numba import jit, njit, prange
from datetime import datetime
import numba as nb
import joblib
import os
from tqdm import tqdm
import multiprocessing
import random
import numpy as np
import time
import copy


def ctime():
    """A formatter on current time used for printing running status."""
    ctime = "[" + datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "]"
    return ctime


@njit()
def FastPrepare(d, m, depth, leaves_info, x, baseline_x):
    temp_sets = np.zeros((m, 2, depth))  # store set_d and set_e
    temp_supp_info = np.zeros((m, 5))
    leaves_counter = 0
    len_set_a = 0
    len_set_b = 0
    len_set_c = 0
    counter_d = 0
    counter_e = 0
    path_length = 0
    for i in range(len(leaves_info)):
        leaves_num = leaves_info[i][0]
        if leaves_counter > leaves_num:
            continue
        temp_f = int(leaves_info[i][1])
        temp_l = leaves_info[i][2]
        temp_r = leaves_info[i][3]
        f_1, f_2 = False, False
        if temp_l < x[temp_f] <= temp_r:
            len_set_a += 1
            f_1 = True
        if temp_l < baseline_x[temp_f] <= temp_r:
            len_set_b += 1
            f_2 = True
        if not f_1 and not f_2:
            leaves_counter += 1
            len_set_a = 0
            len_set_b = 0
            len_set_c = 0
            counter_d = 0
            counter_e = 0
            path_length = 0
            continue
        if f_1 and f_2:
            len_set_c += 1
        if f_1 and not f_2:
            temp_sets[leaves_counter][0][counter_d] = temp_f
            counter_d += 1
        if not f_1 and f_2:
            temp_sets[leaves_counter][1][counter_e] = temp_f
            counter_e += 1
        path_length += 1
        # check whether to record the supp-info
        check_flag = False
        if i == len(leaves_info) - 1:
            check_flag = True
        if i < len(leaves_info) - 1:
            if leaves_info[i + 1][0] != leaves_info[i][0]:
                check_flag = True
        if check_flag:
            temp_supp_info[leaves_counter][0] = 1
            temp_supp_info[leaves_counter][1] = d - path_length
            temp_supp_info[leaves_counter][2] = len_set_a
            temp_supp_info[leaves_counter][3] = len_set_b
            temp_supp_info[leaves_counter][4] = len_set_c
            for j in range(counter_d, depth):
                temp_sets[leaves_counter][0][j] = -1
            for j in range(counter_e, depth):
                temp_sets[leaves_counter][1][j] = -1
            leaves_counter += 1
            # initialize related information
            len_set_a = 0
            len_set_b = 0
            len_set_c = 0
            counter_d = 0
            counter_e = 0
            path_length = 0
    return temp_sets, temp_supp_info


@njit()
def SetValuation(d, leaves_info, leaves_result, x, baseline_x, p, category, T):
    category = int(category)
    ans = 0
    path_length = 0
    n = len(leaves_info)
    leaves_counter = 0
    # initialize some lists
    set_a = nb.typed.List()
    set_a.append(1)
    set_a.clear()
    set_b = nb.typed.List()
    set_b.append(1)
    set_b.clear()
    set_c = nb.typed.List()
    set_c.append(1)
    set_c.clear()
    set_d = nb.typed.List()
    set_d.append(1)
    set_d.clear()
    set_e = nb.typed.List()
    set_e.append(1)
    set_e.clear()
    # scan over all of the leaves
    for j in range(n):
        leaves_num = leaves_info[j][0]
        if leaves_num != leaves_counter:
            continue
        if leaves_result[leaves_counter][category] == 0:
            set_a.clear()
            set_b.clear()
            set_c.clear()
            set_d.clear()
            set_e.clear()
            path_length = 0
            leaves_counter += 1
            continue
        path_length += 1
        temp_f = int(leaves_info[j][1])
        temp_l = leaves_info[j][2]
        temp_r = leaves_info[j][3]
        f_1 = False
        f_2 = False
        if temp_l < x[temp_f] <= temp_r:
            set_a.append(temp_f)
            f_1 = True
        if temp_l < baseline_x[temp_f] <= temp_r:
            set_b.append(temp_f)
            f_2 = True
        if not f_1 and not f_2:
            set_a.clear()
            set_b.clear()
            set_c.clear()
            set_d.clear()
            set_e.clear()
            path_length = 0
            leaves_counter += 1
            continue
        if f_1 and f_2:
            set_c.append(temp_f)
        elif f_1 and not f_2:
            set_d.append(temp_f)
        else:
            set_e.append(temp_f)
        # check the calculation state
        calculation_flag = False
        if j == n - 1:
            calculation_flag = True
        if j < n - 1:
            if leaves_info[j + 1][0] != leaves_info[j][0]:
                calculation_flag = True
        if calculation_flag:
            # number of features that are not on the current decision path
            delta = d - path_length
            # Second check
            counter_d = 0
            counter_e = 0
            counter_c = 0
            flag = True
            for feature in T:
                feature = int(feature)
                if feature in set_d:
                    counter_d += 1
                elif feature in set_e:
                    counter_e += 1
                else:
                    counter_c += 1
                if counter_d > 0 and counter_e > 0:
                    flag = False
                    break
            if flag:
                # calculation phase
                value_of_node = leaves_result[leaves_counter][category]
                if counter_d > 0:
                    temp = p[len(set_c) + delta - counter_c][len(set_a) - len(T) + delta + 1]
                    ans = ans + value_of_node * temp
                if counter_e > 0:
                    temp = p[len(set_c) + delta - counter_c][len(set_b) - len(T) + delta + 1]
                    ans = ans - value_of_node * temp
            set_a.clear()
            set_b.clear()
            set_c.clear()
            set_d.clear()
            set_e.clear()
            path_length = 0
            leaves_counter += 1
    # print(np.sum(results, axis=1)) #  test the correctness of the results
    # return leaves_importance, results
    return ans


@njit()
def SetValuationSS(d, sets, supp_info, leaves_result, p, category, T):
    category = int(category)
    ans = 0
    m = len(sets)
    for j in prange(m):
        valid = supp_info[j][0]
        delta = supp_info[j][1]
        len_set_a = supp_info[j][2]
        len_set_b = supp_info[j][3]
        len_set_c = supp_info[j][4]
        # if A_j \cup B_j = L_j
        if valid == 1 and leaves_result[j][category] > 0:
            counter_d = 0
            counter_e = 0
            counter_c = 0
            flag = True
            for feature in T:
                feature = int(feature)
                if feature in sets[j][0]:
                    counter_d += 1
                elif feature in sets[j][1]:
                    counter_e += 1
                else:
                    counter_c += 1
                if counter_d > 0 and counter_e > 0:
                    flag = False
                    break
            if flag:
                # calculation phase
                value_of_node = leaves_result[j][category]
                if counter_d > 0:
                    temp = p[int(len_set_c + delta - counter_c)][int(len_set_a - len(T) + delta + 1)]
                    ans = ans + value_of_node * temp
                if counter_e > 0:
                    temp = p[int(len_set_c + delta - counter_c)][int(len_set_b - len(T) + delta + 1)]
                    ans = ans - value_of_node * temp
    return ans


@njit()
def FastBaselineShapleyCategory(T, d, leaves_info, leaves_result, x, baseline_x, p, category):
    results = np.zeros((T, d))
    path_length = 0
    n = len(leaves_info)
    leaves_counter = 0
    set_a = nb.typed.List()
    set_a.append(1)
    set_a.clear()
    set_b = nb.typed.List()
    set_b.append(1)
    set_b.clear()
    set_c = nb.typed.List()
    set_c.append(1)
    set_c.clear()
    set_d = nb.typed.List()
    set_d.append(1)
    set_d.clear()
    set_e = nb.typed.List()
    set_e.append(1)
    set_e.clear()
    for j in range(n):
        leaves_num = leaves_info[j][0]
        if leaves_num != leaves_counter:
            continue
        if leaves_result[leaves_counter][category] == 0:
            set_a.clear()
            set_b.clear()
            set_c.clear()
            set_d.clear()
            set_e.clear()
            path_length = 0
            leaves_counter += 1
            continue
        path_length += 1
        temp_f = int(leaves_info[j][1])
        temp_l = leaves_info[j][2]
        temp_r = leaves_info[j][3]
        f_1 = False
        f_2 = False
        if temp_l < x[temp_f] <= temp_r:
            set_a.append(temp_f)
            f_1 = True
        if temp_l < baseline_x[temp_f] <= temp_r:
            set_b.append(temp_f)
            f_2 = True
        if not f_1 and not f_2:
            set_a.clear()
            set_b.clear()
            set_c.clear()
            set_d.clear()
            set_e.clear()
            path_length = 0
            leaves_counter += 1
            continue
        if f_1 and f_2:
            set_c.append(temp_f)
        elif f_1 and not f_2:
            set_d.append(temp_f)
        else:
            set_e.append(temp_f)
        # check the calculation state
        calculation_flag = False
        if j == n - 1:
            calculation_flag = True
        if j < n - 1:
            if leaves_info[j + 1][0] != leaves_info[j][0]:
                calculation_flag = True
        if calculation_flag:
            delta = d - path_length
            t = len(set_c)
            for i in range(T):  # Consider each category
                if leaves_result[leaves_counter][i] > 0:
                    value_of_node = leaves_result[leaves_counter][i]
                else:
                    continue
                if len(set_a) > t:
                    temp = p[t + delta][len(set_a) + delta]
                    for element in set_d:
                        results[i][element] += temp * value_of_node
                if len(set_b) > t:
                    temp = p[t + delta][len(set_b) + delta]
                    for element in set_e:
                        results[i][element] -= temp * value_of_node
            set_a.clear()
            set_b.clear()
            set_c.clear()
            set_d.clear()
            set_e.clear()
            path_length = 0
            leaves_counter += 1
    # print(np.sum(results, axis=1))
    return results


# Calculate individual feature importance when searching salient sets
@njit()
def FastBaselineShapleyCategorySS(T, d, leaves_info, leaves_result, x, baseline_x, p, category):
    results = np.zeros((T, d))
    path_length = 0
    n = len(leaves_info)
    leaves_counter = 0
    set_a = nb.typed.List()
    set_a.append(1)
    set_a.clear()
    set_b = nb.typed.List()
    set_b.append(1)
    set_b.clear()
    set_c = nb.typed.List()
    set_c.append(1)
    set_c.clear()
    set_d = nb.typed.List()
    set_d.append(1)
    set_d.clear()
    set_e = nb.typed.List()
    set_e.append(1)
    set_e.clear()
    for j in range(n):
        leaves_num = leaves_info[j][0]
        if leaves_num != leaves_counter:
            continue
        if leaves_result[leaves_counter][category] == 0:
            set_a.clear()
            set_b.clear()
            set_c.clear()
            set_d.clear()
            set_e.clear()
            path_length = 0
            leaves_counter += 1
            continue
        path_length += 1
        temp_f = int(leaves_info[j][1])
        temp_l = leaves_info[j][2]
        temp_r = leaves_info[j][3]
        f_1 = False
        f_2 = False
        if temp_l < x[temp_f] <= temp_r:
            set_a.append(temp_f)
            f_1 = True
        if temp_l < baseline_x[temp_f] <= temp_r:
            set_b.append(temp_f)
            f_2 = True
        if not f_1 and not f_2:
            set_a.clear()
            set_b.clear()
            set_c.clear()
            set_d.clear()
            set_e.clear()
            path_length = 0
            leaves_counter += 1
            continue
        if f_1 and f_2:
            set_c.append(temp_f)
        elif f_1 and not f_2:
            set_d.append(temp_f)
        else:
            set_e.append(temp_f)
        # check the calculation state
        calculation_flag = False
        if j == n - 1:
            calculation_flag = True
        if j < n - 1:
            if leaves_info[j + 1][0] != leaves_info[j][0]:
                calculation_flag = True
        if calculation_flag:
            delta = d - path_length
            t = len(set_c)
            for i in range(T):  # Consider each category
                if leaves_result[leaves_counter][i] > 0:
                    value_of_node = leaves_result[leaves_counter][i]
                else:
                    continue
                if len(set_a) > t:
                    temp = p[t + delta][len(set_a) + delta]
                    for element in set_d:
                        results[i][element] += temp * value_of_node
                if len(set_b) > t:
                    temp = p[t + delta][len(set_b) + delta]
                    for element in set_e:
                        results[i][element] -= temp * value_of_node
            set_a.clear()
            set_b.clear()
            set_c.clear()
            set_d.clear()
            set_e.clear()
            path_length = 0
            leaves_counter += 1
    # print(np.sum(results, axis=1))
    return results


class Node:
    def __init__(self, feature, point, l_child, r_child):  # t:number of classes
        self.l_child = -1
        self.r_child = -1
        self.result = -1
        self.info = []
        self.split_feature = feature
        self.split_point = point
        self.l_child = l_child
        self.r_child = r_child


class TreeExplainer:
    def __init__(self, d, t, model):
        self.d = d  # dimension of data
        self.t = t  # number of classes
        self.k = None
        self.nodes_array = []
        self.l = []
        self.r = []
        self.result = []
        self.model = None
        self.left = None
        self.right = None
        self.features = None
        self.thresholds = None
        self.values = None
        self.node_sample_weight = None
        self.sets = None
        self.depth = 0
        self.leaves_info = []  # (m*L, 4)
        self.leaves_table = []
        self.leaves_result = []
        self.supp_info = []
        self.leaves_flag = None
        self.leaves_number = 0
        self.recover(model)

    # Calculate the intervals of each random tree
    @jit(forceobj=True)
    def recover(self, model):
        self.model = model
        self.left = model.tree_.children_left
        self.right = model.tree_.children_right
        self.features = model.tree_.feature
        self.thresholds = model.tree_.threshold
        # record the information of leaves
        leaves_counter = 0
        self.leaves_info = []
        self.leaves_result = []
        # self.values = model.tree_.value.reshape(model.tree_.value.shape[0], -1)
        self.node_sample_weight = model.tree_.weighted_n_node_samples
        k = len(self.left)
        self.nodes_array = []
        for i in range(k):
            new_node = Node(self.features[i], self.thresholds[i], self.left[i], self.right[i])
            self.nodes_array.append(new_node)
        for i in range(k):
            split_feature = self.nodes_array[i].split_feature
            split_point = self.nodes_array[i].split_point
            l_child = self.nodes_array[i].l_child
            r_child = self.nodes_array[i].r_child
            if l_child != -1:
                self.nodes_array[l_child].info = deepcopy(self.nodes_array[i].info)
                flag = False
                for j in range(len(self.nodes_array[l_child].info)):
                    if self.nodes_array[l_child].info[j][0] == split_feature:
                        self.nodes_array[l_child].info[j][2] = min(self.nodes_array[l_child].info[j][2], split_point)
                        flag = True
                        break
                if not flag:
                    self.nodes_array[l_child].info.append([split_feature, -1e8, split_point])
            if r_child != -1:
                self.nodes_array[r_child].info = deepcopy(self.nodes_array[i].info)
                flag = False
                for j in range(len(self.nodes_array[r_child].info)):
                    if self.nodes_array[r_child].info[j][0] == split_feature:
                        self.nodes_array[r_child].info[j][1] = max(self.nodes_array[r_child].info[j][1], split_point)
                        flag = True
                        break
                if not flag:
                    self.nodes_array[r_child].info.append([split_feature, split_point, 1e8])
            # for each leaf node
            if l_child == -1 and r_child == -1:
                self.leaves_result.append(model.tree_.value[i].reshape(-1).astype(float) / np.sum(model.tree_.value[i]))
                temp_list = []
                for j in range(len(self.nodes_array[i].info)):
                    temp_f = self.nodes_array[i].info[j][0]
                    temp_l = self.nodes_array[i].info[j][1]
                    temp_r = self.nodes_array[i].info[j][2]
                    self.leaves_info.append([leaves_counter, temp_f, temp_l, temp_r])
                    temp_list.append([temp_f, temp_l, temp_r])
                leaves_counter += 1
                # Record the max depth
                if len(temp_list) > self.depth:
                    self.depth = len(temp_list)
                # Update the leaves_table
        self.leaves_result = np.asarray(self.leaves_result)
        self.leaves_info = np.asarray(self.leaves_info)
        self.leaves_number = len(self.leaves_result)
        del self.nodes_array

    @jit(forceobj=True)
    def prepare(self, x, baseline_set):
        # initialize the set
        self.sets = []
        self.supp_info = []
        # initialize the status of each leave
        self.leaves_flag = np.zeros(len(baseline_set))
        for i in range(len(baseline_set)):
            baseline = baseline_set[i]
            temp_sets, temp_supp_info = FastPrepare(d=self.d, m=self.leaves_number, depth=self.depth,
                                                    leaves_info=self.leaves_info,
                                                    x=np.asarray(x), baseline_x=baseline)
            self.sets.append(temp_sets)
            self.supp_info.append(temp_supp_info)
        self.sets = np.asarray(self.sets)
        self.supp_info = np.asarray(self.supp_info)
        # print(np.shape(self.sets))


class ForestExplainer:
    def __init__(self, model, data_set, baseline_mode='azo'):
        self.m = len(model.estimators_)  # number of trees
        self.d = model.n_features_  # dimension of data
        self.t = model.n_classes_  # number of classes
        self.trees = []
        self.orig_model = model
        # initialize the baseline set
        self.baseline_set = []
        if 'a' in baseline_mode:
            self.baseline_set.append(np.average(data_set, axis=0))
        if 'z' in baseline_mode:
            self.baseline_set.append(np.zeros(self.d))
        if 'o' in baseline_mode:
            self.baseline_set.append(np.ones(self.d))
        # Prepare
        cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=cores)
        pool_list = []
        for estimator in tqdm(model.estimators_):
            pool_list.append(pool.apply_async(TreeExplainer, (self.d, self.t, estimator)))
        self.trees = [x.get() for x in pool_list]
        # Calculate P(n,m), can be saved in external memory
        path_name = os.path.join('BGShapTree', 'p')
        if not os.path.exists(path_name):
            os.makedirs(path_name)
        file_name = os.path.join('BGShapTree', 'p', str(self.d) + '.m')
        if os.path.exists(file_name):
            self.p = joblib.load(file_name)
        else:
            self.p = np.zeros((self.d + 1, self.d + 1))
            c = np.zeros((self.d + 1, self.d + 1))
            for i in range(self.d + 1):
                for j in range(i, self.d + 1):
                    c[j][i] = comb(j, i)
            for n in tqdm(range(self.d + 1)):
                for m in range(n + 1, self.d + 1):
                    for j in range(n + 1):
                        self.p[n][m] += c[n][j] / (self.d * c[self.d - 1][m - j - 1])
            joblib.dump(self.p, file_name)

    # calculate the BShap values for instances in x_set
    def shapley_value(self, x_set):
        ans = []
        target_category = self.orig_model.predict(x_set)
        for j in tqdm(range(len(x_set))):
            x = x_set[j]
            result_set = []
            for i in range(self.m):
                for k in range(len(self.baseline_set)):
                    # if k != target_category[j]:
                    baseline_x = self.baseline_set[k]
                    phi = FastBaselineShapleyCategory(self.trees[i].t, self.trees[i].d, self.trees[i].leaves_info,
                                                      self.trees[i].leaves_result, np.asarray(x),
                                                      np.asarray(baseline_x),
                                                      np.asarray(self.p), target_category[j])
                    result_set.append(phi)
            ans.append(np.average(result_set, axis=0))
        return np.asarray(ans)

    # evaluate the importance of set T for instance x
    def set_valuation(self, x, T):
        target_category = self.orig_model.predict([x])[0]
        # load p array
        p = load_p(self.d - len(T) + 1)
        result = 0
        cnt = 0
        for i in range(self.m):
            for baseline_x in self.baseline_set:
                phi = SetValuation(self.trees[i].d, self.trees[i].leaves_info, self.trees[i].leaves_result,
                                   np.asarray(x), np.asarray(baseline_x),
                                   np.asarray(p), target_category, np.asarray(T))
                result += phi
                cnt += 1
        if cnt == 0:
            return 0
        return result / cnt

    def search_for_salient_set(self, x, tau, size):
        # initialize the list
        list = []
        category = self.orig_model.predict([x])[0]
        # calculate the set for given instance x
        start = time.time()
        for i in range(self.m):
            self.trees[i].prepare(x, self.baseline_set)
        # print(time.time()-start)

        # calculate the importance of individual features
        p = load_p(self.d)
        temp_result = []
        for i in range(self.m):
            for k in range(len(self.baseline_set)):
                baseline_x = self.baseline_set[k]
                phi = FastBaselineShapleyCategory(self.trees[i].t, self.trees[i].d, self.trees[i].leaves_info,
                                                  self.trees[i].leaves_result, np.asarray(x),
                                                  np.asarray(baseline_x), np.asarray(p), category)
                temp_result.append(phi[category])
        end = time.time()
        fi = np.average(temp_result, axis=0)
        list_p = []
        all_features = []
        for j in range(self.d):
            list_p.append([[j], fi[j]])
            all_features.append(j)
        for k in range(1, size):
            p = load_p(self.d - k)
            list_p.sort(key=lambda a: a[1], reverse=True)
            new_list = []
            flag = []
            for l in range(min(tau, len(list_p))):
                set_l = list_p[l][0]
                feature_set = range(self.d)
                # need to be fixed here
                if self.d > 50:
                    temp_set = set(all_features) - set(set_l)
                    feature_set = random.sample(temp_set, min(len(temp_set), 25))
                for t in feature_set:
                    if t not in set_l:
                        temp_set = copy.copy(set_l)
                        temp_set.append(t)
                        if temp_set not in flag:
                            flag.append(temp_set)
                            # evaluate the importance of temp set
                            temp_result = []
                            for i in range(self.m):
                                for j in range(len(self.baseline_set)):
                                    phi = SetValuationSS(d=self.trees[i].d, sets=np.asarray(self.trees[i].sets[j]),
                                                         supp_info=np.asarray(self.trees[i].supp_info[j]),
                                                         leaves_result=self.trees[i].leaves_result,
                                                         p=np.asarray(p), category=category,
                                                         T=np.asarray(temp_set))
                                    temp_result.append(phi)
                            # add the new expanded set into list
                            new_list.append([temp_set, np.average(temp_result)])
            list_p = new_list
        list_p.sort(key=lambda a: a[1], reverse=True)
        return list_p[0][0]


def load_p(d):
    path_name = os.path.join('BGShapTree', 'p')
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    file_name = os.path.join('BGShapTree', 'p', str(d) + '.m')
    if os.path.exists(file_name):
        p = joblib.load(file_name)
    else:
        p = np.zeros((d + 1, d + 1))
        c = np.zeros((d + 1, d + 1))
        for i in range(d + 1):
            for j in range(i, d + 1):
                c[j][i] = comb(j, i)
        for n in tqdm(range(d + 1)):
            for m in range(n + 1, d + 1):
                for j in range(n + 1):
                    p[n][m] += c[n][j] / (d * c[d - 1][m - j - 1])
        joblib.dump(p, file_name)
    return p


if __name__ == "__main__":
    pass
