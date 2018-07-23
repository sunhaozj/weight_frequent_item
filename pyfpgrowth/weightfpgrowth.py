import itertools
from collections import defaultdict


class FPNode(object):
    """
    A node in the FP tree.
    """

    def __init__(self, value, count, weight, parent):
        """
        Create the node.
        """
        self.value  = value
        self.count  = count
        self.weight = weight
        self.parent = parent
        self.link = None
        self.children = []

    def has_child(self, value):
        """
        Check if node has a particular child node.
        """
        for node in self.children:
            if node.value == value:
                return True

        return False

    def get_child(self, value):
        """
        Return a child node with a particular value.
        """
        for node in self.children:
            if node.value == value:
                return node

        return None

    def add_child(self, value, weight):
        """
        Add a node as a child node.
        """
        child = FPNode(value, 1, weight, self)
        self.children.append(child)
        return child

    def disp(self,ind = 1):
        print (' '*ind,self.value,' ',self.count)
        for child in self.children:
            child.disp(ind+1)


class WeightfPTree(object):
    """
    A frequent pattern tree.
    """

    def __init__(self, transactions, supprot_threshold, weight_threadshod,root_value,
                 root_count,default_weight=1,weights=None,isConditional=True):
        """
        Initialize the tree.
        """
        self.weights = weights
        self.default_weight  = default_weight
        self.weight_frequent = self.find_weight_frequent_items(transactions, supprot_threshold,weight_threadshod,isConditional)
        self.headers         = self.build_header_table(self.weight_frequent)
        self.root            = self.build_fptree(
                                transactions, root_value,
                                root_count, self.weight_frequent, self.headers)

    def findMaxAndMinWeight(self):
        '''
        the maximum(minimum) weight of items in a transaction
        :param weightMap:
        :return:
        '''
        maxAndminWeight = {}
        if self.weights is None or len(self.weights.keys()) == 0:
            maxAndminWeight['MaxW'] = self.default_weight
            maxAndminWeight['MinW'] = self.default_weight
        else:
            weights = self.weights.values()
            maxAndminWeight['MaxW'] = max(weights)
            maxAndminWeight['MinW'] = min(weights)
        return  maxAndminWeight

    def find_weight_frequent_items(self,transactions, support_threshold,weight_threshold,isConditional):
        """
        Create a dictionary of items with occurrences above the threshold.
        transactions data structure:[[1,2,3,...],[3,4,5,...],...]
        """
        items   = defaultdict(dict)
        for transaction in transactions:
            for item in transaction:
                if self.weights is None or item not in self.weights:
                    #weights[item] = self.default_weight
                    items[item]['weight'] = self.default_weight
                else:
                    items[item]['weight'] = self.weights[item]
                if 'support' in items[item]:
                    #support[item] += 1
                    #items[item] += 1
                    items[item]['support'] += 1
                else:
                    #support[item] = 1
                    #items[item]   = 1
                    items[item]['support'] = 1

        self.weights = {x:items[x]['weight'] for x in items.keys()}
        for key in list(items.keys()):
            if self.conditionPrune(key,items[key]['support'],support_threshold,weight_threshold,isConditional):
                del items[key]
        return items


    def conditionPrune(self,item_name,item_value,support_threshold,weight_threshold,maxORmin):
        '''
        prune condition:
        1. support < support_threshold && weight < weight_threshold
            OR
        2. support * MaxW(MinW) < support_threshold
        :param item_name:key
        :param item_value:support_value
        :param support_threshold:
        :param weight_threshold:
        :param weights:
        :param maxORmin: True:Max,False:Min
        :return: True represend need del
        '''

        maxAndminWeight = self.findMaxAndMinWeight()
        maxOrMinWeight  = maxAndminWeight['MaxW'] if maxORmin is True else maxAndminWeight['MinW']
        item_weight     = self.weights[item_name] if item_name in self.weights else self.default_weight
        multiValue      = item_value * maxOrMinWeight

        if  (item_value < support_threshold and item_weight < weight_threshold) or \
                multiValue < support_threshold:
            return True
        return False




    @staticmethod
    def build_header_table(frequent):
        """
        Build the header table.
        """
        headers = {}
        for key in frequent.keys():
            headers[key] = None

        return headers

    def build_fptree(self, transactions, root_value,
                     root_count, weight_frequent, headers):
        """
        Build the FP tree and return the root node.
        """
        root = FPNode(root_value, root_count, None, None)#this root is null point

        for transaction in transactions:
            sorted_items = [x for x in transaction if x in weight_frequent]
            sorted_items.sort(key=lambda x: weight_frequent[x]['weight'])#根据weight升序
            if len(sorted_items) > 0:
                self.insert_tree(sorted_items, root, headers)

        return root

    def insert_tree(self, items, node, headers):
        """
        Recursively grow FP tree.
        init：
        item:   is a sorted dataset desc
        node:   is a root,none point
        heasers:the structure is {key:none},key is single frequent data
        """
        first_name   = items[0] # FpNode Value(item name)
        first_weight = self.weight_frequent[first_name]['weight']
        child = node.get_child(first_name)
        if child is not None:
            child.count += 1
        else:
            # Add new child.
            child = node.add_child(first_name,first_weight)

            # Link it to header structure.
            if headers[first_name] is None:
                headers[first_name] = child
            else:
                current = headers[first_name]
                while current.link is not None:
                    current = current.link
                current.link = child

        # Call function recursively.
        remaining_items = items[1:]
        if len(remaining_items) > 0:
            self.insert_tree(remaining_items, child, headers)

    def tree_has_single_path(self, node):
        """
        If there is a single path in the tree,
        return True, else return False.
        """
        num_children = len(node.children)
        if num_children > 1:
            return False
        elif num_children == 0:
            return True
        else:
            return True and self.tree_has_single_path(node.children[0])

    def mine_patterns(self, support_threshold,weight_support):
        """
        Mine the constructed FP tree for frequent patterns.
        """
        if self.tree_has_single_path(self.root):
            return self.generate_pattern_list()
        else:
            return self.zip_patterns(self.mine_sub_trees(support_threshold,weight_support))

    def zip_patterns(self, patterns):
        """
        Append suffix to patterns in dictionary if
        we are in a conditional FP tree.
        """
        suffix = self.root.value

        if suffix is not None:
            # We are in a conditional tree.
            new_patterns = {}
            for key in patterns.keys():
                new_patterns[tuple(sorted(list(key) + [suffix]))] = patterns[key]

            return new_patterns

        return patterns

    def generate_pattern_list(self):
        """
        Generate a list of patterns with support counts.
        """
        patterns = {}
        items = self.weight_frequent.keys()

        # If we are in a conditional tree,
        # the suffix is a pattern on its own.
        if self.root.value is None:
            suffix_value = []
        else:
            suffix_value = [self.root.value]
            patterns[tuple(suffix_value)] = self.root.count

        for i in range(1, len(items) + 1):
            for subset in itertools.combinations(items, i):#返回items中所有长度为i的子序列
                pattern = tuple(sorted(list(subset) + suffix_value))
                patterns[pattern] = \
                    min([self.weight_frequent[x]['support'] for x in subset])

        return patterns

    def mine_sub_trees(self, support_threshold, weight_threshold):
        """
        Generate subtrees and mine them for patterns.
        """
        patterns = {}
        mining_order = sorted(self.weight_frequent.keys(),
                              key=lambda x: self.weight_frequent[x]['weight'],reverse=True)#从根结点开始查找

        weights ={x:self.weight_frequent[x]['weight'] for x in self.weight_frequent}

        # Get items in tree in reverse order of occurrences.
        for item in mining_order:
            suffixes = []
            conditional_tree_input = []#
            node = self.headers[item]

            # Follow node links to get a list of
            # all occurrences of a certain item.
            #找到所有以node节点的路径，er.  (r,2)-->(r,1)--(r,1)
            while node is not None:
                suffixes.append(node)
                node = node.link

            # For each occurrence of the item,
            # trace the path back to the root node.
            for suffix in suffixes:
                frequency = suffix.count
                #weight    = suffix.weight
                path = []
                parent = suffix.parent

                while parent.parent is not None:
                    path.append(parent.value)
                    parent = parent.parent

                for i in range(frequency):
                    conditional_tree_input.append(path)

            # Now we have the input for a subtree,
            # so construct it and grab the patterns.
            subtree = WeightfPTree(conditional_tree_input, support_threshold,weight_threshold,
                                   item, self.weight_frequent[item]['support'],self.default_weight,weights,False)
            subtree_patterns = subtree.mine_patterns(support_threshold,weight_threshold)

            # Insert subtree patterns into main patterns dictionary.
            for pattern in subtree_patterns.keys():
                if pattern in  :
                    patterns[pattern] += subtree_patterns[pattern]
                else:
                    patterns[pattern] = subtree_patterns[pattern]

        return patterns


def find_frequent_patterns(transactions, support_threshold,weight_threshold,weights=None,default_weight=1):
    '''
    Given a set of transactions, find the patterns in it
    over the specified support threshold.

    :param transactions:
    :param support_threshold:
    :param weight_threshold:
    :param weights:
    :param default_weight:
    :return:
    '''
    tree = WeightfPTree(transactions, support_threshold,weight_threshold, None, None,default_weight,weights)
    return tree.mine_patterns(support_threshold,weight_threshold)


def generate_association_rules(patterns, confidence_threshold):
    """
    Given a set of frequent itemsets, return a dict
    of association rules in the form
    {(left): ((right), confidence)}
    """
    rules = {}
    for itemset in patterns.keys():
        upper_support = patterns[itemset]

        for i in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, i):
                antecedent = tuple(sorted(antecedent))
                consequent = tuple(sorted(set(itemset) - set(antecedent)))

                if antecedent in patterns:
                    lower_support = patterns[antecedent]
                    confidence = float(upper_support) / lower_support

                    if confidence >= confidence_threshold:
                        rules[antecedent] = (consequent, confidence)

    return rules
