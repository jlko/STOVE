import numpy as np
# from graphviz import Digraph


class RegionGraph(object):
    """Represents a region graph."""

    def __init__(self, items, seed=12345):

        self._items = tuple(sorted(items))

        # Regions
        self._regions = set()
        self._child_partitions = dict()

        # Partitions
        self._partitions = set()

        # Private random generator
        self._rand_state = np.random.RandomState(seed)

        # layered representation of the region graph
        self._layers = []

        # The root region (== _items) is already part of the region graph
        self._regions.add(self._items)

    def get_root_region(self):
        """Get root region."""
        return self._items

    def get_num_items(self):
        return len(self._items)

    def get_regions(self):
        return self._regions

    def get_child_partitions(self, region):
        return self._child_partitions[region]

    def get_region(self, region):
        """Get a region and create if it does not exist."""
        region = tuple(sorted(region))
        if not region <= self._items:
            raise ValueError('Argument region is not a sub-set of _items.')

        self._regions.add(region)
        return region

    def get_leaf_regions(self):
        """Get leaf regions, i.e. regions which don't have child partitions."""
        return [x for x in self._regions if x not in self._child_partitions]

    def random_split(self, num_parts, num_recursions=1, region=None):
        """Split a region in n random parts and introduce the corresponding partition in the region graph."""

        if num_recursions < 1:
            return None

        if not region:
            region = self._items

        if region not in self._regions:
            raise LookupError('Trying to split non-existing region.')

        if len(region) == 1:
            return None

        region_list = list(self._rand_state.permutation(list(region)))

        num_parts = min(len(region_list), num_parts)
        q = len(region_list) // num_parts
        r = len(region_list) % num_parts

        partition = []
        idx = 0
        for k in range(0, num_parts):
            inc = q + 1 if k < r else q
            sub_region = tuple(sorted(region_list[idx:idx+inc]))
            partition.append(sub_region)
            self._regions.add(sub_region)
            idx = idx + inc

        partition = tuple(sorted(partition))

        if partition not in self._partitions:
            self._partitions.add(partition)
            region_children = self._child_partitions.get(region, [])
            self._child_partitions[region] = region_children + [partition]

        if num_recursions > 1:
            for r in partition:
                self.random_split(num_parts, num_recursions-1, r)

        return partition

    def make_split(self, region, sub_region):

        if region not in self._regions:
            raise LookupError('Trying to split non-existing region.')

        if not sub_region.issubset(region) or len(sub_region) >= len(region) or len(sub_region) == 0:
            raise AssertionError('sub-region is not a proper sub-set.')

        sub_region2 = region.difference(sub_region)

        self._regions.add(sub_region)
        self._regions.add(sub_region2)
        partition = tuple(sorted([sub_region, region.difference(sub_region)]))

        if partition not in self._partitions:
            self._partitions.add(partition)
            region_children = self._child_partitions.get(region, [])
            self._child_partitions[region] = region_children + [partition]

        return partition

    def make_layers(self):
        """Make a layered structure.

        _layer[0] will contain leaf regions
        _layer[k], when k is odd, will contain partitions
        _layer[k], when k is even, will contain regions
        """

        seen_regions = set()
        seen_partitions = set()

        leaf_regions = self.get_leaf_regions()
        # sort regions lexicographically
        leaf_regions = [tuple(sorted(i)) for i in sorted([sorted(j) for j in leaf_regions])]
        self._layers = [leaf_regions]
        if (len(leaf_regions) == 1) and (self._items in leaf_regions):
            return self._layers

        seen_regions.update(leaf_regions)

        while len(seen_regions) != len(self._regions) or len(seen_partitions) != len(self._partitions):
            # the next partition layer contains all partitions which have not been visited (seen)
            # and all its child regions have been visited
            next_partition_layer = [p for p in self._partitions if p not in seen_partitions
                                    and all([r in seen_regions for r in p])]
            self._layers.append(next_partition_layer)
            seen_partitions.update(next_partition_layer)

            # similar as above, but now for regions
            next_region_layer = [r for r in self._regions if r not in seen_regions
                                 and all([p in seen_partitions for p in self._child_partitions[r]])]
            # sort regions lexicographically
            next_region_layer = [tuple(sorted(i)) for i in sorted([sorted(j) for j in next_region_layer])]

            self._layers.append(next_region_layer)
            seen_regions.update(next_region_layer)

        return self._layers

    def make_poon_structure(self, width, height, delta, max_split_depth=None):
        """
        Make a Poon & Domingos like region graph.

        :param width: image width
        :param height: image height
        :param delta: split step-size
        :param max_split_depth: stop splitting at this depth
        :return:
        """

        if self._items != tuple(range(width * height)):
            raise AssertionError('Item set needs to be tuple(range(width * height)).')

        if type(delta) != int or delta <= 0:
            raise AssertionError('delta needs to be a nonnegative integer.')

        def split(A, axis_idx, x):
            """This splits a multi-dimensional numpy array in one axis, at index x.
            For example, if A =
            [[1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]]

            then split(A, 0, 1) delivers
            [[1, 2, 3, 4]],

            [[5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]]
            """
            slc = [slice(None)] * len(A.shape)
            slc[axis_idx] = slice(0, x)
            A1 = A[tuple(slc)]
            slc[axis_idx] = slice(x, A.shape[axis_idx])
            A2 = A[tuple(slc)]
            return A1, A2

        img = np.reshape(range(height * width), (height, width))
        img_tuple = tuple(sorted(img.reshape(-1)))

        # Q is a queue
        Q = [img]
        depth_dict = {img_tuple: 0}

        while Q:
            region = Q.pop(0)
            region_tuple = tuple(sorted(region.reshape(-1)))
            depth = depth_dict[region_tuple]
            if max_split_depth is not None and depth >= max_split_depth:
                continue

            region_children = []

            for axis, length in enumerate(region.shape):
                if length <= delta:
                    continue

                num_splits = int(np.ceil(length / delta) - 1)
                split_points = [(x + 1) * delta for x in range(num_splits)]

                for idx in split_points:
                    region_1, region_2 = split(region, axis, idx)

                    region_1_tuple = tuple(sorted(region_1.reshape(-1)))
                    region_2_tuple = tuple(sorted(region_2.reshape(-1)))

                    if region_1_tuple not in self._regions:
                        self._regions.add(region_1_tuple)
                        depth_dict[region_1_tuple] = depth + 1
                        Q.append(region_1)

                    if region_2_tuple not in self._regions:
                        self._regions.add(region_2_tuple)
                        depth_dict[region_2_tuple] = depth + 1
                        Q.append(region_2)

                    partition = tuple(sorted([region_1_tuple, region_2_tuple]))

                    if partition in self._partitions:
                        raise AssertionError('Partition already generated -- this should not happen.')

                    self._partitions.add(partition)
                    region_children.append(partition)

            if region_children:
                self._child_partitions[region_tuple] = region_children

#    def render_dot(self, path):
#
#        region_to_label = {}
#        partition_to_label = {}
#
#        dot = Digraph()
#
#        for counter, region in enumerate(self._regions):
#            label = 'R' + str(counter)
#            dot.node(label, label=str(list(region)))
#            region_to_label[region] = label
#
#        dot.attr('node', shape='box')
#        for counter, partition in enumerate(self._partitions):
#            label = 'P' + str(counter)
#            dot.node(label, label='X')
#            partition_to_label[partition] = label
#
#        for region in self._regions:
#            if region not in self._child_partitions:
#                continue
#            for partition in self._child_partitions[region]:
#                dot.edge(region_to_label[region], partition_to_label[partition])
#
#        for partition in self._partitions:
#            for region in partition:
#                dot.edge(partition_to_label[partition], region_to_label[region])
#
#        dot.render(path)

if __name__ == '__main__':

    rg = RegionGraph([1, 2, 3, 4, 5, 6, 7])
    for k in range(3):
        rg.random_split(2, 2)
    layers = rg.make_layers()

    for k in reversed(layers):
        print(k)
