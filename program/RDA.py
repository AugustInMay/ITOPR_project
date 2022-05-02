import random


class Location:
    def __init__(self):
        self.coors = list()
        self.dimension = 0

    def __init__(self, coors_: list):
        self.coors = coors_
        self.dimension = len(coors_)

    def get_coor(self, index):
        return self.coors[index]

    def get_dim(self):
        return self.dimension


class Point(Location):
    def __init__(self):
        super().__init__()
        self.func = 0.

    def __init__(self, coors_: list, func_):
        super().__init__(coors_)
        self.func = float(func_)

    def get_func(self):
        return self.func


class Sort:
    def swap(self, p_arr: list, i1, i2):
        tmp = p_arr[i1]
        p_arr[i1] = p_arr[i2]
        p_arr[i2] = tmp

    def radius(self, p_arr: list, midpoint, dimension):
        p_arr_c = p_arr.copy()

        for i in range(len(p_arr)-1):
            for j in range(i, len(p_arr)):
                if abs(midpoint - p_arr_c[i].get_coor(dimension)) > abs(midpoint - p_arr_c[j].get_coor(dimension)):
                    self.swap(p_arr_c, i, j)

        return abs(midpoint - p_arr_c[int(len(p_arr)/2 + len(p_arr) % 2)].get_coor(dimension))


class ITree:
    def get_func(self, location: Location):
        raise NotImplementedError()


class Leaf(ITree):
    def __init__(self, p: Point):
        self.start_p = p

    def get_func(self, location: Location):
        return self.start_p.get_func()


class Branch(ITree):
    def __init__(self, p_arr: list):
        self.global_max_between = 0
        self.global_left_point = 0
        self.dimension = 0
        self.split = 0

        dim_amount = p_arr[0].get_dim()

        for i in range(dim_amount):
            clone = p_arr.copy()
            clone.sort(key=lambda x: x.get_coor(i))

            max_between = (clone[0].get_coor(i) + clone[-1].get_coor(i))
            mid_p = max_between / 2.
            dist = abs(clone[0].get_coor(i) - clone[-1].get_coor(i))

            if dist >= self.global_max_between:
                self.global_max_between = dist
                self.dimension = i
                self.split = mid_p
                self.flag = (clone[0].get_coor(i) == clone[-1].get_coor(i))

        left_mass = list()
        right_mass = list()

        for el in p_arr:
            if el.get_coor(self.dimension) < self.split:
                left_mass.append(el)
            else:
                right_mass.append(el)

        if self.flag:
            self.left_branch = Leaf(right_mass[0])
        elif len(left_mass) > 1:
            self.left_branch = Branch(left_mass)
        else:
            self.left_branch = Leaf(left_mass[0])

        if self.flag:
            self.right_branch = Leaf(right_mass[0])
        elif len(right_mass) > 1:
            self.right_branch = Branch(right_mass)
        else:
            self.right_branch = Leaf(right_mass[0])

    def get_func(self, location: Location):
        if location.get_coor(self.dimension) < self.split:
            return self.left_branch.get_func(location)
        else:
            return self.right_branch.get_func(location)


class Aprox:
    def __init__(self):
        self.points = list()

    def add_point(self, coors: list, func):
        self.points.append(Point(coors, func))

    def get_point(self, index):
        return self.points[index]

    def delete_point(self, index):
        self.points.pop(index)

    def count(self, loc: Location):
        return NotImplementedError()

    def count(self, coors: list):
        loc = Location(coors)
        return self.count(loc)


class RDA(Aprox):
    def __init__(self, dim_am, tree_am=50, choose_per=0.8):
        self.points = list()
        self.forest = list()

        self.dim_am = dim_am
        self.tree_am = tree_am
        self.choose_per = choose_per

        self.error_tree = None

    def do_approx(self):
        points_am = len(self.points)

        excluded_points_am = 1
        if len(self.points) >= 10:
            excluded_points_am = int(len(self.points) * (1 - self.choose_per))

        for i in range(self.tree_am):
            excluded_points = list()
            tree_points = list()

            for j in range(excluded_points_am):
                to_exclude = random.randint(0, points_am)
                while to_exclude in excluded_points:
                    to_exclude = random.randint(0, points_am)

                excluded_points.append(to_exclude)

            for j in range(points_am):
                if j not in excluded_points:
                    tree_points.append(self.points[j])

            self.forest.append(Branch(tree_points))

        errors = list()

        for i in range(points_am):
            s = sum(el.get_func(self.points[i]) for el in self.forest)
            s = float(s) / self.tree_am

            error = self.points[i].get_func() - s
            errors.append(Point(self.points[i].coors, error))

        self.error_tree = Branch(errors)

    def count(self, loc: Location):
        s = sum(el.get_func(loc) for el in self.forest)
        s = float(s) / self.tree_am
        return s + self.error_tree.get_func(loc)
