import numpy as np
from shapely.geometry import Point, LineString, Polygon
from time import perf_counter
from queue import PriorityQueue
import functions.geo_functions as gf


class FMT_STAR:
    """Implementation of the FMT* algorithm."""

    def __init__(self, environment: dict, cruise_speed: float, climb_rate: float):
        """Initializes the FMT* algorithm.

        Args:
            environment (dict): Environment data.
            cruise_speed (float): The cruise speed of the aircraft (m/s).
            climb_rate (float): The climb rate of the aircraft (ft/min).
        """
        self.sector = environment["sector"]
        self.tczs = environment["TCZs"]
        self.vertiports = environment["vertiports"]
        self.frzs = environment["FRZs"]

        self.cruise_speed = cruise_speed
        self.climb_rate = climb_rate

        for tcz in self.tczs:
            self.tczs[tcz] = [
                Polygon(self.tczs[tcz][:, :2]),
                max(self.tczs[tcz][:, -1]),
            ]

        for frz in self.frzs:
            self.frzs[frz] = [
                Polygon(self.frzs[frz][:, :2]),
                max(self.frzs[frz][:, -1]),
            ]

        self.start, self.goal = None, None
        self.start_tcz = ""
        self.goal_tcz = ""

        self._parents = []

    @property
    def obstacle_environment(self) -> dict:
        """A dictionary of the obstacle environment.

        Returns:
            dict: The obstacle environment.
        """
        obstacle_environment = [{"polygon": self.tczs[tcz][0], "max_height": self.tczs[tcz][1]} for tcz in self.tczs]

        obstacle_environment.extend(
            {"polygon": self.frzs[frz][0], "max_height": self.frzs[frz][1]} for frz in self.frzs
        )

        return obstacle_environment

    def initilise_route(
        self,
        start: int = None,
        goal: int = None,
        dep_rad=0.3,
        arr_rad=0.3,
        n_nodes=10,
        dep_alt=500,
        arr_alt=500,
    ) -> None:
        """Initializes the start and goal vertiports for the FMT* algorithm and sets the corresponding experience zones.

        Args:
            self: The instance of the class.
            start (int, optional): The index of the start vertiport. Defaults to None.
            goal (int, optional): The index of the goal vertiport. Defaults to None.

        Returns:
            None

        Example:
            ```python
            fmt = FMTStar()
            start = 0
            goal = 1

            fmt.initilise_route(start, goal)
            ```
        """

        self.start, self.goal = gf.initilise_start_goal_vertiports(self.vertiports, start, goal)

        dep_rad = (
            dep_rad
            if (dep_alt / self.climb_rate) * 60 <= (dep_rad * 1000) / self.cruise_speed
            else self.cruise_speed * ((dep_alt / self.climb_rate) * 60) / 1000
        )
        arr_rad = (
            arr_rad
            if (arr_alt / self.climb_rate) * 60 <= (arr_rad * 1000) / self.cruise_speed
            else self.cruise_speed * ((arr_alt / self.climb_rate) * 60) / 1000
        )

        self.departure_points = gf.calculate_vertiport_points(
            *self.vertiports[self.start][:-1], dep_rad, dep_alt, n_nodes
        )

        self.arrival_points = gf.calculate_vertiport_points(*self.vertiports[self.goal][:-1], arr_rad, arr_alt, n_nodes)

        point_start = Point(self.vertiports[self.start])
        point_goal = Point(self.vertiports[self.goal])

        for tcz, (poly, _) in self.tczs.items():
            if poly.contains(point_start):
                self.start_tcz = tcz

            if poly.contains(point_goal):
                self.goal_tcz = tcz

        self.experience_zones = list(self.frzs.values())
        self.experience_zones.extend(
            value for key, value in self.tczs.items() if key not in [self.start_tcz, self.goal_tcz]
        )

        self.arrival_points = self.validate_nodes(self.arrival_points, True, levels=[arr_alt])

        arr_points = []
        for point in self.arrival_points:
            line = LineString([point[:-1], self.vertiports[self.goal][:-1]])
            valid = all(
                gf.evaluate_intersection(line, point, self.vertiports[self.goal], poly, h)
                for poly, h in self.experience_zones
            )
            if valid:
                arr_points.append(point)

        self.arrival_points = np.array(arr_points)

        if len(self.arrival_points) == 0:
            raise ValueError("No valid arrival points found")

        self.departure_points = self.validate_nodes(self.departure_points, True, levels=[dep_alt])
        arr_points = []
        for point in self.departure_points:
            valid = all(
                gf.evaluate_intersection(
                    LineString([point[:-1], self.vertiports[self.start][:-1]]),
                    point,
                    self.vertiports[self.start],
                    poly,
                    h,
                )
                for poly, h in self.experience_zones
            )
            if valid:
                arr_points.append(point)

        self.departure_points = np.array(arr_points)
        if len(self.departure_points) == 0:
            raise ValueError("No valid departure points found")

        print(
            "TRUE DIST: ",
            round(
                gf.haversine_distance(self.vertiports[self.start][:-1], self.vertiports[self.goal][:-1]),
                2,
            ),
        )

        print(f"Start: {self.start}\nGoal: {self.goal}")

    def gen2D_nodes(self, N: int, validate: bool) -> np.ndarray:
        """Generates 2D nodes within the specified sector.

        Args:
            self: The instance of the class.
            N (int): The number of nodes to generate.
            validate (bool): Flag indicating whether to validate the generated nodes.

        Returns:
            np.ndarray: The generated nodes.

        Example:
            ```python
            fmt = FMTStar()
            N = 10
            validate = True

            nodes = fmt.gen2D_nodes(N, validate)
            print(nodes)
            ```
        """

        nodes = np.random.uniform(self.sector.min(axis=0)[:2], self.sector.max(axis=0)[:2], (N, 2))
        if validate and self.start is not None:
            nodes = self.validate_nodes(nodes, is3D=False)

        running_nodes = np.insert(
            nodes,
            0,
            self.departure_points[:, :2],
            axis=0,
        )

        self.origin_idxs = np.arange(1, self.departure_points.shape[0])

        running_nodes = np.append(self.arrival_points[:, :2], running_nodes, axis=0)
        self.destination_idxs = np.arange(
            running_nodes.shape[0] - self.arrival_points.shape[0],
            running_nodes.shape[0],
        )

        return gf.build_distance_matrix(running_nodes, nodes, gf.haversine_distance)

    def gen3D_nodes(
        self,
        N: int,
        validate: bool,
        floor: float,
        ceil: float,
        height: float,
        random_alts=True,
    ) -> np.ndarray:
        """Generates 3D nodes within the specified sector and height range.

        Args:
            self: The instance of the class.
            N (int): The number of nodes to generate.
            validate (bool): Flag indicating whether to validate the generated nodes.
            floor (float): The minimum height value.
            ceil (float): The maximum height value.
            height (float): The height increment.

        Returns:
            np.ndarray: The generated nodes.

        Example:
            ```python
            fmt = FMTStar()
            N = 10
            validate = True
            floor = 0
            ceil = 100
            height = 10

            nodes = fmt.gen3D_nodes(N, validate, floor, ceil, height)
            print(nodes)
            ```
        """

        levels = list(range(floor, ceil, height))
        if random_alts:
            nodes = np.array(
                [
                    np.random.uniform(
                        np.append(self.sector.min(axis=0)[:2], [l + (height / 2) - 5], axis=0),
                        np.append(self.sector.max(axis=0)[:2], [l + (height / 2) + 5], axis=0),
                        (N, 3),
                    )
                    for l in levels
                ]
            )
        else:
            temp_nodes = np.random.uniform(self.sector.min(axis=0)[:2], self.sector.max(axis=0)[:2], (N, 2))
            nodes = np.array([np.column_stack((temp_nodes.copy(), l * np.ones(temp_nodes.shape[0]))) for l in levels])

            print(nodes.shape)

        nodes[:, :, -1] = np.round(nodes[:, :, -1])

        if validate and self.start is not None:
            nodes = self.validate_nodes(nodes, True, levels=levels)

        running_nodes = np.insert(nodes, 0, self.departure_points, axis=0)

        self.origin_idxs = np.arange(0, self.departure_points.shape[0])

        running_nodes = np.append(running_nodes, self.arrival_points, axis=0)

        self.destination_idxs = np.arange(
            running_nodes.shape[0] - self.arrival_points.shape[0],
            running_nodes.shape[0],
        )

        return gf.build_distance_matrix(running_nodes, nodes, gf.haversine_distance)

    def populate_nodes(self, N: int = 1000, validate=True, is3D=True, floor=500, ceil=3000, height=1000) -> None:
        """Generate nodes, with the start and goal points inserted at the start and 1st index respectively. Also validates the nodes, and builds the distance matrix.

        Args:
            N (int, optional): Number of nodes to generate (for each level if 3D). Defaults to 1000.
            validate (bool, optional): If True, the nodes will be validated. Defaults to True.
            is3D (bool, optional): If True, the nodes will be 3-dimensional. Defaults to True.
            floor (int, optional): The minimum altitude of the nodes. Defaults to 300.
            ceil (int, optional): The maximum altitude of the nodes. Defaults to 5000.
            height (int, optional): The height of each level. Defaults to 500.
        """
        self.nodes, self.running_nodes, self.distance_matrix = (
            self.gen3D_nodes(N, validate, floor, ceil, height) if is3D else self.gen2D_nodes(N, validate)
        )

    def find_path(self, min_dist=0, max_dist=5, is3D=True) -> None:
        """Find the path, using the FMT* algorithm.

        Args:
            min_dist (int, optional): The minimum distance between connected nodes. Defaults to 0.
            max_dist (int, optional): The maximum distance between connected nodes. Defaults to 5.
            is3D (bool, optional): If True, the nodes will be 3-dimensional. Defaults to True.
        """
        print("Finding Path...")

        self.distance_matrix[self.distance_matrix == 0] = np.inf
        self.last_angles = np.zeros(len(self.distance_matrix))

        for _id in self.origin_idxs:
            self.last_angles[_id] = gf.calculate_initial_compass_bearing(
                *self.vertiports[self.start][:2], *self.running_nodes[_id, :-1]
            )

        n = len(self.distance_matrix)

        _open = PriorityQueue()
        _open_costs = np.full(n, np.inf)
        for i in self.origin_idxs:
            min_g_dist = np.min(self.distance_matrix[i, self.destination_idxs]) ** 2
            _open.put((min_g_dist, i))
            _open_costs[i] = min_g_dist

        _visited = np.zeros(n, dtype=bool)
        self._parents = np.full(n, -1)

        _t_costs = np.zeros(n)

        while _open:
            w, active = _open.get()

            if _visited[active]:
                continue

            to_add = self.body(active, _t_costs, _open_costs, min_dist, max_dist, is3D=is3D)

            for n, val in to_add.items():
                if _visited[n]:
                    continue

                self._parents[n] = val[1]
                self.last_angles[n] = val[2]
                cost = (
                    0
                    if any(self.distance_matrix[n, self.destination_idxs] == np.inf)
                    else np.min(self.distance_matrix[n, self.destination_idxs])
                ) ** 2

                if is3D:
                    alt_change = max(self.running_nodes[n][-1] - self.running_nodes[val[1]][-1], 0)
                else:
                    alt_change = 0

                _t_costs[n] = _t_costs[val[1]] + self.distance_matrix[n, val[1]]

                cost += _t_costs[n]

                _open.put((cost, n))
                _open_costs[n] = cost

            _visited[active] = True

            if any(x in self.destination_idxs for x in to_add.keys()):
                print("Path Found")
                break

            if _open.empty():
                print("Failure: No feasible path found")
                return

    def body(
        self,
        active: int,
        _t_costs: np.ndarray,
        _open_costs: np.ndarray,
        min_dist: float,
        max_dist: float,
        is3D: bool,
    ) -> dict:
        """Generates a list of nodes to add to the open list.

        Args:
            active (int): The index of the active node.
            _t_costs (np.ndarray): The traversal cost of each node.
            _open_costs (np.ndarray): The cost of each node in the open list.
            min_dist (float): The minimum distance between connected nodes.
            max_dist (float): The maximum distance between connected nodes.
            is3D (bool): If True, the nodes will be 3-dimensional.

        Returns:
            dict: A dictionary of nodes to add to the open list.
        """

        col = self.distance_matrix[active]

        idxs = np.intersect1d(np.where(col <= max_dist)[0], np.where(col > min_dist)[0])
        open_elems = np.where(_open_costs != np.inf)[0]

        neighbours = np.setdiff1d(idxs, open_elems)

        add_open = {}
        temp_t_costs = {}

        for n in neighbours:

            col_n = self.distance_matrix[n].copy()

            if any(col_n[self.destination_idxs] == np.inf):
                g_dist = 0
            else:
                g_dist = np.min(col_n[self.destination_idxs])

            idxs_n = np.intersect1d(np.where(col_n > min_dist)[0], np.where(col_n <= max_dist * 2)[0])
            open_n = np.intersect1d(idxs_n, open_elems)

            for idx in open_n:
                path = LineString((self.running_nodes[n][:2], self.running_nodes[idx][:2]))

                hdg = gf.calculate_initial_compass_bearing(*self.running_nodes[idx, :-1], *self.running_nodes[n, :-1])

                v_hdg = True if self.last_angles[idx] == -1 else abs(hdg - self.last_angles[idx]) <= 80

                if not v_hdg:
                    continue

                if n in self.destination_idxs:
                    n_hdg = gf.calculate_initial_compass_bearing(
                        *self.running_nodes[n, :-1], *self.vertiports[self.goal][:-1]
                    )
                    if abs(n_hdg - hdg) > 80:
                        continue

                if is3D:
                    valid = all(
                        gf.evaluate_intersection(
                            path,
                            self.running_nodes[n],
                            self.running_nodes[idx],
                            poly,
                            h,
                        )
                        for (poly, h) in self.experience_zones
                    )
                else:
                    valid = not any(poly.intersection(path) for (poly, _) in self.experience_zones)

                if is3D:
                    alt_change = self.running_nodes[n][-1] - self.running_nodes[idx][-1]

                    dist = (col_n[idx] if col_n[idx] != np.inf else 0) * 1000
                    time_to_travel = dist / self.cruise_speed

                    if (abs(alt_change) / self.climb_rate) * 60 > time_to_travel:
                        valid = False

                    alt_change = max(alt_change, 0)

                else:
                    alt_change = 0

                t_cost = _t_costs[idx] + col_n[idx]
                if valid and ((n in add_open and temp_t_costs[n] > t_cost) or (n not in add_open)):
                    add_open[n] = (g_dist, idx, hdg)
                    temp_t_costs[n] = t_cost

            if n in self.destination_idxs and n in add_open:
                print("Here")
                return add_open
        return add_open

    def build_all_path_objects(self) -> np.ndarray:
        """Builds all path objects based on the generated parents.

        Returns:
            np.ndarray: A list of paths represented as numpy arrays.

        Raises:
            ValueError: Raised when the path has not been generated or is not valid.

        Example:
            ```python
            fmt = FMTStar()
            paths = fmt.build_all_path_objects()
            for path in paths:
                print(path)
            ```
        """

        if len(self._parents) == 0 or all(self._parents == -1):
            raise ValueError("No path objects exist")

        paths = []
        visited = set()
        for i, p in enumerate(self._parents):
            if i in visited:
                continue

            active = i

            if active in self.destination_idxs:
                path = [self.vertiports[self.goal]]
            else:
                path = []

            while active not in visited and active != -1:
                path.insert(0, self.running_nodes[active])
                if active in self.origin_idxs and self._parents[active] == -1:
                    path.insert(0, self.vertiports[self.start])
                active = self._parents[active]

            if len(path) > 1:
                paths.append(np.array(path))

        return paths

    def build_path_object(self) -> np.ndarray:
        """Builds a path object based on the generated parents.

        Raises:
            ValueError: Raised when the path has not been generated or is not valid.

        Returns:
            np.ndarray: Returns the path as a numpy array.
        """
        if len(self._parents) == 0 or all(self._parents[self.destination_idxs] == -1):
            raise ValueError("Path has not been generated/is not valid")

        active = np.where(self._parents[self.destination_idxs] != -1)[0]

        if len(active) == 1:
            active = self.destination_idxs[active[0]]

            path = [self.vertiports[self.goal]]
            while active != -1:
                path.insert(0, self.running_nodes[active])
                active = self._parents[active]

            path.insert(0, self.vertiports[self.start])
            return np.array(path)
        else:
            best_distance = np.inf
            best_path = None

            for a in active:
                path = [self.vertiports[self.goal]]
                distance = 0
                a = self.destination_idxs[a]
                while a != -1:
                    path.insert(0, self.running_nodes[a])
                    distance += gf.haversine_distance(path[0][:-1], path[1][:-1])
                    a = self._parents[a]

                path.insert(0, self.vertiports[self.start])
                distance += gf.haversine_distance(path[0][:-1], path[1][:-1])

                if distance < best_distance:
                    best_distance = distance
                    best_path = path

            return np.array(best_path)

    def validate_nodes(self, nodes: np.ndarray, is3D: bool, levels=None) -> np.ndarray:
        """Validates the given nodes.

        Args:
            nodes (np.ndarray): List of nodes to validate.
            is3D (bool): If True, the nodes will be 3-dimensional.
            levels (_type_, optional): List of levels. Defaults to None.

        Raises:
            ValueError: Raised when the levels are not specified and the nodes are 3-dimensional.

        Returns:
            np.ndarray: A list of valid nodes.
        """
        print("Validating nodes...")
        s = perf_counter()

        new_nodes = []

        if not is3D:
            init = len(nodes)
            for point in nodes:
                shp_point = Point(point)

                if not any(poly.contains(shp_point) for (poly, _) in self.experience_zones):
                    new_nodes.append(point)
        else:
            init = nodes.shape[0] * nodes.shape[1]

            if levels is None:
                raise ValueError("Can not validate 3-dimensional nodes if levels is None")
            if len(nodes.shape) != 3:
                nodes = np.expand_dims(nodes, axis=0)

            for l in range(len(levels)):
                for point in nodes[l]:
                    shp_point = Point(point[:2])

                    if all(point[2] > h or (not poly.contains(shp_point)) for (poly, h) in self.experience_zones):
                        new_nodes.append(point)

        print(f"Nodes validated in {round(perf_counter() - s,2)} seconds, yielding {len(new_nodes)}/{init} valid nodes")

        return np.array(new_nodes)
