"""Manage cannon placements."""
import json
import uuid
from collections import defaultdict
from os import getcwd, path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from ares import AresBot, ManagerMediator
from map_analyzer.MapData import MapData
from map_analyzer.Pather import draw_circle
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.position import Point2
from scipy.signal import convolve2d

from bot.consts import (
    BLOCKING,
    DESIRABILITY_KERNEL,
    INVALID_BLOCK,
    POINTS,
    SCORE,
    WEIGHT,
)
from bot.tools.grids import modify_two_by_two


class WallData:
    """Keep track of information for a particular wall off."""

    __slots__ = (
        "buildings_used",
        "enclose_point",
        "next_building_is_final",
        "next_building_location",
        "next_building_type",
        "start_point",
        "wall_complete",
        "wall_id",
        "wall_path",
    )

    def __init__(self, enclose_point: Point2):
        # key is the building position, value is the type of building
        self.buildings_used: Dict[Point2, UnitID] = {}
        self.enclose_point: Point2 = enclose_point
        self.next_building_is_final: bool = False
        self.next_building_location: Optional[Point2] = None
        self.next_building_type: Optional[UnitID] = None
        self.start_point: Optional[Point2] = None
        self.wall_complete: bool = False
        self.wall_id: str = uuid.uuid4().hex
        self.wall_path: Optional[List[Point2]] = None

    def building_placed(self, pos: Point2, type_id: UnitID) -> None:
        """Register that a building has been placed for this wall.

        Parameters
        ----------
        pos : Point2
            Where the building was placed.
        type_id : UnitID
            What type of building was placed.

        Returns
        -------

        """
        self.buildings_used[pos] = type_id
        self.next_building_location = None
        self.next_building_type = None

    def mark_as_completed(self) -> None:
        """Perform bookkeeping to show that this wall is finished.

        Returns
        -------

        """
        self.next_building_location = None
        self.next_building_type = None
        self.wall_complete = True

    def set_new_wall_path(self, new_path: List[Point2]) -> None:
        """Update the current path with the new one.

        Parameters
        ----------
        new_path : List[Point2]
            The new path.

        Returns
        -------

        """
        # don't overwrite the path if there isn't one now
        if new_path:
            self.wall_path = new_path
            self.start_point = new_path[0]

    def set_next_building(
        self, pos: Point2, type_id: UnitID, wall_will_complete: bool
    ) -> None:
        """Update the next building fields.

        Parameters
        ----------
        pos : Point2
            The position of the building.
        type_id : UnitID
            The type of the building.
        wall_will_complete : bool
            Whether this building will complete the wall.

        Returns
        -------

        """
        self.next_building_is_final = wall_will_complete
        self.next_building_location = pos
        self.next_building_type = type_id

    def clear_next_building(self) -> None:
        """Reset the next building fields.

        Returns
        -------

        """
        self.next_building_is_final = False
        self.next_building_location = None
        self.next_building_type = None


class WallCreation:
    """Find places to put wall components.

    This class should:
            - create new walls
            - update existing walls
    """

    def __init__(self, ai: AresBot, map_data: MapData, mediator: ManagerMediator):
        """Set up the walling locations class.

        Parameters
        ----------
        ai
        map_data
        mediator
        """
        self.ai: AresBot = ai
        self.map_data: MapData = map_data
        self.mediator: ManagerMediator = mediator

        self.building_placement: BuildingPlacement = BuildingPlacement(
            ai, map_data, mediator
        )

        # TODO: move these to the config or use function parameters
        self.blocking_building_weight = 1
        self.non_blocking_building_weight = 1
        self.wall_grid_bound_size = 20
        self.terrain_weight = 1

    def _generate_basic_walling_grid(
        self, add_to_blocked_positions: Optional[List[Point2]] = None
    ) -> np.ndarray:
        """Create the pathing grid used for walling.

        Parameters
        ----------
        add_to_blocked_positions : Optional[List[Point2]]
            Placement positions of 2x2 buildings we don't want to use in pathing.

        Returns
        -------
        np.ndarray :
            The pathing grid.

        """
        # get the standard pathing grid
        basic_grid = self.map_data.get_walling_grid()

        # create a new grid of the same size
        basic_walling_grid = np.zeros(basic_grid.shape, dtype=np.float32)

        # swap pathable and unpathable points
        basic_walling_grid[np.where(basic_grid != np.inf)] = np.inf
        basic_walling_grid[np.where(basic_grid == np.inf)] = 1

        # block off the designated positions
        if add_to_blocked_positions:
            for pos in add_to_blocked_positions:
                modify_two_by_two(basic_walling_grid, pos, np.inf)

        return basic_walling_grid

    def _find_wall_path(
        self,
        cannon_placement: Point2,
        cannon_grid: np.ndarray,
        valid_blocks: Dict[Tuple[int, int], int],
        invalid_blocks: Dict[Tuple[int, int], int],
        wall_start_point: Optional[Point2] = None,
    ) -> Optional[List[Point2]]:
        """Given a location to place a cannon, find the path we want to wall.

        Parameters
        ----------
        cannon_placement : Point2
            The cannon we want to wall in.
        cannon_grid : np.ndarray
            The grid to use for walling.
        valid_blocks : Dict[Tuple[int, int], int]
            Dictionary of blocking positions to their convolution score.
        invalid_blocks : Dict[Tuple[int, int], int]
            Dictionary of non-blocking positions to their convolution score.
        wall_start_point : Optional[Point2]
            The point the wall path starts from, if it's known.

        Returns
        -------
        Optional[List[Point2]] :
            The path used for the wall if one is found.

        """
        # add potential placements to the grid used for pathing
        for pos in valid_blocks:
            modify_two_by_two(cannon_grid, Point2(pos), self.blocking_building_weight)

        for pos in invalid_blocks:
            modify_two_by_two(
                cannon_grid, Point2(pos), self.non_blocking_building_weight
            )

        # don't use the cannon as part of the wall
        modify_two_by_two(cannon_grid, cannon_placement, np.inf)

        # try to find a start point for the wall since we don't have one yet
        if not wall_start_point:
            wall_start_point = self._calculate_start_point(
                cannon_placement,
                self.map_data.get_pyastar_grid(),
                blacklist={cannon_placement},
            )

        # only run pathfinding if we have a start point
        if wall_start_point:
            return self.map_data.clockwise_pathfind(
                start=wall_start_point,
                goal=wall_start_point,
                origin=cannon_placement,
                grid=cannon_grid,
            )

        return None

    def _calculate_start_point(
        self,
        cannon_placement: Point2,
        grid: np.ndarray,
        blacklist: Optional[Set[Union[Point2, Tuple[int, int]]]] = None,
    ) -> Optional[Point2]:
        """Given the cannon we want to wall, find the start/end point for our path.

        Parameters
        ----------
        cannon_placement : Point2
            Location of the position we want to wall.
        grid : np.ndarray
            Pathing grid for the wall path.
        blacklist : Optional[Set[Union[Point2, Tuple[int, int]]]]
            Points that should not be considered as potential start point.

        Returns
        -------
        Optional[Point2] :
            The wall start location, if one was found.

        """
        start = None
        if not blacklist:
            blacklist = set()

        if blacklist:
            for pos in blacklist:
                # ensure the point isn't considered valid
                grid[pos] = 1

        point = (int(cannon_placement[0]), int(cannon_placement[1]))
        disk = tuple(draw_circle(point, 10, shape=grid.shape))
        target_weight_cond = np.abs(grid[disk]) == np.inf
        if np.any(target_weight_cond):
            possible_points = np.column_stack(
                (disk[0][target_weight_cond], disk[1][target_weight_cond])
            )

            closest_point_index = np.argmin(
                np.sum((possible_points - point) ** 2, axis=1)
            )
            start = tuple(possible_points[closest_point_index])

        return start

    def create_new_wall(
        self,
        enclose_position: Point2,
        blocked_positions: Optional[List[Point2]] = None,
        wall_closest_to: Optional[Point2] = None,
    ) -> WallData:
        """Create a new WallData object to wall off the given position.

        Parameters
        ----------
        enclose_position : Point2
            The point to enclose in the wall.
        blocked_positions : Optional[List[Point2]]
            Positions to avoid using in the wall.
        wall_closest_to : Optional[Point2]
            Point to place wall components near. Default is the bottom of the enemy main
            base ramp

        Returns
        -------
        WallData :
            Information about the wall.

        """
        # default to enemy main base ramp if we don't have a target
        if not wall_closest_to:
            wall_closest_to = self.mediator.get_enemy_ramp.bottom_center

        new_wall: WallData = WallData(enclose_position)
        grid, valid_blocks, invalid_blocks = self._get_grid_and_blocks(
            enclose_position, blocked_positions
        )

        new_wall.wall_path = self._find_wall_path(
            enclose_position, grid, valid_blocks, invalid_blocks
        )
        if new_wall.wall_path:
            new_wall.start_point = new_wall.wall_path[0]

        self._add_next_building(
            valid_blocks, invalid_blocks, new_wall, wall_closest_to, grid
        )

        return new_wall

    def update_existing_wall(
        self,
        wall: WallData,
        check_for_better_wall: bool = True,
        wall_closest_to: Optional[Point2] = None,
    ) -> None:
        """Perform next steps for an existing wall.

            - check whether the wall is completed
            - change the walling path if needed and allowed
            - select the location and type of the next building to place

        TODO: allow non-Pylon components

        Parameters
        ----------
        wall : WallData
            Information about the existing wall.
        check_for_better_wall : bool
            Whether the walling path should be re-evaluated before finding the next
            building placement.
        wall_closest_to : Optional[Point2]
            Point to place wall components near. Default is the bottom of the enemy main
            base ramp (handled in BuildingPlacement).

        Returns
        -------

        """
        # set the wall as completed and move on if it's finished
        if self._wall_is_finished(wall):
            wall.mark_as_completed()
            return

        grid, valid_blocks, invalid_blocks = self._get_grid_and_blocks(
            wall.enclose_point, [wall.enclose_point]
        )

        wall.set_new_wall_path(
            self._find_wall_path(wall.enclose_point, grid, valid_blocks, invalid_blocks)
        )

        # path might be longer, but the old path may not wall anymore
        # see if a shorter walling path has become available
        # if check_for_better_wall:
        #     if new_wall_path := self._find_wall_path(
        #         wall.enclose_point, grid, valid_blocks, invalid_blocks
        #     ):
        #         # there either wasn't a path before or the new one is shorter
        #         if not wall.wall_path or len(new_wall_path) < len(wall.wall_path):
        #             wall.set_new_wall_path(new_wall_path)

        self._add_next_building(
            valid_blocks, invalid_blocks, wall, wall_closest_to, grid
        )

    def _add_next_building(
        self, valid_blocks, invalid_blocks, wall, wall_closest_to, grid
    ):
        # add another component to the wall if we can
        if next_position := self.building_placement.get_next_walling_position(
            valid_blocks, invalid_blocks, set(wall.wall_path), wall_closest_to
        ):
            # TODO: verify this didn't break in the refactor, the previous code used
            #   np.inf instead of 1
            modify_two_by_two(grid, next_position, 1)
            wall_will_complete = self._wall_is_finished(wall, grid_override=grid)
            wall.set_next_building(next_position, UnitID.PYLON, wall_will_complete)

    def _wall_is_finished(
        self, wall: WallData, grid_override: Optional[np.ndarray] = None
    ) -> bool:
        """Check if the wall is finished.

        Parameters
        ----------
        wall : WallData
            The wall to check.
        grid_override : Optional[np.ndarray]
            The particular grid instead of the basic one. If not supplied, the basic
            walling grid from map_analyzer is used.

        Returns
        -------

        """
        if grid_override is None:
            grid = self._generate_basic_walling_grid([wall.enclose_point])
        else:
            grid = grid_override

        # see if there's already a path in the given grid
        if self.map_data.clockwise_pathfind(
            start=wall.start_point,
            goal=wall.start_point,
            origin=wall.enclose_point,
            grid=grid,
        ):
            return True
        return False

    def _get_grid_and_blocks(
        self,
        enclose_position: Union[Point2, Tuple[int, int]],
        blocked_positions: Optional[List[Union[Point2, Tuple[int, int]]]] = None,
    ):
        """Get the cannon walling grid and allowed Pylon placements.

        Parameters
        ----------
        enclose_position : Union[Point2, Tuple[int, int]]
            The position to wall off.
        blocked_positions : Optional[List[Union[Point2, Tuple[int, int]]]]
            Positions to avoid using in the wall.

        Returns
        -------

        """
        grid = self._generate_basic_walling_grid(
            add_to_blocked_positions=blocked_positions
        )
        valid_blocks, invalid_blocks = self.building_placement.perform_convolutions(
            x_bound=(
                enclose_position[0] - self.wall_grid_bound_size,
                enclose_position[0] + self.wall_grid_bound_size,
            ),
            y_bound=(
                enclose_position[1] - self.wall_grid_bound_size,
                enclose_position[1] + self.wall_grid_bound_size,
            ),
            terrain_height={self.ai.get_terrain_height(enclose_position)},
            pathing_grid=self.map_data.get_pyastar_grid(),
            avoid_positions=blocked_positions,
        )
        return grid, valid_blocks, invalid_blocks


class BuildingPlacement:
    """Find places to put buildings.

    This class should:
            - perform convolutions to find valid building placements
            - determine the next building placement for a wall

    """

    def __init__(self, ai: AresBot, map_data: MapData, mediator: ManagerMediator):
        """Set up the building placement class.

        Parameters
        ----------
        ai
        map_data
        mediator
        """
        self.ai: AresBot = ai
        self.map_data: MapData = map_data
        self.mediator: ManagerMediator = mediator

        __location__ = path.realpath(path.join(getcwd(), path.dirname(__file__)))
        with open(path.join(__location__, "hamming_weight_lookups.json"), "r") as f:
            hamming_lookup = json.load(f)
            self.hamming_lookup = {int(v): hamming_lookup[v] for v in hamming_lookup}

    def perform_convolutions(
        self,
        x_bound: Tuple[int, int],
        y_bound: Tuple[int, int],
        terrain_height: Set[int],
        pathing_grid: np.ndarray,
        avoid_positions: Optional[List[Union[Point2, Tuple[int, int]]]] = None,
    ) -> Tuple[Dict, Dict]:
        """Convolve grids and return the dictionaries.

        Parameters
        ----------
        x_bound : Tuple[int, int]
            Minimum and maximum values of x to include in the possible locations.
        y_bound : Tuple[int, int]
            Minimum and maximum values of y to include in the possible locations.
        terrain_height : Set[int]
            All terrain heights that are part of the target region.
        pathing_grid : np.ndarray
            map_analyzer-style ground pathing grid to use.
        avoid_positions: Optional[List[Union[Point2, Tuple[int, int]]]]
            Positions where buildings are planned and need their tiles considered
            invalid.

        Returns
        -------
        Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], int] :
            Valid and invalid positions as a dictionary of building placement to
            convolution result.

        """
        # get the grid and our boundaries
        grid = self.generate_convolution_grid(
            x_bound,
            y_bound,
            terrain_height,
            pathing_grid,
            avoid_positions,
        )
        x_min, _x_max = x_bound
        y_min, _y_max = y_bound

        # perform convolution and identify valid blocks
        placements = convolve2d(grid, DESIRABILITY_KERNEL, mode="valid")

        # set up location dictionaries
        valid_blocks: Dict[Tuple[int, int], int] = {}
        valid_non_blocking_positions: Dict[Tuple[int, int], int] = {}

        # go through the convolution result and filter into blocks and invalid blocks
        for x in range(placements.shape[0]):
            for y in range(placements.shape[1]):
                point = (x + x_min + 2, y + y_min + 2)
                score = placements[x][y]
                if score >= 4096:
                    # invalid placement
                    continue
                elif score in INVALID_BLOCK:
                    # valid placement, but it doesn't block
                    valid_non_blocking_positions[point] = score
                else:
                    valid_blocks[point] = score
        return valid_blocks, valid_non_blocking_positions

    def generate_convolution_grid(
        self,
        x_bound: Tuple[int, int],
        y_bound: Tuple[int, int],
        terrain_height: Set[int],
        pathing_grid: np.ndarray,
        avoid_positions: Optional[List[Union[Point2, Tuple[int, int]]]] = None,
    ) -> np.ndarray:
        """Generate the grids to convolve based on pathing and placement.

        Parameters
        ----------
        x_bound : Tuple[int, int]
            Minimum and maximum values of x to include in the possible locations.
        y_bound : Tuple[int, int]
            Minimum and maximum values of y to include in the possible locations.
        terrain_height : Set[int]
            All terrain heights that are part of the target region.
        pathing_grid : np.ndarray
            map_analyzer-style ground pathing grid to use.
        avoid_positions: Optional[List[Union[Point2, Tuple[int, int]]]]
            Positions where buildings are planned and need their tiles considered
            invalid.

        Returns
        -------
        np.ndarray :
            Grid of legal tiles to use, ready for convolution.

        """
        # create a grid of all valid tiles of the shape determined by x and y boundaries
        x_min, x_max = x_bound
        y_min, y_max = y_bound
        convolution_grid = np.ones((x_max - x_min + 1, y_max - y_min + 1))

        # if the tile can't actually be used for placements, set its value to 0
        for i in range(*x_bound):
            for j in range(*y_bound):
                if (
                    pathing_grid[i][j] == 1
                    and self.ai.game_info.placement_grid.data_numpy[j][i] == 1
                    and self.ai.game_info.terrain_height.data_numpy[j][i]
                    in terrain_height
                ):
                    convolution_grid[i - x_min][j - y_min] = 0

        # avoid using tiles where buildings are going
        # TODO: see if this broke something in the refactor,
        #   the comment and the code disagreed
        if avoid_positions:
            for pos in avoid_positions:
                if (x_min + 1 < pos[0] < x_max - 1) and (
                    y_min + 1 < pos[1] < y_max - 1
                ):
                    modify_two_by_two(
                        convolution_grid,
                        (pos[0] - x_min, pos[1] - y_min),
                        1,
                    )

        return convolution_grid

    def get_next_walling_position(
        self,
        valid_blocks: Dict[Tuple[int, int], int],
        invalid_blocks: Dict[Tuple[int, int], int],
        wall_path: Set[Point2],
        wall_closest_to: Optional[Point2] = None,
    ) -> Optional[Point2]:
        """Find the next position to place a building.

        Parameters
        ----------
        valid_blocks : Dict[Tuple[int, int], int]
            Dictionary of blocking positions to their convolution score.
        invalid_blocks : Dict[Tuple[int, int], int]
            Dictionary of non-blocking positions to their convolution score.
        wall_path : Set[Point2]
            Points used for the current wall.
        wall_closest_to: Point2
            Prioritize wall components nearest this position. If not supplied, the
            bottom of the enemy main base ramp will be used.

        Returns
        -------

        """
        pylon_usage = self._evaluate_walling_positions(
            valid_blocks, invalid_blocks, wall_path
        )

        # organize points by score
        scores = defaultdict(list)

        for point in pylon_usage:
            scores[pylon_usage[point][SCORE]].append(point)

        # nothing's useful so we didn't find anything
        if not scores or max(scores) == 0:
            return None

        possible_positions = np.array(scores[max(scores)])

        if not wall_closest_to:
            wall_closest_to = self.mediator.get_enemy_ramp.bottom_center

        # get the position with the highest score that's closest to the designated point
        pylon_pos = possible_positions[
            np.argmin(
                np.sum(
                    (possible_positions - wall_closest_to) ** 2,
                    axis=1,
                )
            )
        ]

        return Point2(pylon_pos)

    def _evaluate_walling_positions(
        self,
        valid_blocks: Dict[Tuple[int, int], int],
        invalid_blocks: Dict[Tuple[int, int], int],
        wall_path: Set[Point2],
    ) -> Dict:
        """Score possible locations based on usability for the given path.

        Currently, the score is just the number of tiles shared between the wall path
        and a Pylon placed at the position.

        Parameters
        ----------
        valid_blocks : Dict[Tuple[int, int], int]
            Dictionary of blocking positions to their convolution score.
        invalid_blocks : Dict[Tuple[int, int], int]
            Dictionary of non-blocking positions to their convolution score.
        wall_path : Set[Point2]
            Points used for the current wall.

        Returns
        -------
        Dict :
            Dictionary of the position to information about it.

        """
        scores = {}
        blocking = True
        for d in [valid_blocks, invalid_blocks]:
            for pos in d.keys():
                score = 0
                tiles = []
                for i in [-1, 0]:
                    for j in [-1, 0]:
                        tile = (pos[0] + i, pos[1] + j)
                        if tile in wall_path:
                            score += 1
                            tiles.append(tile)
                # don't bother recording the position if it doesn't help this wall
                if score >= 1:
                    scores[pos] = {
                        SCORE: score,
                        POINTS: tiles,
                        BLOCKING: blocking,
                        WEIGHT: self.hamming_lookup[d[pos]],
                    }
            # blocking = False
        return scores

    def get_high_ground_point_near(
        self, position: Point2, search_distance: float = 6.5
    ) -> Optional[Point2]:
        """Find some high ground near a cannon.

        Parameters
        ----------
        position : Point2
            The position we want to be near.
        search_distance : float
            How far away from the position the high ground point can be.

        Returns
        -------
        Optional[Point2] :
            The high ground point.

        """
        grid = self.ai.game_info.terrain_height.data_numpy
        point = (int(position[0]), int(position[1]))

        # get the target height (the height of the enemy main)
        target_height = self.ai.get_terrain_height(self.ai.enemy_start_locations[0])

        # check if any points within the search distance have the correct terrain height
        disk = tuple(draw_circle(point, search_distance, shape=grid.shape))
        target_weight_cond = np.logical_and(
            abs(np.abs(grid[disk]) - target_height) < 11,
            grid[disk] < np.inf,
        )

        if np.any(target_weight_cond):
            # convolve the region to find valid building placements
            blocks, non_blocks = self.perform_convolutions(
                x_bound=(
                    int(position.x - search_distance),
                    int(position.x + search_distance),
                ),
                y_bound=(
                    int(position.y - search_distance),
                    int(position.y + search_distance),
                ),
                terrain_height={target_height},
                pathing_grid=self.map_data.get_pyastar_grid(),
            )

            # return a point if it's a valid placement
            # TODO: pick a point rather than returning the first one found
            possible_points = np.column_stack(
                (disk[0][target_weight_cond], disk[1][target_weight_cond])
            )
            for point in blocks:
                if point in possible_points:
                    return Point2(point)
            for point in non_blocks:
                if point in possible_points:
                    return Point2(point)
        return None
