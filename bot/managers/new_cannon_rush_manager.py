"""Refactor of CannonRush Manager"""
from typing import Dict, Set, TYPE_CHECKING, Any, Union, List, Optional

import numpy as np
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.position import Point2, Point3
from sc2.unit import Unit

from ares.behaviors.combat import CombatManeuver
from ares.consts import ManagerName, ManagerRequestType, UnitRole

from ares.managers.manager import Manager
from ares.managers.manager_mediator import IManagerMediator, ManagerMediator


from ares.behaviors.combat.individual import (
    KeepUnitSafe,
    PathUnitToTarget,
)

from ares.managers.path_manager import MapData
from bot.behaviors.construct_building import ConstructBuilding
from bot.tools.new_cannon_placement import WallCreation, WallData

if TYPE_CHECKING:
    from ares import AresBot


class CannonRushManager(Manager, IManagerMediator):
    """Handle cannon rush tasks."""

    map_data: MapData
    wall_creation: WallCreation

    def __init__(
        self,
        ai: "AresBot",
        config: Dict,
        mediator: ManagerMediator,
    ) -> None:
        """Set up the manager.

        Parameters
        ----------
        ai :
            Bot object that will be running the game
        config :
            Dictionary with the data from the configuration file
        mediator :
            ManagerMediator used for getting information from other managers.

        Returns
        -------

        """
        super(CannonRushManager, self).__init__(ai, config, mediator)

        self.custom_build_order_complete: bool = False
        self.cannon_rush_complete: bool = False

        self.initial_cannon_placed: bool = False
        self.high_ground_cannon_location: Optional[Point2] = None

        self.walls: Dict[str, WallData] = {}
        # Key is the wall_id of the Wall, value is the  Probe's tag
        # It's allowed, although not recommended, to have multiple probes assigned to
        # the same wall.
        self.wall_to_probe: Dict[str, int] = {}
        self.primary_wall_id: Optional[str] = None
        self.high_ground_wall_id: Optional[str] = None

        # cannon worker roles
        self.cannon_placers = UnitRole.CONTROL_GROUP_ONE
        self.chaos_probes = UnitRole.CONTROL_GROUP_TWO
        self.cannon_roles = {self.cannon_placers, self.chaos_probes}

        # TODO: load based on map
        self.initial_cannon = Point2((32, 99))

    def initialise(self) -> None:
        self.map_data: MapData = self.manager_mediator.get_map_data_object
        self.wall_creation: WallCreation = WallCreation(
            self.ai, self.map_data, self.manager_mediator
        )
        self.primary_wall_id = self.add_new_wall(
            enclose_position=self.initial_cannon,
            blocked_positions=[self.initial_cannon],
        )

    def manager_request(
        self,
        receiver: ManagerName,
        request: ManagerRequestType,
        reason: str = None,
        **kwargs,
    ) -> Any:
        """To be implemented by managers that inherit from IManagerMediator interface.

        Parameters
        ----------
        receiver :
            The Manager the request is being sent to.
        request :
            The Manager that made the request
        reason :
            Why the Manager has made the request
        kwargs :
            If the ManagerRequest is calling a function, that function's keyword
            arguments go here.

        Returns
        -------

        """
        return self.manager_requests_dict[request](kwargs)

    async def update(self, _iteration: int) -> None:
        """Update cannon rush status, tasks, and objectives.

        Parameters
        ----------
        _iteration :
            The current game iteration.

        Returns
        -------

        """
        self.debug_coordinates()

        # don't do anything if the build order is still running
        if not self.ai.build_order_runner.build_completed:
            return

        # run the custom build order while it's needed
        if not self.custom_build_order_complete:
            self.custom_build_order_complete = self.run_custom_build_order()

            grid = self.manager_mediator.get_ground_avoidance_grid
            rusher_tags = self.manager_mediator.get_units_from_roles(
                roles=self.cannon_roles
            )

            for worker in rusher_tags:
                # create and register a maneuver to keep this probe safe while
                # pathing towards the initial cannon placement
                self.ai.register_behavior(
                    self.create_path_if_safe_maneuver(
                        unit=worker,
                        grid=grid,
                        target=self.initial_cannon,
                    )
                )
            return

        # update walls
        for wall in self.walls.values():
            self.wall_creation.update_existing_wall(wall)

        # protocol for getting the first cannon placed
        if not self.initial_cannon_placed:
            self.initial_cannon_placed = self.secure_initial_cannon()
        else:
            self.secure_high_ground()

    def run_custom_build_order(self) -> bool:
        """Run the build order from here rather than the BuildOrderRunner.

        Parameters
        ----------

        Returns
        -------
        bool :
            Whether the build order is completed.

        """
        structure_dict = self.manager_mediator.get_own_structures_dict
        # if the forge is started, we're done with the build order
        if self.ai.structure_present_or_pending(UnitID.FORGE):
            return True

        # get our first pylon at home
        if not self.ai.structure_present_or_pending(UnitID.PYLON):
            if self.ai.minerals >= 25:
                if tag := self.build_structure_at_home_ramp(
                    structure_type=UnitID.PYLON,
                    location=list(self.ai.main_base_ramp.corner_depots)[0],
                    assign_to=self.cannon_placers,
                ):
                    # assign this Probe to the primary wall
                    if self.primary_wall_id and self.primary_wall_id in self.walls:
                        self.wall_to_probe[self.primary_wall_id] = tag
        else:
            # build workers up to 16
            if self.ai.supply_workers + self.ai.already_pending(UnitID.PROBE) < 16:
                if nexus := structure_dict[UnitID.NEXUS]:
                    if self.ai.minerals >= 50 and nexus[0].is_idle:
                        self.ai.train(UnitID.PROBE)
            # then build the forge
            else:
                if self.ai.minerals >= 75:
                    self.build_structure_at_home_ramp(
                        structure_type=UnitID.FORGE,
                        location=self.ai.main_base_ramp.barracks_in_middle,
                        assign_to=self.chaos_probes,
                    )
        return False

    def build_structure_at_home_ramp(
        self,
        structure_type: UnitID,
        location: Point2,
        assign_to: UnitRole,
    ) -> Optional[int]:
        """Construct buildings at the home ramp.

        Parameters
        ----------
        structure_type : UnitID
            Which structure to build.
        location : Point2
            Where to build the structure.
        assign_to : UnitRole
            What role the Probe should have after the building

        Returns
        -------
        Optional[int] :
            The tag of the Probe used to build something, if any.

        """
        if worker := self.manager_mediator.select_worker(target_position=location):
            self.manager_mediator.assign_role(tag=worker.tag, role=assign_to)
            self.manager_mediator.build_with_specific_worker(
                worker=worker,
                structure_type=structure_type,
                pos=location,
                assign_role=False,
            )
            return worker.tag

    def secure_initial_cannon(self) -> bool:
        """Place the first cannon that will be used as our anchor for the cannon rush.

        Parameters
        ----------

        Returns
        -------
        bool :
            Whether this step should be considered completed.

        """
        if self.ai.structure_type_build_progress(UnitID.PHOTONCANNON) >= 0.95:
            return True

        wall = self.walls[self.primary_wall_id]

        # override wall update if we can place a cannon
        if self.ai.tech_requirement_progress(
            UnitID.PHOTONCANNON
        ) == 1 and self.ai.state.psionic_matrix.covers(self.initial_cannon):
            wall.set_next_building(
                pos=self.initial_cannon,
                type_id=UnitID.PHOTONCANNON,
                wall_will_complete=False,
            )
        # place the next building
        self.manager_mediator.build_with_specific_worker(
            worker=self.ai.unit_tag_dict[self.wall_to_probe[self.primary_wall_id]],
            structure_type=wall.next_building_type,
            pos=wall.next_building_location,
            assign_role=False,
        )

        avoidance_grid = self.manager_mediator.get_ground_avoidance_grid
        enemy_nat = self.manager_mediator.get_enemy_nat

        # keep the Probe doing stuff
        self.ai.register_behavior(
            self.create_path_if_safe_maneuver(
                unit=self.ai.unit_tag_dict[self.wall_to_probe[self.primary_wall_id]],
                grid=avoidance_grid,
                target=self.initial_cannon,
            )
        )

        # keep chaos probes active
        for unit in self.manager_mediator.get_units_from_role(role=self.chaos_probes):
            self.ai.register_behavior(
                self.create_path_if_safe_maneuver(
                    unit=unit, grid=avoidance_grid, target=enemy_nat
                )
            )
        return False

    def perform_wall_micro(self, probe: Unit, wall: WallData, grid: np.ndarray):
        """Given the Probe and wall it's supposed to build, execute related tasks."""

    def add_new_wall(
        self,
        enclose_position: Point2,
        blocked_positions: Optional[List[Point2]] = None,
        wall_closest_to: Optional[Point2] = None,
    ) -> str:
        """Create a new WallData object and add it to our wall dictionary.

        Parameters
        ----------
        enclose_position : Point2
            The position we want to wall off, typically the cannon placement.
        blocked_positions : Optional[List[Point2]]
            Positions to avoid using in the wall.
        wall_closest_to : Optional[Point2]
            Point we want to defend the wall from

        Returns
        -------
        str :
            The wall_id of the created Wall.

        """
        new_wall = self.wall_creation.create_new_wall(
            enclose_position=enclose_position,
            blocked_positions=blocked_positions,
            wall_closest_to=wall_closest_to,
        )
        self.walls[new_wall.wall_id] = new_wall
        return new_wall.wall_id

    def remove_wall(self, wall: WallData) -> None:
        """Remove a Wall from the tracking dictionaries.

        Parameters
        ----------
        wall : WallData
            The wall to remove.

        Returns
        -------

        """
        del self.wall_to_probe[wall.wall_id]

        # no longer a primary wall since this one is being removed
        if wall.wall_id == self.primary_wall_id:
            self.primary_wall_id = None

        del self.walls[wall.wall_id]

    def remove_unit(self, unit_tag: int) -> None:
        """Remove a Wall from the tracking dictionaries.

        Parameters
        ----------
        unit_tag : int
            The tag of the unit we need to remove

        Returns
        -------

        """
        to_remove: List[str] = []
        for wall_id in self.wall_to_probe:
            if self.wall_to_probe[wall_id] == unit_tag:
                to_remove.append(wall_id)

        for wall_id in to_remove:
            del self.walls[wall_id]

    @staticmethod
    def create_path_if_safe_maneuver(
        unit: Unit, grid: np.ndarray, target: Point2
    ) -> CombatManeuver:
        """CombatManeuver for Probes to moving towards the cannon location if safe.

        Parameters
        ----------
        unit : Unit
            The Probe being micro'd.
        grid : np.ndarray
            Grid to use for pathing.
        target : Point2
            Where the Probe is going.

        Returns
        -------
        CombatManeuver :
            The completed maneuver.

        """
        probe_maneuver: CombatManeuver = CombatManeuver()
        probe_maneuver.add(KeepUnitSafe(unit=unit, grid=grid))
        probe_maneuver.add(ConstructBuilding(unit=unit))
        probe_maneuver.add(
            PathUnitToTarget(
                unit=unit,
                grid=grid,
                target=target,
                sensitivity=1,
            )
        )
        return probe_maneuver

    def debug_coordinates(self):
        """Draw coordinates on the screen."""
        wall_points = []
        for wall_id in self.walls:
            wall_points.extend(self.walls[wall_id].wall_path)
        wall_points = set(wall_points)
        for i in range(
            int(self.initial_cannon.x - 15), int(self.initial_cannon.x + 15)
        ):
            for j in range(
                int(self.initial_cannon.y - 15), int(self.initial_cannon.y + 15)
            ):
                point = Point2((i, j))
                color = (15, 255, 15) if point in wall_points else (127, 0, 255)
                height = self.ai.get_terrain_z_height(point)
                p_min = Point3((point.x, point.y, height + 0.1))
                p_max = Point3((point.x + 1, point.y + 1, height + 0.1))
                self.ai.client.debug_box_out(p_min, p_max, Point3((0, 0, 127)))
                if height >= 9:
                    self.ai.client.debug_text_world(
                        f"x={i}\ny={j}",
                        Point3((p_min.x, p_min.y + 0.75, p_min.z)),
                        color,
                    )

    def secure_high_ground(self):
        """
        Secure the high ground near the initial cannon.

        Returns
        -------

        """
        if not self.high_ground_wall_id:
            if high_ground_cannon_location := (
                self.wall_creation.building_placement.get_high_ground_point_near(
                    position=self.initial_cannon
                )
            ):
                self.high_ground_wall_id = self.add_new_wall(
                    enclose_position=high_ground_cannon_location,
                    blocked_positions=[high_ground_cannon_location],
                    wall_closest_to=self.ai.enemy_start_locations[0],
                )
                worker = self.manager_mediator.get_units_from_role(
                    role=self.chaos_probes
                ).first
                self.wall_to_probe[self.high_ground_wall_id] = worker.tag
