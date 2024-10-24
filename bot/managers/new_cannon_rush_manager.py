"""Refactor of CannonRush Manager"""
from typing import Dict, Set, TYPE_CHECKING, Any, Union, List, Optional

import numpy as np
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from ares.behaviors.combat import CombatManeuver
from ares.consts import ManagerName, ManagerRequestType, UnitRole, UnitTreeQueryType
from cython_extensions.units_utils import cy_closest_to, cy_sorted_by_distance_to
from ares.managers.manager import Manager
from ares.managers.manager_mediator import IManagerMediator, ManagerMediator
from bot.tools.cannon_placement import CannonPlacement

from bot.consts import (
    BLOCKING,
    DESIRABILITY_KERNEL,
    FINAL_PLACEMENT,
    INVALID_BLOCK,
    LOCATION,
    POINTS,
    SCORE,
    TYPE_ID,
    WEIGHT,
)
from ares.behaviors.combat.individual import (
    KeepUnitSafe,
    PathUnitToTarget,
    AttackTarget,
    AMove,
)

from ares.managers.path_manager import MapData
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

        self.initial_cannon_placed: bool = False
        self.walls: Dict[str, WallData] = {}
        # Key is the Probe's tag, value is the wall_id of the Wall it's assigned to.
        # It's allowed, although not recommended, to have multiple probes assigned to
        # the same wall.
        self.probe_to_wall: Dict[int, str] = {}
        self.primary_wall_id: Optional[str] = None

        # cannon worker roles
        self.cannon_placers = UnitRole.CONTROL_GROUP_ONE
        self.chaos_probes = UnitRole.CONTROL_GROUP_TWO
        self.cannon_roles = {self.cannon_placers, self.chaos_probes}

        # TODO: load based on map
        self.initial_cannon = Point2((31, 99))

    async def initialise(self) -> None:
        self.map_data: MapData = self.manager_mediator.get_map_data_object
        self.wall_creation: WallCreation = WallCreation(
            self.ai, self.map_data, self.manager_mediator
        )
        self.primary_wall_id = self.add_new_wall(enclose_position=self.initial_cannon)

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
                if worker.tag in self.probe_to_wall:
                    # let the wall micro handle this probe
                    self.perform_wall_micro(
                        worker, self.walls[self.probe_to_wall[worker.tag]], grid
                    )
                else:
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
            # self.cause_chaos()
            pass

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
        if UnitID.FORGE in structure_dict:
            return True

        # get our first pylon at home
        if len(structure_dict[UnitID.PYLON]) == 0:
            if self.ai.minerals >= 25:
                if tag := self.build_structure_at_home_ramp(
                    structure_type=UnitID.PYLON,
                    location=list(self.ai.main_base_ramp.corner_depots)[0],
                ):
                    # assign this Probe to the primary wall
                    if self.primary_wall_id and self.primary_wall_id in self.walls:
                        self.probe_to_wall[tag] = self.primary_wall_id
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
                    )
        return False

    def build_structure_at_home_ramp(
        self,
        structure_type: UnitID,
        location: Point2,
    ) -> Optional[int]:
        """Construct buildings at the home ramp.

        Parameters
        ----------
        structure_type : UnitID
            Which structure to build.
        location : Point2
            Where to build the structure.

        Returns
        -------
        Optional[int] :
            The tag of the Probe used to build something, if any.

        """
        if worker := self.manager_mediator.select_worker(target_position=location):
            # assign first worker to cannons, subsequent workers to chaos
            self.manager_mediator.assign_role(
                tag=worker.tag,
                role=self.cannon_placers
                if structure_type == UnitID.PYLON
                else self.chaos_probes,
            )
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

    def perform_wall_micro(self, probe: Unit, wall: WallData, grid: np.ndarray):
        """Given the Probe and wall it's supposed to build, execute related tasks."""

    def add_new_wall(
        self, enclose_position: Point2, blocked_positions: Optional[List[Point2]] = None
    ) -> str:
        """Create a new WallData object and add it to our wall dictionary.

        Parameters
        ----------
        enclose_position : Point2
            The position we want to wall off, typically the cannon placement.
        blocked_positions : Optional[List[Point2]]
            Positions to avoid using in the wall.

        Returns
        -------
        str :
            The wall_id of the created Wall.

        """
        new_wall = self.wall_creation.create_new_wall(
            enclose_position=enclose_position, blocked_positions=blocked_positions
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
        # can't remove while iterating, no matter how many times I try
        tags_to_remove = [
            tag for tag in self.probe_to_wall if self.probe_to_wall[tag] == wall.wall_id
        ]
        for tag in tags_to_remove:
            del self.probe_to_wall[tag]

        # no longer a primary wall since this one is being removed
        if wall.wall_id == self.primary_wall_id:
            self.primary_wall_id = None

        del self.walls[wall.wall_id]

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
        probe_maneuver.add(PathUnitToTarget(unit=unit, grid=grid, target=target))
        return probe_maneuver
