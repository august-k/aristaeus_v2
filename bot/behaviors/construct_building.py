"""CombatBehavior that involves placing buildings."""
from dataclasses import dataclass

from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.unit import Unit

from ares import AresBot
from ares.behaviors.combat.individual.combat_individual_behavior import (
    CombatIndividualBehavior,
)
from ares.consts import ID
from ares.managers.manager_mediator import ManagerMediator


@dataclass
class ConstructBuilding(CombatIndividualBehavior):
    """Construct a building.

    Attributes
    ----------
    unit : Unit
        The unit to use

    """

    unit: Unit

    def execute(self, ai: "AresBot", config: dict, mediator: ManagerMediator) -> bool:
        tracker_dict = mediator.get_building_tracker_dict
        if self.unit.tag in tracker_dict:
            if ai.can_afford(tracker_dict[self.unit.tag][ID]):
                return True
        return False
