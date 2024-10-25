"""CombatBehavior that involves placing buildings."""
from dataclasses import dataclass

from ares import AresBot
from ares.behaviors.combat.individual.combat_individual_behavior import (
    CombatIndividualBehavior,
)
from ares.managers.manager_mediator import ManagerMediator
from sc2.ids.unit_typeid import UnitTypeId as UnitID, UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit


@dataclass
class ConstructBuilding(CombatIndividualBehavior):
    """Construct a building.

    Attributes
    ----------
    unit : Unit
        The unit to use

    """

    unit: Unit
    structure_type: UnitID
    pos: Point2
    assign_role: bool

    def execute(self, ai: "AresBot", config: dict, mediator: ManagerMediator) -> bool:
        if isinstance(self.structure_type, UnitID) and ai.can_afford(
            self.structure_type
        ):
            mediator.build_with_specific_worker(
                worker=self.unit,
                structure_type=self.structure_type,
                pos=self.pos,
                assign_role=self.assign_role,
            )
            return True
        return False
