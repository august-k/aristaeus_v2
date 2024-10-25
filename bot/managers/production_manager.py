from typing import TYPE_CHECKING

from ares.behaviors.macro import (
    AutoSupply,
    BuildStructure,
    BuildWorkers,
    ExpansionController,
    GasBuildingController,
    ProductionController,
    SpawnController,
    UpgradeController,
)
from ares.behaviors.macro.macro_plan import MacroPlan
from ares.managers.manager import Manager
from ares.managers.manager_mediator import ManagerMediator
from cython_extensions.general_utils import cy_unit_pending
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.ids.upgrade_id import UpgradeId
from sc2.unit import Unit

if TYPE_CHECKING:
    from ares import AresBot


class ProductionManager(Manager):
    def __init__(
        self,
        ai: "AresBot",
        config: dict,
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
        super().__init__(ai, config, mediator)

        self._built_single_oracle: bool = False
        self._built_extra_production_pylon: bool = False
        # can use a single chrono for the oracle
        self._oracle_chrono: bool = False

    @property
    def army_comp(self) -> dict:
        return {UnitID.TEMPEST: {"proportion": 1.0, "priority": 0}}

    @property
    def upgrade_list(self) -> list[UpgradeId]:
        return [
            UpgradeId.TEMPESTGROUNDATTACKUPGRADE,
            UpgradeId.PROTOSSAIRWEAPONSLEVEL1,
            UpgradeId.PROTOSSAIRARMORSLEVEL1,
            UpgradeId.PROTOSSAIRWEAPONSLEVEL2,
            UpgradeId.PROTOSSAIRARMORSLEVEL2,
            UpgradeId.PROTOSSAIRWEAPONSLEVEL3,
            UpgradeId.PROTOSSAIRARMORSLEVEL3,
        ]

    @property
    def max_probes(self) -> int:
        return min(80, 22 * len(self.ai.townhalls))

    async def update(self, iteration: int) -> None:
        """Handle production.

        Parameters
        ----------
        iteration :
            The game iteration.
        """

        if not self._built_extra_production_pylon:
            self.ai.register_behavior(
                BuildStructure(self.ai.start_location, UnitID.PYLON)
            )
            self._built_extra_production_pylon = True

        # use ares-sc2 macro behaviors for building pylons and units
        macro_plan: MacroPlan = MacroPlan()
        macro_plan.add(AutoSupply(base_location=self.ai.start_location))
        macro_plan.add(BuildWorkers(to_count=self.max_probes))
        macro_plan.add(
            UpgradeController(
                upgrade_list=self.upgrade_list, base_location=self.ai.start_location
            )
        )
        macro_plan.add(SpawnController(army_composition_dict=self.army_comp))
        macro_plan.add(
            ProductionController(
                army_composition_dict=self.army_comp,
                base_location=self.ai.start_location,
            )
        )
        if self.ai.minerals > 650:
            macro_plan.add(ExpansionController(to_count=100, max_pending=2))
        if self.ai.supply_workers >= 16:
            macro_plan.add(
                GasBuildingController(
                    to_count=200,
                    max_pending=4,
                )
            )
        self.ai.register_behavior(macro_plan)

        self._chrono_structures()

        # one off task to build an oracle
        if not self._built_single_oracle:
            structures_dict: dict[
                UnitID, list[Unit]
            ] = self.manager_mediator.get_own_structures_dict
            if (
                self.ai.can_afford(UnitID.ORACLE)
                and len(structures_dict[UnitID.FLEETBEACON]) > 0
                and self.ai.structures.filter(
                    lambda u: u.type_id == UnitID.STARGATE and u.is_ready and u.is_idle
                )
            ):
                self.ai.train(UnitID.ORACLE)
                self._built_single_oracle = True

    def _chrono_structures(self):
        """Decide what to chrono."""
        stargates: list[Unit] = self.manager_mediator.get_own_structures_dict[
            UnitID.STARGATE
        ]
        for nexus in self.ai.townhalls:
            if nexus.energy >= 50:
                non_idle_stargates = [
                    s
                    for s in stargates
                    if not s.is_idle
                    and not s.has_buff(BuffId.CHRONOBOOSTENERGYCOST)
                    and s.type_id == UnitID.STARGATE
                ]
                if len(non_idle_stargates) > 0:
                    if cy_unit_pending(self.ai, UnitID.TEMPEST):
                        nexus(
                            AbilityId.EFFECT_CHRONOBOOSTENERGYCOST,
                            non_idle_stargates[0],
                        )
                        return
                    if not self._oracle_chrono:
                        nexus(
                            AbilityId.EFFECT_CHRONOBOOSTENERGYCOST,
                            non_idle_stargates[0],
                        )
                        self._oracle_chrono = True
