from cardshark.named_object import NamedObject
from abc import ABC, abstractmethod
import typing


class Player(NamedObject, ABC):
    def __init__(self, **kwargs):
        NamedObject.__init__(self, **kwargs)

        self.reset()

    @abstractmethod
    def reset(self):
        pass

    def giveReward(self, reward: float) -> None:
        """Give reward to this player for a job well done"""

        self.debug("Giving reward:", reward)
        self.rewardAccum += reward
        self.debug("  Reward after:", self.rewardAccum)

    def claimReward(self) -> float:
        """Get the total reward given to this player until now and set it's reward back to 0"""
        self.debug("Claiming reward:", self.rewardAccum)
        ret = self.rewardAccum
        self.rewardAccum = 0.0
        return ret

    def checkReward(self) -> float:
        """Check the reward that this player has accumulated but without resetting it to 0"""
        return self.rewardAccum

    def get_info_str(self) -> str:
        return ""

    def __str__(self) -> str:
        retStr = "Name: " + self.name + "\n" + self.get_info_str()

        return retStr
