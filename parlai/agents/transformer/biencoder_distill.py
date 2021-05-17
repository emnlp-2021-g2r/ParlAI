from typing import Optional

from parlai.agents.transformer.g2r_distill import G2RDistillMixin
from parlai.agents.transformer.biencoder import BiencoderAgent
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser


class BiencoderDistillAgent(G2RDistillMixin, BiencoderAgent):
    """
    Modified version of biencoder distill agent
    that uses candidate & candidate scores while training.
    """
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add command-line arguments specifically for this agent.
        """
        BiencoderAgent.add_cmdline_args(parser, partial_opt=partial_opt)
        G2RDistillMixin.add_cmdline_args(parser, partial_opt=partial_opt)
        return parser
