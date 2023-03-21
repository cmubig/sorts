# ------------------------------------------------------------------------------
# @file:    common.py
# @brief:   This file contains the implementation common utility classes and 
#           function needed by the modules in sorts.
# ------------------------------------------------------------------------------
EPS = 1e-8
FORMAT = '[%(asctime)s: %(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
KM_TO_M = 1000
G_ACC = 9.81
LOSS_OF_SEPARATION_THRESH = 0.5

class Config:
    """ A class for holding configuration parameters. """
    def __init__(self, config):
        self.MAIN = dotdict(config)

        self.DATA = None
        if self.MAIN.data:
            self.DATA = dotdict(self.MAIN.data)
        
        self.GAME = None 
        if self.MAIN.game:
            self.GAME = dotdict(self.MAIN.game)
        
        self.SEARCH_POLICY = None
        if self.MAIN.search_policy:
            self.SEARCH_POLICY = dotdict(self.MAIN.search_policy)
        
        self.SOCIAL_POLICY = None
        if self.MAIN.social_policy:
            self.SOCIAL_POLICY = dotdict(self.MAIN.social_policy)

        self.VISUALIZATION = None
        if self.MAIN.visualization:
            self.VISUALIZATION = dotdict(self.MAIN.visualization)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__