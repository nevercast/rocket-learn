class Backend(object):
    def __init__(self, *args, **kwargs):
        pass


    async def get_worker_configuration(self) -> dict:
        """ Returns a dictionary with the configuration of the worker.
            Keys:
            - gamemode: 'duel', 'doubles', 'standard'
        """
        raise NotImplementedError()


    async def get_latest_parameters(self) -> bytes:
        """ Gets the latest parameters for the currently training model """
        raise NotImplementedError()


    async def get_opponent(self, model_index: int) -> bytes:
        """ Returns the model with the given index.
            If the model does not exist, returns None.
        """
        raise NotImplementedError()


    # Needs to know the game mode, the backend can probably handle that.
    # If it requires an argument now, it'll break workers if it doesn't require an argument later.
    async def get_match_versions(self) -> tuple:
        """ Gets the versions of the models that should play the next match.

            Returns a 2n tuple, one for each team. Containing a tuple of model versions.
            The versions are ints.

            example: ( (1, 2), (3, 4) ) # team 1: (1, 2), team 2: (3, 4)
        """
        raise NotImplementedError()


    async def push_rollout(self, rollout: bytes):
        """ Pushes a rollout to the backend.
            rollout: bytes - the rollout to push.
        """
        raise NotImplementedError()


    async def push_evaluation(self, blue_team_versions, orange_team_versions, blue_team_score, orange_team_score):
        """ Pushes an evaluation to the backend.
            blue_team_versions: tuple - the versions of the blue team models.
            orange_team_versions: tuple - the versions of the orange team models.
            blue_team_score: int - the score of the blue team.
            orange_team_score: int - the score of the orange team.
        """
        raise NotImplementedError()