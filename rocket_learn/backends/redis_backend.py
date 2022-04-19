from threading import Thread
from typing import Tuple
import zlib
import msgpack
from redis import Redis 
import cloudpickle as pickle
from functools import lru_cache

from rocket_learn.backends.abstract_backend import Backend

# Constants for consistent key lookup
QUALITIES = "qualities"
N_UPDATES = "num-updates"
SAVE_FREQ = "save-freq"

MODEL_LATEST = "model-latest"
VERSION_LATEST = "model-version"

ROLLOUTS = "rollout"
OPPONENT_MODELS = "opponent-models"
WORKER_IDS = "worker-ids"
CONTRIBUTORS = "contributors"


# Helper methods for easier changing of byte conversion
def _serialize(obj):
    return zlib.compress(msgpack.packb(obj))


def _unserialize(obj):
    return msgpack.unpackb(zlib.decompress(obj))


def _serialize_model(mdl):
    device = next(mdl.parameters()).device  # Must be a better way right?
    mdl_bytes = pickle.dumps(mdl.cpu())
    mdl.to(device)
    return mdl_bytes


def _unserialize_model(buf):
    agent = pickle.loads(buf)
    return agent


class RedisBackend(Backend):
    def __init__(self, redis: Redis):
        self.redis = redis 
        self.latest_model_version = None 
        self.latest_model_cache = None 
        self.uploader_thread = None 

    def get_worker_configuration(self) -> dict:
        """ Returns a dictionary with the configuration of the worker.
            Keys:
            - gamemode: 'duel', 'doubles', 'standard'
        """
        return 'doubles'


    def get_latest_model(self) -> Tuple[bytes, int]:
        """ Gets the latest parameters for the currently training model """
        latest_version = self.get_latest_version_number()
        if self.latest_model_version == None or self.latest_model_version != latest_version:
            self.latest_model_version = latest_version
            self.latest_model_cache = _unserialize_model(self.redis.get(MODEL_LATEST))
        return self.latest_model_cache, latest_version


    def get_latest_version_number(self) -> int:
        """ Gets the latest version number of the model. """
        return self.redis.get(VERSION_LATEST)


    @lru_cache(maxsize=8)
    def get_opponent(self, model_index: int) -> bytes:
        """ Returns the model with the given index.
            If the model does not exist, returns None.
        """
        assert isinstance(model_index, int)
        return _unserialize_model(self.redis.lindex(OPPONENT_MODELS, model_index))


    # Needs to know the game mode, the backend can probably handle that.
    # If it requires an argument now, it'll break workers if it doesn't require an argument later.
    def get_match_versions(self) -> tuple:
        """ Gets the versions of the models that should play the next match.

            Returns a 2n tuple, one for each team. Containing a tuple of model versions.
            The versions are ints.

            example: ( (1, 2), (3, 4) ) # team 1: (1, 2), team 2: (3, 4)
        """
        latest_version = self.get_latest_version_number()
        return ((latest_version, latest_version), (latest_version, latest_version))


    def push_rollout(self, rollout: bytes):
        """ Pushes a rollout to the backend.
            rollout: bytes - the rollout to push.
        """
        if self.uploader_thread is not None:
            self.uploader_thread.join()

        def upload():
            n_items = self.redis.rpush(ROLLOUTS, rollout)
            if n_items >= 1000:
                print("Had to limit rollouts. Learner may have have crashed, or is overloaded")
                self.redis.ltrim(ROLLOUTS, -100, -1)
                
        self.uploader_thread = Thread(target=upload)
        self.uploader_thread.start()


    def push_evaluation(self, blue_team_versions, orange_team_versions, blue_team_score, orange_team_score):
        """ Pushes an evaluation to the backend.
            blue_team_versions: tuple - the versions of the blue team models.
            orange_team_versions: tuple - the versions of the orange team models.
            blue_team_score: int - the score of the blue team.
            orange_team_score: int - the score of the orange team.
        """
        raise NotImplementedError()