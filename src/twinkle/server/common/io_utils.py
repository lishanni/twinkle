# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Unified IO utilities for managing training runs and checkpoints.

Manager implementations live in dedicated modules:
  - ``tinker_io_utils``  : Tinker-specific managers (use tinker.types)
  - ``twinkle_io_utils`` : Twinkle-specific managers (use twinkle_client.types.training)

This module re-exports everything and provides the factory functions
``create_training_run_manager`` and ``create_checkpoint_manager``.
"""
from twinkle.server.common.tinker_io_utils import TinkerCheckpointManager, TinkerTrainingRunManager
from twinkle.server.common.twinkle_io_utils import TwinkleCheckpointManager, TwinkleTrainingRunManager
from twinkle.server.utils.io_utils import ResolvedLoadPath, validate_ownership, validate_user_path
# Re-export twinkle-native pydantic models from twinkle_client.types
from twinkle_client.types.training import Checkpoint as TwinkleCheckpoint
from twinkle_client.types.training import (CheckpointsListResponse, CreateModelRequest, Cursor, LoraConfig,
                                           ParsedCheckpointTwinklePath)
from twinkle_client.types.training import TrainingRun as TwinkleTrainingRun
from twinkle_client.types.training import TrainingRunsResponse, WeightsInfoResponse

__all__ = [
    'create_checkpoint_manager',
    'create_training_run_manager',
    'validate_user_path',
    'validate_ownership',
    'ResolvedLoadPath',
    'Cursor',
    'TinkerTrainingRunManager',
    'TinkerCheckpointManager',
    'TwinkleTrainingRunManager',
    'TwinkleCheckpointManager',
    # Twinkle-native models (re-exported for convenience)
    'TwinkleCheckpoint',
    'TwinkleTrainingRun',
    'TrainingRunsResponse',
    'CheckpointsListResponse',
    'WeightsInfoResponse',
    'LoraConfig',
    'CreateModelRequest',
    'ParsedCheckpointTwinklePath',
]


def create_training_run_manager(token: str, client_type: str = 'twinkle'):
    """Create a TrainingRunManager for the given token.

    Args:
        token: User authentication token.
        client_type: 'tinker' or 'twinkle' (default 'twinkle').
    """
    if client_type == 'tinker':
        return TinkerTrainingRunManager(token)
    return TwinkleTrainingRunManager(token)


def create_checkpoint_manager(token: str, client_type: str = 'twinkle'):
    """Create a CheckpointManager for the given token.

    Args:
        token: User authentication token.
        client_type: 'tinker' or 'twinkle' (default 'twinkle').
    """
    if client_type == 'tinker':
        run_mgr = TinkerTrainingRunManager(token)
        return TinkerCheckpointManager(token, run_mgr)
    run_mgr = TwinkleTrainingRunManager(token)
    return TwinkleCheckpointManager(token, run_mgr)
