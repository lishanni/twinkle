# Copyright (c) ModelScope Contributors. All rights reserved.
import twinkle

from twinkle import DeviceMesh
from twinkle.model.transformers.strategy.sequence_parallel import (
    _derive_sequence_parallel_sizes,
    _get_sequence_group_specs,
)

twinkle.initialize(mode='local')


class TestDerivedRingSizing:

    def test_derive_cp_sp_sizes(self):
        assert _derive_sequence_parallel_sizes(6, 4) == (2, 2)
        assert _derive_sequence_parallel_sizes(32, 2) == (2, 1)
        assert _derive_sequence_parallel_sizes(1, 4) == (1, 4)

    def test_group_specs_follow_raw_data_order(self):
        device_mesh = DeviceMesh.from_sizes(
            fsdp_size=2,
            dp_size=2,
            ulysses_size=4,
            device_type='cuda',
        )
        specs = _get_sequence_group_specs(device_mesh, seq_world_size=4, sp_world_size=2, rp_world_size=2)
        assert len(specs) == 1
        spec = specs[0]
        assert spec['seq_ranks'] == [0, 2, 1, 3]
        assert spec['sp_groups'] == [[0, 2], [1, 3]]
        assert spec['rp_groups'] == [[0, 1], [2, 3]]
