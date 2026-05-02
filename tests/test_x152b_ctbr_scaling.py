import math
import unittest

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from badminton_intercept.control.ctbr_decoder import decode_ctbr_action
from badminton_intercept.control.x152b_ctbr_controller import compute_x152b_ctbr_wrench
from badminton_intercept.control.x152b_params import DEFAULT_X152B_PARAMS
from badminton_intercept.envs.intercept_env_cfg import CTBR_THRUST_SCALE, _DRONE_MASS_KG


def _compute(actions, axis_sign=(1.0, 1.0, 1.0)):
    batch = actions.shape[0]
    return compute_x152b_ctbr_wrench(
        actions=actions,
        body_rate_rad_s=torch.zeros((batch, 3), dtype=actions.dtype),
        prev_filtered_body_rate_rad_s=torch.zeros((batch, 3), dtype=actions.dtype),
        rate_integral=torch.zeros((batch, 3), dtype=actions.dtype),
        dt=0.0025,
        effective_mass_kg=_DRONE_MASS_KG,
        params=DEFAULT_X152B_PARAMS,
        rate_scale_rad_s=6.0,
        thrust_scale=CTBR_THRUST_SCALE,
        body_rate_axis_sign=axis_sign,
    )


@unittest.skipIf(torch is None, "torch is required for CTBR scaling tests")
class X152bCtbrScalingTest(unittest.TestCase):
    def test_x152b_ctbr_uses_same_tanh_decode_as_fm_for_body_rate(self):
        actions = torch.tensor([[2.0, -0.5, 0.25, 0.0]], dtype=torch.float32)
        axis_sign = (1.0, -1.0, 1.0)

        control_out = _compute(actions, axis_sign=axis_sign)
        decoded_body_rate, _ = decode_ctbr_action(actions, rate_scale_rad_s=6.0, thrust_scale=CTBR_THRUST_SCALE)
        expected = decoded_body_rate * torch.tensor(axis_sign, dtype=actions.dtype).unsqueeze(0)

        self.assertTrue(torch.allclose(control_out.target_body_rate_rad_s, expected))

    def test_x152b_ctbr_collective_thrust_matches_airgym_total_capacity(self):
        max_total_thrust_n = DEFAULT_X152B_PARAMS.airgym_thrust_scale_n * len(DEFAULT_X152B_PARAMS.rotor_positions_m)
        self.assertTrue(math.isclose(max_total_thrust_n, 38.36, rel_tol=0.0, abs_tol=1.0e-6))

        neutral = _compute(torch.zeros((1, 4), dtype=torch.float32))
        self.assertTrue(torch.allclose(neutral.thrust_cmd, torch.tensor([0.5])))
        self.assertTrue(torch.allclose(neutral.rotor_force_n.sum(dim=-1), torch.tensor([max_total_thrust_n * 0.5])))

        high = _compute(torch.tensor([[0.0, 0.0, 0.0, 20.0]], dtype=torch.float32))
        self.assertTrue(torch.allclose(high.thrust_cmd, torch.tensor([1.0])))
        self.assertTrue(torch.allclose(high.rotor_force_n.sum(dim=-1), torch.tensor([max_total_thrust_n])))

        low = _compute(torch.tensor([[0.0, 0.0, 0.0, -20.0]], dtype=torch.float32))
        self.assertTrue(torch.allclose(low.thrust_cmd, torch.tensor([0.0])))
        self.assertTrue(torch.allclose(low.rotor_force_n.sum(dim=-1), torch.tensor([0.0])))


if __name__ == "__main__":
    unittest.main()
