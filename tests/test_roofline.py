import math

import pytest

from perfmodel.roofline import arithmetic_intensity, attainable_flops, op_time, ridge_point

PEAK, BW = 100e12, 1e12  # ridge at 100 FLOP/byte


def test_memory_bound():
    t = op_time(flops=1e9, peak_flops=PEAK, dram_bytes=1e9, dram_bw=BW)
    assert t.bound == "memory" and t.memory_level == "dram"
    assert t.time == pytest.approx(1e-3)


def test_compute_bound():
    t = op_time(flops=1e12, peak_flops=PEAK, dram_bytes=1e6, dram_bw=BW)
    assert t.bound == "compute" and t.memory_level is None
    assert t.time == pytest.approx(1e-2)


def test_exactly_at_ridge_is_balanced():
    t = op_time(flops=100e9, peak_flops=PEAK, dram_bytes=1e9, dram_bw=BW)
    assert t.bound == "balanced"
    assert t.t_compute == t.t_memory == t.time


def test_empty_op():
    t = op_time(0, PEAK, 0, BW)
    assert t.time == 0 and t.bound == "none"


def test_zero_bytes_is_compute_bound():
    t = op_time(1e6, PEAK, 0, BW)
    assert t.bound == "compute"
    assert arithmetic_intensity(1e6, 0) == math.inf


def test_zero_flops_is_memory_bound():
    t = op_time(0, PEAK, 1e3, BW)
    assert t.bound == "memory"
    assert arithmetic_intensity(0, 1e3) == 0
    assert arithmetic_intensity(0, 0) == 0


def test_two_levels_take_max_not_sum():
    t = op_time(0, PEAK, dram_bytes=1e9, dram_bw=1e12, sram_bytes=10e9, sram_bw=100e12)
    assert t.t_dram == pytest.approx(1e-3)
    assert t.t_sram == pytest.approx(1e-4)
    assert t.time == pytest.approx(1e-3)  # concurrent levels
    t2 = op_time(0, PEAK, dram_bytes=0, dram_bw=1e12, sram_bytes=1e12, sram_bw=100e12)
    assert t2.memory_level == "sram"


def test_ridge_and_attainable():
    assert ridge_point(PEAK, BW) == 100
    assert attainable_flops(10, PEAK, BW) == 10e12
    assert attainable_flops(1000, PEAK, BW) == PEAK


def test_invalid_inputs():
    with pytest.raises(ValueError):
        op_time(1, 0, 1, BW)
    with pytest.raises(ValueError):
        op_time(-1, PEAK, 1, BW)
    with pytest.raises(ValueError):
        op_time(1, PEAK, 1, BW, sram_bytes=5, sram_bw=None)
