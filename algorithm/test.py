import os
import csv
import time
import tempfile
from contextlib import contextmanager
from dataclasses import replace
from typing import Dict, List
from storage.memory import Memory
from storage.data_attributes import TradePosition
from config import SYMBOLS_MAP  # assuming this is imported correctly

# For realism in timestamps
NOW = int(time.time())

def create_pending_position(
    symbol: str,
    order_id: str,
    direction: str = "UP",
    entry_proba: float = 0.72,
    entry_price: float = 0.55,
    entry_units: float = 100.0,
) -> TradePosition:
    """Create a position that looks like it just came from TradeEntryAgent"""
    entry_time = NOW - 7200  # pretend entered ~2 hours ago
    return TradePosition(
        symbol=symbol,
        entry_proba=entry_proba,
        entry_time=entry_time,
        entry_price=entry_price,
        entry_units=entry_units,
        direction=direction,
        yes_token_id=f"{symbol}_yes_123",
        no_token_id=f"{symbol}_no_123",
        condition_id=f"cond_{symbol.lower()}_abc123",
        order_id=order_id,
        neg_risk=False,
        exit_time=entry_time + 3600,  # market ends in 1 hour from entry
        exit_price=None,
        exit_units=None,
        position_status="PENDING",
        outcome=None,  # matches real entry
    )


def create_completed_position(
    pending: TradePosition,
    is_win: bool = True,
) -> TradePosition:
    """Take a pending position → mark as resolved (like TradeExitAgent)"""
    return replace(
        pending,
        exit_price=1.0 if is_win else 0.0,
        exit_units=pending.entry_units * (1.8 if is_win else 0.0),  # simplistic payout
        exit_time=NOW,
        position_status="COMPLETE",
        outcome="WIN" if is_win else "LOSS",
    )


def setup_temp_csv() -> tuple[tempfile._TemporaryFileWrapper, str]:
    """Create temp CSV file and return handle + path"""
    tmp = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".csv")
    return tmp, tmp.name


def read_csv_rows(filepath: str) -> list[list[str]]:
    """Helper to read CSV rows safely"""
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", newline="") as f:
        return list(csv.reader(f))

@contextmanager
def patch_trade_log_path(new_path: str):
    """Context manager to patch config.TRADE_LOG_FILEPATH"""
    import config

    original = config.TRADE_LOG_FILEPATH
    config.TRADE_LOG_FILEPATH = new_path
    try:
        yield
    finally:
        config.TRADE_LOG_FILEPATH = original


# ────────────────────────────────────────────────
#                  TESTS
# ────────────────────────────────────────────────


def test_add_new_positions():
    print("\n=== TEST: add_new_trade_position ===")
    with Memory() as memory:
        # Add valid positions
        positions = {
            sym: create_pending_position(sym, f"ord-{sym.lower()}-001")
            for sym in SYMBOLS_MAP
        }
        memory.add_new_trade_position(positions)

        current = memory.return_current_positions() or {}
        assert all(len(current.get(sym, [])) == 1 for sym in SYMBOLS_MAP)
        print("✓ Added valid positions")

        # Empty dict → no change
        memory.add_new_trade_position({})
        assert all(len(current.get(sym, [])) == 1 for sym in SYMBOLS_MAP)

        # Dict with None → should skip None
        none_dict = {sym: None for sym in SYMBOLS_MAP}
        memory.add_new_trade_position(none_dict)
        assert all(len(current.get(sym, [])) == 1 for sym in SYMBOLS_MAP)
        print("✓ Handled empty / None inputs safely")


def test_return_current_positions():
    print("\n=== TEST: return_current_positions ===")
    with Memory() as memory:
        # Empty → None or empty dict (your code returns None)
        assert memory.return_current_positions() is None
        print("✓ Returns None when empty")

        # Add something
        pos = create_pending_position("BTCUSD", "ord-btc-001")
        memory.add_new_trade_position({"BTCUSD": pos})

        current = memory.return_current_positions() or {}
        assert len(current["BTCUSD"]) == 1

        # Modify returned copy → original unchanged
        current["BTCUSD"].append(create_pending_position("BTCUSD", "fake"))
        assert len(memory.pending_trades["BTCUSD"]) == 1
        print("✓ Returns deep copy")


def test_remove_finished_positions():
    print("\n=== TEST: remove_finished_positions + logging ===")

    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".csv") as tmp:
        temp_path = tmp.name

    try:
        with patch_trade_log_path(temp_path):
            with Memory() as memory:
                # Setup pending positions
                positions = {
                    sym: create_pending_position(sym, f"ord-{sym.lower()}-00{i+1}")
                    for i, sym in enumerate(SYMBOLS_MAP)
                }
                memory.add_new_trade_position(positions)

                # Complete BTC & ETH only
                redeemed = {
                    "BTCUSD": [create_completed_position(positions["BTCUSD"], is_win=True)],
                    "ETHUSD": [create_completed_position(positions["ETHUSD"], is_win=False)],
                    "SOLUSD": [],
                    "XRPUSD": [],
                }

                memory.remove_finished_positions(redeemed)

                remaining = memory.return_current_positions() or {}
                assert len(remaining.get("BTCUSD", [])) == 0
                assert len(remaining.get("ETHUSD", [])) == 0
                assert len(remaining.get("SOLUSD", [])) == 1
                assert len(remaining.get("XRPUSD", [])) == 1
                print("✓ Correctly removed completed positions")

                # Check CSV logging (header + 2 rows)
                rows = read_csv_rows(temp_path)
                assert len(rows) >= 3, f"Expected header + ≥2 rows, got {len(rows)}"
                assert len(rows) - 1 == 2, "Should have logged exactly 2 completed trades"
                print("✓ Completed trades logged to CSV")

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_emergency_logging():
    print("\n=== TEST: emergency logging on crash ===")

    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".csv") as tmp:
        temp_path = tmp.name

    try:
        with patch_trade_log_path(temp_path):
            try:
                with Memory() as memory:
                    positions = {
                        sym: create_pending_position(sym, f"ord-{sym.lower()}-crash")
                        for sym in SYMBOLS_MAP
                    }
                    memory.add_new_trade_position(positions)
                    raise RuntimeError("Simulated crash for emergency logging test")
            except RuntimeError:
                pass  # expected

            # After exception, emergency log should have written pending positions
            rows = read_csv_rows(temp_path)
            data_rows = rows[1:] if rows else []
            assert len(data_rows) == 4, f"Expected 4 emergency-logged positions, got {len(data_rows)}"
            print("✓ Emergency logging wrote pending positions on crash")

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_partial_completion():
    print("\n=== TEST: partial completion (multiple positions per symbol) ===")
    with Memory() as memory:
        # Add two positions per symbol
        for i in range(2):
            batch = {
                sym: create_pending_position(sym, f"ord-{sym.lower()}-{i+1:03d}")
                for sym in SYMBOLS_MAP
            }
            memory.add_new_trade_position(batch)

        # Complete only the first BTC position
        btc_pos_1 = memory.pending_trades["BTCUSD"][0]
        redeemed = {
            "BTCUSD": [create_completed_position(btc_pos_1, is_win=True)],
            "ETHUSD": [],
            "SOLUSD": [],
            "XRPUSD": [],
        }

        memory.remove_finished_positions(redeemed)

        remaining = memory.return_current_positions() or {}
        assert len(remaining["BTCUSD"]) == 1
        assert remaining["BTCUSD"][0].order_id == "ord-btcusd-002"
        assert len(remaining["ETHUSD"]) == 2
        print("✓ Partial completion keeps correct positions")


def test_edge_cases():
    print("\n=== TEST: edge cases ===")
    with Memory() as memory:
        # Remove from empty
        memory.remove_finished_positions({sym: [] for sym in SYMBOLS_MAP})
        print("✓ Handled remove on empty memory")

        # Pass None
        memory.remove_finished_positions(None)
        print("✓ Handled None input")

        # Add → remove everything
        positions = {
            sym: create_pending_position(sym, f"ord-{sym.lower()}-999")
            for sym in SYMBOLS_MAP
        }
        memory.add_new_trade_position(positions)

        redeemed = {sym: [create_completed_position(pos)] for sym, pos in positions.items()}
        memory.remove_finished_positions(redeemed)

        assert memory.return_current_positions() is None
        print("✓ Add → complete all → memory empty")


if __name__ == "__main__":
    print("=" * 60)
    print(" MEMORY & TRACKING UNIT TESTS ".center(60, "="))
    print("Symbols:", ", ".join(SYMBOLS_MAP))
    print("=" * 60)

    tests = [
        test_add_new_positions,
        test_return_current_positions,
        test_remove_finished_positions,
        test_emergency_logging,
        test_edge_cases,
        test_partial_completion,
    ]

    failures = 0
    for test_func in tests:
        try:
            test_func()
        except AssertionError as e:
            print(f"\n❌ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failures += 1
        except Exception as e:
            print(f"\n❌ {test_func.__name__} UNEXPECTED ERROR: {e}")
            traceback.print_exc()
            failures += 1

    print("\n" + "=" * 60)
    if failures == 0:
        print(" ALL TESTS PASSED ".center(60, " "))
    else:
        print(f" {failures} TEST(S) FAILED ".center(60, " "))
    print("=" * 60)