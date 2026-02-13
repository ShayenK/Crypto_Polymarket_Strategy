import os
import csv
import tempfile
from typing import Dict, List
from storage.memory import Memory
from storage.data_attributes import TradePosition

# Your actual SYMBOLS_MAP
SYMBOLS_MAP = {"BTCUSD": None, "ETHUSD": None, "SOLUSD": None, "XRPUSD": None}

def create_test_position(symbol: str, order_id: str, position_status: str = "PENDING") -> TradePosition:
    """Helper to create test positions"""
    return TradePosition(
        symbol=symbol,
        entry_proba=0.75,
        entry_time=1234567890,
        entry_price=100.0,
        entry_units=10.0,
        direction="LONG",
        yes_token_id="yes_123",
        no_token_id="no_123",
        condition_id="cond_123",
        order_id=order_id,
        neg_risk=False,
        exit_time=0,
        exit_price=0.0,
        exit_units=0.0,
        position_status=position_status,
        outcome=""
    )

def test_add_new_positions():
    """Test adding new trade positions"""
    print("\n=== TEST: add_new_trade_position ===")
    
    with Memory() as memory:
        # Test 1: Add valid positions
        entered = {
            "BTCUSD": create_test_position("BTCUSD", "order_001"),
            "ETHUSD": create_test_position("ETHUSD", "order_002"),
            "SOLUSD": create_test_position("SOLUSD", "order_003"),
            "XRPUSD": create_test_position("XRPUSD", "order_004")
        }
        memory.add_new_trade_position(entered)
        
        current = memory.return_current_positions()
        assert current is not None, "Should have positions"
        assert len(current["BTCUSD"]) == 1, "BTCUSD should have 1 position"
        assert len(current["ETHUSD"]) == 1, "ETHUSD should have 1 position"
        print("✅ Added valid positions successfully")
        
        # Test 2: Add empty dict - should not crash
        memory.add_new_trade_position({})
        assert len(memory.pending_trades["BTCUSD"]) == 1, "Should still have 1 position"
        print("✅ Handled empty dict correctly")
        
        # Test 3: Add None values - should not crash
        none_dict = {"BTCUSD": None, "ETHUSD": None, "SOLUSD": None, "XRPUSD": None}
        memory.add_new_trade_position(none_dict)
        assert len(memory.pending_trades["BTCUSD"]) == 1, "Should still have 1 position"
        print("✅ Handled None values correctly")
        
        # Test 4: Add partial dict (only some symbols)
        partial = {
            "BTCUSD": create_test_position("BTCUSD", "order_005"),
            "SOLUSD": create_test_position("SOLUSD", "order_006")
        }
        memory.add_new_trade_position(partial)
        assert len(memory.pending_trades["BTCUSD"]) == 2, "BTCUSD should have 2 positions"
        assert len(memory.pending_trades["ETHUSD"]) == 1, "ETHUSD should still have 1 position"
        print("✅ Handled partial dict correctly")

def test_return_current_positions():
    """Test returning current positions"""
    print("\n=== TEST: return_current_positions ===")
    
    with Memory() as memory:
        # Test 1: Empty memory
        result = memory.return_current_positions()
        assert result is None, "Should return None when empty"
        print("✅ Returns None when empty")
        
        # Test 2: With positions
        entered = {
            "BTCUSD": create_test_position("BTCUSD", "order_001"),
            "ETHUSD": create_test_position("ETHUSD", "order_002"),
            "SOLUSD": create_test_position("SOLUSD", "order_003"),
            "XRPUSD": create_test_position("XRPUSD", "order_004")
        }
        memory.add_new_trade_position(entered)
        
        result = memory.return_current_positions()
        assert result is not None, "Should return positions"
        assert len(result["BTCUSD"]) == 1, "Should have 1 BTCUSD position"
        assert len(result["XRPUSD"]) == 1, "Should have 1 XRPUSD position"
        
        # Test 3: Verify deep copy (modifications don't affect original)
        result["BTCUSD"].append(create_test_position("BTCUSD", "order_999"))
        assert len(memory.pending_trades["BTCUSD"]) == 1, "Original should be unchanged"
        print("✅ Returns deep copy correctly")

def test_remove_finished_positions():
    """Test removing finished positions with logging"""
    print("\n=== TEST: remove_finished_positions ===")
    
    # Ensure test directory exists
    os.makedirs('algorithm/storage', exist_ok=True)
    
    # Create temporary CSV file in the correct location
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', 
                                      dir='algorithm/storage') as f:
        temp_csv = f.name
        temp_filename = os.path.basename(temp_csv)
    
    try:
        # Set TRADE_LOG_FILEPATH to just the filename (not full path)
        import config
        original_path = config.TRADE_LOG_FILEPATH
        config.TRADE_LOG_FILEPATH = temp_filename
        
        # Reload Memory to pick up new config
        import importlib
        import storage.memory as mem_module
        importlib.reload(mem_module)
        from storage.memory import Memory
        
        with Memory() as memory:
            # Add positions
            entered = {
                "BTCUSD": create_test_position("BTCUSD", "order_001", "PENDING"),
                "ETHUSD": create_test_position("ETHUSD", "order_002", "PENDING"),
                "SOLUSD": create_test_position("SOLUSD", "order_003", "PENDING"),
                "XRPUSD": create_test_position("XRPUSD", "order_004", "PENDING")
            }
            memory.add_new_trade_position(entered)
            
            # Mark some as complete
            redeemed = {
                "BTCUSD": [create_test_position("BTCUSD", "order_001", "COMPLETE")],
                "ETHUSD": [create_test_position("ETHUSD", "order_002", "COMPLETE")],
                "SOLUSD": [],
                "XRPUSD": []
            }
            
            memory.remove_finished_positions(redeemed)
            
            # Check positions were removed
            remaining = memory.return_current_positions()
            assert remaining is not None, "Should still have SOLUSD and XRPUSD"
            assert len(remaining["BTCUSD"]) == 0, "BTCUSD should be removed"
            assert len(remaining["ETHUSD"]) == 0, "ETHUSD should be removed"
            assert len(remaining["SOLUSD"]) == 1, "SOLUSD should remain"
            assert len(remaining["XRPUSD"]) == 1, "XRPUSD should remain"
            print("✅ Removed completed positions correctly")
            
            # Check CSV was written (header + completed trades)
            with open(f'algorithm/storage/{temp_filename}', 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
                # Header + 2 completed positions (BTCUSD, ETHUSD)
                assert len(rows) >= 3, f"Should have at least header + 2 data rows, got {len(rows)}"
                print(f"✅ Logged completed trades to CSV ({len(rows)-1} rows)")
        
        # Restore original path
        config.TRADE_LOG_FILEPATH = original_path
        
    finally:
        # Cleanup
        if os.path.exists(temp_csv):
            os.remove(temp_csv)

def test_emergency_logging():
    """Test emergency logging on exception"""
    print("\n=== TEST: emergency logging ===")
    
    os.makedirs('algorithm/storage', exist_ok=True)
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv',
                                      dir='algorithm/storage') as f:
        temp_csv = f.name
        temp_filename = os.path.basename(temp_csv)
    
    try:
        import config
        original_path = config.TRADE_LOG_FILEPATH
        config.TRADE_LOG_FILEPATH = temp_filename
        
        import importlib
        import storage.memory as mem_module
        importlib.reload(mem_module)
        from storage.memory import Memory
        
        try:
            with Memory() as memory:
                # Add positions
                entered = {
                    "BTCUSD": create_test_position("BTCUSD", "order_001"),
                    "ETHUSD": create_test_position("ETHUSD", "order_002"),
                    "SOLUSD": create_test_position("SOLUSD", "order_003"),
                    "XRPUSD": create_test_position("XRPUSD", "order_004")
                }
                memory.add_new_trade_position(entered)
                
                # Simulate crash
                raise RuntimeError("Simulated crash!")
                
        except RuntimeError:
            print("✅ Exception caught (expected)")
        
        # Check emergency log was written
        with open(f'algorithm/storage/{temp_filename}', 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            # Header + 4 emergency logged positions
            assert len(rows) >= 5, f"Should have at least header + 4 emergency logged positions, got {len(rows)}"
            print(f"✅ Emergency logging activated on crash ({len(rows)-1} positions logged)")
        
        config.TRADE_LOG_FILEPATH = original_path
        
    finally:
        if os.path.exists(temp_csv):
            os.remove(temp_csv)

def test_edge_cases():
    """Test edge cases"""
    print("\n=== TEST: edge cases ===")
    
    with Memory() as memory:
        # Test removing from empty memory
        memory.remove_finished_positions({
            "BTCUSD": [],
            "ETHUSD": [],
            "SOLUSD": [],
            "XRPUSD": []
        })
        print("✅ Handled remove from empty memory")
        
        # Test removing None
        memory.remove_finished_positions(None)
        print("✅ Handled None in remove_finished_positions")
        
        # Test adding then immediately removing all
        entered = {
            "BTCUSD": create_test_position("BTCUSD", "order_001"),
            "ETHUSD": create_test_position("ETHUSD", "order_002"),
            "SOLUSD": create_test_position("SOLUSD", "order_003"),
            "XRPUSD": create_test_position("XRPUSD", "order_004")
        }
        memory.add_new_trade_position(entered)
        
        redeemed = {
            "BTCUSD": [create_test_position("BTCUSD", "order_001", "COMPLETE")],
            "ETHUSD": [create_test_position("ETHUSD", "order_002", "COMPLETE")],
            "SOLUSD": [create_test_position("SOLUSD", "order_003", "COMPLETE")],
            "XRPUSD": [create_test_position("XRPUSD", "order_004", "COMPLETE")]
        }
        memory.remove_finished_positions(redeemed)
        
        result = memory.return_current_positions()
        assert result is None, "All positions should be removed"
        print("✅ Add then remove all positions works correctly")
        
def test_partial_completion():
    """Test when only some symbols have completed trades"""
    print("\n=== TEST: partial completion ===")
    
    with Memory() as memory:
        # Add multiple positions for same symbol
        memory.add_new_trade_position({
            "BTCUSD": create_test_position("BTCUSD", "order_001"),
            "ETHUSD": create_test_position("ETHUSD", "order_002"),
            "SOLUSD": create_test_position("SOLUSD", "order_003"),
            "XRPUSD": create_test_position("XRPUSD", "order_004")
        })
        
        memory.add_new_trade_position({
            "BTCUSD": create_test_position("BTCUSD", "order_005"),
            "ETHUSD": create_test_position("ETHUSD", "order_006"),
            "SOLUSD": create_test_position("SOLUSD", "order_007"),
            "XRPUSD": create_test_position("XRPUSD", "order_008")
        })
        
        # Complete only one position for BTCUSD
        redeemed = {
            "BTCUSD": [create_test_position("BTCUSD", "order_001", "COMPLETE")],
            "ETHUSD": [],
            "SOLUSD": [],
            "XRPUSD": []
        }
        memory.remove_finished_positions(redeemed)
        
        remaining = memory.return_current_positions()
        assert len(remaining["BTCUSD"]) == 1, "BTCUSD should have 1 remaining"
        assert remaining["BTCUSD"][0].order_id == "order_005", "Should keep order_005"
        assert len(remaining["ETHUSD"]) == 2, "ETHUSD should still have 2"
        print("✅ Partial completion works correctly")

def test_mixed_status():
    """Test with mix of PENDING and COMPLETE in same redeemed list"""
    print("\n=== TEST: mixed status ===")
    
    with Memory() as memory:
        # Add positions
        memory.add_new_trade_position({
            "BTCUSD": create_test_position("BTCUSD", "order_001"),
            "ETHUSD": create_test_position("ETHUSD", "order_002"),
            "SOLUSD": create_test_position("SOLUSD", "order_003"),
            "XRPUSD": create_test_position("XRPUSD", "order_004")
        })
        
        # Some complete, some still pending
        redeemed = {
            "BTCUSD": [
                create_test_position("BTCUSD", "order_001", "COMPLETE"),
            ],
            "ETHUSD": [
                create_test_position("ETHUSD", "order_002", "PENDING"),  # Still pending
            ],
            "SOLUSD": [],
            "XRPUSD": []
        }
        
        memory.remove_finished_positions(redeemed)
        
        remaining = memory.return_current_positions()
        assert len(remaining["BTCUSD"]) == 0, "BTCUSD COMPLETE should be removed"
        assert len(remaining["ETHUSD"]) == 1, "ETHUSD PENDING should remain"
        assert len(remaining["SOLUSD"]) == 1, "SOLUSD should remain"
        print("✅ Mixed status handled correctly")

if __name__ == "__main__":
    print("=" * 50)
    print("RUNNING MEMORY & TRACKING TESTS")
    print("Testing with: BTCUSD, ETHUSD, SOLUSD, XRPUSD")
    print("=" * 50)
    
    try:
        test_add_new_positions()
        test_return_current_positions()
        test_remove_finished_positions()
        test_emergency_logging()
        test_edge_cases()
        test_partial_completion()
        test_mixed_status()
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("=" * 50)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()