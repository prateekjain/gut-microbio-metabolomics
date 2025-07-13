#!/usr/bin/env python3
"""
Simple performance test script to verify optimizations
"""

import time
import logging
from compare_tumor.data_functions import (
    get_multiple_bacteria_top_metabolites,
    get_multiple_bacteria_cumm_top_metabolites,
    log_performance_status
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_top_metabolites_performance():
    """Test the optimized top metabolites function"""
    print("Testing Top Metabolites Performance...")
    
    # Test with a small set of bacteria
    test_bacteria = ['Bacteria_1', 'Bacteria_2', 'Bacteria_3']
    table_name = "gmm_test_1"  # Use a test table
    
    start_time = time.time()
    
    try:
        result = get_multiple_bacteria_top_metabolites(table_name, test_bacteria)
        execution_time = time.time() - start_time
        
        print(f"✅ Top Metabolites Query completed in {execution_time:.2f} seconds")
        print(f"   Result shape: {result.shape if result is not None else 'None'}")
        
        return execution_time < 10.0  # Should complete within 10 seconds
        
    except Exception as e:
        print(f"❌ Top Metabolites Query failed: {e}")
        return False

def test_cumulative_metabolites_performance():
    """Test the optimized cumulative metabolites function"""
    print("Testing Cumulative Metabolites Performance...")
    
    # Test with a small set of bacteria
    test_bacteria = ['Bacteria_1', 'Bacteria_2']
    table_name = "gmm_test_1"  # Use a test table
    
    start_time = time.time()
    
    try:
        result = get_multiple_bacteria_cumm_top_metabolites(table_name, test_bacteria)
        execution_time = time.time() - start_time
        
        print(f"✅ Cumulative Metabolites Query completed in {execution_time:.2f} seconds")
        print(f"   Result shape: {result.shape if result is not None else 'None'}")
        
        return execution_time < 10.0  # Should complete within 10 seconds
        
    except Exception as e:
        print(f"❌ Cumulative Metabolites Query failed: {e}")
        return False

def main():
    """Run performance tests"""
    print("🚀 Performance Optimization Test")
    print("=" * 50)
    
    # Log initial performance status
    log_performance_status()
    
    # Test top metabolites
    top_success = test_top_metabolites_performance()
    
    # Test cumulative metabolites
    cumm_success = test_cumulative_metabolites_performance()
    
    # Log final performance status
    log_performance_status()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   Top Metabolites: {'✅ PASS' if top_success else '❌ FAIL'}")
    print(f"   Cumulative Metabolites: {'✅ PASS' if cumm_success else '❌ FAIL'}")
    
    if top_success and cumm_success:
        print("\n🎉 All performance tests passed! Optimizations are working.")
    else:
        print("\n⚠️  Some tests failed. Check the logs for details.")

if __name__ == "__main__":
    main() 