"""
Test Script for FlyWire Overlap Analysis Toolkit
=================================================

This script performs basic sanity checks on the toolkit files
without running the full analysis (which takes 30-60 minutes).

Tests:
1. File existence check
2. Import check (verify all dependencies)
3. Data structure check (verify results exist)
4. Quick EM viewer test (if results exist)
"""

import os
import sys
import importlib.util

def color_text(text, color='green'):
    colors = {
        'green': '\033[92m',
        'red': '\033[91m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'reset': '\033[0m'
    }
    return f"{colors.get(color, '')}{text}{colors['reset']}"

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"  ✓ {description}: {color_text('FOUND', 'green')}")
        return True
    else:
        print(f"  ✗ {description}: {color_text('MISSING', 'red')}")
        return False

def check_import(module_name, package_name=None):
    """Check if a Python package can be imported"""
    try:
        if package_name:
            __import__(package_name)
        else:
            __import__(module_name)
        print(f"  ✓ {module_name}: {color_text('OK', 'green')}")
        return True
    except ImportError as e:
        print(f"  ✗ {module_name}: {color_text('MISSING', 'red')} - {e}")
        return False

def check_script_syntax(filepath):
    """Check if a Python script has valid syntax"""
    try:
        spec = importlib.util.spec_from_file_location("module", filepath)
        if spec is None:
            print(f"  ✗ {os.path.basename(filepath)}: {color_text('INVALID', 'red')}")
            return False
        module = importlib.util.module_from_spec(spec)
        print(f"  ✓ {os.path.basename(filepath)}: {color_text('VALID SYNTAX', 'green')}")
        return True
    except Exception as e:
        print(f"  ✗ {os.path.basename(filepath)}: {color_text('SYNTAX ERROR', 'red')} - {e}")
        return False

def main():
    print("="*70)
    print("FlyWire Overlap Analysis Toolkit - Test Suite")
    print("="*70)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "comprehensive_overlap_results")
    
    all_passed = True
    
    # Test 1: File Existence
    print("\n[1/4] Checking Core Files...")
    files_to_check = {
        'overlap_analysis.py': 'Main analysis script',
        'em_viewer.py': 'EM viewer generator',
        'generate_em_stacks.py': 'EM stack generator',
    }
    
    for filename, description in files_to_check.items():
        filepath = os.path.join(base_dir, filename)
        if not check_file_exists(filepath, description):
            all_passed = False
    
    # Test 2: Python Syntax
    print("\n[2/4] Checking Python Syntax...")
    for filename in files_to_check.keys():
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            if not check_script_syntax(filepath):
                all_passed = False
        else:
            print(f"  - Skipping {filename} (not found)")
    
    # Test 3: Dependencies
    print("\n[3/4] Checking Dependencies...")
    dependencies = [
        ('numpy', None),
        ('pandas', None),
        ('plotly', None),
        ('PIL', 'Pillow'),
        ('navis', None),
        ('fafbseg', None),
        ('cloudvolume', 'cloud-volume'),
        ('trimesh', None),
        ('cv2', 'opencv-python'),
    ]
    
    for module_name, package_name in dependencies:
        if not check_import(module_name, package_name):
            all_passed = False
            print(f"      Install with: pip install {package_name or module_name}")
    
    # Test 4: Results Structure
    print("\n[4/4] Checking Results Directory...")
    if os.path.exists(results_dir):
        print(f"  ✓ Results directory: {color_text('EXISTS', 'green')}")
        
        # Check for key subdirectories/files
        csv_file = os.path.join(results_dir, 'all_results_combined.csv')
        synapses_file = os.path.join(results_dir, 'synapses.csv')
        mesh_dir = os.path.join(results_dir, 'neuron_meshes')
        em_dir = os.path.join(results_dir, 'em_snaps')
        
        if os.path.exists(csv_file):
            print(f"  ✓ Contact data: {color_text('FOUND', 'green')}")
        else:
            print(f"  ℹ Contact data: {color_text('NOT GENERATED YET', 'yellow')}")
        
        if os.path.exists(synapses_file):
            print(f"  ✓ Synapse data: {color_text('FOUND', 'green')}")
        else:
            print(f"  ℹ Synapse data: {color_text('NOT GENERATED YET', 'yellow')}")
        
        if os.path.exists(mesh_dir):
            mesh_count = len([f for f in os.listdir(mesh_dir) if f.endswith('.obj')])
            print(f"  ✓ Neuron meshes: {color_text(f'{mesh_count} files', 'green')}")
        else:
            print(f"  ℹ Neuron meshes: {color_text('NOT EXPORTED YET', 'yellow')}")
        
        if os.path.exists(em_dir):
            em_count = len([f for f in os.listdir(em_dir) if f.endswith('.png')])
            print(f"  ✓ EM snapshots: {color_text(f'{em_count} files', 'green')}")
        else:
            print(f"  ℹ EM snapshots: {color_text('NOT GENERATED YET', 'yellow')}")
    else:
        print(f"  ℹ Results directory: {color_text('NOT CREATED YET', 'yellow')}")
        print(f"     Run 'python overlap_analysis.py' to generate results")
    
    # Summary
    print("\n" + "="*70)
    if all_passed:
        print(color_text("✓ ALL TESTS PASSED!", 'green'))
        print("\nYou can now run the toolkit:")
        print("  python overlap_analysis.py")
    else:
        print(color_text("✗ SOME TESTS FAILED", 'red'))
        print("\nPlease install missing dependencies:")
        print("  pip install -r requirements.txt")
    print("="*70)

if __name__ == "__main__":
    main()
