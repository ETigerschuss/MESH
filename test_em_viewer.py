"""
Quick EM Viewer Test
====================

Tests the EM viewer generation without running the full analysis.
This assumes you already have results from a previous run.

Requirements:
- comprehensive_overlap_results/all_results_combined.csv
- comprehensive_overlap_results/synapses.csv
- comprehensive_overlap_results/neuron_meshes/*.obj
- comprehensive_overlap_results/em_snaps/*.png (at least center slices)
"""

import os
import sys
import subprocess
import time

def main():
    print("="*70)
    print("Quick EM Viewer Test")
    print("="*70)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "comprehensive_overlap_results")
    viewer_script = os.path.join(base_dir, "em_viewer.py")
    
    # Check if results exist
    print("\n[1/3] Checking prerequisites...")
    
    if not os.path.exists(results_dir):
        print("  ✗ Results directory not found!")
        print("  Please run 'python overlap_analysis.py' first")
        return
    
    csv_file = os.path.join(results_dir, 'all_results_combined.csv')
    if not os.path.exists(csv_file):
        print("  ✗ Contact data not found!")
        print("  Please run 'python overlap_analysis.py' first")
        return
    
    synapses_file = os.path.join(results_dir, 'synapses.csv')
    if not os.path.exists(synapses_file):
        print("  ✗ Synapse data not found!")
        print("  Please run 'python overlap_analysis.py' first")
        return
    
    mesh_dir = os.path.join(results_dir, 'neuron_meshes')
    if not os.path.exists(mesh_dir):
        print("  ✗ Mesh directory not found!")
        print("  Please run 'python overlap_analysis.py' first")
        return
    
    mesh_count = len([f for f in os.listdir(mesh_dir) if f.endswith('.obj')])
    print(f"  ✓ Found {mesh_count} mesh files")
    
    em_dir = os.path.join(results_dir, 'em_snaps')
    if os.path.exists(em_dir):
        em_count = len([f for f in os.listdir(em_dir) if f.endswith('.png')])
        print(f"  ✓ Found {em_count} EM snapshot files")
    else:
        print("  ⚠ No EM snapshots found (will be generated on-the-fly)")
    
    if not os.path.exists(viewer_script):
        print(f"  ✗ EM viewer script not found: {viewer_script}")
        return
    
    print("  ✓ All prerequisites met!")
    
    # Run the viewer generator
    print("\n[2/3] Running EM viewer generator...")
    print(f"  Command: python {os.path.basename(viewer_script)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, viewer_script],
            cwd=base_dir,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"  ✓ Viewer generated successfully in {elapsed:.1f}s")
            if result.stdout:
                print("\n  Output:")
                for line in result.stdout.split('\n'):
                    if line.strip():
                        print(f"    {line}")
        else:
            print(f"  ✗ Viewer generation failed!")
            if result.stderr:
                print("\n  Error:")
                for line in result.stderr.split('\n'):
                    if line.strip():
                        print(f"    {line}")
            return
            
    except subprocess.TimeoutExpired:
        print("  ✗ Viewer generation timed out (120s limit)")
        return
    except Exception as e:
        print(f"  ✗ Error running viewer: {e}")
        return
    
    # Check output
    print("\n[3/3] Verifying output...")
    
    viewer_html = os.path.join(results_dir, "em_viewer.html")
    if os.path.exists(viewer_html):
        file_size = os.path.getsize(viewer_html) / (1024 * 1024)  # MB
        print(f"  ✓ Viewer HTML created: {file_size:.1f} MB")
        print(f"\n  >>> Open in browser: file:///{os.path.abspath(viewer_html)}")
    else:
        print(f"  ✗ Viewer HTML not found at: {viewer_html}")
        return
    
    print("\n" + "="*70)
    print("✓ TEST PASSED!")
    print("="*70)

if __name__ == "__main__":
    main()
