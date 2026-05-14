import os
import shutil
import glob

def build_release():
    source_dir = r'D:\medical'
    release_dir = os.path.join(source_dir, 'Norm-AImed_Release')
    
    print(f"[SYSTEM] Starting industrial-grade asset extraction...")
    
    # Step 1: Establish Clean Room
    if os.path.exists(release_dir):
        print(f"[CLEANUP] Removing existing release directory: {release_dir}")
        shutil.rmtree(release_dir)
    os.makedirs(release_dir)
    
    # Step 2: Whitelist Asset Transfer
    # 1. .py files
    # 2. .yaml files
    # 3. README.md & requirements.txt
    # 4. .png result charts
    
    extensions = ['*.py', '*.yaml', 'README.md', 'requirements.txt', '*.png', '.gitignore']
    
    copied_count = 0
    for pattern in extensions:
        files = glob.glob(os.path.join(source_dir, pattern))
        for f in files:
            if os.path.isfile(f):
                shutil.copy2(f, release_dir)
                copied_count += 1
    
    print(f"[TRANSFER] Copied {copied_count} core asset files to root.")
    
    # Step 3: Precise Weight Extraction
    weights_dest = os.path.join(release_dir, 'weights')
    os.makedirs(weights_dest)
    
    source_weight = os.path.join(source_dir, 'ablation_runs', 'Exp_EMA', 'weights', 'best.pt')
    if os.path.exists(source_weight):
        shutil.copy2(source_weight, os.path.join(weights_dest, 'best.pt'))
        print(f"[WEIGHTS] Successfully extracted SOTA weights (best.pt) to /weights/.")
    else:
        print(f"[WARNING] EMA best.pt NOT found at {source_weight}!")
    
    # Step 4: Safety Audit & Size Calculation
    total_size = 0
    for root, dirs, files in os.walk(release_dir):
        for f in files:
            fp = os.path.join(root, f)
            total_size += os.path.getsize(fp)
            
    size_mb = total_size / (1024 * 1024)
    print("\n" + "="*50)
    print(f"RELEASE PACKAGE BUILT SUCCESSFULLY!")
    print(f"Location: {release_dir}")
    print(f"Total Physical Size: {size_mb:.2f} MB")
    print("="*50)

if __name__ == "__main__":
    build_release()
