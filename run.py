import sys
import subprocess
from pathlib import Path

def main():
    project_root = Path(__file__).parent
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(project_root / "scripts" / "app.py")
        ], cwd=project_root)
    except KeyboardInterrupt:
        print("\n Application stopped.")

if __name__ == "__main__":
    main()
