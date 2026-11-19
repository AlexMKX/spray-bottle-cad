"""Headless OpenSCAD camera renders."""
from pathlib import Path
import argparse,shutil,subprocess

CAMERAS={
 "front":"0,0,48,75,0,0,380",
 "rear":"0,0,48,75,0,180,380",
 "left":"0,0,48,75,0,90,380",
 "right":"0,0,48,75,0,270,380",
 "iso":"0,0,48,65,0,35,395",
 "overhead":"0,0,48,0,0,0,380",
}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path("out"))
    out=p.parse_args().out.resolve()
    if not shutil.which("openscad"):
        raise SystemExit("OpenSCAD required")
    prefix=["xvfb-run","-a"] if shutil.which("xvfb-run") else []
    for name,camera in CAMERAS.items():
        target=out/(name+".png")
        subprocess.run(prefix+["openscad","-o",str(target),
          "--imgsize=960,720","--camera="+camera,"--projection=o",
          "--colorscheme=Tomorrow",str(out/"scene.scad")],
          cwd=out,check=True,capture_output=True,text=True)
        assert target.stat().st_size>1000
        print(name,target)
if __name__=="__main__":
    main()
