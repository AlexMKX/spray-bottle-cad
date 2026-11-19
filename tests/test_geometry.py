from pathlib import Path
import sys
import trimesh
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from model import build_parts,DIM
from compare import render_mask,contour_reference,compare_masks

def test_brep():
    parts=build_parts()
    assert parts["body"].val().isValid()
    assert parts["pump"].val().isValid()
    assert len(parts["body"].solids().vals())==1
    assert len(parts["pump"].solids().vals())==1
    assert 50<DIM["body_width"]<100

def test_stl():
    for name in ("body","pump","silhouette"):
        mesh=trimesh.load(str(ROOT/"out"/(name+".stl")),force="mesh")
        assert mesh.is_watertight,name
        assert mesh.volume>100,name

def test_renders_not_clipped():
    for name in ("front","rear","left","right","iso","overhead"):
        mask=render_mask(ROOT/"out"/(name+".png"))
        ys,xs=(mask>0).nonzero()
        h,w=mask.shape
        assert min(xs)>8 and max(xs)<w-8,name
        assert min(ys)>8 and max(ys)<h-8,name

def test_comparison():
    photo=contour_reference("rear")
    model=render_mask(ROOT/"out/rear.png")
    iou,chamfer,_=compare_masks(photo,model)
    assert 0.3<iou<=1,iou
    assert chamfer<40,chamfer
