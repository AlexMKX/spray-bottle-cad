"""Parametric cosmetic reconstruction, units mm, estimates from photos."""
from pathlib import Path
import argparse
import json
import cadquery as cq
import trimesh

DIM = {
    "body_width":72.0, "body_depth":28.0, "body_height":55.0,
    "body_edge_radius":3.8, "loop_center_x":22.1,
    "loop_width":27.0, "loop_height":30.0, "loop_depth":21.0,
    "loop_bottom_z":52.4, "loop_opening_width":14.8,
    "loop_opening_height":14.1, "nozzle_center_x":-21.8,
    "nozzle_base_z":54.8, "collar_radius":12.5,
    "label_width":63.0, "label_height":29.0, "label_center_z":24.7,
}

def cylinder(r,z0,h,x):
    return cq.Workplane("XY",origin=(x,0,z0)).circle(r).extrude(h)

def build_parts():
    p=DIM
    body=(cq.Workplane("XY").box(p["body_width"],p["body_depth"],
        p["body_height"],centered=(True,True,False))
        .edges().fillet(p["body_edge_radius"]))
    lz=p["loop_bottom_z"]+p["loop_height"]/2
    loop=(cq.Workplane("XY").box(p["loop_width"],p["loop_depth"],p["loop_height"])
        .translate((p["loop_center_x"],0,lz)).edges("|Y").fillet(3.7))
    opening=(cq.Workplane("XY")
        .box(p["loop_opening_width"],p["loop_depth"]+4,p["loop_opening_height"])
        .translate((p["loop_center_x"],0,lz+1.8)).edges("|Y").fillet(1.8))
    body=body.union(loop.cut(opening))
    x,z=p["nozzle_center_x"],p["nozzle_base_z"]
    body=body.union(cylinder(10.4,z-0.8,2.5,x))
    pump=cylinder(11.8,z+0.8,17.8,x)
    for r,zoff,h in [(12.6,1.5,3.1),(11,18,4.8),(7.5,22,10.5),
                      (6.7,31.5,12),(7,43,2)]:
        pump=pump.union(cylinder(r,z+zoff,h,x))
    hole=(cq.Workplane("XZ",origin=(x,-7.5,z+40))
          .circle(1.25).extrude(-2.2))
    pump=pump.cut(hole)
    label=(cq.Workplane("XY").box(p["label_width"],0.35,p["label_height"])
           .translate((0,-p["body_depth"]/2-0.08,p["label_center_z"])))
    label=label.edges("|Y").fillet(0.85)
    parts={"body":body,"pump":pump,"label":label}
    try:
        plane=cq.Plane(origin=(0,-p["body_depth"]/2-0.30,0),
                       xDir=(1,0,0),normal=(0,-1,0))
        lettering=None
        for s,tx,tz,size in [("PUMP",-20,30.4,5.6),
                              ("SPRAY",-20,22.6,5.2),("80",14,27.2,13.7),
                              ("ISE MOUNT",-19,13.1,3.8)]:
            part=(cq.Workplane(plane).center(tx,tz)
                  .text(s,size,0.55,font="DejaVu Sans",
                        kind="bold",combine=False))
            lettering=part if lettering is None else lettering.add(part)
        parts["lettering"]=lettering
    except Exception as exc:
        print("lettering omitted:",exc)
    return parts

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,default=Path("out"))
    out=ap.parse_args().out
    out.mkdir(exist_ok=True,parents=True)
    parts=build_parts()
    for name,part in parts.items():
        target=out/(name+".stl")
        cq.exporters.export(part,str(target),tolerance=0.18,angularTolerance=0.35)
        mesh=trimesh.load(str(target),force="mesh")
        mesh.update_faces(mesh.nondegenerate_faces(height=1e-6))
        mesh.remove_unreferenced_vertices()
        mesh.export(str(target))
    colors={"body":(0.92,0.78,0.62),"pump":(0.09,0.08,0.08),
            "label":(0.10,0.09,0.11),"lettering":(0.94,0.94,0.92)}
    assy=cq.Assembly()
    for name,part in parts.items():
        assy.add(part,name=name,color=cq.Color(*colors[name]))
    assy.save(str(out/"spray-bottle.step"))
    total=trimesh.util.concatenate([
        trimesh.load(str(out/(n+".stl")),force="mesh")
        for n in ("body","pump")])
    total.export(str(out/"silhouette.stl"))
    (out/"dimensions.json").write_text(json.dumps(DIM,indent=2)+"\n")
    lines=["// Generated color preview", "$fn=72;"]
    for name,color in colors.items():
        if name in parts:
            lines.append("color([%s]) import(\"%s.stl\");" %
                         (",".join(map(str,color)),name))
    (out/"scene.scad").write_text("\n".join(lines)+"\n")
    assert len(parts["body"].solids().vals())==1
    assert len(parts["pump"].solids().vals())==1
    assert total.is_watertight
    print("CAD bounds:",total.bounds.tolist())

if __name__=="__main__":
    main()
