"""Parametric cosmetic reconstruction, units mm, estimates from photos."""
from math import cos, sin, tau
from pathlib import Path
import argparse
import json
import cadquery as cq
import trimesh

DIM = {
    "body_width": 59.0,
    "body_depth": 29.0,
    "body_center_x": 0.5,
    "body_height": 58.0,
    "body_edge_radius": 2.2,
    "shoulder_height": 13.0,
    "shoulder_left_inset": 4.0,
    "shoulder_depth_inset": 6.2,
    "base_recess_depth": 1.7,
    "loop_top_z": 85.0,
    "loop_root_width": 26.0,
    "loop_top_width": 24.0,
    "loop_root_depth": 17.0,
    "loop_top_depth": 8.0,
    "loop_outside_x": 29.2,
    "loop_opening_width": 13.2,
    "loop_opening_height": 16.8,
    "loop_opening_center_x": 16.8,
    "loop_opening_bottom_z": 58.5,
    "nozzle_center_x": -13.5,
    "nozzle_center_y": 0.0,
    "pump_top_z": 97.5,
    "collar_radius": 10.7,
    "actuator_radius": 6.8,
    "label_width": 53.6,
    "label_height": 29.0,
    "label_center_z": 24.0,
}


def rounded_rect(width, depth, radius, x=0, z=0):
    """Rounded horizontal section, centered at (x, 0, z)."""
    x0, x1 = x - width / 2, x + width / 2
    y0, y1 = -depth / 2, depth / 2
    wire = cq.Wire.makePolygon([(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)],
                               close=True)
    return wire.fillet2D(radius, wire.Vertices())


def cylinder(radius, bottom, height, x, y=0):
    return cq.Workplane("XY", origin=(x, y, bottom)).circle(radius).extrude(height)


def build_body(p):
    width, depth, height = (p["body_width"], p["body_depth"], p["body_height"])
    inset = p["body_edge_radius"]
    shoulder = p["shoulder_height"]
    left, slim = p["shoulder_left_inset"], p["shoulder_depth_inset"]
    # XY stations describe the rounded foot, straight label faces and drafted shoulder.
    stations = [
        (0, 2 * inset, 2 * inset, 2.5, 0),
        (0.7, 2.9, 2.9, 2.8, 0),
        (2.0, 1.9, 1.9, 3.3, 0),
        (7, 0.1, 0.4, 3.8, 0),
        (11, 0, 0, 3.8, 0),
        (height - shoulder - 1, 0, 0, 3.8, 0),
        (height - shoulder + 1, 0.25, 0.4, 3.8, 0),
        (height - 1.6, left, slim, 3.0, 0),
        (height - 0.5, left + 0.8, slim + 1.1, 2.8, 0.2),
        (height, left + 2.0, slim + 2.4, 2.6, 0.65),
    ]
    sections = [rounded_rect(width - dw, depth - dd, radius, p["body_center_x"] + dw / 2 - right, z)
                for z, dw, dd, radius, right in stations]
    body = cq.Workplane(obj=cq.Solid.makeLoft(sections, ruled=True))

    # The outer rail follows the side of the shoulder. Depth tapers into the hand grip.
    top, right = p["loop_top_z"], p["loop_outside_x"]
    root_w, grip_w = p["loop_root_width"], p["loop_top_width"]
    root_d, grip_d = p["loop_root_depth"], p["loop_top_depth"]
    loop_stations = [
        (height - 3, root_w + 2.0, root_d + 2.0, 0.1, 3.0),
        (height - 0.7, root_w, root_d, 0, 2.9),
        (height + 4, root_w - 1.8, root_d - 5, 0, 2.5),
        (height + 10, grip_w + 0.6, grip_d + 1.5, 0.2, 2.0),
        (top - 3, grip_w, grip_d, 0.5, 1.65),
        (top - 1.0, grip_w - 1.1, grip_d - 1.0, 0.9, 1.65),
        (top, grip_w - 3.0, grip_d - 3.2, 1.8, 1.4),
    ]
    loop_sections = [rounded_rect(w, d, radius, right - setback - w / 2, z)
                     for z, w, d, setback, radius in loop_stations]
    loop = cq.Workplane(obj=cq.Solid.makeLoft(loop_sections, ruled=True))
    body = body.union(loop)

    opening_h = p["loop_opening_height"]
    opening_center = p["loop_opening_bottom_z"] + opening_h / 2
    opening = (cq.Workplane("XY")
               .box(p["loop_opening_width"], depth + 2, opening_h)
               .edges("|Y").fillet(2.0)
               .translate((p["loop_opening_center_x"], 0, opening_center)))
    body = body.cut(opening)
    pocket = cq.Solid.extrudeLinear(
        rounded_rect(width - 2 * inset - 7, depth - 2 * inset - 7, 2.4, x=p["body_center_x"], z=-1),
        [], (0, 0, 1 + p["base_recess_depth"]))
    body = body.cut(pocket)

    # A short molded neck is visible between the shoulder and the cap.
    body = body.union(cylinder(p["collar_radius"] - 1.0, height - 1.1, 2.6,
                               p["nozzle_center_x"], p["nozzle_center_y"]))

    # The rear molded capacity mark is recessed within a shallow raised frame.
    plane = cq.Plane(origin=(p["body_center_x"], depth / 2 - 0.1, 19.0),
                     xDir=(-1, 0, 0), normal=(0, 1, 0))
    frame = (cq.Workplane(plane).rect(36.5, 12.6).rect(34.1, 10.3)
             .extrude(0.65).edges("|Y").fillet(0.75))
    body = body.union(frame)
    for text, z, size in (("CAPACITY", 2.2, 4.4), ("2.7 fl oz   80 mL", -2.5, 3.4)):
        mark = (cq.Workplane(plane).center(0, z)
                .text(text, size, 0.48, font="DejaVu Sans", kind="bold", combine=False))
        body = body.union(mark)
    return body


def build_pump(p):
    bottom = p["body_height"] + 1.5
    length = p["pump_top_z"] - bottom
    collar, stem = p["collar_radius"], p["actuator_radius"]
    # Revolve the skirt, shoulder and actuator as one solid.
    profile = [(0, 0), (collar - 0.9, 0), (collar - 0.48, 0.7),
               (collar - 0.48, length * 0.40), (collar - 0.9, length * 0.47),
               (stem + 1.4, length * 0.56), (stem + 1.0, length * 0.67),
               (stem, length * 0.71), (stem, length - 1.0),
               (stem - 0.65, length), (0, length)]
    pump = cq.Workplane("XZ").polyline(profile).close().revolve()
    # The grip is a continuous serrated extrusion joined to the revolved core.
    ribs = 64
    grip = [(r * cos((i + f) * tau / ribs), r * sin((i + f) * tau / ribs))
            for i in range(ribs)
            for f, r in ((0, collar - 0.70), (0.2, collar), (0.53, collar),
                         (0.73, collar - 0.70))]
    skirt = (cq.Workplane("XY").workplane(offset=1.0)
             .polyline(grip).close().extrude(length * 0.39))
    pump = pump.union(skirt)
    outlet = (cq.Workplane("XZ", origin=(0, -stem - 1, length - 4.4))
              .circle(1.4).extrude(-2.45))
    return pump.cut(outlet).translate((p["nozzle_center_x"], p["nozzle_center_y"], bottom))


def angular_digit(plane, x, z, slash):
    """Re-create the angular capacity numeral as shallow decal geometry."""
    outline = [(-4.1, -6), (4.1, -6), (5.1, -5), (5.1, 5),
               (4.1, 6), (-4.1, 6), (-5.1, 5), (-5.1, -5)]
    digit = cq.Workplane(plane).center(x, z).polyline(outline).close().extrude(0.07)
    if slash:
        holes = [[(-3.45, -3.9), (-3.45, 3.7), (-2.95, 4.25), (3.2, 4.25)],
                 [(3.45, 3.9), (3.45, -3.7), (2.95, -4.25), (-3.2, -4.25)]]
    else:
        holes = [[(-3.45, top - 3.4), (3.45, top - 3.4), (3.45, top - 0.5),
                  (2.95, top), (-2.95, top), (-3.45, top - 0.5)]
                 for top in (4.5, -1.1)]
    for hole in holes:
        cut = cq.Workplane(plane).center(x, z).polyline(hole).close().extrude(0.1)
        digit = digit.cut(cut)
    return digit


def build_parts():
    p = DIM
    body = build_body(p)
    pump = build_pump(p)
    label_y = -p["body_depth"] / 2
    label_bottom = p["label_center_z"] - p["label_height"] / 2
    label = (cq.Workplane("XY").box(p["label_width"], 0.24, p["label_height"])
             .edges("|Y").fillet(0.65)
             .translate((0, label_y + 0.02, p["label_center_z"])))
    parts = {"body": body, "pump": pump, "label": label}
    try:
        plane = cq.Plane(origin=(0, label_y - 0.10, 0), xDir=(1, 0, 0), normal=(0, -1, 0))
        lines = [("PUMP", -14, 31.0, 5.0), ("SPRAY", -14, 25.3, 4.4),
                 ("ISE MOUNT", -14, label_bottom + 6.0, 3.3),
                 ("2.7 FL OZ", 16.5, label_bottom + 11.5, 1.6),
                 ("CAMPING SERIES PRODUCTS", 11.0, label_bottom + 4.0, 1.35)]
        lettering = cq.Workplane(plane)
        for word, x, z, size in lines:
            piece = (cq.Workplane(plane).center(x, z)
                     .text(word, size, 0.07, font="DejaVu Sans", kind="bold", combine=False))
            lettering = lettering.add(piece)
        lettering = lettering.add(angular_digit(plane, 4.5, 28.7, slash=False))
        lettering = lettering.add(angular_digit(plane, 16.3, 28.7, slash=True))
        rule = cq.Workplane(plane).center(10.3, 21.3).rect(22.2, 0.32).extrude(0.07)
        parts["lettering"] = lettering.add(rule)
    except Exception as exc:
        print("lettering omitted:", exc)
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
