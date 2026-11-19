"""Uncalibrated photo-to-CAD QA. Uses original JPGs if supplied, otherwise traced silhouette contours."""
from pathlib import Path
import argparse,json
import cv2
import numpy as np

REFERENCE={
 "1000048543.jpg":{"view":"front","bbox":[238,108,327,457]},
 "1000048547.jpg":{"view":"rear","bbox":[264,238,365,458]},
 "1000048548.jpg":{"view":"right","bbox":[368,244,155,468]},
}
ROOT=Path(__file__).resolve().parents[1]

def segment_photo(src,bbox):
    im=cv2.imread(str(src))
    if im is None:raise FileNotFoundError(src)
    h,w=im.shape[:2]
    scale_x=w/1152;scale_y=h/1536
    x,y,bw,bh=[round(v*s) for v,s in zip(bbox,[scale_x,scale_y,scale_x,scale_y])]
    pad=max(5,round(62*scale_x))
    l,t=max(0,x-pad),max(0,y-pad)
    r,b=min(w,x+bw+pad),min(h,y+bh+pad)
    crop=im[t:b,l:r]
    sf=min(1,650/max(crop.shape[:2]))
    small=cv2.resize(crop,None,fx=sf,fy=sf,interpolation=cv2.INTER_AREA)
    rect=tuple(max(1,round(v*sf)) for v in (x-l,y-t,bw,bh))
    mask=np.zeros(small.shape[:2],np.uint8)
    bg=np.zeros((1,65),np.float64);fg=np.zeros((1,65),np.float64)
    cv2.grabCut(small,mask,rect,bg,fg,6,cv2.GC_INIT_WITH_RECT)
    binary=np.isin(mask,[cv2.GC_FGD,cv2.GC_PR_FGD]).astype("uint8")
    n,labels,stats,_=cv2.connectedComponentsWithStats(binary,8)
    if n<2:raise RuntimeError("No foreground for "+str(src))
    return small,((labels==1+np.argmax(stats[1:,cv2.CC_STAT_AREA]))*255).astype("uint8")

def contour_reference(view,source=ROOT/"refs/silhouettes.json"):
    contours=json.loads(source.read_text())["contours"]
    pts=np.asarray(contours[view],dtype=np.int32).reshape((-1,1,2))
    mask=np.zeros((384,384),np.uint8)
    cv2.fillPoly(mask,[pts],255)
    return mask

def render_mask(src):
    im=cv2.imread(str(src))
    if im is None:raise FileNotFoundError(src)
    diff=np.max(np.abs(im.astype(np.int16)-im[0,0].astype(np.int16)),axis=2)
    binary=(diff>25).astype("uint8")
    n,labels,stats,_=cv2.connectedComponentsWithStats(binary,8)
    if n<2:raise RuntimeError("No model silhouette")
    return ((labels==1+np.argmax(stats[1:,cv2.CC_STAT_AREA]))*255).astype("uint8")

def normalize(mask,canvas=384,target=330):
    ys,xs=np.where(mask>0)
    if not len(xs):raise ValueError("Empty silhouette")
    cut=mask[ys.min():ys.max()+1,xs.min():xs.max()+1]
    s=target/max(cut.shape)
    h,w=[max(1,round(v*s)) for v in cut.shape]
    cut=cv2.resize(cut,(w,h),interpolation=cv2.INTER_NEAREST)
    out=np.zeros((canvas,canvas),np.uint8)
    out[(canvas-h)//2:(canvas-h)//2+h,
        (canvas-w)//2:(canvas-w)//2+w]=cut
    return out

def compare_masks(photo,model):
    a=normalize(photo)>0;b=normalize(model)>0
    iou=float(np.logical_and(a,b).sum()/max(1,np.logical_or(a,b).sum()))
    ae=cv2.Canny(a.astype("uint8")*255,80,150)
    be=cv2.Canny(b.astype("uint8")*255,80,150)
    ad=cv2.distanceTransform((ae==0).astype("uint8"),cv2.DIST_L2,3)
    bd=cv2.distanceTransform((be==0).astype("uint8"),cv2.DIST_L2,3)
    chamfer=float((ad[be>0].mean()+bd[ae>0].mean())/2)
    overlay=np.zeros((384,384,3),np.uint8)
    overlay[a&b]=(175,210,175)
    overlay[a&~b]=(100,100,230)
    overlay[b&~a]=(235,150,100)
    return iou,chamfer,overlay

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--photos",type=Path,default=Path("refs/photos"))
    p.add_argument("--out",type=Path,default=Path("out"))
    p.add_argument("--min-iou",type=float,default=None)
    args=p.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    results=[]
    for filename,conf in REFERENCE.items():
        photo_path=args.photos/filename
        if photo_path.exists():
            _,photo=segment_photo(photo_path,conf["bbox"])
            source="original JPG"
        else:
            photo=contour_reference(conf["view"])
            source="stored photo contour"
        model=render_mask(args.out/(conf["view"]+".png"))
        iou,chamfer,overlay=compare_masks(photo,model)
        cv2.imwrite(str(args.out/(filename[:-4]+"-mask.png")),photo)
        cv2.imwrite(str(args.out/(filename[:-4]+"-overlay.png")),overlay)
        results.append({"photo":filename,"view":conf["view"],
                        "source":source,"iou":round(iou,4),
                        "chamfer_px":round(chamfer,2)})
        print(filename,source,"IoU",round(iou,3),"Chamfer",round(chamfer,2))
    (args.out/"comparison.json").write_text(json.dumps(
        {"method":"uncalibrated, normalized silhouettes","results":results},indent=2)+"\n")
    if args.min_iou is not None and any(x["iou"]<args.min_iou for x in results):
        raise SystemExit("Silhouette below threshold")
if __name__=="__main__":
    main()
