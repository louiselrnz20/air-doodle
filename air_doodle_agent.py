"""Air Doodle Agent: trace a shape in the webcam view and stamp its picture.

Point with the index finger to draw. Hold a fist to recognize and replace the
stroke with a clean picture. Open your palm to clear the scene. Q quits.
"""
from __future__ import annotations
import argparse, math, time, urllib.request
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp

ROOT=Path(__file__).resolve().parent
MODEL=ROOT/'hand_landmarker.task'
MODEL_URL='https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'
ART=ROOT/'object_pictures'
TRAINING=ROOT/'learned_shape_templates.npz'
MASK_SIDE=128
STABLE_FRAMES=4
FIST_STABLE_FRAMES=18
OBJECTS=('flower','house','cat','tree','sun','heart','star','fish')
LABELS={x:x.title() for x in OBJECTS}

def trace_mask(points,side=MASK_SIDE):
    p=np.asarray(points,dtype=np.float32)
    if p.ndim!=2 or len(p)<4:return None
    lo=p.min(axis=0);hi=p.max(axis=0);span=hi-lo;extent=float(max(span))
    if extent<1e-3:return None
    scale=(side-20)/extent;xy=np.round((p-(lo+hi)/2)*scale+side/2).astype(np.int32)
    gap=float(np.linalg.norm(p[0]-p[-1]))
    is_closed=gap<.42*float(np.linalg.norm(span))
    mask=np.zeros((side,side),np.uint8)
    cv2.polylines(mask,[xy.reshape(-1,1,2)],is_closed,255,4,cv2.LINE_AA)
    _,mask=cv2.threshold(mask,100,255,cv2.THRESH_BINARY)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    return cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations=1)

def chamfer_score(query,template):
    if query is None or not np.any(query) or not np.any(template):return float('inf')
    dq=cv2.distanceTransform(255-query,cv2.DIST_L2,3)
    dt=cv2.distanceTransform(255-template,cv2.DIST_L2,3)
    q=query>0;t=template>0
    return float((dq[t].mean()+dt[q].mean())/(2*MASK_SIDE))

def load_templates():
    if not TRAINING.exists():raise FileNotFoundError(f'Missing learned examples: {TRAINING}')
    data=np.load(TRAINING);masks=data['masks']
    labels=[str(x) for x in (data['labels'] if 'labels' in data else data['names'])]
    samples={name:[] for name in OBJECTS}
    for label,mask in zip(labels,masks):samples.setdefault(label,[]).append(mask)
    return samples

def save_templates(samples):
    masks=[];labels=[]
    for name,items in samples.items():
      for mask in items:masks.append(mask);labels.append(name)
    np.savez_compressed(TRAINING,masks=np.stack(masks),labels=np.array(labels),names=np.array(list(samples)))

def learn_example(points,name,samples):
    mask=trace_mask(points)
    if mask is None:return False
    samples.setdefault(name,[]).append(mask)
    # Keep the original outline plus the 15 most recent corrections per class.
    samples[name]=samples[name][:1]+samples[name][1:][-15:]
    save_templates(samples)
    return True

def classify(points,trained=None):
    return classify_mask(trace_mask(points),trained)

def classify_mask(query,trained=None):
    samples=trained or load_templates()
    if query is None:return None,float('inf')
    scores=sorted((min(chamfer_score(query,mask) for mask in masks),name) for name,masks in samples.items() if masks)
    return (scores[0][1],scores[0][0]) if scores else (None,float('inf'))

def make_stamp(points,samples,force_name=None):
    name,score=classify(points,samples) if force_name is None else (force_name,0.0)
    if name is None:return None,'Trace a little longer, then hold fist or press SPACE'
    xs=[p[0] for p in points];ys=[p[1] for p in points]
    center=(sum(xs)/len(xs),sum(ys)/len(ys));size=max(max(xs)-min(xs),max(ys)-min(ys),105)*1.35
    learn_example(points,name,samples) if force_name is not None else None
    stamp=(name,center[0],center[1],size,list(points))
    quality='close fit' if score<.10 else 'best fit'
    return stamp,f'{LABELS[name]} picture | {quality} ({score:.2f})'

def finger_extended(lm,tip,pip,w,h):
    wrist=np.array([lm[0].x*w,lm[0].y*h]); t=np.array([lm[tip].x*w,lm[tip].y*h]); p=np.array([lm[pip].x*w,lm[pip].y*h])
    return np.linalg.norm(t-wrist)>np.linalg.norm(p-wrist)*1.16

def hand_pose(lm,w,h):
    # Index=8/6, middle=12/10, ring=16/14, little=20/18.
    e=[finger_extended(lm,t,p,w,h) for t,p in ((8,6),(12,10),(16,14),(20,18))]
    if all(e):return 'OPEN'
    if not any(e):return 'FIST'
    if e[0] and not any(e[1:]):return 'POINT'
    return 'OTHER'

class GestureGate:
    def __init__(self): self.candidate=None;self.count=0;self.active=None
    def update(self,pose):
      if pose!=self.candidate:self.candidate=pose;self.count=1
      else:self.count+=1
      required=FIST_STABLE_FRAMES if pose=='FIST' else STABLE_FRAMES
      if self.count>=required:self.active=pose
      return self.active

def ensure_model():
    if not MODEL.exists():
      print('Downloading official MediaPipe hand-landmarker model...')
      urllib.request.urlretrieve(MODEL_URL,MODEL)

def load_art():
    # Archive previews sometimes launch the script from a temporary copy that
    # contains the .py file but not its sibling asset folder. Check the script
    # directory first, then the current working directory and its parent.
    candidates=[]
    for folder in (ART,Path.cwd()/'object_pictures',ROOT.parent/'object_pictures'):
      if folder not in candidates:candidates.append(folder)
    asset_dir=next((folder for folder in candidates if all((folder/f'{name}.png').is_file() for name in OBJECTS)),None)
    if asset_dir is None:
      expected=ART/'flower.png'
      raise FileNotFoundError(
        f'Missing picture assets. Extract the complete Air_Doodle_Source.zip to a regular folder, '
        f'keep object_pictures beside air_doodle_agent.py, and run the script from that folder. '
        f'Expected at least: {expected}'
      )
    images={}
    for name in OBJECTS:
      image=cv2.imread(str(asset_dir/f'{name}.png'),cv2.IMREAD_UNCHANGED)
      if image is None:raise FileNotFoundError(f'Could not read picture file: {asset_dir/name}.png')
      images[name]=image
    return images

def alpha_stamp(frame,image,cx,cy,size):
    h,w=frame.shape[:2];tw=max(64,min(270,int(size)));th=tw
    image=cv2.resize(image,(tw,th),interpolation=cv2.INTER_AREA)
    x=max(0,min(w-tw,int(cx-tw/2)));y=max(0,min(h-th,int(cy-th/2)))
    if image.shape[2]==3:frame[y:y+th,x:x+tw]=image;return
    alpha=image[:,:,3:4].astype(np.float32)/255.;roi=frame[y:y+th,x:x+tw].astype(np.float32)
    frame[y:y+th,x:x+tw]=(alpha*image[:,:,:3]+(1-alpha)*roi).astype(np.uint8)

def demo():
    trained=load_templates()
    print('Training-sheet contour round-trip (self-check only; not a new-drawing/user evaluation).')
    for name,masks in trained.items():
      mask=masks[0] if masks else None
      if mask is None:print(f'{name:8} -> no example');continue
      contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
      contour=max(contours,key=cv2.contourArea) if contours else None
      found,score=classify(contour.reshape(-1,2).tolist(),trained) if contour is not None else (None,float('inf'))
      print(f'{name:8} -> {found or "no match":8} contour-distance={score:.3f}  {"PASS" if found==name else "CHECK"}')

def camera(index=0):
    ensure_model();art=load_art();trained=load_templates();cap=cv2.VideoCapture(index)
    if not cap.isOpened():raise RuntimeError(f'Cannot open camera {index}; try --camera-index 1')
    V=mp.tasks.vision; opts=V.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),running_mode=V.RunningMode.VIDEO,num_hands=1)
    gate=GestureGate();last_pose=None;trace=[];stamps=[];last_stamp='Draw with one finger; hold fist or press SPACE when finished';last_time=0;fist_acted=False;filt=None;drawing_blocked=False
    try:
      with V.HandLandmarker.create_from_options(opts) as detector:
       while True:
        ok,frame=cap.read()
        if not ok:break
        frame=cv2.flip(frame,1);h,w=frame.shape[:2]
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB);now=max(last_time+1,int(time.monotonic()*1000));last_time=now
        result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),now)
        pose=None;tip=None
        if result.hand_landmarks:
          lm=result.hand_landmarks[0];pose=hand_pose(lm,w,h);tip=(int(lm[8].x*w),int(lm[8].y*h))
        stable=gate.update(pose)
        if stable!='POINT':drawing_blocked=False
        if stable=='POINT' and pose=='POINT' and tip and not drawing_blocked:
          point=np.array(tip,dtype=np.float32);filt=point if filt is None else .35*point+.65*filt
          q=tuple(np.round(filt).astype(int))
          if not trace or math.dist(trace[-1],q)>3:trace.append(q)
          fist_acted=False
        elif stable=='FIST' and trace and not fist_acted:
          stamp,last_stamp=make_stamp(trace,trained);fist_acted=True
          if stamp:stamps.append(stamp)
          trace=[];filt=None;drawing_blocked=True
        elif stable=='OPEN' and last_pose!='OPEN':
          trace=[];stamps=[];filt=None;last_stamp='Scene cleared'
        last_pose=stable
        for name,cx,cy,size,_stroke in stamps:alpha_stamp(frame,art[name],cx,cy,size)
        if len(trace)>1:
          for a,b in zip(trace,trace[1:]):cv2.line(frame,a,b,(40,240,255),5,cv2.LINE_AA)
          cv2.circle(frame,trace[-1],7,(255,255,255),-1,cv2.LINE_AA)
        cv2.rectangle(frame,(8,8),(w-8,108),(25,29,42),-1)
        cv2.putText(frame,'AIR DOODLE AGENT',(20,32),cv2.FONT_HERSHEY_SIMPLEX,.64,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(frame,last_stamp[:86],(20,56),cv2.FONT_HERSHEY_SIMPLEX,.48,(180,226,237),1,cv2.LINE_AA)
        cv2.putText(frame,'POINT draw | FIST hold 0.6 s or SPACE finish | OPEN palm / C clear',(20,80),cv2.FONT_HERSHEY_SIMPLEX,.43,(225,225,234),1,cv2.LINE_AA)
        cv2.putText(frame,f'1-8 correct last picture   |   Gesture: {stable or "show hand"}   |   Points: {len(trace)}',(20,99),cv2.FONT_HERSHEY_SIMPLEX,.42,(210,210,220),1,cv2.LINE_AA)
        cv2.imshow('Air Doodle Agent',frame);key=cv2.waitKey(1)&0xff
        if key in (ord('q'),27):break
        if key==ord('c'):trace=[];stamps=[];last_stamp='Scene cleared'
        if key in (ord(' '),13) and trace:
          stamp,last_stamp=make_stamp(trace,trained);trace=[];filt=None;drawing_blocked=True
          if stamp:stamps.append(stamp)
        if ord('1')<=key<=ord('8'):
          chosen=OBJECTS[key-ord('1')]
          if trace:
            stamp,last_stamp=make_stamp(trace,trained,chosen);trace=[];filt=None;drawing_blocked=True
            if stamp:stamps.append(stamp);last_stamp=f'Learned this outline as {LABELS[chosen]}'
          elif stamps:
            old=stamps[-1];stamps[-1]=(chosen,old[1],old[2],old[3],old[4]);learn_example(old[4],chosen,trained);last_stamp=f'Corrected to {LABELS[chosen]} - learned for next time'
    finally:cap.release();cv2.destroyAllWindows()

def main():
    p=argparse.ArgumentParser(description='Draw in the air and replace recognized shapes with object pictures.')
    p.add_argument('--camera-index',type=int,default=0);p.add_argument('--camera',action='store_true',help='open the webcam (also the default)');p.add_argument('--demo',action='store_true',help='run synthetic template checks without opening webcam')
    a=p.parse_args()
    if a.demo:demo()
    else:camera(a.camera_index)
if __name__=='__main__':main()
