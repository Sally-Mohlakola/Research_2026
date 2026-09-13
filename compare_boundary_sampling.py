"""Two-seed diagnostic of image variance, not a ground-truth accuracy benchmark."""
import argparse
import json
from pathlib import Path
import numpy as np
import render_boundary as rb
import mitsuba as mi


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',nargs=2,type=Path,required=True)
    p.add_argument('--prior',nargs=2,type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    dirs=args.baseline+args.prior
    reports=[json.loads((d/'render.json').read_text()) for d in dirs]
    for report in reports:
        for key in ('width','height','spp','scene_depth','azimuth','checkpoint_sha256','geometry_sha256'):
            if report[key]!=reports[0][key]:
                raise ValueError(f'Incompatible render {key}')
        if report['mode']!='neural':
            raise ValueError('Compare neural proposal variants only')
    if reports[0]['seed']==reports[1]['seed'] or reports[2]['seed']==reports[3]['seed']:
        raise ValueError('Each pair requires independent seeds')
    if any(r.get('rdm_prior') is not None for r in reports[:2]) or any(r.get('rdm_prior') is None for r in reports[2:]):
        raise ValueError('Need baseline pair without prior and prior pair with prior')
    _,checkpoint=rb.load_model(reports[0]['model'])
    w,h=reports[0]['width'],reports[0]['height']
    scene,_,mesh=rb.make_scene(checkpoint,w,h,reports[0]['azimuth'])
    sensor=scene.sensors()[0]
    mask=np.zeros((h,w),dtype=bool)
    for y in range(h):
        for x in range(w):
            ray,_=sensor.sample_ray(0.,.5,mi.Point2f((x+.5)/w,(y+.5)/h),mi.Point2f(.5))
            si=scene.ray_intersect(ray)
            mask[y,x]=si.is_valid() and si.shape==mesh
    if not mask.any():
        raise ValueError('No primary diamond pixels')
    images=[np.asarray(mi.Bitmap(str(d/'frames/frame_0000.exr')),dtype=np.float64) for d in dirs]
    lum=[a @ np.array([.2126,.7152,.0722]) for a in images]
    results={}
    for name,offset in [('baseline',0),('prior',2)]:
        a,b=lum[offset:offset+2]
        results[name]=dict(diamond_mean_luminance=float(((a+b)/2)[mask].mean()),
                          paired_variance_estimate=float((.5*(a-b)**2)[mask].mean()),
                          seconds=[r['seconds'] for r in reports[offset:offset+2]])
    results['variance_ratio_prior_over_baseline']=results['prior']['paired_variance_estimate']/results['baseline']['paired_variance_estimate']
    results['diamond_pixels']=int(mask.sum())
    results['limitations']='Two seeds, one view, low spp; measures noise around learned target, not accuracy. Concurrent timings are not a controlled benchmark.'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
