"""Build a model-specific directional RDM prior using training entries only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from neural.boundary_model import load_model
from neural.boundary_data import load_boundary
from neural.boundary_prior import BoundaryPrior


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--theta_bins',type=int,default=4)
    p.add_argument('--phi_bins',type=int,default=8)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.output.suffix != '.npz' or min(args.theta_bins,args.phi_bins)<1:
        p.error('Use an .npz output and positive bin counts')
    model,checkpoint=load_model(args.model)
    if hashlib.sha256(args.data.read_bytes()).hexdigest()!=checkpoint['source_sha256']:
        raise ValueError('Use the exact training-source dataset for this checkpoint')
    data,metadata=load_boundary(args.data)
    keep=np.isin(data['entry_id'],checkpoint['train_entry_ids']) & (data['status']==0)
    if not keep.any():
        raise ValueError('No escaped training entries')
    size=args.theta_bins*args.phi_bins
    # Temporary uniform table provides the shared binning convention.
    prior=BoundaryPrior(torch.full((size,size),1/size),args.theta_bins,args.phi_bins,model)
    incoming=prior.bins(torch.from_numpy(-data['entry_direction'][keep]))
    outgoing=prior.bins(torch.from_numpy(data['exit_direction'][keep]))
    counts=torch.bincount(incoming*size+outgoing,minlength=size*size).reshape(size,size).float()
    uniform=prior.solid_angle/(4*np.pi)
    rows=counts.sum(-1,keepdim=True)
    pmf=torch.where(rows>0,counts/rows.clamp_min(1),uniform[None,:])
    pmf=.95*pmf+.05*uniform[None,:]
    info=dict(schema_version=1,theta_bins=args.theta_bins,phi_bins=args.phi_bins,
              model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
              geometry_sha256=metadata['geometry_sha256'],training_records=int(keep.sum()),
              uniform_fraction=.05,coordinates='object-space world directions',
              conditioning='entry transmission and finite-depth escape',
              note='Boundary-record RDM, not legacy first-hit-local depth>k RDM')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,pmf=pmf.numpy(),metadata_json=np.array(json.dumps(info)))
    print(json.dumps(info,indent=2))


if __name__=='__main__':
    main()
