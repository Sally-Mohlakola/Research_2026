"""Re-point a trained boundary operator at a different stone, without retraining.

This is the cross-cut generalisation probe. A boundary model is bound to the
mesh it was trained on in three places: the geometry buffers it decodes exit
coordinates against, the facet categorical whose width is the face count, and
the facet embedding that conditions the coordinate head. Swapping the mesh
means rewriting all three while keeping every learned weight.

Two things transfer and one does not, which is exactly what makes the test
informative. The encoder, the escape head and the coordinate head are all
expressed in a facet's own local frame -- barycentric logits and tangent slopes
-- so they are geometry-agnostic and carry over unchanged. The facet
categorical is not: class k means "the k-th triangle of the training mesh", and
that label has no meaning on another stone.

So the facet head needs a correspondence. Identity is the honest null: keep
index k as index k and accept that the labels are arbitrary. The default is
better -- a greedy assignment pairing each new facet with the unused old facet
whose outward normal points most nearly the same way, so the operator's learned
preference for a direction survives the swap. A permutation rather than nearest
matching keeps the softmax a distribution over distinct facets.

The resulting checkpoint renders through the ordinary pipeline, and the analytic
mode of the same run traces the new stone for real, so the pair is a matched
comparison on the target geometry.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from config.parameters import get_diamond_parameters
from ground_truth.cuts import make_diamond
from neural.boundary_model import frames

GEOMETRY_BUFFERS = ('triangles', 'tangent', 'bitangent', 'normal')


def greedy_assignment(new_normals, old_normals):
    """Permutation pairing new facets to old ones by outward normal agreement.

    Hungarian assignment would be optimal, but scipy is not a dependency here
    and greedy on a sorted similarity matrix is within a degree or two of it on
    64 facets. The mean alignment achieved is reported so the quality of the
    correspondence is visible rather than assumed.
    """
    similarity = new_normals @ old_normals.T
    order = np.dstack(np.unravel_index(np.argsort(similarity, axis=None)[::-1],
                                       similarity.shape))[0]
    permutation = np.full(len(new_normals), -1, dtype=np.int64)
    taken = np.zeros(len(old_normals), dtype=bool)
    for new, old in order:
        if permutation[new] < 0 and not taken[old]:
            permutation[new], taken[old] = old, True
    if (permutation < 0).any():
        raise ValueError('Assignment left a facet unmatched')
    return permutation


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, required=True, help='Trained source checkpoint')
    parser.add_argument('--diamond_name', required=True,
                        help='Target variant from config.parameters')
    parser.add_argument('--output', type=Path, required=True, help='Retargeted checkpoint')
    parser.add_argument('--correspondence', choices=('normal', 'identity'), default='normal',
                        help='How facet classes map from the old mesh to the new one')
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location='cpu', weights_only=True)
    if checkpoint.get('schema_version') != 1:
        raise ValueError('Unsupported model schema')
    parameters = get_diamond_parameters(args.diamond_name)
    vertices, faces = make_diamond(parameters)
    old_faces = checkpoint['faces'].numpy()
    if len(faces) != len(old_faces):
        raise ValueError('Target has %d faces, source head expects %d. The facet '
                         'categorical cannot be resized without retraining; pick a '
                         'tessellation with a matching face count.'
                         % (len(faces), len(old_faces)))

    triangles, tangent, bitangent, normal = frames(vertices, faces)
    old_normal = checkpoint['state_dict']['normal'].numpy()
    if args.correspondence == 'identity':
        permutation = np.arange(len(faces))
    else:
        permutation = greedy_assignment(normal, old_normal)
    alignment = np.degrees(np.arccos(np.clip(
        (normal*old_normal[permutation]).sum(1), -1., 1.)))

    state = dict(checkpoint['state_dict'])
    for name, value in zip(GEOMETRY_BUFFERS, (triangles, tangent, bitangent, normal)):
        state[name] = torch.as_tensor(value, dtype=torch.float32)
    index = torch.as_tensor(permutation, dtype=torch.long)
    state['facet.weight'] = state['facet.weight'][index].clone()
    state['facet.bias'] = state['facet.bias'][index].clone()
    state['embedding.weight'] = state['embedding.weight'][index].clone()

    radius = float(np.linalg.norm(vertices, axis=1).max())
    new_vertices = torch.as_tensor(vertices, dtype=torch.float32)
    new_faces = torch.as_tensor(np.asarray(faces), dtype=checkpoint['faces'].dtype)
    metadata = dict(checkpoint['metadata'])
    metadata['diamond_name'] = args.diamond_name
    metadata['parameters'] = parameters
    # render_boundary re-derives this hash and refuses to run on a mismatch, so
    # it has to be recomputed here over exactly the bytes that check sees.
    metadata['geometry_sha256'] = hashlib.sha256(
        new_vertices.numpy().tobytes()
        + new_faces.numpy().astype(np.uint32).tobytes()).hexdigest()

    retargeted = dict(checkpoint)
    retargeted.update(state_dict=state, vertices=new_vertices, faces=new_faces,
                      metadata=metadata, input_radius=radius)
    retargeted['retarget'] = dict(
        source=str(args.model),
        source_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
        source_diamond=checkpoint['metadata']['diamond_name'],
        target_diamond=args.diamond_name,
        correspondence=args.correspondence,
        permutation=permutation.tolist(),
        mean_alignment_deg=float(alignment.mean()),
        median_alignment_deg=float(np.median(alignment)),
        worst_alignment_deg=float(alignment.max()),
        source_input_radius=float(checkpoint['input_radius']),
        target_input_radius=radius,
        note='Weights are unchanged apart from a permutation of the facet '
             'categorical and embedding. No training was performed on the '
             'target geometry, so every rendered difference is transfer error.')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(retargeted, args.output)
    print('source %s (%s, %d faces)' % (args.model, checkpoint['metadata']['diamond_name'],
                                        len(old_faces)))
    print('target %s (%d vertices, %d faces)' % (args.diamond_name, len(vertices), len(faces)))
    print('correspondence %s: alignment mean %.1f deg, median %.1f deg, worst %.1f deg'
          % (args.correspondence, alignment.mean(), np.median(alignment), alignment.max()))
    print('input radius %.4f -> %.4f' % (checkpoint['input_radius'], radius))
    print('wrote %s' % args.output)
    args.output.with_suffix('.json').write_text(json.dumps(retargeted['retarget'], indent=2))


if __name__ == '__main__':
    main()
