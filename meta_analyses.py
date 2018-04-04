import json
import numpy as np

import nibabel as nib
from nilearn.masking import apply_mask
from nilearn.plotting import plot_stat_map

import nimare
from nimare.meta.cbma import MKDADensity, ALE
from nimare.meta.ibma import (stouffers, fishers, weighted_stouffers,
                              rfx_glm, ffx_glm)

dset_file = '/Users/tsalo/Documents/tsalo/NiMARE/nimare/tests/data/nidm_pain_dset.json'
with open(dset_file, 'r') as fo:
    dset_dict = json.load(fo)
db = nimare.dataset.Database(dset_file)
dset = db.get_dataset()

mask_img = dset.mask

logp_thresh = -np.log(.05)

def _get_file(cdict, t):
    """Return the file associated with a given data type within a
    folder if it exists. Otherwise, returns an empty list.
    """
    temp = ''
    if t == 'con':
        temp = cdict['images'].get('con')
    elif t == 'se':
        temp = cdict['images'].get('se')
    elif t == 't':
        temp = cdict['images'].get('t')
    elif t == 'z':
        temp = cdict['images'].get('z')
    elif t == 'n':
        temp = cdict.get('sample_sizes', [])
        if temp:
            temp = np.mean(temp)
    else:
        raise Exception('Input type "{0}" not recognized.'.format(t))

    return temp


def get_files(ddict, types):
    """Returns a list of files associated with a given data type
    from a set of subfolders within a directory. Allows for
    multiple data types and only returns a set of files from folders
    with all of the requested types.
    """
    all_files = []
    for study in ddict.keys():
        file_list = []
        cdict = ddict[study]['contrasts']['1']
        for type_ in types:
            temp = _get_file(cdict, type_)
            if temp:
                files.append(temp)

        if len(file_list) == len(types):
            all_files.append(file_list)
    all_files = list(map(list, zip(*all_files)))
    return all_files

files = get_files(dset_dict, ['z'])
z_imgs = [nib.load(f) for f in files[0]]
z_data = apply_mask(z_imgs, mask_img)

result = fishers(z_data, mask_img)
plot_stat_map(result.images['log_p'], threshold=logp_thresh,
              cut_coords=[0, 0, -8], draw_cross=False)

result = stouffers(z_data, mask_img, inference='ffx',
                   null='theoretical', n_iters=None)
plot_stat_map(result.images['log_p'], threshold=logp_thresh,
              cut_coords=[0, 0, -8], draw_cross=False)

result = stouffers(z_data, mask_img, inference='rfx',
                   null='theoretical', n_iters=None)
plot_stat_map(result.images['log_p'], threshold=logp_thresh,
              cut_coords=[0, 0, -8], draw_cross=False)

result = stouffers(z_data, mask_img, inference='rfx',
                   null='empirical', n_iters=10000)
plot_stat_map(result.images['log_p'], threshold=logp_thresh,
              cut_coords=[0, 0, -8], draw_cross=False)

z_files, ns = get_files(dset_dict, ['z', 'n'])
z_imgs = [nib.load(f) for f in z_files]
z_data = apply_mask(z_imgs, mask_img)
sample_sizes = np.array(ns)

result = weighted_stouffers(z_data, sample_sizes, mask_img)
plot_stat_map(result.images['log_p'], threshold=logp_thresh,
              cut_coords=[0, 0, -8], draw_cross=False)

con_files, se_files, ns = get_files(dset_dict, ['con', 'se', 'n'])
con_imgs = [nib.load(f) for f in con_files]
se_imgs = [nib.load(f) for f in se_files]
con_data = apply_mask(con_imgs, mask_img)
se_data = apply_mask(se_imgs, mask_img)
sample_sizes = np.array(ns)

result = ffx_glm(con_data, se_data, sample_sizes, mask_img, equal_var=True)
plot_stat_map(result.images['log_p'], threshold=logp_thresh,
              cut_coords=[0, 0, -8], draw_cross=False)

con_files = get_files(dset_dict, ['con'])
con_files = con_files[0]
con_imgs = [nib.load(f) for f in con_files]
con_data = apply_mask(con_imgs, mask_img)

result = rfx_glm(con_data, mask_img, null='theoretical', n_iters=None)
plot_stat_map(result.images['log_p'], threshold=logp_thresh,
              cut_coords=[0, 0, -8], draw_cross=False)

result = rfx_glm(con_data, mask_img, null='empirical', n_iters=10000)
plot_stat_map(result.images['log_p'], threshold=logp_thresh,
              cut_coords=[0, 0, -8], draw_cross=False)

mkda = MKDADensity(dset, ids=dset.ids, kernel__r=10)
mkda.fit(n_iters=10000)
plot_stat_map(mkda.results.images['vfwe'], cut_coords=[0, 0, -8],
              draw_cross=False)

ale = ALE(dset, ids=dset.ids)
ale.fit(n_iters=10000)
plot_stat_map(ale.results.images['vthresh'], cut_coords=[0, 0, -8],
              draw_cross=False)
