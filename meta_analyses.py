import json
import numpy as np

import nibabel as nib
from nilearn.masking import apply_mask

import nimare
from nimare.meta.ibma import (stouffers, fishers, weighted_stouffers,
                              rfx_glm, ffx_glm, mfx_glm)

dset_file = 'nidm_pain_dset_with_subpeaks_hpc.json'
with open(dset_file, 'r') as fo:
    dset_dict = json.load(fo)
db = nimare.dataset.Database(dset_file)
dset = db.get_dataset()

mask_img = dset.mask


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
    elif t == 't!z':
        # Get t-image only if z-image doesn't exist
        temp = cdict['images'].get('z')
        if temp is None:
            temp = cdict['images'].get('t')
        else:
            temp = None
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
        files = []
        cdict = ddict[study]['contrasts']['1']
        for t in types:
            temp = _get_file(cdict, t)
            if temp:
                files.append(temp)

        if len(files) == len(types):
            all_files.append(files)
    all_files = list(map(list, zip(*all_files)))
    return all_files

#
#
# CBMAs
#
#
# MKDA Chi2 analysis with FDR
mkda_chi2_fdr = nimare.meta.cbma.MKDAChi2(dset, ids=dset.ids, ids2=dset.ids,
                                          kernel__r=10)
mkda_chi2_fdr.fit(corr='FDR')
mkda_chi2_fdr.results.save_results(output_dir='results/', prefix='mkda_chi2_fdr')

# ALE
ale = nimare.meta.cbma.ALE(dset, ids=dset.ids)
ale.fit(n_iters=10000, n_cores=12)
ale.results.save_results(output_dir='results/', prefix='ale')

# SCALE
ijk = np.loadtxt('neurosynth_mni_2mm_ijk.txt')
scale = nimare.meta.cbma.SCALE(dset, ids=dset.ids, ijk=ijk)
scale.fit(n_iters=10000, n_cores=12)
scale.results.save_results(output_dir='results/', prefix='scale')

# MKDA Density analysis
mkda_density = nimare.meta.cbma.MKDADensity(dset, ids=dset.ids, kernel__r=10)
mkda_density.fit(n_iters=10000, n_cores=12)
mkda_density.results.save_results(output_dir='results/', prefix='mkda_density')

# KDA
kda = nimare.meta.cbma.KDA(dset, ids=dset.ids, kernel__r=10)
kda.fit(n_iters=10000, n_cores=12)
kda.results.save_results(output_dir='results/', prefix='kda')

#
#
# Z-based IBMAs
#
#
# Get z-maps
# Regular z maps
z_files = get_files(dset_dict, ['z'])
z_imgs = [nib.load(f) for f in z_files[0]]
z_data = apply_mask(z_imgs, mask_img)

# T maps to be converted to z
t_files, t_ns = get_files(dset_dict, ['t!z', 'n'])
t_imgs = [nib.load(f) for f in t_files]
t_data_list = [apply_mask(t_img, mask_img) for t_img in t_imgs]
tz_data_list = [nimare.utils.t_to_z(t_data, t_ns[i]-1) for i, t_data
                in enumerate(t_data_list)]
tz_data = np.vstack(tz_data_list)

# Combine
z_data = np.vstack((z_data, tz_data))

# Fisher's
result1 = fishers(z_data, mask_img)
result1.save_results(output_dir='results/', prefix='fishers')

# Stouffer's
# Fixed-effects inference
result2 = stouffers(z_data, mask_img, inference='ffx', null='theoretical',
                    n_iters=None)
result2.save_results(output_dir='results/', prefix='stouffers_ffx')

# Random-effects inference with theoretical null
result3 = stouffers(z_data, mask_img, inference='rfx', null='theoretical',
                    n_iters=None)
result3.save_results(output_dir='results/', prefix='stouffers_rfx')

# Random-effects inference with empirical null
# Do not use FWE with empirical null
result4 = stouffers(z_data, mask_img, inference='rfx', null='empirical',
                    n_iters=10000, corr='FDR')
result4.save_results(output_dir='results/', prefix='z_perm')

# Get z-maps + sample sizes
# Regular z maps
z_files, ns = get_files(dset_dict, ['z', 'n'])
z_imgs = [nib.load(f) for f in z_files]
z_data = apply_mask(z_imgs, mask_img)

# T maps to be converted to z
t_files, t_ns = get_files(dset_dict, ['t!z', 'n'])
t_imgs = [nib.load(f) for f in t_files]
t_data_list = [apply_mask(t_img, mask_img) for t_img in t_imgs]
tz_data_list = [nimare.utils.t_to_z(t_data, t_ns[i]-1) for i, t_data
                in enumerate(t_data_list)]
tz_data = np.vstack(tz_data_list)

# Combine
z_data = np.vstack((z_data, tz_data))
ns = np.concatenate((ns, t_ns))
sample_sizes = np.array(ns)

# Weighted Stouffer's
result5 = weighted_stouffers(z_data, sample_sizes, mask_img)
result5.save_results(output_dir='results/', prefix='stouffers_weighted')

#
#
# Contrast-based IBMAs
#
#
# Get contrast maps + contrast standard error maps + sample sizes
con_files, se_files, ns = get_files(dset_dict, ['con', 'se', 'n'])
con_imgs = [nib.load(f) for f in con_files]
se_imgs = [nib.load(f) for f in se_files]
con_data = apply_mask(con_imgs, mask_img)
se_data = apply_mask(se_imgs, mask_img)
sample_sizes = np.array(ns)

# FFX GLM
result6 = ffx_glm(con_data, se_data, sample_sizes, mask_img, equal_var=True)
result6.save_results(output_dir='results/', prefix='ffx_glm')

# MFX GLM
result0 = mfx_glm(con_data, se_data, sample_sizes, mask_img,
                  work_dir='mfx_glm/')
result0.save_results(output_dir='results/', prefix='mfx_glm')

# Get contrast maps
con_files = get_files(dset_dict, ['con'])
con_files = con_files[0]
con_imgs = [nib.load(f) for f in con_files]
con_data = apply_mask(con_imgs, mask_img)

# RFX GLM
# Theoretical null distribution
result7 = rfx_glm(con_data, mask_img, null='theoretical', n_iters=None)
result7.save_results(output_dir='results/', prefix='rfx_glm')

# Empirical null distribution
result8 = rfx_glm(con_data, mask_img, null='empirical', n_iters=10000,
                  corr='FDR')
result8.save_results(output_dir='results/', prefix='contrast_perm')

# MKDA Chi2 analysis with FWE
mkda_chi2_fwe = nimare.meta.cbma.MKDAChi2(dset, ids=dset.ids, ids2=dset.ids,
                                          kernel__r=10)
mkda_chi2_fwe.fit(corr='FWE', n_iters=10000, n_cores=12)
mkda_chi2_fwe.results.save_results(output_dir='results/', prefix='mkda_chi2_fwe')

