


from h5py import Group
from itertools import groupby




def get_meta_data(dset):
    # convert_key = lambda key: key.lower().replace(" ", "_")

    field_mapping = {
        "PDE": "pde",
        "Fields Scheme": "fields_scheme",
        "Fields": "fields",
        "Constants": "const",
        "Field Desc": "field_desc",
        "Constant Desc": "const_desc",
        "Dt": "dt",
    }

    meta_attrs = dset["sims"].attrs

    meta = {field_mapping[field]: meta_attrs[field] for field in field_mapping.keys() if field in meta_attrs}

    group = dset["sims"][f'{next(iter(dset["sims"]))}']
    if isinstance(group, Group):
        first_sim = dset["sims"][f'{next(iter(dset["sims"]))}/0']
        sim_shape = first_sim.shape
        num_spatial_dim = len(sim_shape) - 1
    else:
        first_sim = group
        sim_shape = first_sim.shape
        num_spatial_dim = len(sim_shape) - 2 # subtract 2 for frame and field dimension

    num_fields = len(list(groupby(meta["fields_scheme"])))  # TODO

    meta.update(
        {
            "num_sims": len(dset["sims"]),
            "num_const": len(meta["const"]),
            "sim_shape": sim_shape,
            "num_frames": sim_shape[0],
            "num_sca_fields": sim_shape[1],
            "num_fields": num_fields,
            "num_spatial_dim": num_spatial_dim,
        }
    )

    return meta