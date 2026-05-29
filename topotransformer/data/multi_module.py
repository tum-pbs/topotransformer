from typing import TypeVar, List, Iterable, Union, Optional, Tuple

import math

import lightning.pytorch
import torch
import torch.distributed
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset, Dataset
from torchvision.transforms.v2 import Transform

from omegaconf import ListConfig
from .metadata_dataset import MetadataDataset


from .pbdl_module import get_transforms, get_datasets
from .metadata_remapping import update_boundary_condition

from pdetransformer.data import MultiDataModule as MD
from pdetransformer.data.multi_module import ConcatDatasetDifferentShapes as CDDS
from pdetransformer.data.cached_dataset import CachedDataset
from pdetransformer.data.utils import SubsetSequentialSampler


def collate_with_configs(batch):
    """
    Custom collate function that handles config and topology_config specially.
    
    These fields are kept as lists of dicts instead of being converted to tensors.
    All other fields are handled with default collation.
    """
    if not batch:
        return {}
    
    # Check if any sample has config or topology_config
    has_config = any("config" in sample for sample in batch)
    has_topology_config = any("topology_config" in sample for sample in batch)
    
    # Extract configs if they exist
    configs = []
    topology_configs = []
    
    if has_config:
        configs = [sample.pop("config", None) for sample in batch]
    if has_topology_config:
        topology_configs = [sample.pop("topology_config", None) for sample in batch]
    
    # Use default collate for the rest
    from torch.utils.data._utils.collate import default_collate
    collated = default_collate(batch)
    
    # Add configs back as lists
    if has_config:
        collated["config"] = configs
    if has_topology_config:
        collated["topology_config"] = topology_configs
    
    return collated

class MultiDataModule(MD):
    r'''
    Wrapper around multiple PBDL datasets for joint datasets with PyTorch Lightning.
    In comparison to the joint module, this class concatenates the datasets before attaching them to a LightningDataModule.

    Some arguments are either individual objects if they are the same for all sub datasets,
    or lists of the corresponding type for individual configuration of each sub datasets.
    Args:
        path_index: index dictionary with the directory for each dataset type
        dataset_names: list with names of local datasets
        dataset_type: type of the dataset
        unrolling_steps: number of time steps between start and end of sequence
        batch_size: batch size for each GPU, i.e. larger total batch size depending on number of GPUs
        num_workers: number of worker threads in the dataloaders
        test_unrolling_steps: number of time steps between start and end of sequence for the test dataset
        cache_strategy: strategy for caching sub data sets from ["none", "testOnly", testAndVal", "all"]
        different_resolution_strategy: strategy for handling different resolutions in the sub datasets from ["none", "rescale", "crop"]
        target_size: target size for spatial dimensions of the concatenated datasets
        max_cache_size: maximum number of cached items
    '''


    # runs on every GPU
    def setup(self, stage: str):
        # prevent reload when all datasets already exist
        if self.set_train and self.set_val and self.set_test:
            return

        for i in range(len(self.dataset_names)):
            name = self.dataset_names[i]
            datatype = self.dataset_type[i] if (isinstance(self.dataset_type, ListConfig) or isinstance(self.dataset_type, list)) else self.dataset_type
            directory = self.path_index[datatype]
            steps = self.unrolling_steps[i] if (isinstance(self.unrolling_steps, ListConfig) or isinstance(self.unrolling_steps, list)) else self.unrolling_steps
            test_steps = self.test_unrolling_steps[i] if (isinstance(self.test_unrolling_steps, ListConfig) or isinstance(self.test_unrolling_steps, list)) else self.test_unrolling_steps
            variable_dt = self.variable_dt[i] if isinstance(self.variable_dt, ListConfig) else self.variable_dt
            test_variable_dt = self.test_variable_dt[i] if isinstance(self.test_variable_dt, ListConfig) else self.test_variable_dt
            variable_dt_stride = self.variable_dt_stride_maximum[i] if isinstance(self.variable_dt_stride_maximum, ListConfig) else self.variable_dt_stride_maximum
            variable_dt_stride = variable_dt_stride if variable_dt else 1

            test_variable_dt_stride = self.test_variable_dt_stride_maximum[i] if isinstance(self.test_variable_dt_stride_maximum, ListConfig) else self.test_variable_dt_stride_maximum
            test_variable_dt_stride = test_variable_dt_stride if test_variable_dt else 1

            # prepare transforms and datasets
            transform_train, transform_val, transform_test = get_transforms(datatype, name)
            # Extract normalize_target and normalize_channels from additional_kwargs to avoid duplicate arguments
            additional_kwargs_copy = dict(self.additional_kwargs)
            normalize_target = additional_kwargs_copy.pop("normalize_target", True)
            normalize_channels = additional_kwargs_copy.pop("normalize_channels", None)
            set_train, set_val, set_test = get_datasets(datatype, name, directory, steps, test_steps, variable_dt_stride,
                                                        test_variable_dt_stride, self.normalize_data, self.normalize_const,
                                                        self.crop_size, normalize_target,
                                                        normalize_channels,
                                                        **additional_kwargs_copy)

            # add metadata
            loading_metadata_train = {"type": datatype, "unrolling_steps": steps, "dataset_idx": i}
            loading_metadata_test = {"type": datatype, "unrolling_steps": test_steps, "dataset_idx": i}

            meta_train = MetadataDataset(set_train, loading_metadata_train, transform_train)
            meta_val = MetadataDataset(set_val, loading_metadata_train, transform_val)
            meta_test = MetadataDataset(set_test, loading_metadata_test, transform_test)

            self.subsets_train += [meta_train]
            self.subsets_val += [meta_val]
            self.subsets_test += [meta_test]

        self._prepare_gpu_split_and_cache("fit", ConcatDatasetDifferentShapes(self.subsets_train,
                                                                              self.different_resolution_strategy,
                                                                              downsample_factor=self.downsample_factor,
                                                                              max_channels=self.max_channels,
                                                                              max_constants=self.max_constants,
                                                                              target_size=self.target_size))
        self._prepare_gpu_split_and_cache("validate", ConcatDatasetDifferentShapes(self.subsets_val,
                                                                                   self.different_resolution_strategy,
                                                                                   downsample_factor=self.downsample_factor,
                                                                                   max_channels=self.max_channels,
                                                                                   max_constants=self.max_constants,
                                                                                   target_size=self.target_size))
        self._prepare_gpu_split_and_cache("test", ConcatDatasetDifferentShapes(self.subsets_test,
                                                                               self.different_resolution_strategy,
                                                                               downsample_factor=self.downsample_factor,
                                                                               max_channels=self.max_channels,
                                                                               max_constants=self.max_constants,
                                                                               target_size=self.target_size))


    def _prepare_gpu_split_and_cache(self, stage:str, dataset:Dataset, after_cache_transforms:Transform = None):
        r'''
        Creates split indices for each GPU and loads the dataset to the cache according to the cache strategy.
        Does not reload the cache if already loaded. Supplied dataset should match the stage.
        '''
        if stage == "fit" and not self.set_train:
            self.subset_indices_train = self._compute_gpu_subset_indices(len(dataset))

            if self.cache_strategy in ["all"] :
                self.set_train = CachedDataset(dataset, after_cache_transforms, self.max_cache_size)
                self.set_train.fill_cache_sequentially(self.subset_indices_train)
            else:
                self.set_train = dataset


        if stage == "validate" and not self.set_val:
            self.subset_indices_val = self._compute_gpu_subset_indices(len(dataset))

            if self.cache_strategy in ["testAndVal", "all"] :
                self.set_val = CachedDataset(dataset, after_cache_transforms, self.max_cache_size)
                self.set_val.fill_cache_sequentially(self.subset_indices_val)
            else:
                self.set_val = dataset


        if stage == "test" and not self.set_test:
            self.subset_indices_test = self._compute_gpu_subset_indices(len(dataset))

            if self.cache_strategy in ["testOnly", "testAndVal", "all"] :
                self.set_test = CachedDataset(dataset, after_cache_transforms, self.max_cache_size)
                self.set_test.fill_cache_sequentially(self.subset_indices_test)
            else:
                self.set_test = dataset

    def _compute_gpu_subset_indices(self, dataset_length:int) -> list:
        r'''
        Computes local data indices for the current GPU. Discards the last few data samples to ensure even subset size across GPUs.
        '''
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

        dataset_size = dataset_length - (dataset_length % world_size) # ensure even subset size
        subset_start = int(dataset_size * (rank / world_size))
        subset_end = int(dataset_size * ((rank + 1) / world_size))
        return range(subset_start, subset_end, 1)


    def train_dataloader(self):
        if self.shuffle_train_set:
            sampler = SubsetRandomSampler(self.subset_indices_train)
        else:
            sampler = SubsetSequentialSampler(self.subset_indices_train)
        sampler = SubsetRandomSampler(self.subset_indices_train)
        return DataLoader(
            self.set_train,
            drop_last=True,
            # prefetch_factor=self.prefetch_factor,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=collate_with_configs,
            num_workers = 0 if self.cache_strategy in ["all"] else self.num_workers)

    def val_dataloader(self):
        return DataLoader(
            self.set_val,
            # prefetch_factor=self.prefetch_factor,
            batch_size=self.batch_size,
            sampler=SubsetSequentialSampler(self.subset_indices_val),
            collate_fn=collate_with_configs,
            num_workers = 0 if self.cache_strategy in ["testAndVal", "all"] else self.num_workers)

    def test_dataloader(self):
        return DataLoader(
            self.set_test,
            # prefetch_factor=self.prefetch_factor,
            batch_size=100,
            sampler=SubsetSequentialSampler(self.subset_indices_test),
            collate_fn=collate_with_configs,
            drop_last=False,
            num_workers = 0 if self.cache_strategy in ["testOnly", "testAndVal", "all"] else self.num_workers)


class ConcatDatasetDifferentShapes(CDDS):
    def process_sample(self, sample, add_batch_dim=False):

        returnDictNoConditioning = super().process_sample(sample, add_batch_dim)

        data = sample["conditioning"]
        constants_norm = sample["constants_norm"]
        constants = sample["constants"]
        time_step_stride = sample["time_step_stride"].long()
        metadata = sample["physical_metadata"]
        loading_metadata = sample["loading_metadata"]

        # loading_metadata["dataset_idx"] = torch.Tensor([loading_metadata["dataset_idx"]]).long()

        # some datasets contain a larger number of dimensions in the domain extent -> TODO fix this in the underlying h5py file
        if self.dimension == 2:
            metadata['Domain Extent'] = metadata['Domain Extent'][:2]

        # downsampling
        if self.downsample_factor > 1:
            # average pooling
            data = torch.nn.functional.avg_pool2d(data, self.downsample_factor)

        # 2D spatial adjustment
        if self.dimension == 2 and self.target_size is not None:

            # shape already matches
            if data.shape[2] == self.target_size[0] and data.shape[3] == self.target_size[1]:
                pass

            # data too small -> bilinear interpolation
            elif data.shape[2] <= self.target_size[0] and data.shape[3] <= self.target_size[1]:
                data = torch.nn.functional.interpolate(data, size=self.target_size, mode="bilinear",
                                                       align_corners=False)

            # data too large -> random crop
            elif data.shape[2] >= self.target_size[0] and data.shape[3] >= self.target_size[1]:
                start = (
                    torch.randint(0, data.shape[2] - self.target_size[0] + 1, (1,)).item(),
                    torch.randint(0, data.shape[3] - self.target_size[1] + 1, (1,)).item(),
                )
                end = (
                    start[0] + self.target_size[0],
                    start[1] + self.target_size[1],
                )

                data = data[:, :, start[0]:end[0], start[1]:end[1]]

                if start[0] > 0:
                    metadata["Boundary Conditions"] = update_boundary_condition(metadata["Boundary Conditions"], "open",
                                                                                "x negative")
                if end[0] < data.shape[2]:
                    metadata["Boundary Conditions"] = update_boundary_condition(metadata["Boundary Conditions"], "open",
                                                                                "x positive")
                if start[1] > 0:
                    metadata["Boundary Conditions"] = update_boundary_condition(metadata["Boundary Conditions"], "open",
                                                                                "y negative")
                if end[1] < data.shape[3]:
                    metadata["Boundary Conditions"] = update_boundary_condition(metadata["Boundary Conditions"], "open",
                                                                                "y positive")

            else:
                raise NotImplementedError(
                    "Interpolating some dimensions while cropping others is currently not implemented.")

        # 3D spatial adjustment
        elif self.dimension == 3 and self.target_size is not None:
            # shape already matches
            if data.shape[2] == self.target_size[0] and data.shape[3] == self.target_size[1] and data.shape[4] == \
                    self.target_size[2]:
                pass

            # data too small -> trilinear interpolation
            elif data.shape[2] <= self.target_size[0] and data.shape[3] <= self.target_size[1] and data.shape[4] <= \
                    self.target_size[2]:

                data = torch.nn.functional.interpolate(data, size=(data.shape[1],) + self.target_size, mode="trilinear",
                                                       align_corners=False)

            # data too large -> random crop
            elif data.shape[2] >= self.target_size[0] and data.shape[3] >= self.target_size[1] and data.shape[4] >= \
                    self.target_size[2]:
                start = (
                    torch.randint(0, data.shape[2] - self.target_size[0] + 1, (1,)).item(),
                    torch.randint(0, data.shape[3] - self.target_size[1] + 1, (1,)).item(),
                    torch.randint(0, data.shape[4] - self.target_size[2] + 1, (1,)).item(),
                )
                end = (
                    start[0] + self.target_size[0],
                    start[1] + self.target_size[1],
                    start[2] + self.target_size[2],
                )

                data = data[:, :, start[0]:end[0], start[1]:end[1], start[2]:end[2]]

                if start[0] > 0:
                    metadata["Boundary Conditions"] = update_boundary_condition(metadata["Boundary Conditions"], "open",
                                                                                "x negative")
                if end[0] < data.shape[2]:
                    metadata["Boundary Conditions"] = update_boundary_condition(metadata["Boundary Conditions"], "open",
                                                                                "x positive")
                if start[1] > 0:
                    metadata["Boundary Conditions"] = update_boundary_condition(metadata["Boundary Conditions"], "open",
                                                                                "y negative")
                if end[1] < data.shape[3]:
                    metadata["Boundary Conditions"] = update_boundary_condition(metadata["Boundary Conditions"], "open",
                                                                                "y positive")
                if start[2] > 0:
                    metadata["Boundary Conditions"] = update_boundary_condition(metadata["Boundary Conditions"], "open",
                                                                                "z negative")
                if end[2] < data.shape[4]:
                    metadata["Boundary Conditions"] = update_boundary_condition(metadata["Boundary Conditions"], "open",
                                                                                "z positive")

            else:
                raise NotImplementedError(
                    "Interpolating some dimensions while cropping others is currently not implemented.")

        elif self.dimension not in [2, 3]:
            raise ValueError("Datasets with dimensionality %d are not supported." % self.dimension)

        # NOTE: We do NOT pad channels for conditioning since sel_channels_input 
        # can differ from sel_channels_target. The conditioning channels are 
        # determined by sel_channels_input and should not be modified here.

        # pad constants and constant metadata
        '''
        if constants.shape[0] == self.target_constants:
            pass
        elif constants.shape[0] < self.target_constants:
            pad = torch.zeros(self.target_constants - constants.shape[0])
            constants = torch.cat([constants, pad], 0)

            pad_norm = torch.zeros(self.target_constants - constants_norm.shape[0])
            constants_norm = torch.cat([constants_norm, pad_norm], 0)

            pad_constants = torch.zeros(self.target_constants - len(metadata["Constants"]))
            metadata["Constants"] = torch.cat([metadata["Constants"], pad_constants], 0)
        else:
            raise NotImplementedError("Reducing the number of constants is currently not implemented.")
        '''
        # convert metadata to correct data type -> not doing this gives errors for multiple workers
        metadata["Domain Extent"] = metadata["Domain Extent"].float()
        metadata["Dimension"] = metadata["Dimension"].long()
        metadata["PDE"] = metadata["PDE"].long()
        metadata["Fields"] = metadata["Fields"].long()
        metadata["Constants"] = metadata["Constants"].long()
        metadata["Boundary Conditions"] = metadata["Boundary Conditions"].long()

        loading_metadata["dataset_idx"] = torch.Tensor([loading_metadata["dataset_idx"]]).long()
        loading_metadata["unrolling_steps"] = torch.Tensor([loading_metadata["unrolling_steps"]]).long()

        if add_batch_dim:
            data = data.unsqueeze(0)
            constants = constants.unsqueeze(0)
            constants_norm = constants_norm.unsqueeze(0)
            time_step_stride = time_step_stride.unsqueeze(0)
            '''
            metadata["Domain Extent"] = metadata["Domain Extent"].unsqueeze(0)
            metadata["Dimension"] = metadata["Dimension"].unsqueeze(0)
            metadata["PDE"] = metadata["PDE"].unsqueeze(0)
            metadata["Fields"] = metadata["Fields"].unsqueeze(0)
            metadata["Constants"] = metadata["Constants"].unsqueeze(0)
            metadata["Boundary Conditions"] = metadata["Boundary Conditions"].unsqueeze(0)

            loading_metadata["dataset_idx"] = loading_metadata["dataset_idx"].unsqueeze(0)
            loading_metadata["unrolling_steps"] = loading_metadata["unrolling_steps"].unsqueeze(0)
            '''
        result = {
            "conditioning": data,
            "constants_norm": constants_norm,
            "constants": constants,
            "time_step_stride": time_step_stride,
            "physical_metadata": metadata,
            "loading_metadata": loading_metadata,
            "data": returnDictNoConditioning["data"],
        }
        
        # Add config and topology_config if they exist in the sample
        if "config" in sample:
            result["config"] = sample["config"]
        if "topology_config" in sample:
            result["topology_config"] = sample["topology_config"]
        
        return result