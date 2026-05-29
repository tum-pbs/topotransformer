from pdetransformer.data.pbdl_dataloader.dataset import Dataset as PBDLDataset

from pdetransformer.data.pbdl_dataloader.utilities import get_sel_const_sim, get_meta_data, scan_local_dset_dir, get_sel_const_sim_v2
from pdetransformer.data.pbdl_dataloader.logging import info, success, warn, fail, corrupt
import sys
import numpy as np
class Dataset(PBDLDataset):
    def __init__(
        self,
        dset_name,
        time_steps=None,  # yb default num_frames - 1
        all_time_steps=False,  # sets time_steps=max time steps, intermediate_time_steps=True
        intermediate_time_steps=None,  # by default False
        normalize_data=None,  # by default no normalization
        normalize_const=None,  # by default no normalization
        sel_sims=None,  # if None, all simulations are loaded
        sel_const=None,  # if None, all constants are returned
        trim_start=None,  # by default 0
        trim_end=None,  # by default 0
        step_size=None,  # by default 1
        disable_progress=False,
        crop_size=None,
        seed=0,
        clear_norm_data=False,
        sel_channels_input=None, # if None, all channels are returned
        sel_channels_target=None, # if None, all channels are returned
        normalize_target=True,  # if False, target will not be normalized
        normalize_channels=None,  # if None, normalize all channels; if list, only normalize those channels
        **kwargs,
    ):
        

        super().__init__(
            dset_name,
            time_steps,
            all_time_steps,
            intermediate_time_steps,
            normalize_data,
            normalize_const,
            sel_sims,
            sel_const,
            None,
            trim_start,
            trim_end,
            step_size,
            disable_progress,
            crop_size,
            seed,
            clear_norm_data,
            **kwargs
        )

        self.sel_channels_input = sel_channels_input
        self.sel_channels_target = sel_channels_target
        self.normalize_target = normalize_target
        self.normalize_channels = normalize_channels
        
        # Compute samples per sim for each simulation (variable timesteps support)
        self._compute_samples_per_sim()

    def _compute_samples_per_sim(self):
        """Compute number of samples for each simulation based on its timesteps."""
        self.sim_timesteps = []  # Store timesteps for each sim
        self.sim_samples = []  # Store number of samples for each sim
        self.cumulative_samples = [0]  # Cumulative sum for indexing
        
        sims_to_iterate = self.sel_sims if self.sel_sims else range(self.num_sims)
        
        for sim_idx in sims_to_iterate:
            sim = self.dset["sims/sim" + str(sim_idx)]
            num_timesteps = sim.shape[0]
            self.sim_timesteps.append(num_timesteps)
            
            # Calculate samples for this sim based on its timesteps
            # available_steps is the number of frames we can use after trimming
            available_steps = num_timesteps - self.trim_start - self.trim_end
            
            if self.intermediate_time_steps:
                # When intermediate_time_steps=True, we need:
                # - input frame at position i
                # - target frames from i+1 to i+time_steps (inclusive)
                # So we need at least (time_steps + 1) frames
                # The last valid starting position is: available_steps - time_steps - 1
                samples = max(0, (available_steps - self.time_steps - 1) // self.step_size + 1)
            else:
                # When intermediate_time_steps=False, we only need:
                # - input frame at position i
                # - target frame at same position i (or i+time_steps if time_steps>0)
                # For single-frame prediction: available_steps frames gives us available_steps samples
                if self.time_steps > 0:
                    samples = max(0, (available_steps - self.time_steps) // self.step_size + 1)
                else:
                    samples = max(0, available_steps // self.step_size)
            
            self.sim_samples.append(samples)
            self.cumulative_samples.append(self.cumulative_samples[-1] + samples)
    
    def __len__(self):
        """Return total number of samples across all simulations."""
        return self.cumulative_samples[-1]

    def _normalize_selective_channels(self, data):
        """
        Normalize only the channels specified in self.normalize_channels.
        Other channels are left unchanged (zero mean, unit std - i.e., no normalization).
        
        Args:
            data: numpy array with shape (channels, spatial dims...)
        
        Returns:
            numpy array with selective normalization applied
        """
        result = data.copy()
        for ch_idx in self.normalize_channels:
            if ch_idx < data.shape[0]:
                # Normalize only this channel using the normalization strategy
                # We need to normalize channel-by-channel since norm_strat normalizes all channels
                ch_data = data[ch_idx:ch_idx+1]  # Keep dimension for broadcasting
                result[ch_idx:ch_idx+1] = self.norm_strat_data.normalize(ch_data)
        return result

    def _validate_dataset(self):
        # basic validation checks on shape
        if len(self.sim_shape) < 3:
            corrupt(
                "Simulations data must have shape (frames, fields, spatial dim [...])."
            )
            sys.exit(0)

        if len(self.fields_scheme) != self.sim_shape[1]:
            raise ValueError(
                f"Inconsistent number of fields between metadata ({len(self.fields_scheme) }) and simulations ({ self.sim_shape[1]})."
            )

        for sim in self.dset["sims/"]:
            sim_data = self.dset["sims/" + sim]
            
            # shape must be consistent through all sims (except time dimension)
            # Check that all dimensions except time (index 0) match
            if sim_data.shape[1:] != self.sim_shape[1:]:
                corrupt(
                    f"The shape of all simulations must be consistent (excluding time): Shape of first sim {self.sim_shape[1:]} and sim {sim} {sim_data.shape[1:]} do not match)."
                )
                sys.exit(0)

            # all sims must define the declared constants
            missing = set(self.const) - set(sim_data.attrs.keys())
            if missing:
                corrupt(
                    f"Simulation {sim} does not define all declared constants: {missing}."
                )
                sys.exit(0)

    def __getitem__(self, idx):
        """
        The data provided has the shape (channels, spatial dims...).

        Returns:
            numpy.ndarray: Input data (without constants)
            tuple: Constants
            numpy.ndarray: Target data
            tuple: Non-normalized constants (only if solver flag is set)
        """

        #input, target, const, const_nnorm = super().__getitem__(idx)
        if idx >= len(self):
            raise IndexError

        # Find which simulation this index belongs to using cumulative samples
        sim_list_idx = 0
        for i, cumsum in enumerate(self.cumulative_samples[1:]):
            if idx < cumsum:
                sim_list_idx = i
                break
        
        # Get actual simulation index
        if self.sel_sims:
            sim_idx = self.sel_sims[sim_list_idx]
        else:
            sim_idx = sim_list_idx
        
        # Get the sample index within this specific simulation
        idx_within_sim = idx - self.cumulative_samples[sim_list_idx]

        sim = self.dset["sims/sim" + str(sim_idx)]
        const = get_sel_const_sim(self.dset, sim_idx, self.sel_const)

        input_frame_idx = (
                self.trim_start + idx_within_sim * self.step_size
        )
        target_frame_idx = input_frame_idx + self.time_steps

        dim_list = sim.shape[2:]

        if self.crop_size is None:

            input = sim[input_frame_idx]

        else:

            crop_dim_list = [self.rng.integers(low=0, high=dim-self.crop_size, size=1)[0] for dim in dim_list]
        
            # 2D
            if len(dim_list) == 2:
                input = sim[input_frame_idx, :, crop_dim_list[0]:crop_dim_list[0]+self.crop_size,
                                                crop_dim_list[1]:crop_dim_list[1]+self.crop_size]
            # 3D
            elif len(dim_list) == 3:
                input = sim[input_frame_idx, :, crop_dim_list[0]:crop_dim_list[0]+self.crop_size,
                                                crop_dim_list[1]:crop_dim_list[1]+self.crop_size,
                                                crop_dim_list[2]:crop_dim_list[2]+self.crop_size]
            else:
                raise ValueError(f'Dimension {self.sim_shape} not supported')

        if self.intermediate_time_steps:

            if self.crop_size is None:

                target = sim[-1:]

            else:

                if len(dim_list) == 2:
                    target = sim[input_frame_idx + 1: target_frame_idx + 1, :, crop_dim_list[0]:crop_dim_list[0]+self.crop_size,
                                                                               crop_dim_list[1]:crop_dim_list[1]+self.crop_size]

                elif len(dim_list) == 3:
                    target = sim[input_frame_idx + 1: target_frame_idx + 1, :, crop_dim_list[0]:crop_dim_list[0]+self.crop_size,
                                                                               crop_dim_list[1]:crop_dim_list[1]+self.crop_size,
                                                                               crop_dim_list[2]:crop_dim_list[2]+self.crop_size]
        
        else:

            if self.crop_size is None:

                target = sim[input_frame_idx]

            else:

                if len(dim_list) == 2:
                    target = sim[input_frame_idx, :, crop_dim_list[0]:crop_dim_list[0]+self.crop_size,
                                                                               crop_dim_list[1]:crop_dim_list[1]+self.crop_size]

                elif len(dim_list) == 3:
                    target = sim[input_frame_idx, :, crop_dim_list[0]:crop_dim_list[0]+self.crop_size,
                                                                               crop_dim_list[1]:crop_dim_list[1]+self.crop_size,
                                                                               crop_dim_list[2]:crop_dim_list[2]+self.crop_size]

        const_nnorm = const

        # normalize
        if self.norm_strat_data:
            if self.normalize_channels is None:
                # Normalize all channels with the normalization strategy
                input = self.norm_strat_data.normalize(input)
            else:
                # Selective channel normalization: only normalize specified channels
                input = self._normalize_selective_channels(input)

            if self.normalize_target:
                if self.normalize_channels is None:
                    if self.intermediate_time_steps:
                        target = np.array(
                            [self.norm_strat_data.normalize(frame) for frame in target]
                        )
                    else:
                        target = self.norm_strat_data.normalize(target)
                else:
                    # Selective channel normalization for target
                    if self.intermediate_time_steps:
                        target = np.array(
                            [self._normalize_selective_channels(frame) for frame in target]
                        )
                    else:
                        target = self._normalize_selective_channels(target)

        if self.norm_strat_const:
            const = self.norm_strat_const.normalize(const)

        if self.sel_channels is not None:
            input = input[self.sel_channels]
            if self.intermediate_time_steps:
                target = target[:, self.sel_channels]
            else:
                target = target[self.sel_channels]


        if self.sel_channels_input is not None:
            input = input[self.sel_channels_input]
        if self.sel_channels_target is not None:
            if self.intermediate_time_steps:
                target = target[:, self.sel_channels_target]
            else:
                target = target[self.sel_channels_target]

        return (
            input, #(input+0.01)*1.5,#[..., :1, :, :],
            target,#[..., -2:-1, :, :],
            const,  # required by loader
            const_nnorm,  # needed by pbdl.torch.phi.loader
        )

