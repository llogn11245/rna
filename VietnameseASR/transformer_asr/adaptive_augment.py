from speechbrain.lobes.augment import SpecAugment
import torch

class AdaptiveSpecAugment(SpecAugment):
    def __init__(self, *args, max_n_mask = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_n_mask = max_n_mask
    def forward(self, x, l):
        """Takes in input a tensors and returns an augmented one."""
        if self.apply_time_warp:
            x = self.time_warp(x, l)
        if self.freq_mask:
            x = self.mask_along_axis(x, dim=2, l = l)
        if self.time_mask:
            x = self.mask_along_axis(x, dim=1, l = l)
        return x

    def time_warp(self, x, l):
        """Time warping with torch.nn.functional.interpolate"""
        original_size = x.shape
        window = self.time_warp_window

        # 2d interpolation requires 4D or higher dimension tensors
        # x: (Batch, Time, Freq) -> (Batch, 1, Time, Freq)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        time = x.shape[2]
        if time - window <= window:
            return x.view(*original_size)

        # compute center and corresponding window
        c = torch.randint(window, time - window, (1,))[0]
        w = torch.randint(c - window, c + window, (1,))[0] + 1

        left = torch.nn.functional.interpolate(
            x[:, :, :c],
            (w, x.shape[3]),
            mode=self.time_warp_mode,
            align_corners=True,
        )
        right = torch.nn.functional.interpolate(
            x[:, :, c:],
            (time - w, x.shape[3]),
            mode=self.time_warp_mode,
            align_corners=True,
        )

        x[:, :, :w] = left
        x[:, :, w:] = right

        return x.view(*original_size)

    def mask_along_axis(self, x, dim, l):
        original_size = x.shape
        if x.dim() == 4:
            x = x.view(-1, x.shape[2], x.shape[3])

        batch, time, fea = x.shape

        if dim == 1:
            D = time
            n_mask = self.n_time_mask
            width_range = self.time_mask_width
        else:
            D = fea
            n_mask = self.n_freq_mask
            width_range = self.freq_mask_width

        if self.replace_with_zero:
            val = 0.0
        else:
            with torch.no_grad():
                val = x.mean()

        if isinstance(width_range, float):
            # mask_len = torch.zeros(batch, n_mask).to(x.device)
            # mask_pos = torch.zeros(batch, n_mask).to(x.device)
            mask_len = []
            mask_pos = []
            arange = torch.arange(D, device=x.device).view(1, -1)

            for i in range(batch):
                actual_time = int(torch.round(D * l[i]))

                n_mask = min(self.max_n_mask, round(width_range*actual_time))

                mask_len = torch.randint(0, round(width_range*actual_time), (1, n_mask)).unsqueeze(2).squeeze(0).to(x.device)
                mask_pos = torch.randint(0, int(max(1, actual_time - mask_len.max())), (1, n_mask)).unsqueeze(2).squeeze(0).to(x.device)

                mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))

                mask = mask.any(0)

                if dim == 1:
                    mask = mask.unsqueeze(1)
                else:
                    mask = mask.unsqueeze(0)

                x[i] = x[i].masked_fill_(mask, val)
        else: 
            mask_len = torch.randint(
                width_range[0], width_range[1], (batch, n_mask), device=x.device
            ).unsqueeze(2)

            mask_pos = torch.randint(
                0, max(1, D - mask_len.max()), (batch, n_mask), device=x.device
            ).unsqueeze(2)

            # compute masks
            arange = torch.arange(D, device=x.device).view(1, 1, -1)
            mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
            mask = mask.any(dim=1)

            if dim == 1:
                mask = mask.unsqueeze(2)
            else:
                mask = mask.unsqueeze(1)

            x = x.masked_fill_(mask, val)

        return x.view(*original_size)