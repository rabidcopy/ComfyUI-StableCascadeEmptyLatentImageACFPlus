import torch
import nodes
import comfy.utils
import math

class SC_EmptyLatentImageACF_plus_min_768:
    def __init__(self, device="cpu"):
        self.device = device
        self.smallest_gap = float('inf')  # Initialize smallest_gap as an instance attribute

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 768, "min": 384, "max": 4096, "step": 32}),
            "height": ("INT", {"default": 768, "min": 384, "max": 4096, "step": 32}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})
        }}
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("stage_c", "stage_b")
    FUNCTION = "generate"

    CATEGORY = "latent/stable_cascade"

    def calc_compression_factor(self, width, height):
        final_compression_factor = None
        self.smallest_gap = float('inf')  # Initialize with a very large number

        for compression in range(168, 15, -1):
            res_se = min(width, height)
            res_le = max(width, height)
            aspect = res_le / res_se

            latent_min = res_se // compression
            latent_max = res_le // compression
            latent_div = (latent_max + latent_min) / 2

            new_center = self.remap(aspect, 1, 3.75, 24, 28.875)
            new_center = self.clamp(new_center, 24, 28.875)

            # Calculate the absolute difference between latent_div and new_center
            gap = abs(latent_div - new_center)

            # Update the smallest_gap and final_compression_factor accordingly
            if gap < self.smallest_gap:
                self.smallest_gap = gap
#                print(f"Compression: {compression}, Latent Div: {latent_div}, New Center: {new_center}, Smallest Gap: {self.smallest_gap}")
                final_compression_factor = compression

        if final_compression_factor is None:
            final_compression_factor = 32  # Set default compression factor to 32

        return final_compression_factor

    def remap(self, value, from1, to1, from2, to2):
        return (value - from1) / (to1 - from1) * (to2 - from2) + from2

    def clamp(self, value, min_value, max_value):
        return max(min_value, min(value, max_value))

    def round_half_up(self, value):
        return int(math.floor(value + 0.5))

    def generate(self, width, height, batch_size=1):
        compression = self.calc_compression_factor(width, height)
        if compression is None:
            raise ValueError("Unable to determine an appropriate compression factor.")

        print(f"Compression factor set to: {compression}, Smallest Gap was: {self.smallest_gap}")

        c_latent = torch.zeros([batch_size, 16, height // compression, width // compression])
        b_latent = torch.zeros([batch_size, 4, height // 4, width // 4])
        return ({
            "samples": c_latent,
        }, {
            "samples": b_latent,
        })

NODE_CLASS_MAPPINGS = {

    "SC_EmptyLatentImageACF_plus_min_768": SC_EmptyLatentImageACF_plus_min_768,
}
