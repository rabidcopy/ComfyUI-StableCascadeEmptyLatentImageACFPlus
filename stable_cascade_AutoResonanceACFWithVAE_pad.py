import torch
import comfy.utils
import math

class AutoResonanceAdvancedACF:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 32}),
            "height": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 32}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            "offset": ("INT", {"default": 0, "min": -16, "max": 16}),
            "pad_shortest_to_32": ("BOOLEAN", {"default": False}),
            "target_mean": ("BOOLEAN", {"default": False}),
            "mean": ("FLOAT", {"default": 32, "min": 1, "max": 64, "step": 0.5}),
        }, "optional": {
            "image": ("IMAGE", {}),
            "vae": ("VAE", {})
        }}
    
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("stage_c", "stage_b")
    FUNCTION = "generate"

    CATEGORY = "latent/stable_cascade"

    def calc_compression_factor(self, width, height, target_mean=False, mean=32):
        final_compression_factor = None
        self.smallest_gap = float('inf')  # Initialize with a very large number

        for compression in range(128, 15, -1):
            res_se = min(width, height)
            res_le = max(width, height)
            aspect = res_le / res_se

            latent_min = res_se // compression
            latent_max = res_le // compression
            latent_div = (latent_max + latent_min) / 2

            new_center = self.remap(aspect, 1, 3.75, 32, 38.5)
            new_center = self.clamp(new_center, 32, 38.5)

            # Calculate the absolute difference between latent_div and new_center
            if target_mean is True:
                gap = abs(latent_div - mean)
            elif target_mean is False:
                gap = abs(latent_div - new_center)

            # Update the smallest_gap and final_compression_factor accordingly
            if gap < self.smallest_gap:
                self.smallest_gap = gap
#                print(f"Compression: {compression}, Latent Div: {latent_div}, New Center: {new_center}, Smallest Gap: {self.smallest_gap}")
                final_compression_factor = compression
        if final_compression_factor >= 81:
            print(f"Warning! Compression factors over 80 are likely to not work when the latent is passed to Stage B. Consider a lower resolution or using Img2Img at 32 compression for higher resolutions.")

        if final_compression_factor is None:
            final_compression_factor = 32  # Set default compression factor to 32

        return final_compression_factor

    def remap(self, value, from1, to1, from2, to2):
        return (value - from1) / (to1 - from1) * (to2 - from2) + from2

    def clamp(self, value, min_value, max_value):
        return max(min_value, min(value, max_value))

    def generate(self, width, height, offset, batch_size=1, image=None, vae=None, pad_shortest_to_32=False, target_mean=False, mean=32):

        if image is not None and vae is not None:
            # Get the dimensions of the input image
            image_width = image.shape[-2]
            image_height = image.shape[-3]
            
            compression = self.calc_compression_factor(image_width, image_height, target_mean, mean)
            if compression is None:
                raise ValueError("Unable to determine an appropriate compression factor.")

            print(f"Compression factor set to: {compression}, Smallest Gap was: {self.smallest_gap}")
                        
            # Determine latent size from compression
            c_width = (image_width // compression) + offset
            c_height = (image_height // compression) + offset

            # If target_mean is True, adjust c_width and c_height
            if target_mean:
                # Calculate the desired total dimension
                target_total = mean * 2

                # Compute the current total dimension
                current_total = c_width + c_height

                # Calculate the scaling factor to achieve the target total dimension
                scale_factor = target_total / current_total

                # Adjust c_width and c_height based on the scaling factor
                c_width = int(c_width * scale_factor)
                c_height = int(c_height * scale_factor)

                # Ensure the sum of c_width and c_height is exactly target_total
                if c_width + c_height != target_total:
                    difference = target_total - (c_width + c_height)
                    # Adjust the larger dimension to account for rounding differences
                    if c_width > c_height:
                        c_width = int(c_width + difference)
                    else:
                        c_height = int(c_height + difference)

                print(f"Scaling factor is {scale_factor}, adjusted dimensions to total of {target_total}")
            
            shortest_edge = min(c_width, c_height)
            if shortest_edge < 32 and pad_shortest_to_32:
                padding_factor = (32 / shortest_edge)
                c_width = int(c_width * padding_factor)
                c_height = int(c_height * padding_factor)
                print(f"Padding factor is {padding_factor}, padding shortest edge to 32")

            print(f"Stage C latent dimensions set to: {c_width}x{c_height}")

            # Resize the image to match the best matching latent size using comfy.utils
            image_tensor = image.movedim(-1, 1)  # Move the channel dimension
            resized_image = comfy.utils.common_upscale(image_tensor, c_width * vae.downscale_ratio, c_height * vae.downscale_ratio, "bicubic", "center").movedim(1, -1)

            # Encode the image using VAE
            c_latent = vae.encode(resized_image[:, :, :, :3])

            # Calculate means of user-configured dimensions and the matched latent size
            input_dimension_mean = (width + height) / 2
            c_dimension_mean = (c_width + c_height) / 2

            # Calculate factor to multiply the matched latent by
            upscale_factor = input_dimension_mean / c_dimension_mean

            # Check if the calculated b_width and b_height match the user-configured width and height
            if image_width == width and image_height == height:
                b_width = image_width // 4
                b_height = image_height // 4

            else:
                # Make multiple of 32
                def round_to_multiple(value, multiple):
                    return int(math.ceil(value / multiple) * multiple)

                b_width = round_to_multiple(c_width * upscale_factor, 32) // 4
                b_height = round_to_multiple(c_height * upscale_factor, 32) // 4

        else:

            compression = self.calc_compression_factor(width, height, target_mean, mean)
            if compression is None:
                raise ValueError("Unable to determine an appropriate compression factor.")

            print(f"Compression factor set to: {compression}, Smallest Gap was: {self.smallest_gap}")
                    
            # Calculate aspect ratio of the input dimensions
#            input_aspect_ratio = width / height

            # Use the dimensions of the best matching latent size
            c_width = (width // compression) + offset
            c_height = (height // compression) + offset

            # If target_mean is True, adjust c_width and c_height
            if target_mean:
                # Calculate the desired total dimension
                target_total = mean * 2

                # Compute the current total dimension
                current_total = c_width + c_height

                # Calculate the scaling factor to achieve the target total dimension
                scale_factor = target_total / current_total

                # Adjust c_width and c_height based on the scaling factor
                c_width = int(c_width * scale_factor)
                c_height = int(c_height * scale_factor)

                # Ensure the sum of c_width and c_height is exactly target_total
                if c_width + c_height != target_total:
                    difference = target_total - (c_width + c_height)
                    # Adjust the larger dimension to account for rounding differences
                    if c_width > c_height:
                        c_width = int(c_width + difference)
                    else:
                        c_height = int(c_height + difference)

                print(f"Scaling factor is {scale_factor}, adjusted dimensions to total of {target_total}")

            shortest_edge = min(c_width, c_height)
            if shortest_edge < 32 and pad_shortest_to_32:
                padding_factor = (32 / shortest_edge)
                c_width = int(c_width * padding_factor)
                c_height = int(c_height * padding_factor)
                print(f"Padding factor is {padding_factor}, padding shortest edge to 32")

            print(f"Stage C latent dimensions set to: {c_width}x{c_height}")

            c_latent = torch.zeros([batch_size, 16, c_height, c_width])

            b_width = width // 4
            b_height = height // 4

        print(f"Stage B latent dimensions set to: {b_width}x{b_height}")

        b_latent = torch.zeros([batch_size, 4, b_height, b_width])
        
        return ({
            "samples": c_latent,
        }, {
            "samples": b_latent,
        })


NODE_CLASS_MAPPINGS = {
    "AutoResonanceAdvancedACF": AutoResonanceAdvancedACF,
}
