import torch
import comfy.utils
import math

class AutoResonanceAdvanced:
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

    PRESET_LATENT_SIZES = [
        (61, 16), (60, 16), (59, 17), (58, 17), (57, 17), (56, 18), (55, 18), (54, 18), 
        (53, 19), (52, 19), (51, 19), (50, 20), (49, 20), (48, 20), (48, 21), (47, 21), 
        (46, 21), (46, 22), (45, 22), (44, 22), (44, 23), (43, 23), (42, 23), (42, 24), 
        (41, 24), (40, 24), (40, 25), (39, 25), (39, 26), (38, 26), (37, 26), (37, 27), 
        (36, 27), (36, 28), (35, 28), (35, 29), (34, 29), (34, 30), (33, 30), (33, 31), 
        (32, 31), (32, 32), (31, 32), (30, 33), (30, 34), (29, 34), (29, 35), (28, 35), 
        (28, 36), (27, 36), (27, 37), (26, 37), (26, 38), (26, 39), (25, 39), (25, 40), 
        (24, 40), (24, 41), (24, 42), (23, 42), (23, 43), (23, 44), (22, 44), (22, 45), 
        (22, 46), (21, 46), (21, 47), (21, 48), (20, 48), (20, 49), (20, 50), (20, 51), 
        (19, 51), (19, 52), (19, 53), (18, 53), (18, 54), (18, 55), (18, 56), (17, 56), 
        (17, 57), (17, 58), (17, 59), (17, 60), (16, 60), (16, 61)
    ]

    def generate(self, width, height, offset, batch_size=1, image=None, vae=None, pad_shortest_to_32=False, target_mean=False, mean=32):

        if image is not None and vae is not None:
            # Get the dimensions of the input image
            image_width = image.shape[-2]
            image_height = image.shape[-3]
            input_aspect_ratio = image_width / image_height
            
            # Find the best matching latent size based on aspect ratio
            best_match = min(self.PRESET_LATENT_SIZES, key=lambda size: abs((size[0] / size[1]) - input_aspect_ratio))
            
            # Use the dimensions of the best matching latent size
            c_width = best_match[0] + offset
            c_height = best_match[1] + offset

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


#            # If target_mean is True, adjust c_width and c_height
#            if target_mean:
#                c_dimension_mean = (c_width + c_height) / 2
#                scale_factor = mean / c_dimension_mean
#                c_width = int(c_width * scale_factor)
#                c_height = int(c_height * scale_factor)
#                print(f"Scaling factor is {scale_factor}, adjusted dimensions to mean of {mean}")

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
            # Calculate aspect ratio of the input dimensions
            input_aspect_ratio = width / height

            # Find the best matching latent size based on aspect ratio
            best_match = min(self.PRESET_LATENT_SIZES, key=lambda size: abs((size[0] / size[1]) - input_aspect_ratio))

            # Use the dimensions of the best matching latent size
            c_width = best_match[0] + offset
            c_height = best_match[1] + offset

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


#            # If target_mean is True, adjust c_width and c_height
#            if target_mean:
#                c_dimension_mean = (c_width + c_height) / 2
#                scale_factor = mean / c_dimension_mean
#                c_width = int(c_width * scale_factor)
#                c_height = int(c_height * scale_factor)
#                print(f"Scaling factor is {scale_factor}, adjusted dimensions to mean of {mean}")

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
    "AutoResonanceAdvanced": AutoResonanceAdvanced,
}
