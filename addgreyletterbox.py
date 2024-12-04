import torch

class AddGreyLetterbox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input single image or batch of images."}),
                "grey_value": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Grey level for the letterbox, default is 50%."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_letterbox"

    CATEGORY = "image/transform"

    def add_letterbox(self, images, grey_value=0.5):
        # Handle single image by adding batch dimension
        if len(images.shape) == 3:
            images = images.unsqueeze(0)

        batch, height, width, channels = images.shape
        max_dim = max(height, width)

        padded_images = []
        for img in images:
            # Determine padding
            padding_top = (max_dim - img.shape[0]) // 2
            padding_bottom = max_dim - img.shape[0] - padding_top
            padding_left = (max_dim - img.shape[1]) // 2
            padding_right = max_dim - img.shape[1] - padding_left

            # Add grey padding
            padded_img = torch.nn.functional.pad(
                img,
                (0, 0, padding_left, padding_right, padding_top, padding_bottom),
                mode="constant",
                value=grey_value,
            )
            padded_images.append(padded_img)

        # Stack into a batch
        return (torch.stack(padded_images),)

NODE_CLASS_MAPPINGS = {
        "Add Grey Letterbox": AddGreyLetterbox
}
