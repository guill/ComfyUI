from comfy_api.v0_0_2 import ComfyAPIAdapter_v0_0_2


# This version only exists to serve as a template for future version adapters.
# There is no reason anyone should ever use it.
class ComfyAPIAdapter_v0_0_1(ComfyAPIAdapter_v0_0_2):
    VERSION = "0.0.1"
    STABLE = True

ComfyAPI = ComfyAPIAdapter_v0_0_1
