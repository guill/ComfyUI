from comfy_api.latest import ComfyAPI_latest

class ComfyAPIAdapter_v0_0_2(ComfyAPI_latest):
    VERSION = "0.0.2"
    STABLE = False

ComfyAPI = ComfyAPIAdapter_v0_0_2
