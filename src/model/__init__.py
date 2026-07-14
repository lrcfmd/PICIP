from .picip_model import PicipModel

# Wire the concrete model for the runner. This is the single tool-owned wiring
# point — tools replace PicipModel with their own model class.
handler = PicipModel()