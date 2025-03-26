from medseg.evaluation.medcam import medcam_inject
from medseg.evaluation.medcam import medcam_utils
from functools import wraps


@wraps(medcam_inject.inject)
def inject(*args, **kwargs):
    return medcam_inject.inject(*args, **kwargs)


@wraps(medcam_utils.get_layers)
def get_layers(model, reverse=False):
    return medcam_utils.get_layers(model, reverse)




@wraps(medcam_utils.save_attention_map)
def save(attention_map, filename, heatmap, raw_input=None):
    medcam_utils.save_attention_map(filename, attention_map, heatmap=heatmap, raw_input=raw_input)
