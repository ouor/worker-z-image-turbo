INPUT_SCHEMA = {
    'prompt': {
        'type': str,
        'required': True,
    },
    'height': {
        'type': int,
        'required': False,
        'default': 1024
    },
    'width': {
        'type': int,
        'required': False,
        'default': 1024
    },
    'seed': {
        'type': int,
        'required': False,
        'default': 43
    },
    'num_inference_steps': {
        'type': int,
        'required': False,
        'default': 9
    },
    'guidance_scale': {
        'type': float,
        'required': False,
        'default': 1.0
    }
}
