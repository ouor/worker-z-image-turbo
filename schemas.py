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
        'default': 512 # Default width for Z-Image vertical preference or turbo speed
    },
    'seed': {
        'type': int,
        'required': False,
        'default': None
    },
    'num_inference_steps': {
        'type': int,
        'required': False,
        'default': 8 # Z-Image-Turbo typically needs fewer steps (e.g. 4-10)
    },
    'guidance_scale': {
        'type': float,
        'required': False,
        'default': 1.0 # Recommended CFG for Turbo models
    },
    'num_images': {
        'type': int,
        'required': False,
        'default': 1,
        'constraints': lambda img_count: 5 > img_count > 0
    }
}
