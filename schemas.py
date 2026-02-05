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
    'cfg_scale': {
        'type': float,
        'required': False,
        'default': 1.0
    }
}
