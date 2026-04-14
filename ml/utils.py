def clean_params(params):
    return {k.replace('model__', ''): v for k, v in params.items()}