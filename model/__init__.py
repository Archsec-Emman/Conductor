from .conductor import ConductorTokenizer, Conductor, ConductorPredictor

model_dict = {
    'conductor_tokenizer': ConductorTokenizer,
    'conductor': Conductor,
    'conductor_predictor': ConductorPredictor
}


def get_model_class(model_name):
    if model_name in model_dict:
        return model_dict[model_name]
    else:
        print(f"Model {model_name} not found in model_dict")
        raise NotImplementedError


