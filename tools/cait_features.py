import tools.cait_models_attn as cait_models_attn
from timm.models import create_model

def cait_xxs24_224_features(pretrained=False, nb_classes=1000, drop=0.0, drop_path=0.1, **kwargs):
    model_name = 'cait_xxs24_224'
    model = create_model(
            model_name,
            pretrained=pretrained,
            num_classes=nb_classes,
            drop_rate=drop,
            drop_path_rate=drop_path,
        )

    return model

def cait_s24_224_features(pretrained=False, nb_classes=1000, drop=0.0, drop_path=0.1, **kwargs):
    model_name = 'cait_s24_224'
    model = create_model(
            model_name,
            pretrained=pretrained,
            num_classes=nb_classes,
            drop_rate=drop,
            drop_path_rate=drop_path,
        )

    return model