from comet.models import load_from_checkpoint


def calculate_comet(sentence1, sentence2, device):
    model = load_from_checkpoint("wmt22-cometkiwi-da/checkpoints/model.ckpt").to(device)
    data = [{"src": src, "mt": mt} for src, mt in zip([sentence1], [sentence2])]
    return model.predict(data, batch_size=32, gpus=1)
