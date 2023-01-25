import argparse
import torch

from models import YoLoss
from models import YoloEmb
from models import MobileNetV2Emb


def load_model(path: str):
    """Loading pretrained weights
    """
    model = YoloEmb(MobileNetV2Emb(), YoLoss())

    model.load_state_dict(torch.load(path))
    model.to('cpu')
    model.eval()

    return model


def convert_to_onnx(model: YoloEmb, onnx_save_path: str = 'pretrained/Detector.onnx'):
    """Convert loaded model to .onnx
    """

    torch.onnx.export(model, torch.zeros(1, 3, 640, 640).to('cpu'), onnx_save_path, training=torch.onnx.TrainingMode.EVAL,
                      do_constant_folding=False, input_names=['images'], output_names=['output'], dynamic_axes=None)


def convert_to_torchscript(model: YoloEmb, pt_save_path: str = "pretrained/Detector.pt"):
    """Convert loaded model to .pth
    """

    scripted_model = model.to_torchscript()
    torch.jit.save(scripted_model, pt_save_path)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, default="False",
                        help="convert to onnx")
    parser.add_argument('--weight_path', type=str, default="./pretrained/Detector.pth",
                        help="pretrained model weight's path")
    parser.add_argument('--save_path', type=str, default="./pretrained/",
                        help="saving directory")

    return parser.parse_known_args()[0]


def main(opt):
    model = load_model(opt.weight_path)

    if opt.onnx != "False":
        saving_path = opt.save_path + "Detector.onnx"

        convert_to_onnx(model, saving_path)
        print(f"Model to onnx converted: destination is {saving_path}")
    else:
        saving_path = opt.save_path + "Detector.pt"

        convert_to_torchscript(model, saving_path)
        print(f"Model to pt converted: destination is {saving_path}")


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
