import os
import sys
import argparse
import numpy as np
import torch
from neural_style.transformer_net import *
from neural_style.utils import *
from torchvision import transforms
import torch.onnx


def transfer_style(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    content_image = load_image(args.content_image, scale=args.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)
    if args.model.endswith(".onnx"):
        output = stylize_onnx_caffe2(content_image, args)
    else:
        with torch.no_grad():
            style_model = TransformerNet()
            state_dict = torch.load(args.model)
            for k in list(state_dict.keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del state_dict[k]
            style_model.load_state_dict(state_dict)
            style_model.to(device)
            if args.export_onnx:
                assert args.export_onnx.endswith(".onnx"), "Export model file should end with .onnx"
                output = torch.onnx._export(style_model, content_image, args.export_onnx).cpu()
            else:
                output = style_model(content_image).cpu()
    save_image(args.output_image, output[0])

def stylize_onnx_caffe2(content_image, args):
    assert not args.export_onnx
    import onnx
    import onnx_caffe2.backend
    model = onnx.load(args.model)
    prepared_backend = onnx_caffe2.backend.prepare(model, device='CUDA' if args.cuda else 'CPU')
    inp = {model.graph.input[0].name: content_image.numpy()}
    c2_out = prepared_backend.run(inp)[0]
    return torch.from_numpy(c2_out)

def get_routine_args():
    routine_args = argparse.ArgumentParser(description="parser for the style transfer task")
    subroutine_parser = routine_args.add_subparsers(title="subcommands", dest="subcommand")
    style_nodel_parser = subroutine_parser.add_parser("model", help="choice of model(1 of : candy.ph, mosaic.ph, rain_princess.ph, udnie.ph)")
    style_nodel_parser.add_argument("--log-path", help="path to log dir",type=str, required=True)
    style_nodel_parser.add_argument("--segmentation", help="segment the content image", type=bool, default=False)
    style_nodel_parser.add_argument("--content-image", help="path of the content image", type=str, required=True)
    style_nodel_parser.add_argument("--cuda", help="set it to 1 for running on GPU, 0 for CPU", type=bool, required=True)
    return style_nodel_parser.parse_args()


if __name__ == "__main__":
    args = get_routine_args()
    args = vars(args)
    transfer_style(args)