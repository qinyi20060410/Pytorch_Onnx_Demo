'''
Author: your name
Date: 2021-03-16 17:36:59
LastEditTime: 2021-03-16 19:41:49
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Demo/export_onn.py
'''

import sys
import torch  # Must import torch before C extension
from torch.onnx import register_custom_op_symbolic
from Mish import mish_forward, mish_backward
import json
import onnxruntime


class MishCudaFunction(torch.autograd.Function):
    @staticmethod
    def symbolic(g, inp):
        return g.op("plugin::Mish", inp, name_s="Mish")

    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        return mish_forward(inp)

    @staticmethod
    def backward(ctx, grad_out):
        inp, = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return (None, )
        return mish_backward(inp, grad_out)


register_custom_op_symbolic("plugin::mish_plugin", MishCudaFunction.symbolic,
                            11)


class MishCuda(torch.nn.Module):
    def forward(self, inp):
        return MishCudaFunction.apply(inp)


class TestModel(torch.nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2))
        self.mish = MishCuda()
        self.conv2 = torch.nn.Conv2d(64,
                                     128,
                                     kernel_size=(3, 3),
                                     stride=(2, 2))
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.mish(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


def transform_to_onnx(batch_size=1, image_h=600, image_w=600):
    model = TestModel()

    input_names = ["input"]
    output_names = ['output']

    dynamic = False
    if batch_size <= 0:
        dynamic = True
        if dynamic:
            x = torch.randn((1, 3, image_h, image_w), requires_grad=True)
        onnx_file_name = "test-1_3_{}_{}_dynamic.onnx".format(image_h, image_w)
        dynamic_axes = {
            "input": {
                0: "batch_size"
            },
            "classes": {
                0: "batch_size"
            }
        }
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names,
                          output_names=output_names,
                          dynamic_axes=dynamic_axes)

        print('Onnx model exporting done')
        return onnx_file_name

    else:
        x = torch.randn((batch_size, 3, image_h, image_w), requires_grad=True)
        onnx_file_name = "test_{}_3_{}_{}_static.onnx".format(
            batch_size, image_h, image_w)
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names,
                          output_names=output_names,
                          dynamic_axes=None)

        print('Onnx model exporting done')
        return onnx_file_name


def main(batch_size, IN_IMAGE_H, IN_IMAGE_W):

    if batch_size <= 0:
        onnx_path_demo = transform_to_onnx(batch_size, IN_IMAGE_H, IN_IMAGE_W)
    else:
        # Transform to onnx as specified batch size
        transform_to_onnx(batch_size, IN_IMAGE_H, IN_IMAGE_W)
        # Transform to onnx for demo
        onnx_path_demo = transform_to_onnx(1, IN_IMAGE_H, IN_IMAGE_W)

    # session = onnxruntime.InferenceSession(onnx_path_demo)
    # # session = onnx.load(onnx_path)
    # print("The model expects input shape: ", session.get_inputs()[0].shape)


if __name__ == '__main__':
    print("Converting to onnx and running demo ...")
    if len(sys.argv) == 4:

        batch_size = int(sys.argv[1])
        IN_IMAGE_H = int(sys.argv[2])
        IN_IMAGE_W = int(sys.argv[3])

        main(batch_size, IN_IMAGE_H, IN_IMAGE_W)
    else:
        print('Please run this way:\n')
        print(
            '  python demo_onnx.py <weight_file>  <batch_size> <n_classes> <IN_IMAGE_H> <IN_IMAGE_W>'
        )
