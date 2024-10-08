import tensorflow as tf

import sys
sys.path.extend(['./','../','../Models/','../Data_Loader/','../Custom_Functions/'])

def create_model(args, input_shape):
    if args.gpus <= 1:  # CPU or single GPU
        if args.net == "udet":
            from Models.UDet import UDet
            return [UDet(input_shape, args.activation == "mish")]
        elif args.net == "bifpn":
            from Models.Encoder_BiFPN import BIFPN  
            return [BIFPN(input_shape, args.activation == "mish")]
        elif args.net == "unet":
            from Models.UNet import UNet
            return [UNet(input_shape, args.activation == "mish")]
        elif args.net == "udet_small_sized":
            from Models.UDet_small_sized import UDet_small_sized
            return [UDet_small_sized(input_shape)]
        elif args.net == "udet_mixed":
            from Models.UDet import UDet
            from Models.UDet_small_sized import UDet_small_sized
            return [UDet(input_shape, args.activation == "mish"), UDet_small_sized(input_shape)]
        else:
            raise ValueError(f"Unknown network type: {args.net}")
    else:  # Multiple GPUs
        with tf.device("/cpu:0"):
            if args.net == "udet":
                from Models.UDet import UDet
                return [UDet(input_shape, args.activation == "mish")]  
            elif args.net == "unet":
                from Models.UNet import UNet  
                return [UNet(input_shape, args.activation == "mish")]
            elif args.net == "udet_small_sized":
                from Models.UDet_small_sized import UDet_small_sized
                return [UDet_small_sized(input_shape)]
            else:
                raise ValueError(f"Unknown network type: {args.net}")