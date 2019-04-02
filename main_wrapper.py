from main import parse_args

what = 'train'
dataset_path = '/home/laurent.lejeune/medical-labeling/VOC2012/'
out_path = '/home/laurent.lejeune/medical-labeling/gain/'
model_type = 'vgg16'
gradient_layer_name = 'features'
input_dims = '224'
input_channels = '3'

argv = [
    '{}'.format(what),
    '--dataset-path',
    dataset_path,
    '--model-type',
    model_type,
    '--gradient-layer-name',
    gradient_layer_name,
    '--omega',
    '10',
    '--input-dims',
    input_dims, input_dims,
    '--gpus',
    '0',
    '--pretrain-epochs',
    '-1',
    '--output-dir',
    out_path,
    '--input-channels',
    input_channels]

args = parse_args(argv)
args.func(args)
