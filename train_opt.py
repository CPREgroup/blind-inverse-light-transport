import socket
from argparse import ArgumentParser

def args_myParser():
    parser = ArgumentParser(description='FullCNN-Net')
    machine_name = socket.gethostname()
    parser.add_argument('--flag',
                        # required=True,
                        default='fake_and_real_peppers_ms',
                        help="flag of log, or dataset img name for not forget")
    parser.add_argument('--dataset', type=str,default='dots', help='Name of the dataset')
    parser.add_argument('-d', '--data_dir', type=str,
                        default='./ablation/data/light_transport', help='Location for data directory')
    parser.add_argument('-f', '--folder', type=str,
                        default='last_final', help='Name of the folder')
    parser.add_argument('-s', '--seq_name', type=str,
                        default='disc', help='Name of the sequence')
    parser.add_argument('-o', '--out_dir', type=str,
                        default='output_light_transport', help='Location for output directory')
    parser.add_argument('-sf', '--skip_frames', type=int,
                        default=0, help='How many frames to skip (fast forward)')
    parser.add_argument('-dev', '--device', type=int,
                        default=0, help='GPU ID')
    parser.add_argument('--nvec', '--num_singular_vectors', type=int,
                        default=32, help='Number of singular vectors')
    parser.add_argument('-n_ds', '--num_downsample', type=int,
                        default=3, help='Num downsamples')
    parser.add_argument('--mis', type=str, default="unixy", help="reminder")
    parser.add_argument('--nerf', type=str, default="sin", help="reminder")
    parser.add_argument('--gpuind', type=str, default="0", help="gpu for train")

    parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
    parser.add_argument('--end_epoch', type=int, default=50000, help='epoch number of end training')
    parser.add_argument('--layer_num', type=int, default=4, help='phase number of ResCNN-Net')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=1, help='batchsize')
    parser.add_argument('--rgb_wei', type=float, default=1, help='ryb loss weight')

    parser.add_argument('--model_dir', type=str, default='ablation/resblock', help='trained or pre-trained model directory')
    parser.add_argument('--log_dir', type=str, default='ablation/resblock', help='no need')
    parser.add_argument('--L', type=int, default=64, help='position encoding')
    parser.add_argument('--eta', type=float, default=1.0, help='weight')
    parser.add_argument('--ker_sz', type=int, default=8, help='kernel border')
    parser.add_argument('--imsz', type=int, default=512, help='rgb border')
    parser.add_argument('--hsi_slice_xy', type=str, default='0,0', help='rgb border')

    parser.add_argument('-vis_a', '--visdom_address', type=str,
                        default='localhost', help='Network address of the visdom server')
    parser.add_argument('-vis_p', '--visdom_port', type=int,
                        default=8097, help='Port of the visdom server')
    parser.add_argument('-vis_v', '--visdom_video', type=int,
                        default=1,
                        help='Output the estimate of the hidden video to visdom. Requires ffmpeg command line.')

    args = parser.parse_args()
    # args.log_dir = args.model_dir + '/trainlog'
    # args.model_dir = args.model_dir + '/trainmodel'

    return args