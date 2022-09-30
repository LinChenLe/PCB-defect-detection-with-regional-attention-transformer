import argparse
import torch
def get_args_parser():
    parser = argparse.ArgumentParser('Set detector parameter', add_help=False)
    parser.add_argument('--seed', type=int, default=42,
                        help='seed for reproducibility')
    parser.add_argument('--device', type=torch.device, default=torch.device('cuda:0'),
                        help='compute on this device')
    # load data
    # =============================================================================
    parser.add_argument('--data_foloder', type=str, default='./DeepPCB-master/PCBData/',
                        help='DeepPCB data foloder')
    
    parser.add_argument('--num_object',type=int, default=6,
                        help='total class num of defect')
    
    parser.add_argument('--save_model_path',type=str, default='ckpt/',
                        help='save check point path')
    
    # parser.add_argument('--load_ckpt',type=str, default='ckpt/'+'',
    #                     help='load check point file path')
    
    parser.add_argument('--load_backbone_ckpt',type=str, default='ckpt/'+'',
                        help='load backbone check point file path')
    

    # =============================================================================
    # =============================================================================
    # training config
    parser.add_argument('--val_rate',type=int, default=5,
                        help='validation after val_rate epoch')
    
    # parser.add_argument('--classify_lr',type=float, default=1e-4,
    #                     help='learning rate for optimizer')
    # parser.add_argument('--classify_min_lr',type=float, default=1e-5,
    #                     help='smallest learning rate for warm up')
    
    parser.add_argument('--backbone_lr',type=float, default=1e-4,
                        help='learning rate for optimizer')
    
    parser.add_argument('--backbone_min_lr',type=float, default=1e-5,
                        help='smallest learning rate for warm up')
    
    # parser.add_argument('--lr_drop',type=int, default=5,
    #                     help='learning rate drop by lr_drop epochs after')
    # parser.add_argument('--lr_gamma',type=float, default=0.5,
    #                     help='learning rate drop by lr*lr_gamma')
    
    parser.add_argument('--max_lr_epoch',type=float, default=10,
                        help='max lreaning rate epoch for wram up')
    
    parser.add_argument('--training_mode',type=str, default='backbone',# backbone or classify
                        help="training mode for segmentation or bounddingbox")
    
    

    # =============================================================================
    # model config
    # parser.add_argument('--num_queries',type=int, default=20,
    #                     help='decoder layers inputs and outputs number')
    # parser.add_argument('--hidden_dim',type=int, default=512,
    #                     help='all model hidden dimension')
    # parser.add_argument('--num_channel',type=int, default=256,
    #                     help='each feature vactor length')
    # parser.add_argument('--non_OJ_cost',type=int, default=0.1,
                        # help='non object cost')
    # =============================================================================
    # =============================================================================
    # Logger
    
    parser.add_argument('--logger_pfreq',type=int, default=1,
                        help='print logger after logger_pfreq data')
    
    parser.add_argument('--not_show_list',type=list, default=['confusion_matrix'],
                        help='when print logger, information in this list ignore it')
    
    parser.add_argument('--not_smooth_list',type=list, default=['lr'],
                        help='when print logger, information in this list ignore it')
    # =============================================================================
    parser.add_argument('--batch_size',type=int, default=4,
                        help='batch size for load [batch_size] data at once')
    
    parser.add_argument('--model_out_size',type=int, default=16,
                        help='resize mask to be model outputs size')

    
    return parser
if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    print(args.load_data_mode)