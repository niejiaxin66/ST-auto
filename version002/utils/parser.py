
import argparse
def getparse():
    parse = argparse.ArgumentParser()
    
    #Spatial
    
    #model
    parse.add_argument('-s_model_d',type=int,default=32)
    parse.add_argument('-model_N',type=int,default=6)
    parse.add_argument('-k',type=int,default=20)
    parse.add_argument('-spatial',type=str,choices=['gcn','transformer'],help='choose the spatial model type',default='transformer')
    parse.add_argument('-mode',type=str,default='corr',choices=['cos','corr'],help='choose the way to get adj metrix')
    parse.add_argument('-c',action='store_true')
    parse.add_argument('-s',action='store_true')
    parse.add_argument('-FS',action='store_true')
    #parse.add_argument('-nb_flow', type=int, default=3)
    #parse.add_argument('-flow',type=int,choices=[0,1,2],default=0,help='in--0,out--1')
    #parse.add_argument('-c_t',type=str,default='p',choices=['t','p','tp','c','r'])
    #parse.add_argument('-s_t',type=str,default='c',choices=['t','p','tp','c','r'])
    #training
    parse.add_argument('-train', dest='train', action='store_true')
    parse.add_argument('-no-train', dest='train', action='store_false')
    parse.set_defaults(train=True)
    parse.add_argument('-rows', nargs='+', type=int, default=[40, 60])
    parse.add_argument('-cols', nargs='+', type=int, default=[40, 60])
    parse.add_argument('-lr', type=float)
    parse.add_argument('-se',type=int)
    parse.add_argument('-epoch_size', type=int, default=200, help='epochs')
    
    #这两行是用来干啥的。。。先注释掉
    #parse.add_argument('-test_row', type=int, default=51, help='test row')
    #parse.add_argument('-test_col', type=int, default=60, help='test col')
    parse.add_argument('-g',type=str,default=None)
    parse.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parse.add_argument('-w',type=str)
    parse.add_argument('-save_dir', type=str, default='results')
    parse.add_argument('-best_valid_loss',type=float,default=1)
    parse.add_argument('-lr-scheduler', type=str, default='poly',choices=['poly', 'step', 'cos'],
                            help='lr scheduler mode: (default: poly)')

    parse.add_argument('-warmup',type=int,default=100)
    parse.add_argument('-test_batch_size',type=int,default=1)
    
    
    
    
    #Autoformer

    # basic config
    parse.add_argument('--is_training', type=int, default=1, help='status')
    parse.add_argument('--model_id', type=str, default='test', help='model id')
    parse.add_argument('--model', type=str, default='Autoformer',
                    help='model name, options: [Autoformer, Transformer]')

    # data loader
    #parse.add_argument('--data', type=str, default='pems04', help='dataset type')
    parse.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parse.add_argument('--freq', type=str, default='d',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parse.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parse.add_argument('--label_len', type=int, default=3, help='start token length')
    parse.add_argument('--pred_len', type=int, default=7,help='prediction sequence length')

    # model define
    parse.add_argument('--enc_in', type=int, default=66, help='encoder input size')
    parse.add_argument('--dec_in', type=int, default=66, help='decoder input size')
    parse.add_argument('--c_out', type=int, default=66, help='output size')
    parse.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parse.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parse.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parse.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parse.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parse.add_argument('--moving_avg', type=int, default=7, help='window size of moving average')
    parse.add_argument('--factor', type=int, default=1, help='attn factor')
    parse.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
    parse.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parse.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
    parse.add_argument('--activation', type=str, default='gelu', help='activation')
    parse.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parse.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parse.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parse.add_argument('--itr', type=int, default=2, help='experiments times')
    parse.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parse.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parse.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parse.add_argument('--des', type=str, default='test', help='exp description')
    parse.add_argument('--loss', type=str, default='mse', help='loss function')
    parse.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parse.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    # GPU
    parse.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parse.add_argument('--gpu', type=int, default=0, help='gpu')
    parse.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parse.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')
    
    #common
    parse.add_argument('-seq_len', type=int, default=7)#
    parse.add_argument('-test_size', type=int, default=15)
    parse.add_argument('-batch_size', type=int, default=32, help='batch size')
    parse.add_argument('-group_type',type=int,default=1)
    #这里是为了处理我们高维数据的读取，比如说，疫情中，R，I和S是三种人群，储存在数据里就是三列，这里就是在筛选
    parse.add_argument('-split',type=int, default=0.9, help='split the data into train and valid')
    parse.add_argument('-data_type',type=str, default='train',help='train or valid')
    return parse.parse_args()
