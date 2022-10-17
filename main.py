import os
import socket

from datetime import datetime
import traceback

import utils
from trainer import run_training
from get_argparser import get_argparser

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

def mkLogDir(args, verbose=False):

    hostname = socket.gethostname()
    pth_prefix = os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) )
    folder_name = os.path.dirname( os.path.abspath(__file__) ).split('/')[-1] + '-result'
    current_time = datetime.now().strftime('%b%d-%H-%M-%S') + ('_demo' if args.run_demo else '')

    if verbose:
        print(f'hostname: {hostname}\nfolder: {folder_name}\ncurrent time: {current_time}\nprefix: {pth_prefix}')
    
    args.current_time = current_time
    args.prefix = os.path.join(pth_prefix, folder_name, current_time)
    args.TB_pth = os.path.join(pth_prefix, folder_name, current_time, 'tensorboard')
    args.BP_pth = os.path.join(pth_prefix, folder_name, current_time, 'best-param')

    if not os.path.exists(args.prefix):
        os.makedirs(args.prefix)
        os.makedirs(os.path.join(args.prefix, 'best-param'))
        os.makedirs(os.path.join(args.prefix, 'tensorboard'))

def main_worker(args, ) -> dict:
 
    params = utils.save_argparser(args, args.prefix)
  
    results = {}
    for exp_itrs in range(0, args.exp_itrs):
        RUN_ID = 'run_' + str(exp_itrs).zfill(2)
        print(f"{exp_itrs + 1}-th iteration")
        
        start_time = datetime.now()
        results[RUN_ID] = {}
        
        f1bg, f1roi, bbox = 0.0, 0.0, 0.0
        for i in range(0, args.kfold):
            args.k = i
            DATA_FOLD = 'fold_' + str(i).zfill(2)
            print(f"{i + 1}-th data fold")
            os.makedirs(os.path.join( args.TB_pth, RUN_ID, DATA_FOLD ))
            os.makedirs(os.path.join( args.BP_pth, RUN_ID, DATA_FOLD ))
            
            results[RUN_ID][DATA_FOLD] = run_training(args, RUN_ID, DATA_FOLD)
            
            f1bg += results[RUN_ID][DATA_FOLD]['F1 score']['background']
            f1roi += results[RUN_ID][DATA_FOLD]['F1 score']['RoI']
            bbox += results[RUN_ID][DATA_FOLD]['Bbox regression']['MSE']

        subject = f"[{args.model}]" + args.current_time
        msg_body = {
            "Description" : f"kfold (k={args.kfold}) average summary",
            "Short Memo" : args.short_memo,
            "Model" : "Model: " + args.model,
            "Dir" : "Dir: " + args.current_time,
            "F1 score" : {
                "background" : f"{f1bg / args.kfold:.5f}",
                "RoI" : f"{f1roi / args.kfold:.5f}"
            },
            "Bbox regression" : {
                "MSE" : f"{bbox / args.kfold:.5f}",
            },
            "time elapsed" : str(datetime.now() - start_time)
        }
        utils.Email(subject=subject, msg=msg_body, ).send()

    return results

def main():
    total_time = datetime.now()

    try:
        is_error = False

        args = get_argparser().parse_args()
        
        mkLogDir(args, verbose=True, )

        results = main_worker(args, )
        utils.save_dict_to_json(d=results ,json_path=os.path.join(args.prefix, 'result-summary.json'))

    except KeyboardInterrupt:
        is_error = True
        print("KeyboardInterrupt: Stop !!!")

    except Exception as e:
        is_error = True
        print("Error: ", e)
        print(traceback.format_exc())
    
    if is_error:
        os.rename(args.prefix, args.prefix + '_aborted')

    total_time = datetime.now() - total_time
    print('Time elapsed (h:m:s.ms) {}'.format(total_time))
    

if __name__ == "__main__":
    main()