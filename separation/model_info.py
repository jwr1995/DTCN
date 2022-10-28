
import os, sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from thop import profile, clever_format

from train import Separation

hparam_file_list = [
    "baselines/tcn/convtasnet-whamr.yaml",
    "deformable/convtasnet-whamr.yaml",
    "deformable/shared_weights/convtasnet-whamr.yaml",
]

model_names = ["_".join(f.split("/")[:-1]) for f in hparam_file_list]
# print(model_names);exit()

run_opts_list = ["--data_folder /fastdata/acp19jwr/data/mono-whamr"]*len(hparam_file_list)

overrides_list = [""]*len(hparam_file_list)

sig_len=6

for h,r,o,m in zip(hparam_file_list,run_opts_list,overrides_list,model_names):
    hparams_file, run_opts, overrides = sb.parse_arguments([h]+r.split(" ")+o.split(" "))
    # print(sb.parse_arguments([h]+r.split(" ")+o.split(" ")));exit()
    
    with open(os.path.join("hparams",hparams_file)) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    run_opts["auto_mix_prec"] = hparams["auto_mix_prec"]


    input = torch.randn(1, hparams["sample_rate"]*sig_len).cuda()
    
    model = Separation(
            modules=hparams["modules"],
            opt_class=hparams["optimizer"],
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=hparams["checkpointer"],
        )

    model.hparams.encode_rirs = False

    prof_macs, prof_params = model.compute_forward(input, None, sb.Stage.TEST, None, profiler=True)
    prof_macs["decoder"] = prof_macs["decoder"]*model.hparams.num_spks
    total_macs = sum(prof_macs.values())/sig_len
    total_params = sum(prof_params.values())
    macs, params = clever_format([total_macs, total_params], "%.3f")
    print(prof_macs["decoder"])
    print(m,macs,params)
    
    