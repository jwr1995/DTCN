"""
Author
 * Cem Subakan 2020
 * Will Ravenscroft 2021

The .csv preperation functions for WHAMR
"""

import os, glob, csv, json
import soundfile as sf

set_map = {"tr":"train","cv":"valid","tt":"test"}

def get_rir_paths(rir_path, sets=set_map.keys()):
    rir_directory = {
        set :
        {
            "s1" :
            {
                os.path.basename(filepath).replace("0_0_","") : filepath 
                    for filepath in glob.glob(os.path.join(rir_path,set,"0_0_*"))
            },
            "s2" :
            {
                os.path.basename(filepath).replace("0_1_","") : filepath 
                    for filepath in glob.glob(os.path.join(rir_path,set,"0_1_*"))
            }    
        }
        for set in sets
    }
    return rir_directory

def get_meta_data(creation_path, wham_noise_path):
    """
    creation_path = Path to data folder in WHAMR! creation scripts
    """
    rir_csv_paths = {
        os.path.basename(fname).replace("reverb_params_","").replace(".csv","") : fname 
        for fname in glob.glob(os.path.join(creation_path,"data","reverb_params*.csv"))
        }
    # mix_csv_paths = {
    #     fname.replace("reverb_params_","").replace(".csv") : fname 
    #     for fname in glob.glob(os.path.join(creation_path,"data","reverb_params*.csv"))
    #     }
    noise_csv_paths = {
        os.path.basename(fname).replace("mix_param_meta_","").replace(".csv","") : fname 
        for fname in glob.glob(os.path.join(wham_noise_path,"metadata","mix_param_meta*.csv"))
        }
 
    sets = ["cv","tr","tt"]
    
    directory = {}

    for i, set in enumerate(sets):
        try:
            with open(rir_csv_paths[set],'r') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    room_size = float(row['room_x'])*float(row['room_y'])*float(row['room_z'])
                    directory[row["utterance_id"]]={
                        "t60":row["T60"],
                        "room_size":room_size,
                        "room_x":row["room_x"],
                        "room_y":row["room_y"],
                        "room_z":row["room_z"],
                        "micL_x":row["micL_x"],
                        "micL_y":row["micL_y"],
                        "micR_x":row["micR_x"],
                        "micR_y":row["micR_y"],
                        "mic_z":row["mic_z"],
                        "s1_x":row["s1_x"],
                        "s1_y":row["s1_y"],
                        "s1_z":row["s1_z"],
                        "s2_x":row["s2_x"],
                        "s2_y":row["s2_y"],
                        "s2_z":row["s2_z"]
                        }
                
            with open(noise_csv_paths[set],'r') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate (reader):
                    directory[row["utterance_id"]]["snr"]=float(row["target_speaker1_snr_db"])
        except KeyError as e:
            rir_key = "Keys found (RIRs):"+str(rir_csv_paths.keys())
            noise_key = "Keys found (noises):"+str(noise_csv_paths.keys())
            raise KeyError(str(e)+'. '+rir_key+'. '+noise_key)
        
    return directory

def get_transcriptions(
    wsj0_path, 
    case='u', 
    filters=["\.PERIOD","\.COMMA","\-HYPHEN","\-\-DASH","\\\"DOUBLE\-QUOTE"]
    ): #/path/to/...11.-1.1 etc.
    dot_files = (glob.glob(os.path.join(wsj0_path,"*/*/*/*.dot"))+
                glob.glob(os.path.join(wsj0_path,"*/*/*/*/*.dot"))+
                glob.glob(os.path.join(wsj0_path,"*/*/*/*/*/*.dot"))+
                glob.glob(os.path.join(wsj0_path,"*/*/*/*/*/*/*.dot"))+
                glob.glob(os.path.join(wsj0_path,"*/*/*/*/*/*/*/*.dot")))

    test_file = "/share/mini1/data/audvis/pub/asr/studio/us/wsj/v2/wsj0/11-10.1/wsj0/transcrp/dots/si_tr_s/01v/01vo0300.dot"
    test_label = "01vo030q"
    assert test_file in dot_files
                
    dot_directory = {}
    for dot in dot_files:
        with open(dot,'r') as f:
            for line in f:
                try:
                    line = line.replace(")","").replace("(","")
                    utterance, label = line[:-9], line[-9:]
                except ValueError as e:
                    raise ValueError("Can't split line \""+line+"\"")
                label = label.replace(")","").replace("\n","")
                dot_directory[label] = utterance.upper() if 'u' else utterance.lower()
        
    return dot_directory

def prepare_wham_whamr_csv(
    datapath, 
    savepath, 
    skip_prep=False, 
    fs=8000, 
    mini=False, 
    mix_folder="mix_both_reverb",
    target_condition="anechoic",
    set_types=["tr", "cv", "tt"],
    num_spks=2,
    alternate_path=None,
    creation_path=None,
    wham_noise_path=None,
    version="min",
    wsj0_path=None, # for transcriptions
    eval_original=False,
    meta_dump=False,
    use_rirs=False,
    rir_path=None,
    extended=False,
    savename="whamr_"
):
    """
    Prepares the csv files for wham or whamr dataset

    Arguments:
    ----------
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file.
        skip_prep (bool): If True, skip data preparation
    """

    if skip_prep:
        return

    if "whamr" in datapath:
        # if we want to train a model on the whamr dataset
        create_wham_whamr_csv(datapath, savepath, fs,mix_folder=mix_folder,
            alternate_path=alternate_path, num_spks=num_spks, target_condition=target_condition,
            version=version, creation_path=creation_path, wham_noise_path=wham_noise_path, 
            wsj0_path=wsj0_path, eval_original=eval_original, meta_dump=meta_dump, rir_path=rir_path,
            use_rirs=use_rirs,set_types=set_types,extended=extended,savename=savename
        )
    elif "wham" in datapath:
        # if we want to train a model on the original wham dataset
        create_wham_whamr_csv(
            datapath, savepath, fs, savename="whamorg_", add_reverb=False, mini=mini,
            mix_folder=mix_folder.replace("reverb","").replace("anechoic","")
        )
    else:
        raise ValueError("Unsupported Dataset at: "+datapath)

def create_wham_whamr_csv(
    datapath,
    savepath,
    fs,
    version="min",
    mix_folder="mix_both_reverb",
    target_condition="anechoic",
    savename="whamr_",
    set_types=["tr", "cv", "tt"],
    add_reverb=True,
    mini=False,
    num_spks=2,
    alternate_path=None,
    creation_path=None,
    wham_noise_path=None,
    wsj0_path=None, # for transcriptions
    eval_original=False,
    meta_dump = False, # grabs every kind of meta data for room simulation
    use_rirs = False,
    rir_path=None,
    extended=False
):
    """
    This function creates the csv files to get the speechbrain data loaders for the whamr dataset.

    Arguments:
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file
        fs (int) : the sampling rate
        version (str) : min or max
        savename (str) : the prefix to use for the .csv files
        set_types (list) : the sets to create
    """

    if alternate_path==None:
        target_path = datapath
    else:
        target_path = alternate_path
    
    if creation_path != None and wham_noise_path != None:
        extra = True
        directory = get_meta_data(creation_path, wham_noise_path)
    else: 
        extra = False
    
    if not wsj0_path == None:
        transcription_directory =  get_transcriptions(wsj0_path)
        transcribe = True
    else:
        transcribe = False
    
    # if not rir_path == None:
    #     rir_directory= get_rir_paths(rir_path)
    #     use_rirs = True
    # else:
    #     use_rirs = False

    if fs == 8000:
        sample_rate = "8k"
    elif fs == 16000:
        sample_rate = "16k"
    else:
        raise ValueError("Unsupported sampling rate")

    if add_reverb:
        mix = mix_folder+"/"
        s1 = "s1_"+target_condition+"/"
        s2 = "s2_"+target_condition+"/"
    else:
        mix = mix_folder+"/"
        s1 = "s1/"
        s2 = "s2/"

    if eval_original:
        s1_eval = "s1_anechoic/"
        s2_eval = "s2_anechoic/"
    

    for set_type in set_types:
        mix_path = os.path.join(
            datapath, "wav{}".format(sample_rate), version, set_type, mix,
        )
        if eval_original and (set_type == "tt" or set_type == "cv"):
            s1_path = os.path.join(
                datapath, "wav{}".format(sample_rate), version, set_type, s1_eval,
            )
            if num_spks==2:
                s2_path = os.path.join(
                    datapath, "wav{}".format(sample_rate), version, set_type, s2_eval,
                )
        else:
            s1_path = os.path.join(
                target_path, "wav{}".format(sample_rate), version, set_type, s1,
            )
            if num_spks==2:
                s2_path = os.path.join(
                    target_path, "wav{}".format(sample_rate), version, set_type, s2,
                )
       
        noise_path = os.path.join(
            datapath, "wav{}".format(sample_rate), version, set_type, "noise/"
        )
        # rir_path = os.path.join(
        #     datapath, "wav{}".format(sample_rate), version, set_type, "rirs/"
        # )

        files = os.listdir(mix_path)

        mix_fl_paths = [mix_path + fl for fl in files]
        s1_fl_paths = [s1_path + fl for fl in files]
        if num_spks==2:
            s2_fl_paths = [s2_path + fl for fl in files]
        noise_fl_paths = [noise_path + fl for fl in files]
        # rir_fl_paths = [rir_path + fl + ".t" for fl in files]

        # if not rir_path ==None:
        #     rir_paths = rir_directory[set_type]

        if num_spks==1:
            csv_columns = [
                "ID",
                "duration",
                "mix_wav",
                "mix_wav_format",
                "mix_wav_opts",
                "s1_wav",
                "s1_wav_format",
                "s1_wav_opts",
                "noise_wav",
                "noise_wav_format",
                "noise_wav_opts",
                # "rir_t",
                # "rir_format",
                # "rir_opts",
                "t60",
                "room_size",
                "snr",
                "s1_dot"
            ]
        else:
            csv_columns = [
                "ID",
                "duration",
                "mix_wav",
                "mix_wav_format",
                "mix_wav_opts",
                "s1_wav",
                "s1_wav_format",
                "s1_wav_opts",
                "s2_wav",
                "s2_wav_format",
                "s2_wav_opts",
                "noise_wav",
                "noise_wav_format",
                "noise_wav_opts",
                # "rir_t",
                # "rir_format",
                # "rir_opts",
                "t60",
                "room_size",
                "snr",
                "s1_dot",
                "s2_dot"
            ]

        if meta_dump:
            extra_cols = [
                "room_x",
                "room_y",
                "room_z",
                "micL_x",
                "micL_y",
                "micR_x",
                "micR_y",
                "mic_z",
                "s1_x",
                "s1_y",
                "s1_z",
                "s2_x",
                "s2_y",
                "s2_z"
                ]
            csv_columns = csv_columns + extra_cols
        
        if use_rirs:
            extra_cols = [
                "s1_rir",
                "s2_rir"
            ]
            csv_columns = csv_columns + extra_cols

        with open(
            (
            os.path.join(savepath, savename + set_type + ".csv") if not extended else os.path.join(savepath, savename + set_type + "_ext.csv")
            ), "w"
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            if mini and (num_spks==1):
                zipped = list(zip(
                    mix_fl_paths,
                    s1_fl_paths,
                    noise_fl_paths,
                    # rir_fl_paths,
                ))[:len(mix_fl_paths)//4]
            elif (not mini) and (num_spks==1):
                zipped = zip(
                    mix_fl_paths,
                    s1_fl_paths,
                    noise_fl_paths,
                    # rir_fl_paths,
                )
            elif mini and (num_spks==2):
                zipped = list(zip(
                    mix_fl_paths,
                    s1_fl_paths,
                    s2_fl_paths,
                    noise_fl_paths,
                    # rir_fl_paths,
                ))[:len(mix_fl_paths)//4]
            else:
                zipped = zip(
                    mix_fl_paths,
                    s1_fl_paths,
                    s2_fl_paths,
                    noise_fl_paths,
                    # rir_fl_paths,
                )
            for (i, packed,) in enumerate(zipped):
                if num_spks==1:
                    mix_path, s1_path, noise_path = packed
                    basename = os.path.basename(mix_path)
                    if transcribe:
                        s1_basename = os.path.basename(s1_path).replace(".wav","")
                        s1_dot = transcription_directory[s1_basename]
                    else:
                        s1_dot=None
                    row = {
                    "ID": i,
                    "duration": 1.0,
                    "mix_wav": mix_path,
                    "mix_wav_format": "wav",
                    "mix_wav_opts": None,
                    "s1_wav": s1_path,
                    "s1_wav_format": "wav",
                    "s1_wav_opts": None,
                    "noise_wav": noise_path,
                    "noise_wav_format": "wav",
                    "noise_wav_opts": None,
                    # "rir_t": rir_path,
                    # "rir_format": ".t",
                    # "rir_opts": None,
                    "t60": directory[basename]["t60"] if extra else None,
                    "room_size": directory[basename]["room_size"] if extra else None,
                    "snr": directory[basename]["snr"] if extra else None,
                    "s1_dot": s1_dot
                    }
                else:
                    mix_path, s1_path, s2_path, noise_path = packed
                    basename = os.path.basename(mix_path)
                    if transcribe:
                        s1_basename = os.path.basename(s1_path).replace(".wav","")
                        s1_dot = transcription_directory[s1_basename]
                        s2_basename = os.path.basename(s2_path).replace(".wav","")
                        s2_dot = transcription_directory[s2_basename]
                    else:
                        s1_dot=None
                        s2_dot=None
                    row = {
                        "ID": i,
                        "duration": 1.0,
                        "mix_wav": mix_path,
                        "mix_wav_format": "wav",
                        "mix_wav_opts": None,
                        "s1_wav": s1_path,
                        "s1_wav_format": "wav",
                        "s1_wav_opts": None,
                        "s2_wav": s2_path,
                        "s2_wav_format": "wav",
                        "s2_wav_opts": None,
                        "noise_wav": noise_path,
                        "noise_wav_format": "wav",
                        "noise_wav_opts": None,
                        # "rir_t": rir_path,
                        # "rir_format": ".t",
                        # "rir_opts": None,
                        "t60": directory[basename]["t60"] if extra else None,
                        "room_size": directory[basename]["room_size"] if extra else None,
                        "snr": directory[basename]["snr"] if extra else None,
                        "s1_dot": s1_dot,
                        "s2_dot": s2_dot,
                    }

                if meta_dump:
                    row["room_x"] = directory[basename]["room_x"] 
                    row["room_y"] = directory[basename]["room_y"] 
                    row["room_z"] = directory[basename]["room_z"] 
                    row["micL_x"] = directory[basename]["micL_x"] 
                    row["micL_y"] = directory[basename]["micL_y"] 
                    row["micR_x"] = directory[basename]["micR_x"] 
                    row["micR_y"] = directory[basename]["micR_y"] 
                    row["mic_z"] = directory[basename]["mic_z"] 
                    row["s1_x"] = directory[basename]["mic_z"] 
                    row["s1_y"] = directory[basename]["s1_y"] 
                    row["s1_z"] = directory[basename]["s1_z"] 
                    row["s2_x"] = directory[basename]["s2_x"] 
                    row["s2_y"] = directory[basename]["s2_y"] 
                    row["s2_z"] = directory[basename]["s2_z"] 
                
                if use_rirs:
                    # row["s1_rir"] = rir_paths["s1"][basename]
                    # row["s2_rir"] = rir_paths["s2"][basename]
                    row["s1_rir"] = s1_path.replace("s1_anechoic","s1_rir")
                    if num_spks == 2:
                        row["s2_rir"] = s2_path.replace("s2_anechoic","s2_rir")


                writer.writerow(row)

def create_whamr_rir_csv(datapath, savepath):
    """
    This function creates the csv files to get the data loaders for the whamr  dataset.

    Arguments:
        datapath (str) : path for the whamr rirs.
        savepath (str) : path where we save the csv file
    """

    csv_columns = ["ID", "duration", "wav", "wav_format", "wav_opts"]

    files = os.listdir(datapath)
    all_paths = [os.path.join(datapath, fl) for fl in files]

    with open(savepath + "/whamr_rirs.csv", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for i, wav_path in enumerate(all_paths):

            row = {
                "ID": i,
                "duration": 2.0,
                "wav": wav_path,
                "wav_format": "wav",
                "wav_opts": None,
            }
            writer.writerow(row)

def create_wham_whamr_json(
    datapath,
    savepath,
    fs,
    version="min",
    mix_folder="mix_both_reverb",
    target_condition="anechoic",
    savename="whamr_",
    set_types=["tr", "cv", "tt"],
    add_reverb=True,
    mini=False,
    num_spks=2,
    alternate_path=None,
    creation_path=None,
    wham_noise_path=None,
    wsj0_path=None, # for transcriptions
    rir_path=None,
    eval_original=False,
):
    """
    This function creates the csv files to get the speechbrain data loaders for the whamr dataset.

    Arguments:
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file
        fs (int) : the sampling rate
        version (str) : min or max
        savename (str) : the prefix to use for the .csv files
        set_types (list) : the sets to create
    """

    if alternate_path==None:
        target_path = datapath
    else:
        target_path = alternate_path
    
    if creation_path != None and wham_noise_path != None:
        extra = True
        directory = get_meta_data(creation_path, wham_noise_path)
    else: 
        extra = False
    
    if not wsj0_path == None:
        transcription_directory =  get_transcriptions(wsj0_path)
        transcribe = True
    else:
        transcribe = False

    if fs == 8000:
        sample_rate = "8k"
    elif fs == 16000:
        sample_rate = "16k"
    else:
        raise ValueError("Unsupported sampling rate")

    if add_reverb:
        mix = mix_folder+"/"
        s1 = "s1_"+target_condition+"/"
        s2 = "s2_"+target_condition+"/"
    else:
        mix = mix_folder+"/"
        s1 = "s1/"
        s2 = "s2/"

    if eval_original:
        s1_eval = "s1_anechoic/"
        s2_eval = "s2_anechoic/"
    

    for set_type in set_types:
        mix_path = os.path.join(
            datapath, "wav{}".format(sample_rate), version, set_type, mix,
        )
        if eval_original and (set_type == "tt" or set_type == "cv"):
            s1_path = os.path.join(
                datapath, "wav{}".format(sample_rate), version, set_type, s1_eval,
            )
            if num_spks==2:
                s2_path = os.path.join(
                    datapath, "wav{}".format(sample_rate), version, set_type, s2_eval,
                )
        else:
            s1_path = os.path.join(
                target_path, "wav{}".format(sample_rate), version, set_type, s1,
            )
            if num_spks==2:
                s2_path = os.path.join(
                    target_path, "wav{}".format(sample_rate), version, set_type, s2,
                )
       
        noise_path = os.path.join(
            datapath, "wav{}".format(sample_rate), version, set_type, "noise/"
        )
        # rir_path = os.path.join(
        #     datapath, "wav{}".format(sample_rate), version, set_type, "rirs/"
        # )

        files = os.listdir(mix_path)

        mix_fl_paths = [mix_path + fl for fl in files]
        s1_fl_paths = [s1_path + fl for fl in files]
        if num_spks==2:
            s2_fl_paths = [s2_path + fl for fl in files]
        noise_fl_paths = [noise_path + fl for fl in files]
        # rir_fl_paths = [rir_path + fl + ".t" for fl in files]

        if num_spks==1:
            csv_columns = [
                "ID",
                "duration",
                "mix_wav",
                "mix_wav_format",
                "mix_wav_opts",
                "s1_wav",
                "s1_wav_format",
                "s1_wav_opts",
                "noise_wav",
                "noise_wav_format",
                "noise_wav_opts",
                # "rir_t",
                # "rir_format",
                # "rir_opts",
                "t60",
                "room_size",
                "snr",
                "s1_dot"
            ]
        else:
            csv_columns = [
                "ID",
                "duration",
                "mix_wav",
                "mix_wav_format",
                "mix_wav_opts",
                "s1_wav",
                "s1_wav_format",
                "s1_wav_opts",
                "s2_wav",
                "s2_wav_format",
                "s2_wav_opts",
                "noise_wav",
                "noise_wav_format",
                "noise_wav_opts",
                # "rir_t",
                # "rir_format",
                # "rir_opts",
                "t60",
                "room_size",
                "snr",
                "s1_dot",
                "s2_dot"
            ]

        with open(
            os.path.join(savepath, savename + set_type + ".csv"), "w"
        ) as csvfile:
            json_dict = {}
            if mini and (num_spks==1):
                zipped = list(zip(
                    mix_fl_paths,
                    s1_fl_paths,
                    noise_fl_paths,
                    # rir_fl_paths,
                ))[:len(mix_fl_paths)//4]
            elif (not mini) and (num_spks==1):
                zipped = zip(
                    mix_fl_paths,
                    s1_fl_paths,
                    noise_fl_paths,
                    # rir_fl_paths,
                )
            elif mini and (num_spks==2):
                zipped = list(zip(
                    mix_fl_paths,
                    s1_fl_paths,
                    s2_fl_paths,
                    noise_fl_paths,
                    # rir_fl_paths,
                ))[:len(mix_fl_paths)//4]
            else:
                zipped = zip(
                    mix_fl_paths,
                    s1_fl_paths,
                    s2_fl_paths,
                    noise_fl_paths,
                    # rir_fl_paths,
                )
            for (i, packed,) in enumerate(zipped):
                if num_spks==1:
                    mix_path, s1_path, noise_path = packed
                    basename = os.path.basename(mix_path)
                    data, fs = sf.read(mix_path)
                    duration = len(data)/fs
                    if transcribe:
                        s1_basename = os.path.basename(s1_path).replace(".wav","")
                        s1_dot = transcription_directory[s1_basename]
                    else:
                        s1_dot=None
                    row = {
                    "ID": i,
                    "duration": duration,
                    "mix_wav": mix_path,
                    "mix_wav_format": "wav",
                    "mix_wav_opts": None,
                    "s1_wav": s1_path,
                    "s1_wav_format": "wav",
                    "s1_wav_opts": None,
                    "noise_wav": noise_path,
                    "noise_wav_format": "wav",
                    "noise_wav_opts": None,
                    # "rir_t": rir_path,
                    # "rir_format": ".t",
                    # "rir_opts": None,
                    "t60": directory[basename]["t60"] if extra else None,
                    "room_size": directory[basename]["room_size"] if extra else None,
                    "snr": directory[basename]["snr"] if extra else None,
                    "s1_dot": s1_dot
                    }
                else:
                    mix_path, s1_path, s2_path, noise_path = packed
                    basename = os.path.basename(mix_path)
                    data, fs = sf.read(mix_path)
                    duration = len(data)/fs
                    if transcribe:
                        s1_basename = os.path.basename(s1_path).replace(".wav","")
                        s1_dot = transcription_directory[s1_basename]
                        s2_basename = os.path.basename(s2_path).replace(".wav","")
                        s2_dot = transcription_directory[s2_basename]
                    else:
                        s1_dot=None
                        s2_dot=None
                    row = {
                        "ID": i,
                        "duration": duration,
                        "mix_wav": mix_path,
                        "mix_wav_format": "wav",
                        "mix_wav_opts": None,
                        "s1_wav": s1_path,
                        "s1_wav_format": "wav",
                        "s1_wav_opts": None,
                        "s2_wav": s2_path,
                        "s2_wav_format": "wav",
                        "s2_wav_opts": None,
                        "noise_wav": noise_path,
                        "noise_wav_format": "wav",
                        "noise_wav_opts": None,
                        # "rir_t": rir_path,
                        # "rir_format": ".t",
                        # "rir_opts": None,
                        "t60": directory[basename]["t60"] if extra else None,
                        "room_size": directory[basename]["room_size"] if extra else None,
                        "snr": directory[basename]["snr"] if extra else None,
                        "s1_dot": s1_dot,
                        "s2_dot": s2_dot,
                    }

                json_dict[basename.replace(".wav","")+"_#_"+target_condition] = row
        json_fname = os.path.join(savepath,set_type+".json")
        with open(json_fname, 'w') as f:
            json_string= json.dumps(json_dict, sort_keys=False, indent=4)
            f.write(json_string)


def tokenizer_data_prep(
    datapath,
    savepath,
    fs,
    version="min",
    mix_folder="mix_both_reverb",
    target_condition="anechoic",
    savename="whamr_",
    set_types=["tr", "cv", "tt"],
    add_reverb=True,
    mini=False,
    num_spks=2,
    wsj0_path=None, # for transcriptions
):
    """
    This function creates the csv files to get the speechbrain data loaders for the whamr dataset.

    Arguments:
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file
        fs (int) : the sampling rate
        version (str) : min or max
        savename (str) : the prefix to use for the .csv files
        set_types (list) : the sets to create
    """

    os.makedirs(savepath, exist_ok=True)
    
    if not wsj0_path == None:
        transcription_directory =  get_transcriptions(wsj0_path)
        transcribe = True
    else:
        transcribe = False

    if fs == 8000:
        sample_rate = "8k"
    elif fs == 16000:
        sample_rate = "16k"
    else:
        raise ValueError("Unsupported sampling rate")

    if add_reverb:
        mix = mix_folder+"/"
        s1 = "s1_"+target_condition+"/"
        s2 = "s2_"+target_condition+"/"
    else:
        mix = mix_folder+"/"
        s1 = "s1/"
        s2 = "s2/"


    for set_type in set_types:
        mix_path = os.path.join(
            datapath, "wav{}".format(sample_rate), version, set_type, mix,
        )
        
        s1_path = os.path.join(
            datapath, "wav{}".format(sample_rate), version, set_type, s1,
        )
        if num_spks==2:
            s2_path = os.path.join(
                datapath, "wav{}".format(sample_rate), version, set_type, s2,
            )
    
        noise_path = os.path.join(
            datapath, "wav{}".format(sample_rate), version, set_type, "noise/"
        )
        # rir_path = os.path.join(
        #     datapath, "wav{}".format(sample_rate), version, set_type, "rirs/"
        # )

        files = os.listdir(mix_path)

        mix_fl_paths = [mix_path + fl for fl in files]
        s1_fl_paths = [s1_path + fl for fl in files]
        if num_spks==2:
            s2_fl_paths = [s2_path + fl for fl in files]
        noise_fl_paths = [noise_path + fl for fl in files]
        # rir_fl_paths = [rir_path + fl + ".t" for fl in files]


        with open(
            os.path.join(savepath, set_map[set_type] + ".json"), "w"
        ) as jsonfile:
            json_dict = {}
            if mini and (num_spks==1):
                zipped = list(zip(
                    mix_fl_paths,
                    s1_fl_paths,
                    noise_fl_paths,
                    # rir_fl_paths,
                ))[:len(mix_fl_paths)//4]
            elif (not mini) and (num_spks==1):
                zipped = zip(
                    mix_fl_paths,
                    s1_fl_paths,
                    noise_fl_paths,
                    # rir_fl_paths,
                )
            elif mini and (num_spks==2):
                zipped = list(zip(
                    mix_fl_paths,
                    s1_fl_paths,
                    s2_fl_paths,
                    noise_fl_paths,
                    # rir_fl_paths,
                ))[:len(mix_fl_paths)//4]
            else:
                zipped = zip(
                    mix_fl_paths,
                    s1_fl_paths,
                    s2_fl_paths,
                    noise_fl_paths,
                    # rir_fl_paths,
                )
            for (i, packed,) in enumerate(zipped):
                if num_spks==1:
                    mix_path, s1_path, noise_path = packed
                    basename = os.path.basename(mix_path)
                    data, fs = sf.read(mix_path)
                    duration = len(data)/fs
                    if transcribe:
                        s1_basename = os.path.basename(s1_path).replace(".wav","")
                        s1_dot = transcription_directory[s1_basename]
                    else:
                        s1_dot=None
                    json_dict[basename.replace(".wav","")+"#s1"] = {
                                    "length": duration,
                                    "wav": s1_path,
                                    "words": s1_dot
                            }
                else:
                    mix_path, s1_path, s2_path, noise_path = packed
                    basename = os.path.basename(mix_path)
                    data, fs = sf.read(mix_path)
                    duration = len(data)/fs
                    try:
                        if transcribe:
                            s1_key, _, s2_key, _ = basename.replace(".wav","").split("_")
                            s1_dot = transcription_directory[s1_key]
                            s2_dot = transcription_directory[s2_key]
                        else:
                            s1_dot=None
                            s2_dot=None
                    except KeyError as e:
                        err_str = "Error with key "+str(e)+". Available keys: "+str(list(transcription_directory.keys())[:20])+"..."
                        raise KeyError(err_str)

                    json_dict[basename.replace(".wav","")+"#s1"] = {
                                    "length": duration,
                                    "wav": s1_path,
                                    "words": s1_dot
                            }
                    json_dict[basename.replace(".wav","")+"#s2"] = {
                                "length": duration,
                                "wav": s2_path,
                                "words": s2_dot
                            }
            json_string= json.dumps(json_dict, sort_keys=False, indent=4)
            try:
                jsonfile.write(json_string)
            except ValueError as e:
                raise ValueError("Error writing json dump "+json_string)

def prepare_multi_whamr_csv(
    datapath, 
    savepath, 
    skip_prep=False, 
    fs=8000, 
    mini=False, 
    mix_folder="mix_both_reverb",
    target_folders=["s1_anechoic","s2_anechoic"],
    set_types=["tr", "cv", "tt"],
    num_spks=2,
    alternate_path=None,
    creation_path=None,
    wham_noise_path=None,
    version="min",
    wsj0_path=None, # for transcriptions
    eval_original=False,
    meta_dump=False,
    use_rirs=False,
    rir_path=None,
    extended=False,
    savename="whamr_"
):
    """
    Prepares the csv files for wham or whamr dataset

    Arguments:
    ----------
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file.
        skip_prep (bool): If True, skip data preparation
    """

    if skip_prep:
        return

    if "whamr" in datapath:
        # if we want to train a model on the whamr dataset
        create_multi_whamr_csv(datapath, savepath, fs,mix_folder=mix_folder,
            alternate_path=alternate_path, num_spks=num_spks, target_folders=target_folders,
            version=version, creation_path=creation_path, wham_noise_path=wham_noise_path, 
            wsj0_path=wsj0_path, eval_original=eval_original, meta_dump=meta_dump, rir_path=rir_path,
            use_rirs=use_rirs,set_types=set_types,extended=extended,savename=savename
        )
    elif "wham" in datapath:
        # if we want to train a model on the original wham dataset
        create_wham_whamr_csv(
            datapath, savepath, fs, savename="whamorg_", add_reverb=False, mini=mini,
            mix_folder=mix_folder.replace("reverb","").replace("anechoic","")
        )
    else:
        raise ValueError("Unsupported Dataset at: "+datapath)


def create_multi_whamr_csv(
    datapath,
    savepath,
    fs,
    version="min",
    mix_folder="mix_both_reverb",
    target_folders=["s1_anechoic","s2_anechoic"],
    savename="whamr_",
    set_types=["tr", "cv", "tt"],
    add_reverb=True,
    mini=False,
    num_spks=2,
    alternate_path=None,
    creation_path=None,
    wham_noise_path=None,
    wsj0_path=None, # for transcriptions
    eval_original=False,
    meta_dump = False, # grabs every kind of meta data for room simulation
    use_rirs = False,
    rir_path=None,
    extended=False
):
    """
    This function creates the csv files to get the speechbrain data loaders for the whamr dataset.

    Arguments:
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file
        fs (int) : the sampling rate
        version (str) : min or max
        savename (str) : the prefix to use for the .csv files
        set_types (list) : the sets to create
    """

    if alternate_path==None:
        target_path = datapath
    else:
        target_path = alternate_path
    
    if creation_path != None and wham_noise_path != None:
        extra = True
        directory = get_meta_data(creation_path, wham_noise_path)
    else: 
        extra = False
    
    if not wsj0_path == None:
        transcription_directory =  get_transcriptions(wsj0_path)
        transcribe = True
    else:
        transcribe = False
    
    # if not rir_path == None:
    #     rir_directory= get_rir_paths(rir_path)
    #     use_rirs = True
    # else:
    #     use_rirs = False

    if fs == 8000:
        sample_rate = "8k"
    elif fs == 16000:
        sample_rate = "16k"
    else:
        raise ValueError("Unsupported sampling rate")

    mix = mix_folder+"/"
    speakers = [folder+"/" for folder in target_folders]


    if eval_original:
        s1_eval = "s1_anechoic/"
        s2_eval = "s2_anechoic/"
    

    for set_type in set_types:
        mix_path = os.path.join(
            datapath, "wav{}".format(sample_rate), version, set_type, mix,
        )
        speaker_paths = [os.path.join(
            target_path, "wav{}".format(sample_rate), version, set_type, speaker,
        ) for speaker in speakers]
       
        noise_path = os.path.join(
            datapath, "wav{}".format(sample_rate), version, set_type, "noise/"
        )

        files = os.listdir(mix_path)

        mix_fl_paths = [mix_path + fl for fl in files]
        speaker_fl_paths = [
            [speaker_path + fl for fl in files] for speaker_path in speaker_paths
        ]
        noise_fl_paths = [noise_path + fl for fl in files]

        csv_columns = [
            "ID",
            "duration",
            "mix_wav",
            "mix_wav_format",
            "mix_wav_opts"]
        for i in range(len(speaker_paths)):
            pre_spk = "s"+str(i+1)    
            spk_cols = [pre_spk+"_wav",pre_spk+"_wav_format",pre_spk+"_wav_opts"]
            csv_columns = csv_columns+spk_cols
        csv_columns = csv_columns + ["noise_wav",
            "noise_wav_format",
            "noise_wav_opts",
            "t60",
            "room_size",
            "snr",
        ]

        if meta_dump:
            extra_cols = [
                "room_x",
                "room_y",
                "room_z",
                "micL_x",
                "micL_y",
                "micR_x",
                "micR_y",
                "mic_z",
            ]
            for i in range(len(speaker_paths)):
                pre_spk = "s"+str(i+1)
                spk_cols = [pre_spk+"_x",pre_spk+"_y",pre_spk+"_z"]
                extra_cols = extra_cols+spk_cols
            csv_columns = csv_columns + extra_cols
        
        if use_rirs:
            extra_cols = ["s"+str(i+1)+"_rir" for i in range(len(speaker_paths))]
            csv_columns = csv_columns + extra_cols

        with open(
            (
            os.path.join(savepath, savename + set_type + ".csv") if not extended else os.path.join(savepath, savename + set_type + "_ext.csv")
            ), "w"
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            if mini:
                zipped = list(zip(
                    mix_fl_paths,
                    *speaker_fl_paths,
                    noise_fl_paths,
                    # rir_fl_paths,
                ))[:len(mix_fl_paths)//4]
            elif (not mini):
                zipped = zip(
                    mix_fl_paths,
                    *speaker_fl_paths,
                    noise_fl_paths,
                    # rir_fl_paths,
                )
            
            for (i, packed,) in enumerate(zipped):
                mix_path = packed[0]
                speaker_paths_pack = packed[1:-1]
                noise_path = packed[-1]
                basename = os.path.basename(mix_path)
                
                row = {
                    "ID": i,
                    "duration": 1.0,
                    "mix_wav": mix_path,
                    "mix_wav_format": "wav",
                    "mix_wav_opts": None}
                for i in range(len(speaker_paths)):
                    pre_spk="s"+str(i+1)
                    spk_dict = {pre_spk+"_wav": speaker_paths_pack[i],
                    pre_spk+"_wav_format": "wav",
                    pre_spk+"_wav_opts": None}
                    row = {**row, **spk_dict}
                    
                row = {**row,**{"noise_wav": noise_path,
                    "noise_wav_format": "wav",
                    "noise_wav_opts": None,
                    "t60": directory[basename]["t60"] if extra else None,
                    "room_size": directory[basename]["room_size"] if extra else None,
                    "snr": directory[basename]["snr"] if extra else None,
                }}

                if meta_dump:
                    row["room_x"] = directory[basename]["room_x"] 
                    row["room_y"] = directory[basename]["room_y"] 
                    row["room_z"] = directory[basename]["room_z"] 
                    row["micL_x"] = directory[basename]["micL_x"] 
                    row["micL_y"] = directory[basename]["micL_y"] 
                    row["micR_x"] = directory[basename]["micR_x"] 
                    row["micR_y"] = directory[basename]["micR_y"] 
                    row["mic_z"] = directory[basename]["mic_z"] 
                    for i in range(len(speaker_paths)):
                        row["s"+(i+1)+"_x"] = directory[basename]["s"+(i+1)+"_x"] 
                        row["s"+(i+1)+"_y"] = directory[basename]["s"+(i+1)+"_y"] 
                        row["s"+(i+1)+"_z"] = directory[basename]["s"+(i+1)+"_z"] 
                
                # if use_rirs:
                #     row["s1_rir"] = s1_path.replace("s1_anechoic","s1_rir")
                #     if num_spks == 2:
                #         row["s2_rir"] = s2_path.replace("s2_anechoic","s2_rir")

                # print(row)
                writer.writerow(row)


###### WSJ0MIX ########
def prepare_wsjmix_csv(
    datapath,
    savepath,
    n_spks=2,
    skip_prep=False,
    librimix_addnoise=False,
    fs=8000,
):
    """
    Prepared wsj2mix if n_spks=2 and wsj3mix if n_spks=3.

    Arguments:
    ----------
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file.
        n_spks (int): number of speakers
        skip_prep (bool): If True, skip data preparation
        librimix_addnoise: If True, add whamnoise to librimix datasets
    """

    if skip_prep:
        return

    
    if n_spks == 2:
        assert (
            "2mix" in datapath
        ), "Inconsistent number of speakers and datapath"
        create_wsj_csv(datapath, savepath)
    elif n_spks == 3:
        assert (
            "3mix" in datapath
        ), "Inconsistent number of speakers and datapath"
        create_wsj_csv_3spks(datapath, savepath)
    else:
        raise ValueError("Unsupported Number of Speakers")

def create_wsj_csv(datapath, savepath):
    """
    This function creates the csv files to get the speechbrain data loaders for the wsj0-2mix dataset.

    Arguments:
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file
    """
    for set_type in ["tr", "cv", "tt"]:
        mix_path = os.path.join(datapath, "wav8k/min/" + set_type + "/mix/")
        s1_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s1/")
        s2_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s2/")

        files = os.listdir(mix_path)

        mix_fl_paths = [mix_path + fl for fl in files]
        s1_fl_paths = [s1_path + fl for fl in files]
        s2_fl_paths = [s2_path + fl for fl in files]

        csv_columns = [
            "ID",
            "duration",
            "mix_wav",
            "mix_wav_format",
            "mix_wav_opts",
            "s1_wav",
            "s1_wav_format",
            "s1_wav_opts",
            "s2_wav",
            "s2_wav_format",
            "s2_wav_opts",
        ]

        with open(savepath + "/wsj_" + set_type + ".csv", "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for i, (mix_path, s1_path, s2_path) in enumerate(
                zip(mix_fl_paths, s1_fl_paths, s2_fl_paths)
            ):

                row = {
                    "ID": i,
                    "duration": 1.0,
                    "mix_wav": mix_path,
                    "mix_wav_format": "wav",
                    "mix_wav_opts": None,
                    "s1_wav": s1_path,
                    "s1_wav_format": "wav",
                    "s1_wav_opts": None,
                    "s2_wav": s2_path,
                    "s2_wav_format": "wav",
                    "s2_wav_opts": None,
                }
                writer.writerow(row)


def create_wsj_csv_3spks(datapath, savepath):
    """
    This function creates the csv files to get the speechbrain data loaders for the wsj0-3mix dataset.

    Arguments:
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file
    """
    for set_type in ["tr", "cv", "tt"]:
        mix_path = os.path.join(datapath, "wav8k/min/" + set_type + "/mix/")
        s1_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s1/")
        s2_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s2/")
        s3_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s3/")

        files = os.listdir(mix_path)

        mix_fl_paths = [mix_path + fl for fl in files]
        s1_fl_paths = [s1_path + fl for fl in files]
        s2_fl_paths = [s2_path + fl for fl in files]
        s3_fl_paths = [s3_path + fl for fl in files]

        csv_columns = [
            "ID",
            "duration",
            "mix_wav",
            "mix_wav_format",
            "mix_wav_opts",
            "s1_wav",
            "s1_wav_format",
            "s1_wav_opts",
            "s2_wav",
            "s2_wav_format",
            "s2_wav_opts",
            "s3_wav",
            "s3_wav_format",
            "s3_wav_opts",
        ]

        with open(savepath + "/wsj_" + set_type + ".csv", "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for i, (mix_path, s1_path, s2_path, s3_path) in enumerate(
                zip(mix_fl_paths, s1_fl_paths, s2_fl_paths, s3_fl_paths)
            ):

                row = {
                    "ID": i,
                    "duration": 1.0,
                    "mix_wav": mix_path,
                    "mix_wav_format": "wav",
                    "mix_wav_opts": None,
                    "s1_wav": s1_path,
                    "s1_wav_format": "wav",
                    "s1_wav_opts": None,
                    "s2_wav": s2_path,
                    "s2_wav_format": "wav",
                    "s2_wav_opts": None,
                    "s3_wav": s3_path,
                    "s3_wav_format": "wav",
                    "s3_wav_opts": None,
                }
                writer.writerow(row)