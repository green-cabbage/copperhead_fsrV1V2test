from coffea.nanoevents import NanoAODSchema
import coffea.processor as processor
from distributed import Client
import json 
from fsr_recovery import fsr_recovery
import glob
import os
import numpy as np
import awkward as ak
from coffea.processor import DaskExecutor, Runner
import argparse
import traceback
import pandas as pd

class DimuonProcessor(processor.ProcessorABC):
    def __init__(self):
        pass
    
    def process(self, events):
        fsr_recovery(events)
        muon_columns = [
                "pt",
                "pt_fsr",
        ] 
        muons = ak.to_pandas(events.Muon[muon_columns])
        print(f"input length : {len(muons)}")   
        nmuons = (
            muons
            .reset_index()
            .groupby("entry")["subentry"]
            .nunique()
        )
        
        output = pd.DataFrame({"run": events.run, "event": events.event})
        print(f"muons \n : {muons}")       
        mu1 = muons.loc[muons.pt.groupby("entry").idxmax()]
        mu2 = muons.loc[muons.pt.groupby("entry").idxmin()]
        mu1.index = mu1.index.droplevel("subentry")
        mu2.index = mu2.index.droplevel("subentry")
        print(f"mu1 \n : {mu1}") 
        output["mu1_pt"] = mu1.pt
        output["mu1_pt_fsr"] = mu1.pt_fsr
        output["mu2_pt"] = mu2.pt
        output["mu2_pt_fsr"] = mu2.pt_fsr
        
        event_selection =(nmuons == 2)
        print(f"event_selection \n : {event_selection}") 
        output["event_selection"] = event_selection
        output = output[output["event_selection"]==True]
        print(f"output \n : {output}")
        save_path = "./output/V1/"
        # remove previously existing files
        filelist = glob.glob(f"{save_path}/*.parquet")
        for file in filelist:
            os.remove(file)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        output.to_parquet(path=save_path+"0.parquet")
        return None
    # @property
    # def accumulator(self):
    #     return processor.defaultdict_accumulator(int)

    # @property
    # def columns(self):
    #     return branches

    def postprocess(self, accumulator):
        return accumulator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ch",
        "--chunksize",
        dest="chunksize",
        default=100000,
        action="store",
        help="Approximate chunk size",
    )
    parser.add_argument(
        "-mch",
        "--maxchunks",
        dest="maxchunks",
        default=-1,
        action="store",
        help="Max. number of chunks",
    )
    args = parser.parse_args()

    sample_path = "./input_file.json"
    with open(sample_path) as file:
        samples = json.loads(file.read())
    V1_samples = {}
    for dataset, sample_dict in samples.items():
        V1_samples[dataset] = {
            "files" : list(sample_dict["files"].keys()),
            'treename': 'Events'
        }
    
    client = Client(n_workers=1,  threads_per_worker=1, processes=True, memory_limit='3 GiB')
    executor_args = {"client": client, "retries": 2}
    executor = DaskExecutor(**executor_args)

    run = Runner(
        executor=executor,
        schema=NanoAODSchema,
        chunksize=int(args.chunksize),
        maxchunks=int(args.maxchunks),
        xrootdtimeout=2400,
    )
    processor = DimuonProcessor()
    try:
        run(
            V1_samples,
            "Events",
            processor_instance=processor,
        )
    except Exception as e:
        tb = traceback.format_exc()
        print( "Failed: " + str(e) + " " + tb)