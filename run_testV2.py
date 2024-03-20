import numpy as np
import awkward as ak
import dask_awkward as dak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from distributed import Client
import json 
from fsr_recovery import fsr_recovery
import glob
import os

# file_dict = {"root://eos.cms.rcac.purdue.edu:1094///store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/40000/AA6F89B0-EDAA-3942-A3BB-BC3709722EB4.root": {"object_path": "Events", "steps": [[0, 100017]], "uuid": "b16b5dea-fbcd-11ed-bae7-a2a0b8bcbeef"}}

if __name__ == '__main__':
    with Client(n_workers=1,  threads_per_worker=1, processes=True, memory_limit='3 GiB') as client:
        print("test")
        sample_path = "./input_file.json"
        with open(sample_path) as file:
            samples = json.loads(file.read())
        file_dict = samples["dy_M-100To200"]["files"]
        events = NanoEventsFactory.from_root(
            file_dict,
            schemaclass=NanoAODSchema,
            metadata={"dataset": "DYJets"},
        ).events()
        nmuons = ak.num(events.Muon, axis=1)
        nmuon_selection = nmuons==2
        good_events = events[nmuon_selection]
        fsr_recovery(good_events)
        padded_muons = ak.pad_none(good_events.Muon, target=2, clip=True)
        sorted_args = ak.argsort(padded_muons.pt, ascending=False)
        sorted_muons = padded_muons[sorted_args]
        # take the subleading muon values if that now has higher pt after corrections
        mu1 = sorted_muons[:,0]
        mu2 = sorted_muons[:,1]
        out_dict = {
            "mu1_pt" : mu1.pt,
            "mu1_pt_fsr" : mu1.pt_fsr,
            "mu2_pt" : mu2.pt,
            "mu2_pt_fsr" : mu2.pt_fsr,
        }
        zip = ak.zip(out_dict)
        save_path = "./output/V2/"
        # remove previously existing files
        filelist = glob.glob(f"{save_path}/*.parquet")
        for file in filelist:
            os.remove(file)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        dak.to_parquet(zip, save_path) # save data



