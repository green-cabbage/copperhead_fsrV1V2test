import numpy as np
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from distributed import Client


with Client(n_workers=41,  threads_per_worker=1, processes=True, memory_limit='3 GiB', silence_logs=logging.ERROR) as client:

    events = NanoEventsFactory.from_root(
        # {fname: "Events"},
        file_dict,
        schemaclass=NanoAODSchema,
        metadata={"dataset": "DYJets"},
        # delayed=False,
        # entry_stop = 10017,
    ).events()
    nmuons = ak.num(events.Muon, axis=1)
    good_events = nmuons==2
    # events = events[good_events]
    fsr_recovery_old(events)
    padded_muons = ak.pad_none(events.Muon, target=2, clip=True)
    muon_flip = padded_muons.pt[:,0] < padded_muons.pt[:,1]  
    muon_flip = ak.fill_none(muon_flip, value=False)
    # take the subleading muon values if that now has higher pt after corrections
    mu1 = ak.where(muon_flip, padded_muons[:,1], padded_muons[:,0])
    mu2 = ak.where(muon_flip, padded_muons[:,0], padded_muons[:,1])
    mu1_pt_old = mu1.pt
    mu2_pt_old = mu2.pt
    print(mu1_pt_old.compute())