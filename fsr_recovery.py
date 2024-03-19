import numpy as np
import awkward as ak


def fsr_recovery(events):
    mask = (
        (events.Muon.fsrPhotonIdx >= 0)
        & (events.Muon.matched_fsrPhoton.relIso03 < 1.8)
        & (events.Muon.matched_fsrPhoton.dROverEt2 < 0.012)
        & (events.Muon.matched_fsrPhoton.pt / events.Muon.pt < 0.4)
        & (abs(events.Muon.matched_fsrPhoton.eta) < 2.4)
    )
    mask = ak.fill_none(mask, False)


    px = ak.zeros_like(events.Muon.pt)
    py = ak.zeros_like(events.Muon.pt)
    pz = ak.zeros_like(events.Muon.pt)
    e = ak.zeros_like(events.Muon.pt)

    fsr = {
        "pt": events.Muon.matched_fsrPhoton.pt,
        "eta": events.Muon.matched_fsrPhoton.eta,
        "phi": events.Muon.matched_fsrPhoton.phi,
        "mass": 0.0,
    }

    for obj in [events.Muon, fsr]:
        px_ = obj["pt"] * np.cos(obj["phi"])
        py_ = obj["pt"] * np.sin(obj["phi"])
        pz_ = obj["pt"] * np.sinh(obj["eta"])
        e_ = np.sqrt(px_**2 + py_**2 + pz_**2 + obj["mass"] ** 2)

        px = px + px_
        py = py + py_
        pz = pz + pz_
        e = e + e_

    pt = np.sqrt(px**2 + py**2)
    print(f"type(pt): {(pt.type)}")
    print(f"total nmuons applied with fsrPhotons: {ak.sum(mask,axis=None)}")
    eta = np.arcsinh(pz / pt)
    phi = np.arctan2(py, px)
    mass = np.sqrt(e**2 - px**2 - py**2 - pz**2)
    print(f"type(eta): {(eta.type)}")
    print(f"type(phi): {(phi.type)}")
    print(f"type(mass): {(mass.type)}")
    iso = (events.Muon.pfRelIso04_all * events.Muon.pt - events.Muon.matched_fsrPhoton.pt) / pt

    events["Muon", "pt_fsr"] = ak.where(mask, pt, events.Muon.pt)
    events["Muon", "eta_fsr"] = ak.where(mask, eta, events.Muon.eta)
    events["Muon", "phi_fsr"] = ak.where(mask, phi, events.Muon.phi)
    events["Muon", "mass_fsr"] = ak.where(mask, mass, events.Muon.mass)
    # events["Muon", "mass_fsr"] = events.Muon.mass
    events["Muon", "iso_fsr"] = ak.where(mask, iso, events.Muon.pfRelIso04_all)
    fsr_event_mask = ak.sum(mask, axis=1) > 0
    # print(f"fsr_event_mask len : {ak.sum(fsr_event_mask)}")
    # print(f"events[mask].Muon.pt_fsr: {events[fsr_event_mask].Muon.pt_fsr}")
    # print(f"events[mask].Muon.pt: {events[fsr_event_mask].Muon.pt}")
    # print(f"fsr[pt][mask]: \n {fsr['pt'][fsr_event_mask]}")
    return mask
