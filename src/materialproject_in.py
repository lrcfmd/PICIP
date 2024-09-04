from mp_api.client import MPRester
from pymatgen.entries.computed_entries import ComputedEntry
from emmet.core.thermo import ThermoType
from pymatgen.analysis.phase_diagram import PhaseDiagram
import itertools
import pickle
import copy
el=['Li','Mg','Ni','Fe','Al','Zn','Sn','Ru','Cu','Co']
x=itertools.combinations(el,3)
x=[['Mg','B','O','F']]
for els in x:
    print(els)
    cs="-".join(els)
    print(cs)
    file="".join(els)+".txt"
    #print(cs)
    with MPRester("6RIrYbfyl2utrngtDAkfn20ivRkZVEXr") as mpr:
        docs = mpr.summary.search(material_ids=["mp-149"], fields=["structure"])
        print('yay')
        pd = mpr.thermo.get_phase_diagram_from_chemsys(
            chemsys=cs,thermo_type=ThermoType.GGA_GGA_U_R2SCAN)
        dd={}
        for i in pd.entries:
            print(i.composition)
            dd[i.composition]=i.energy
        with open(file, 'wb') as f:
            pickle.dump(dd, f)
    with open(file, 'rb') as f:
        in_dict=pickle.load(f)
    entries=[]
    for k,v in in_dict.items():
        entries.append(ComputedEntry(k,v))
    #for i in entries:
    #    print(i.composition)
    pd=PhaseDiagram(entries)
    #print('--------')
    count=0
    print('------')
    for i in pd.stable_entries:
        if len(i.composition)==4:
            count+=1
            print(i.composition)
    print(count)


