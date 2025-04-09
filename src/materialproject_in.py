from mp_api.client import MPRester
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition
import itertools
import pickle
import copy
from collections import defaultdict
elementsa=['Li','Na','K','Rb','Cs','Be','Mg','Ca','Sr','Ba','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu']#,'Zn','Al','Ga','In','Sn','Tl','Pb']
ans=['O']
#x=itertools.combinations(el,3)
#x=[tuple(list(p)+['O']) for p in x]
#x=[['Mg','B','O','F']]
x=[1]
for an in ans:
    cs='S-*-*'
    #file="".join(els)+".txt"
    #print(cs)
    comps=[]
    with MPRester("rH3eb6i3mHNcQR9oJ1dpeWAmKr1YdD0s") as mpr:
        #docs = mpr.materials.summary.search(
                #chemsys=cs,fields='composition',is_stable=True,exclude_elements=['Cl','F','O','I','C','H','Br','P','N','As'])
        docs = mpr.materials.summary.search(
        material_ids=["mp-149", "mp-13", "mp-22526"])
        print(docs)
        dd={}
        charged_dic=defaultdict(list)
        for i in docs:
            print(i)
            comp=Composition(i.composition.as_dict())
            oxi_prob={'S':[-2]}
            c_comp=comp.add_charges_from_oxi_state_guesses(oxi_states_override=oxi_prob)
            #print(c_comp.as_dict())
            b=c_comp.charge_balanced
            if b:
                charges=[]
                valid=True
                if len(c_comp)==3:
                    for n,sp in enumerate(list(c_comp.keys())):
                        if n<2:
                            charge=sp.as_dict()['oxidation_state']
                            if charge<0:
                                valid=False
                            else:
                                charges.append(charge)
                        elif n==2:
                            charge=sp.as_dict()['oxidation_state']
                            if charge!=-2:
                                valid=False
                    if valid:
                        charges.sort()
                        charge_key=tuple(charges)
                        comp=list(i.composition.as_dict().values())[:-1]
                        charged_dic[charge_key].append(comp)
        for i in charged_dic.keys():
            print(i)
            print(charged_dic[i])
        with open("bin_S_charged.pkl", "wb") as f:
            pickle.dump(charged_dic, f)


                            
        #comps.append(list(i.composition.as_dict().values())[:-1])
        #with open("tern_O.pkl", "wb") as f:
        #    pickle.dump(comps, f)
'''
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
    '''


