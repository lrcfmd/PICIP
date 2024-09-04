import numpy as np
class wt_converter:
    def __init__(self,relative=False):
        self.relative=relative
    # Dictionary of all elements matched with their atomic masses.
        self.elements_dict = {'H' : 1.008,'HE' : 4.003, 'LI' : 6.941, 'BE' : 9.012,\
                         'B' : 10.811, 'C' : 12.011, 'N' : 14.007, 'O' : 15.999,\
                         'F' : 18.998, 'NE' : 20.180, 'NA' : 22.990, 'MG' : 24.305,\
                         'AL' : 26.982, 'SI' : 28.086, 'P' : 30.974, 'S' : 32.066,\
                         'CL' : 35.453, 'AR' : 39.948, 'K' : 39.098, 'CA' : 40.078,\
                         'SC' : 44.956, 'TI' : 47.867, 'V' : 50.942, 'CR' : 51.996,\
                         'MN' : 54.938, 'FE' : 55.845, 'CO' : 58.933, 'NI' : 58.693,\
                         'CU' : 63.546, 'ZN' : 65.38, 'GA' : 69.723, 'GE' : 72.631,\
                         'AS' : 74.922, 'SE' : 78.971, 'BR' : 79.904, 'KR' : 84.798,\
                         'RB' : 84.468, 'SR' : 87.62, 'Y' : 88.906, 'ZR' : 91.224,\
                         'NB' : 92.906, 'MO' : 95.95, 'TC' : 98.907, 'RU' : 101.07,\
                         'RH' : 102.906, 'PD' : 106.42, 'AG' : 107.868, 'CD' : 112.414,\
                         'IN' : 114.818, 'SN' : 118.711, 'SB' : 121.760, 'TE' : 126.7,\
                         'I' : 126.904, 'XE' : 131.294, 'CS' : 132.905, 'BA' : 137.328,\
                         'LA' : 138.905, 'CE' : 140.116, 'PR' : 140.908, 'ND' : 144.243,\
                         'PM' : 144.913, 'SM' : 150.36, 'EU' : 151.964, 'GD' : 157.25,\
                         'TB' : 158.925, 'DY': 162.500, 'HO' : 164.930, 'ER' : 167.259,\
                         'TM' : 168.934, 'YB' : 173.055, 'LU' : 174.967, 'HF' : 178.49,\
                         'TA' : 180.948, 'W' : 183.84, 'RE' : 186.207, 'OS' : 190.23,\
                         'IR' : 192.217, 'PT' : 195.085, 'AU' : 196.967, 'HG' : 200.592,\
                         'TL' : 204.383, 'PB' : 207.2, 'BI' : 208.980, 'PO' : 208.982,\
                         'AT' : 209.987, 'RN' : 222.081, 'FR' : 223.020, 'RA' : 226.025,\
                         'AC' : 227.028, 'TH' : 232.038, 'PA' : 231.036, 'U' : 238.029,\
                         'NP' : 237, 'PU' : 244, 'AM' : 243, 'CM' : 247, 'BK' : 247,\
                         'CT' : 251, 'ES' : 252, 'FM' : 257, 'MD' : 258, 'NO' : 259,\
                         'LR' : 262, 'RF' : 261, 'DB' : 262, 'SG' : 266, 'BH' : 264,\
                         'HS' : 269, 'MT' : 268, 'DS' : 271, 'RG' : 272, 'CN' : 285,\
                         'NH' : 284, 'FL' : 289, 'MC' : 288, 'LV' : 292, 'TS' : 294,\
                         'OG' : 294}

        # List of all elements to allow for easy inclusion testing.
        self.elements_list = ['H', 'HE', 'LI', 'BE', 'B', 'C', 'N', 'O', 'F', 'NE', 'NA',\
                         'MG', 'AL', 'SI', 'P', 'S', 'CL', 'AR', 'K', 'CA', 'SC', 'TI',\
                         'V', 'CR', 'MN', 'FE', 'CO', 'NI', 'CU', 'ZN', 'GA', 'GE',\
                         'AS', 'SE', 'BR', 'KR', 'RB', 'SR', 'Y', 'ZR', 'NB', 'MO',\
                         'TC', 'RU', 'RH', 'PD', 'AG', 'CD', 'IN', 'SN', 'SB', 'TE',\
                         'I', 'XE', 'CS', 'BA', 'LA', 'CE', 'PR', 'ND', 'PM', 'SM',\
                         'EU', 'GD', 'TB', 'DY', 'HO', 'ER', 'TM', 'YB', 'LU', 'HF',\
                         'TA', 'W', 'RE', 'OS', 'IR', 'PT', 'AU', 'HG', 'TL', 'PB',\
                         'BI', 'PO', 'AT', 'RN', 'FR', 'RA', 'AC', 'TH', 'PA', 'U',\
                         'NP', 'PU', 'AM', 'CM', 'BK', 'CT', 'ES', 'FM', 'MD', 'NO',\
                         'LR', 'RF', 'DB', 'SG', 'BH', 'HS', 'MT', 'DS', 'RG', 'CN',\
                         'NH', 'FL', 'MC', 'LV', 'TS', 'OG']

    def get_molar_mass(self,formula):
        #function to get molar mass from a valid formula
        #formula must be a space seperated string 
        if formula is not None:
            formula=formula.upper().split()
            atomic_mass_float = 0.0
            invalid_input = False
            coefficient = 1
            formula_standard=[]
            for i, ch in enumerate(formula):
                if ch in self.elements_list:
                    element_mass = self.elements_dict.get(ch)
                    # If the next character is an integer, multiply the element's mass 
                    # by that integer and add it to the running sum.
                    if len(formula)==i+1:
                        atomic_mass_float += element_mass
                    else:
                        try:
                            c=float(formula[i+1])
                            atomic_mass_float += element_mass * c
                            formula_standard.append(c)
                        except:
                            atomic_mass_float+=element_mass
                            formula_standard.append(0)
                            pass
                    # If not, just add that element's mass to the sum.
                # If the charcter is an integer
                elif ch.isdigit() == True:
                    # If the first index is an integer, assume that that is a
                    # coefficient for the formula and multiply the formula by that
                    # number
                    if i == 0:
                        coefficient = int(ch)
                    # Make sure the previous index wasn't an integer as well
                    # (the integer needs an element to multiply), if it was, let the
                    # program know that there is an error.
                    elif formula[i - 1].isalpha() == False:
                        invalid_input = True
                    else:
                        pass
                # If the only character in the formula is an integer, let the program
                # know that there is an error.
                elif ch.isdigit() == True and len(formula) == 1:
                    invalid_input = True
                # If a character is not an element or an integer, let the program know
                else:
                    try:
                        float(ch)
                    except:
                        # that there is an error.
                        invalid_input = True
            # Muliply the entire atomic mass by the coefficient.
            atomic_mass_float *= coefficient
            # If there was an error in the program, display the appropriate error.
            if invalid_input == True:
                # Error for if there is only an integer entered.
                if ch.isdigit() == True and len(formula) == 1:
                    print("You must enter at least one element for every integer\
         subscript.")
                    print()
                # Error for if there is a float or non-elemnt entered.
                else:
                    print("Please enter only the atomic symbols of elements or an integer\
         subscript.")
                    print()
            # If there was no error, then print the claculated atomic mass
            else:
                #print("Atomic Mass: ", round(atomic_mass_float, 3))
                return (atomic_mass_float,formula_standard)

    def test(self,f1,f2,w1,w2):
        mf1 = self.get_molar_mass(f1)
        mf2 = self.get_molar_mass(f2)
        c1 = (w1/w2)*(mf2/mf1)
        print('c1=',c1,',c2=1.')

    def wt_to_moles(self,formulas,weights,weights_error=None,
                    total_mass=None,total_mass_error=None,phase_field=None):
        #converts wt% to mole% and error
        #if weights error is not given assumes 3%
        #else can take a constant error or different error for each weight
        #if total mass is not given assumes 0.5g
        #assumes error on total mass is 2%
        #returns moles of each formula with std
        if phase_field is not None:
            n_formulas=[]
            for i in formulas:
                chunks = i.split()
                f_stan=""
                for el in phase_field:
                    f_stan+=el + " "
                    try:
                        n=chunks.index(el)
                    except ValueError:
                        f_stan += "0 "
                    else:
                        try:
                            f_stan+= str(float(chunks[n+1])) + " "
                        except:
                            f_stan+= "1 "
                n_formulas.append(f_stan)
            formulas=n_formulas

        if len(weights)!=len(formulas):
            print('Error - need weight for each formula')
            print(weights)
            print(formulas)
        #weights/=sum(weights)
        molar_masses=[0]*len(formulas)
        formulas_standard=[0]*len(formulas)
        for i,f in enumerate(formulas):
            molar_masses[i],formulas_standard[i]=self.get_molar_mass(f)
        if weights_error is not None:
            if len(weights_error)==1:
                weights_error=[weights_error[0]]*len(formulas)
            if self.relative:
                weights_error=[e*w for e,w in zip(weights_error,weights)]
            elif len(weights_error)!=len(formulas):
                print('Error - number of errors must be 1 or number of inputs')
        else:
            weights_error = [0.035*x+2.835 for x in weights]
        if total_mass is None:
            total_mass=0.5
        if total_mass_error is None:
            total_mass_error = 0.02*total_mass
        moles_error = [0]*len(formulas)
        moles=[0]*len(formulas)
        for i in range(len(formulas)):
            mole=weights[i]*total_mass/molar_masses[i]
            mole_error=(((total_mass*weights_error[i])**2+
                          (weights[i]*total_mass_error)**2)**0.5)/molar_masses[i]
            moles[i]=mole
            moles_error[i]=mole_error
        moles_t=np.array(moles)/sum(moles)
        return (np.array(moles),np.array(moles_error),formulas_standard)

        
